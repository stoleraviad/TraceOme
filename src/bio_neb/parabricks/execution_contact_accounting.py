"""One evidence transaction for the public post-HITL execution flow.

Boundary decision: this module owns diagnostic evidence, not provider Adapter
mechanics.  Contact-capable adapters depend only on ``ExecutionContactObserver`` and
cannot open, finalize, compile, or review a journal.  The sink stays beside its strict
row schema and longest-prefix parser so durable bytes cannot acquire a second,
drifting interpretation; Journey mutation is confined to the finalizer below, after
the Adapter outcome already exists.  Splitting persistence from that canonical parser
would widen the seam while duplicating the invariant that makes crash evidence true.
The public application operation owns the transaction and covers its Parabricks Adapter
contacts.  One bound execution may run more than one pass (issue 41: the pass that stops
at the CPU-preparation review and the pass that continues after ``approve``): each pass
has its own journal and its own Journey pointer, the pointer names its pass, and the
ledger written by a pass composes every pass so far in order, so the latest ledger is
the execution's one ledger.  The first pass keeps the unnumbered locators every retained
receipt already names; a later pass lives under ``pass-<n>/`` beside them.
The selected Nebius context is supplied before this transaction; this ledger never
claims per-invocation or full-predecessor totals for how that selection was produced.

Physical Source attempts are owned by transport scope facts, while typed stage facts
retain scientific meaning. An unmeasurable total may retain observed units as a lower
bound; those units never turn incomplete accounting into fixture eligibility. Journal
completion and fixture eligibility remain separate diagnostic decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from bio_neb.contracts import StrictFrozenContract
from bio_neb.failure import ResolvedFailurePolicy
from bio_neb.journey import canonical_json
from bio_neb.persistence import append_durable, retain_immutable, write_once
from bio_neb.providers.nebius_common import RegistryResponseEvidence
from bio_neb.providers.ports import ContactReceipt

COMPILER_VERSION = "bio-neb.execution-contact-compiler/2"
JOURNAL_SCHEMA_VERSION = "bio-neb.execution-contact-journal/1"
LEDGER_SCHEMA_VERSION = "bio-neb.fresh-full-predecessor-contact-ledger/3"
REVIEW_SCHEMA_VERSION = "bio-neb.execution-contact-review/2"
EXECUTION_CONTACT_SCOPE = "Parabricks operation transaction"
EXECUTION_CONTACT_SCOPE_ID = "execution_flow_parabricks_adapter_transaction"


def _unsupported_accounting_version(value: Any, schema_version: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    journal = value.get("journal", {})
    return (
        value.get("schema_version", schema_version) != schema_version
        or value.get("compiler_version", COMPILER_VERSION) != COMPILER_VERSION
        or isinstance(journal, Mapping)
        and journal.get("compiler_version", COMPILER_VERSION) != COMPILER_VERSION
    )


ContactCategory = Literal[
    "local_runtime", "provider", "storage", "credential", "source"
]
ContactDisposition = Literal["attempted", "succeeded", "failed", "unknown"]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_EXECUTION_PATTERN = r"^parabricks-execution:[0-9a-f]{64}$"
_UNIT_PATTERN = r"^execution-contact:[0-9a-f]{64}:[0-9]{8}$"

_CREDENTIAL_NAMES = (
    "control_plane_authentication",
    "host_secret_resolution",
    "worker_secret_binding",
    "session_credentials",
)
_NEBIUS_OPERATIONS = frozenset(
    {
        "nebius.version",
        "iam.project.get",
        "iam.project.list",
        "iam.tenant.list",
        "iam.v2.access-key.list",
        "iam.group-membership.list-member-of",
        "iam.group-membership.list-members",
        "iam.access-permit.list",
        "mysterybox.payload.get",
        "mysterybox.secret-version.payload.get",
        "mysterybox.secret-version.get",
        "mysterybox.secret.get",
        "iam.v2.access-key.get",
        "iam.group-membership.get",
        "iam.service-account.get",
        "iam.access-permit.get",
        "iam.group.get",
        "iam.whoami",
        "storage.bucket.create",
        "storage.bucket.get",
        "storage.bucket.list",
        "registry.v2.repository.get",
        "ai.job.create",
        "ai.job.get",
        "nebius.allowlisted-other",
    }
)
_S3_OPERATIONS = frozenset(
    {
        "s3.head-object",
        "s3.get-object",
        "s3.list-objects-v2",
        "s3.list-multipart-uploads",
        "s3.put-object",
        "s3.get-bucket-lifecycle-configuration",
        "s3.put-bucket-lifecycle-configuration",
        "s3.create-multipart-upload",
        "s3.upload-part",
        "s3.upload-part-copy",
        "s3.complete-multipart-upload",
        "s3.copy-object",
        "s3.allowlisted-other",
    }
)
_REGISTRY_OPERATIONS = frozenset(
    {
        f"registry.http.{phase}"
        for phase in ("challenge", "token", "index", "child", "config")
    }
)
_PARSE_CLASSIFICATIONS = frozenset(
    {
        "timeout",
        "invalid_utf8",
        "unavailable",
        "runtime_interrupted",
        "interactive_authentication_required",
        "authentication_failed",
        "authorization_denied",
        "invalid_argument",
        "not_found",
        "unknown",
        "version_text",
        "json_parse_pending",
        "text",
        "absent",
        "access",
        "transient",
        "provider_error",
        "response_received",
        "credential_unavailable",
        "selected_identity",  # Historical rows do not distinguish resolution from reuse.
        "credential_resolved",
        "credential_cached",
    }
)


def _canonical_safe_line(value: str, field_name: str, maximum: int = 160) -> str:
    if (
        not value
        or value.strip() != value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{field_name} must be bounded canonical safe text")
    return value


def _strict_json_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    """Admit only the JSON-array representation of an otherwise strict tuple."""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(value)


class ExecutionContactJournalRow(StrictFrozenContract):
    """One safe canonical lifecycle row; terminal rows repeat only safe identity."""

    schema_version: Literal["bio-neb.execution-contact-journal/1"] = (
        JOURNAL_SCHEMA_VERSION
    )
    sequence: StrictInt = Field(ge=1)
    execution_id: StrictStr = Field(pattern=_EXECUTION_PATTERN)
    row_kind: Literal["start", "terminal", "final", "step"]
    unit_id: StrictStr | None = Field(default=None, pattern=_UNIT_PATTERN)
    category: ContactCategory | None = None
    operation: StrictStr | None = None
    executable_identity: StrictStr | None = None
    disposition: ContactDisposition | None = None
    exit_status: StrictInt | None = None
    parse_classification: StrictStr | None = None
    registry_response: RegistryResponseEvidence | None = None
    credential_names: tuple[StrictStr, ...] = ()
    selected_identity: StrictStr | None = None
    prefix_sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    complete: StrictBool | None = None
    stage: StrictStr | None = None
    step: StrictStr | None = None
    phase: Literal["claimed", "invoked", "verified", "failed"] | None = None
    attempt: StrictInt = Field(default=1, ge=1)
    ceiling_count: StrictInt | None = Field(default=None, ge=0)
    receipt: ContactReceipt | None = None
    provider_ids: tuple[StrictStr, ...] = ()
    duration: StrictFloat | None = Field(default=None, ge=0, allow_inf_nan=False)
    logical_operation: StrictStr | None = None
    resource_kind: Literal["bucket", "cpu_job", "h100_job"] | None = None
    resource_intent: StrictStr | None = None
    invocation_ref: StrictStr | None = None
    failure_class: StrictStr | None = None
    failure_detail: StrictStr | None = None
    failure_policy: ResolvedFailurePolicy | None = None

    @field_validator("provider_ids", mode="before")
    @classmethod
    def provider_ids_json_boundary(cls, value: Any) -> tuple[Any, ...]:
        return _strict_json_tuple(value, "provider_ids")

    @field_validator("provider_ids")
    @classmethod
    def safe_provider_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _canonical_safe_line(item, "provider_ids", maximum=2048) for item in value
        )

    @field_validator("credential_names", mode="before")
    @classmethod
    def credential_names_json_boundary(cls, value: Any) -> tuple[Any, ...]:
        return _strict_json_tuple(value, "credential_names")

    @field_validator(
        "operation",
        "executable_identity",
        "parse_classification",
        "selected_identity",
        "stage",
        "step",
        "logical_operation",
        "resource_intent",
        "invocation_ref",
        "failure_class",
    )
    @classmethod
    def safe_optional_lines(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _canonical_safe_line(value, info.field_name)

    @field_validator("failure_detail")
    @classmethod
    def safe_failure_detail(cls, value):
        return (
            None
            if value is None
            else _canonical_safe_line(value, "failure_detail", 2000)
        )

    @field_validator("credential_names")
    @classmethod
    def safe_credential_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _canonical_safe_line(item, "credential_names", 80) for item in value
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("credential names must be unique")
        return normalized

    @model_validator(mode="after")
    def step_shape(self) -> "ExecutionContactJournalRow":
        identity = (
            self.unit_id,
            self.category,
            self.operation,
            self.executable_identity,
        )
        extension = (self.stage, self.step, self.phase)
        if any(item is not None for item in extension) and not all(
            item is not None for item in extension
        ):
            raise ValueError("stage, step and phase must be supplied together")
        if self.row_kind == "step":
            if not all(item is not None for item in extension) or any(
                item is not None for item in identity
            ):
                raise ValueError(
                    "step row requires stage/step/phase without a contact unit"
                )
            if any(
                item is not None
                for item in (
                    self.disposition,
                    self.exit_status,
                    self.parse_classification,
                    self.prefix_sha256,
                    self.complete,
                )
            ):
                raise ValueError("step row carries contact/final fields")
        if self.row_kind == "final" and (
            any(item is not None for item in extension)
            or self.receipt is not None
            or self.provider_ids
            or self.duration is not None
            or self.ceiling_count is not None
        ):
            raise ValueError("final row carries step evidence")
        return self

    @model_validator(mode="after")
    def registry_shape(self) -> "ExecutionContactJournalRow":
        registry_contact = (
            self.category == "provider" and self.operation in _REGISTRY_OPERATIONS
        )
        if registry_contact and self.executable_identity != "registry-http-client":
            raise ValueError("Registry contact requires its HTTP identity")
        if self.registry_response is not None and (
            not registry_contact or self.row_kind != "terminal"
        ):
            raise ValueError("Registry response belongs only to a Registry terminal")
        if (
            registry_contact
            and self.row_kind == "terminal"
            and self.registry_response is None
        ):
            raise ValueError("Registry terminal lacks response evidence")
        return self

    @model_validator(mode="after")
    def exact_shape(self) -> "ExecutionContactJournalRow":
        identity = (
            self.unit_id,
            self.category,
            self.operation,
            self.executable_identity,
        )
        if self.row_kind == "start":
            if (
                any(item is None for item in identity)
                or self.disposition != "attempted"
            ):
                raise ValueError("journal start lacks its exact attempted identity")
            if (
                self.exit_status is not None
                or self.parse_classification is not None
                or self.prefix_sha256 is not None
                or self.complete is not None
            ):
                raise ValueError("journal start carries terminal or final material")
        elif self.row_kind == "terminal":
            if (
                any(item is None for item in identity)
                or self.disposition not in {"succeeded", "failed", "unknown"}
                or self.prefix_sha256 is not None
                or self.complete is not None
            ):
                raise ValueError("journal terminal lacks its exact unit disposition")
        elif self.row_kind == "final" and (
            any(item is not None for item in identity)
            or self.disposition is not None
            or self.exit_status is not None
            or self.parse_classification is not None
            or self.credential_names
            or self.selected_identity is not None
            or self.prefix_sha256 is None
            or self.complete is None
        ):
            raise ValueError("journal final row changed its prefix binding shape")
        if self.category == "credential":
            if (
                self.credential_names != _CREDENTIAL_NAMES
                or self.selected_identity is None
                or not self.selected_identity.startswith(
                    "nebius-object-storage-topology:"
                )
                or len(self.selected_identity) != 95
                or any(
                    item not in "0123456789abcdef"
                    for item in self.selected_identity.removeprefix(
                        "nebius-object-storage-topology:"
                    )
                )
            ):
                raise ValueError(
                    "credential row lacks interface names or selected identity"
                )
        elif self.credential_names or self.selected_identity is not None:
            raise ValueError("noncredential row carries credential material")
        if self.category == "local_runtime" and self.operation != "nebius.version":
            raise ValueError("only the pinned version subprocess is local runtime")
        if self.category == "provider" and self.operation == "nebius.version":
            raise ValueError("the pinned version subprocess is not provider contact")
        registry_contact = (
            self.category == "provider" and self.operation in _REGISTRY_OPERATIONS
        )
        if (
            self.category in {"local_runtime", "provider"}
            and not registry_contact
            and (
                self.executable_identity is None
                or not self.executable_identity.startswith("nebius-cli:")
                or len(self.executable_identity) != 75
                or any(
                    item not in "0123456789abcdef"
                    for item in self.executable_identity.removeprefix("nebius-cli:")
                )
            )
        ):
            raise ValueError("Nebius executable identity is not a bound payload")
        if (
            self.category == "provider"
            and self.operation not in _NEBIUS_OPERATIONS | _REGISTRY_OPERATIONS
        ):
            raise ValueError("provider operation is not normalized")
        if self.category == "source" and (
            self.operation != "http.get" or self.executable_identity != "http-client"
        ):
            raise ValueError("source operation is not normalized")
        if self.category == "storage" and (
            self.executable_identity != "s3-sdk" or self.operation not in _S3_OPERATIONS
        ):
            raise ValueError("storage operation is not a normalized SDK request")
        if self.category == "credential" and (
            self.executable_identity != "credential-resolver"
            or self.operation != "selected-object-storage-credential"
        ):
            raise ValueError("credential operation identity changed")
        if (
            self.parse_classification is not None
            and self.parse_classification not in _PARSE_CLASSIFICATIONS
        ):
            raise ValueError("parse classification is not normalized")
        return self


class ExecutionContactJournalBinding(StrictFrozenContract):
    """Exact immutable identity of the surviving external journal bytes."""

    locator: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    execution_id: StrictStr = Field(pattern=_EXECUTION_PATTERN)
    compiler_version: Literal["bio-neb.execution-contact-compiler/2"] = COMPILER_VERSION

    @field_validator("locator")
    @classmethod
    def canonical_absolute_locator(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or os.path.normpath(value) != value:
            raise ValueError("journal binding locator must be absolute and canonical")
        return value


class ExecutionContactUnit(StrictFrozenContract):
    unit_id: StrictStr = Field(pattern=_UNIT_PATTERN)
    category: ContactCategory
    operation: StrictStr
    executable_identity: StrictStr
    disposition: ContactDisposition
    exit_status: StrictInt | None = None
    parse_classification: StrictStr | None = None
    registry_response: RegistryResponseEvidence | None = None
    credential_names: tuple[StrictStr, ...] = ()
    selected_identity: StrictStr | None = None

    @field_validator("credential_names", mode="before")
    @classmethod
    def credential_names_json_boundary(cls, value: Any) -> tuple[Any, ...]:
        return _strict_json_tuple(value, "credential_names")


class MeasuredContactMetric(StrictFrozenContract):
    status: Literal["measured"] = "measured"
    count: StrictInt = Field(ge=0)
    unit_ids: tuple[StrictStr, ...]
    evidence_bindings: tuple[StrictStr, ...] = Field(min_length=1)
    derivation_version: Literal["bio-neb.execution-contact-compiler/2"] = (
        COMPILER_VERSION
    )

    @field_validator("unit_ids", "evidence_bindings", mode="before")
    @classmethod
    def tuple_json_boundaries(cls, value: Any, info: Any) -> tuple[Any, ...]:
        return _strict_json_tuple(value, info.field_name)

    @model_validator(mode="after")
    def exact_count(self) -> "MeasuredContactMetric":
        if (
            len(set(self.unit_ids)) != len(self.unit_ids)
            or len(self.unit_ids) != self.count
        ):
            raise ValueError("measured count differs from its unique evidenced units")
        return self


class UnmeasurableContactMetric(StrictFrozenContract):
    status: Literal["unmeasurable"] = "unmeasurable"
    count: None = None
    observed_count: StrictInt = Field(default=0, ge=0)
    observed_unit_ids: tuple[StrictStr, ...] = ()
    reason: StrictStr
    evidence_bindings: tuple[StrictStr, ...] = Field(min_length=1)
    derivation_version: Literal["bio-neb.execution-contact-compiler/2"] = (
        COMPILER_VERSION
    )

    @field_validator("evidence_bindings", "observed_unit_ids", mode="before")
    @classmethod
    def tuple_json_boundary(cls, value: Any, info: Any) -> tuple[Any, ...]:
        return _strict_json_tuple(value, info.field_name)

    @model_validator(mode="after")
    def exact_observed_lower_bound(self) -> "UnmeasurableContactMetric":
        if len(set(self.observed_unit_ids)) != len(
            self.observed_unit_ids
        ) or self.observed_count != len(self.observed_unit_ids):
            raise ValueError(
                "observed lower bound differs from its unique evidenced units"
            )
        return self

    @field_validator("reason")
    @classmethod
    def bounded_reason(cls, value: str) -> str:
        return _canonical_safe_line(value, "reason", 120)


ContactMetric = MeasuredContactMetric | UnmeasurableContactMetric


class FreshFullPredecessorContactLedger(StrictFrozenContract):
    """Evidence-derived contacts spent by one already-completed predecessor."""

    schema_version: Literal["bio-neb.fresh-full-predecessor-contact-ledger/3"] = (
        LEDGER_SCHEMA_VERSION
    )
    provenance: Literal["fresh_full_predecessor"] = "fresh_full_predecessor"
    contact_scope: Literal["execution_flow_parabricks_adapter_transaction"] = (
        EXECUTION_CONTACT_SCOPE_ID
    )
    compiler_version: Literal["bio-neb.execution-contact-compiler/2"] = COMPILER_VERSION
    journal_binding: ExecutionContactJournalBinding | None
    # The journals of the passes before the one bound above, in pass order; empty
    # for a single-pass execution, so earlier ledgers keep their shape.
    prior_pass_journal_bindings: tuple[ExecutionContactJournalBinding, ...] = ()
    journey_binding: StrictStr
    credential_access: ContactMetric
    model_contact: ContactMetric
    source_contact: ContactMetric
    storage_contact: ContactMetric
    provider_contact: ContactMetric
    network_contact: ContactMetric
    fresh_bucket: ContactMetric
    cpu_preparation_create: ContactMetric
    h100_create: ContactMetric
    post_execution_vcf_query: ContactMetric

    @field_validator("prior_pass_journal_bindings", mode="before")
    @classmethod
    def tuple_json_boundary(cls, value: Any) -> tuple[Any, ...]:
        return _strict_json_tuple(value, "prior_pass_journal_bindings")


class LogicalContactAttempt(StrictFrozenContract):
    invocation_ref: StrictStr
    execution_pass: StrictInt
    operation: StrictStr
    resource_kind: Literal["bucket", "cpu_job", "h100_job"] | None = None
    resource_intent: StrictStr | None = None
    stage: StrictStr
    step: StrictStr
    attempt: StrictInt
    disposition: Literal["unfinished", "succeeded", "failed", "inconsistent"]
    provider_ids: tuple[StrictStr, ...] = ()
    resource_count: StrictInt | None = None
    failure_class: StrictStr | None = None
    failure_detail: StrictStr | None = None
    receipt: ContactReceipt | None = None

    @field_validator("provider_ids", mode="before")
    @classmethod
    def ids_json_boundary(cls, value):
        return _strict_json_tuple(value, "provider_ids")


class RetainedContactPass(StrictFrozenContract):
    execution_pass: StrictInt
    journal_locator: StrictStr
    outcome_event_id: StrictStr | None
    disposition: StrictStr | None
    journal_complete: StrictBool
    binding_matches: StrictBool
    reasons: tuple[StrictStr, ...]

    @field_validator("reasons", mode="before")
    @classmethod
    def reasons_json_boundary(cls, value):
        return _strict_json_tuple(value, "reasons")


class ExecutionContactReview(StrictFrozenContract):
    schema_version: Literal["bio-neb.execution-contact-review/2"] = (
        REVIEW_SCHEMA_VERSION
    )
    compiler_version: Literal["bio-neb.execution-contact-compiler/2"] = COMPILER_VERSION
    execution_id: StrictStr = Field(pattern=_EXECUTION_PATTERN)
    binding: ExecutionContactJournalBinding | None
    binding_matches: StrictBool
    complete: StrictBool
    fixture_eligible: StrictBool
    valid_row_count: StrictInt = Field(ge=0)
    longest_valid_prefix_byte_count: StrictInt = Field(ge=0)
    truncated_tail: StrictBool
    unmatched_unit_ids: tuple[StrictStr, ...]
    discrepancies: tuple[StrictStr, ...]
    units: tuple[ExecutionContactUnit, ...]
    logical_attempts: tuple[LogicalContactAttempt, ...] = ()
    retained_passes: tuple[RetainedContactPass, ...] = ()
    logical_results: tuple[ExecutionContactJournalRow, ...] = ()
    ledger: FreshFullPredecessorContactLedger

    @field_validator(
        "unmatched_unit_ids",
        "discrepancies",
        "units",
        "logical_attempts",
        "retained_passes",
        "logical_results",
        mode="before",
    )
    @classmethod
    def tuple_json_boundaries(cls, value: Any, info: Any) -> tuple[Any, ...]:
        return _strict_json_tuple(value, info.field_name)


class ExecutionContactJourneyPointer(StrictFrozenContract):
    """One ancillary Journey fact binding the external journal, never authority."""

    schema_version: Literal["bio-neb.execution-contact-journey-pointer/2"] = (
        "bio-neb.execution-contact-journey-pointer/2"
    )
    execution_id: StrictStr = Field(pattern=_EXECUTION_PATTERN)
    journal: ExecutionContactJournalBinding
    ledger_locator: StrictStr
    compiler_version: Literal["bio-neb.execution-contact-compiler/2"] = COMPILER_VERSION
    journal_complete: StrictBool
    # Which pass of the bound execution this journal recorded; pointers retained
    # before passes existed read as the first.
    execution_pass: StrictInt = Field(ge=1, default=1)

    @model_validator(mode="before")
    @classmethod
    def supported_accounting_version(cls, value: Any) -> Any:
        if _unsupported_accounting_version(
            value, "bio-neb.execution-contact-journey-pointer/2"
        ):
            raise ValueError("unsupported_execution_contact_accounting_version")
        return value

    @field_validator("ledger_locator")
    @classmethod
    def canonical_ledger_locator(cls, value: str) -> str:
        if not Path(value).is_absolute() or os.path.normpath(value) != value:
            raise ValueError("ledger locator must be absolute and canonical")
        return value

    @model_validator(mode="after")
    def exact_external_binding(self) -> "ExecutionContactJourneyPointer":
        if self.journal.execution_id != self.execution_id or Path(
            self.ledger_locator
        ) != Path(self.journal.locator).with_name("predecessor-contact-ledger.json"):
            raise ValueError("Journey pointer changed its execution-derived binding")
        return self


class ExecutionContactLedgerBinding(StrictFrozenContract):
    locator: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("locator")
    @classmethod
    def canonical_locator(cls, value: str) -> str:
        if not Path(value).is_absolute() or os.path.normpath(value) != value:
            raise ValueError("ledger binding locator must be absolute and canonical")
        return value


class PublicExecutionContactReview(StrictFrozenContract):
    schema_version: Literal["bio-neb.public-execution-contact-review/2"] = (
        "bio-neb.public-execution-contact-review/2"
    )
    contact_scope: Literal["execution_flow_parabricks_adapter_transaction"] = (
        EXECUTION_CONTACT_SCOPE_ID
    )
    status: Literal["complete", "incomplete", "drift"]
    execution_journey_file: StrictStr
    journal_locator: StrictStr
    ledger_locator: StrictStr
    journey_pointer_event_id: StrictStr | None
    journal_artifact_equal: StrictBool
    materialized_ledger: ExecutionContactLedgerBinding | None
    projection_equal: StrictBool
    review: ExecutionContactReview


class ExecutionContactJournal:
    """Fail-soft append-only sink whose failures never escape into Adapter mechanics."""

    def __init__(
        self,
        path: Path,
        execution_id: str,
        *,
        execution_pass: int = 1,
        unit_offset: int = 0,
        processing_interval=None,
        owner_event_id: str | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.execution_id = execution_id
        self.execution_pass = execution_pass
        # Unit ordinals continue across passes so every unit id of the
        # execution stays unique in the composed ledger.
        self._unit_offset = unit_offset
        self._lock = threading.Lock()
        self._descriptors: dict[str, dict[str, Any]] = {}
        self._finished: set[str] = set()
        self._degraded = False
        self._closed = False
        self._sequence = 0
        self._interval = processing_interval
        self._owner_event_id = owner_event_id
        self._binding = None
        if processing_interval is not None:
            if owner_event_id is None:
                raise ValueError("execution contacts require their bound operation")
            processing_interval.start()
            if self.path != processing_interval.journal.path.resolve():
                raise ValueError("execution contacts require the coordinator journal")
            from bio_neb.recording import DomainContactRecord

            processing_interval.journal.record_domain(
                DomainContactRecord(
                    owner_event_id=owner_event_id,
                    after_event_id=processing_interval.journey.head_event_id,
                    operation_id=self.execution_id,
                    iteration=self.execution_pass,
                    phase="started",
                    payload={},
                )
            )
            return
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch(mode=0o600, exist_ok=False)
            content = self.path.read_bytes()
            rows, _, _, _ = _valid_prefix(content)
            self._sequence = len(rows)
            self._closed = bool(rows and rows[-1][0].row_kind == "final")
            for row, _, _ in rows:
                if row.row_kind == "start" and row.unit_id is not None:
                    self._descriptors[row.unit_id] = _descriptor(row)
                elif row.row_kind == "terminal" and row.unit_id is not None:
                    self._finished.add(row.unit_id)
        except (OSError, ValueError):
            self._degraded = True

    @property
    def degraded(self) -> bool:
        return self._degraded

    def retain_receipt(self, *, kind: str, body: Mapping[str, Any]) -> ContactReceipt:
        """Retain a secret-free contact result before its verified journal row."""
        if self._closed:
            raise ValueError("contact journal is not writable")
        content = canonical_json(body).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        path = self.path.parent / "receipts" / (digest + ".json")
        retain_immutable(path, content)
        return ContactReceipt(kind=kind, ref=str(path))

    def retained_transfers(self, journey):
        """Read verified transfer receipts; the caller decides intent eligibility.

        Closed passes must still match their Journey binding. The current pass
        can contain an unfinished step, but only its completed PUTs are returned.
        """
        passes = {}
        try:
            passes[self.execution_pass] = self.path.read_bytes()
        except OSError:
            # Diagnostic read damage grants no reuse, but does not prevent a
            # fresh transfer through the provider's normal collision controls.
            pass
        for event in journey.read():
            if (
                event.event_type != "ExecutionOutcome"
                or event.pass_ >= self.execution_pass
            ):
                continue
            payload = event.payload
            try:
                content = Path(payload.contact_journal).read_bytes()
            except OSError:
                continue
            if (
                len(content) != payload.journal_bytes
                or hashlib.sha256(content).hexdigest() != payload.journal_sha256
            ):
                raise ValueError("retained transfer journal binding changed")
            passes[event.pass_] = content
        for number, content in sorted(passes.items()):
            rows, _, truncated, discrepancies = _valid_prefix(content)
            if truncated or discrepancies:
                raise ValueError("retained transfer journal is malformed")
            invoked = set()
            for row, _, _ in rows:
                if row.execution_id != self.execution_id:
                    raise ValueError("retained transfer execution changed")
                if row.logical_operation != "store.put":
                    continue
                if row.phase == "invoked":
                    invoked.add(f"cj:{number}:{row.sequence}")
                if (
                    row.phase != "verified"
                    or row.invocation_ref not in invoked
                    or row.receipt is None
                    or row.receipt.kind != "transfer_proof"
                ):
                    continue
                path = Path(row.receipt.ref)
                receipt_bytes = path.read_bytes()
                if hashlib.sha256(receipt_bytes).hexdigest() != path.stem:
                    raise ValueError("retained transfer receipt changed")
                body = json.loads(receipt_bytes)
                if body.get("operation") == "put" and body.get("transfer_intent"):
                    yield body, f"cj:{number}:{row.sequence}", row.receipt

    @property
    def closed(self) -> bool:
        return self._closed

    def start(
        self,
        *,
        category: ContactCategory,
        operation: str,
        executable_identity: str,
        credential_names: tuple[str, ...] = (),
        selected_identity: str | None = None,
    ) -> str:
        with self._lock:
            if self._closed:
                raise ValueError("contact journal is not writable")
            ordinal = self._unit_offset + len(self._descriptors) + 1
            digest = self.execution_id.removeprefix("parabricks-execution:")
            unit_id = f"execution-contact:{digest}:{ordinal:08d}"
            descriptor = {
                "unit_id": unit_id,
                "category": category,
                "operation": operation,
                "executable_identity": executable_identity,
                "credential_names": credential_names,
                "selected_identity": selected_identity,
            }
            self._descriptors[unit_id] = descriptor
            self._append(
                ExecutionContactJournalRow(
                    sequence=self._sequence + 1,
                    execution_id=self.execution_id,
                    row_kind="start",
                    disposition="attempted",
                    **descriptor,
                )
            )
            return unit_id

    def finish(
        self,
        unit_id: str | None,
        *,
        disposition: Literal["succeeded", "failed", "unknown"],
        exit_status: int | None = None,
        parse_classification: str | None = None,
        registry_response: RegistryResponseEvidence | None = None,
    ) -> None:
        if self._closed:
            raise ValueError("contact journal is not writable")
        if unit_id is None:
            self._degraded = True
            return
        with self._lock:
            descriptor = self._descriptors.get(unit_id)
            if descriptor is None or unit_id in self._finished:
                self._degraded = True
                return
            self._append(
                ExecutionContactJournalRow(
                    sequence=self._sequence + 1,
                    execution_id=self.execution_id,
                    row_kind="terminal",
                    disposition=disposition,
                    exit_status=exit_status,
                    parse_classification=parse_classification,
                    registry_response=registry_response,
                    **descriptor,
                )
            )
            self._finished.add(unit_id)

    def step(
        self,
        *,
        stage: str,
        step: str,
        phase: Literal["claimed", "invoked", "verified", "failed"],
        attempt: int = 1,
        ceiling_count: int | None = None,
        receipt: ContactReceipt | None = None,
        provider_ids: tuple[str, ...] = (),
        duration: float | None = None,
        logical_operation: str | None = None,
        resource_kind: Literal["bucket", "cpu_job", "h100_job"] | None = None,
        resource_intent: str | None = None,
        invocation_ref: str | None = None,
        failure_class: str | None = None,
        failure_detail: str | None = None,
        failure_policy: ResolvedFailurePolicy | None = None,
    ) -> str:
        """Append engine/effect evidence to this journal, never to the path.

        A claimed/invoked row is written before contact; a submission receipt is
        written as soon as the provider returns its identity. The returned row
        locator is suitable for a path artifact's proof_ref.
        """
        with self._lock:
            if self._closed or self._degraded:
                raise ValueError("contact journal is not writable")
            sequence = self._sequence + 1
            self._append(
                ExecutionContactJournalRow(
                    sequence=sequence,
                    execution_id=self.execution_id,
                    row_kind="step",
                    stage=stage,
                    step=step,
                    phase=phase,
                    attempt=attempt,
                    ceiling_count=ceiling_count,
                    receipt=receipt,
                    provider_ids=provider_ids,
                    duration=duration,
                    logical_operation=logical_operation,
                    resource_kind=resource_kind,
                    resource_intent=resource_intent,
                    invocation_ref=invocation_ref,
                    failure_class=failure_class,
                    failure_detail=failure_detail,
                    failure_policy=failure_policy,
                )
            )
            if self._degraded:
                raise OSError("contact journal step was not retained")
            return f"cj:{self.execution_pass}:{sequence}"

    def finalize(self) -> ExecutionContactJournalBinding | None:
        with self._lock:
            if self._closed:
                return self.binding()
            try:
                prefix = self.path.read_bytes()
            except OSError:
                self._degraded = True
                return None
            existing, _, _, _ = _valid_prefix(prefix)
            if existing and existing[-1][0].row_kind == "final":
                self._closed = True
                return self.binding()
            starts = set(self._descriptors)
            terminal = set(self._finished)
            final = ExecutionContactJournalRow(
                sequence=self._sequence + 1,
                execution_id=self.execution_id,
                row_kind="final",
                prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                complete=not self._degraded
                and starts == terminal
                and not _ScannedJournal(
                    prefix, self.execution_id, self.execution_pass
                ).discrepancies,
            )
            self._append(final)
            self._closed = not self._degraded
            if self._interval is not None:
                content = self._interval.journal.snapshot()
                digest = hashlib.sha256(content).hexdigest()
                snapshot = self.path.with_name(f"execution-prefix-{digest}.jsonl")
                retain_immutable(snapshot, content)
                self._binding = ExecutionContactJournalBinding(
                    locator=str(snapshot),
                    byte_count=len(content),
                    sha256=digest,
                    execution_id=self.execution_id,
                )
                self._closed = True
            return self.binding()

    def binding(self) -> ExecutionContactJournalBinding | None:
        if self._interval is not None:
            return self._binding
        try:
            content = self.path.read_bytes()
        except OSError:
            return None
        if not content:
            return None
        return ExecutionContactJournalBinding(
            locator=str(self.path),
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            execution_id=self.execution_id,
        )

    def _append(self, row: ExecutionContactJournalRow) -> None:
        if self._closed:
            self._degraded = True
            return
        content = (canonical_json(row.model_dump(mode="json")) + "\n").encode("utf-8")
        try:
            if self._interval is None:
                append_durable(self.path, content)
            else:
                from bio_neb.recording import DomainContactRecord

                self._interval.journal.record_domain(
                    DomainContactRecord(
                        owner_event_id=self._owner_event_id,
                        after_event_id=self._interval.journey.head_event_id,
                        operation_id=self.execution_id,
                        iteration=self.execution_pass,
                        phase="closed" if row.row_kind == "final" else "record",
                        complete=row.complete if row.row_kind == "final" else None,
                        payload=row.model_dump(mode="json"),
                    )
                )
        except OSError:
            self._degraded = True
            return
        self._sequence += 1


def _descriptor(row: ExecutionContactJournalRow) -> dict[str, Any]:
    return {
        "unit_id": row.unit_id,
        "category": row.category,
        "operation": row.operation,
        "executable_identity": row.executable_identity,
        "credential_names": row.credential_names,
        "selected_identity": row.selected_identity,
    }


def _valid_prefix(
    content: bytes,
) -> tuple[
    list[tuple[ExecutionContactJournalRow, int, int]], int, bool, tuple[str, ...]
]:
    rows: list[tuple[ExecutionContactJournalRow, int, int]] = []
    offset = 0
    discrepancies: list[str] = []
    for raw in content.splitlines(keepends=True):
        if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
            discrepancies.append("truncated_tail")
            break
        try:
            text = raw[:-1].decode("utf-8")
            value = json.loads(text)
            if canonical_json(value) != text:
                raise ValueError("noncanonical")
            if "run_id" in value and "pass_id" in value:
                from bio_neb.recording import ContactRow

                envelope = ContactRow.model_validate(value)
                if envelope.domain is None:
                    offset += len(raw)
                    continue
                if envelope.domain.phase == "started":
                    offset += len(raw)
                    continue
                value = envelope.domain.payload
                if value.get("execution_id") != envelope.domain.operation_id:
                    raise ValueError("domain operation identity changed")
            row = ExecutionContactJournalRow.model_validate(value, strict=True)
        except (UnicodeError, json.JSONDecodeError, ValueError):
            discrepancies.append("invalid_row")
            break
        start = offset
        offset += len(raw)
        rows.append((row, start, offset))
    truncated = offset != len(content)
    return rows, offset, truncated, tuple(discrepancies)


def _measured(unit_ids: Sequence[str], *bindings: str) -> MeasuredContactMetric:
    unique = tuple(dict.fromkeys(unit_ids))
    return MeasuredContactMetric(
        count=len(unique), unit_ids=unique, evidence_bindings=tuple(bindings)
    )


def _unmeasurable(
    reason: str, *bindings: str, observed_unit_ids: Sequence[str] = ()
) -> UnmeasurableContactMetric:
    observed = tuple(dict.fromkeys(observed_unit_ids))
    return UnmeasurableContactMetric(
        reason=reason,
        evidence_bindings=tuple(bindings),
        observed_unit_ids=observed,
        observed_count=len(observed),
    )


def _event_ids(events: Sequence[Mapping[str, Any]], event_type: str) -> tuple[str, ...]:
    return tuple(
        str(event.get("event_id"))
        for event in events
        if event.get("event_type") == event_type
        and isinstance(event.get("event_id"), str)
    )


def _vcf_query_units(
    events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], bool, bool]:
    from bio_neb.application.planning_read import PrefixSnapshot
    from bio_neb.investigation.investigation_observation import (
        validate_investigation_collection_evidence_fact,
        validate_investigation_evidence_fact,
        validate_investigation_work_item_started_fact,
    )
    from bio_neb.investigation.item_evidence import (
        InvestigationItemWindowRecorded,
        validate_item_evidence_fact,
    )

    context = None
    evidence = {}
    starts = {}
    units = []
    observation_types = {
        "InvestigationWorkItemStarted",
        "InvestigationEvidenceRecorded",
        "InvestigationCollectionEvidenceRecorded",
    }
    for index, event in enumerate(events):
        window = validate_item_evidence_fact(event)
        if isinstance(window, InvestigationItemWindowRecorded):
            # An approved item's window reads the prepared VCF (ADR 0047).
            if window.result.query_invocation is not None:
                units.append(f"journey-vcf:{event['event_id']}")
            continue
        if event.get("event_type") not in observation_types:
            continue
        if index == 0 or event.get("parent_ids") != [events[index - 1]["event_id"]]:
            raise ValueError("VCF accounting lost its exact observation parent")
        event_authority_id = event.get("payload", {}).get("authority_id") or event.get(
            "payload", {}
        ).get("authority", {}).get("authority_id")
        if context is None:
            context = PrefixSnapshot(events).investigation
        # Observation facts are no Investigation facts: every observation of one
        # batch reads the authorities of the one prefix before the batch.
        anchor = next(
            (
                item
                for item in reversed(events[:index])
                if item.get("event_type") not in observation_types
            ),
            None,
        )
        authority = next(
            (
                grant
                for grant in (
                    context.through(anchor["event_id"]).projection.authorities
                    if anchor is not None
                    else ()
                )
                if grant.authority_id == event_authority_id
            ),
            None,
        )
        if authority is None:
            raise ValueError("VCF accounting lacks current Investigation authority")
        previous = events[index - 1]
        started = validate_investigation_work_item_started_fact(event)
        if started is not None:
            key = (started.authority_id, started.work_item_id)
            if key in starts or key in evidence:
                raise ValueError("VCF accounting repeated or reordered a Work item")
            if (
                started.authority_id != authority.authority_id
                or not 1
                <= started.selected_ordinal
                <= len(authority.authorized_work_items)
            ):
                raise ValueError("VCF Work item start changed its collection authority")
            work_item = authority.authorized_work_items[started.selected_ordinal - 1]
            if (
                started.work_item_id != work_item.work_item_id
                or started.sample_id != work_item.sample_id
                or previous.get("event_type")
                not in observation_types
                | {
                    "InvestigationAuthorityGranted",
                    "InvestigationReviewRequested",
                    "InvestigationAssertionRecorded",
                    "InvestigationDownloadReviewAnswered",
                }
            ):
                raise ValueError("VCF Work item start changed its selected member")
            starts[key] = (event, started)
        if event.get("event_type") == "InvestigationEvidenceRecorded":
            fact = validate_investigation_evidence_fact(event)
            key = (fact.authority.authority_id, fact.work_item.work_item_id)
            if key in evidence:
                raise ValueError("VCF accounting repeated current observation evidence")
            if fact.authority != authority:
                raise ValueError("VCF observation changed its retained authority")
            start = starts.get(key)
            if (
                start is None
                or previous["event_id"] != start[0]["event_id"]
                or start[1].target != fact.sample_observation.result.target
            ):
                raise ValueError("VCF observation lost its exact Work item start")
            evidence[key] = fact
            if fact.sample_observation.result.query_invocation is not None:
                units.append(f"journey-vcf:{event['event_id']}")
        if event.get("event_type") == "InvestigationCollectionEvidenceRecorded":
            collection = validate_investigation_collection_evidence_fact(event)
            if collection.authority != authority or previous.get(
                "event_type"
            ) not in observation_types | {
                "InvestigationAuthorityGranted",
                "InvestigationReviewRequested",
                "InvestigationDownloadReviewAnswered",
            }:
                raise ValueError(
                    "VCF collection changed its retained authority or parent"
                )
            if context.through(event["event_id"]).read.evidence != collection:
                raise ValueError("VCF accounting changed its retained collection")
    return (
        tuple(units),
        bool(evidence) and set(starts) <= set(evidence),
        bool(set(starts) - set(evidence)),
    )


def _resource_discrepancies(rows):
    reasons = []
    intents = {}
    for row in rows:
        if row.resource_kind is None:
            continue
        if row.ceiling_count is not None and row.ceiling_count != len(
            set(row.provider_ids)
        ):
            reasons.append("resource_attestation_conflict")
        if row.resource_intent and row.provider_ids:
            key = row.resource_kind, row.resource_intent
            ids = set(row.provider_ids)
            if intents.setdefault(key, ids) != ids:
                reasons.append("resource_intent_identity_conflict")
    return reasons


def _logical_contact_discrepancies(rows, execution_pass):
    discrepancies = []
    invocations = {
        f"cj:{execution_pass}:{row.sequence}": row
        for row in rows
        if row.phase == "invoked" and row.logical_operation
    }
    result_counts = {}
    for row in rows:
        if row.invocation_ref is not None:
            try:
                invoked = invocations[row.invocation_ref]
            except KeyError:
                discrepancies.append("logical_result_without_invocation")
                continue
            result_counts[row.invocation_ref] = (
                result_counts.get(row.invocation_ref, 0) + 1
            )
            if row.phase not in {"verified", "failed"} or any(
                getattr(row, name) != getattr(invoked, name)
                for name in (
                    "logical_operation",
                    "resource_kind",
                    "resource_intent",
                    "stage",
                    "step",
                    "attempt",
                )
            ):
                discrepancies.append("logical_result_identity_drift")
    if any(sequence not in result_counts for sequence in invocations):
        discrepancies.append("unfinished_logical_contact")
    if any(count != 1 for count in result_counts.values()):
        discrepancies.append("duplicate_logical_result")
    return discrepancies


class _ScannedJournal:
    """The strict row reading of one pass's journal bytes."""

    def __init__(
        self, journal_bytes: bytes, expected_execution_id: str, execution_pass: int = 1
    ) -> None:
        rows_with_offsets, prefix_count, truncated, prefix_discrepancies = (
            _valid_prefix(journal_bytes)
        )
        self.journal_bytes = journal_bytes
        self.rows = [item[0] for item in rows_with_offsets]
        self.prefix_count = prefix_count
        self.truncated = truncated
        self.discrepancies = list(prefix_discrepancies)
        for line in journal_bytes.splitlines():
            try:
                value = json.loads(line)
            except (ValueError, UnicodeError):
                break
            domain = value.get("domain")
            if domain is not None and (
                domain.get("operation_id") != expected_execution_id
                or domain.get("iteration") != execution_pass
            ):
                self.discrepancies.append("domain_operation_scope_drift")
        self.starts: dict[str, ExecutionContactJournalRow] = {}
        self.terminals: dict[str, ExecutionContactJournalRow] = {}
        self.final: ExecutionContactJournalRow | None = None
        for index, (row, start, _) in enumerate(rows_with_offsets, start=1):
            if row.sequence != index or row.execution_id != expected_execution_id:
                self.discrepancies.append("sequence_or_execution_identity_drift")
            if self.final is not None:
                self.discrepancies.append("row_after_final")
            if row.row_kind == "start" and row.unit_id is not None:
                if row.unit_id in self.starts:
                    self.discrepancies.append("duplicate_unit_start")
                self.starts[row.unit_id] = row
            elif row.row_kind == "terminal" and row.unit_id is not None:
                if row.unit_id not in self.starts or row.unit_id in self.terminals:
                    self.discrepancies.append("invalid_unit_terminal")
                elif _descriptor(row) != _descriptor(self.starts[row.unit_id]):
                    self.discrepancies.append("unit_identity_drift")
                self.terminals[row.unit_id] = row
            elif row.row_kind == "final":
                if self.final is not None or index != len(self.rows):
                    self.discrepancies.append("invalid_final_position")
                self.final = row
                if (
                    row.prefix_sha256
                    != hashlib.sha256(journal_bytes[:start]).hexdigest()
                ):
                    self.discrepancies.append("final_prefix_drift")
        step_ends = {}
        for row in self.rows:
            if row.row_kind == "step":
                step_ends[row.stage, row.step, row.attempt] = row.phase
        if any(phase not in {"verified", "failed"} for phase in step_ends.values()):
            self.discrepancies.append("unfinished_step")
        self.discrepancies.extend(
            _logical_contact_discrepancies(self.rows, execution_pass)
        )
        self.discrepancies.extend(_resource_discrepancies(self.rows))
        self.unmatched = tuple(
            unit_id for unit_id in self.starts if unit_id not in self.terminals
        )
        if self.unmatched:
            self.discrepancies.append("unmatched_unit_starts")

    def binding_matches(
        self,
        binding: ExecutionContactJournalBinding | None,
        expected_execution_id: str,
    ) -> bool:
        return bool(
            binding is not None
            and binding.execution_id == expected_execution_id
            and binding.byte_count == len(self.journal_bytes)
            and binding.sha256 == hashlib.sha256(self.journal_bytes).hexdigest()
        )

    @property
    def complete(self) -> bool:
        return bool(
            not self.truncated
            and not self.discrepancies
            and self.final is not None
            and self.final.complete is True
            and not self.unmatched
        )


def read_complete_execution_contacts(
    content: bytes, execution_id: str, *, execution_pass: int = 1
) -> tuple[ExecutionContactJournalRow, ...]:
    """Read one closed journal, refusing any incomplete or inconsistent evidence."""
    scanned = _ScannedJournal(content, execution_id, execution_pass)
    if not scanned.complete or scanned.discrepancies:
        raise ValueError("execution_contact_journal_incomplete_or_inconsistent")
    return tuple(scanned.rows)


def _pointer_validity(
    pointer_event: Mapping[str, Any],
    *,
    binding: ExecutionContactJournalBinding | None,
    expected_execution_id: str,
    execution_pass: int,
    final: ExecutionContactJournalRow | None,
    discrepancies: list[str],
) -> tuple[bool, bool]:
    """Return (valid, complete) for one pass's pointer, appending its discrepancies."""

    valid = False
    complete = False
    declared: bool | None = None
    try:
        pointer = ExecutionContactJourneyPointer.model_validate(
            pointer_event.get("payload", {}), strict=True
        )
        valid = bool(
            binding is not None
            and pointer.execution_id == expected_execution_id
            and pointer.journal == binding
            and pointer.execution_pass == execution_pass
            and len(pointer_event.get("parent_ids", [])) == 1
            and len(pointer_event.get("artifact_references", [])) == 1
        )
        complete = valid and pointer.journal_complete is True
        declared = pointer.journal_complete
    except ValueError:
        if _unsupported_accounting_version(
            pointer_event.get("payload"),
            "bio-neb.execution-contact-journey-pointer/2",
        ):
            discrepancies.append("unsupported_execution_contact_accounting_version")
    if not valid:
        discrepancies.append("execution_contact_pointer_drift")
    if valid and final is not None and declared is not final.complete:
        discrepancies.append("execution_contact_pointer_completion_drift")
    return valid, complete


def compile_execution_contact_evidence(  # noqa: C901, PLR0915 - Existing orchestration complexity; contract-base migration preserves this flow.
    *,
    journal_bytes: bytes,
    binding: ExecutionContactJournalBinding | None,
    expected_execution_id: str,
    journey_events: Sequence[Mapping[str, Any]],
    journey_binding: str,
    prior_passes: Sequence[tuple[bytes, ExecutionContactJournalBinding | None]] = (),
) -> ExecutionContactReview:
    """Purely compile retained bytes and Journey facts without contact or mutation.

    ``journal_bytes`` and ``binding`` are the latest pass's; ``prior_passes`` are
    the earlier passes of the same execution in order, each as its journal bytes
    and its pointer's binding (``None`` when the file no longer matches the
    retained artifact). The review composes every pass: units in pass order, one
    pointer per pass, and the Journey's create decisions cross-checked against
    the contacts of all passes together.
    """

    passes = [
        _ScannedJournal(prior_bytes, expected_execution_id, number)
        for number, (prior_bytes, _) in enumerate(prior_passes, start=1)
    ]
    latest = _ScannedJournal(
        journal_bytes, expected_execution_id, len(prior_passes) + 1
    )
    passes.append(latest)
    bindings = [*(prior_binding for _, prior_binding in prior_passes), binding]
    rows = latest.rows
    discrepancies: list[str] = []
    for scanned in passes:
        discrepancies.extend(scanned.discrepancies)
    discrepancies.extend(
        _resource_discrepancies(row for scanned in passes for row in scanned.rows)
    )
    binding_matches = latest.binding_matches(binding, expected_execution_id)
    prior_bindings_match = all(
        scanned.binding_matches(prior_binding, expected_execution_id)
        for scanned, prior_binding in zip(passes[:-1], bindings[:-1], strict=True)
    )
    if not binding_matches or not prior_bindings_match:
        discrepancies.append("missing_or_drifted_binding")
    if any(not scanned.complete for scanned in passes[:-1]):
        discrepancies.append("prior_pass_journal_incomplete")

    starts: dict[str, ExecutionContactJournalRow] = {}
    terminals: dict[str, ExecutionContactJournalRow] = {}
    for scanned in passes:
        for unit_id, row in scanned.starts.items():
            if unit_id in starts:
                discrepancies.append("duplicate_unit_start")
            starts[unit_id] = row
        terminals.update(scanned.terminals)
    final = latest.final
    truncated = latest.truncated
    prefix_count = latest.prefix_count
    unmatched = tuple(unit_id for scanned in passes for unit_id in scanned.unmatched)
    path_mode = any(
        event.get("event_type") == "ExecutionBound" and "kind" in event
        for event in journey_events
    )
    if path_mode:
        outcome_events = {}
        for event in journey_events:
            if event.get("event_type") == "ExecutionOutcome":
                outcome_events[event["pass"]] = event
        pointer_complete = len(outcome_events) == len(passes)
        for number, pass_binding in enumerate(bindings, start=1):
            payload = outcome_events.get(number, {}).get("payload", {})
            if (
                pass_binding is None
                or payload.get("journal_sha256") != pass_binding.sha256
                or payload.get("journal_bytes") != pass_binding.byte_count
                or payload.get("contact_journal") != pass_binding.locator
            ):
                pointer_complete = False
        if not pointer_complete:
            discrepancies.append("missing_or_drifted_outcome_journal_binding")
    else:
        pointer_events = [
            event
            for event in journey_events
            if event.get("event_type") == "ExecutionContactJournalBound"
        ]
        pointer_complete = False
        if len(pointer_events) != len(passes):
            discrepancies.append("missing_or_duplicate_execution_contact_pointer")
        else:
            validity = [
                _pointer_validity(
                    pointer_event,
                    binding=pass_binding,
                    expected_execution_id=expected_execution_id,
                    execution_pass=execution_pass,
                    final=scanned.final,
                    discrepancies=discrepancies,
                )
                for execution_pass, (pointer_event, pass_binding, scanned) in enumerate(
                    zip(pointer_events, bindings, passes, strict=True), start=1
                )
            ]
            pointer_complete = all(complete for _, complete in validity)
    complete = bool(
        binding_matches
        and pointer_complete
        and not truncated
        and final is not None
        and final.complete is True
        and not unmatched
        and not discrepancies
    )
    units = tuple(
        ExecutionContactUnit(
            **_descriptor(start),
            registry_response=(
                terminals[unit_id].registry_response if unit_id in terminals else None
            ),
            disposition=(
                terminals[unit_id].disposition if unit_id in terminals else "unknown"
            ),
            exit_status=(
                terminals[unit_id].exit_status if unit_id in terminals else None
            ),
            parse_classification=(
                terminals[unit_id].parse_classification
                if unit_id in terminals
                else None
            ),
        )
        for unit_id, start in starts.items()
    )
    # One evidence binding per pass, in pass order; a single pass keeps its one.
    evidence_bindings = tuple(
        f"journal:{pass_binding.sha256}"
        if pass_binding is not None
        else f"execution:{expected_execution_id}"
        for pass_binding in bindings
    )
    if complete:
        provider = _measured(
            [unit.unit_id for unit in units if unit.category == "provider"],
            *evidence_bindings,
        )
        storage = _measured(
            [unit.unit_id for unit in units if unit.category == "storage"],
            *evidence_bindings,
        )
        credential = _measured(
            [unit.unit_id for unit in units if unit.category == "credential"],
            *evidence_bindings,
        )
    else:
        reason = "execution_contact_evidence_incomplete"
        provider = _unmeasurable(
            reason,
            *evidence_bindings,
            observed_unit_ids=[
                unit.unit_id for unit in units if unit.category == "provider"
            ],
        )
        storage = _unmeasurable(
            reason,
            *evidence_bindings,
            observed_unit_ids=[
                unit.unit_id for unit in units if unit.category == "storage"
            ],
        )
        credential = _unmeasurable(
            reason,
            *evidence_bindings,
            observed_unit_ids=[
                unit.unit_id for unit in units if unit.category == "credential"
            ],
        )

    from bio_neb.application.queries import model_contact_units

    model_ids, model_complete = model_contact_units(journey_events)
    model = (
        _measured(model_ids, journey_binding)
        if model_complete
        else _unmeasurable(
            "model_contact_evidence_incomplete"
            if model_ids
            else "model_contact_evidence_absent",
            journey_binding,
            observed_unit_ids=model_ids,
        )
    )
    from bio_neb.agentic_journey.source_contacts import source_contact_units

    source_ids, source_complete = source_contact_units(journey_events)
    source = (
        _measured(source_ids, journey_binding)
        if source_complete
        else _unmeasurable(
            "source_contact_evidence_incomplete"
            if source_ids
            else "source_contact_evidence_absent",
            journey_binding,
            observed_unit_ids=source_ids,
        )
    )

    if all(
        isinstance(metric, MeasuredContactMetric)
        for metric in (provider, storage, model, source)
    ):
        assert isinstance(provider, MeasuredContactMetric)
        assert isinstance(storage, MeasuredContactMetric)
        assert isinstance(model, MeasuredContactMetric)
        assert isinstance(source, MeasuredContactMetric)
        network_ids = tuple(
            dict.fromkeys(
                (
                    *provider.unit_ids,
                    *storage.unit_ids,
                    *model.unit_ids,
                    *source.unit_ids,
                )
            )
        )
        network: ContactMetric = _measured(
            network_ids, *evidence_bindings, journey_binding
        )
    else:
        network = _unmeasurable(
            "deduplicated_external_contact_set_incomplete",
            *evidence_bindings,
            journey_binding,
            observed_unit_ids=tuple(
                unit_id
                for metric in (provider, storage, model, source)
                for unit_id in (
                    metric.unit_ids
                    if isinstance(metric, MeasuredContactMetric)
                    else metric.observed_unit_ids
                )
            ),
        )

    # Resource counts come from current journal receipts below. Archived stage
    # facts do not become measured resource identities in this compiler.
    bucket_ids, cpu_ids, h100_ids, cache_hit = (), (), (), False
    fresh_bucket: ContactMetric = (
        _measured(
            tuple(f"journey-create:{item}" for item in bucket_ids), journey_binding
        )
        if bucket_ids
        else _unmeasurable("fresh_bucket_create_evidence_absent", journey_binding)
    )
    cpu_create: ContactMetric = (
        _measured(tuple(f"journey-create:{item}" for item in cpu_ids), journey_binding)
        if cpu_ids
        else _measured((), journey_binding)
        if cache_hit
        else _unmeasurable("cpu_create_decision_evidence_absent", journey_binding)
    )
    h100_create: ContactMetric = (
        _measured(tuple(f"journey-create:{item}" for item in h100_ids), journey_binding)
        if h100_ids
        else _unmeasurable("h100_create_evidence_absent", journey_binding)
    )

    logical_attempts = []
    for number, scanned in enumerate(passes, start=1):
        results = {}
        for result_row in scanned.rows:
            if result_row.invocation_ref:
                results.setdefault(result_row.invocation_ref, []).append(result_row)
        for row in scanned.rows:
            if row.phase != "invoked" or row.logical_operation is None:
                continue
            ref = f"cj:{number}:{row.sequence}"
            matches = results.get(ref, [])
            result = matches[0] if len(matches) == 1 else None
            logical_attempts.append(
                LogicalContactAttempt(
                    invocation_ref=ref,
                    execution_pass=number,
                    operation=row.logical_operation,
                    resource_kind=row.resource_kind,
                    resource_intent=row.resource_intent,
                    stage=row.stage,
                    step=row.step,
                    attempt=row.attempt,
                    disposition="inconsistent"
                    if len(matches) > 1
                    else "unfinished"
                    if result is None
                    else "failed"
                    if result.phase == "failed"
                    else "succeeded",
                    provider_ids=tuple(
                        sorted(
                            {item for match in matches for item in match.provider_ids}
                        )
                    ),
                    resource_count=None if result is None else result.ceiling_count,
                    failure_class=None if result is None else result.failure_class,
                    failure_detail=None if result is None else result.failure_detail,
                    receipt=row.receipt if result is None else result.receipt,
                )
            )
    if path_mode:

        def resources(kind):
            attempts = [a for a in logical_attempts if a.resource_kind == kind]
            ids = {
                item
                for scanned in passes
                for row in scanned.rows
                if row.resource_kind == kind
                for item in row.provider_ids
            }
            uncertain = not complete
            if any(
                attempt.disposition != "succeeded" or attempt.resource_count is None
                for attempt in attempts
            ):
                uncertain = True
            # Source-era step rows cannot establish new logical-operation meaning.
            if any(
                row.row_kind == "step"
                and row.phase == "invoked"
                and row.logical_operation is None
                for scanned in passes
                for row in scanned.rows
            ):
                uncertain = True
            return (
                _unmeasurable(
                    "resource_result_incomplete",
                    *evidence_bindings,
                    observed_unit_ids=sorted(ids),
                )
                if uncertain
                else _measured(sorted(ids), *evidence_bindings)
            )

        fresh_bucket = resources("bucket")
        cpu_create = resources("cpu_job")
        h100_create = resources("h100_job")
        complete = complete and not discrepancies
        # Step wrappers are logical operations, not a count of wire exchanges.
        # Registry rows expose HTTP exchanges; CLI subprocesses still hide wire totals.
        network = _unmeasurable(
            "cli_registry_contact_coverage_incomplete",
            *evidence_bindings,
            observed_unit_ids=tuple(unit.unit_id for unit in units),
        )
        if not units:
            provider = _unmeasurable(
                "physical_contact_evidence_absent", *evidence_bindings
            )
            storage = _unmeasurable(
                "physical_contact_evidence_absent", *evidence_bindings
            )
            credential = _unmeasurable(
                "physical_contact_evidence_absent", *evidence_bindings
            )

    if complete and not path_mode:
        observed_bucket_creates = sum(
            unit.operation == "storage.bucket.create" for unit in units
        )
        observed_job_creates = sum(unit.operation == "ai.job.create" for unit in units)
        if fresh_bucket.status == "measured" and (
            observed_bucket_creates != fresh_bucket.count
        ):
            discrepancies.append("fresh_bucket_create_cross_check_drift")
        if (
            cpu_create.status == "measured"
            and h100_create.status == "measured"
            and observed_job_creates != cpu_create.count + h100_create.count
        ):
            discrepancies.append("job_create_cross_check_drift")
        complete = not discrepancies

    vcf_ids, vcf_observed, vcf_incomplete = _vcf_query_units(journey_events)
    vcf_query: ContactMetric = (
        _measured(vcf_ids, journey_binding)
        if vcf_observed or complete and not vcf_incomplete
        else _unmeasurable(
            "execution_contact_evidence_incomplete",
            journey_binding,
            observed_unit_ids=vcf_ids,
        )
    )

    ledger = FreshFullPredecessorContactLedger(
        journal_binding=binding,
        prior_pass_journal_bindings=tuple(
            prior_binding
            for prior_binding in bindings[:-1]
            if prior_binding is not None
        ),
        journey_binding=journey_binding,
        credential_access=credential,
        model_contact=model,
        source_contact=source,
        storage_contact=storage,
        provider_contact=provider,
        network_contact=network,
        fresh_bucket=fresh_bucket,
        cpu_preparation_create=cpu_create,
        h100_create=h100_create,
        post_execution_vcf_query=vcf_query,
    )
    constrained = (
        fresh_bucket.status == "measured"
        and fresh_bucket.count == 1
        and cpu_create.status == "measured"
        and cpu_create.count in {0, 1}
        and h100_create.status == "measured"
        and h100_create.count == 1
        and vcf_query.count == 0
    )
    # An artifact/flow fixture may retain unknown physical contact totals. The
    # resource ceilings above still require measured create counts.
    return ExecutionContactReview(
        execution_id=expected_execution_id,
        binding=binding,
        binding_matches=binding_matches,
        complete=complete,
        fixture_eligible=complete and constrained,
        valid_row_count=len(rows),
        longest_valid_prefix_byte_count=prefix_count,
        truncated_tail=truncated,
        unmatched_unit_ids=unmatched,
        discrepancies=tuple(dict.fromkeys(discrepancies)),
        units=units,
        logical_attempts=tuple(logical_attempts),
        logical_results=tuple(
            row for scanned in passes for row in scanned.rows if row.invocation_ref
        ),
        ledger=ledger,
    )


def _journey_binding(events: Sequence[Mapping[str, Any]]) -> str:
    return (
        "journey:"
        + hashlib.sha256(
            b"".join(
                (canonical_json(dict(event)) + "\n").encode("utf-8") for event in events
            )
        ).hexdigest()
    )


def _write_ledger_binding(path: Path, content: bytes) -> ExecutionContactLedgerBinding:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(path, content)
    return ExecutionContactLedgerBinding(
        locator=str(path),
        byte_count=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _journal_complete(
    content: bytes, binding: ExecutionContactJournalBinding, execution_pass: int = 1
) -> bool:
    scanned = _ScannedJournal(content, binding.execution_id, execution_pass)
    return scanned.complete and scanned.binding_matches(binding, binding.execution_id)


def retained_contact_unit_count(journey) -> int:
    """Reserve unique unit IDs after the distinct closed contact passes."""
    paths = {
        event.payload.contact_journal
        for event in journey.read()
        if event.event_type == "ExecutionOutcome"
    }
    return sum(
        row.row_kind == "start"
        for path in paths
        for row, _, _ in _valid_prefix(Path(path).read_bytes())[0]
    )


def _retained_path_outcomes(journey, events, bound, path_reasons):
    outcomes = {}
    closed = {}
    for event in events:
        if event["event_type"] != "ExecutionOutcome":
            continue
        payload = event["payload"]
        locator = payload["contact_journal"]
        if locator in closed:
            previous = closed[locator]
            # A human abort closes the answer pass without making contacts.
            # It must retain exactly the already closed journal binding.
            if (
                event["grounding"] != "human"
                or payload["disposition"] != "failed"
                or any(
                    payload.get(key) != previous.get(key)
                    for key in ("journal_sha256", "journal_bytes")
                )
            ):
                path_reasons.append("execution_contact_reused_binding_drift")
            continue
        closed[locator] = payload
        outcomes[event["pass"]] = event
    if journey._owner is not None:
        from bio_neb.recording import read_passes

        intervals = read_passes(journey._owner)
        for interval in intervals:
            for row in interval.contacts:
                if (
                    row.domain is None
                    or row.domain.operation_id != bound["execution_id"]
                ):
                    continue
                number = row.domain.iteration
                outcomes.setdefault(
                    number,
                    {
                        "event_id": None,
                        "payload": {
                            "contact_journal": str(
                                journey.file.parent
                                / "contacts"
                                / interval.pass_id
                                / "journal.jsonl"
                            )
                        },
                    },
                )
        if not outcomes and intervals:
            # Failure to retain the first mechanical row still leaves the
            # admitted bound operation and its interrupted processing interval.
            interval = intervals[-1]
            outcomes[1] = {
                "event_id": None,
                "payload": {
                    "contact_journal": str(
                        journey.file.parent
                        / "contacts"
                        / interval.pass_id
                        / "journal.jsonl"
                    )
                },
            }
        return outcomes
    root = (
        journey.file.parent
        / "execution-contact"
        / bound["execution_id"].rsplit(":", 1)[1]
    )
    discovered = (
        {1: root / "journal.jsonl"} if (root / "journal.jsonl").exists() else {}
    )
    for path in root.glob("pass-*/journal.jsonl"):
        suffix = path.parent.name.removeprefix("pass-")
        if suffix.isdecimal() and int(suffix) > 1 and suffix == str(int(suffix)):
            discovered[int(suffix)] = path
    for number, path in discovered.items():
        outcomes.setdefault(
            number, {"event_id": None, "payload": {"contact_journal": str(path)}}
        )
    if not outcomes:
        outcomes[1] = {
            "event_id": None,
            "payload": {"contact_journal": str(root / "journal.jsonl")},
        }
    if sorted(outcomes) != list(range(1, max(outcomes) + 1)):
        path_reasons.append("execution_contact_pass_gap")
        for number in range(1, max(outcomes) + 1):
            path = (
                root / "journal.jsonl"
                if number == 1
                else root / f"pass-{number}" / "journal.jsonl"
            )
            outcomes.setdefault(
                number, {"event_id": None, "payload": {"contact_journal": str(path)}}
            )
    return outcomes


def _path_accounting(journey):
    """Reconcile canonical retained passes with their available outcome bindings."""
    events = [event.model_dump(mode="json", by_alias=True) for event in journey.read()]
    bound = next(
        event["payload"] for event in events if event["event_type"] == "ExecutionBound"
    )
    path_reasons = []
    outcomes = _retained_path_outcomes(journey, events, bound, path_reasons)
    passes = []
    retained_passes = []
    receipt_drift = False
    shared_prefixes = {}
    if journey._owner is not None:
        from bio_neb.recording import read_passes

        shared_prefixes = {
            interval.pass_id: b"".join(
                (
                    canonical_json(row.model_dump(mode="json", exclude_unset=True))
                    + "\n"
                ).encode()
                for row in interval.contacts
            )
            for interval in read_passes(journey._owner)
        }
    for number, event in sorted(outcomes.items()):
        payload = event["payload"]
        path = Path(payload["contact_journal"])
        expected = (
            journey.file.parent
            / "execution-contact"
            / bound["execution_id"].rsplit(":", 1)[1]
        )
        if number > 1:
            expected = expected / f"pass-{number}"
        if journey._owner is not None:
            root = journey.file.parent / "contacts"
            if path.parent.parent != root or not (
                path.name == "journal.jsonl"
                or path.name
                == f"execution-prefix-{payload.get('journal_sha256')}.jsonl"
            ):
                path_reasons.append("execution_contact_locator_drift")
        elif path != expected / "journal.jsonl":
            path_reasons.append("execution_contact_locator_drift")
            path = expected / "journal.jsonl"
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            content = b""
            path_reasons.append("journal_missing")
        except OSError:
            content = b""
            path_reasons.append("journal_unreadable")
        if (
            journey._owner is not None
            and payload.get("journal_sha256") is not None
            and not shared_prefixes.get(path.parent.name, b"").startswith(content)
        ):
            path_reasons.append("execution_contact_prefix_drift")
        binding = None
        if payload.get("journal_sha256") and payload.get("journal_bytes"):
            binding = ExecutionContactJournalBinding(
                locator=str(path),
                execution_id=bound["execution_id"],
                sha256=payload["journal_sha256"],
                byte_count=payload["journal_bytes"],
            )
        scanned = _ScannedJournal(content, bound["execution_id"], number)
        reasons = list(scanned.discrepancies)
        if binding is None:
            reasons.append("outcome_binding_absent")
        elif not scanned.binding_matches(binding, bound["execution_id"]):
            reasons.append("journal_binding_drift")
        if scanned.final is None:
            reasons.append("journal_unclosed")
        path_reasons.extend(reasons)
        retained_passes.append(
            RetainedContactPass(
                execution_pass=number,
                journal_locator=str(path),
                outcome_event_id=event["event_id"],
                disposition=payload.get("disposition"),
                journal_complete=scanned.complete,
                binding_matches=scanned.binding_matches(binding, bound["execution_id"]),
                reasons=tuple(reasons),
            )
        )
        for row, _, _ in _valid_prefix(content)[0]:
            ref = row.receipt.ref if row.receipt is not None else None
            if ref is not None and Path(ref).is_absolute():
                receipt_path = Path(ref)
                try:
                    if (
                        receipt_path.parent != path.parent / "receipts"
                        or receipt_path.name
                        != hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                        + ".json"
                    ):
                        receipt_drift = True
                except OSError:
                    receipt_drift = True
        passes.append((content, binding))
    latest = outcomes[max(outcomes)]
    # Review closes at the last outcome, excluding later review answers.
    index = events.index(latest) if latest in events else len(events) - 1
    selected = [*journey.planning_evidence(), *events[: index + 1]]
    review = compile_execution_contact_evidence(
        journal_bytes=passes[-1][0],
        binding=passes[-1][1],
        expected_execution_id=bound["execution_id"],
        journey_events=selected,
        journey_binding=_journey_binding(selected),
        prior_passes=passes[:-1],
    )
    if receipt_drift:
        path_reasons.append("contact_receipt_drift")
    review = review.model_copy(update={"retained_passes": tuple(retained_passes)})
    if path_reasons:
        review = review.model_copy(
            update={
                "complete": False,
                "fixture_eligible": False,
                "discrepancies": tuple(
                    dict.fromkeys((*review.discrepancies, *path_reasons))
                ),
            }
        )
    latest = {
        **latest,
        "payload": {
            **latest["payload"],
            "contact_journal": retained_passes[-1].journal_locator,
        },
    }
    return review, latest


def retain_path_accounting(journey):
    review, latest = _path_accounting(journey)
    path = Path(latest["payload"]["contact_journal"]).with_name(
        "predecessor-contact-ledger.json"
    )
    _write_ledger_binding(
        path, canonical_json(review.ledger.model_dump(mode="json")).encode()
    )
    return review


def review_path_contacts(journey):
    review, latest = _path_accounting(journey)
    journal_path = Path(latest["payload"]["contact_journal"])
    ledger_path = journal_path.with_name("predecessor-contact-ledger.json")
    materialized = None
    equal = False
    reason = None
    try:
        content = ledger_path.read_bytes()
        materialized = (
            ExecutionContactLedgerBinding(
                locator=str(ledger_path),
                byte_count=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            if content
            else None
        )
        retained = FreshFullPredecessorContactLedger.model_validate_json(
            content, strict=True
        )
        equal = retained == review.ledger
        if not equal:
            reason = "materialized_ledger_projection_drift"
    except FileNotFoundError:
        reason = "materialized_ledger_missing"
    except OSError:
        reason = "materialized_ledger_unreadable"
    except ValueError:
        reason = "materialized_ledger_malformed"
    incomplete_reasons = {
        "missing_or_drifted_binding",
        "missing_or_drifted_outcome_journal_binding",
        "prior_pass_journal_incomplete",
        "unfinished_step",
        "unfinished_logical_contact",
        "unmatched_unit_starts",
        "materialized_ledger_missing",
        "materialized_ledger_unreadable",
        "journal_missing",
        "journal_unreadable",
        "outcome_binding_absent",
        "journal_unclosed",
    }
    reasons = (*review.discrepancies, *((reason,) if reason else ()))
    drift = any(item not in incomplete_reasons for item in reasons)
    complete = review.complete and equal
    return PublicExecutionContactReview(
        status="complete" if complete else "drift" if drift else "incomplete",
        execution_journey_file=str(journey.file),
        journal_locator=str(journal_path),
        ledger_locator=str(ledger_path),
        journey_pointer_event_id=latest["event_id"],
        journal_artifact_equal=review.binding_matches,
        materialized_ledger=materialized,
        projection_equal=equal,
        review=review
        if complete
        else review.model_copy(
            update={
                "fixture_eligible": False,
                "discrepancies": reasons,
            }
        ),
    )


def review_execution_contacts(
    execution_journey_file: str,
) -> PublicExecutionContactReview:
    from bio_neb.journey_path import JourneyPath

    # Accounting interprets the retained semantic outcomes and journal bindings;
    # it never reconstructs CPU/H100 completion or grants artifact validity.
    return review_path_contacts(JourneyPath.open(Path(execution_journey_file)))


__all__ = [
    "COMPILER_VERSION",
    "EXECUTION_CONTACT_SCOPE",
    "EXECUTION_CONTACT_SCOPE_ID",
    "ExecutionContactJournal",
    "ExecutionContactJournalBinding",
    "ExecutionContactJourneyPointer",
    "ExecutionContactLedgerBinding",
    "ExecutionContactReview",
    "FreshFullPredecessorContactLedger",
    "MeasuredContactMetric",
    "PublicExecutionContactReview",
    "UnmeasurableContactMetric",
    "compile_execution_contact_evidence",
    "read_complete_execution_contacts",
    "review_execution_contacts",
]
