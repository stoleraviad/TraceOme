"""The Run's own observation inputs: its verified VCF and BAM, and its tools.

An approved grant needs the Run's Execution VCF on local disk and the observation
tools the Run named; the BAM comes local beside it for later questions such as
coverage (#363). The first grant of a Run supplies them under the Run's own
directory: the tools are probed once, and the VCF, then the BAM, each download in
ranged parts, every part retried only for a transient fault by its stage policy
(``vcf_materialization``, ``bam_materialization``) and ended by its part deadline
(#418). Each part attempt is a contact row in the Pass journal, owned by the
grant, one scope per file, with why it ended; the scope opens with the stage's
resolved part limits and counts the file's credential lookups. Later grants
reuse the retained inputs without another transfer, hash or probe.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, StrictInt, StrictStr, model_validator

from bio_neb.application.input_journey import RunInputsRegistered
from bio_neb.application.verification_policy import (
    ArtifactVerificationOverrides,
    ResolvedArtifactVerificationPolicy,
)
from bio_neb.contracts import FrozenContract
from bio_neb.failure import ResolvedStageLimits, run_failure_tuning
from bio_neb.investigation.bam_materialization import (
    BamMaterializationRefused,
    CurrentBamAcquisitionReceipt,
    MaterializeCurrentBamCommand,
    materialize_current_bam,
)
from bio_neb.investigation.download_review import (
    REVIEWED_CODES,
    InputDownloadRefused,
    InvestigationDownloadReviews,
)
from bio_neb.investigation.investigation_observation import (
    InvestigationObservationCommand,
)
from bio_neb.investigation.observation_tools import probe_observation_tools
from bio_neb.investigation.vcf_materialization import (
    MaterializeCurrentVcfCommand,
    VcfMaterializationReceipt,
    VcfMaterializationRefused,
    materialize_current_vcf,
)
from bio_neb.journey import Journey
from bio_neb.providers.nebius_object_storage import PartEndCause
from bio_neb.verified_artifacts import VerifiedObject

CONTACT_SCHEMA = "bio-neb.investigation-object-contact/4"
# Recorded before part causes: part attempts without a cause, in a scope opened
# without limits. Such rows read under that recorded contract.
RECORDED_PART_SCHEMA = "bio-neb.investigation-object-contact/3"
# Recorded before parts: one whole-object GET per attempt, started then ended.
# Such rows read under that recorded contract.
RECORDED_ATTEMPT_SCHEMA = "bio-neb.investigation-object-contact/2"
# The first live Run (2ac5fd60) recorded one row per finished attempt, with no
# start; such rows read under that recorded contract (D3 A).
RECORDED_RESULT_SCHEMA = "bio-neb.investigation-object-contact/1"
# A /4 scope's first row: its stage's resolved part limits.
LIMITS_SCHEMA = "bio-neb.investigation-part-limits/1"
# A file's credential lookups, counted in its scope before its parts.
CREDENTIAL_SCHEMA = "bio-neb.investigation-credential-contact/1"
CONTACT_SCHEMAS = frozenset(
    {
        CONTACT_SCHEMA,
        LIMITS_SCHEMA,
        RECORDED_PART_SCHEMA,
        RECORDED_ATTEMPT_SCHEMA,
        RECORDED_RESULT_SCHEMA,
        CREDENTIAL_SCHEMA,
    }
)
# One contact scope per downloaded file under its grant, under its stage.
_SCOPE_ITERATION = {"VCF": 1, "BAM": 2}
_SCOPE_STAGE = {1: "vcf_materialization", 2: "bam_materialization"}
_ROOT = "investigation"
# Where the grant's preparation keeps the Run's tool probe and its BAM's receipt.
_TOOLS = "tools"
_BAM_RECEIPT = Path("bam") / "current-bam-acquisition-receipt.json"
_Classification = Literal[
    "success", "transient", "absent", "access", "transfer", "malformed", "failed"
]


class RecordedObjectResult(FrozenContract):
    """A ``/1`` row: one finished GET attempt, as it was recorded."""

    schema_version: Literal["bio-neb.investigation-object-contact/1"]
    authority_id: StrictStr = Field(min_length=1)
    role: Literal["VCF"]
    bucket: StrictStr = Field(min_length=1)
    object_key: StrictStr = Field(min_length=1)
    attempt: StrictInt = Field(ge=1)
    classification: _Classification
    byte_count: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def bytes_on_success(self) -> "RecordedObjectResult":
        if (self.classification == "success") != (self.byte_count is not None):
            raise ValueError("investigation contact changed its transfer result")
        return self


class RecordedObjectAttempt(FrozenContract):
    """A ``/2`` row: one whole-object GET attempt's start or end, as recorded."""

    schema_version: Literal["bio-neb.investigation-object-contact/2"]
    authority_id: StrictStr = Field(min_length=1)
    role: Literal["VCF"]
    bucket: StrictStr = Field(min_length=1)
    object_key: StrictStr = Field(min_length=1)
    attempt: StrictInt = Field(ge=1)
    stage: Literal["started", "ended"]
    classification: _Classification | None = None
    byte_count: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def result_on_its_end(self) -> "RecordedObjectAttempt":
        if (self.stage == "ended") != (self.classification is not None) or (
            self.classification == "success"
        ) != (self.byte_count is not None):
            raise ValueError("an attempt's result belongs to its end")
        return self


class RecordedPartAttempt(FrozenContract):
    """A ``/3`` row: one part attempt's start or end, without its cause, as recorded."""

    schema_version: Literal["bio-neb.investigation-object-contact/3"]
    authority_id: StrictStr = Field(min_length=1)
    role: Literal["VCF", "BAM"]
    bucket: StrictStr = Field(min_length=1)
    object_key: StrictStr = Field(min_length=1)
    part: StrictInt = Field(ge=1)
    offset: StrictInt = Field(ge=0)
    part_byte_count: StrictInt = Field(gt=0)
    attempt: StrictInt = Field(ge=1)
    stage: Literal["started", "ended"]
    classification: _Classification | None = None
    byte_count: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def result_on_its_end(self) -> "RecordedPartAttempt":
        if (
            (self.stage == "ended") != (self.classification is not None)
            or (self.classification == "success") != (self.byte_count is not None)
            or self.byte_count not in (None, self.part_byte_count)
        ):
            raise ValueError("an attempt's result belongs to its end")
        return self


class InvestigationObjectContact(RecordedPartAttempt):
    """One part attempt of a grant's file download, as its Pass journal records it.

    A ``started`` row precedes the ranged GET; its ``ended`` row carries the
    result and its cause: success, the part deadline, or the failure's safe
    detail. An attempt without an end was interrupted. Attempts count per part.
    """

    schema_version: Literal["bio-neb.investigation-object-contact/4"] = CONTACT_SCHEMA
    cause: PartEndCause | None = None

    @model_validator(mode="after")
    def cause_on_its_end(self) -> "InvestigationObjectContact":
        if (
            (self.stage == "ended") != (self.cause is not None)
            or (self.cause is not None)
            and (
                (self.cause.kind == "success") != (self.classification == "success")
                or self.cause.kind == "deadline"
                and self.classification != "transient"
            )
        ):
            raise ValueError("an attempt's cause belongs to its end")
        return self


class InvestigationCredentialContact(FrozenContract):
    """One credential lookup for a grant's file download, started then ended.

    The selected context's read-only resolution (``api.nebius.cloud``) or its
    cached material; a lookup without an end was interrupted.
    """

    schema_version: Literal["bio-neb.investigation-credential-contact/1"] = (
        CREDENTIAL_SCHEMA
    )
    authority_id: StrictStr = Field(min_length=1)
    role: Literal["VCF", "BAM"]
    lookup: StrictInt = Field(ge=1)
    stage: Literal["started", "ended"]
    outcome: (
        Literal["credential_resolved", "credential_cached", "credential_unavailable"]
        | None
    ) = None

    @model_validator(mode="after")
    def outcome_on_its_end(self) -> "InvestigationCredentialContact":
        if (self.stage == "ended") != (self.outcome is not None):
            raise ValueError("a lookup's outcome belongs to its end")
        return self


class ScopeOpening(FrozenContract):
    """A ``/4`` file scope's first row: its stage's resolved part limits."""

    schema_version: Literal["bio-neb.investigation-part-limits/1"] = LIMITS_SCHEMA
    limits: ResolvedStageLimits


class GrantContactOwner(FrozenContract):
    """The grant event as a contact owner: only its exact authority identity."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    class _Authority(FrozenContract):
        model_config = ConfigDict(extra="ignore", frozen=True)
        authority_id: StrictStr

    authority: _Authority


def grant_contact_identity(event: Mapping[str, Any]) -> str:
    return GrantContactOwner.model_validate(event["payload"]).authority.authority_id


def validate_investigation_contacts(rows: Sequence) -> None:
    """A grant's download rows, each scope under the contract its schema names.

    One scope per file under its grant. ``/4``: the scope's first row is its
    stage's resolved part limits; each part's attempts start, then end with a cause, in
    order; a success ends that part and the next part starts at its end; the
    file's credential lookups each start then end, outside any part attempt.
    The scope closes complete only after every started attempt and lookup has
    ended, so an interrupted one leaves it open and its Pass journal reads
    incomplete. ``/3``: the same parts without causes, limits or lookups. ``/2``:
    whole-object attempts. ``/1``: one row per finished attempt, in order,
    success last. One scope's rows share one version and file.
    """
    scopes: dict[tuple[str, int], _Scope] = {}
    for row in rows:
        domain = row.domain
        schema = None if domain is None else domain.payload.get("schema_version")
        if schema not in CONTACT_SCHEMAS:
            continue
        key = (domain.operation_id, domain.iteration)
        if schema == LIMITS_SCHEMA:
            opening = ScopeOpening.model_validate(domain.payload)
            if (
                domain.phase != "record"
                or key in scopes
                or opening.limits.surface.name != _SCOPE_STAGE.get(domain.iteration)
            ):
                raise ValueError("investigation contact changed its scoped opening")
            scopes[key] = _Scope(CONTACT_SCHEMA)
            continue
        if schema == CREDENTIAL_SCHEMA:
            scope = scopes.get(key)
            if scope is None or scope.version != CONTACT_SCHEMA:
                raise ValueError("investigation contact changed its schema version")
            scope.lookup(
                domain, InvestigationCredentialContact.model_validate(domain.payload)
            )
            continue
        if schema == CONTACT_SCHEMA and key not in scopes:
            raise ValueError("investigation contact changed its scoped opening")
        scope = scopes.setdefault(key, _Scope(schema))
        if scope.version != schema:
            raise ValueError("investigation contact changed its schema version")
        seen = scope.parts
        if domain.phase == "closed":
            if (
                domain.payload != {"schema_version": schema}
                or not domain.complete
                # A /4 scope opens before any contact, so a refused lookup
                # closes it with no part.
                or (schema != CONTACT_SCHEMA and not seen)
                or (
                    schema != RECORDED_RESULT_SCHEMA
                    and seen
                    and seen[-1].stage != "ended"
                )
                or scope.open_lookup is not None
            ):
                raise ValueError("investigation contact changed its scoped closure")
            continue
        if domain.phase != "record":
            raise ValueError("investigation contact changed its grant")
        previous = seen[-1] if seen else None
        if schema == RECORDED_RESULT_SCHEMA:
            contact = RecordedObjectResult.model_validate(domain.payload)
            expected_attempt = 1 if previous is None else previous.attempt + 1
            in_order = contact.attempt == expected_attempt
            done = previous is not None and previous.classification == "success"
        elif schema == RECORDED_ATTEMPT_SCHEMA:
            contact = RecordedObjectAttempt.model_validate(domain.payload)
            in_order = (contact.stage, contact.attempt) == _next_attempt(previous)
            done = previous is not None and previous.classification == "success"
        else:
            contact = (
                InvestigationObjectContact
                if schema == CONTACT_SCHEMA
                else RecordedPartAttempt
            ).model_validate(domain.payload)
            in_order = (
                _next_part_attempt(previous, contact) and scope.open_lookup is None
            )
            done = False
        scope.same_file(domain, contact)
        if not in_order or done:
            raise ValueError("investigation contact attempts are out of order")
        seen.append(contact)


class _Scope:
    """One file scope's rows so far: its version, part rows and open lookup."""

    def __init__(self, version: str) -> None:
        self.version = version
        self.parts: list = []
        self.file: tuple[str, str] | None = None
        self.lookups = 0
        self.open_lookup: int | None = None

    def same_file(self, domain, contact) -> None:
        """One file per scope; a ``/4`` file is the one its limits and lookups name."""
        if (
            contact.authority_id != domain.operation_id
            or (
                self.file is not None
                and self.file != (contact.role, contact.object_key)
            )
            or (
                self.version == CONTACT_SCHEMA
                and _SCOPE_ITERATION[contact.role] != domain.iteration
            )
        ):
            raise ValueError("investigation contact changed its grant")
        self.file = (contact.role, contact.object_key)

    def lookup(self, domain, contact: InvestigationCredentialContact) -> None:
        """A lookup starts outside any part attempt and ends before the next."""
        if domain.phase != "record" or contact.authority_id != domain.operation_id:
            raise ValueError("investigation contact changed its grant")
        if _SCOPE_ITERATION[contact.role] != domain.iteration:
            raise ValueError("investigation contact changed its grant")
        part_open = bool(self.parts) and self.parts[-1].stage == "started"
        if contact.stage == "started":
            in_order = (
                self.open_lookup is None
                and not part_open
                and contact.lookup == self.lookups + 1
            )
            self.lookups = contact.lookup
            self.open_lookup = contact.lookup
        else:
            in_order = self.open_lookup == contact.lookup
            self.open_lookup = None
        if not in_order:
            raise ValueError("investigation contact lookups are out of order")


def _next_attempt(previous) -> tuple[str, int]:
    if previous is None:
        return ("started", 1)
    if previous.stage == "started":
        return ("ended", previous.attempt)
    return ("started", previous.attempt + 1)


def _next_part_attempt(previous, contact: InvestigationObjectContact) -> bool:
    """A part attempt follows its predecessor: its end, a retry, or the next part."""
    if previous is None:
        return (contact.stage, contact.part, contact.offset, contact.attempt) == (
            "started",
            1,
            0,
            1,
        )
    if previous.stage == "started":
        return (
            contact.stage,
            contact.part,
            contact.offset,
            contact.part_byte_count,
            contact.attempt,
        ) == (
            "ended",
            previous.part,
            previous.offset,
            previous.part_byte_count,
            previous.attempt,
        )
    if previous.classification == "success":
        return (contact.stage, contact.part, contact.offset, contact.attempt) == (
            "started",
            previous.part + 1,
            previous.offset + previous.part_byte_count,
            1,
        )
    return (
        contact.stage,
        contact.part,
        contact.offset,
        contact.part_byte_count,
        contact.attempt,
    ) == (
        "started",
        previous.part,
        previous.offset,
        previous.part_byte_count,
        previous.attempt + 1,
    )


def fold_grant_contact_owners(_events: Sequence[Mapping[str, Any]]) -> None:
    """Contact ownership needs no folded state; the owner's identity is enough."""
    return None


class _GrantGetJournal:
    """Record one file's download under its grant: limits, lookups and parts.

    It is the download's part observer and its credential resolver's contact
    observer. The file's scope opens with its stage's resolved part limits,
    before any contact; a refusal before then leaves no contact record.
    """

    def __init__(
        self,
        journey: Journey,
        grant_event_id: str,
        authority_id: str,
        role: Literal["VCF", "BAM"],
        descriptor: VerifiedObject,
    ) -> None:
        from bio_neb.recording import current_processing_pass

        interval = current_processing_pass()
        if interval is None or interval.journey is not journey:
            raise RuntimeError("a grant's GET requires its owning processing Pass")
        interval.start()
        self._interval = interval
        self._journey = journey
        self._grant_event_id = grant_event_id
        self._authority_id = authority_id
        self._role = role
        self._descriptor = descriptor
        self._scoped = False
        self._open: tuple[int, int, int, int] | None = None
        self._lookups = 0
        self._open_lookup: int | None = None

    def _record(
        self, phase: str, payload: dict[str, Any], complete: bool | None = None
    ) -> None:
        from bio_neb.recording import DomainContactRecord

        self._interval.journal.record_domain(
            DomainContactRecord(
                owner_event_id=self._grant_event_id,
                after_event_id=self._journey.head_event_id,
                operation_id=self._authority_id,
                iteration=_SCOPE_ITERATION[self._role],
                phase=phase,
                complete=complete,
                payload=payload,
            )
        )

    def _attempt(
        self, part: int, attempt: int, offset: int, size: int, stage: str, **result
    ) -> None:
        self._record(
            "record",
            InvestigationObjectContact(
                authority_id=self._authority_id,
                role=self._role,
                bucket=self._descriptor.bucket,
                object_key=self._descriptor.object_key,
                part=part,
                offset=offset,
                part_byte_count=size,
                attempt=attempt,
                stage=stage,
                **result,
            ).model_dump(mode="json"),
        )

    def opened(self, limits: ResolvedStageLimits) -> None:
        self._record("started", {})
        self._record("record", ScopeOpening(limits=limits).model_dump(mode="json"))
        self._scoped = True

    def started(self, part: int, attempt: int, offset: int, byte_count: int) -> None:
        self._attempt(part, attempt, offset, byte_count, "started")
        self._open = (part, attempt, offset, byte_count)

    def ended(
        self,
        part: int,
        attempt: int,
        classification: str,
        byte_count: int | None,
        cause: PartEndCause,
    ) -> None:
        assert self._open is not None and self._open[:2] == (part, attempt)
        _, _, offset, size = self._open
        self._attempt(
            part,
            attempt,
            offset,
            size,
            "ended",
            classification=classification,
            byte_count=byte_count,
            cause=cause,
        )
        self._open = None

    def _lookup(self, lookup: int, stage: str, outcome: str | None = None) -> None:
        self._record(
            "record",
            InvestigationCredentialContact(
                authority_id=self._authority_id,
                role=self._role,
                lookup=lookup,
                stage=stage,
                outcome=outcome,
            ).model_dump(mode="json"),
        )

    def start(
        self,
        *,
        category: str,
        operation: str,
        executable_identity: str,
        credential_names: tuple[str, ...] = (),
        selected_identity: str | None = None,
    ) -> str:
        """A credential lookup starts (``ExecutionContactObserver``)."""
        if category != "credential" or not self._scoped:
            raise ValueError("a grant's journal counts only its file's lookups")
        self._lookups += 1
        self._lookup(self._lookups, "started")
        self._open_lookup = self._lookups
        return str(self._lookups)

    def finish(
        self,
        unit_id: str | None,
        *,
        disposition: str,
        exit_status: int | None = None,
        parse_classification: str | None = None,
        registry_response: object | None = None,
    ) -> None:
        """A credential lookup ends with its outcome."""
        if unit_id is None or unit_id != str(self._open_lookup):
            raise ValueError("a lookup ends only after its start")
        self._lookup(self._open_lookup, "ended", parse_classification)
        self._open_lookup = None

    def close(self) -> None:
        """Close the scope complete; an interrupted attempt or lookup leaves it open."""
        if self._scoped and self._open is None and self._open_lookup is None:
            self._record("closed", {"schema_version": CONTACT_SCHEMA}, complete=True)


@dataclass(frozen=True)
class SuppliedObservationInputs:
    """The Run-root command, and how to download its files when preparation needs them."""

    command: InvestigationObservationCommand
    verification_policy: ResolvedArtifactVerificationPolicy
    acquire_inputs: Callable[[], object]


def supply_observation_inputs(
    journey: Journey,
    *,
    authority_id: str,
    grant_event_id: str,
    descriptor: VerifiedObject,
    registration: RunInputsRegistered,
    downloads: InvestigationDownloadReviews,
    object_read_factory: Callable[[Any], Any] | None = None,
) -> SuppliedObservationInputs:
    """Probe the Run's tools and lay out its first observation inputs.

    A tool the Run's section does not provide raises ``RuntimeConfigError`` before
    any contact; the registered selections are the only configuration read.
    The VCF and BAM are downloaded only when the observation's preparation needs
    them, as the grant's answered ``downloads`` Reviews still require; the BAM is
    named by the Execution result's own ``aligned_bam``.
    """
    root = journey.run_directory / _ROOT
    selections = registration.runtime_selections
    probe = probe_observation_tools(selections.investigation_tools, root / _TOOLS)
    (root / "derivation").mkdir(mode=0o700, parents=True, exist_ok=True)
    vcf_directory = root / "vcf"
    command = MaterializeCurrentVcfCommand(
        execution_journey_file=str(journey.path),
        output_directory=str(vcf_directory),
        verification_defaults=selections.verification_defaults,
    )

    bam_command = MaterializeCurrentBamCommand(
        execution_journey_file=str(journey.path),
        output_directory=str(root / "bam"),
    )
    vcf_receipt = vcf_directory / "vcf-materialization-receipt.json"
    bam_receipt = root / _BAM_RECEIPT

    def download(role, item, materialize, selected) -> None:
        observe = _GrantGetJournal(journey, grant_event_id, authority_id, role, item)
        try:
            materialize(
                selected,
                object_read_factory=object_read_factory,
                observe_part=observe,
                observe_credentials=observe,
            )
        except (VcfMaterializationRefused, BamMaterializationRefused) as error:
            if error.code in REVIEWED_CODES:
                raise InputDownloadRefused(role, item, error) from error
            raise
        finally:
            observe.close()

    def acquire_inputs() -> None:
        """Download each file this preparation has not yet published.

        A file whose receipt is published is reused (an interrupted or reviewed
        preparation resumes after it); a refused download left nothing. An
        answered ``abort`` skips the BAM or refuses the VCF with its recorded
        refusal (owner B, #363).
        """
        if downloads.latest_action(authority_id, "VCF") == "abort":
            failure = downloads.failure(authority_id, "VCF")
            raise VcfMaterializationRefused(failure.code, *failure.details)
        bam = _run_bam(journey)
        # Each part's stage policy is the Run's recorded one.
        with run_failure_tuning(registration.runtime_selections.failure):
            if not _published(vcf_receipt, VcfMaterializationReceipt, descriptor):
                download("VCF", descriptor, materialize_current_vcf, command)
            if downloads.latest_action(
                authority_id, "BAM"
            ) != "abort" and not _published(
                bam_receipt, CurrentBamAcquisitionReceipt, bam
            ):
                download("BAM", bam, materialize_current_bam, bam_command)

    return SuppliedObservationInputs(
        command=InvestigationObservationCommand(
            journey_file=str(journey.path),
            authority_id=authority_id,
            vcf_file=str(vcf_directory / "variants.vcf"),
            materialization_receipt_file=str(vcf_receipt),
            bcftools_executable=probe.receipt.tool("bcftools").executable,
            bcftools_probe_receipt_file=probe.receipt_path,
            derivation_root=str(root / "derivation"),
        ),
        verification_policy=selections.artifact_verification_policy(
            ArtifactVerificationOverrides()
        ),
        acquire_inputs=acquire_inputs,
    )


@dataclass(frozen=True)
class PublishedObservationInputs:
    """The local BAM the grant's preparation published and the Run's probed
    samtools, under the Investigation's own root."""

    root: Path
    bam: Path
    samtools: str


def published_observation_inputs(journey: Journey) -> PublishedObservationInputs | None:
    """What ``supply_observation_inputs`` published for the Run's measurement
    (#414); None when no download published the BAM (it was aborted, #363).

    The receipt is held to the Execution's ``aligned_bam``; the copy it names is
    trusted, never reread (ADR 0029). The probe is the one the preparation kept.
    """
    root = journey.run_directory / _ROOT
    receipt_path = root / _BAM_RECEIPT
    if not _published(receipt_path, CurrentBamAcquisitionReceipt, _run_bam(journey)):
        return None
    receipt = CurrentBamAcquisitionReceipt.model_validate_json(
        receipt_path.read_bytes()
    )
    probe = probe_observation_tools(None, root / _TOOLS)
    return PublishedObservationInputs(
        root=root,
        bam=Path(receipt.materialized_bam.path),
        samtools=probe.receipt.tool("samtools").executable,
    )


def _published(path: Path, receipt_type, descriptor: VerifiedObject) -> bool:
    """Whether this file's download already published its receipt here.

    The receipt is a small document read for its meaning; the replica it names is
    trusted, never reread (ADR 0029). A receipt for another object is refused.
    """
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return False
    if receipt_type.model_validate_json(payload).descriptor != descriptor:
        raise ValueError("a published receipt names another object")
    return True


def _run_bam(journey: Journey) -> VerifiedObject:
    """The Execution result's one ``aligned_bam``, which names the Run's BAM."""
    from bio_neb.parabricks.result import read_completed_parabricks_result

    (bam,) = (
        item
        for item in read_completed_parabricks_result(str(journey.path)).verified_objects
        if item.role == "aligned_bam"
    )
    return bam


__all__ = [
    "CONTACT_SCHEMA",
    "GrantContactOwner",
    "InvestigationObjectContact",
    "PublishedObservationInputs",
    "SuppliedObservationInputs",
    "fold_grant_contact_owners",
    "grant_contact_identity",
    "published_observation_inputs",
    "supply_observation_inputs",
    "validate_investigation_contacts",
]
