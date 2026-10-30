"""Contact-free export and review of the active Parabricks fixture bundle.

The bundle is evidence, never executable authority. Export preserves the rigorously
reviewed Ticket-94 detail packet and combines it atomically with one byte-bound,
strictly validated Ticket-93 cache-miss projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

from pydantic import (
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from bio_neb.content import ContentId
from bio_neb.contracts import StrictFrozenContract
from bio_neb.journey import (
    Journey,
    JourneyValidationError,
    canonical_json,
    canonical_json_digest,
)
from bio_neb.journey_path import ArtifactPointer, JourneyPath
from bio_neb.parabricks.execution_contact_accounting import (
    FreshFullPredecessorContactLedger,
    review_execution_contacts,
)
from bio_neb.parabricks.result import parabricks_result
from bio_neb.verified_artifacts import VerifiedObject

EXPORTER_VERSION = "bio-neb.parabricks-fixture-exporter/2"
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_FIXTURE_BYTES = 256 * 1024
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_TEXT = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=512,
        pattern=r"^(?:\S|\S.*\S)$",
    ),
]


class FixtureExportRefused(RuntimeError):
    """Fail-closed fixture export or review refusal."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class EvidenceFileIdentity(StrictFrozenContract):
    path: _TEXT
    byte_count: int = Field(ge=1, le=_MAX_EVIDENCE_BYTES)
    sha256: _SHA256

    @field_validator("path")
    @classmethod
    def absolute_canonical_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or os.path.normpath(value) != value:
            raise ValueError("evidence path must be absolute and canonical")
        return value


class FixtureObject(StrictFrozenContract):
    role: _TEXT
    object_key: _TEXT
    byte_count: int = Field(ge=1)
    sha256: _SHA256
    metadata_sha256: _SHA256

    @field_validator("object_key")
    @classmethod
    def relative_object_key(cls, value: str) -> str:
        if value.startswith(("/", "\\")) or "\\" in value or ".." in value.split("/"):
            raise ValueError("fixture object key must be relative and canonical")
        return value


class CreateCounts(StrictFrozenContract):
    claimed: int = Field(ge=0, le=1)
    contacted: int = Field(ge=0, le=1)
    created: int = Field(ge=0, le=1)


class MutationCounts(StrictFrozenContract):
    bucket: CreateCounts
    cpu: CreateCounts
    h100: CreateCounts


class ProviderFacts(StrictFrozenContract):
    bucket_create_disposition: Literal["created"]
    cpu_create_disposition: Literal["not_claimed"]
    h100_create_disposition: Literal["created"]
    h100_job_id: _TEXT
    requested_envelope_sha256: _SHA256
    fetched_envelope_sha256: _SHA256
    envelopes_match: Literal[True]
    provider_states: tuple[_TEXT, ...]
    timestamp_facts_sha256: _SHA256
    detail_facts_sha256: _SHA256
    bounded_log_sha256: _SHA256

    @field_validator("provider_states")
    @classmethod
    def exact_terminal_states(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value[-1] != "COMPLETED" or "RUNNING" not in value:
            raise ValueError("provider states do not reach RUNNING then COMPLETED")
        return value


class RuntimeIdentity(StrictFrozenContract):
    name: Literal["python", "pbrun", "bwa", "samtools"]
    path: _TEXT
    version: _TEXT

    @field_validator("path")
    @classmethod
    def exact_tool_path(cls, value: str) -> str:
        if not value.startswith(
            ("/opt/bio-neb/", "/usr/bin/", "/usr/local/parabricks/")
        ):
            raise ValueError("runtime identity path is outside the admitted worker")
        return value


class WorkerFacts(StrictFrozenContract):
    request_schema: _TEXT
    receipt_schema: _TEXT
    request_sha256: _SHA256
    worker_sha256: _SHA256
    image_child_digest: _TEXT
    runtime_identities: tuple[RuntimeIdentity, ...]
    germline_argv_sha256: _SHA256
    tool_started_count: Literal[1]
    tool_exited_count: Literal[1]
    germline_exit_code: Literal[0]
    worker_input_verified_count: Literal[10]

    @model_validator(mode="after")
    def exact_runtime_roles(self) -> "WorkerFacts":
        if tuple(item.name for item in self.runtime_identities) != (
            "python",
            "pbrun",
            "bwa",
            "samtools",
        ):
            raise ValueError("worker runtime identities are incomplete or unordered")
        return self


class InventoryFacts(StrictFrozenContract):
    pre_manifest_keys: tuple[_TEXT, ...]
    manifest_key: _TEXT
    final_keys: tuple[_TEXT, ...]
    manifest_published_last: Literal[True]
    multipart_upload_count: Literal[0]

    @model_validator(mode="after")
    def exact_inventory(self) -> "InventoryFacts":
        if (
            len(set(self.pre_manifest_keys)) != len(self.pre_manifest_keys)
            or tuple(sorted(self.pre_manifest_keys)) != self.pre_manifest_keys
            or self.manifest_key in self.pre_manifest_keys
            or self.final_keys
            != tuple(sorted((*self.pre_manifest_keys, self.manifest_key)))
        ):
            raise ValueError("fixture inventory is not exact and manifest-last")
        return self


class InputInventoryFacts(StrictFrozenContract):
    exact_keys: tuple[_TEXT, ...]
    multipart_upload_count: Literal[0]

    @field_validator("exact_keys")
    @classmethod
    def exact_inventory(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 3 or len(set(value)) != 3 or tuple(sorted(value)) != value:
            raise ValueError("fixture input inventory is not exact")
        return value


class StableFixtureFacts(StrictFrozenContract):
    request_id: _TEXT
    proposal_id: _TEXT
    authority_id: _TEXT
    packet_binding_id: _TEXT
    reference_asset_id: _TEXT
    assembly_accession: _TEXT
    source_derivation_rule: _TEXT
    input_objects: tuple[FixtureObject, ...]
    prepared_objects: tuple[FixtureObject, ...]
    prepared_worker_receipt: FixtureObject
    prepared_manifest: FixtureObject
    output_objects: tuple[FixtureObject, ...]
    mutation_counts: MutationCounts
    provider: ProviderFacts
    worker: WorkerFacts
    input_inventory: InputInventoryFacts
    prepared_inventory: InventoryFacts
    output_inventory: InventoryFacts
    prepared_reference_receipt_id: _TEXT
    h100_receipt_id: _TEXT
    status: Literal["execution_artifacts_verified"]
    frontier_state: Literal["artifacts_verified"]
    cache_outcome: Literal["hit"]
    prepared_reference_source: Literal["cache_hit"]

    @model_validator(mode="after")
    def exact_roles_and_inventories(self) -> "StableFixtureFacts":
        if (
            (
                self.mutation_counts.bucket.claimed,
                self.mutation_counts.bucket.contacted,
                self.mutation_counts.bucket.created,
            )
            != (1, 1, 1)
            or (
                self.mutation_counts.cpu.claimed,
                self.mutation_counts.cpu.contacted,
                self.mutation_counts.cpu.created,
            )
            != (0, 0, 0)
            or (
                self.mutation_counts.h100.claimed,
                self.mutation_counts.h100.contacted,
                self.mutation_counts.h100.created,
            )
            != (1, 1, 1)
        ):
            raise ValueError("Ticket-94 facts lost the accepted cache-hit topology")
        if tuple(item.role for item in self.input_objects) != (
            "reference_source",
            "read_1",
            "read_2",
        ):
            raise ValueError("fixture input roles are not exact")
        if tuple(item.role for item in self.prepared_objects) != (
            "genomic_fasta",
            "fasta_fai",
            "reference_dict",
            "bwa_amb",
            "bwa_ann",
            "bwa_bwt",
            "bwa_pac",
            "bwa_sa",
        ):
            raise ValueError("fixture prepared-reference roles are not exact")
        if self.prepared_manifest.role != "prepared_reference_manifest":
            raise ValueError("fixture lacks the admission manifest")
        if self.prepared_worker_receipt.role != "worker_receipt":
            raise ValueError("fixture lacks the copied CPU worker receipt")
        if tuple(item.role for item in self.output_objects) != (
            "aligned_bam",
            "VCF",
            "worker_receipt",
            "execution_manifest",
        ):
            raise ValueError("fixture output roles are not exact")
        input_keys = tuple(item.object_key for item in self.input_objects)
        if tuple(sorted(input_keys)) != self.input_inventory.exact_keys:
            raise ValueError("fixture inputs disagree with their exact inventory")
        checks = (
            (
                (
                    *self.prepared_objects,
                    self.prepared_worker_receipt,
                    self.prepared_manifest,
                ),
                self.prepared_inventory,
            ),
            (self.output_objects, self.output_inventory),
        )
        for objects, inventory in checks:
            keys = tuple(item.object_key for item in objects)
            if (
                tuple(sorted(keys[:-1])) != inventory.pre_manifest_keys
                or keys[-1] != inventory.manifest_key
            ):
                raise ValueError("fixture objects disagree with their exact inventory")
        return self


class FixtureEvidenceDocument(StrictFrozenContract):
    # Stored schema ids keep their original names (only code identifiers were
    # renamed); rename them at a clean fixture refresh.
    schema_version: Literal[
        "bio-neb.ticket94-live-receipt/1",
        "bio-neb.ticket94-independent-verification/1",
    ]
    journey_id: _TEXT
    journey_sha256: _SHA256
    facts: StableFixtureFacts


class HostileReopenFacts(StrictFrozenContract):
    schema_version: Literal["bio-neb.ticket94-hostile-reopen/1"]
    journey_id: _TEXT
    journey_byte_count: int = Field(ge=1)
    journey_sha256: _SHA256
    returncodes: tuple[Literal[0], Literal[0]]
    stdout_byte_count: int = Field(ge=1)
    stdout_sha256: _SHA256
    stdout_byte_stable: Literal[True]
    journey_byte_stable: Literal[True]
    model_tool_provider_storage_process_contact_count: Literal[0]
    filesystem_write_count: Literal[0]
    fixture_contact_count: Literal[0]


class FixtureEvidenceIndex(StrictFrozenContract):
    schema_version: Literal["bio-neb.ticket94-evidence-index/1"]
    journey: EvidenceFileIdentity
    live_receipt: EvidenceFileIdentity
    independent_verification: EvidenceFileIdentity
    hostile_reopen: EvidenceFileIdentity
    facts_sha256: _SHA256


class CacheHitEvidenceInput(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-cache-hit-evidence-input/1"] = (
        "bio-neb.parabricks-cache-hit-evidence-input/1"
    )
    journey: EvidenceFileIdentity
    live_receipt: EvidenceFileIdentity
    independent_verification: EvidenceFileIdentity
    evidence_index: EvidenceFileIdentity
    hostile_reopen: EvidenceFileIdentity


class FixtureSourceEvidence(StrictFrozenContract):
    ticket: Literal[94] = 94
    journey_id: _TEXT
    journey_byte_count: int = Field(ge=1)
    journey_sha256: _SHA256
    live_receipt_sha256: _SHA256
    independent_verification_sha256: _SHA256
    evidence_index_sha256: _SHA256
    hostile_reopen_sha256: _SHA256


class FixtureManifestIdentity(StrictFrozenContract):
    object_key: _TEXT
    byte_count: int = Field(ge=1)
    sha256: _SHA256

    @field_validator("object_key")
    @classmethod
    def relative_object_key(cls, value: str) -> str:
        if value.startswith(("/", "\\")) or "\\" in value or ".." in value.split("/"):
            raise ValueError("fixture manifest key must be relative and canonical")
        return value


class FixtureHostileProjection(StrictFrozenContract):
    returncodes: tuple[Literal[0], Literal[0]]
    stdout_byte_count: int = Field(ge=1)
    stdout_sha256: _SHA256
    stdout_byte_stable: Literal[True]
    journey_byte_stable: Literal[True]
    external_contact_count: Literal[0]
    filesystem_mutation_count: Literal[0]


class _FixtureScenario(StrictFrozenContract):
    fixture_grants_live_authority: Literal[False] = False
    source_ticket: Literal[93, 94]
    run_label: _TEXT
    journey_id: _TEXT
    journey_byte_count: int = Field(ge=1)
    journey_sha256: _SHA256
    live_receipt_sha256: _SHA256
    independent_verification_sha256: _SHA256
    evidence_index_sha256: _SHA256
    hostile_reopen_sha256: _SHA256
    proposal_id: _TEXT
    authority_id: _TEXT
    packet_binding_id: _TEXT
    mutation_counts: MutationCounts
    prepared_reference_receipt_id: _TEXT
    prepared_manifest: FixtureManifestIdentity
    h100_receipt_id: _TEXT
    execution_manifest: FixtureManifestIdentity
    status: Literal["execution_artifacts_verified"]
    frontier_state: Literal["artifacts_verified"]
    hostile_reopen_projection: FixtureHostileProjection


class CacheMissFixtureScenario(_FixtureScenario):
    cache_outcome: Literal["miss"] = "miss"
    prepared_reference_source: Literal["cpu_built"] = "cpu_built"
    source_ticket: Literal[93] = 93

    @model_validator(mode="after")
    def exact_cache_miss_topology(self) -> "CacheMissFixtureScenario":
        for counts in (
            self.mutation_counts.bucket,
            self.mutation_counts.cpu,
            self.mutation_counts.h100,
        ):
            if (counts.claimed, counts.contacted, counts.created) != (1, 1, 1):
                raise ValueError("cache-miss scenario lost an exact 1/1/1 triplet")
        return self


class CacheHitFixtureScenario(_FixtureScenario):
    cache_outcome: Literal["hit"] = "hit"
    prepared_reference_source: Literal["cache_hit"] = "cache_hit"
    source_ticket: Literal[94] = 94
    source_evidence: FixtureSourceEvidence
    facts: StableFixtureFacts
    hostile_reopen: HostileReopenFacts

    @model_validator(mode="after")
    def exact_cache_hit_projection(self) -> "CacheHitFixtureScenario":
        expected = ((1, 1, 1), (0, 0, 0), (1, 1, 1))
        actual = tuple(
            (counts.claimed, counts.contacted, counts.created)
            for counts in (
                self.mutation_counts.bucket,
                self.mutation_counts.cpu,
                self.mutation_counts.h100,
            )
        )
        if actual != expected or self.mutation_counts != self.facts.mutation_counts:
            raise ValueError("cache-hit projection lost the exact mutation topology")
        execution_manifest = next(
            (
                item
                for item in self.facts.output_objects
                if item.role == "execution_manifest"
            ),
            None,
        )
        if (
            self.source_evidence.ticket != 94
            or self.journey_id != self.source_evidence.journey_id
            or self.journey_byte_count != self.source_evidence.journey_byte_count
            or self.journey_sha256 != self.source_evidence.journey_sha256
            or self.live_receipt_sha256 != self.source_evidence.live_receipt_sha256
            or self.independent_verification_sha256
            != self.source_evidence.independent_verification_sha256
            or self.evidence_index_sha256 != self.source_evidence.evidence_index_sha256
            or self.hostile_reopen_sha256 != self.source_evidence.hostile_reopen_sha256
            or self.proposal_id != self.facts.proposal_id
            or self.authority_id != self.facts.authority_id
            or self.packet_binding_id != self.facts.packet_binding_id
            or self.prepared_reference_receipt_id
            != self.facts.prepared_reference_receipt_id
            or self.h100_receipt_id != self.facts.h100_receipt_id
            or self.cache_outcome != self.facts.cache_outcome
            or self.prepared_reference_source != self.facts.prepared_reference_source
            or self.prepared_manifest.object_key
            != self.facts.prepared_manifest.object_key
            or self.prepared_manifest.byte_count
            != self.facts.prepared_manifest.byte_count
            or self.prepared_manifest.sha256 != self.facts.prepared_manifest.sha256
            or execution_manifest is None
            or self.execution_manifest.object_key != execution_manifest.object_key
            or self.execution_manifest.byte_count != execution_manifest.byte_count
            or self.execution_manifest.sha256 != execution_manifest.sha256
            or self.status != self.facts.status
            or self.frontier_state != self.facts.frontier_state
            or self.journey_id != self.hostile_reopen.journey_id
            or self.journey_byte_count != self.hostile_reopen.journey_byte_count
            or self.journey_sha256 != self.hostile_reopen.journey_sha256
            or self.hostile_reopen_projection.returncodes
            != self.hostile_reopen.returncodes
            or self.hostile_reopen_projection.stdout_byte_count
            != self.hostile_reopen.stdout_byte_count
            or self.hostile_reopen_projection.stdout_sha256
            != self.hostile_reopen.stdout_sha256
            or self.hostile_reopen_projection.stdout_byte_stable
            != self.hostile_reopen.stdout_byte_stable
            or self.hostile_reopen_projection.journey_byte_stable
            != self.hostile_reopen.journey_byte_stable
            or self.hostile_reopen_projection.external_contact_count
            != self.hostile_reopen.model_tool_provider_storage_process_contact_count
            or self.hostile_reopen_projection.filesystem_mutation_count
            != self.hostile_reopen.filesystem_write_count
        ):
            raise ValueError(
                "cache-hit common projection conflicts with retained detail"
            )
        return self


FixtureScenario = Annotated[
    CacheMissFixtureScenario | CacheHitFixtureScenario,
    Field(discriminator="cache_outcome"),
]


class ParabricksRunFixtureBundle(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-run-fixture-bundle/1"] = (
        "bio-neb.parabricks-run-fixture-bundle/1"
    )
    exporter_version: Literal[EXPORTER_VERSION] = EXPORTER_VERSION
    fixture_grants_live_authority: Literal[False] = False
    scenarios: tuple[FixtureScenario, FixtureScenario]

    @model_validator(mode="after")
    def exact_active_scenarios(self) -> "ParabricksRunFixtureBundle":
        if tuple(item.cache_outcome for item in self.scenarios) != ("miss", "hit"):
            raise ValueError(
                "fixture bundle must contain ordered miss then hit scenarios"
            )
        return self

    def scenario(self, cache_outcome: Literal["miss", "hit"]) -> FixtureScenario:
        """Return one inert scenario explicitly by cache outcome for tests only."""

        return next(
            item for item in self.scenarios if item.cache_outcome == cache_outcome
        )


class ParabricksFixtureExportInput(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-fixture-export-input/2"] = (
        "bio-neb.parabricks-fixture-export-input/2"
    )
    cache_miss_scenario: EvidenceFileIdentity
    cache_hit_run_label: _TEXT
    cache_hit_evidence: CacheHitEvidenceInput


class _CacheHitFixtureDetail(StrictFrozenContract):
    source_evidence: FixtureSourceEvidence
    facts: StableFixtureFacts
    hostile_reopen: HostileReopenFacts


class ExportedFixtureReceipt(StrictFrozenContract):
    schema_version: Literal["bio-neb.parabricks-fixture-export-receipt/1"] = (
        "bio-neb.parabricks-fixture-export-receipt/1"
    )
    byte_count: int = Field(ge=1, le=_MAX_FIXTURE_BYTES)
    sha256: _SHA256


def _read_verified(identity: EvidenceFileIdentity) -> bytes:
    path = Path(identity.path)
    try:
        if path.resolve(strict=True) != path:
            raise FixtureExportRefused("evidence_path_not_canonical")
        stat = path.stat()
        if not path.is_file() or stat.st_size != identity.byte_count:
            raise FixtureExportRefused("evidence_size_mismatch")
        content = path.read_bytes()
    except (FileNotFoundError, OSError) as error:
        raise FixtureExportRefused("evidence_unavailable") from error
    if len(content) != identity.byte_count:
        raise FixtureExportRefused("evidence_size_mismatch")
    if hashlib.sha256(content).hexdigest() != identity.sha256:
        raise FixtureExportRefused("evidence_hash_mismatch")
    return content


def _load_json(content: bytes, model: type[StrictFrozenContract], reason: str) -> Any:
    try:
        text = content.decode("utf-8")
        value = json.loads(text)
        if canonical_json(value).encode("utf-8") != content:
            raise ValueError("not canonical")
        return model.model_validate_json(content)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FixtureExportRefused(reason) from error


def _facts_sha256(facts: StableFixtureFacts) -> str:
    return canonical_json_digest(facts.model_dump(mode="json"))


def _contains(value: Any, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains(item, expected) for item in value)
    return False


def _validate_journey(identity: EvidenceFileIdentity, facts: StableFixtureFacts) -> str:
    try:
        journey = Journey.open(Path(identity.path))
        events = journey.read()
    except (JourneyValidationError, OSError) as error:
        raise FixtureExportRefused("journey_invalid") from error
    required_values = (
        facts.proposal_id,
        facts.authority_id,
        facts.prepared_reference_receipt_id,
        facts.h100_receipt_id,
        facts.provider.h100_job_id,
    )
    if any(not _contains(events, value) for value in required_values):
        raise FixtureExportRefused("journey_fixture_identity_mismatch")
    counts = {}
    for event in events:
        event_type = event["event_type"]
        counts[event_type] = counts.get(event_type, 0) + 1
    exact_counts = {
        "RunInputStorageBucketCreateClaimed": 1,
        "RunInputStorageBucketCreateInvoked": 1,
        "ReferencePreparationCreateClaimed": 0,
        "ReferencePreparationCreateInvoked": 0,
        "H100ExecutionCreateClaimed": 1,
        "H100ExecutionCreateInvoked": 1,
        "execution_inputs_verified": 1,
        "prepared_reference_cache_hit": 1,
        "H100ExecutionArtifactsVerified": 1,
    }
    if any(counts.get(name, 0) != expected for name, expected in exact_counts.items()):
        raise FixtureExportRefused("journey_cache_hit_topology_mismatch")
    return journey.journey_id


def _forbidden_output(value: Any, *, key: str = "") -> bool:
    forbidden_keys = {
        "credential",
        "credentials",
        "selector",
        "authorization",
        "provider_body",
        "raw_log",
        "url",
        "query",
        "path_locator",
    }
    if key.lower() in forbidden_keys:
        return True
    if isinstance(value, dict):
        return any(
            _forbidden_output(item, key=str(name)) for name, item in value.items()
        )
    if isinstance(value, list):
        return any(_forbidden_output(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        if any(
            marker in lowered
            for marker in ("http://", "https://", "s3://", "file://", "?")
        ):
            return True
        if lowered.startswith(("/home/", "/tmp/", "/mnt/", "\\\\")):
            return True
        if len(value) >= 3 and value[1:3] in {":\\", ":/"}:
            return True
        if any(
            term in lowered
            for term in ("access_key", "secret_key", "bearer ", "token=")
        ):
            return True
    return False


def _compile_cache_hit_detail(
    input_manifest: CacheHitEvidenceInput,
) -> _CacheHitFixtureDetail:
    evidence = {
        "journey": _read_verified(input_manifest.journey),
        "live_receipt": _read_verified(input_manifest.live_receipt),
        "independent_verification": _read_verified(
            input_manifest.independent_verification
        ),
        "evidence_index": _read_verified(input_manifest.evidence_index),
        "hostile_reopen": _read_verified(input_manifest.hostile_reopen),
    }
    live = _load_json(
        evidence["live_receipt"], FixtureEvidenceDocument, "live_receipt_invalid"
    )
    independent = _load_json(
        evidence["independent_verification"],
        FixtureEvidenceDocument,
        "independent_verification_invalid",
    )
    if live.schema_version != "bio-neb.ticket94-live-receipt/1":
        raise FixtureExportRefused("live_receipt_schema_mismatch")
    if independent.schema_version != "bio-neb.ticket94-independent-verification/1":
        raise FixtureExportRefused("independent_verification_schema_mismatch")
    index = _load_json(
        evidence["evidence_index"], FixtureEvidenceIndex, "evidence_index_invalid"
    )
    hostile = _load_json(
        evidence["hostile_reopen"], HostileReopenFacts, "hostile_reopen_invalid"
    )
    if (
        index.journey != input_manifest.journey
        or index.live_receipt != input_manifest.live_receipt
        or index.independent_verification != input_manifest.independent_verification
        or index.hostile_reopen != input_manifest.hostile_reopen
    ):
        raise FixtureExportRefused("evidence_index_conflict")
    if live.facts != independent.facts:
        raise FixtureExportRefused("independent_facts_conflict")
    facts_digest = _facts_sha256(live.facts)
    if index.facts_sha256 != facts_digest:
        raise FixtureExportRefused("evidence_index_facts_conflict")
    journey_id = _validate_journey(input_manifest.journey, live.facts)
    if (
        live.journey_id != journey_id
        or independent.journey_id != journey_id
        or hostile.journey_id != journey_id
        or live.journey_sha256 != input_manifest.journey.sha256
        or independent.journey_sha256 != input_manifest.journey.sha256
        or hostile.journey_sha256 != input_manifest.journey.sha256
        or hostile.journey_byte_count != input_manifest.journey.byte_count
    ):
        raise FixtureExportRefused("journey_evidence_conflict")
    return _CacheHitFixtureDetail(
        source_evidence=FixtureSourceEvidence(
            journey_id=journey_id,
            journey_byte_count=input_manifest.journey.byte_count,
            journey_sha256=input_manifest.journey.sha256,
            live_receipt_sha256=input_manifest.live_receipt.sha256,
            independent_verification_sha256=(
                input_manifest.independent_verification.sha256
            ),
            evidence_index_sha256=input_manifest.evidence_index.sha256,
            hostile_reopen_sha256=input_manifest.hostile_reopen.sha256,
        ),
        facts=live.facts,
        hostile_reopen=hostile,
    )


def compile_fixture(input_manifest: ParabricksFixtureExportInput) -> bytes:
    miss = _load_json(
        _read_verified(input_manifest.cache_miss_scenario),
        CacheMissFixtureScenario,
        "cache_miss_scenario_invalid",
    )
    hit_detail = _compile_cache_hit_detail(input_manifest.cache_hit_evidence)
    facts = hit_detail.facts
    source = hit_detail.source_evidence
    hostile = hit_detail.hostile_reopen
    execution_manifest = next(
        item for item in facts.output_objects if item.role == "execution_manifest"
    )
    hit = CacheHitFixtureScenario(
        source_ticket=94,
        run_label=input_manifest.cache_hit_run_label,
        journey_id=source.journey_id,
        journey_byte_count=source.journey_byte_count,
        journey_sha256=source.journey_sha256,
        live_receipt_sha256=source.live_receipt_sha256,
        independent_verification_sha256=source.independent_verification_sha256,
        evidence_index_sha256=source.evidence_index_sha256,
        hostile_reopen_sha256=source.hostile_reopen_sha256,
        proposal_id=facts.proposal_id,
        authority_id=facts.authority_id,
        packet_binding_id=facts.packet_binding_id,
        mutation_counts=facts.mutation_counts,
        prepared_reference_receipt_id=facts.prepared_reference_receipt_id,
        prepared_manifest=FixtureManifestIdentity(
            object_key=facts.prepared_manifest.object_key,
            byte_count=facts.prepared_manifest.byte_count,
            sha256=facts.prepared_manifest.sha256,
        ),
        h100_receipt_id=facts.h100_receipt_id,
        execution_manifest=FixtureManifestIdentity(
            object_key=execution_manifest.object_key,
            byte_count=execution_manifest.byte_count,
            sha256=execution_manifest.sha256,
        ),
        status=facts.status,
        frontier_state=facts.frontier_state,
        hostile_reopen_projection=FixtureHostileProjection(
            returncodes=hostile.returncodes,
            stdout_byte_count=hostile.stdout_byte_count,
            stdout_sha256=hostile.stdout_sha256,
            stdout_byte_stable=hostile.stdout_byte_stable,
            journey_byte_stable=hostile.journey_byte_stable,
            external_contact_count=(
                hostile.model_tool_provider_storage_process_contact_count
            ),
            filesystem_mutation_count=hostile.filesystem_write_count,
        ),
        source_evidence=source,
        facts=facts,
        hostile_reopen=hostile,
    )
    fixture = ParabricksRunFixtureBundle(scenarios=(miss, hit))
    value = fixture.model_dump(mode="json")
    if _forbidden_output(value):
        raise FixtureExportRefused("fixture_contains_forbidden_content")
    content = canonical_json(value).encode("utf-8")
    if len(content) > _MAX_FIXTURE_BYTES:
        raise FixtureExportRefused("fixture_too_large")
    return content


def load_export_input(path: Path) -> ParabricksFixtureExportInput:
    try:
        content = path.read_bytes()
        identity = EvidenceFileIdentity(
            path=str(path),
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    except (OSError, ValueError) as error:
        raise FixtureExportRefused("export_input_unavailable") from error
    return _load_json(
        _read_verified(identity),
        ParabricksFixtureExportInput,
        "export_input_invalid",
    )


def publish_exclusive(path: Path, content: bytes) -> None:
    if not path.is_absolute() or path.resolve() != path:
        raise FixtureExportRefused("output_path_not_canonical")
    if not path.parent.is_dir():
        raise FixtureExportRefused("output_parent_missing")
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return
        raise FixtureExportRefused("output_conflict")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise FixtureExportRefused("output_conflict")  # noqa: B904 - Preserve existing implicit exception context during contract-base migration.
        finally:
            temporary_path.unlink(missing_ok=True)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def export_fixture(
    input_manifest: ParabricksFixtureExportInput, output: Path
) -> ExportedFixtureReceipt:
    content = compile_fixture(input_manifest)
    publish_exclusive(output, content)
    return ExportedFixtureReceipt(
        byte_count=len(content), sha256=hashlib.sha256(content).hexdigest()
    )


def review_fixture(path: Path, *, expected_sha256: str) -> bytes:
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
        fixture = ParabricksRunFixtureBundle.model_validate_json(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FixtureExportRefused("fixture_invalid") from error
    if canonical_json(fixture.model_dump(mode="json")).encode("utf-8") != content:
        raise FixtureExportRefused("fixture_not_canonical")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise FixtureExportRefused("fixture_hash_mismatch")
    if _forbidden_output(value):
        raise FixtureExportRefused("fixture_contains_forbidden_content")
    return content


def compile_input_storage_fixture(journey_file: Path, *, source_commit: str) -> bytes:
    """Export the input milestone and its proofs without replaying its authority.

    The capsule supplies the source binding. Contact counts describe the journal;
    synthetic journal evidence remains synthetic evidence in offline tests.
    """
    from bio_neb.journey_path import JourneyPath, project_execution
    from bio_neb.parabricks.execution_contact_accounting import (
        read_complete_execution_contacts,
    )

    def identity(path: Path, content: bytes) -> dict:
        return EvidenceFileIdentity(
            path=str(path.resolve(strict=True)),
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        ).model_dump(mode="json")

    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise FixtureExportRefused("source_commit_invalid")
    try:
        journey = JourneyPath.open(journey_file)
        events = journey.read()
        if [event.event_type for event in events] != [
            "ExecutionBound",
            "InputsVerified",
            "ExecutionOutcome",
        ]:
            raise ValueError("input milestone path required")
        state = project_execution(events)
        if state.outcome.milestone != "input_storage":
            raise ValueError("input milestone outcome required")
        journal_file = Path(state.outcome.contact_journal)
        journal_content = journal_file.read_bytes()
        rows = read_complete_execution_contacts(
            journal_content, state.bound.execution_id
        )
        pointers = state.inputs.pointers
        if [pointer.role for pointer in pointers] != [
            "read_1",
            "read_2",
            "reference_source",
            "input_manifest",
        ]:
            raise ValueError("input roles incomplete")
        manifest = json.loads(journey.read_artifact(pointers[-1]))
        if manifest["execution_id"] != state.bound.execution_id or manifest[
            "inputs"
        ] != [pointer.model_dump(mode="json") for pointer in pointers[:-1]]:
            raise ValueError("manifest input identities differ")
        receipts = []
        for pointer in pointers:
            _, pass_number, sequence = pointer.proof_ref.split(":")
            row = rows[int(sequence) - 1]
            if (
                int(pass_number) != state.pass_
                or row.phase != "verified"
                or row.stage != "input_storage"
                or row.receipt is None
                or row.receipt.kind != "transfer_proof"
            ):
                raise ValueError("input proof row invalid")
            receipt_file = Path(row.receipt.ref)
            if receipt_file.parent != journal_file.parent / "receipts":
                raise ValueError("input receipt outside journal bundle")
            content = receipt_file.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if receipt_file != journal_file.parent / "receipts" / (digest + ".json"):
                raise ValueError("input receipt identity changed")
            receipt = json.loads(content)
            values = receipt["values"]
            provider_ids = tuple(receipt["provider_ids"])
            if (
                receipt["operation"] != "put"
                or provider_ids != row.provider_ids
                or len(provider_ids) != 1
                or (
                    pointer.role != "input_manifest"
                    and provider_ids != (pointer.location,)
                )
                or values["bytes"] != pointer.bytes
                or values["sha256"] != ContentId.parse(pointer.content_id).digest
            ):
                raise ValueError("input verification receipt differs")
            receipts.append(
                {
                    "role": pointer.role,
                    "proof_ref": pointer.proof_ref,
                    "receipt_sha256": digest,
                    "verification_mode": "reuse_creation_proof",
                    "bytes": values["bytes"],
                    "sha256": values["sha256"],
                }
            )
        journey_content = journey_file.read_bytes()
        if hashlib.sha256(journey_content).hexdigest() != journey.prefix_sha256:
            raise ValueError("Journey changed after validation")
        return canonical_json(
            {
                "schema_version": "bio-neb.input-storage-fixture/1",
                "source_commit": source_commit,
                "journey": identity(journey_file, journey_content),
                "journal": identity(journal_file, journal_content),
                "execution_id": state.bound.execution_id,
                "path_events": [event.event_type for event in events],
                "milestone": state.outcome.milestone,
                "disposition": state.disposition,
                "inputs": [pointer.model_dump(mode="json") for pointer in pointers],
                "manifest": manifest,
                "verification_receipts": receipts,
                "contact_operations": dict(
                    Counter(row.operation for row in rows if row.row_kind == "start")
                ),
            }
        ).encode("utf-8")
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        raise FixtureExportRefused("input_storage_evidence_invalid") from error


class ExecutionFixtureScenario(StrictFrozenContract):
    """Inert artifact and flow evidence; unknown metrics retain their typed meaning."""

    source_ticket: int = Field(gt=0)
    source_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    evidence_kind: Literal["synthetic", "retained_live"]
    fixture_grants_live_authority: Literal[False] = False
    journey: EvidenceFileIdentity
    journey_id: _TEXT
    path_events: tuple[str, ...]
    disposition: Literal["succeeded"]
    inputs: tuple[ArtifactPointer, ...]
    reference: tuple[ArtifactPointer, ...]
    verified_objects: tuple[VerifiedObject, ...]
    contacts: FreshFullPredecessorContactLedger

    @model_validator(mode="after")
    def completed_artifact_flow(self):
        if (
            not self.path_events
            or self.path_events[0] != "ExecutionBound"
            or self.path_events[-1] != "ExecutionOutcome"
            or "H100ArtifactsVerified" not in self.path_events
            or tuple(item.role for item in self.verified_objects)
            != ("aligned_bam", "VCF", "worker_receipt", "execution_manifest")
            # Two FASTQs and the input manifest; the source was placed by input
            # storage or fetched by the CPU job, so it appears once in either.
            or len(self.inputs)
            != 3 + any(item.role == "reference_source" for item in self.inputs)
            or [
                item.role
                for item in (*self.inputs, *self.reference)
                if item.role == "reference_source"
            ]
            != ["reference_source"]
            or not self.reference
        ):
            raise ValueError("completed execution fixture roles/path required")
        return self


class ExecutionFixture(StrictFrozenContract):
    schema_version: Literal["bio-neb.execution-fixture/1"] = (
        "bio-neb.execution-fixture/1"
    )
    fixture_grants_live_authority: Literal[False] = False
    scenarios: tuple[ExecutionFixtureScenario, ...] = Field(min_length=1)


def compile_execution_fixture(
    journey_file: Path, *, source_commit: str, source_ticket: int, evidence_kind: str
) -> bytes:
    """Export a checked completed path and accounting, without replayable approval.

    The caller supplies the source revision and origin classification. They are
    provenance declarations, not an inference of live proof from valid fake bytes.
    Physical totals can remain unknown while the artifact/flow purpose is valid.
    """
    try:
        journey_file = journey_file.resolve(strict=True)
        owner = Journey.open(journey_file)
        path = JourneyPath.from_journey(owner)
        result = parabricks_result(path)
        if result.disposition != "succeeded":
            raise ValueError("completed execution required")
        contacts = review_execution_contacts(str(journey_file))
        if contacts.status != "complete":
            raise ValueError("complete undrifted accounting required")
        # The current result reader validates the bound manifest and worker
        # documents; the contact reader owns receipt/pass bindings and metrics.
        if not owner.is_current():
            raise ValueError("Journey changed during export")
        identity = owner.identity()
        fixture = ExecutionFixture(
            scenarios=(
                ExecutionFixtureScenario(
                    source_ticket=source_ticket,
                    source_commit=source_commit,
                    evidence_kind=evidence_kind,
                    journey=EvidenceFileIdentity(
                        path=str(journey_file),
                        byte_count=identity.byte_count,
                        sha256=identity.journey_sha256,
                    ),
                    journey_id=result.journey_id,
                    path_events=tuple(event.event_type for event in path.read()),
                    disposition=result.disposition,
                    inputs=result.input_proofs,
                    reference=result.reference_proofs,
                    verified_objects=result.verified_objects,
                    contacts=contacts.review.ledger,
                ),
            )
        )
        return canonical_json(fixture.model_dump(mode="json")).encode()
    except (OSError, ValueError, KeyError, AttributeError) as error:
        raise FixtureExportRefused("execution_evidence_invalid") from error


def load_execution_fixture(content: bytes) -> ExecutionFixture:
    """Strictly load inert exported evidence; never consult its original Run."""
    return ExecutionFixture.model_validate_json(content)


_PRE_EXECUTION_OPERATIONS = frozenset(
    {"model.request", "source.get", "source.head", "tool.execute"}
)


class PreExecutionModelCall(StrictFrozenContract):
    tool_name: _TEXT
    arguments: str


class PreExecutionModelTurn(StrictFrozenContract):
    """One model response as sent and returned; reasoning text is not retained."""

    attempt_id: _TEXT
    request_number: int = Field(gt=0)
    offered_tools: tuple[str, ...]
    model_settings: dict[str, Any]
    finish_reason: str | None
    calls: tuple[PreExecutionModelCall, ...]
    text: tuple[str, ...]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class PreExecutionToolCall(StrictFrozenContract):
    """One validated Tool call and the turn disposition that admitted or refused it."""

    attempt_id: _TEXT
    tool_name: _TEXT
    status: _TEXT
    errors: tuple[str, ...]
    arguments: dict[str, Any]
    disposition: _TEXT
    correction_key: str | None
    correction_count: int = Field(ge=0)


class PreExecutionReviewCandidate(StrictFrozenContract):
    candidate_id: _TEXT
    display_label: str
    asserted_value: str | None


class PreExecutionPlanningReview(StrictFrozenContract):
    """A planning question and its answer as recorded through the public path."""

    request_event_id: _TEXT
    attempt_id: _TEXT
    subject_key: _TEXT
    question: str
    issue: str
    candidates: tuple[PreExecutionReviewCandidate, ...]
    response: dict[str, Any]
    value: str | None
    authority: _TEXT
    authorizes_execution: Literal[False]


class PreExecutionContacts(StrictFrozenContract):
    """Pass accounting as the public observation reads it; measured, not inferred."""

    pass_count: int = Field(gt=0)
    complete: Literal[True]
    operations: dict[str, int]
    model_logical_requests: int = Field(ge=0)
    model_physical_contacts: int = Field(ge=0)


class PreExecutionFixtureScenario(StrictFrozenContract):
    """Inert planning flow to the unanswered Execution review; never an approval."""

    source_ticket: int = Field(gt=0)
    source_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    evidence_kind: Literal["synthetic", "retained_live"]
    fixture_grants_live_authority: Literal[False] = False
    journey: EvidenceFileIdentity
    journey_id: _TEXT
    path_events: tuple[str, ...]
    status: Literal["execution_hitl_pending"]
    execution_review_event_id: _TEXT
    execution_proposal: dict[str, Any]
    planning_reviews: tuple[PreExecutionPlanningReview, ...]
    model_turns: tuple[PreExecutionModelTurn, ...] = Field(min_length=1)
    tool_calls: tuple[PreExecutionToolCall, ...]
    contacts: PreExecutionContacts
    # The Contact review (E12) reads only execution paths; whether measured pass
    # accounting may decide pre-Execution eligibility is still open.
    fixture_eligibility: Literal["not_computed"] = "not_computed"

    @model_validator(mode="after")
    def unanswered_execution_review(self):
        # Pass intervals only bracket processing; the last domain event is the proposal.
        domain = [
            event
            for event in self.path_events
            if event not in {"PassStarted", "PassClosed"}
        ]
        if (
            not domain
            or domain[0] != "JourneyStarted"
            or domain[-1] != "ExecutionProposalSubmitted"
            or "ExecutionHitlResponded" in domain
        ):
            raise ValueError("pre-Execution fixture must stop at the Execution review")
        return self


class PreExecutionFixture(StrictFrozenContract):
    schema_version: Literal["bio-neb.pre-execution-fixture/1"] = (
        "bio-neb.pre-execution-fixture/1"
    )
    fixture_grants_live_authority: Literal[False] = False
    scenarios: tuple[PreExecutionFixtureScenario, ...] = Field(min_length=1)


def _model_turns(events) -> tuple[PreExecutionModelTurn, ...]:
    requests = {
        (event["payload"]["attempt_id"], event["payload"]["request_number"]): event[
            "payload"
        ]["compatible_adapter_evidence"]["request"]
        for event in events
        if event["event_type"] == "ModelRequestRecorded"
    }
    turns = []
    for event in events:
        if event["event_type"] != "ModelResponseRecorded":
            continue
        payload = event["payload"]
        request = requests[payload["attempt_id"], payload["request_number"]]
        message = payload["compatible_adapter_evidence"]["response"]["message"]
        parts = message["parts"]
        turns.append(
            PreExecutionModelTurn(
                attempt_id=payload["attempt_id"],
                request_number=payload["request_number"],
                offered_tools=tuple(tool["name"] for tool in request["function_tools"]),
                model_settings=request["model_settings"],
                finish_reason=message["finish_reason"],
                calls=tuple(
                    PreExecutionModelCall(
                        tool_name=part["tool_name"],
                        arguments=part["args"]
                        if isinstance(part["args"], str)
                        else canonical_json(part["args"]),
                    )
                    for part in parts
                    if part["part_kind"] == "tool-call"
                ),
                text=tuple(
                    part["content"] for part in parts if part["part_kind"] == "text"
                ),
                input_tokens=message["usage"]["input_tokens"],
                output_tokens=message["usage"]["output_tokens"],
            )
        )
    return tuple(turns)


def _tool_calls(events) -> tuple[PreExecutionToolCall, ...]:
    dispositions = {
        validation_id: event["payload"]
        for event in events
        if event["event_type"] == "AgentActionTurnDispositionRecorded"
        for validation_id in event["payload"]["validation_event_ids"]
    }
    calls = []
    for event in events:
        if event["event_type"] != "ToolCallValidated":
            continue
        payload = event["payload"]
        disposition = dispositions[event["event_id"]]
        calls.append(
            PreExecutionToolCall(
                attempt_id=payload["attempt_id"],
                tool_name=payload["tool_name"],
                status=payload["status"],
                errors=tuple(payload["errors"]),
                arguments=payload["arguments"],
                disposition=disposition["disposition"],
                correction_key=disposition.get("correction_key"),
                correction_count=disposition.get("correction_count") or 0,
            )
        )
    return tuple(calls)


def _planning_reviews(events) -> tuple[PreExecutionPlanningReview, ...]:
    answers = {
        event["payload"]["request_event_id"]: event["payload"]
        for event in events
        if event["event_type"] == "PlanningHitlResponded"
    }
    reviews = []
    for event in events:
        if event["event_type"] != "PlanningHitlRequested":
            continue
        request = event["payload"]
        answer = answers[event["event_id"]]
        reviews.append(
            PreExecutionPlanningReview(
                request_event_id=event["event_id"],
                attempt_id=request["attempt_id"],
                subject_key=request["subject_key"],
                question=request["question"],
                issue=request["issue"],
                candidates=tuple(
                    PreExecutionReviewCandidate(
                        candidate_id=candidate["candidate_id"],
                        display_label=candidate["display_label"],
                        asserted_value=candidate.get("asserted_value"),
                    )
                    for candidate in request["candidates"]
                ),
                response=answer["response"],
                value=answer.get("value"),
                authority=answer["authority"],
                authorizes_execution=answer["authorizes_execution"],
            )
        )
    return tuple(reviews)


def _pre_execution_contacts(view) -> PreExecutionContacts:
    postmortem = view.model_contact_postmortem
    if (
        not view.passes
        or postmortem is None
        or any(
            item.disposition == "active"
            or not item.journal_complete
            or item.accounting_error is not None
            for item in view.passes
        )
    ):
        raise ValueError("complete pass accounting required")
    operations = Counter(
        row.operation
        for item in view.passes
        for row in item.contacts
        if row.kind == "start"
    )
    if not set(operations) <= _PRE_EXECUTION_OPERATIONS:
        raise ValueError("pre-Execution contact outside planning operations")
    return PreExecutionContacts(
        pass_count=len(view.passes),
        complete=True,
        operations=dict(sorted(operations.items())),
        model_logical_requests=postmortem.logical_request_count,
        model_physical_contacts=postmortem.physical_contact_count,
    )


def compile_pre_execution_fixture(
    journey_file: Path, *, source_commit: str, source_ticket: int, evidence_kind: str
) -> bytes:
    """Export the checked planning path of a Run stopped at its Execution review.

    It reads the public observation and the validated Journey events; it neither
    answers the Execution review nor retains anything that could approve one.
    """
    from bio_neb.application.queries import ObserveRun, observe
    from bio_neb.application.run_interaction import run_interaction

    try:
        journey_file = journey_file.resolve(strict=True)
        owner = Journey.open(journey_file)
        path = JourneyPath.from_journey(owner)
        events = path.planning_events()
        view = run_interaction(observe(ObserveRun(journey_file=str(journey_file))))
        review = view.review
        if (
            view.status != "execution_hitl_pending"
            or review is None
            or review.kind != "execution"
            or len(view.pending_reviews) != 1
            or view.execution_proposal is None
        ):
            raise ValueError("Run must stop at its one unanswered Execution review")
        if not owner.is_current():
            raise ValueError("Journey changed during export")
        identity = owner.identity()
        fixture = PreExecutionFixture(
            scenarios=(
                PreExecutionFixtureScenario(
                    source_ticket=source_ticket,
                    source_commit=source_commit,
                    evidence_kind=evidence_kind,
                    journey=EvidenceFileIdentity(
                        path=str(journey_file),
                        byte_count=identity.byte_count,
                        sha256=identity.journey_sha256,
                    ),
                    journey_id=view.journey_id,
                    path_events=tuple(event["event_type"] for event in events),
                    status=view.status,
                    execution_review_event_id=review.request_event_id,
                    execution_proposal=view.execution_proposal.model_dump(mode="json"),
                    planning_reviews=_planning_reviews(events),
                    model_turns=_model_turns(events),
                    tool_calls=_tool_calls(events),
                    contacts=_pre_execution_contacts(view),
                ),
            )
        )
        return canonical_json(fixture.model_dump(mode="json")).encode()
    except (
        OSError,
        ValueError,
        KeyError,
        AttributeError,
        TypeError,
        JourneyValidationError,
    ) as error:
        raise FixtureExportRefused("pre_execution_evidence_invalid") from error


def load_pre_execution_fixture(content: bytes) -> PreExecutionFixture:
    """Strictly load inert exported planning evidence; never consult its original Run."""
    return PreExecutionFixture.model_validate_json(content)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--input-manifest", required=True)
    export.add_argument("--output", required=True)
    review = subparsers.add_parser("review")
    review.add_argument("--fixture", required=True)
    review.add_argument("--expected-sha256", required=True)
    inputs = subparsers.add_parser("export-input-storage")
    inputs.add_argument("--journey-file", required=True)
    inputs.add_argument("--source-commit", required=True)
    inputs.add_argument("--output", required=True)
    execution = subparsers.add_parser("export-execution")
    execution.add_argument("--journey-file", required=True)
    execution.add_argument("--source-commit", required=True)
    execution.add_argument("--source-ticket", required=True, type=int)
    execution.add_argument(
        "--evidence-kind", required=True, choices=("synthetic", "retained_live")
    )
    execution.add_argument("--output", required=True)
    pre_execution = subparsers.add_parser("export-pre-execution")
    pre_execution.add_argument("--journey-file", required=True)
    pre_execution.add_argument("--source-commit", required=True)
    pre_execution.add_argument("--source-ticket", required=True, type=int)
    pre_execution.add_argument(
        "--evidence-kind", required=True, choices=("synthetic", "retained_live")
    )
    pre_execution.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation in {"export-execution", "export-pre-execution"}:
        compiler = (
            compile_execution_fixture
            if args.operation == "export-execution"
            else compile_pre_execution_fixture
        )
        content = compiler(
            Path(args.journey_file),
            source_commit=args.source_commit,
            source_ticket=args.source_ticket,
            evidence_kind=args.evidence_kind,
        )
        publish_exclusive(Path(args.output), content)
        print(
            canonical_json(
                {
                    "fixture": args.output,
                    "byte_count": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        )
        return 0
    if args.operation == "export-input-storage":
        content = compile_input_storage_fixture(
            Path(args.journey_file), source_commit=args.source_commit
        )
        publish_exclusive(Path(args.output), content)
        print(
            canonical_json(
                {
                    "fixture": args.output,
                    "byte_count": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        )
        return 0
    if args.operation == "review":
        os.write(
            1,
            review_fixture(Path(args.fixture), expected_sha256=args.expected_sha256),
        )
        return 0
    receipt = export_fixture(
        load_export_input(Path(args.input_manifest)), Path(args.output)
    )
    print(canonical_json(receipt.model_dump(mode="json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPORTER_VERSION",
    "EvidenceFileIdentity",
    "ExportedFixtureReceipt",
    "FixtureExportRefused",
    "FixtureScenario",
    "CacheHitFixtureScenario",
    "CacheMissFixtureScenario",
    "ParabricksFixtureExportInput",
    "ParabricksRunFixtureBundle",
    "CacheHitEvidenceInput",
    "ExecutionFixture",
    "ExecutionFixtureScenario",
    "PreExecutionFixture",
    "PreExecutionFixtureScenario",
    "compile_execution_fixture",
    "compile_pre_execution_fixture",
    "load_execution_fixture",
    "load_pre_execution_fixture",
    "compile_fixture",
    "compile_input_storage_fixture",
    "export_fixture",
    "load_export_input",
    "main",
    "publish_exclusive",
    "review_fixture",
]
