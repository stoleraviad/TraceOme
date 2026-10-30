"""Compile and read exact execution capability snapshots for Planning."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Tuple, Union
from urllib.parse import parse_qsl, urlsplit

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from bio_neb.contracts import FrozenContract
from bio_neb.journey import canonical_json, canonical_json_digest
from bio_neb.parabricks.execution_runtime import (
    CpuReferenceRuntime,
    H100Runtime,
    Runtimes,
)
from bio_neb.parabricks.recipe_resources import (
    CapabilityImageClaim,
    CapabilitySource,
    CapabilitySourceManifestEntry,
    CapabilityToolClaim,
    ListRecipesInput,
    RecipeAdapterObservation,
    RecipeCandidate,
    ResourceAdapterObservation,
    ResourceWorkItem,
    make_recipe_candidate,
    make_resource_candidate,
)

_SHA256 = r"^[0-9a-f]{64}$"
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class CapabilitySnapshotError(ValueError):
    """Typed, contact-free authoring or publication refusal."""

    def __init__(self, code: str, fields: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


def _credential_free_locator(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("capability source locator must be canonical")
    parsed = urlsplit(value)
    sensitive = {"token", "key", "secret", "password", "credential", "signature"}
    if (
        parsed.scheme not in {"file", "http", "https"}
        or parsed.scheme != parsed.scheme.casefold()
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or any(
            any(marker in name.casefold() for marker in sensitive)
            for name, _ in parse_qsl(parsed.query, keep_blank_values=True)
        )
    ):
        raise ValueError(
            "capability source locator must be canonical and credential-free"
        )
    if parsed.scheme == "file":
        if parsed.netloc or not parsed.path.startswith("/") or parsed.query:
            raise ValueError(
                "file capability locator must be one canonical absolute URI"
            )
    elif not parsed.netloc or parsed.netloc != parsed.netloc.casefold():
        raise ValueError("remote capability locator must have one canonical host")
    if parsed.geturl() != value:
        raise ValueError("capability source locator must be canonical")
    return value


def _retrieval_time(value: str) -> str:
    if not _TIMESTAMP.fullmatch(value):
        raise ValueError("retrieved_at must be canonical UTC seconds")
    datetime.fromisoformat(value[:-1] + "+00:00")
    return value


class LocalCapabilitySourceInput(FrozenContract):
    kind: Literal["local_bytes"]
    locator: StrictStr
    retrieved_at: StrictStr
    path: StrictStr

    _safe_locator = field_validator("locator")(_credential_free_locator)
    _canonical_time = field_validator("retrieved_at")(_retrieval_time)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError(
                "local capability source path must be canonical and absolute"
            )
        return value


class RetainedCapabilitySourceInput(FrozenContract):
    kind: Literal["retained_nonlocal"]
    locator: StrictStr
    retrieved_at: StrictStr
    content_digest: StrictStr = Field(pattern=_SHA256)

    _safe_locator = field_validator("locator")(_credential_free_locator)
    _canonical_time = field_validator("retrieved_at")(_retrieval_time)


CapabilitySourceInput = Annotated[
    Union[LocalCapabilitySourceInput, RetainedCapabilitySourceInput],
    Field(discriminator="kind"),
]


class CapabilityPolicyRecipe(FrozenContract):
    recipe_id: StrictStr
    assay: StrictStr
    read_layout: Literal["paired", "single"]
    output_kind: StrictStr
    requires_prepared_reference: StrictBool = True
    requires_companion_roles: Tuple[StrictStr, ...]


class CapabilityPolicyResource(FrozenContract):
    profile_id: StrictStr
    recipe_id: StrictStr
    image_id: StrictStr
    tool_ids: Tuple[StrictStr, ...] = Field(min_length=1)
    accelerator_count: StrictInt = Field(ge=1)
    cpu_count: StrictInt = Field(ge=1)
    memory_gib: StrictInt = Field(ge=1)
    accelerator_memory_gib: StrictInt = Field(ge=1)


CapabilityPolicyImage = CapabilityImageClaim
CapabilityPolicyTool = CapabilityToolClaim


class CapabilityRuntimeProfile(FrozenContract):
    """Complete runtime claim, including provider target values absent from resources."""

    profile_id: StrictStr
    cpu: CpuReferenceRuntime
    h100: H100Runtime


class CapabilitySnapshotAuthoring(FrozenContract):
    schema_version: Literal["bio-neb.execution-capability-authoring/1"]
    capability_version: StrictStr
    sources: Tuple[CapabilitySourceInput, ...] = Field(min_length=1)
    recipes: Tuple[CapabilityPolicyRecipe, ...]
    resources: Tuple[CapabilityPolicyResource, ...]
    images: Tuple[CapabilityPolicyImage, ...] = Field(min_length=1)
    tools: Tuple[CapabilityPolicyTool, ...] = Field(min_length=1)
    runtime_profiles: tuple[CapabilityRuntimeProfile, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class CapabilitySnapshot(FrozenContract):
    schema_version: Literal["bio-neb.execution-capability-snapshot/1"]
    snapshot_id: StrictStr = Field(pattern=_SHA256)
    capability_version: StrictStr
    source_manifest: Tuple[CapabilitySourceManifestEntry, ...] = Field(min_length=1)
    recipes: Tuple[CapabilityPolicyRecipe, ...]
    resources: Tuple[CapabilityPolicyResource, ...]
    images: Tuple[CapabilityPolicyImage, ...] = Field(min_length=1)
    tools: Tuple[CapabilityPolicyTool, ...] = Field(min_length=1)
    runtime_profiles: tuple[CapabilityRuntimeProfile, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def canonical_and_cross_bound(self) -> "CapabilitySnapshot":
        _validate_claims(
            self.source_manifest, self.recipes, self.resources, self.images, self.tools
        )
        self._validate_runtime_profiles()
        if self.source_manifest != tuple(
            sorted(
                self.source_manifest,
                key=lambda item: (item.locator, item.retrieved_at, item.content_digest),
            )
        ):
            raise ValueError("capability source manifest must be canonical")
        if self.recipes != tuple(sorted(self.recipes, key=lambda item: item.recipe_id)):
            raise ValueError("capability recipes must be canonical")
        if self.resources != tuple(
            sorted(self.resources, key=lambda item: item.profile_id)
        ):
            raise ValueError("capability resources must be canonical")
        if self.images != tuple(sorted(self.images, key=lambda item: item.image_id)):
            raise ValueError("capability images must be canonical")
        if self.tools != tuple(sorted(self.tools, key=lambda item: item.tool_id)):
            raise ValueError("capability tools must be canonical")
        claims = self.model_dump(mode="json", exclude={"snapshot_id"})
        if self.snapshot_id != canonical_json_digest(claims):
            raise ValueError(
                "snapshot_id must cover the exact canonical capability claims"
            )
        return self

    def _validate_runtime_profiles(self) -> None:
        if self.runtime_profiles != tuple(
            sorted(self.runtime_profiles, key=lambda item: item.profile_id)
        ) or len({item.profile_id for item in self.runtime_profiles}) != len(
            self.runtime_profiles
        ):
            raise ValueError("runtime profiles must be canonical and unique")
        for profile in self.runtime_profiles:
            gpu = profile.h100
            resource = next(
                (
                    item
                    for item in self.resources
                    if item.profile_id == gpu.resource_profile_id
                    and item.recipe_id == gpu.recipe_id
                ),
                None,
            )
            if resource is None or (
                resource.accelerator_count,
                resource.cpu_count,
                resource.memory_gib,
                resource.accelerator_memory_gib,
            ) != (
                gpu.gpu_count,
                gpu.cpu_count,
                gpu.memory_gib,
                gpu.accelerator_memory_gib,
            ):
                raise ValueError("runtime profile must match its capability resource")
            image = next(
                item for item in self.images if item.image_id == resource.image_id
            )
            for runtime in (profile.cpu, gpu):
                if (
                    runtime.provider_image_reference,
                    runtime.provider_image_index_digest,
                    runtime.worker_runtime_image,
                    runtime.worker_sha256,
                ) != (
                    image.provider_image,
                    image.image_index_digest,
                    image.provider_image.rpartition(":")[0]
                    + "@"
                    + image.platform_child_digest,
                    image.worker_sha256,
                ):
                    raise ValueError("runtime profile must match its capability image")
            if (
                profile.cpu.worker_module != image.worker_module
                or gpu.python_path != image.worker_entrypoint[0]
            ):
                raise ValueError("runtime profile must match its capability entrypoint")
            tools = {
                (item.absolute_path, item.version)
                for item in self.tools
                if item.image_id == image.image_id and item.tool_id in resource.tool_ids
            }
            if (
                not {
                    ("/usr/bin/bwa", profile.cpu.bwa_version),
                    ("/usr/bin/samtools", profile.cpu.samtools_version),
                    (gpu.pbrun_path, gpu.pbrun_version),
                    (gpu.python_path, gpu.python_version),
                }
                <= tools
            ):
                raise ValueError("runtime profile must match its capability tools")


@dataclass(frozen=True)
class SelectedRuntime:
    """Current approved choice and the automatic successors it still permits."""

    profile_id: str
    remaining_profile_ids: tuple[str, ...]
    runtimes: Runtimes

    @property
    def next_profile_id(self) -> str | None:
        return next(iter(self.remaining_profile_ids), None)


class ExecutionRuntimeSelection(FrozenContract):
    capability: CapabilitySnapshot
    profile_ids: tuple[StrictStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_profiles(self):
        profiles = {item.profile_id: item for item in self.capability.runtime_profiles}
        if len(set(self.profile_ids)) != len(self.profile_ids) or any(
            item not in profiles for item in self.profile_ids
        ):
            raise ValueError("runtime selection requires exact capability profiles")
        first = profiles[self.profile_ids[0]]
        if any(
            profiles[item].cpu != first.cpu
            or profiles[item].h100.recipe_id != first.h100.recipe_id
            for item in self.profile_ids
        ):
            raise ValueError(
                "runtime fallbacks must preserve CPU preparation and recipe"
            )
        return self

    def select(
        self,
        *,
        amended_profile_id: str | None = None,
        substituted_profile_id: str | None = None,
    ) -> SelectedRuntime:
        """Resolve manual compatibility separately from ordered automatic authority."""
        profiles = {item.profile_id: item for item in self.capability.runtime_profiles}
        first = profiles[self.profile_ids[0]]
        current = (
            self.profile_ids[0] if amended_profile_id is None else amended_profile_id
        )
        if not isinstance(current, str) or current not in profiles:
            raise ValueError("unknown runtime amendment")
        profile = profiles[current]
        if profile.cpu != first.cpu or profile.h100.recipe_id != first.h100.recipe_id:
            raise ValueError("amendment changes preparation or recipe")
        remaining = (
            self.profile_ids[self.profile_ids.index(current) + 1 :]
            if current in self.profile_ids
            else ()
        )
        if substituted_profile_id is not None:
            if substituted_profile_id not in remaining:
                raise ValueError("unapproved runtime substitution")
            current = substituted_profile_id
            profile = profiles[current]
            remaining = remaining[remaining.index(current) + 1 :]
        return SelectedRuntime(
            profile_id=current,
            remaining_profile_ids=remaining,
            runtimes=Runtimes(
                cpu=first.cpu,
                h100=profile.h100,
                fallbacks=tuple(profiles[item].h100 for item in remaining),
            ),
        )

    def resolve(self) -> Runtimes:
        return self.select().runtimes


class CapabilitySnapshotAudit(FrozenContract):
    status: Literal["capability_snapshot_compiled"] = "capability_snapshot_compiled"
    snapshot_id: StrictStr = Field(pattern=_SHA256)
    capability_version: StrictStr
    source_manifest: Tuple[CapabilitySourceManifestEntry, ...]
    recipe_ids: Tuple[StrictStr, ...]
    resource_ids: Tuple[StrictStr, ...]
    image_ids: Tuple[StrictStr, ...]
    tool_ids: Tuple[StrictStr, ...]


def _unique(label: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} identities must be unique")


def _validate_claims(
    manifest: Tuple[CapabilitySourceManifestEntry, ...],
    recipes: Tuple[CapabilityPolicyRecipe, ...],
    resources: Tuple[CapabilityPolicyResource, ...],
    images: Tuple[CapabilityPolicyImage, ...],
    tools: Tuple[CapabilityPolicyTool, ...],
) -> None:
    for entry in manifest:
        _credential_free_locator(entry.locator)
        _retrieval_time(entry.retrieved_at)
    _unique("source locator", tuple(item.locator for item in manifest))
    _unique("recipe", tuple(item.recipe_id for item in recipes))
    _unique("resource", tuple(item.profile_id for item in resources))
    _unique("image", tuple(item.image_id for item in images))
    _unique("tool", tuple(item.tool_id for item in tools))
    recipe_ids = {item.recipe_id for item in recipes}
    image_ids = {item.image_id for item in images}
    tool_ids = {item.tool_id for item in tools}
    for recipe in recipes:
        if recipe.requires_companion_roles != tuple(
            sorted(set(recipe.requires_companion_roles))
        ):
            raise ValueError("recipe companion roles must be canonical and unique")
    for tool in tools:
        if tool.image_id is not None and tool.image_id not in image_ids:
            raise ValueError("tool image identity does not exist")
    used_images: set[str] = set()
    used_tools: set[str] = set()
    for resource in resources:
        if resource.recipe_id not in recipe_ids:
            raise ValueError("resource recipe identity does not exist")
        if resource.image_id not in image_ids:
            raise ValueError("resource image identity does not exist")
        if resource.tool_ids != tuple(sorted(set(resource.tool_ids))):
            raise ValueError("resource tool identities must be canonical and unique")
        if not set(resource.tool_ids) <= tool_ids:
            raise ValueError("resource tool identity does not exist")
        worker_images = {
            tool.image_id
            for tool in tools
            if tool.tool_id in resource.tool_ids
            and tool.execution_scope == "worker_image"
        }
        if worker_images != {resource.image_id}:
            raise ValueError("resource worker tools must bind its exact image")
        used_images.add(resource.image_id)
        used_tools.update(resource.tool_ids)
    if resources and (used_images != image_ids or used_tools != tool_ids):
        raise ValueError("capability images and tools must be cross-bound to resources")


def _hash_local_source(
    source: LocalCapabilitySourceInput,
) -> CapabilitySourceManifestEntry:
    path = Path(source.path)
    try:
        before = path.stat()
        if not path.is_file():
            raise OSError("not a regular file")
        digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                byte_count += len(chunk)
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise CapabilitySnapshotError(
            "capability_source_unavailable", (source.path,)
        ) from error
    if (
        before.st_size != byte_count
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or getattr(before, "st_ino", None) != getattr(after, "st_ino", None)
    ):
        raise CapabilitySnapshotError("capability_source_changed", (source.path,))
    return CapabilitySourceManifestEntry(
        locator=source.locator,
        content_digest=digest.hexdigest(),
        retrieved_at=source.retrieved_at,
        content_assurance="application_hashed_local_bytes",
    )


def _canonical_authoring(
    authoring: CapabilitySnapshotAuthoring,
) -> tuple[
    Tuple[CapabilitySourceManifestEntry, ...],
    Tuple[CapabilityPolicyRecipe, ...],
    Tuple[CapabilityPolicyResource, ...],
    Tuple[CapabilityPolicyImage, ...],
    Tuple[CapabilityPolicyTool, ...],
]:
    manifest = []
    for source in authoring.sources:
        if isinstance(source, LocalCapabilitySourceInput):
            manifest.append(_hash_local_source(source))
        else:
            manifest.append(
                CapabilitySourceManifestEntry(
                    locator=source.locator,
                    content_digest=source.content_digest,
                    retrieved_at=source.retrieved_at,
                    content_assurance="retained_exact_digest",
                )
            )
    recipes = tuple(
        sorted(
            (
                item.model_copy(
                    update={
                        "requires_companion_roles": tuple(
                            sorted(item.requires_companion_roles)
                        )
                    }
                )
                for item in authoring.recipes
            ),
            key=lambda item: item.recipe_id,
        )
    )
    resources = tuple(
        sorted(
            (
                item.model_copy(update={"tool_ids": tuple(sorted(item.tool_ids))})
                for item in authoring.resources
            ),
            key=lambda item: item.profile_id,
        )
    )
    return (
        tuple(
            sorted(
                manifest,
                key=lambda item: (item.locator, item.retrieved_at, item.content_digest),
            )
        ),
        recipes,
        resources,
        tuple(sorted(authoring.images, key=lambda item: item.image_id)),
        tuple(sorted(authoring.tools, key=lambda item: item.tool_id)),
    )


def compile_capability_snapshot_value(
    authoring: CapabilitySnapshotAuthoring,
) -> CapabilitySnapshot:
    manifest, recipes, resources, images, tools = _canonical_authoring(authoring)
    _validate_claims(manifest, recipes, resources, images, tools)
    claims = {
        "schema_version": "bio-neb.execution-capability-snapshot/1",
        "capability_version": authoring.capability_version,
        "source_manifest": [item.model_dump(mode="json") for item in manifest],
        "recipes": [item.model_dump(mode="json") for item in recipes],
        "resources": [item.model_dump(mode="json") for item in resources],
        "images": [item.model_dump(mode="json") for item in images],
        "tools": [item.model_dump(mode="json") for item in tools],
    }
    if authoring.runtime_profiles:
        claims["runtime_profiles"] = [
            item.model_dump(mode="json")
            for item in sorted(
                authoring.runtime_profiles, key=lambda item: item.profile_id
            )
        ]
    return CapabilitySnapshot.model_validate(
        {**claims, "snapshot_id": canonical_json_digest(claims)}
    )


def read_capability_snapshot(path: Path) -> CapabilitySnapshot:
    """Independently reopen one canonical compiled snapshot without source contact."""

    try:
        raw = path.read_bytes()
        snapshot = CapabilitySnapshot.model_validate_json(raw)
    except (OSError, UnicodeError, ValidationError) as error:
        raise CapabilitySnapshotError(
            "capability_snapshot_invalid", (str(path),)
        ) from error
    expected = canonical_json(snapshot.model_dump(mode="json")).encode("utf-8")
    if raw != expected:
        raise CapabilitySnapshotError("capability_snapshot_not_canonical", (str(path),))
    return snapshot


def capability_snapshot_audit(snapshot: CapabilitySnapshot) -> CapabilitySnapshotAudit:
    return CapabilitySnapshotAudit(
        snapshot_id=snapshot.snapshot_id,
        capability_version=snapshot.capability_version,
        source_manifest=snapshot.source_manifest,
        recipe_ids=tuple(item.recipe_id for item in snapshot.recipes),
        resource_ids=tuple(item.profile_id for item in snapshot.resources),
        image_ids=tuple(item.image_id for item in snapshot.images),
        tool_ids=tuple(item.tool_id for item in snapshot.tools),
    )


def compile_capability_snapshot(
    authoring_path: Path, target_path: Path
) -> CapabilitySnapshotAudit:
    """Compile, publish without overwrite, then independently reopen a snapshot."""

    try:
        raw = authoring_path.read_bytes()
        authoring = CapabilitySnapshotAuthoring.model_validate_json(raw)
    except (OSError, UnicodeError, ValidationError) as error:
        raise CapabilitySnapshotError(
            "capability_authoring_invalid", (str(authoring_path),)
        ) from error
    snapshot = compile_capability_snapshot_value(authoring)
    payload = canonical_json(snapshot.model_dump(mode="json")).encode("utf-8")
    target_path = target_path.resolve()
    if not target_path.parent.is_dir():
        raise CapabilitySnapshotError(
            "capability_snapshot_target_invalid", (str(target_path),)
        )
    temporary = target_path.parent / f".{target_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target_path)
        directory_descriptor = os.open(target_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as error:
        raise CapabilitySnapshotError(
            "capability_snapshot_target_exists", (str(target_path),)
        ) from error
    except OSError as error:
        raise CapabilitySnapshotError(
            "capability_snapshot_publication_failed", (str(target_path),)
        ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    reopened = read_capability_snapshot(target_path)
    if reopened != snapshot:
        raise CapabilitySnapshotError(
            "capability_snapshot_reopen_mismatch", (str(target_path),)
        )
    return capability_snapshot_audit(reopened)


class CapabilitySnapshotAdapter:
    """Compare exact Journey subjects with one compiled capability snapshot."""

    def __init__(self, snapshot: CapabilitySnapshot) -> None:
        self.snapshot = snapshot
        self.contact_count = 0

    def _source(self, request: object, response: object) -> CapabilitySource:
        return CapabilitySource(
            authority="compiled_capability_snapshot",
            capability_version=self.snapshot.capability_version,
            snapshot_id=self.snapshot.snapshot_id,
            source_manifest=self.snapshot.source_manifest,
            images=self.snapshot.images,
            tools=self.snapshot.tools,
            request_identity=canonical_json_digest(request),
            response_digest=canonical_json_digest(response),
        )

    def list_recipes(self, tool_input: ListRecipesInput) -> RecipeAdapterObservation:
        self.contact_count += 1
        work_item = tool_input.work_item
        candidates: list[RecipeCandidate] = []
        for claim in self.snapshot.recipes:
            reasons = []
            if not set(claim.requires_companion_roles) <= set(
                work_item.companion_roles
            ):
                reasons.append("policy_required_companion_roles_absent")
            # Requested outputs are typed kinds, compared as recorded.
            if claim.output_kind not in work_item.proposal_output_values:
                reasons.append("policy_output_kind_not_requested")
            assertion = tool_input.eligibility_assertion
            if assertion is not None:
                if claim.assay.casefold() != assertion.assay.casefold():
                    reasons.append("policy_assay_mismatch")
                if claim.read_layout != assertion.read_layout:
                    reasons.append("policy_read_layout_mismatch")
            compatibility = (
                "incompatible"
                if reasons
                else "ambiguous"
                if assertion is None
                else "compatible"
            )
            if compatibility == "ambiguous":
                reasons.append("input_eligibility_assertion_required")
            candidates.append(
                make_recipe_candidate(
                    recipe_id=claim.recipe_id,
                    assay=claim.assay,
                    read_layout=claim.read_layout,
                    output_kind=claim.output_kind,
                    requires_prepared_reference=claim.requires_prepared_reference,
                    requires_companion_roles=claim.requires_companion_roles,
                    compatibility=compatibility,
                    incompatibility_reasons=tuple(reasons),
                    snapshot_id=self.snapshot.snapshot_id,
                    unresolved_constraints=(
                        "prepared_reference_not_verified",
                        "software_installation_not_verified",
                        "provider_environment_availability_not_assessed",
                    ),
                )
            )
        candidates.sort(key=lambda item: item.candidate_id)
        request = tool_input.model_dump(mode="json")
        response = [item.model_dump(mode="json") for item in candidates]
        return RecipeAdapterObservation(
            status="ok",
            candidates=tuple(candidates),
            source=self._source(request, response),
        )

    def lookup_resources(
        self, work_item: ResourceWorkItem
    ) -> ResourceAdapterObservation:
        self.contact_count += 1
        values = []
        for claim in self.snapshot.resources:
            reasons = (
                ()
                if claim.recipe_id == work_item.recipe_candidate.recipe_id
                else ("policy_recipe_mismatch",)
            )
            values.append(
                make_resource_candidate(
                    profile_id=claim.profile_id,
                    recipe_candidate_id=work_item.recipe_candidate.candidate_id,
                    accelerator_count=claim.accelerator_count,
                    cpu_count=claim.cpu_count,
                    memory_gib=claim.memory_gib,
                    accelerator_memory_gib=claim.accelerator_memory_gib,
                    compatibility="compatible" if not reasons else "incompatible",
                    incompatibility_reasons=reasons,
                    snapshot_id=self.snapshot.snapshot_id,
                    unresolved_constraints=(
                        "accelerator_model_and_driver_not_verified",
                        "software_installation_not_verified",
                        "provider_environment_availability_not_assessed",
                        "single_gpu_supported_not_recommended"
                        if claim.accelerator_count == 1
                        else "recommended_envelope_not_availability_proof",
                    ),
                )
            )
        values.sort(key=lambda item: item.candidate_id)
        request = work_item.model_dump(mode="json")
        response = [item.model_dump(mode="json") for item in values]
        return ResourceAdapterObservation(
            status="ok",
            candidates=tuple(values),
            source=self._source(request, response),
        )


__all__ = [
    "CapabilityPolicyImage",
    "CapabilityPolicyRecipe",
    "CapabilityPolicyResource",
    "CapabilityPolicyTool",
    "CapabilitySnapshot",
    "CapabilitySnapshotAdapter",
    "CapabilitySnapshotAudit",
    "CapabilitySnapshotAuthoring",
    "CapabilitySnapshotError",
    "CapabilitySourceManifestEntry",
    "LocalCapabilitySourceInput",
    "RetainedCapabilitySourceInput",
    "capability_snapshot_audit",
    "compile_capability_snapshot",
    "compile_capability_snapshot_value",
    "read_capability_snapshot",
]
