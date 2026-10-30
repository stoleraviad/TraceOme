"""Typed artifact verification defaults and retained invocation interpretation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_serializer, model_validator

from bio_neb.journey import canonical_json_digest

ObjectVerificationMode = Literal["reuse_creation_proof", "independent_full_read"]


class ObjectVerificationModes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_storage: ObjectVerificationMode
    prepared_candidate: ObjectVerificationMode
    reference_preparation: ObjectVerificationMode
    h100_outputs: ObjectVerificationMode


# All placements share these modes. Ordinary use carries original creation proof;
# only a reasoned explicit override performs an independent full read.
DEFAULT_OBJECT_VERIFICATION_MODE: ObjectVerificationMode = "reuse_creation_proof"


class ObjectVerificationOverrides(BaseModel):
    """Sparse per-slot selections; a full read must say why it was selected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_storage: ObjectVerificationMode | None = None
    prepared_candidate: ObjectVerificationMode | None = None
    reference_preparation: ObjectVerificationMode | None = None
    h100_outputs: ObjectVerificationMode | None = None
    reason: StrictStr | None = None

    @model_validator(mode="after")
    def explicit_values(self) -> ObjectVerificationOverrides:
        if any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("explicit object verification overrides cannot be null")
        full_read_selected = "independent_full_read" in self.slot_settings().values()
        if full_read_selected and not (self.reason or "").strip():
            raise ValueError(
                "an independent_full_read override requires a recorded reason"
            )
        if self.reason is not None and not full_read_selected:
            raise ValueError(
                "an override reason requires an independent_full_read selection"
            )
        return self

    def slot_settings(self) -> dict[str, ObjectVerificationMode]:
        """The explicitly overridden slots only, without the reason."""

        return {
            name: getattr(self, name)
            for name in ObjectVerificationModes.model_fields
            if name in self.model_fields_set
        }

    @model_serializer
    def explicit_settings(self) -> dict:
        settings: dict = self.slot_settings()
        if self.reason is not None:
            settings["reason"] = self.reason
        return settings


class ObjectVerificationCheckpoints(BaseModel):
    """Fixed deployed producer/consumer guarantees, separate from host observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    controller_upload: Literal["stream_sha256"] = "stream_sha256"
    worker_download: Literal["stream_sha256"] = "stream_sha256"
    worker_output_admission: Literal["full_sha256"] = "full_sha256"
    worker_upload: Literal["stream_sha256"] = "stream_sha256"
    local_destination_admission: Literal["transfer_integrity"] = "transfer_integrity"
    integrity_failure: Literal["refuse_without_retry_or_adoption"] = (
        "refuse_without_retry_or_adoption"
    )


class ResolvedObjectVerificationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_mode: Literal["reuse_creation_proof"]
    overrides: ObjectVerificationOverrides
    effective: ObjectVerificationModes
    checkpoints: ObjectVerificationCheckpoints
    policy_id: StrictStr

    @model_validator(mode="before")
    @classmethod
    def complete_checkpoints(cls, value: object) -> object:
        if isinstance(value, Mapping):
            checks = value.get("checkpoints")
            if (
                isinstance(checks, Mapping)
                and checks
                != ObjectVerificationCheckpoints.model_validate(checks).model_dump()
            ):
                raise ValueError("retained object checkpoints must be complete")
        return value

    @model_validator(mode="after")
    def exact_resolution(self) -> ResolvedObjectVerificationPolicy:
        expected = {
            name: self.default_mode for name in ObjectVerificationModes.model_fields
        }
        expected.update(self.overrides.slot_settings())
        if self.effective.model_dump() != expected or self.policy_id != (
            "object-verification-policy:"
            + canonical_json_digest(self.model_dump(mode="json", exclude={"policy_id"}))
        ):
            raise ValueError("retained object verification policy changed")
        return self


def resolve_object_verification_policy(
    default_mode: ObjectVerificationMode, overrides: ObjectVerificationOverrides
) -> ResolvedObjectVerificationPolicy:
    if default_mode != DEFAULT_OBJECT_VERIFICATION_MODE:
        raise ValueError("independent reads require an explicit reasoned override")
    effective = {name: default_mode for name in ObjectVerificationModes.model_fields}
    effective.update(overrides.slot_settings())
    material = {
        "default_mode": default_mode,
        "overrides": overrides.explicit_settings(),
        "effective": effective,
        "checkpoints": ObjectVerificationCheckpoints().model_dump(),
    }
    return ResolvedObjectVerificationPolicy(
        **material,
        policy_id="object-verification-policy:" + canonical_json_digest(material),
    )


class ArtifactVerificationDefaults(BaseModel):
    """The same reasoned selection applies to local and remote artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    mode: ObjectVerificationMode = DEFAULT_OBJECT_VERIFICATION_MODE
    reason: StrictStr | None = None

    @model_validator(mode="after")
    def reasoned_selection(self):
        if self.mode == "independent_full_read":
            if not (self.reason or "").strip():
                raise ValueError(
                    "an independent_full_read override requires a recorded reason"
                )
        elif self.reason is not None:
            raise ValueError("a reason requires an independent_full_read selection")
        return self


class ArtifactVerificationOverrides(BaseModel):
    """An explicit selection amends an invocation, including its reason."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    mode: ObjectVerificationMode | None = None
    reason: StrictStr | None = None

    @model_validator(mode="after")
    def explicit_selection(self):
        if self.model_fields_set:
            if self.mode is None:
                raise ValueError("an explicit artifact override requires a mode")
            ArtifactVerificationDefaults(mode=self.mode, reason=self.reason)
        return self

    @model_serializer
    def explicit_settings(self) -> dict:
        return (
            {"mode": self.mode, "reason": self.reason} if self.mode is not None else {}
        )


def _policy_identity(material: dict) -> str:
    return "artifact-verification-policy:" + canonical_json_digest(material)


class ResolvedArtifactVerificationPolicy(BaseModel):
    """Complete historical interpretation, independent of today's defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    defaults: ArtifactVerificationDefaults
    overrides: ArtifactVerificationOverrides
    effective: ArtifactVerificationDefaults
    policy_id: StrictStr

    @model_validator(mode="before")
    @classmethod
    def retained_settings_are_complete(cls, value: object) -> object:
        if isinstance(value, Mapping):
            for name in ("defaults", "effective"):
                settings = value.get(name)
                if isinstance(settings, Mapping) and (
                    ArtifactVerificationDefaults.model_validate(settings).model_dump(
                        mode="json"
                    )
                    != settings
                ):
                    raise ValueError(
                        "retained artifact verification settings must be complete"
                    )
        return value

    @model_validator(mode="after")
    def retained_interpretation_matches(self) -> "ResolvedArtifactVerificationPolicy":
        expected = ArtifactVerificationDefaults.model_validate(
            {
                **self.defaults.model_dump(mode="json"),
                **self.overrides.explicit_settings(),
            }
        )
        material = self.model_dump(mode="json", exclude={"policy_id"})
        if self.effective != expected or self.policy_id != _policy_identity(material):
            raise ValueError("retained artifact verification policy binding is invalid")
        return self


def resolve_artifact_verification_policy(
    defaults: ArtifactVerificationDefaults,
    overrides: ArtifactVerificationOverrides,
) -> ResolvedArtifactVerificationPolicy:
    """Resolve one invocation without changing defaults or previously retained policy."""
    effective = ArtifactVerificationDefaults.model_validate(
        {
            **defaults.model_dump(mode="json"),
            **overrides.explicit_settings(),
        }
    )
    material = {
        "defaults": defaults.model_dump(mode="json"),
        "overrides": overrides.explicit_settings(),
        "effective": effective.model_dump(mode="json"),
    }
    return ResolvedArtifactVerificationPolicy(
        **material, policy_id=_policy_identity(material)
    )


def retained_artifact_verification_policy(
    value: object, *, defaults: ArtifactVerificationDefaults | None = None
) -> ResolvedArtifactVerificationPolicy:
    """Read complete retained provenance, optionally binding its recorded defaults.

    Documented reader: no product path calls it; tests and inspection read retained evidence through it.
    """
    policy = ResolvedArtifactVerificationPolicy.model_validate(value)
    if defaults is not None and policy.defaults != defaults:
        raise ValueError("retained artifact verification defaults do not match")
    return policy
