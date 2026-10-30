"""Local replica metadata and the verification checks a selected policy measures."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from bio_neb.application.verification_policy import (
    ResolvedArtifactVerificationPolicy,
)


class ReplicaMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    device: StrictInt
    inode: StrictInt
    mode: StrictInt
    byte_count: StrictInt = Field(gt=0)
    modified_at_ns: StrictInt
    changed_at_ns: StrictInt
    owner: StrictInt
    group: StrictInt
    links: Literal[1]
    directory_device: StrictInt
    directory_inode: StrictInt
    directory_mode: StrictInt
    directory_owner: StrictInt


class ArtifactVerificationCheck(BaseModel):
    """Measured local verification, retained alongside the bytes actually consumed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    policy: ResolvedArtifactVerificationPolicy
    proof_id: StrictStr | None = Field(
        default=None, pattern=r"^local-artifact-proof:[0-9a-f]{64}$"
    )
    path: StrictStr
    reason: Literal[
        "controlled_proof_reuse",
        "sensitive_checkpoint",
        "policy_full_hash",
        "missing_proof_admission",
        "destination_admission",
    ]
    bytes_read: StrictInt = Field(ge=0)
    bytes_hashed: StrictInt = Field(ge=0)
    elapsed_seconds: StrictFloat = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def selected_check(self) -> ArtifactVerificationCheck:
        if self.reason == "controlled_proof_reuse":
            if self.bytes_hashed or self.proof_id is None:
                raise ValueError("proof reuse changed selected verification policy")
        elif self.bytes_hashed != self.bytes_read:
            raise ValueError("designated verification did not hash all consumed bytes")
        return self
