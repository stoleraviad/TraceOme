"""Secret-free value types naming one selected Nebius execution context."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.content import canonical_json_digest
from bio_neb.contracts import FrozenContract


class NebiusProfileCandidate(FrozenContract):
    profile: StrictStr


class NebiusTenantCandidate(FrozenContract):
    profile: StrictStr
    tenant_id: StrictStr
    name: StrictStr | None = None


class NebiusProjectCandidate(FrozenContract):
    profile: StrictStr
    tenant_id: StrictStr
    project_id: StrictStr
    name: StrictStr | None = None


class NebiusObjectStorageTopologyCandidate(FrozenContract):
    candidate_id: StrictStr = Field(
        pattern=r"^nebius-object-storage-topology:[0-9a-f]{64}$"
    )
    profile: StrictStr
    tenant_id: StrictStr
    project_id: StrictStr
    service_account_id: StrictStr
    access_key_resource_id: StrictStr
    access_key_id: StrictStr
    group_id: StrictStr
    group_membership_id: StrictStr
    access_permit_id: StrictStr
    access_key_state: Literal["ACTIVE"] = "ACTIVE"
    secret_delivery_mode: Literal["mystery_box"] = "mystery_box"

    @model_validator(mode="after")
    def exact_identity(self) -> "NebiusObjectStorageTopologyCandidate":
        values = self.model_dump(mode="json", exclude={"candidate_id"})
        expected = "nebius-object-storage-topology:" + canonical_json_digest(values)
        if self.candidate_id != expected:
            raise ValueError("candidate_id does not bind the topology identity")
        return self

    @classmethod
    def bind(cls, **values: str) -> "NebiusObjectStorageTopologyCandidate":
        body = {
            **values,
            "access_key_state": "ACTIVE",
            "secret_delivery_mode": "mystery_box",
        }
        candidate_id = "nebius-object-storage-topology:" + canonical_json_digest(body)
        return cls(candidate_id=candidate_id, **body)


class NebiusExecutionContextSelection(FrozenContract):
    """Selected native-Ubuntu Nebius CLI identity and its one Object Storage topology."""

    schema_version: Literal["bio-neb.nebius-execution-context-selection/2"] = (
        "bio-neb.nebius-execution-context-selection/2"
    )
    controller_runtime: Literal["native_ubuntu"]
    nebius_executable: StrictStr
    resolved_nebius_executable: StrictStr
    nebius_cli_version: Literal["0.12.263"]
    nebius_executable_byte_count: StrictInt = Field(gt=0)
    nebius_executable_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    topology: NebiusObjectStorageTopologyCandidate
    exact_topology_match_count: Literal[1] = 1
    bootstrap_mutation_count: Literal[0] = 0


__all__ = [
    "NebiusExecutionContextSelection",
    "NebiusObjectStorageTopologyCandidate",
    "NebiusProfileCandidate",
    "NebiusProjectCandidate",
    "NebiusTenantCandidate",
]
