"""Shared validation postures for domain-owned typed contracts."""

from pydantic import BaseModel, ConfigDict


class FrozenContract(BaseModel):
    """Reject extra fields and freeze assignments, with Pydantic coercion."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StrictFrozenContract(BaseModel):
    """Reject extra fields, freeze assignments, and require strict validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
