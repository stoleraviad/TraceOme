"""Secret-free provider connection and storage targets."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, StrictInt, StrictStr, field_validator

from bio_neb.contracts import FrozenContract


def _canonical_text(value: str, field_name: str) -> str:
    if not value or value.strip() != value or any(c in value for c in "\r\n\0"):
        raise ValueError(f"{field_name} must be one canonical line")
    return value


def _absolute_posix_path(value: str, field_name: str) -> str:
    _canonical_text(value, field_name)
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or value == "/"
        or str(path) != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be an absolute canonical POSIX path")
    return value


class NebiusCliTarget(FrozenContract):
    controller_runtime: Literal["native_ubuntu"]
    executable: StrictStr
    resolved_executable: StrictStr
    cli_version: Literal["0.12.263"]
    executable_byte_count: StrictInt = Field(gt=0)
    executable_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    profile: StrictStr
    tenant_id: StrictStr
    project_id: StrictStr
    subnet_id: StrictStr = Field(pattern=r"^vpcsubnet-[a-z0-9]+$")
    registry_id: StrictStr = Field(pattern=r"^registry-[a-z0-9]+$")

    @field_validator("profile", "tenant_id", "project_id")
    @classmethod
    def canonical_values(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)

    @field_validator("executable", "resolved_executable")
    @classmethod
    def absolute_executable(cls, value: str, info: Any) -> str:
        return _absolute_posix_path(value, info.field_name)


class RunObjectStorageTarget(FrozenContract):
    endpoint_url: StrictStr
    region: StrictStr
    bucket: StrictStr = Field(pattern=r"^[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])$")
    mount_path: StrictStr

    @field_validator("endpoint_url", "region", "bucket")
    @classmethod
    def canonical_values(cls, value: str, info: Any) -> str:
        return _canonical_text(value, info.field_name)

    @field_validator("endpoint_url")
    @classmethod
    def https_endpoint(cls, value: str) -> str:
        selected = urlsplit(value)
        try:
            _ = selected.port
        except ValueError as error:
            raise ValueError("endpoint_url must be one exact HTTPS origin") from error
        if (
            selected.scheme != "https"
            or not selected.hostname
            or selected.username is not None
            or selected.password is not None
            or selected.path
            or selected.query
            or selected.fragment
            or selected.geturl() != value
        ):
            raise ValueError("endpoint_url must be one exact HTTPS origin")
        return value
