"""Authoritative reference source identities admitted without source contact."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from bio_neb.journey import canonical_json_digest
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
)

GZIP_REFERENCE_SEQUENCES = "decompress_gzip_reference_sequences"
SourceDerivationRule = Literal["decompress_gzip_reference_sequences", "identity"]
"""How the prepared FASTA derives from the admitted source: a gzip source is
decompressed and keeps the Run's reference sequences only (its excluded sequences are
dropped, owner #464); a plain one is taken as it is."""


class ReferenceSourceAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admission_id: StrictStr = Field(
        pattern=r"^reference-source-admission:[0-9a-f]{64}$"
    )
    reference_asset_id: StrictStr
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    authority_locator: StrictStr
    source_mode: Literal["public_url", "local_file", "existing_object"]
    source_locator: StrictStr
    source_representation: Literal["gzip_fasta", "plain_fasta"]
    expected_size_bytes: StrictInt = Field(gt=0)
    expected_checksum_algorithm: Literal["md5", "sha256", "sha512"]
    expected_checksum: StrictStr
    checksum_authority_locator: StrictStr
    derivation_rule: SourceDerivationRule

    @model_validator(mode="after")
    def exact_identity(self) -> "ReferenceSourceAdmission":
        for name in (
            "reference_asset_id",
            "authority_locator",
            "source_locator",
            "checksum_authority_locator",
        ):
            value = getattr(self, name)
            if (
                not value
                or value.strip() != value
                or any(char in value for char in "\r\n\0")
            ):
                raise ValueError(f"{name} must be one canonical line")
        expected_length = {"md5": 32, "sha256": 64, "sha512": 128}[
            self.expected_checksum_algorithm
        ]
        if len(self.expected_checksum) != expected_length or any(
            character not in "0123456789abcdef" for character in self.expected_checksum
        ):
            raise ValueError("expected checksum is not canonical")
        if self.source_representation == "gzip_fasta":
            if self.derivation_rule != GZIP_REFERENCE_SEQUENCES:
                raise ValueError("gzip FASTA source requires gzip decompression")
        elif self.derivation_rule != "identity":
            raise ValueError("plain FASTA source requires identity derivation")
        if self.source_mode == "public_url":
            selected = urlsplit(self.source_locator)
            if (
                selected.scheme != "https"
                or not selected.hostname
                or selected.username is not None
                or selected.password is not None
                or selected.fragment
                or selected.geturl() != self.source_locator
            ):
                raise ValueError("public reference source must be one exact HTTPS URL")
        elif self.source_mode == "local_file":
            if (
                not os.path.isabs(self.source_locator)
                or os.path.normpath(self.source_locator) != self.source_locator
            ):
                raise ValueError(
                    "local reference source must be one canonical absolute path"
                )
        else:
            selected = urlsplit(self.source_locator)
            if (
                selected.scheme != "object"
                or not selected.netloc
                or not selected.path.startswith("/")
                or selected.path == "/"
                or selected.username is not None
                or selected.password is not None
                or selected.query
                or selected.fragment
                or selected.geturl() != self.source_locator
            ):
                raise ValueError(
                    "existing reference source must use object://<scope>/<key>"
                )
        expected = reference_source_admission_id(
            self.model_dump(mode="json", exclude={"admission_id"})
        )
        if self.admission_id != expected:
            raise ValueError("source admission omitted consequence-bearing bytes")
        return self


def reference_source_admission_id(value: object) -> str:
    return "reference-source-admission:" + canonical_json_digest(value)


class ReferenceSourceAdmitted(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_type: Literal["ReferenceSourceAdmitted"] = "ReferenceSourceAdmitted"
    admission: ReferenceSourceAdmission
    reference_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    selection_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def validate_reference_source_fact(
    event: Mapping[str, Any],
) -> ReferenceSourceAdmitted | None:
    if event.get("event_type") != "ReferenceSourceAdmitted":
        return None
    return ReferenceSourceAdmitted.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


__all__ = [
    "ReferenceSourceAdmission",
    "ReferenceSourceAdmitted",
    "reference_source_admission_id",
    "validate_reference_source_fact",
]
