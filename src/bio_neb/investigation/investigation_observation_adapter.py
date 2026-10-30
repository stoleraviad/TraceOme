"""Local receipt mechanics for approved Investigation observation."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from bio_neb.application.local_file_observation import (
    LocalFileObservationError,
    ObservationFiles,
    stable_file_observation,
)
from bio_neb.investigation.investigation_observation_commands import (
    InvestigationObservationApplicationRefused,
    MaterializationObservationEvidence,
    RuntimeProbeObservationEvidence,
)
from bio_neb.investigation.observation_tools import ObservationToolsReceipt
from bio_neb.investigation.vcf_materialization import (
    VcfMaterializationReceipt,
)
from bio_neb.investigation.vcf_observation import (
    AttestedPlainVcf,
    BcftoolsRuntimeBinding,
)
from bio_neb.journey import canonical_json


def _stable_receipt_payload(
    path: str, *, require_read_only: bool, files: ObservationFiles | None
) -> tuple[bytes, Any]:
    try:
        payload, observation = (
            stable_file_observation(path, retain_bytes=True)
            if files is None
            else files.read(path, retain_bytes=True)
        )
    except LocalFileObservationError as error:
        code = (
            "investigation_observation_receipt_unsafe"
            if error.code == "unsafe"
            else "investigation_observation_receipt_changed"
        )
        raise InvestigationObservationApplicationRefused(code, path) from error
    if payload is None or not observation.exists:
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_receipt_unavailable", path
        )
    if require_read_only and observation.is_writable:
        raise InvestigationObservationApplicationRefused(
            "investigation_observation_receipt_not_immutable", path
        )
    return payload, observation


class DirectInvestigationObservationEvidence:
    """Parse stable local receipts without following links."""

    def __init__(self, *, files: ObservationFiles | None = None) -> None:
        self.files = files

    def materialization(
        self, *, receipt_file: str, vcf_file: str
    ) -> MaterializationObservationEvidence:
        payload, observation = _stable_receipt_payload(
            receipt_file, require_read_only=True, files=self.files
        )
        try:
            receipt = VcfMaterializationReceipt.model_validate_json(
                payload, strict=True
            )
        except ValidationError as error:
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_materialization_receipt_invalid",
                "materialization_receipt_file",
            ) from error
        if payload != canonical_json(receipt.model_dump(mode="json")).encode("utf-8"):
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_materialization_receipt_invalid",
                "materialization_receipt_file",
            )
        if receipt.materialized_vcf.path != vcf_file:
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_vcf_identity_changed", "vcf_file"
            )
        return MaterializationObservationEvidence(
            receipt_id=receipt.receipt_id,
            descriptor=receipt.descriptor,
            attested_vcf=AttestedPlainVcf(
                local_path=receipt.materialized_vcf.path,
                byte_count=receipt.materialized_vcf.byte_count,
                sha256=receipt.materialized_vcf.sha256,
                acquisition_receipt_path=receipt_file,
                acquisition_receipt_sha256=observation.sha256,
            ),
            receipt_file=observation,
        )

    def runtime(
        self, *, receipt_file: str, executable: str
    ) -> RuntimeProbeObservationEvidence:
        """Bind the bcftools the Run's own tool probe named (#107)."""
        payload, observation = _stable_receipt_payload(
            receipt_file, require_read_only=False, files=self.files
        )
        try:
            receipt = ObservationToolsReceipt.model_validate_json(payload)
        except ValidationError as error:
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_probe_receipt_invalid",
                "bcftools_probe_receipt_file",
            ) from error
        tool = receipt.tool("bcftools")
        if tool.executable != executable:
            raise InvestigationObservationApplicationRefused(
                "investigation_observation_probe_receipt_invalid",
                "bcftools_executable",
            )
        runtime = BcftoolsRuntimeBinding(
            executable_path=executable,
            executable_sha256=tool.executable_sha256,
            bcftools_version=tool.tool_version,
            htslib_version=tool.htslib_version,
            probe_receipt_path=receipt_file,
            probe_receipt_sha256=observation.sha256,
        )
        return RuntimeProbeObservationEvidence(
            runtime=runtime, receipt_file=observation
        )


__all__ = ["DirectInvestigationObservationEvidence"]
