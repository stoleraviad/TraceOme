"""Shared fakes for the run-health tool tests (lane H)."""

import hashlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

from bio_neb.agentic_journey.runtime_config import InvestigationTools, ObservationTool
from bio_neb.content import canonical_json
from bio_neb.investigation.bam_materialization import (
    MaterializeCurrentBamCommand,
    materialize_current_bam,
)
from bio_neb.investigation.observation_tools import (
    RECEIPT_NAME,
    ObservationToolsReceipt,
    ProbedTool,
    probe_observation_tools,
)
from bio_neb.investigation.vcf_materialization import (
    MaterializeCurrentVcfCommand,
    materialize_current_vcf,
)
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt


def kept_probe(directory, *, samtools="b" * 64, bcftools="c" * 64):
    """A Run's tool probe as main keeps it: a receipt that the probe reads back."""
    tools = tuple(
        ProbedTool(
            name=name,
            executable=f"/run/.venv/bin/bio-neb-{name}",
            executable_sha256=digest,
            package="pysam",
            package_version="0.24.1",
            tool_version="1.24",
            htslib_version="1.24",
        )
        for name, digest in (("bcftools", bcftools), ("samtools", samtools))
    )
    receipt = ObservationToolsReceipt(tools=tools)
    directory.mkdir(parents=True)
    (directory / RECEIPT_NAME).write_text(
        canonical_json(receipt.model_dump(mode="json"))
    )
    return probe_observation_tools(None, directory)


def pysam_probed_runner(directory):
    """This environment's samtools, as main's tool probe names it."""
    from tools.run_health import ProbedProcessRunner

    pysam_tool = ObservationTool(
        package="pysam", package_version=importlib.metadata.version("pysam")
    )
    return ProbedProcessRunner(
        probe_observation_tools(
            InvestigationTools(bcftools=pysam_tool, samtools=pysam_tool), directory
        )
    )


def credential_receipt():
    """The credential reuse a Run's exact object read reports in its selected context."""
    return ObjectStorageCredentialReuseReceipt(
        service_account_id="service-account-exact",
        access_key_resource_id="access-key-resource-exact",
        access_key_id="aws-access-key-exact",
        exact_topology_match_count=1,
        access_key_state="ACTIVE",
        secret_state="ACTIVE",
        secret_delivery_mode="mystery_box",
        host_secret_resolved=True,
        worker_secret_version_bound=True,
    )


class ObjectGet:
    """Object-read port fake that answers one exact GET with `body`, reporting the
    identity it streamed."""

    def __init__(self, body):
        self.body = body

    def __call__(self, packet):
        return self

    def download(
        self,
        object_key,
        destination,
        *,
        expected_byte_count,
        expected_sha256,
        observe_part=None,
    ):
        destination.write(self.body)
        return SimpleNamespace(
            byte_count=len(self.body),
            sha256=hashlib.sha256(self.body).hexdigest(),
            credential_receipt=credential_receipt(),
            part_bytes=32 * 1024 * 1024,
            part_count=1,
        )


def local_run_inputs(journey_file, *, vcf, bam, tools=kept_probe):
    """The Run's VCF and BAM downloaded beside its Journey, as its Investigation does
    (#363): `<run>/investigation/{vcf,bam}` with main's receipts, and its kept tool
    probe in `<run>/investigation/tools`, written by `tools` (a fake receipt, or
    `pysam_probed_runner` for this environment's real tools). Returns the
    investigation directory."""
    root = Path(journey_file).parent / "investigation"
    root.mkdir(mode=0o700)
    materialize_current_vcf(
        MaterializeCurrentVcfCommand(
            execution_journey_file=str(journey_file),
            output_directory=str(root / "vcf"),
        ),
        object_read_factory=ObjectGet(vcf),
    )
    materialize_current_bam(
        MaterializeCurrentBamCommand(
            execution_journey_file=str(journey_file),
            output_directory=str(root / "bam"),
        ),
        object_read_factory=ObjectGet(bam),
    )
    tools(root / "tools")
    return root
