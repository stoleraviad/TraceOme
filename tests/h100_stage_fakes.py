"""Synthetic H100 worker observations; no worker or network executes.

Existing observation vectors provide the protocol markers. Output identity and
failure expectations live in the tests, independently of the production closure.
"""

import hashlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from bio_neb.content import canonical_json
from bio_neb.journey_path import JourneyPath, project_execution
from bio_neb.parabricks.h100_stage import compile_request
from bio_neb.providers.ports import Receipt
from tests.execution_observation_fakes import durable_receipt, worker_marker
from tests.h100_observation_fakes import h100_observations
from tests.reference_stage_fakes import CpuCli
from tests.tiny_bam import tiny_bam

OUTPUTS = {
    # A real BAM: every Investigation measures it with the Run's samtools.
    "aligned_bam": tiny_bam(),
    "VCF": b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
}


class H100Cli(CpuCli):
    """One persistent named H100 resource using the actual contact CLI protocol."""

    def __init__(self, *args, job_id="aijob-exact", **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = job_id

    def _job(self, state):
        result = super()._job(state)
        result["metadata"]["id"] = self.job_id
        if result["spec"]:
            result["spec"]["disk"]["size_bytes"] = str(
                self.config.runtime.disk_size_bytes
            )
            result["spec"]["shm_size_bytes"] = str(
                self.config.runtime.shared_memory_bytes
            )
        return result


def h100_observation_inputs(plan):
    return SimpleNamespace(
        nebius=plan.scope,
        object_storage=plan.storage,
        runtime=plan.runtimes.h100,
        request_container_path="/opt/bio-neb/request.json",
        outputs=tuple(
            SimpleNamespace(role=role, object_key=key)
            for role, key in zip(
                ("aligned_bam", "VCF", "worker_receipt"),
                plan.layout.h100_output_keys,
                strict=True,
            )
        ),
    )


def h100_provider(
    provider, plan, approved, *, amend=None, job_id="aijob-exact", output_payloads=None
):
    outputs = {**OUTPUTS, **(output_payloads or {})}
    original = provider.for_step
    seeded = False

    def for_step(stage, step):
        nonlocal seeded
        if stage == "h100" and not seeded:
            seeded = True
            state = project_execution(
                JourneyPath.open(Path(approved.journey_file)).read()
            )
            request = compile_request(plan, state)
            observations = h100_observations(
                h100_observation_inputs(plan),
                request,
                output_payloads=outputs,
                job_id=job_id,
            )
            names = {
                "worker_startup": "BIO_NEB_PARABRICKS_WORKER_START",
                "worker_request": "BIO_NEB_WORKER_REQUEST_ACCEPTED",
                "worker_tools": "BIO_NEB_WORKER_TOOLS",
                "worker_input_verified": "BIO_NEB_WORKER_INPUT_VERIFIED",
                "worker_reference_support": "BIO_NEB_WORKER_REFERENCE_SUPPORT",
                "worker_prepared_reference_verified": "BIO_NEB_WORKER_PREPARED_REFERENCE_VERIFIED",
                "tool_started": "BIO_NEB_WORKER_TOOL_START",
                "tool_exited": "BIO_NEB_WORKER_TOOL_EXIT",
                "upload_started": "BIO_NEB_WORKER_UPLOAD_START",
                "upload_completed": "BIO_NEB_WORKER_UPLOAD_COMPLETE",
                "worker_receipt": "BIO_NEB_WORKER_RECEIPT",
            }
            lines = []
            for obs in observations:
                if obs.kind not in names:
                    continue
                payload = obs.model_dump(
                    mode="json", exclude={"kind"}, exclude_none=True
                )
                if obs.kind == "worker_tools":
                    payload = {
                        "tools": {
                            i.name: i.model_dump(
                                mode="json", exclude={"name"}, exclude_none=True
                            )
                            for i in obs.tools
                        }
                    }
                lines.append(worker_marker(names[obs.kind], payload, len(lines) + 1))
            by_kind = {o.kind: o for o in observations}
            provider.scripts["h100", "preflight"] = deque(
                Receipt(values={"bytes": i.byte_count, "sha256": i.sha256})
                for i in request.inputs
            )
            provider.scripts["h100", "submit_h100"] = deque(
                [
                    Receipt(
                        values={
                            "create": by_kind["provider_create"].model_dump(
                                mode="json"
                            ),
                            "envelope": by_kind["fetched_envelope"].model_dump(
                                mode="json"
                            ),
                        },
                        provider_ids=(job_id,),
                        resource_count=1,
                    )
                ]
            )
            provider.scripts["h100", "await_h100"] = deque(
                [
                    Receipt(
                        values={
                            "state": "COMPLETED",
                            "log": "\n".join(lines),
                            "window": 1000,
                            "details": by_kind["provider_state"].model_dump(
                                mode="json"
                            ),
                        },
                        provider_ids=(job_id,),
                    )
                ]
            )
            for role, key in zip(
                ("aligned_bam", "VCF"), plan.layout.h100_output_keys, strict=False
            ):
                provider.objects[plan.storage.bucket, key] = outputs[role]
            provider.objects[plan.storage.bucket, plan.layout.h100_output_keys[2]] = (
                canonical_json(durable_receipt(observations)).encode()
            )
            if amend:
                amend(provider, plan)
        return original(stage, step)

    provider.for_step = for_step
    return provider


class OneGet:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def download(
        self,
        object_key,
        destination,
        *,
        expected_byte_count,
        expected_sha256,
        observe_part=None,
    ):
        from tests.parabricks_operation_fakes import _credential_receipt

        self.calls.append(object_key)
        if (expected_byte_count, expected_sha256) != (
            len(self.body),
            hashlib.sha256(self.body).hexdigest(),
        ):
            raise AssertionError("descriptor changed")
        destination.write(self.body)
        return SimpleNamespace(
            byte_count=len(self.body),
            sha256=hashlib.sha256(self.body).hexdigest(),
            credential_receipt=_credential_receipt(),
            part_bytes=32 * 1024 * 1024,
            part_count=1,
        )
