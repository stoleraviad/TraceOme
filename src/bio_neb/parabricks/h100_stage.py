"""H100 recipe: verified input pointers through a manifest published last."""

import hashlib
from pathlib import Path

from bio_neb.application.artifact_verification import use_produced_object
from bio_neb.content import ContentId, canonical_json, canonical_json_digest
from bio_neb.failure import Invariant, ProviderError
from bio_neb.journey_path import (
    Artifact,
    ArtifactPointer,
    Hit,
    PathFact,
    Prepared,
    Producer,
    Verified,
)
from bio_neb.parabricks import h100_recipe as contracts
from bio_neb.parabricks.execution_observations import (
    JobObservation,
    durable_worker_observations,
    observe_worker,
)
from bio_neb.parabricks.run_method import GERMLINE_COMMAND
from bio_neb.parabricks.runner import StageDef
from bio_neb.parabricks.worker_request import (
    PREPARED_ROLES,
    ExactWorkerObject,
    WorkerRequestIdentity,
)
from bio_neb.parabricks.worker_request_placement import place_execution_request
from bio_neb.providers.object_locations import object_address, object_location
from bio_neb.providers.ports import Receipt
from bio_neb.verified_artifacts import VerifiedObject


def _object(pointer):
    try:
        bucket, key = object_address(pointer.location)
    except ValueError as error:
        raise Invariant(
            "h100_request_invalid", "h100_input_location_invalid"
        ) from error
    return VerifiedObject(
        role=pointer.role,
        bucket=bucket,
        object_key=key,
        byte_count=pointer.bytes,
        sha256=ContentId.parse(pointer.content_id).digest,
    )


def compile_request(plan, state):
    """Read the accepted reference answer uniformly for Hit and Prepared."""
    try:
        if not isinstance(state.inputs, Verified) or not isinstance(
            state.reference, (Hit, Prepared)
        ):
            raise Invariant("h100_request_invalid", "h100_inputs_not_verified")
        targets = contracts.output_targets(plan)
        roles = ("read_1", "read_2", *PREPARED_ROLES)
        aliases = {"forward_read": "read_1", "reverse_read": "read_2"}
        selected = {}
        for pointer in (*state.inputs.pointers, *state.reference.pointers):
            role = aliases.get(pointer.role, pointer.role)
            if role in roles:
                if role in selected:
                    raise Invariant("h100_request_invalid", "h100_input_roles_invalid")
                selected[role] = pointer
        if set(selected) != set(roles):
            raise Invariant("h100_request_invalid", "h100_input_roles_invalid")
        paths = contracts._worker_paths()
        inputs = []
        for role in roles:
            item = _object(selected[role])
            if item.bucket != plan.storage.bucket:
                raise Invariant("h100_request_invalid", "h100_input_bucket_mismatch")
            inputs.append(
                ExactWorkerObject(
                    role=role,
                    object_key=item.object_key,
                    local_target=paths[role],
                    byte_count=item.byte_count,
                    sha256=item.sha256,
                )
            )
        outputs = {item.role: item for item in targets}
        package = state.bound.approved_input["proposal"]["execution_package"]
        argv = (
            *GERMLINE_COMMAND,
            "--ref",
            paths["genomic_fasta"],
            "--in-fq",
            paths["read_1"],
            paths["read_2"],
            "--out-bam",
            outputs["aligned_bam"].local_target,
            "--out-variants",
            outputs["VCF"].local_target,
            "--read-group-sm",
            package["workflow"]["sample_id"],
            "--num-gpus",
            str(plan.runtimes.h100.gpu_count),
        )
        base = dict(
            journey_id=plan.identity.journey_id,
            authority_id=plan.identity.authority_id,
            proposal_id=plan.identity.proposal_id,
            configuration_id=contracts.configuration_id(plan),
            prepared_reference_receipt_id=state.reference.receipt_id,
            execution_package_id=package["package_id"],
            inputs=tuple(inputs),
            outputs=targets,
            parabricks_argv=argv,
            source_proof_ids=(
                selected["read_1"].proof_ref,
                selected["read_2"].proof_ref,
                state.reference.receipt_id,
            ),
        )
        material = {
            **base,
            "inputs": [i.model_dump(mode="json") for i in inputs],
            "outputs": [i.model_dump(mode="json") for i in targets],
        }
        request_id = "h100-execution-request:" + canonical_json_digest(material)
        provisional = contracts.H100ExecutionRequest.model_construct(
            request_id=request_id,
            deterministic_job_name=plan.layout.h100_job_name,
            worker_request_identity=WorkerRequestIdentity(
                schema_version="bio-neb.parabricks-worker-request/7",
                operation="consume_prepared_reference",
                byte_count=1,
                sha256="0" * 64,
            ),
            **base,
        )
        compiled = contracts._compile_worker_request(
            plan.storage,
            plan.runtimes.h100,
            plan.layout.h100_manifest_object_key.rsplit("/", 1)[0] + "/",
            provisional,
        )
        return contracts.H100ExecutionRequest(
            **base,
            request_id=request_id,
            deterministic_job_name=provisional.deterministic_job_name,
            worker_request_identity=compiled.identity,
        )

    except (ValueError, KeyError) as error:
        raise Invariant("h100_request_invalid", "h100_request_invalid") from error


def _expect(receipt, size, digest):
    if (receipt.values.get("bytes"), receipt.values.get("sha256")) != (size, digest):
        raise ProviderError("semantic", "h100_object_identity_mismatch")


def preflight(ctx, store):
    compile_request(ctx.plan, ctx.state)
    return Receipt()


def submit_h100(ctx, jobs):
    request = compile_request(ctx.plan, ctx.state)
    try:
        compiled = contracts.compile_h100_worker_request(
            ctx.plan.storage,
            ctx.plan.runtimes.h100,
            ctx.plan.layout.h100_manifest_object_key.rsplit("/", 1)[0] + "/",
            request,
        )
    except ValueError as error:
        raise Invariant("h100_request_invalid", "h100_request_invalid") from error
    placed = place_execution_request(
        compiled,
        source_path=str(
            Path(ctx.plan.launch.work_directory)
            / ctx.plan.layout.h100_job_name
            / compiled.identity.sha256
            / "h100-request.json"
        ),
        container_path="/opt/bio-neb/request.json",
    )
    runtime = ctx.plan.runtimes.h100
    return jobs.submit(
        ctx.plan.layout.h100_job_name,
        {
            "target": ctx.plan.scope.model_dump(mode="json"),
            "provider_image_reference": runtime.provider_image_reference,
            "image_index_digest": runtime.provider_image_index_digest,
            "image_child_digest": runtime.worker_runtime_image.rsplit("@", 1)[1],
            "platform": runtime.platform,
            "preset": runtime.preset,
            "disk_size": runtime.disk_size,
            "disk_size_bytes": runtime.disk_size_bytes,
            "shared_memory_size": str(runtime.shared_memory_bytes),
            "shared_memory_bytes": runtime.shared_memory_bytes,
            "restart_policy": runtime.restart_policy,
            "normalized_arguments": list(placed.normalized_arguments),
            "injection_provider_values": list(placed.injection_provider_values),
            "injected_paths": list(placed.injected_paths),
            "environment_names": [
                "AWS_ENDPOINT_URL",
                "AWS_REGION",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
            ],
            "version_bound_secret_environment_names": ["AWS_SECRET_ACCESS_KEY"],
        },
    )


def h100_terminal(samples):
    sample = samples[-1]
    state = sample.values["state"]
    if state in {"FAILED", "CANCELLED", "ERROR"}:
        raise ProviderError("worker_failed", "h100_job_failed", sample.values)
    return sample if state == "COMPLETED" else None


H100_OBSERVATION = JobObservation()


def await_h100(ctx, jobs):
    return observe_worker(ctx, jobs, H100_OBSERVATION)


def _pointer(ctx, role, key, receipt):
    return ArtifactPointer(
        role=role,
        content_id=str(ContentId(receipt.values["sha256"])),
        location=object_location(ctx.plan.storage.bucket, key),
        bytes=receipt.values["bytes"],
        producer=Producer(stage="h100", authority_id=ctx.plan.identity.authority_id),
        grounding="verified",
        proof_ref=receipt.proof_ref,
    )


def verify_outputs(ctx, store):
    request = compile_request(ctx.plan, ctx.state)
    key = ctx.plan.layout.h100_output_keys[2]
    raw = store.read_small(ctx.plan.storage.bucket, key)
    try:
        observations = durable_worker_observations(
            ctx.receipts[-1], raw, ctx.plan.storage.bucket, key
        )
        observed = contracts.validate_h100_observation_closure(
            ctx.plan.scope,
            ctx.plan.storage,
            ctx.plan.runtimes.h100,
            request,
            contracts.H100SubmissionReceipt(
                provider_job_id=ctx.receipts[-1].provider_ids[0],
                deterministic_job_name=request.deterministic_job_name,
            ),
            observations,
            durable=True,
            container_path="/opt/bio-neb/request.json",
        )
        worker = observed.receipt
        outputs = tuple(
            VerifiedObject.model_validate(item) for item in worker["uploads"]
        )
        if tuple(item.role for item in outputs) != ("aligned_bam", "VCF"):
            raise ValueError("output roles changed")
    except (ValueError, KeyError) as error:
        raise ProviderError(
            "semantic", "h100_receipt_invalid", {"reason": str(error)}
        ) from None
    pointers = []
    for item, target in zip(outputs, ctx.plan.layout.h100_output_keys[:2], strict=True):
        if item.bucket != ctx.plan.storage.bucket or item.object_key != target:
            raise ProviderError("semantic", "h100_output_location_mismatch")
        checked = use_produced_object(
            store,
            item,
            proof_ref=raw.proof_ref,
            mode=ctx.plan.verification.effective.h100_outputs,
        )
        _expect(checked, item.byte_count, item.sha256)
        pointers.append(
            _pointer(ctx, item.role, target, checked).model_dump(mode="json")
        )
    receipt_pointer = _pointer(ctx, "worker_receipt", key, raw)
    closure = contracts.H100OutputClosure(
        outputs=outputs,
        worker_receipt=VerifiedObject(
            role="worker_receipt",
            bucket=ctx.plan.storage.bucket,
            object_key=key,
            byte_count=raw.values["bytes"],
            sha256=raw.values["sha256"],
        ),
        pre_manifest_inventory=tuple(sorted((*ctx.plan.layout.h100_output_keys,))),
    )
    manifest = contracts.compile_h100_manifest(
        ctx.plan.storage,
        ctx.plan.runtimes.h100,
        ctx.plan.verification.effective.h100_outputs,
        request,
        contracts.H100SubmissionReceipt(
            provider_job_id=ctx.receipts[-1].provider_ids[0],
            deterministic_job_name=request.deterministic_job_name,
        ),
        closure,
    )
    return Receipt(
        content=raw.content,
        values={
            "manifest": manifest.model_dump(mode="json"),
            "pointers": pointers,
            "worker_receipt": receipt_pointer.model_dump(mode="json"),
        },
    )


def publish_manifest(ctx, store):
    verified = ctx.receipts[-1]
    content = canonical_json(verified.values["manifest"]).encode()
    key = ctx.plan.layout.h100_manifest_object_key
    checked = store.put(ctx.plan.storage.bucket, key, content)
    _expect(checked, len(content), hashlib.sha256(content).hexdigest())
    producer = Producer(stage="h100", authority_id=ctx.plan.identity.authority_id)
    manifest = Artifact(
        content=content,
        role="execution_manifest",
        producer=producer,
        grounding="verified",
        proof_ref=checked.proof_ref,
    )
    reference_id = ctx.state.latest_event_ids[
        "ReferencePrepared"
        if isinstance(ctx.state.reference, Prepared)
        else "ReferenceResolved"
    ]
    rests_on = [
        ctx.state.latest_event_ids["ExecutionBound"],
        ctx.state.latest_event_ids["InputsVerified"],
        reference_id,
    ]
    rests_on.extend(ctx.state.stage_answer_ids.get("h100", ()))
    if "RuntimeSubstituted" in ctx.state.latest_event_ids:
        rests_on.append(ctx.state.latest_event_ids["RuntimeSubstituted"])
    if "gpu_class" in ctx.state.amendment_event_ids:
        rests_on.append(ctx.state.amendment_event_ids["gpu_class"])
    from bio_neb.parabricks.result import execution_result_context

    return Receipt(
        content=content,
        values={
            "context": execution_result_context(ctx.plan, ctx.state),
            "manifest_pointer": manifest.pointer().model_dump(mode="json"),
            "remote_manifest": _pointer(
                ctx, "execution_manifest_object", key, checked
            ).model_dump(mode="json"),
            "pass": ctx.state.pass_,
            "rests_on": list(dict.fromkeys(rests_on)),
        },
    )


def h100_artifacts(receipts):
    verified, published = receipts[-2:]
    documents = tuple(
        Artifact(
            content=content,
            role=p.role,
            producer=p.producer,
            grounding=p.grounding,
            proof_ref=p.proof_ref,
        )
        for content, p in (
            (
                verified.content,
                ArtifactPointer.model_validate(verified.values["worker_receipt"]),
            ),
            (
                published.content,
                ArtifactPointer.model_validate(published.values["manifest_pointer"]),
            ),
        )
    )
    context = Artifact(
        content=canonical_json(published.values["context"]).encode(),
        role="execution_result_context",
        producer=documents[1].producer,
        grounding="agent_composed",
        proof_ref=published.proof_ref,
    )
    return (*documents, context)


def h100_verified(receipts):
    published = receipts[-1].values
    artifacts = h100_artifacts(receipts)
    return PathFact.model_validate(
        {
            "event_type": "H100ArtifactsVerified",
            "kind": "verified_artifact",
            "grounding": "verified",
            "pass": published["pass"],
            "rests_on": published["rests_on"],
            "payload": {
                "state": "verified",
                "receipt_id": artifacts[1].pointer().content_id,
                "pointers": [
                    *receipts[-2].values["pointers"],
                    published["remote_manifest"],
                    *(a.pointer().model_dump(mode="json") for a in artifacts),
                ],
            },
        }
    )


H100 = StageDef(
    name="h100",
    steps=(preflight, submit_h100, await_h100, verify_outputs, publish_manifest),
    ports={
        "preflight": "store",
        "submit_h100": "jobs",
        "await_h100": "jobs",
        "verify_outputs": "store",
        "publish_manifest": "store",
    },
    closure=h100_terminal,
    path_fact=h100_verified,
    artifacts=h100_artifacts,
    runtime="h100",
)
