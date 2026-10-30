"""Reference admission and CPU preparation recipes for the shared stage engine.

The old typed manifest and worker validators remain the recipe contracts during
migration. No legacy adapter or Journey runner is invoked here.
"""

import hashlib
from pathlib import Path

from pydantic import ValidationError

from bio_neb.application.artifact_verification import use_produced_object
from bio_neb.content import ContentId, canonical_json, canonical_json_digest
from bio_neb.failure import Invariant, ProviderError
from bio_neb.journey_path import (
    Artifact,
    ArtifactPointer,
    PathFact,
    Producer,
    ReferenceAcquisition,
    Unresolved,
)
from bio_neb.parabricks import reference_recipe as contracts
from bio_neb.parabricks.bwa_index import FormatInvalid, read_fai, sequence_dictionary
from bio_neb.parabricks.execution_observations import (
    JobObservation,
    durable_worker_observations,
    observe_worker,
    source_mismatch,
)
from bio_neb.parabricks.runner import StageDef
from bio_neb.parabricks.worker_exchange import PREPARED_ROLES
from bio_neb.parabricks.worker_request_placement import place_execution_request
from bio_neb.prebuilt_index.rename import ContigRename, renamed_byte_count
from bio_neb.providers.ports import Receipt, TransferSource
from bio_neb.verified_artifacts import ArtifactIdentity


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _expect(receipt, size, digest):
    if (receipt.values.get("bytes"), receipt.values.get("sha256")) != (size, digest):
        raise ProviderError(
            "semantic",
            "reference_identity_mismatch",
            {
                "expected_bytes": size,
                "expected_sha256": digest,
                "observed_bytes": receipt.values.get("bytes"),
                "observed_sha256": receipt.values.get("sha256"),
            },
        )


def _pointer(ctx, stage, role, bucket, key, receipt):
    return ArtifactPointer(
        role=role,
        content_id=str(ContentId(receipt.values["sha256"])),
        location=f"s3://{bucket}/{key}",
        bytes=receipt.values["bytes"],
        producer=Producer(stage=stage, authority_id=ctx.plan.identity.authority_id),
        grounding="verified",
        proof_ref=receipt.proof_ref,
    ).model_dump(mode="json")


def _dependencies(ctx, *, prepared=False):
    latest = ctx.state.latest_event_ids
    names = ["InputsVerified"]
    if prepared:
        names.append(
            "ExecutionBound"
            if isinstance(ctx.state.reference, Unresolved)
            else "ReferenceResolved"
        )
    if "CpuPreparationReviewAnswered" in latest:
        names.append("CpuPreparationReviewAnswered")
    ids = [latest[name] for name in names]
    ids.extend(
        ctx.state.stage_answer_ids.get(
            "cpu_preparation" if prepared else "admit_candidate", ()
        )
    )
    return list(dict.fromkeys(ids))


def _miss(reason, evidence=None, *, state="miss"):
    return Receipt(
        values={"miss": reason, "reference_state": state},
        content=canonical_json({"reason": reason, "evidence": evidence or {}}).encode(),
    )


_SOURCE_REFUSALS = frozenset(
    {"source_http_unavailable", "source_unavailable", "source_or_transport_unavailable"}
)
"""A Galaxy file refused at its URL: the index is not there (the stage's
`candidate_absent`); the bucket's refusals stay the stage's failure."""
_NAMED_FORMATS = {"genomic_fasta": "fasta", "bwa_ann": "ann"}
"""The streamed files that name each contig once, besides the `.fai`."""


def _source_miss(error):
    """A changed Galaxy file is drift and an unreachable one a miss, each opening the
    CPU preparation Review (owner I7, #464); any other fault is the stage's."""
    if error.failure_class == "semantic" and error.detail in {
        "source_identity_mismatch",
        "reference_identity_mismatch",
    }:
        return _miss(error.detail, error.evidence, state="drifted")
    if error.failure_class == "rejected" and error.detail in _SOURCE_REFUSALS:
        return _miss("candidate_absent", {"detail": error.detail, **error.evidence})
    raise error


def _stream(ctx, store, role, rename=None):
    """One bound Galaxy file streamed from its URL into its candidate key, checked
    against its bound size; the placed object and the source's measured identity."""
    plan = ctx.plan
    file = next(f for f in plan.reference.prebuilt_index.files if f.role == role)
    key = plan.layout.candidate_object_keys[PREPARED_ROLES.index(role)]
    receipt = store.put(
        plan.storage.bucket,
        key,
        TransferSource("url", file.url, file.byte_count, None, transform=rename),
    )
    # A rewritten transfer proves its original source separately (ST6): its
    # placed identity is not the source's.
    measured = receipt.values.get("source") if rename is not None else receipt.values
    try:
        # The one shared identity contract (ADR 0029) validates the source proof.
        identity = ArtifactIdentity(
            byte_count=measured["bytes"], sha256=measured["sha256"]
        )
    except (KeyError, TypeError, ValidationError) as error:
        raise ProviderError(
            "semantic", "source_proof_invalid", {"url": file.url}
        ) from error
    source = {"bytes": identity.byte_count, "sha256": identity.sha256}
    if source["bytes"] != file.byte_count:
        raise ProviderError(
            "semantic",
            "source_identity_mismatch",
            {"url": file.url, "expected_bytes": file.byte_count},
        )
    placed = {
        "role": role,
        "bucket": plan.storage.bucket,
        "object_key": key,
        "byte_count": receipt.values["bytes"],
        "sha256": receipt.values["sha256"],
        "proof_ref": receipt.proof_ref,
    }
    origin = {
        "role": role,
        "url": file.url,
        "byte_count": source["bytes"],
        "sha256": source["sha256"],
    }
    return placed, origin


def inspect_candidate(ctx, store):
    """The Run's one candidate: the Execution Review's bound prebuilt index (owner,
    2026-10-29, #464). Without one the Run never admits; it prepares on CPU.

    The model's choice is trusted (owner I7: no fit, no download). The bound `.fai`
    streams first, its names made NCBI's and its offsets shifted
    (`prebuilt_index.rename`), and is read back for the `.dict`."""
    if ctx.plan.reference.prebuilt_index is None:
        return _miss("candidate_absent")
    names = tuple(ctx.plan.reference.prebuilt_index.contigs.ncbi_names().items())
    try:
        placed, origin = _stream(
            ctx, store, "fasta_fai", ContigRename("fai", names) if names else None
        )
        read = store.read_small(placed["bucket"], placed["object_key"])
        _expect(read, placed["byte_count"], placed["sha256"])
    except ProviderError as error:
        return _source_miss(error)
    try:
        contigs = read_fai(read.content)
    except FormatInvalid as error:
        return _miss(
            "candidate_unreadable", {"detail": error.reason, "file": "fasta_fai"}
        )
    if len({name for name, _ in contigs}) != len(contigs):
        return _miss("candidate_incompatible", {"detail": "contig_names_not_unique"})
    return Receipt(
        values={
            "names": [list(pair) for pair in names],
            "contigs": [list(contig) for contig in contigs],
            "placed": [placed],
            "origin": [origin],
        }
    )


def place_candidate(ctx, store):
    """Stream the other six bound files and write the `.dict` from the renamed `.fai`:
    the FASTA's headers and the `.ann`'s names rewritten in flight, the `.amb`,
    `.bwt`, `.pac` and `.sa` as published (owner I7, #464)."""
    inspected = ctx.receipts[-1]
    if "miss" in inspected.values:
        return inspected
    names = tuple(tuple(pair) for pair in inspected.values["names"])
    files = {f.role: f for f in ctx.plan.reference.prebuilt_index.files}
    placed = {"fasta_fai": inspected.values["placed"][0]}
    origins = list(inspected.values["origin"])
    try:
        for role in (
            "genomic_fasta",
            "bwa_amb",
            "bwa_ann",
            "bwa_bwt",
            "bwa_pac",
            "bwa_sa",
        ):
            rename = (
                ContigRename(
                    _NAMED_FORMATS[role],
                    names,
                    renamed_byte_count(files[role].byte_count, names),
                )
                if names and role in _NAMED_FORMATS
                else None
            )
            placed[role], origin = _stream(ctx, store, role, rename)
            origins.append(origin)
        content = sequence_dictionary(
            (name, length) for name, length in inspected.values["contigs"]
        )
        key = ctx.plan.layout.candidate_object_keys[
            PREPARED_ROLES.index("reference_dict")
        ]
        receipt = store.put(ctx.plan.storage.bucket, key, content)
        _expect(receipt, len(content), _digest(content))
    except ProviderError as error:
        return _source_miss(error)
    placed["reference_dict"] = {
        "role": "reference_dict",
        "bucket": ctx.plan.storage.bucket,
        "object_key": key,
        "byte_count": len(content),
        "sha256": _digest(content),
        "proof_ref": receipt.proof_ref,
    }
    bound = ctx.plan.reference.prebuilt_index
    origin = {
        "source": "galaxy",
        "catalogue": bound.candidate.catalogue,
        "candidate_id": bound.candidate.candidate_id,
        "display_name": bound.candidate.display_name,
        "stated_bwa_version": bound.stated_bwa_version,
        "files": origins,
        "renamed": [list(pair) for pair in names],
    }
    return Receipt(
        values={
            "placed": [placed[role] for role in PREPARED_ROLES],
            "origin": origin,
            "logical_bundle_id": "galaxy-prebuilt:" + canonical_json_digest(origin),
        }
    )


def candidate_metadata(ctx, store):
    try:
        return _candidate_metadata(ctx, store)
    except ProviderError as error:
        if error.failure_class != "semantic" or error.detail not in {
            "reference_identity_mismatch",
            "artifact_identity_mismatch",
        }:
            raise
        result = _miss(error.detail, error.evidence, state="drifted")
        return result.model_copy(
            update={
                "values": {
                    **result.values,
                    "pass": ctx.state.pass_,
                    "rests_on": _dependencies(ctx),
                }
            }
        )


def _candidate_metadata(ctx, store):
    result = ctx.receipts[-1]
    if "miss" in result.values:
        return result.model_copy(
            update={
                "values": {
                    **result.values,
                    "pass": ctx.state.pass_,
                    "rests_on": _dependencies(ctx),
                }
            }
        )
    pointers = []
    for item in result.values["placed"]:
        checked = use_produced_object(
            store,
            contracts.VerifiedObject.model_validate(
                {k: v for k, v in item.items() if k != "proof_ref"}
            ),
            proof_ref=item["proof_ref"],
            mode=ctx.plan.verification.effective.prepared_candidate,
        )
        _expect(checked, item["byte_count"], item["sha256"])
        pointers.append(
            _pointer(
                ctx,
                "admit_candidate",
                item["role"],
                item["bucket"],
                item["object_key"],
                checked,
            )
        )
    # Application-owned admission describes the fresh placement and its original provenance.
    document = {
        "execution_id": ctx.plan.identity.execution_id,
        "logical_bundle_id": result.values["logical_bundle_id"],
        "origin": result.values["origin"],
        "objects": result.values["placed"],
        "pointers": pointers,
    }
    return _publish(
        ctx,
        store,
        "admit_candidate",
        "prepared_reference_admission",
        ctx.plan.layout.candidate_manifest_object_key,
        document,
        pointers,
    )


def _publish(ctx, store, stage, role, key, document, pointers):
    content = canonical_json(document).encode()
    if stage == "admit_candidate":
        key = key.removesuffix(".json") + f"/{_digest(content)}.json"
    checked = store.put(ctx.plan.storage.bucket, key, content)
    _expect(checked, len(content), _digest(content))
    return Receipt(
        content=content,
        values={
            "pointers": pointers,
            "manifest": _pointer(
                ctx, stage, role, ctx.plan.storage.bucket, key, checked
            ),
            "receipt_id": str(ContentId.of(content)),
            "pass": ctx.state.pass_,
            "rests_on": _dependencies(ctx, prepared=stage == "cpu_preparation"),
        },
    )


def reference_artifacts(receipts):
    receipt = receipts[-1]
    if receipt.content is None:
        return ()
    if "miss" in receipt.values:
        return (
            Artifact(
                content=receipt.content,
                role="candidate_miss",
                producer=Producer(stage="admit_candidate"),
                grounding="ambiguous",
                proof_ref=receipt.proof_ref,
            ),
        )
    pointer = ArtifactPointer.model_validate(receipt.values["manifest"])
    documents = [(receipt.content, pointer)]
    for prior in receipts[:-1]:
        if "artifact_pointer" in prior.values:
            documents.append(
                (
                    prior.content,
                    ArtifactPointer.model_validate(prior.values["artifact_pointer"]),
                )
            )
    return tuple(
        Artifact(
            content=content,
            role=p.role,
            producer=p.producer,
            grounding=p.grounding,
            proof_ref=p.proof_ref,
        )
        for content, p in documents
    )


def _result(receipts):
    result = receipts[-1].values
    return {
        "pointers": [
            *result["pointers"],
            *(
                artifact.pointer().model_dump(mode="json")
                for artifact in reference_artifacts(receipts)
            ),
        ],
        "receipt_id": result["receipt_id"],
    }


def reference_resolved(receipts):
    result = receipts[-1].values
    reference = (
        {
            "state": result.get("reference_state", "miss"),
            "reason": result["miss"],
            "evidence": [
                artifact.pointer().model_dump(mode="json")
                for artifact in reference_artifacts(receipts)
            ],
        }
        if "miss" in result
        else {"state": "hit", **_result(receipts)}
    )
    return PathFact.model_validate(
        {
            "event_type": "ReferenceResolved",
            "kind": "decision",
            "grounding": "verified",
            "pass": result["pass"],
            "rests_on": result["rests_on"],
            "payload": {"reference": reference},
        }
    )


ADMIT_CANDIDATE = StageDef(
    name="admit_candidate",
    steps=(inspect_candidate, place_candidate, candidate_metadata),
    path_fact=reference_resolved,
    artifacts=reference_artifacts,
    ports={
        "inspect_candidate": "store",
        "place_candidate": "store",
        "candidate_metadata": "store",
    },
)


def preflight(ctx, identity):
    from bio_neb.journey_path import AdmissionFailed, Answered, Drifted, Miss, Verified

    if not isinstance(ctx.state.inputs, Verified):
        raise ProviderError("semantic", "cpu_inputs_not_verified")
    reference = ctx.state.reference
    if isinstance(reference, Unresolved) and ctx.plan.reference.prebuilt_index is None:
        return identity.whoami()
    if (
        isinstance(reference, (Miss, Drifted, AdmissionFailed))
        and isinstance(ctx.state.review, Answered)
        and ctx.state.review.action == "approve_cpu_preparation"
    ):
        return identity.whoami()
    raise ProviderError("semantic", "cpu_preparation_not_authorized")


def verify_source(ctx, store):
    """Bind the CPU request to its source: the placed object, or the approved URL.

    Without a recorded acquisition input storage placed no source; the job
    fetches it, and ``verify_bundle`` establishes it from the job's receipt.
    """
    plan = ctx.plan
    acquisition = ctx.state.inputs.reference_acquisition
    try:
        request = contracts.compile_cpu_request(plan, acquisition)
    except ValueError as error:
        raise Invariant("cpu_request_invalid", "cpu_request_invalid") from error
    if acquisition is None:
        return Receipt(values={"request": request.model_dump(mode="json")})
    selected = next(
        p for p in ctx.state.inputs.pointers if p.role == "reference_source"
    )
    if (
        selected.location
        != f"s3://{plan.storage.bucket}/{plan.layout.source_object_key}"
    ):
        raise ProviderError("semantic", "reference_source_location_mismatch")
    checked = Receipt(
        values={
            "bytes": selected.bytes,
            "sha256": ContentId.parse(selected.content_id).digest,
        },
        proof_ref=selected.proof_ref,
    )
    _expect(checked, acquisition.source_byte_count, acquisition.source_sha256)
    source_object = contracts.VerifiedObject(
        role="reference_source",
        bucket=plan.storage.bucket,
        object_key=plan.layout.source_object_key,
        byte_count=checked.values["bytes"],
        sha256=checked.values["sha256"],
    )
    content = canonical_json(
        {
            "execution_id": plan.identity.execution_id,
            "source": source_object.model_dump(mode="json"),
        }
    ).encode()
    key = plan.layout.source_manifest_object_key
    manifest_receipt = store.put(plan.storage.bucket, key, content)
    _expect(manifest_receipt, len(content), _digest(content))
    source = contracts.VerifiedReferenceSource(
        source_mode="existing_object",
        reference_asset_id=plan.reference.reference_asset_id,
        source_locator=f"object://{plan.storage.bucket}/{plan.layout.source_object_key}",
        object=source_object,
        manifest_object=contracts.VerifiedObject(
            role="source_manifest",
            bucket=plan.storage.bucket,
            object_key=key,
            byte_count=len(content),
            sha256=_digest(content),
        ),
        exact_inventory=tuple(sorted((key, plan.layout.source_object_key))),
        verification_mode="reuse_creation_proof",
        independently_full_read=False,
    )
    return Receipt(
        content=content,
        values={
            "request": request.model_dump(mode="json"),
            "source": source.model_dump(mode="json"),
            "artifact_pointer": _pointer(
                ctx,
                "cpu_preparation",
                "source_manifest",
                plan.storage.bucket,
                key,
                manifest_receipt,
            ),
        },
    )


def _cpu_inputs(ctx):
    """The compiled request and, for a placed source, its verified identity."""
    result = ctx.receipts[1].values
    return (
        contracts.ReferencePreparationRequest.model_validate(result["request"]),
        contracts.VerifiedReferenceSource.model_validate(result["source"])
        if "source" in result
        else None,
    )


def submit_cpu(ctx, jobs):
    try:
        request, _ = _cpu_inputs(ctx)
        compiled = contracts._compile_worker_request(request)
    except ValueError as error:
        raise Invariant("cpu_request_invalid", "cpu_request_invalid") from error
    placed = place_execution_request(
        compiled,
        source_path=str(
            Path(ctx.plan.launch.work_directory)
            / ctx.plan.layout.cpu_job_name
            / compiled.identity.sha256
            / "cpu-request.json"
        ),
        container_path="/opt/bio-neb/request.json",
    )
    runtime = ctx.plan.runtimes.cpu
    return jobs.submit(
        ctx.plan.layout.cpu_job_name,
        {
            "target": ctx.plan.scope.model_dump(mode="json"),
            "provider_image_reference": runtime.provider_image_reference,
            "image_index_digest": runtime.provider_image_index_digest,
            "image_child_digest": runtime.worker_runtime_image.rsplit("@", 1)[1],
            "platform": runtime.platform,
            "preset": runtime.preset,
            "disk_size": runtime.disk_size,
            "disk_size_bytes": runtime.disk_size_bytes,
            "shared_memory_size": None,
            "shared_memory_bytes": 0,
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


def cpu_terminal(samples):
    sample = samples[-1]
    state = sample.values["state"]
    if state in {"FAILED", "CANCELLED", "ERROR"}:
        # Every observed sample carries its log window (null when unavailable)
        # and the job identity; a missing key is a broken port, not a fallback.
        if not {"log", "provider_job_id"} <= set(sample.values):
            raise Invariant(
                "worker_observation_invalid", "terminal sample lacks log or job id"
            )
        mismatch = source_mismatch(sample.values["log"])
        if mismatch is not None:
            # The same typed stop as a controller-streamed md5 mismatch.
            raise ProviderError(
                "semantic",
                "source_identity_mismatch",
                {
                    **mismatch.model_dump(mode="json", exclude={"kind"}),
                    "provider_job_id": sample.values["provider_job_id"],
                    "terminal_state": state,
                },
            )
        raise ProviderError("worker_failed", "cpu_job_failed", sample.values)
    return sample if state == "COMPLETED" else None


CPU_OBSERVATION = JobObservation()


def await_cpu(ctx, jobs):
    return observe_worker(ctx, jobs, CPU_OBSERVATION)


def verify_bundle(ctx, store):
    request, source = _cpu_inputs(ctx)
    raw = store.read_small(
        ctx.plan.storage.bucket, ctx.plan.layout.reference_receipt_object_key
    )
    try:
        observations = durable_worker_observations(
            ctx.receipts[-1],
            raw,
            ctx.plan.storage.bucket,
            ctx.plan.layout.reference_receipt_object_key,
        )
        observed = contracts.validate_preparation_observation_closure(
            ctx.plan.scope,
            ctx.plan.storage,
            ctx.plan.runtimes.cpu,
            request,
            source,
            ctx.receipts[-1].provider_ids[0],
            observations,
            durable=True,
        )
        worker = observed.receipt
        uploads = tuple(
            contracts.VerifiedObject.model_validate(item)
            for item in worker["prepared_uploads"]
        )
        if tuple(item.role for item in uploads) != PREPARED_ROLES:
            raise ValueError("worker output roles changed")
    except (ValueError, KeyError) as error:
        raise ProviderError(
            "semantic", "cpu_receipt_invalid", {"reason": str(error)}
        ) from None
    acquired = (
        _verify_acquired_source(ctx, store, request, worker, raw)
        if source is None
        else None
    )
    pointers = []
    for item, key, role in zip(
        uploads, ctx.plan.layout.prepared_object_keys, PREPARED_ROLES, strict=True
    ):
        if (
            item.object_key != key
            or item.bucket != ctx.plan.storage.bucket
            or item.role != role
        ):
            raise ProviderError("semantic", "cpu_output_location_mismatch")
        checked = use_produced_object(
            store,
            item,
            proof_ref=raw.proof_ref,
            mode=ctx.plan.verification.effective.reference_preparation,
        )
        _expect(checked, item.byte_count, item.sha256)
        pointers.append(
            _pointer(ctx, "cpu_preparation", role, item.bucket, key, checked)
        )
    acquisition = None
    if acquired is not None:
        source, source_pointers = _publish_acquired_source(
            ctx, store, request, acquired[0]
        )
        pointers.extend(source_pointers)
        acquisition = acquired[1].model_dump(mode="json")
    closure = contracts.PreparedReferenceObjectClosure(
        objects=uploads,
        worker_receipt=contracts.VerifiedObject.model_validate(
            observed.receipt_object.model_dump()
        ),
        pre_manifest_inventory=tuple(
            sorted(
                [
                    *(item.object_key for item in uploads),
                    request.worker_receipt_object_key,
                ]
            )
        ),
    )
    manifest = contracts.compile_prepared_reference_manifest(
        ctx.plan.storage,
        ctx.plan.runtimes.cpu,
        ctx.plan.verification.effective.reference_preparation,
        request,
        source,
        closure,
    )
    return Receipt(
        values={
            "manifest_document": manifest.model_dump(mode="json"),
            "pointers": pointers,
            "reference_acquisition": acquisition,
            "artifact_pointer": _pointer(
                ctx,
                "cpu_preparation",
                "worker_receipt",
                ctx.plan.storage.bucket,
                ctx.plan.layout.reference_receipt_object_key,
                raw,
            ),
        },
        content=raw.content,
        proof_ref=raw.proof_ref,
    )


def _verify_acquired_source(ctx, store, request, worker, raw):
    """Check the job-fetched .gz the receipt names; return it and its acquisition.

    The worker checked the approved md5 and size while fetching; the host checks
    the stored object against the receipt and records the acquired identities.
    """
    observed = worker["source"]
    plain = worker["prepared_uploads"][0]
    source_object = contracts.VerifiedObject(
        role="reference_source",
        bucket=ctx.plan.storage.bucket,
        object_key=request.source.object_key,
        byte_count=observed["byte_count"],
        sha256=observed["sha256"],
    )
    checked = use_produced_object(
        store,
        source_object,
        proof_ref=raw.proof_ref,
        mode=ctx.plan.verification.effective.reference_preparation,
    )
    _expect(checked, source_object.byte_count, source_object.sha256)
    return (
        _pointer(
            ctx,
            "cpu_preparation",
            "reference_source",
            source_object.bucket,
            source_object.object_key,
            checked,
        ),
        ReferenceAcquisition(
            source_md5=observed["md5"],
            source_byte_count=observed["byte_count"],
            source_sha256=observed["sha256"],
            plain_fasta_byte_count=plain["byte_count"],
            plain_fasta_sha256=plain["sha256"],
        ),
    )


def _publish_acquired_source(ctx, store, request, source_pointer):
    """Publish the fetched source's manifest once every object is verified."""
    plan = ctx.plan
    source_object = contracts.VerifiedObject(
        role="reference_source",
        bucket=plan.storage.bucket,
        object_key=request.source.object_key,
        byte_count=source_pointer["bytes"],
        sha256=ContentId.parse(source_pointer["content_id"]).digest,
    )
    content = canonical_json(
        {
            "execution_id": plan.identity.execution_id,
            "source": source_object.model_dump(mode="json"),
        }
    ).encode()
    key = plan.layout.source_manifest_object_key
    manifest_receipt = store.put(plan.storage.bucket, key, content)
    _expect(manifest_receipt, len(content), _digest(content))
    source = contracts.VerifiedReferenceSource(
        source_mode="public_url",
        reference_asset_id=plan.reference.reference_asset_id,
        source_locator=request.source.url,
        object=source_object,
        manifest_object=contracts.VerifiedObject(
            role="source_manifest",
            bucket=plan.storage.bucket,
            object_key=key,
            byte_count=len(content),
            sha256=_digest(content),
        ),
        exact_inventory=tuple(sorted((key, request.source.object_key))),
        verification_mode="reuse_creation_proof",
        independently_full_read=False,
    )
    manifest_pointer = _pointer(
        ctx,
        "cpu_preparation",
        "source_manifest",
        plan.storage.bucket,
        key,
        manifest_receipt,
    )
    return source, [source_pointer, manifest_pointer]


def publish_prepared(ctx, store):
    result = ctx.receipts[-1]
    published = _publish(
        ctx,
        store,
        "cpu_preparation",
        "prepared_reference_manifest",
        ctx.plan.layout.reference_manifest_object_key,
        result.values["manifest_document"],
        result.values["pointers"],
    )
    return published.model_copy(
        update={
            "values": {
                **published.values,
                "reference_acquisition": result.values["reference_acquisition"],
            }
        }
    )


def reference_prepared(receipts):
    result = receipts[-1].values
    return PathFact.model_validate(
        {
            "event_type": "ReferencePrepared",
            "kind": "verified_artifact",
            "grounding": "verified",
            "pass": result["pass"],
            "rests_on": result["rests_on"],
            "payload": {
                "state": "prepared",
                **_result(receipts),
                "reference_acquisition": result["reference_acquisition"],
            },
        }
    )


CPU_PREPARATION = StageDef(
    name="cpu_preparation",
    steps=(
        preflight,
        verify_source,
        submit_cpu,
        await_cpu,
        verify_bundle,
        publish_prepared,
    ),
    closure=cpu_terminal,
    path_fact=reference_prepared,
    artifacts=reference_artifacts,
    ports={
        "preflight": "identity",
        "verify_source": "store",
        "submit_cpu": "jobs",
        "await_cpu": "jobs",
        "verify_bundle": "store",
        "publish_prepared": "store",
    },
)
