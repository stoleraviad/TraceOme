"""Input placement recipe; effects receive names and approved stream identities."""

import hashlib

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from bio_neb.application.artifact_verification import use_produced_object
from bio_neb.content import ContentId, canonical_json
from bio_neb.failure import ProviderError
from bio_neb.input_inspection.tool import is_staged_object
from bio_neb.journey_path import (
    Artifact,
    ArtifactPointer,
    PathFact,
    Producer,
    ReferenceAcquisition,
)
from bio_neb.parabricks.runner import StageDef
from bio_neb.providers.object_locations import object_address, object_location
from bio_neb.providers.ports import Receipt, TransferSource
from bio_neb.verified_artifacts import ArtifactIdentity, ArtifactProof, VerifiedObject


def preflight(ctx, identity):
    return identity.whoami()


def fresh_bucket(ctx, store):
    return store.fresh_bucket(ctx.plan.storage.bucket)


# The staged scope's manifest, beside its objects.
STAGING_MANIFEST = "staging-manifest.json"


class _StagedManifestEntry(BaseModel):
    """One file of a scope's ``staging-manifest.json``; other keys are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    object_key: StrictStr
    bytes: StrictInt
    # Their form is checked where InputsVerified records them.
    sha256: StrictStr | None = None
    md5: StrictStr | None = None


class _StagedManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    files: tuple[_StagedManifestEntry, ...]


def _source(fastq):
    if is_staged_object(fastq.registered_path):
        return fastq.registered_path
    return TransferSource(
        "local", fastq.registered_path, fastq.byte_count, fastq.sha256
    )


def _inputs(plan):
    fastqs = tuple(
        (fastq.role, fastq.object_key, _source(fastq)) for fastq in plan.fastqs
    )
    if plan.input_source is None:
        return fastqs
    return fastqs + (
        ("reference_source", plan.layout.source_object_key, plan.input_source),
    )


def place_inputs(ctx, store):
    """Place the inputs; the acquisition exists only when the source streamed here."""
    pointers, acquisition, staged = [], None, []
    sizes = {fastq.role: fastq.byte_count for fastq in ctx.plan.fastqs}
    for role, key, source in _inputs(ctx.plan):
        if isinstance(source, str):
            receipt, basis = _copy_staged(
                store, ctx.plan.storage.bucket, key, role, source, sizes[role]
            )
            pointers.append(_pointer(ctx, role, key, receipt).model_dump(mode="json"))
            staged.append(basis)
            continue
        receipt = store.put(ctx.plan.storage.bucket, key, source)
        if (
            receipt.values.get("bytes") != source.byte_count
            or (
                source.sha256 is not None
                and receipt.values.get("sha256") != source.sha256
            )
            or receipt.values.get("md5") != source.md5
        ):
            raise ProviderError("semantic", "upload_identity_mismatch")
        pointers.append(_pointer(ctx, role, key, receipt).model_dump(mode="json"))
        if role == "reference_source":
            acquisition = _acquisition(receipt)
    return Receipt(
        values={
            "pointers": pointers,
            "reference_acquisition": None
            if acquisition is None
            else acquisition.model_dump(mode="json"),
            "staged_inputs": staged,
        }
    )


def _copy_staged(store, bucket, key, role, location, byte_count):
    """Trust a staged object: copy it in the provider to the key an upload uses.

    The scope's staging manifest supplies its sha256 when it lists one; otherwise
    the object is read once to hash it, since the H100 worker checks every input
    it downloads against that sha256. No byte is uploaded from here.
    """
    source_bucket, source_key = object_address(location)
    folder, _separator, _name = source_key.rpartition("/")
    manifest_key = f"{folder}/{STAGING_MANIFEST}" if folder else STAGING_MANIFEST
    entry = _manifest_entry(store, source_bucket, manifest_key, source_key)
    if entry is not None and entry.sha256 is not None:
        observed, sha256, sha256_basis = entry.bytes, entry.sha256, "staging_manifest"
    else:
        read = store.head(
            source_bucket, source_key, verification="independent_full_read"
        )
        observed, sha256 = read.values["bytes"], read.values["sha256"]
        sha256_basis = "source_read"
    if observed != byte_count or (entry is not None and entry.bytes != byte_count):
        raise ProviderError(
            "semantic",
            "staged_input_size_mismatch",
            {"role": role, "inspected_bytes": byte_count, "staged_bytes": observed},
        )
    receipt = store.copy(
        ArtifactProof(
            identity=ArtifactIdentity(byte_count=byte_count, sha256=sha256),
            location=location,
            producer=sha256_basis,
            expected_identity=None,
        ),
        bucket,
        key,
    )
    # InputsVerified types this as its StagedInputBasis.
    return receipt, {
        "role": role,
        "source_location": location,
        "manifest_location": None
        if entry is None
        else object_location(source_bucket, manifest_key),
        "sha256": sha256,
        "sha256_basis": sha256_basis,
        "md5": None if entry is None else entry.md5,
    }


def _manifest_entry(store, bucket, manifest_key, object_key):
    """The manifest's entry for this object, or None without a manifest or entry."""
    try:
        content = store.read_small(bucket, manifest_key).content
    except ProviderError as error:
        if error.detail == "object_missing":
            return None
        raise
    files = _StagedManifest.model_validate_json(content).files
    return next((item for item in files if item.object_key == object_key), None)


def _acquisition(receipt):
    """The approved source's measured bytes: md5 checked, sha256 derived, content."""
    try:
        decompressed = receipt.values["decompressed"]
        return ReferenceAcquisition(
            source_md5=receipt.values["md5"],
            source_byte_count=receipt.values["bytes"],
            source_sha256=receipt.values["sha256"],
            plain_fasta_byte_count=decompressed["bytes"],
            plain_fasta_sha256=decompressed["sha256"],
        )
    except (KeyError, TypeError, ValueError):
        raise ProviderError("semantic", "upload_identity_mismatch") from None


def _pointer(ctx, role, key, receipt):
    return ArtifactPointer(
        role=role,
        content_id=str(ContentId(receipt.values["sha256"])),
        location=f"s3://{ctx.plan.storage.bucket}/{key}",
        bytes=receipt.values["bytes"],
        producer=Producer(
            stage="input_storage", authority_id=ctx.plan.identity.authority_id
        ),
        grounding="verified",
        proof_ref=receipt.proof_ref,
    )


def verify_inputs(ctx, store):
    placed = ctx.receipts[-1]
    for (role, key, _source), pointer in zip(
        _inputs(ctx.plan), placed.values["pointers"], strict=True
    ):
        use_produced_object(
            store,
            VerifiedObject(
                role=role,
                bucket=ctx.plan.storage.bucket,
                object_key=key,
                byte_count=pointer["bytes"],
                sha256=ContentId.parse(pointer["content_id"]).digest,
            ),
            proof_ref=pointer["proof_ref"],
            mode=ctx.plan.verification.effective.input_storage,
        )
    return placed


def publish_manifest(ctx, store):
    pointers = ctx.receipts[-1].values["pointers"]
    acquisition = ctx.receipts[-1].values["reference_acquisition"]
    staged = ctx.receipts[-1].values["staged_inputs"]
    content = canonical_json(
        {
            "execution_id": ctx.plan.identity.execution_id,
            "inputs": pointers,
            "reference_acquisition": acquisition,
            # Present only when an input was a staged object.
            **({"staged_inputs": staged} if staged else {}),
        }
    ).encode()
    digest = hashlib.sha256(content).hexdigest()
    # Observation documents retain their exact proofs. New proof bytes get a
    # new immutable document, while the described input content stays unchanged.
    key = (
        ctx.plan.layout.input_manifest_object_key.removesuffix(".json")
        + f"/{digest}.json"
    )
    checked = store.put(ctx.plan.storage.bucket, key, content)
    if checked.values.get("sha256") != digest or checked.values.get("bytes") != len(
        content
    ):
        raise ProviderError("semantic", "input_manifest_identity_mismatch")
    return Receipt(
        content=content,
        values={
            "pointers": [
                *pointers,
                _pointer(ctx, "input_manifest", key, checked).model_dump(mode="json"),
            ],
            "receipt_id": str(ContentId(digest)),
            "reference_acquisition": acquisition,
            "staged_inputs": staged,
            "bound_event_id": ctx.state.latest_event_ids["ExecutionBound"],
            "answer_ids": list(ctx.state.stage_answer_ids.get("input_storage", ())),
            "pass": ctx.state.pass_,
        },
    )


def inputs_verified(receipts):
    result = receipts[-1].values
    pointers = [
        *result["pointers"][:-1],
        manifest_artifacts(receipts)[0].pointer().model_dump(mode="json"),
    ]
    return PathFact.model_validate(
        {
            "event_type": "InputsVerified",
            "kind": "verified_artifact",
            "grounding": "verified",
            "pass": result["pass"],
            "rests_on": [result["bound_event_id"], *result["answer_ids"]],
            "payload": {
                "state": "verified",
                "pointers": pointers,
                "receipt_id": result["receipt_id"],
                "reference_acquisition": result["reference_acquisition"],
                "staged_inputs": result["staged_inputs"],
            },
        }
    )


def manifest_artifacts(receipts):
    receipt = receipts[-1]
    verified = ArtifactPointer.model_validate(receipt.values["pointers"][-1])
    return (
        Artifact(
            content=receipt.content,
            role=verified.role,
            producer=verified.producer,
            grounding=verified.grounding,
            proof_ref=verified.proof_ref,
        ),
    )


INPUT_STORAGE = StageDef(
    name="input_storage",
    steps=(preflight, fresh_bucket, place_inputs, verify_inputs, publish_manifest),
    path_fact=inputs_verified,
    artifacts=manifest_artifacts,
    ports={
        "preflight": "identity",
        "fresh_bucket": "store",
        "place_inputs": "store",
        "verify_inputs": "store",
        "publish_manifest": "store",
    },
)
