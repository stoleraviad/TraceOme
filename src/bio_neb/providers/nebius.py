"""Layout-blind Nebius store and identity, with existing transfer mechanics.

Resource names, keys and approved stream identities come from the caller. The
provider owns CLI/credential construction and physical contact observation.
Jobs use deterministic names and bounded provider samples.
"""

import hashlib
from functools import cached_property, wraps
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from bio_neb.failure import ProviderError
from bio_neb.providers.gzip_content import GzipContentMeasure
from bio_neb.providers.nebius_buckets import NebiusBucketAccess
from bio_neb.providers.nebius_common import (
    NebiusAccessScope,
    NebiusCliError,
    S3ContactError,
    nebius_common_arguments,
    observe_execution_contact_finish,
    observe_execution_contact_start,
)
from bio_neb.providers.nebius_controller_runtime import make_accepted_nebius_cli_runner
from bio_neb.providers.nebius_object_storage import (
    NebiusObjectStorageError,
    NebiusObjectStorageMechanics,
    ObjectIdentity,
    StorageAccessError,
    StorageConditionalCollision,
    StorageMalformedResponse,
    StorageReadAbsent,
    nebius_object_storage_client,
)
from bio_neb.providers.nebius_object_storage_credentials import (
    SelectedNebiusCredentialResolver,
)
from bio_neb.providers.nebius_serverless_job_contact import NebiusJobContactError
from bio_neb.providers.ports import Ports, Receipt, TransferSource, VerificationMode


class _ObservedResponse:
    """Track response consumption independently of upload and content validation."""

    def __init__(self, source):
        self.source = source
        self.complete = False
        self.byte_count = 0
        value = getattr(source, "headers", {}).get("Content-Length")
        self.content_length = (
            int(value) if isinstance(value, str) and value.isdecimal() else None
        )

    def read(self, size=-1):
        chunk = self.source.read(size)
        self.byte_count += len(chunk)
        if size != 0 and (not chunk or size < 0):
            self.complete = (
                self.content_length is None or self.byte_count == self.content_length
            )
        elif self.content_length is not None and self.byte_count == self.content_length:
            self.complete = True
        return chunk


def _storage_failure_evidence(error):
    """Project the transfer core's safe facts, never exception text or responses."""
    evidence = {"kind": error.kind, "operation": error.operation}
    for name in ("safe_error_category", "bytes_read"):
        value = getattr(error, name)
        if value is not None:
            evidence[name] = value
    detail = {
        name: value
        for name in ("status", "code")
        if (value := getattr(error.detail, name)) is not None
    }
    if detail:
        evidence["detail"] = detail
    if error.contacts:
        evidence["contacts"] = [
            {
                "operation": contact.operation,
                "ordinal": contact.ordinal,
                "classification": contact.classification,
                "detail": {
                    name: value
                    for name in ("status", "code")
                    if (value := getattr(contact.detail, name)) is not None
                },
            }
            for contact in error.contacts
        ]
    if error.residue:
        evidence["residue"] = [
            {"object_key": item.object_key, "upload_id": item.upload_id}
            for item in error.residue
        ]
    return evidence


def _classified(effect):
    @wraps(effect)
    def call(*args, **kwargs):
        try:
            return effect(*args, **kwargs)
        except NebiusObjectStorageError as error:
            evidence = _storage_failure_evidence(error)
            if isinstance(error, StorageConditionalCollision):
                failure_class, detail = "rejected", "object_content_conflict"
            elif isinstance(error, StorageReadAbsent):
                failure_class, detail = "rejected", "object_missing"
            elif isinstance(error, StorageAccessError):
                failure_class, detail = "rejected", "storage_access_refused"
            elif isinstance(error, StorageMalformedResponse):
                failure_class, detail = "semantic", "object_identity_invalid"
            elif error.safe_error_category == "checksum_mismatch":
                failure_class, detail = "semantic", "source_identity_mismatch"
            else:
                # The core owns bounded part retries and response-loss reconciliation.
                # Interrupted mutation remains a review stop, never a new replay.
                failure_class, detail = "exhausted", "storage_transfer_incomplete"
            raise ProviderError(failure_class, detail, evidence) from None
        except S3ContactError as error:
            raise ProviderError(
                "rejected", "storage_access_refused", {"kind": error.kind}
            ) from None
        except NebiusJobContactError as error:
            raise ProviderError(
                "worker_failed" if error.category == "provider_failed" else "rejected",
                "no_capacity"
                if error.provider_reason == "no_capacity"
                else error.reason,
                {
                    "provider_job_id": error.provider_job_id,
                    "resource_absent": error.operation == "create"
                    and error.category == "create_rejected"
                    and error.provider_reason == "no_capacity",
                    "log_window": None
                    if error.log_window is None
                    else error.log_window.model_dump(mode="json"),
                },
            ) from None
        except NebiusCliError as error:
            evidence = {"code": error.code, "journey_reason": error.journey_reason()}
            if error.provider_reason is not None:
                evidence["provider_reason"] = error.provider_reason
            if error.failure_evidence is not None:
                evidence["failure_evidence"] = error.failure_evidence.public_fields()
            raise ProviderError("rejected", "nebius_command_failed", evidence) from None
        except HTTPError as error:
            raise ProviderError(
                "transient"
                if error.code in {408, 429, 500, 502, 503, 504}
                else "rejected",
                "source_http_unavailable",
                {"http_status": error.code},
            ) from None
        except (URLError, TimeoutError, ConnectionError):
            raise ProviderError("transient", "source_transport_unavailable") from None
        except OSError:
            raise ProviderError("rejected", "source_or_transport_unavailable") from None

    return call


class NebiusProvider:
    def __init__(
        self,
        *,
        selection,
        endpoint_url: str,
        region: str,
        journal,
        cli_runner=None,
        credential_resolver=None,
        s3_client_factory=nebius_object_storage_client,
        url_opener=urlopen,
        registry_resolver=None,
    ):
        self._selection = selection
        self._endpoint, self._region = endpoint_url, region
        self._journal = journal
        self._runner = cli_runner or make_accepted_nebius_cli_runner(
            selection.nebius_executable, execution_contact_observer=journal
        )
        self._credentials = credential_resolver or SelectedNebiusCredentialResolver(
            None, selection, cli_runner=self._runner, execution_contact_observer=journal
        )
        self._client_factory, self._url_opener = s3_client_factory, url_opener
        from bio_neb.providers.nebius_registry_v2 import NebiusRegistryV2Resolver

        self._registry = registry_resolver or NebiusRegistryV2Resolver(
            cli_runner=self._runner,
            profile=selection.topology.profile,
            execution_contact_observer=journal,
        )
        self.store = _Store(self)
        self.identity = _Identity(self)
        self.jobs = _Jobs(self)

    def for_step(self, stage: str, step: str) -> Ports:
        return Ports(self.store, self.jobs, self.identity)

    @cached_property
    def authentication(self):
        from bio_neb.providers.nebius_authentication import NebiusAuthenticationProbe

        return NebiusAuthenticationProbe(cli_runner=self._runner)

    def _scope(self):
        scope = self._selection.topology
        return NebiusAccessScope(scope.profile, scope.tenant_id, scope.project_id)

    def _common(self):
        return nebius_common_arguments(self._scope())

    def _objects(self, bucket):
        return NebiusObjectStorageMechanics(
            endpoint_url=self._endpoint,
            region=self._region,
            bucket=bucket,
            credential_resolver=self._credentials,
            client_factory=self._client_factory,
            execution_contact_observer=self._journal,
        )


class _Identity:
    def __init__(self, provider):
        self.provider = provider

    @_classified
    def whoami(self):
        p = self.provider
        scope = p._selection.topology
        kind = p.authentication.authenticate(p._scope())
        p._credentials()
        return Receipt(
            values={
                "profile_kind": kind,
                "project_id": scope.project_id,
                "tenant_id": scope.tenant_id,
            }
        )


def _receipt(bucket, key, identity, verification, *, expected=None, source_proof=None):
    from bio_neb.verified_artifacts import ArtifactIdentity, ArtifactProof

    proof = ArtifactProof(
        identity=ArtifactIdentity(
            byte_count=identity.byte_count, sha256=identity.sha256
        ),
        location=f"s3://{bucket}/{key}",
        producer=verification,
        expected_identity=expected,
        source_proof=source_proof,
    )
    return Receipt(
        values={
            "bytes": identity.byte_count,
            "sha256": identity.sha256,
            "verification_mode": verification,
            "artifact_proof": proof.model_dump(mode="json"),
        },
        provider_ids=(f"s3://{bucket}/{key}",),
    )


def _put_renamed(objects, key, stream, content):
    """Stream one source with its contig names rewritten (#464, owner I7); return the
    placed identity and the source's own, measured in the same pass."""
    renamed = content.transform.open(stream)
    placed = content.transform.placed_byte_count
    if placed is None:
        # A .fai is small and rewritten whole: its offsets' digits may change.
        body = renamed.read()
        if renamed.source_identity()[0] != content.byte_count:
            raise ProviderError("semantic", "source_identity_mismatch")
        identity = objects.put_conditional_bytes(
            key, body, content_type="application/octet-stream"
        ).identity
    else:
        identity = objects.put_conditional_stream(
            key, renamed, expected=None, expected_byte_count=placed
        ).identity
    source = renamed.source_identity()
    if source[0] != content.byte_count:
        raise ProviderError("semantic", "source_identity_mismatch")
    return identity, source


def _put_source(objects, key, stream, content, expected):
    """Stream one selected source once; return its identity and measured values."""
    if content.transform is not None:
        identity, source = _put_renamed(objects, key, stream, content)
        return identity, {"source": {"bytes": source[0], "sha256": source[1]}}
    measure = (
        GzipContentMeasure(stream, content.excluded_sequences)
        if content.measure_gzip_content
        else None
    )
    reader = measure or stream
    if content.md5 is not None:
        placed = objects.put_conditional_authority_stream(
            key,
            reader,
            expected_byte_count=content.byte_count,
            approved_algorithm="md5",
            approved_checksum=content.md5,
        )
        identity = ObjectIdentity(placed.byte_count, placed.sha256)
    else:
        identity = objects.put_conditional_stream(
            key, reader, expected=expected, expected_byte_count=content.byte_count
        ).identity
    if measure is None:
        return identity, None
    # The publisher md5 matched while streaming; the decompressed content
    # was measured in the same pass.
    plain = measure.identity()
    return identity, {
        "md5": content.md5,
        "decompressed": {"bytes": plain[0], "sha256": plain[1]},
    }


class _Store:
    def __init__(self, provider):
        self.provider = provider

    @_classified
    def fresh_bucket(self, name):
        p = self.provider
        project = p._selection.topology.project_id
        access = NebiusBucketAccess(
            p._runner,
            common=p._common(),
            project_id=project,
            region=p._region,
            domain=urlsplit(p._endpoint).hostname,
        )
        matches = access.find(name)
        if len(matches) > 1:
            raise ProviderError("rejected", "bucket_name_ambiguous")
        created = not matches
        bucket = (
            access.create(name)[0] if created else access.validate(name, matches[0])
        )
        metadata = bucket["metadata"]
        if created:
            # The existing accepted readiness policy removes only incomplete
            # multipart uploads. Existing resources are never reconfigured.
            objects = p._objects(name)
            lifecycle = {
                "Rules": [
                    {
                        "ID": "bio-neb-abort-incomplete-multipart",
                        "Status": "Enabled",
                        "Filter": {},
                        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                    }
                ]
            }
            objects.put_lifecycle_once(lifecycle)
            if objects.get_lifecycle().value != lifecycle:
                raise ProviderError("rejected", "bucket_readiness_mismatch")
        return Receipt(
            values={"name": name, "reused": not created},
            provider_ids=(metadata["id"],),
            resource_count=1,
        )

    def _existing(self, objects, key, expected):
        try:
            observed = objects.head(key).value
        except StorageReadAbsent:
            return False
        if observed != expected:
            raise ProviderError("rejected", "object_content_conflict")
        return True

    @_classified
    def put(self, bucket, key, content: bytes | TransferSource):
        objects = self.provider._objects(bucket)
        expected = (
            (
                ObjectIdentity(content.byte_count, content.sha256)
                if content.sha256 is not None
                else None
            )
            if isinstance(content, TransferSource)
            else ObjectIdentity(len(content), hashlib.sha256(content).hexdigest())
        )
        reused = self._existing(objects, key, expected)
        measured = None
        if not reused:
            if isinstance(content, bytes):
                objects.put_conditional_bytes(key, content)
            elif content.kind == "local":
                path = Path(content.locator)
                with path.open("rb") as stream:
                    expected, measured = _put_source(
                        objects, key, stream, content, expected
                    )
            else:
                request = Request(
                    content.locator, headers={"Accept-Encoding": "identity"}
                )
                unit = observe_execution_contact_start(
                    self.provider._journal,
                    category="source",
                    operation="http.get",
                    executable_identity="http-client",
                )
                response = None
                try:
                    with self.provider._url_opener(request) as stream:
                        if getattr(stream, "status", 200) != 200:
                            raise ProviderError("rejected", "source_unavailable")
                        response = _ObservedResponse(stream)
                        expected, measured = _put_source(
                            objects, key, response, content, expected
                        )
                finally:
                    received = response is not None and response.complete
                    observe_execution_contact_finish(
                        self.provider._journal,
                        unit,
                        disposition="succeeded" if received else "failed",
                        parse_classification="response_received"
                        if received
                        else "provider_error",
                    )
        receipt = _receipt(
            bucket,
            key,
            expected,
            "named_creation_reuse" if reused else "streamed_sha256",
            expected=(
                {"byte_count": content.byte_count, "sha256": content.sha256}
                if isinstance(content, TransferSource) and content.sha256 is not None
                else None
            ),
        )
        if measured is None:
            return receipt
        return receipt.model_copy(update={"values": {**receipt.values, **measured}})

    @_classified
    def copy(self, source, bucket, key):
        from bio_neb.providers.object_locations import object_address

        source_bucket, source_key = object_address(source.location)
        expected = ObjectIdentity(source.identity.byte_count, source.identity.sha256)
        objects = self.provider._objects(bucket)
        if not self._existing(objects, key, expected):
            objects.copy_conditional(
                source_bucket=source_bucket,
                source_key=source_key,
                destination_key=key,
                expected=expected,
            )
        receipt = _receipt(
            bucket, key, expected, "controlled_copy", source_proof=source.proof_id
        )
        return receipt.model_copy(
            update={
                "values": {
                    **receipt.values,
                    "source_proof": source.model_dump(mode="json"),
                }
            }
        )

    @_classified
    def head(
        self,
        bucket,
        key,
        *,
        verification: VerificationMode = "reuse_creation_proof",
    ):
        objects = self.provider._objects(bucket)
        if verification == "independent_full_read":
            identity = objects.full_hash(key).value
        elif verification == "reuse_creation_proof":
            identity = objects.head(key).value
        else:
            raise ValueError("unknown verification mode")
        return _receipt(bucket, key, identity, verification)

    @_classified
    def read_small(self, bucket, key):
        content = (
            self.provider._objects(bucket)
            .get_bytes(key, maximum_bytes=1024 * 1024)
            .value
        )
        identity = ObjectIdentity(len(content), hashlib.sha256(content).hexdigest())
        return _receipt(bucket, key, identity, "independent_full_read").model_copy(
            update={"content": content}
        )


class _Jobs:
    """Named submission and one provider sample, without recipe knowledge."""

    def __init__(self, provider):
        self.provider = provider

    @cached_property
    def contact(self):
        from bio_neb.providers.nebius_serverless_job_contact import (
            NebiusServerlessJobContact,
        )

        return NebiusServerlessJobContact(
            cli_runner=self.provider._runner,
            authentication=self.provider.authentication,
        )

    @_classified
    def submit(self, name, request):
        from bio_neb.providers.nebius_serverless_job_contact import (
            ProviderReadyJobRequest,
        )
        from bio_neb.providers.runtime import NebiusCliTarget

        p = self.provider
        target = NebiusCliTarget.model_validate(request["target"])
        if (target.project_id, target.tenant_id, target.profile) != (
            p._selection.topology.project_id,
            p._selection.topology.tenant_id,
            p._selection.topology.profile,
        ):
            raise ProviderError("rejected", "job_scope_mismatch")
        ready = ProviderReadyJobRequest(
            target=target,
            provider_job_name=name,
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in request.items()
                if key not in {"target", "image_index_digest", "image_child_digest"}
            },
        )
        credentials = p._credentials()
        if (
            credentials.worker_secret_selector is None
            or credentials.session_token is not None
        ):
            raise ProviderError("rejected", "worker_credentials_unavailable")

        def attest():
            lineage = p._registry(ready.provider_image_reference)
            if (lineage.index_digest, lineage.linux_amd64_child_digest) != (
                request["image_index_digest"],
                request["image_child_digest"],
            ):
                raise ProviderError("rejected", "job_image_drift")

        result = self.contact.create(
            ready,
            environment_bindings=(
                f"AWS_ENDPOINT_URL={p._endpoint}",
                f"AWS_REGION={p._region}",
                f"AWS_ACCESS_KEY_ID={credentials.access_key_id}",
            ),
            secret_environment_bindings=(
                f"AWS_SECRET_ACCESS_KEY={credentials.worker_secret_selector}",
            ),
            reattest_before_create=attest,
            record_create_invoked=lambda: None,
            reuse_named=True,
        )
        return Receipt(
            values={
                "name": name,
                "reused": result.create.disposition == "reused",
                "create": result.create.model_dump(mode="json"),
                "envelope": result.envelope.model_dump(mode="json"),
            },
            resource_count=1,
            provider_ids=(result.provider_job_id,),
        )

    @_classified
    def observe(self, job_id, *, window):
        state, changed, output = self.contact.poll(job_id, window=window)
        return Receipt(
            values={
                "state": state.provider_state,
                "state_changed": changed,
                "details": state.model_dump(mode="json"),
                "log": output,
                "window": window,
            },
            provider_ids=(job_id,),
        )
