"""Storage readiness effects over shared authentication, bucket and object mechanics."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from bio_neb.failure import ProviderError
from bio_neb.providers.nebius_authentication import NebiusAuthenticationProbe
from bio_neb.providers.nebius_buckets import NebiusBucketAccess
from bio_neb.providers.nebius_common import (
    AwsCredentialMaterial,
    EnvironmentAwsCredentialResolver,
    ExecutionContactObserver,
    NebiusAccessScope,
    NebiusCliError,
    nebius_common_arguments,
)
from bio_neb.providers.nebius_controller_runtime import (
    accepted_nebius_cli_identity,
    make_accepted_nebius_cli_runner,
)
from bio_neb.providers.nebius_object_storage import (
    NebiusObjectStorageError,
    NebiusObjectStorageMechanics,
    ReadContact,
    StorageAccessError,
    StorageMalformedResponse,
    StorageMutationAmbiguous,
    nebius_object_storage_client,
)
from bio_neb.storage_contracts import (
    ExecutionInputStorageAdapterError,
    ExecutionStorageReadinessConfig,
    FreshBucketNameAvailable,
    FreshBucketReceipt,
    InputStorageAuthenticationReceipt,
    InputStorageFailureEvidence,
    InputStorageInventory,
    InputStorageReadinessWriteReceipt,
    InventoryObject,
    MultipartUpload,
    ObjectStorageCredentialReuseReceipt,
)


class NebiusStorageReadinessAdapter:
    """One fresh bucket with no scientific placement or cleanup capability."""

    def __init__(
        self,
        *,
        authentication=None,
        credential_resolver: Callable[[], AwsCredentialMaterial] | None = None,
        cli_runner_factory: Callable[[ExecutionStorageReadinessConfig], Any]
        | None = None,
        s3_client_factory: Callable[
            [str, str, AwsCredentialMaterial], Any
        ] = nebius_object_storage_client,
        contact_observer: Callable[[ReadContact], None] | None = None,
        execution_contact_observer: ExecutionContactObserver | None = None,
    ) -> None:
        self._authentication = authentication
        self._authentication_by_runtime = {}
        self._credentials = credential_resolver or EnvironmentAwsCredentialResolver()
        self._cli_factory = cli_runner_factory or self._default_cli_runner
        self._s3_factory = s3_client_factory
        self._contact_observer = contact_observer
        self._execution_contact_observer = execution_contact_observer

    @staticmethod
    def _default_cli_runner(config: ExecutionStorageReadinessConfig) -> Any:
        target = config.nebius
        expected = accepted_nebius_cli_identity(target.executable)
        if (
            target.resolved_executable != expected.resolved_executable
            or target.cli_version != expected.version
            or target.executable_byte_count != expected.byte_count
            or target.executable_sha256 != expected.sha256
        ):
            raise NebiusCliError("provider_cli_identity_mismatch")
        return make_accepted_nebius_cli_runner(target.executable)

    @staticmethod
    def _scope(config: ExecutionStorageReadinessConfig):
        target = config.nebius
        return NebiusAccessScope(
            target.nebius_cli_profile, target.tenant_id, target.project_id
        )

    @classmethod
    def _common(cls, config: ExecutionStorageReadinessConfig) -> tuple[str, ...]:
        return nebius_common_arguments(cls._scope(config))

    def _buckets(self, config: ExecutionStorageReadinessConfig, runner):
        return NebiusBucketAccess(
            runner,
            common=self._common(config),
            project_id=config.nebius.project_id,
            region=config.object_storage.region,
            domain=config.bucket_domain,
        )

    @staticmethod
    def _fail(
        reason: str,
        phase: str,
        inventory: InputStorageInventory | None = None,
        failure_evidence: InputStorageFailureEvidence | None = None,
    ) -> None:
        raise ExecutionInputStorageAdapterError(  # type: ignore[arg-type]
            reason,
            phase,
            inventory=inventory,
            failure_evidence=failure_evidence,
        )

    def _credential_material(
        self, config: ExecutionStorageReadinessConfig, phase: str
    ) -> AwsCredentialMaterial:
        try:
            credentials = self._credentials()
        except Exception:
            self._fail(
                "authentication_unavailable"
                if phase == "authentication"
                else "authentication_lost",
                phase,
            )
        selector = credentials.worker_secret_selector
        selector_parts = selector.split("@") if selector is not None else []
        if (
            credentials.credentials_profile == config.nebius.nebius_cli_profile
            or selector
            in {
                None,
                credentials.credentials_profile,
                config.nebius.nebius_cli_profile,
            }
            or len(selector_parts) != 2
            or any(not part or part.strip() != part for part in selector_parts)
            or credentials.session_token is not None
        ):
            self._fail(
                "authentication_unavailable"
                if phase == "authentication"
                else "authentication_lost",
                phase,
            )
        return credentials

    def _objects(
        self,
        config: ExecutionStorageReadinessConfig,
        bucket: FreshBucketReceipt,
        phase: str,
    ) -> NebiusObjectStorageMechanics:
        return NebiusObjectStorageMechanics(
            endpoint_url=config.object_storage.endpoint_url,
            region=config.object_storage.region,
            bucket=bucket.name,
            credential_resolver=lambda: self._credential_material(config, phase),
            client_factory=self._s3_factory,
            contact_observer=self._contact_observer,
            execution_contact_observer=self._execution_contact_observer,
        )

    def authenticate(
        self, config: ExecutionStorageReadinessConfig
    ) -> InputStorageAuthenticationReceipt:
        runner = self._cli_factory(config)
        target = config.nebius
        runtime = (
            target.executable,
            target.resolved_executable,
            target.cli_version,
            target.executable_byte_count,
            target.executable_sha256,
        )
        authentication = self._authentication
        if authentication is None:
            if runtime not in self._authentication_by_runtime:
                self._authentication_by_runtime[runtime] = NebiusAuthenticationProbe(
                    cli_runner=runner
                )
            authentication = self._authentication_by_runtime[runtime]
        try:
            profile_kind = authentication.authenticate(self._scope(config))
            self._credential_material(config, "authentication")
            object_storage_credential = (
                ObjectStorageCredentialReuseReceipt.model_validate(
                    getattr(self._credentials, "receipt")  # noqa: B009 - Preserve the existing credential boundary during relocation.
                )
            )
            return InputStorageAuthenticationReceipt(
                nebius_cli_profile=config.nebius.nebius_cli_profile,
                profile_kind=profile_kind,
                tenant_id=config.nebius.tenant_id,
                project_id=config.nebius.project_id,
                control_plane_identity_observed=True,
                object_storage_credential=object_storage_credential,
            )
        except ExecutionInputStorageAdapterError:
            raise
        except (NebiusCliError, KeyError, TypeError, ValueError):
            self._fail("authentication_unavailable", "authentication")

    def _parse_bucket(
        self,
        config: ExecutionStorageReadinessConfig,
        payload: dict[str, Any],
        *,
        reconciled: int,
    ) -> FreshBucketReceipt:
        # Identity was checked by the shared bucket operation before receipt conversion.
        parent_id = payload["metadata"]["parent_id"]
        name = payload["metadata"]["name"]
        bucket_id = payload["metadata"]["id"]
        domain = payload["status"]["domain_name"]
        region = payload["status"]["region"]
        return FreshBucketReceipt(
            bucket_id=bucket_id,
            parent_id=parent_id,
            name=name,
            state="ACTIVE",
            suspended=False,
            domain=domain,
            region=region,
            reconciled_after_ambiguous_create=reconciled,
        )

    def precheck_fresh_bucket(
        self, config: ExecutionStorageReadinessConfig
    ) -> FreshBucketNameAvailable:
        try:
            matches = self._buckets(config, self._cli_factory(config)).find(
                config.bucket_name
            )
        except ExecutionInputStorageAdapterError:
            self._fail("bucket_freshness_unknown", "bucket_freshness")
        except NebiusCliError as error:
            self._fail(
                "authentication_lost"
                if error.provider_reason == "authentication"
                else "bucket_freshness_unknown",
                "bucket_freshness",
            )
        except Exception:
            self._fail("bucket_freshness_unknown", "bucket_freshness")
        if matches:
            self._fail("bucket_name_exists", "bucket_freshness")
        return FreshBucketNameAvailable(bucket_name=config.bucket_name)

    def create_fresh_bucket(
        self,
        config: ExecutionStorageReadinessConfig,
        record_create_invoked: Callable[[], None],
    ) -> FreshBucketReceipt:
        runner = self._cli_factory(config)
        record_create_invoked()
        try:
            created, reconciled = self._buckets(config, runner).create(
                config.bucket_name
            )
            return self._parse_bucket(config, created, reconciled=reconciled)
        except ProviderError as error:
            self._fail(error.detail, "bucket_create")
        except NebiusCliError as error:
            if error.provider_reason == "authentication":
                self._fail("authentication_lost", "bucket_create")
            if error.code == "provider_command_unavailable":
                self._fail("bucket_create_unavailable", "bucket_create")
            self._fail("bucket_create_rejected", "bucket_create")

    def inspect_inventory(
        self,
        config: ExecutionStorageReadinessConfig,
        bucket: FreshBucketReceipt,
    ) -> InputStorageInventory:
        mechanics = self._objects(config, bucket, "inventory_closure")
        try:
            closure = mechanics.closure("")
            return InputStorageInventory(
                contact_complete=1,
                objects=tuple(
                    InventoryObject(
                        object_key=item.object_key, byte_count=item.byte_count
                    )
                    for item in closure.objects
                ),
                multipart_uploads=tuple(
                    MultipartUpload(
                        object_key=item.object_key, upload_id=item.upload_id
                    )
                    for item in closure.multipart
                ),
            )
        except NebiusObjectStorageError as error:
            reason = (
                "authentication_lost"
                if isinstance(error, StorageAccessError)
                else "adapter_output_invalid"
                if isinstance(error, StorageMalformedResponse)
                else "authentication_lost"
            )
            self._fail(
                reason,
                "inventory_closure",
                InputStorageInventory(contact_complete=0),
            )

    @staticmethod
    def _readiness_lifecycle(config: ExecutionStorageReadinessConfig) -> dict[str, Any]:
        identity = hashlib.sha256(
            f"{config.configuration_id}:{config.bucket_name}".encode()
        ).hexdigest()[:24]
        # The disabled rule proves the bucket accepts a lifecycle write. The enabled
        # rule bounds residue from interrupted multipart uploads; run-level accounting
        # still reports any residue that exists while the run is judged.
        return {
            "Rules": [
                {
                    "Expiration": {"Days": 1},
                    "Filter": {},
                    "ID": f"bio-neb-readiness-{identity}",
                    "Status": "Disabled",
                },
                {
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                    "Filter": {},
                    "ID": f"bio-neb-readiness-{identity}-abort-multipart",
                    "Status": "Enabled",
                },
            ]
        }

    def verify_readiness_write(
        self,
        config: ExecutionStorageReadinessConfig,
        bucket: FreshBucketReceipt,
        record_write_claimed: Callable[[], None],
        record_write_invoked: Callable[[], None],
    ) -> InputStorageReadinessWriteReceipt:
        """Persist and read back the lifecycle rules without writing an object."""

        if bucket.name != config.bucket_name:
            self._fail("adapter_output_invalid", "bucket_readiness")
        mechanics = self._objects(config, bucket, "bucket_readiness")
        lifecycle = self._readiness_lifecycle(config)
        try:
            existing = mechanics.get_lifecycle()
        except StorageAccessError:
            self._fail("authentication_lost", "bucket_readiness")
        except NebiusObjectStorageError:
            self._fail("readiness_write_rejected", "bucket_readiness")
        if existing.value is not None:
            self._fail("readiness_write_mismatch", "bucket_readiness")
        record_write_claimed()
        record_write_invoked()
        try:
            written = mechanics.put_lifecycle_once(lifecycle)
        except StorageMutationAmbiguous:
            self._fail("readiness_write_ambiguous", "bucket_readiness")
        except StorageAccessError:
            self._fail("authentication_lost", "bucket_readiness")
        except NebiusObjectStorageError:
            self._fail("readiness_write_rejected", "bucket_readiness")
        if not written.reconciled_after_response_loss:
            try:
                observed = mechanics.get_lifecycle()
            except NebiusObjectStorageError:
                self._fail("readiness_write_ambiguous", "bucket_readiness")
            if observed.value != lifecycle:
                self._fail("readiness_write_mismatch", "bucket_readiness")
        rule = lifecycle["Rules"][0]
        return InputStorageReadinessWriteReceipt(
            lifecycle_rule_id=rule["ID"],
            status="Disabled",
            exact_readback=True,
            reconciled_after_ambiguous_write=int(
                written.reconciled_after_response_loss
            ),
        )
