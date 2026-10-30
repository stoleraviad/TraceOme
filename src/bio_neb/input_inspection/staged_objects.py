"""Metadata-only inspection of registered inputs: local files and staged objects."""

from __future__ import annotations

from typing import Callable, Sequence

from bio_neb.input_inspection.local_files import LocalFilesAdapter, _utc_now
from bio_neb.input_inspection.tool import (
    FilesystemIdentity,
    InputMetadataObservation,
    ObservedInputMetadata,
    RegisteredInputPath,
    is_staged_object,
)

# A staged object's size and version facts, or None when it is absent.
StagedObjectHead = Callable[[str], "tuple[int, str, int] | None"]


class RegisteredInputsAdapter:
    """Stat a local file; HEAD a staged object, reading none of its bytes.

    A staged object fills the same observation: its bucket stands in the device,
    its ETag in the inode and its Last-Modified in the modification time.
    """

    def __init__(
        self,
        staged_head: Callable[[], StagedObjectHead],
        local: LocalFilesAdapter | None = None,
        clock: Callable[[], str] = _utc_now,
    ):
        self._staged_head = staged_head
        self._local = local or LocalFilesAdapter(clock=clock)
        self._clock = clock

    def observe(
        self, requests: Sequence[RegisteredInputPath]
    ) -> InputMetadataObservation:
        ordered = tuple(requests)
        if not ordered:
            raise ValueError("at least one metadata request is required")
        started_at = self._clock()
        head = (
            self._staged_head()
            if any(is_staged_object(item.registered_path) for item in ordered)
            else None
        )
        inputs = tuple(
            self._staged(item, head)
            if is_staged_object(item.registered_path)
            else self._local.observe((item,)).inputs[0]
            for item in ordered
        )
        return InputMetadataObservation(
            started_at=started_at, completed_at=self._clock(), inputs=inputs
        )

    @staticmethod
    def _staged(request: RegisteredInputPath, head) -> ObservedInputMetadata:
        from bio_neb.providers.object_locations import object_address

        observed = head(request.registered_path)
        if observed is None:
            return ObservedInputMetadata(
                role=request.role,
                registered_path=request.registered_path,
                exists=False,
                is_regular=False,
                size_bytes=None,
                filesystem_identity=None,
                modified_at_ns=None,
            )
        byte_count, etag, modified_at_ns = observed
        bucket, _key = object_address(request.registered_path)
        return ObservedInputMetadata(
            role=request.role,
            registered_path=request.registered_path,
            exists=True,
            is_regular=True,
            size_bytes=byte_count,
            filesystem_identity=FilesystemIdentity(device=f"s3://{bucket}", inode=etag),
            modified_at_ns=modified_at_ns,
        )


def registered_inputs_adapter(journey) -> RegisteredInputsAdapter:
    """The Run's inspection adapter; staged objects use its registered storage."""
    return RegisteredInputsAdapter(lambda: _object_storage_head(journey))


def _object_storage_head(journey) -> StagedObjectHead:
    from bio_neb.parabricks.configuration import current_configuration
    from bio_neb.providers.nebius_object_storage import (
        NebiusObjectStorageMechanics,
        StorageReadAbsent,
    )
    from bio_neb.providers.nebius_object_storage_credentials import (
        SelectedNebiusCredentialResolver,
    )
    from bio_neb.providers.object_locations import object_address

    _event, configuration = current_configuration(journey.events_view())
    if configuration is None:
        raise ValueError("staged inputs need the Run's object storage settings")
    launch = configuration.launch
    credentials = SelectedNebiusCredentialResolver(None, launch.context)

    def head(location: str) -> tuple[int, str, int] | None:
        bucket, key = object_address(location)
        try:
            observed = (
                NebiusObjectStorageMechanics(
                    endpoint_url=launch.object_storage.endpoint_url,
                    region=launch.object_storage.region,
                    bucket=bucket,
                    credential_resolver=credentials,
                )
                .observe(key)
                .value
            )
        except StorageReadAbsent:
            return None
        return observed.byte_count, observed.etag, observed.modified_at_ns

    return head
