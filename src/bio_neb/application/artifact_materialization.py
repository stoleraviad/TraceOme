"""Shared byte I/O and replica admission; callers retain selection, evidence and policy."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Literal, Protocol

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
)

from bio_neb.application.local_artifact_proof import ReplicaMetadata
from bio_neb.application.local_artifact_proof_adapter import (
    CONTROLLED_DIRECTORY_MODE,
    ControlledDirectoryDrift,
    controlled_directory,
    controlled_replica_metadata,
)
from bio_neb.contracts import FrozenContract
from bio_neb.parabricks.execution_runtime import ExecutionCompositionRuntime
from bio_neb.providers.nebius_common import ExecutionContactObserver
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
)
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt
from bio_neb.verified_artifacts import VerifiedObject

_READ_CHUNK = 8 * 1024 * 1024


class ObjectMaterializationRefused(ValueError):
    def __init__(self, code: str, *details: str) -> None:
        super().__init__(code)
        self.code = code
        self.details = tuple(details)


class ObjectReadResult(FrozenContract):
    """One object downloaded in parts; its streamed digest matched the known one."""

    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    credential_receipt: ObjectStorageCredentialReuseReceipt
    part_bytes: StrictInt = Field(gt=0)
    part_count: StrictInt = Field(gt=0)


class PartObserver(Protocol):
    def opened(self, limits: object) -> None: ...

    def started(
        self, part: int, attempt: int, offset: int, byte_count: int
    ) -> None: ...

    def ended(
        self,
        part: int,
        attempt: int,
        classification: str,
        byte_count: int | None,
        cause: object,
    ) -> None: ...


class ObjectReadPort(Protocol):
    """The sole true-external boundary reachable by this application command."""

    def download(
        self,
        object_key: str,
        destination: BinaryIO,
        *,
        expected_byte_count: int,
        expected_sha256: str,
        observe_part: PartObserver | None = None,
    ) -> object: ...


class MaterializedFile(FrozenContract):
    path: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    independently_full_read: Literal[True] = True
    read_only: StrictBool

    @field_validator("path")
    @classmethod
    def absolute_canonical_path(cls, value: str) -> str:
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("materialized file paths must be absolute and canonical")
        return value


class ObjectMaterializationLineage(FrozenContract):
    journey_id: StrictStr
    journey_file: MaterializedFile
    execution_id: StrictStr = Field(pattern=r"^parabricks-execution:[0-9a-f]{64}$")
    composition_binding_id: StrictStr = Field(
        pattern=r"^(nebius-execution-composition:[0-9a-f]{64}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
    )
    authority_id: StrictStr
    proposal_id: StrictStr
    h100_configuration_id: StrictStr
    h100_request_id: StrictStr
    h100_receipt_id: StrictStr = Field(
        pattern=r"^(h100-execution-receipt|sha256):[0-9a-f]{64}$"
    )
    h100_checkpoint_event_id: StrictStr
    provider_job_id: StrictStr
    selected_context: NebiusExecutionContextSelection
    runtime: ExecutionCompositionRuntime


class MaterializationFiles[FileT: MaterializedFile]:
    """Perform stable reads and durable fresh writes in the caller's file vocabulary.

    The caller chooses the result type and refusal vocabulary once. This owner
    admits fresh replicas under the controlled-replica rule but neither selects
    artifacts nor admits transport responses or publishes evidence.
    """

    def __init__(
        self,
        file_type: type[FileT],
        refused: Callable[..., ObjectMaterializationRefused],
    ) -> None:
        self.file_type = file_type
        self.refused = refused

    def stable_full_read(
        self,
        path: Path,
        *,
        require_read_only: bool,
        capture: bool = False,
        destination: BinaryIO | None = None,
    ) -> tuple[FileT, bytes | None]:
        try:
            resolved = path.resolve(strict=True)
            flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            descriptor = os.open(path, flags)
        except OSError as error:
            raise self.refused("file_unavailable", str(path)) from error
        try:
            before = os.fstat(descriptor)
            if resolved != path or not stat.S_ISREG(before.st_mode):
                raise self.refused("file_invalid", str(path))
            read_only = stat.S_IMODE(before.st_mode) & 0o222 == 0
            if require_read_only and not read_only:
                raise self.refused("candidate_mutable", str(path))
            digest = hashlib.sha256()
            byte_count = 0
            retained = bytearray() if capture else None
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while True:
                    chunk = stream.read(_READ_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if retained is not None:
                        retained.extend(chunk)
                    if destination is not None and destination.write(chunk) != len(
                        chunk
                    ):
                        raise OSError("short local artifact write")
            after = os.fstat(descriptor)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_mode",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise self.refused("file_unstable", str(path))
            if byte_count <= 0 or byte_count != after.st_size:
                raise self.refused("file_invalid", str(path))
            return (
                self.file_type(
                    path=str(path),
                    byte_count=byte_count,
                    sha256=digest.hexdigest(),
                    read_only=read_only,
                ),
                bytes(retained) if retained is not None else None,
            )
        except ObjectMaterializationRefused:
            raise
        except OSError as error:
            raise self.refused("file_unavailable", str(path)) from error
        finally:
            os.close(descriptor)

    def open_source(self, path: Path) -> BinaryIO:
        """Open proved local bytes for one transfer; no stat or custody check.

        A link is not followed; a missing or unreadable file refuses.
        """
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            return os.fdopen(os.open(path, flags), "rb")
        except OSError as error:
            raise self.refused("file_unavailable", str(path)) from error

    def copy_stream(self, source: BinaryIO, destination: BinaryIO) -> int:
        """Stream every byte once and count them: the transfer's own accounting."""
        byte_count = 0
        try:
            while chunk := source.read(_READ_CHUNK):
                if destination.write(chunk) != len(chunk):
                    raise OSError("short local artifact write")
                byte_count += len(chunk)
        except OSError as error:
            raise self.refused("file_unavailable", str(error)) from error
        return byte_count

    def fresh_directory(self, path: Path) -> None:
        """Create a replica's own directory and admit it before any transfer begins."""
        try:
            path.mkdir(mode=CONTROLLED_DIRECTORY_MODE)
        except FileExistsError as error:
            raise self.refused("output_exists") from error
        except OSError as error:
            raise self.refused("output_unavailable") from error
        try:
            controlled_directory(path)
        except ControlledDirectoryDrift as error:
            raise self.refused(
                "output_directory_invalid", str(path), error.observed
            ) from error
        except OSError as error:
            raise self.refused("output_directory_invalid", str(path)) from error

    def seal_replica(self, path: Path) -> ReplicaMetadata:
        """Seal one fresh replica read-only and admit the replica rule, once.

        Its transfer already established its identity; nothing rereads it here.
        """
        try:
            os.chmod(path, 0o444)
            return controlled_replica_metadata(str(path))
        except (ValueError, OSError) as error:
            raise self.refused(
                "output_replica_invalid", str(path), str(error)
            ) from error

    def exclusive_output(self, path: Path) -> BinaryIO:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as error:
            raise self.refused("output_exists", str(path)) from error
        except OSError as error:
            raise self.refused("output_unavailable", str(path)) from error
        return os.fdopen(descriptor, "wb")

    def read_object(
        self,
        path: Path,
        source: ObjectReadPort,
        descriptor: VerifiedObject,
        observe_part: PartObserver | None = None,
    ) -> object:
        """One part download into an exclusive destination, flushed before admission.

        A failed download leaves nothing: its file and its fresh directory are
        removed, so a retry starts from scratch (#363).
        """
        try:
            with self.exclusive_output(path) as output:
                result = source.download(
                    descriptor.object_key,
                    output,
                    expected_byte_count=descriptor.byte_count,
                    expected_sha256=descriptor.sha256,
                    observe_part=observe_part,
                )
                output.flush()
                os.fsync(output.fileno())
        except ObjectMaterializationRefused:
            raise
        except BaseException:
            self.discard(path)
            raise
        return result

    def discard(self, path: Path) -> None:
        """Remove a refused download's file and its own, now empty, directory."""
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass


def execution_object_storage(
    packet,
    context_missing: Exception,
    credential_observer: ExecutionContactObserver | None = None,
):
    """Object-storage mechanics for a retained Execution's own bucket and context.

    ``packet`` is the Run's verified Execution result; its selected context
    resolves the credential, and each lookup is reported to
    ``credential_observer`` (a grant's journal counts them, #418).
    ``context_missing`` is raised when it has none.
    """
    from bio_neb.providers.nebius_object_storage import (
        NebiusObjectStorageMechanics,
    )
    from bio_neb.providers.nebius_object_storage_credentials import (
        SelectedNebiusCredentialResolver,
    )

    selection = packet.selected_context
    if selection is None:
        raise context_missing
    config = packet.storage
    return NebiusObjectStorageMechanics(
        endpoint_url=config.endpoint_url,
        region=config.region,
        bucket=config.bucket,
        credential_resolver=SelectedNebiusCredentialResolver(
            None, selection, execution_contact_observer=credential_observer
        ),
    )
