"""Secure native placement for one compiler-owned worker request.

The caller owns path selection and lifetime.  This module owns only exact local
bytes, the request-specific provider route, and closure of the worker-observed
request identity.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass

from bio_neb.parabricks.worker_request import (
    CompiledWorkerRequest,
    WorkerRequestContractError,
    WorkerRequestIdentity,
)

_PLACED_REQUEST_TOKEN = object()
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+")
_READ_CHUNK_BYTES = 1024 * 1024


class WorkerRequestPlacementError(ValueError):
    """A local placement or request-specific reconciliation refusal."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, init=False)
class PlacedWorkerRequest:
    """One exact compiled request placed at one provider-visible route."""

    compiled: CompiledWorkerRequest
    source_path: str
    container_path: str

    def __new__(cls, token: object = None) -> PlacedWorkerRequest:
        if token is not _PLACED_REQUEST_TOKEN:
            raise TypeError("placed worker requests are created only by placement")
        return super().__new__(cls)

    @classmethod
    def _create(
        cls,
        compiled: CompiledWorkerRequest,
        *,
        source_path: str,
        container_path: str,
    ) -> PlacedWorkerRequest:
        placed = cls(_PLACED_REQUEST_TOKEN)
        object.__setattr__(placed, "compiled", compiled)
        object.__setattr__(placed, "source_path", source_path)
        object.__setattr__(placed, "container_path", container_path)
        return placed

    @property
    def request_identity(self) -> WorkerRequestIdentity:
        return self.compiled.identity

    @property
    def injection_provider_values(self) -> tuple[str, ...]:
        return (f"{self.source_path}:{self.container_path}",)

    @property
    def normalized_arguments(self) -> tuple[str, ...]:
        return (self.container_path,)

    @property
    def injected_paths(self) -> tuple[str, ...]:
        return (self.container_path,)


def _canonical_native_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
        or ":" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise WorkerRequestPlacementError("worker_request_path_invalid")
    components = value[1:].split("/")
    if any(
        component in {"", ".", ".."} or _SAFE_COMPONENT.fullmatch(component) is None
        for component in components
    ):
        raise WorkerRequestPlacementError("worker_request_path_invalid")
    return value


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _metadata(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IMODE(value.st_mode),
    )


def _require_directory(descriptor: int, *, private: bool) -> os.stat_result:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode) or (
        private and stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise OSError("request parent is not an effective private directory")
    return observed


def _open_parent(source_path: str) -> tuple[int, str]:
    components = source_path[1:].split("/")
    leaf = components.pop()
    current: int | None = os.open("/", _directory_flags())
    try:
        _require_directory(current, private=False)
        for component in components:
            created = False
            try:
                child = os.open(component, _directory_flags(), dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(current)
                child = os.open(component, _directory_flags(), dir_fd=current)
            _require_directory(child, private=created)
            os.close(current)
            current = child
        return current, leaf
    except Exception:
        if current is not None:
            os.close(current)
        raise


def _read_exact(descriptor: int, expected: bytes) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    retained = bytearray()
    while len(retained) <= len(expected):
        remaining = len(expected) + 1 - len(retained)
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        retained.extend(chunk)
    return bytes(retained)


def reconcile_fetched_request_route(
    *,
    expected_container_path: str,
    normalized_arguments: tuple[str, ...],
    injected_paths: tuple[str, ...],
) -> None:
    """Reconcile retained or live provider facts through one route rule."""

    if normalized_arguments != (expected_container_path,) or injected_paths != (
        expected_container_path,
    ):
        raise WorkerRequestPlacementError("provider_request_route_drift")


def reconcile_observed_worker_request(
    expected_identity: WorkerRequestIdentity,
    *,
    request_schema: str,
    byte_count: int,
    request_sha256: str,
) -> WorkerRequestIdentity:
    """Reconcile retained or live worker facts through one identity rule."""

    if (
        request_schema != expected_identity.schema_version
        or byte_count != expected_identity.byte_count
        or request_sha256 != expected_identity.sha256
    ):
        raise WorkerRequestPlacementError("worker_request_identity_drift")
    return expected_identity


def _path_stat(parent: int, leaf: str) -> os.stat_result:
    return os.stat(leaf, dir_fd=parent, follow_symlinks=False)


def _require_descriptor_path_identity(
    descriptor_stat: os.stat_result, path_stat: os.stat_result
) -> None:
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
    ):
        raise OSError("request descriptor and source path identity drifted")


class NativeWorkerRequestPlacement:
    """Place, render, and reconcile one native worker-request lifecycle."""

    def place(
        self,
        compiled: CompiledWorkerRequest,
        *,
        source_path: str,
        container_path: str,
    ) -> PlacedWorkerRequest:
        source = _canonical_native_path(source_path)
        container = _canonical_native_path(container_path)
        try:
            if not isinstance(compiled, CompiledWorkerRequest):
                raise WorkerRequestPlacementError("compiled_worker_request_required")
            compiled._require_exact_identity()
            parent, leaf = _open_parent(source)
            try:
                self._place_at(parent, leaf, compiled.request_bytes)
            finally:
                os.close(parent)
        except WorkerRequestPlacementError:
            raise
        except (OSError, TypeError, UnicodeError, WorkerRequestContractError) as error:
            raise WorkerRequestPlacementError(
                "worker_request_placement_failed"
            ) from error
        return PlacedWorkerRequest._create(
            compiled, source_path=source, container_path=container
        )

    @staticmethod
    def _place_at(parent: int, leaf: str, expected: bytes) -> None:
        descriptor: int | None = None
        created = False
        try:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            try:
                descriptor = os.open(leaf, flags, 0o600, dir_fd=parent)
                created = True
            except FileExistsError:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
                descriptor = os.open(leaf, flags, dir_fd=parent)

            before = os.fstat(descriptor)
            before_path = _path_stat(parent, leaf)
            _require_descriptor_path_identity(before, before_path)
            if stat.S_IMODE(before.st_mode) != 0o600:
                raise OSError("request file does not have effective mode 0600")

            if created:
                view = memoryview(expected)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("request write made no progress")
                    view = view[written:]
                os.fsync(descriptor)

            stable_before = os.fstat(descriptor)
            if stable_before.st_size != len(expected):
                raise OSError("request file size differs from compiled bytes")
            retained = _read_exact(descriptor, expected)
            after = os.fstat(descriptor)
            after_path = _path_stat(parent, leaf)
            _require_descriptor_path_identity(after, after_path)
            if (
                _metadata(stable_before) != _metadata(after)
                or _metadata(after) != _metadata(after_path)
                or retained != expected
                or stat.S_IMODE(after.st_mode) != 0o600
            ):
                raise OSError("request file changed during stable readback")

            os.close(descriptor)
            descriptor = None
            if created:
                os.fsync(parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def reconcile_fetched_route(
        placed: PlacedWorkerRequest,
        *,
        normalized_arguments: tuple[str, ...],
        injected_paths: tuple[str, ...],
    ) -> WorkerRequestIdentity:
        reconcile_fetched_request_route(
            expected_container_path=placed.container_path,
            normalized_arguments=normalized_arguments,
            injected_paths=injected_paths,
        )
        return placed.request_identity

    @staticmethod
    def reconcile_worker_identity(
        placed: PlacedWorkerRequest,
        *,
        request_schema: str,
        byte_count: int,
        request_sha256: str,
    ) -> WorkerRequestIdentity:
        return reconcile_observed_worker_request(
            placed.request_identity,
            request_schema=request_schema,
            byte_count=byte_count,
            request_sha256=request_sha256,
        )


def place_execution_request(
    compiled: CompiledWorkerRequest, *, source_path: str, container_path: str
) -> PlacedWorkerRequest:
    """Use native placement with the execution engine's existing failure meaning."""
    from bio_neb.failure import Invariant, ProviderError

    try:
        return NativeWorkerRequestPlacement().place(
            compiled, source_path=source_path, container_path=container_path
        )
    except WorkerRequestPlacementError as error:
        if error.reason == "compiled_worker_request_required" or isinstance(
            error.__cause__, (TypeError, WorkerRequestContractError)
        ):
            raise Invariant("worker_request_invalid", error.reason) from error
        raise ProviderError("rejected", error.reason) from error


__all__ = [
    "NativeWorkerRequestPlacement",
    "PlacedWorkerRequest",
    "place_execution_request",
    "WorkerRequestPlacementError",
    "reconcile_fetched_request_route",
    "reconcile_observed_worker_request",
]
