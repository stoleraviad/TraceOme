"""Stable no-follow full reads for local Adapter evidence."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from typing import Literal

from bio_neb.application.verification_policy import (
    ArtifactVerificationDefaults,
    ArtifactVerificationOverrides,
    ResolvedArtifactVerificationPolicy,
    resolve_artifact_verification_policy,
)
from bio_neb.investigation.vcf_observation import LocalFileObservation


class LocalFileObservationError(ValueError):
    def __init__(
        self, code: Literal["unsafe", "changed", "path_changed"], path: str
    ) -> None:
        super().__init__(code, path)
        self.code = code
        self.path = path


def stable_file_observation(
    path: str, *, retain_bytes: bool = False
) -> tuple[bytes | None, LocalFileObservation]:
    """Observe one regular path through a stable no-follow full read."""
    payload, observation, _metadata = _read_file_observation(
        path, retain_bytes=retain_bytes
    )
    return payload, observation


@dataclass(frozen=True)
class _OwnedFile:
    observation: LocalFileObservation
    payload: bytes | None


class ObservationFiles:
    """Own one full read per admitted path across an observation session.

    A path read once is trusted for the rest of the session (0029): a later need
    reuses that read without another stat, hash or ownership check.
    """

    def __init__(
        self,
        *,
        verification_policy: ResolvedArtifactVerificationPolicy | None = None,
    ) -> None:
        self._files: dict[str, _OwnedFile] = {}
        self._closed = False
        self.verification_policy = (
            verification_policy
            or resolve_artifact_verification_policy(
                ArtifactVerificationDefaults(),
                ArtifactVerificationOverrides(),
            )
        )

    def __enter__(self) -> ObservationFiles:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._files.clear()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def read(
        self, path: str, *, retain_bytes: bool = False
    ) -> tuple[bytes | None, LocalFileObservation]:
        if self._closed:
            raise ValueError("observation files owner is closed")
        owned = self._files.get(path)
        if owned is not None and (not retain_bytes or owned.payload is not None):
            return owned.payload if retain_bytes else None, owned.observation
        payload, observation, metadata = _read_file_observation(
            path, retain_bytes=retain_bytes
        )
        if metadata is not None:
            self._files[path] = _OwnedFile(observation, payload)
        return payload, observation


def _read_file_observation(
    path: str,
    *,
    retain_bytes: bool,
) -> tuple[bytes | None, LocalFileObservation, os.stat_result | None]:
    try:
        path_before = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return (
            None,
            LocalFileObservation(
                path=path,
                exists=False,
                is_regular=False,
                is_symlink=False,
                is_writable=False,
                is_executable=False,
                byte_count=None,
                sha256=None,
                device=None,
                inode=None,
                modified_at_ns=None,
            ),
            None,
        )
    if not stat.S_ISREG(path_before.st_mode) or stat.S_ISLNK(path_before.st_mode):
        raise LocalFileObservationError("unsafe", path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise LocalFileObservationError("unsafe", path) from error
        if isinstance(error, (FileNotFoundError, NotADirectoryError)):
            raise LocalFileObservationError("changed", path) from error
        raise
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    read_count = 0
    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        opened_before = os.fstat(stream.fileno())
        _require_stable(path_before, opened_before, path)
        if not stat.S_ISREG(opened_before.st_mode):
            raise LocalFileObservationError("unsafe", path)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            read_count += len(chunk)
            if retain_bytes:
                chunks.append(chunk)
            digest.update(chunk)
        opened_after = os.fstat(stream.fileno())
    try:
        path_after = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise LocalFileObservationError("changed", path) from error
    _require_stable(opened_before, opened_after, path)
    _require_stable(opened_after, path_after, path)
    if read_count != path_after.st_size:
        raise LocalFileObservationError("changed", path)
    observation = LocalFileObservation(
        path=path,
        exists=True,
        is_regular=True,
        is_symlink=False,
        is_writable=bool(path_after.st_mode & 0o222),
        is_executable=bool(path_after.st_mode & 0o111),
        byte_count=path_after.st_size,
        sha256=digest.hexdigest(),
        device=str(path_after.st_dev),
        inode=str(path_after.st_ino),
        modified_at_ns=path_after.st_mtime_ns,
    )
    return (b"".join(chunks) if retain_bytes else None), observation, path_after


def _identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _require_stable(before: os.stat_result, after: os.stat_result, path: str) -> None:
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise LocalFileObservationError("path_changed", path)
    if _identity(before) != _identity(after):
        raise LocalFileObservationError("changed", path)


__all__ = [
    "LocalFileObservationError",
    "stable_file_observation",
]
