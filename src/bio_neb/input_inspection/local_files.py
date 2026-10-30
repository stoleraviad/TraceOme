"""Local filesystem mechanics for metadata-only FASTQ inspection."""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone
from typing import Callable, Sequence

from bio_neb.input_inspection.tool import (
    FilesystemIdentity,
    InputMetadataObservation,
    ObservedInputMetadata,
    RegisteredInputPath,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LocalFilesAdapter:
    """Observe only filesystem metadata; never open or read a FASTQ path."""

    def __init__(
        self,
        stat_provider: Callable[..., os.stat_result] = os.stat,
        clock: Callable[[], str] = _utc_now,
    ):
        self._stat = stat_provider
        self._clock = clock

    def observe(
        self, requests: Sequence[RegisteredInputPath]
    ) -> InputMetadataObservation:
        ordered = tuple(requests)
        if not ordered:
            raise ValueError("at least one metadata request is required")
        started_at = self._clock()
        inputs = tuple(self._observe(item) for item in ordered)
        return InputMetadataObservation(
            started_at=started_at,
            completed_at=self._clock(),
            inputs=inputs,
        )

    def _observe(self, request: RegisteredInputPath) -> ObservedInputMetadata:
        try:
            observed = self._stat(request.registered_path, follow_symlinks=True)
        except (FileNotFoundError, NotADirectoryError):
            return ObservedInputMetadata(
                role=request.role,
                registered_path=request.registered_path,
                exists=False,
                is_regular=False,
                size_bytes=None,
                filesystem_identity=None,
                modified_at_ns=None,
            )
        return ObservedInputMetadata(
            role=request.role,
            registered_path=request.registered_path,
            exists=True,
            is_regular=stat.S_ISREG(observed.st_mode),
            size_bytes=observed.st_size,
            filesystem_identity=FilesystemIdentity(
                device=str(observed.st_dev), inode=str(observed.st_ino)
            ),
            modified_at_ns=observed.st_mtime_ns,
        )
