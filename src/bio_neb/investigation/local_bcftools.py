"""Local shell-free mechanics for deterministic VCF observation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Tuple

from bio_neb.application.local_file_observation import (
    LocalFileObservationError,
    ObservationFiles,
    stable_file_observation,
)
from bio_neb.investigation.vcf_observation import (
    CompletedProcess,
    LocalFileObservation,
)


class LocalBcftoolsAdapter:
    """Hash bound files and execute only argv compiled by the owning Tool."""

    def __init__(self, *, files: ObservationFiles | None = None) -> None:
        self.files = files

    def inspect_file(self, path: str) -> LocalFileObservation:
        try:
            _payload, observation = (
                stable_file_observation(path)
                if self.files is None
                else self.files.read(path)
            )
        except LocalFileObservationError as error:
            message = (
                "VCF observation accepts regular non-symlink files only"
                if error.code == "unsafe"
                else (
                    "VCF observation path changed while hashing"
                    if error.code == "path_changed"
                    else "VCF observation file changed while hashing"
                )
            )
            raise ValueError(message) from error
        return observation

    def path_exists(self, path: str) -> bool:
        return os.path.lexists(path)

    def create_fresh_directory(self, path: str) -> None:
        os.mkdir(path, mode=0o700)

    def write_fresh_file(self, path: str, data: bytes) -> None:
        """Publish ``data`` at ``path`` only once it is whole: written to a fresh file
        beside it, synced, then linked into place, which refuses an existing path. A
        failure leaves nothing at ``path`` (#467 Standards r1 ST1, ADR 0029)."""
        directory, name = os.path.split(path)
        descriptor, staged = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as written:
                written.write(data)
                written.flush()
                os.fsync(written.fileno())
            os.link(staged, path)
        finally:
            os.unlink(staged)

    def run(
        self,
        argv: Tuple[str, ...],
        *,
        timeout_seconds: int,
        stdout_path: str | None = None,
    ) -> CompletedProcess:
        if stdout_path is None:
            completed = subprocess.run(
                list(argv),
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
            return CompletedProcess(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(stdout_path, flags, 0o600)
        try:
            completed = subprocess.run(
                list(argv),
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=descriptor,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        finally:
            os.close(descriptor)
        os.chmod(stdout_path, 0o400)
        return CompletedProcess(
            return_code=completed.returncode, stdout=b"", stderr=completed.stderr
        )


__all__ = ["LocalBcftoolsAdapter"]
