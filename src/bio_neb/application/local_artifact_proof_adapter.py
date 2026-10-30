"""Controlled-replica rule: the file and its own directory; metadata alone grants no proof."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from bio_neb.application.local_artifact_proof import ReplicaMetadata

CONTROLLED_DIRECTORY_MODE = 0o700


class ControlledDirectoryDrift(ValueError):
    """The replica's own directory is not a user-owned mode-0700 directory."""

    def __init__(self, observed: str) -> None:
        super().__init__("controlled_replica_directory_drift")
        self.observed = observed


def controlled_directory(path: Path) -> os.stat_result:
    """Admit a replica's own directory: user-owned, mode 0700, not a symlink.

    Ancestors are deliberately unchecked; the own directory alone bounds who can
    replace or relink the replica.
    """
    directory = path.lstat()
    if (
        not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != os.getuid()
        or stat.S_IMODE(directory.st_mode) != CONTROLLED_DIRECTORY_MODE
    ):
        raise ControlledDirectoryDrift(
            f"required user-owned directory mode {oct(CONTROLLED_DIRECTORY_MODE)}; "
            f"observed {stat.filemode(directory.st_mode)} "
            f"({oct(stat.S_IMODE(directory.st_mode))}) owner uid {directory.st_uid}"
        )
    return directory


def controlled_replica_metadata(path: str) -> ReplicaMetadata:
    selected = Path(path)
    observed = selected.lstat()
    if (
        selected.resolve(strict=True) != selected
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_mode & 0o222
        or observed.st_nlink != 1
        or observed.st_uid != os.getuid()
    ):
        raise ValueError("controlled_replica_ownership_drift")
    directory = controlled_directory(selected.parent)
    return ReplicaMetadata(
        device=observed.st_dev,
        inode=observed.st_ino,
        mode=observed.st_mode,
        byte_count=observed.st_size,
        modified_at_ns=observed.st_mtime_ns,
        changed_at_ns=observed.st_ctime_ns,
        owner=observed.st_uid,
        group=observed.st_gid,
        links=observed.st_nlink,
        directory_device=directory.st_dev,
        directory_inode=directory.st_ino,
        directory_mode=directory.st_mode,
        directory_owner=directory.st_uid,
    )
