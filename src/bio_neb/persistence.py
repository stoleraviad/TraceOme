"""Exclusive byte persistence mechanics."""

import os
from pathlib import Path


def write_once(path: Path, content: bytes) -> None:
    """Create and fsync exact bytes; the owner supplies directories and error policy."""
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def append_durable(path: Path, content: bytes) -> None:
    """Append exact bytes; schemas, serialization locks and damage policy belong to callers."""
    with path.open("ab") as stream:
        if stream.write(content) != len(content):
            raise OSError("short journal write")
        stream.flush()
        os.fsync(stream.fileno())


def retain_immutable(path: Path, content: bytes) -> None:
    """Retain exact content once; an existing identity must contain identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("immutable content changed")
    else:
        write_once(path, content)
