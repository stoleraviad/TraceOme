"""Contact-free configured-secret scanning for one exact evidence root."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    computed_field,
    model_validator,
)

from bio_neb.contracts import StrictFrozenContract
from bio_neb.journey import canonical_json_digest

_CREDENTIAL_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_AUTH_MATERIAL_MARKERS = (
    b"authorization:",
    b"proxy-authorization:",
    b"bearer ",
    b"cookie:",
    b"set-cookie:",
    b"?api_key=",
    b"&api_key=",
    b"?token=",
    b"&token=",
    b"?access_token=",
    b"&access_token=",
    b"?credential=",
    b"&credential=",
    b"?password=",
    b"&password=",
    b"?secret=",
    b"&secret=",
)


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    byte_count: int
    modified_ns: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileIdentity":
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            byte_count=value.st_size,
            modified_ns=value.st_mtime_ns,
            mode=value.st_mode,
        )


@dataclass(frozen=True)
class _StableFileRead:
    content: bytes
    identity: _FileIdentity


class ConfiguredSecretScannedFile(StrictFrozenContract):
    path: StrictStr
    byte_count: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    matched_transform_names: tuple[StrictStr, ...] = ()


class ConfiguredSecretScanReceipt(StrictFrozenContract):
    schema_version: Literal["bio-neb.configured-secret-scan/1"] = (
        "bio-neb.configured-secret-scan/1"
    )
    scan_kind: Literal["value_aware_contact_free"] = "value_aware_contact_free"
    scanned_root: StrictStr
    credential_names: tuple[StrictStr, ...]
    scanned_file_count: StrictInt = Field(ge=0)
    scanned_files: tuple[ConfiguredSecretScannedFile, ...]
    matching_credential_names: tuple[StrictStr, ...]
    matching_transform_names: tuple[StrictStr, ...]
    matching_value_count: StrictInt = Field(ge=0)
    zero_secret_result: StrictBool
    contacted_external_surface: Literal[False] = False
    secret_values_retained: Literal[False] = False

    @model_validator(mode="after")
    def exact_result(self) -> "ConfiguredSecretScanReceipt":
        if self.scanned_file_count != len(self.scanned_files):
            raise ValueError("scan file count changed manifest cardinality")
        if self.matching_value_count != len(self.matching_credential_names):
            raise ValueError("matching credential count changed")
        if self.zero_secret_result != (self.matching_value_count == 0):
            raise ValueError("zero-secret result contradicts matching values")
        return self

    @computed_field
    @property
    def scan_id(self) -> str:
        material = self.model_dump(mode="json", exclude={"scan_id"})
        return "configured-secret-scan:" + canonical_json_digest(material)


def _exact_directory(path: Path, name: str) -> Path:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve() != path
    ):
        raise ValueError(f"{name} must be an exact absolute non-symlink directory")
    return path


def _credential_values(
    names: Sequence[str], environ: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    exact_names = tuple(names)
    if (
        not exact_names
        or len(exact_names) != len(set(exact_names))
        or any(not _CREDENTIAL_NAME.fullmatch(name) for name in exact_names)
    ):
        raise ValueError("credential names must be unique canonical environment names")
    values = tuple((name, environ.get(name, "")) for name in exact_names)
    if any(not value for _, value in values):
        raise ValueError("configured credential is missing or empty")
    return values


def _credential_representations(value: str) -> tuple[tuple[str, bytes], ...]:
    return (
        ("raw", value.encode("utf-8")),
        (
            "json_escaped",
            json.dumps(value, ensure_ascii=True)[1:-1].encode("utf-8"),
        ),
        ("url_encoded", urllib.parse.quote(value, safe="").encode("ascii")),
        (
            "url_form_encoded",
            urllib.parse.quote_plus(value, safe="").encode("ascii"),
        ),
        ("base64", base64.b64encode(value.encode("utf-8"))),
    )


def _configured_secret_matches(
    content: bytes, credentials: Sequence[tuple[str, str]]
) -> tuple[set[str], set[str]]:
    matching_names: set[str] = set()
    matching_transforms: set[str] = set()
    for name, value in credentials:
        for transform_name, encoded in _credential_representations(value):
            if encoded and encoded in content:
                matching_names.add(name)
                matching_transforms.add(transform_name)
    return matching_names, matching_transforms


def _refuse_configured_secret_in_path(
    path: Path, credentials: Sequence[tuple[str, str]]
) -> None:
    matching_names, _ = _configured_secret_matches(os.fsencode(path), credentials)
    if matching_names:
        raise ValueError("configured secret value detected in evidence path")


def _stable_regular_file_read(path: Path) -> _StableFileRead:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("evidence-root scan requires no-follow file reads")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError(
            "evidence-root scan refuses symlinks or unsafe files"
        ) from None
    try:
        before = _FileIdentity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode):
            raise ValueError("evidence-root scan refuses symlinks or unsafe files")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = _FileIdentity.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if before != after:
        raise ValueError("evidence file changed while scanning")
    return _StableFileRead(content=b"".join(chunks), identity=after)


def scan_configured_secret_values(
    root: Path,
    *,
    credential_names: Sequence[str],
    environ: Mapping[str, str] = os.environ,
) -> ConfiguredSecretScanReceipt:
    exact_root = _exact_directory(root, "scan root")
    credentials = _credential_values(credential_names, environ)
    _refuse_configured_secret_in_path(exact_root, credentials)
    before_entries = tuple(
        sorted(str(path.relative_to(exact_root)) for path in exact_root.rglob("*"))
    )
    matching_names: set[str] = set()
    matching_transforms: set[str] = set()
    scanned_files: list[ConfiguredSecretScannedFile] = []
    final_identities: dict[Path, _FileIdentity] = {}
    for path in sorted(exact_root.rglob("*")):
        _refuse_configured_secret_in_path(path, credentials)
        if path.is_symlink():
            raise ValueError("evidence-root scan refuses symlinks")
        if not path.is_file():
            continue
        stable_read = _stable_regular_file_read(path)
        content = stable_read.content
        final_identities[path] = stable_read.identity
        lowered = content.lower()
        if any(marker in lowered for marker in _AUTH_MATERIAL_MARKERS):
            raise ValueError(
                "authentication header, cookie, bearer, or credential query material detected"
            )
        file_names, file_transforms = _configured_secret_matches(content, credentials)
        matching_names.update(file_names)
        matching_transforms.update(file_transforms)
        scanned_files.append(
            ConfiguredSecretScannedFile(
                path=str(path),
                byte_count=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                matched_transform_names=tuple(sorted(file_transforms)),
            )
        )
    after_entries = tuple(
        sorted(str(path.relative_to(exact_root)) for path in exact_root.rglob("*"))
    )
    if before_entries != after_entries:
        raise ValueError("evidence-root file set changed while scanning")
    for path, identity in final_identities.items():
        current = _FileIdentity.from_stat(os.lstat(path))
        if identity != current or not stat.S_ISREG(current.mode):
            raise ValueError("evidence file changed after its scan read")
    return ConfiguredSecretScanReceipt(
        scanned_root=str(exact_root),
        credential_names=tuple(name for name, _ in credentials),
        scanned_file_count=len(scanned_files),
        scanned_files=tuple(scanned_files),
        matching_credential_names=tuple(sorted(matching_names)),
        matching_transform_names=tuple(sorted(matching_transforms)),
        matching_value_count=len(matching_names),
        zero_secret_result=not matching_names,
    )


def _exact_receipt_path(path: Path, root: Path) -> Path:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.exists()
        or path.is_symlink()
    ):
        raise ValueError("scan receipt must be an unused exact absolute path")
    parent = _exact_directory(path.parent, "scan receipt parent")
    try:
        path.relative_to(root)
    except ValueError:
        return parent / path.name
    raise ValueError("scan receipt must be outside the scanned root")


def run_scan(
    *,
    root: Path,
    receipt_path: Path,
    credential_names: Sequence[str],
    environ: Mapping[str, str] = os.environ,
) -> int:
    exact_root = _exact_directory(root, "scan root")
    exact_receipt = _exact_receipt_path(receipt_path, exact_root)
    receipt = scan_configured_secret_values(
        exact_root, credential_names=credential_names, environ=environ
    )
    encoded = (
        json.dumps(
            receipt.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    with exact_receipt.open("xb") as stream:
        stream.write(encoded)
    print(receipt.model_dump_json())
    if receipt.zero_secret_result:
        return 0
    print(
        "configured_secret_scan_refused:configured secret value detected",
        file=sys.stderr,
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="contact-free configured-secret value scan"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--credential-name", required=True, action="append")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        return run_scan(
            root=arguments.root,
            receipt_path=arguments.receipt,
            credential_names=arguments.credential_name,
        )
    except (OSError, ValueError) as error:
        print(f"configured_secret_scan_refused:{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
