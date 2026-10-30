"""Application-owned accepted Ubuntu Nebius CLI identity."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import StrictInt, StrictStr

from bio_neb import locations
from bio_neb.content import canonical_json
from bio_neb.contracts import FrozenContract
from bio_neb.providers.nebius_common import (
    ExecutionContactObserver,
    NebiusCliError,
    NebiusCliInstallIdentity,
    NebiusCliRunner,
)

# The accepted payload is identified by version, byte count and SHA-256 below; its
# location comes from the repository's single location policy.
ACCEPTED_NEBIUS_EXECUTABLE = locations.nebius_executable()
ACCEPTED_NEBIUS_CLI_VERSION = "0.12.263"
ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT = 83_259_554
ACCEPTED_NEBIUS_EXECUTABLE_SHA256 = (
    "8712903f152120e40ec64e96d242e754496e46dffde3b491a8f1a0d4cf67896d"
)


class NativeControllerPreflightReceipt(FrozenContract):
    """Immutable local proof before context or provider contact."""

    schema_version: Literal["bio-neb.native-controller-preflight/1"] = (
        "bio-neb.native-controller-preflight/1"
    )
    controller_runtime: Literal["native_ubuntu"]
    executable: StrictStr
    resolved_executable: StrictStr
    nebius_cli_version: Literal["0.12.263"]
    executable_byte_count: StrictInt
    executable_sha256: StrictStr
    run_root: StrictStr
    filesystem_namespace: Literal["linux_visible_paths_only"] = (
        "linux_visible_paths_only"
    )
    exclusive_create_verified: Literal[True] = True
    hard_link_no_overwrite_verified: Literal[True] = True
    file_fsync_verified: Literal[True] = True
    directory_fsync_verified: Literal[True] = True
    provider_contact_count: Literal[0] = 0
    provider_mutation_count: Literal[0] = 0


class NativeControllerPreflightRefused(ValueError):
    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


def accepted_nebius_cli_identity(executable: str) -> NebiusCliInstallIdentity:
    """Return the APP-owned pin only for its one accepted canonical target."""

    if executable != ACCEPTED_NEBIUS_EXECUTABLE:
        raise NebiusCliError("provider_cli_target_mismatch")
    return NebiusCliInstallIdentity(
        executable=ACCEPTED_NEBIUS_EXECUTABLE,
        resolved_executable=ACCEPTED_NEBIUS_EXECUTABLE,
        version=ACCEPTED_NEBIUS_CLI_VERSION,
        byte_count=ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT,
        sha256=ACCEPTED_NEBIUS_EXECUTABLE_SHA256,
    )


def make_accepted_nebius_cli_runner(
    executable: str,
    *,
    execution_contact_observer: ExecutionContactObserver | None = None,
) -> NebiusCliRunner:
    """Construct the direct process seam around the accepted pin."""

    return NebiusCliRunner(
        expected_identity=accepted_nebius_cli_identity(executable),
        execution_contact_observer=execution_contact_observer,
    )


def _publish_preflight(receipt: NativeControllerPreflightReceipt, target: Path) -> None:
    descriptor: int | None = None
    staging: str | None = None
    directory_descriptor: int | None = None
    linked = False
    try:
        descriptor, staging = tempfile.mkstemp(
            prefix=".bio-neb-native-preflight-", dir=target.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = None
            handle.write(canonical_json(receipt.model_dump(mode="json")))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        linked = True
        os.unlink(staging)
        staging = None
        directory_descriptor = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)
        directory_descriptor = None
    except (OSError, UnicodeError) as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if staging is not None:
            try:
                os.unlink(staging)
            except OSError:
                pass
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        if linked:
            try:
                os.unlink(target)
            except OSError:
                pass
        raise NativeControllerPreflightRefused(
            "native_controller_preflight_write_failed", "receipt_file"
        ) from error


def run_native_controller_preflight(
    *,
    executable: str,
    run_root: str,
    receipt_file: str,
    runner_factory: Callable[[str], Any] = make_accepted_nebius_cli_runner,
) -> NativeControllerPreflightReceipt:
    """Verify the accepted CLI and exact filesystem mechanics without provider contact."""

    root = Path(run_root)
    target = Path(receipt_file)
    if (
        not root.is_absolute()
        or not root.is_dir()
        or not target.is_absolute()
        or target.parent != root
        or target.exists()
    ):
        raise NativeControllerPreflightRefused(
            "native_controller_preflight_paths_invalid", "run_root", "receipt_file"
        )
    try:
        runner = runner_factory(executable)
        identity = runner.preflight()
        probe = root / ".bio-neb-native-filesystem-probe"
        linked = root / ".bio-neb-native-filesystem-linked"
        with probe.open("x", encoding="utf-8", newline="") as handle:
            handle.write("bio-neb-native-filesystem-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(probe, linked)
        try:
            os.link(probe, linked)
        except FileExistsError:
            pass
        else:
            raise OSError("hard-link publication overwrote an existing path")
        if linked.read_text(encoding="utf-8") != "bio-neb-native-filesystem-probe\n":
            raise OSError("filesystem probe bytes changed")
        linked.unlink()
        probe.unlink()
        directory_descriptor = os.open(
            root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except (NebiusCliError, OSError, UnicodeError, ValueError) as error:
        for residue in (
            root / ".bio-neb-native-filesystem-linked",
            root / ".bio-neb-native-filesystem-probe",
        ):
            try:
                residue.unlink()
            except OSError:
                pass
        if isinstance(error, NebiusCliError):
            raise
        raise NativeControllerPreflightRefused(
            "native_controller_filesystem_unsupported", "run_root"
        ) from error
    receipt = NativeControllerPreflightReceipt(
        controller_runtime="native_ubuntu",
        executable=identity.executable,
        resolved_executable=identity.resolved_executable,
        nebius_cli_version=identity.version,
        executable_byte_count=identity.byte_count,
        executable_sha256=identity.sha256,
        run_root=root.as_posix(),
    )
    _publish_preflight(receipt, target)
    return receipt


__all__ = [
    "ACCEPTED_NEBIUS_CLI_VERSION",
    "ACCEPTED_NEBIUS_EXECUTABLE",
    "ACCEPTED_NEBIUS_EXECUTABLE_BYTE_COUNT",
    "ACCEPTED_NEBIUS_EXECUTABLE_SHA256",
    "NativeControllerPreflightReceipt",
    "NativeControllerPreflightRefused",
    "accepted_nebius_cli_identity",
    "make_accepted_nebius_cli_runner",
    "run_native_controller_preflight",
]
