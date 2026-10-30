"""Native Ubuntu ticket-capsule lifecycle and read-only postmortem helper.

This module deliberately owns local operator mechanics only.  It never interprets a
Journey, contacts or cancels a provider, retries a command, resumes an attempt, or
deletes retained state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from bio_neb import locations

# Fixed Controller roots, bound once at import from the repository's single location
# policy so that tests can patch them and the checks below see canonical paths.
BARE_CACHE = locations.bare_cache()
CHECKOUTS_ROOT = locations.checkouts_root()
# The OS releases the capsule has been accepted on: the original native Ubuntu host and,
# since the Controller runtime moved into the devcontainer, Boxdown's Debian 13
# image.  The recorded identity keeps its "ubuntu" key so retained receipts stay readable.
ACCEPTED_OS_RELEASES = frozenset({("ubuntu", "24.04"), ("debian", "13")})
RUN_ROOTS = locations.run_roots()
CONTROLLER_ENV = locations.controller_env()
MODEL_ENV = locations.model_env()
UV_EXECUTABLE = locations.uv_executable()
# Linux pathname sockets allow 107 encoded bytes plus a terminating NUL. Keep the
# transport independent of retained-root and TMPDIR lengths; command.json owns its
# exact locator. Directories remain private and are never reused or removed here.
_TMUX_SOCKET_ROOT = Path("/tmp")
_UNIX_SOCKET_PATH_MAX_BYTES = 107

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LITERAL_ENV_VALUE_RE = re.compile(rb"[A-Za-z0-9_./:+,@%=-]+")
_GLOB_MARKERS = frozenset("*?[]{}")
_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "TERM",
        "COLORTERM",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
        "UV_CACHE_DIR",
        "UV_PYTHON_DOWNLOADS",
        *locations.OVERRIDES,
        "NO_COLOR",
        "FORCE_COLOR",
    }
)


class OperatorRefused(ValueError):
    """A fail-closed operator boundary refusal with a stable code."""

    def __init__(self, code: str, *fields: str) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_publish(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    """Publish one fsynced JSON record without replacing an existing fact."""

    if path.exists():
        raise OperatorRefused("operator_record_collision", path.name)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    staging = Path(staging_name)
    linked = False
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(_canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, path)
        linked = True
        staging.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        if linked:
            try:
                path.unlink()
            except OSError:
                pass
        raise OperatorRefused("operator_record_publish_failed", path.name) from error
    finally:
        try:
            staging.unlink()
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OperatorRefused("operator_record_invalid", path.name) from error
    if not isinstance(value, dict):
        raise OperatorRefused("operator_record_invalid", path.name)
    return value


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        shell=False,
    )


def _full_sha(value: str, field: str = "expected_sha") -> str:
    if _SHA_RE.fullmatch(value) is None:
        raise OperatorRefused("full_candidate_sha_required", field)
    return value


def _native_path(value: str | Path, field: str) -> Path:
    rendered = str(value)
    pure = PurePosixPath(rendered)
    if (
        not pure.is_absolute()
        or rendered != pure.as_posix()
        or pure == PurePosixPath("/")
        or pure.parts[:2] == ("/", "mnt")
        or any(part in (".", "..") for part in pure.parts)
    ):
        raise OperatorRefused("native_ext4_path_required", field)
    return Path(rendered)


def _require_no_symlink_components(
    path: Path, field: str, *, missing_tail_may_be_created: bool = False
) -> None:
    """Reject symlinks in every existing component without following the leaf."""

    current = Path("/")
    parts = PurePosixPath(str(path)).parts[1:]
    for part in parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if missing_tail_may_be_created:
                return
            raise OperatorRefused("canonical_path_missing", field) from None
        except OSError as error:
            raise OperatorRefused("canonical_path_unreadable", field) from error
        if stat.S_ISLNK(mode):
            raise OperatorRefused("symlink_path_refused", field)


def _mkdir_private_tail(path: Path, field: str) -> None:
    """Create only an absent, previously no-symlink-validated canonical suffix."""

    _require_no_symlink_components(path, field, missing_tail_may_be_created=True)
    current = Path("/")
    for part in PurePosixPath(str(path)).parts[1:]:
        current /= part
        if current.exists():
            continue
        try:
            current.mkdir(mode=0o700)
        except OSError as error:
            raise OperatorRefused("canonical_directory_create_failed", field) from error


def _canonical_existing_path(value: str | Path, field: str) -> Path:
    path = _native_path(value, field)
    _require_no_symlink_components(path, field)
    try:
        if path.resolve(strict=True) != path:
            raise OperatorRefused("canonical_path_required", field)
    except OSError as error:
        raise OperatorRefused("canonical_path_required", field) from error
    return path


def _require_ext4(path: Path, field: str) -> None:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    result = _run(("stat", "-f", "-c", "%T", str(existing)))
    if result.stdout.strip() not in {"ext2/ext3", "ext4"}:
        raise OperatorRefused("native_ext4_path_required", field)


def _git_output(repository: Path, *arguments: str, check: bool = True) -> str:
    return _run(("git", "-C", str(repository), *arguments), check=check).stdout.strip()


def _checkout_identity(checkout: Path, expected_sha: str) -> None:
    try:
        actual = _git_output(checkout, "rev-parse", "HEAD")
        detached = (
            _run(
                ("git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"),
                check=False,
            ).returncode
            != 0
        )
        clean = (
            _git_output(checkout, "status", "--porcelain", "--untracked-files=all")
            == ""
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OperatorRefused("candidate_checkout_invalid", "checkout") from error
    if actual != expected_sha:
        raise OperatorRefused("candidate_checkout_mismatch", "checkout", "expected_sha")
    if not detached:
        raise OperatorRefused("candidate_checkout_not_detached", "checkout")
    if not clean:
        raise OperatorRefused("candidate_checkout_dirty", "checkout")
    if not (checkout / "uv.lock").is_file():
        raise OperatorRefused("candidate_lock_missing", "checkout")


def _referencing_operator_records(
    checkout: Path, run_roots: Path, *, skipped_unreadable: list[Path] | None = None
) -> list[Path]:
    """Run roots whose operator command names `checkout`.

    A root whose record cannot be opened refuses, unless the caller collects it in
    `skipped_unreadable`; a record that opens but does not parse always refuses.
    """
    references: list[Path] = []
    if not run_roots.is_dir():
        return references
    exact = str(checkout)
    for run_root in run_roots.iterdir():
        if not run_root.is_dir() or run_root.is_symlink():
            continue
        command_path = run_root / "operator" / "command.json"
        try:
            command = _read_json(command_path)
        except OperatorRefused as refused:
            if skipped_unreadable is None or not isinstance(
                refused.__cause__, PermissionError
            ):
                raise
            skipped_unreadable.append(run_root)
            continue
        if command is not None and command.get("checkout") == exact:
            references.append(run_root)
    return references


def provision_candidate(
    *,
    bundle: str,
    expected_sha: str,
    named_run_roots: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify and provision one self-contained bundle into an immutable worktree."""

    return _provision_candidate_at(
        bundle=bundle,
        expected_sha=expected_sha,
        bare_cache=BARE_CACHE,
        checkouts_root=CHECKOUTS_ROOT,
        run_roots=RUN_ROOTS,
        named_run_roots=named_run_roots,
    )


def _provision_candidate_at(
    *,
    bundle: str,
    expected_sha: str,
    bare_cache: Path,
    checkouts_root: Path,
    run_roots: Path,
    named_run_roots: Sequence[str] = (),
) -> dict[str, Any]:
    """Test seam for the fixed-root public provisioning operation."""

    sha = _full_sha(expected_sha)
    for name in named_run_roots:
        if _RUN_ID_RE.fullmatch(name) is None:
            raise OperatorRefused("run_id_invalid", "named_run_root")
    bundle_path = Path(bundle)
    if not bundle_path.is_absolute() or not bundle_path.is_file():
        raise OperatorRefused("candidate_bundle_invalid", "bundle")
    cache = _native_path(bare_cache, "bare_cache")
    checkout_root = _native_path(checkouts_root, "checkouts_root")
    roots = _native_path(run_roots, "run_roots")
    _require_no_symlink_components(
        cache.parent, "bare_cache", missing_tail_may_be_created=True
    )
    _require_no_symlink_components(
        checkout_root, "checkouts_root", missing_tail_may_be_created=True
    )
    _require_no_symlink_components(roots, "run_roots")
    _require_ext4(cache, "bare_cache")
    _require_ext4(checkout_root, "checkouts_root")
    target = checkout_root / sha
    _require_no_symlink_components(target, "checkout", missing_tail_may_be_created=True)
    if target.exists() or target.is_symlink():
        raise OperatorRefused("candidate_checkout_collision", "expected_sha")
    # An unreadable root is skipped and reported whatever it references, unless the
    # caller named it: a named root is authoritative and never skipped. A named root
    # that exists must be one the scan reads: a real directory.
    for name in named_run_roots:
        named = roots / name
        if named.is_symlink() or (named.exists() and not named.is_dir()):
            raise OperatorRefused("named_run_root_invalid", name)
    skipped: list[Path] = []
    if _referencing_operator_records(target, roots, skipped_unreadable=skipped):
        raise OperatorRefused("candidate_checkout_referenced", "expected_sha")
    for root in skipped:
        if root.name in named_run_roots:
            raise OperatorRefused("named_run_root_unreadable", root.name)

    _mkdir_private_tail(cache.parent, "bare_cache")
    _mkdir_private_tail(checkout_root, "checkouts_root")
    # Verification against an empty bare repository proves that the bundle does not
    # silently rely on objects already present in the native cache.
    with tempfile.TemporaryDirectory(
        prefix="bio-neb-bundle-verify-", dir=cache.parent
    ) as td:
        verifier = Path(td) / "empty.git"
        _run(("git", "init", "--bare", str(verifier)))
        try:
            _run(("git", "-C", str(verifier), "bundle", "verify", str(bundle_path)))
            heads = _run(("git", "bundle", "list-heads", str(bundle_path))).stdout
        except subprocess.SubprocessError as error:
            raise OperatorRefused(
                "candidate_bundle_not_self_contained", "bundle"
            ) from error
        if not any(line.split(maxsplit=1)[0] == sha for line in heads.splitlines()):
            raise OperatorRefused(
                "candidate_bundle_sha_missing", "bundle", "expected_sha"
            )

    if cache.exists():
        try:
            is_bare = _git_output(cache, "rev-parse", "--is-bare-repository")
        except (OSError, subprocess.SubprocessError) as error:
            raise OperatorRefused("native_bare_cache_invalid", "bare_cache") from error
        if is_bare != "true":
            raise OperatorRefused("native_bare_cache_invalid", "bare_cache")
    else:
        _run(("git", "init", "--bare", str(cache)))
    candidate_ref = f"refs/bio-neb/candidates/{sha}"
    _run(
        (
            "git",
            "-C",
            str(cache),
            "fetch",
            "--no-tags",
            str(bundle_path),
            f"{sha}:{candidate_ref}",
        )
    )
    if _git_output(cache, "rev-parse", f"{candidate_ref}^{{commit}}") != sha:
        raise OperatorRefused("native_cache_candidate_mismatch", "expected_sha")
    _run(("git", "-C", str(cache), "worktree", "add", "--detach", str(target), sha))
    _run((str(UV_EXECUTABLE), "sync", "--locked", "--project", str(target)))
    _run((str(UV_EXECUTABLE), "lock", "--check", "--project", str(target)))
    _checkout_identity(target, sha)
    return {
        "schema_version": "bio-neb.native-operator-provision/1",
        "status": "provisioned",
        "candidate_sha": sha,
        "bundle": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bare_cache": str(cache),
        "checkout": str(target),
        "detached": True,
        "clean": True,
        "locked": True,
        "named_run_roots": list(named_run_roots),
        "skipped_unreadable_run_roots": [str(root) for root in sorted(skipped)],
    }


def allowlisted_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return the exact non-secret environment allowed into the tmux server."""

    selected = {key: value for key, value in source.items() if key in _ENV_ALLOWLIST}
    if "PATH" not in selected:
        selected["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    selected.pop("TMUX", None)
    selected.pop("TMUX_PANE", None)
    return selected


def _version_identity(executable: str, *version_arguments: str) -> dict[str, Any]:
    resolved = Path(executable).resolve(strict=True)
    result = _run((str(resolved), *version_arguments))
    version = (result.stdout or result.stderr).strip()
    if not version:
        raise OperatorRefused("operator_tool_version_missing", resolved.name)
    return {
        "path": str(resolved),
        "version": version,
        "byte_count": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _operator_identity() -> dict[str, Any]:
    """Bind the exact native OS, helper, and executable identities before launch."""

    try:
        release: dict[str, str] = {}
        for raw_line in (
            Path("/etc/os-release").read_text(encoding="utf-8").splitlines()
        ):
            name, separator, raw_value = raw_line.partition("=")
            if separator and name in {"ID", "VERSION_ID", "PRETTY_NAME"}:
                release[name.lower()] = raw_value.strip().strip('"')
        if (release.get("id"), release.get("version_id")) not in ACCEPTED_OS_RELEASES:
            raise OperatorRefused("accepted_os_release_required", "ubuntu")
        kernel = _run(("uname", "-srmo")).stdout.strip()
        helper = Path(__file__).resolve(strict=True)
        tools = {}
        for name, executable, version_arguments in (
            ("python", sys.executable, ("--version",)),
            ("uv", str(UV_EXECUTABLE), ("--version",)),
            ("git", shutil.which("git"), ("--version",)),
            ("tmux", shutil.which("tmux"), ("-V",)),
        ):
            if executable is None:
                raise OperatorRefused("operator_tool_missing", name)
            tools[name] = _version_identity(executable, *version_arguments)
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        raise OperatorRefused(
            "operator_identity_unavailable", "ubuntu", "tools"
        ) from error
    return {
        "ubuntu": {**release, "kernel": kernel},
        "helper": {
            "module": "bio_neb.devtools.native_ubuntu_operator",
            "path": str(helper),
            "byte_count": helper.stat().st_size,
            "sha256": _sha256(helper),
        },
        "tools": tools,
    }


def _pane_tty_identity() -> dict[str, Any]:
    """Prove fd 0 is the pane TTY that the child will inherit unchanged."""

    if not sys.stdin.isatty():
        raise OperatorRefused("tmux_pane_tty_required", "stdin")
    try:
        name = os.ttyname(sys.stdin.fileno())
        identity = os.fstat(sys.stdin.fileno())
    except OSError as error:
        raise OperatorRefused("tmux_pane_tty_required", "stdin") from error
    return {
        "path": name,
        "device": identity.st_dev,
        "inode": identity.st_ino,
        "inherited_by_child": True,
    }


def validate_credential_environment(
    path: str, expected_names: Sequence[str]
) -> dict[str, Any]:
    """Validate a fixed credential file's ownership/modes; return names, never values.

    Only the two fixed files are accepted: ``model.env`` for planning-only Runs and
    ``controller.env`` for Runs that reach storage or execution stages.
    """

    credential_path = _native_path(path, "credential_env")
    if credential_path not in {CONTROLLER_ENV, MODEL_ENV}:
        raise OperatorRefused("credential_environment_path_refused", "credential_env")
    _require_no_symlink_components(credential_path, "credential_env")
    expected = tuple(expected_names)
    if not expected or any(_ENV_NAME_RE.fullmatch(name) is None for name in expected):
        raise OperatorRefused(
            "credential_environment_names_invalid", "credential_names"
        )
    try:
        parent_stat = credential_path.parent.lstat()
        file_stat = credential_path.lstat()
    except OSError as error:
        raise OperatorRefused(
            "credential_environment_missing", "credential_env"
        ) from error
    owner = os.geteuid()
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or parent_stat.st_uid != owner
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise OperatorRefused(
            "credential_environment_parent_mode_refused", "credential_env"
        )
    if (
        stat.S_ISLNK(file_stat.st_mode)
        or not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid != owner
        or stat.S_IMODE(file_stat.st_mode) != 0o600
    ):
        raise OperatorRefused(
            "credential_environment_file_mode_refused", "credential_env"
        )
    literals = _parse_credential_literals(credential_path)
    names = list(literals)
    if any(name not in names for name in expected):
        raise OperatorRefused("credential_environment_name_missing", "credential_names")
    return {
        "path": str(credential_path),
        "expected_names": list(expected),
        "present_names": sorted(names),
        "owner_uid": owner,
        "parent_mode": "0700",
        "file_mode": "0600",
        "values_recorded": False,
    }


def _validate_credential_argv(argv: Sequence[str], credential_path: Path) -> None:
    if len(argv) < 5 or Path(argv[0]).name != "uv" or argv[1] != "run":
        raise OperatorRefused("credential_command_must_use_uv_env_file", "argv")
    if argv.count("--env-file") != 1 or any(
        item.startswith("--env-file=") for item in argv
    ):
        raise OperatorRefused("credential_command_must_use_one_env_file", "argv")
    try:
        index = argv.index("--env-file")
    except ValueError as error:
        raise OperatorRefused(
            "credential_command_must_use_uv_env_file", "argv"
        ) from error
    if index != 2 or index + 1 >= len(argv) or argv[index + 1] != str(credential_path):
        raise OperatorRefused("credential_command_env_file_mismatch", "argv")


def _credential_value_patterns(path: Path) -> tuple[bytes, ...]:
    """Load values only into transient scanner memory; never serialize them."""

    return tuple(
        sorted(set(_parse_credential_literals(path).values()), key=len, reverse=True)
    )


def _parse_credential_literals(path: Path) -> dict[str, bytes]:
    """Parse the exact unquoted literal subset shared with ``uv --env-file``."""

    try:
        document = path.read_bytes()
    except OSError as error:
        raise OperatorRefused(
            "credential_environment_unreadable", "credential_env"
        ) from error
    if b"\r" in document:
        raise OperatorRefused(
            "credential_environment_literal_refused", "credential_env"
        )
    lines = document.split(b"\n")
    if lines[-1:] == [b""]:
        lines.pop()
    literals: dict[str, bytes] = {}
    for line in lines:
        name_bytes, separator, value = line.partition(b"=")
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise OperatorRefused(
                "credential_environment_literal_refused", "credential_env"
            ) from error
        if (
            not separator
            or _ENV_NAME_RE.fullmatch(name) is None
            or _LITERAL_ENV_VALUE_RE.fullmatch(value) is None
        ):
            raise OperatorRefused(
                "credential_environment_literal_refused", "credential_env"
            )
        if name in literals:
            raise OperatorRefused(
                "credential_environment_duplicate_name", "credential_env"
            )
        literals[name] = value
    if not literals:
        raise OperatorRefused(
            "credential_environment_literal_refused", "credential_env"
        )
    return literals


def _parse_proc_stat(document: str) -> dict[str, int]:
    """Parse proc stat fields after the final parenthesized command name."""

    opening = document.find(" (")
    closing = document.rfind(")")
    if opening <= 0 or closing <= opening + 2:
        raise ValueError("malformed proc stat")
    pid = int(document[:opening])
    suffix = document[closing + 1 :].strip().split()
    # suffix begins at field 3 (state); pgrp is field 5 and starttime is 22.
    if len(suffix) < 20 or len(suffix[0]) != 1:
        raise ValueError("malformed proc stat")
    return {"pid": pid, "pgid": int(suffix[2]), "start_ticks": int(suffix[19])}


def _process_identity(pid: int) -> dict[str, int] | None:
    try:
        identity = _parse_proc_stat(
            (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
        )
        return {
            "pid": identity["pid"],
            "pgid": identity["pgid"],
            "start_ticks": identity["start_ticks"],
        }
    except (OSError, UnicodeError, ValueError):
        return None


def _process_identity_matches(started: Mapping[str, Any]) -> bool:
    try:
        expected = {
            "pid": int(started["child_pid"]),
            "pgid": int(started["child_pgid"]),
            "start_ticks": int(started["child_start_ticks"]),
        }
    except (KeyError, TypeError, ValueError):
        return False
    return _process_identity(expected["pid"]) == expected


def _tmux_base(command: Mapping[str, Any]) -> tuple[str, ...]:
    socket = command.get("tmux_socket")
    tools = command.get("tools")
    tmux = tools.get("tmux") if isinstance(tools, Mapping) else None
    executable = tmux.get("path") if isinstance(tmux, Mapping) else None
    if not isinstance(socket, str) or not isinstance(executable, str):
        raise OperatorRefused("operator_record_invalid", "command.json")
    return (executable, "-S", socket)


def _tmux_facts(command: Mapping[str, Any]) -> dict[str, Any]:
    base = _tmux_base(command)
    session = command.get("tmux_session")
    if not isinstance(session, str):
        raise OperatorRefused("operator_record_invalid", "command.json")
    exists = _run((*base, "has-session", "-t", session), check=False).returncode == 0
    if not exists:
        return {
            "server_present": False,
            "session_present": False,
            "viewer_attached": False,
            "pane_dead": None,
            "pane_pid": None,
            "pane_current_command": None,
        }
    result = _run(
        (
            *base,
            "display-message",
            "-p",
            "-t",
            session,
            "#{session_attached}\t#{pane_dead}\t#{pane_pid}\t#{pane_current_command}",
        )
    )
    attached, dead, pane_pid, pane_command = result.stdout.rstrip("\n").split("\t", 3)
    return {
        "server_present": True,
        "session_present": True,
        "viewer_attached": int(attached) > 0,
        "pane_dead": dead == "1",
        "pane_pid": int(pane_pid),
        "pane_current_command": pane_command,
    }


def _operator_paths(run_root: str) -> tuple[Path, Path]:
    root = _canonical_existing_path(run_root, "run_root")
    canonical_roots = _canonical_existing_path(RUN_ROOTS, "run_root")
    if root.parent != canonical_roots:
        raise OperatorRefused("canonical_run_root_required", "run_root")
    operator = _canonical_existing_path(root / "operator", "run_root")
    if not root.is_dir() or not operator.is_dir() or operator.parent != root:
        raise OperatorRefused("operator_run_not_found", "run_root")
    command = _read_json(operator / "command.json")
    if (
        command is None
        or command.get("run_id") != root.name
        or command.get("run_root") != str(root)
        or command.get("operator_directory") != str(operator)
    ):
        raise OperatorRefused("operator_run_identity_mismatch", "run_root", "run_id")
    return root, operator


def start(
    *,
    run_root: str,
    run_id: str,
    checkout: str,
    expected_sha: str,
    argv: Sequence[str],
    cwd: str | None = None,
    credential_env: str | None = None,
    credential_names: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    wait_seconds: float = 15.0,
) -> dict[str, Any]:
    """Reserve one fresh capsule and start its single command in isolated tmux."""

    sha = _full_sha(expected_sha)
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise OperatorRefused("run_id_invalid", "run_id")
    root = _native_path(run_root, "run_root")
    checkout_path = _canonical_existing_path(checkout, "checkout")
    expected_checkout = CHECKOUTS_ROOT / sha
    if checkout_path != expected_checkout:
        raise OperatorRefused("canonical_checkout_required", "checkout", "expected_sha")
    if root.parent != RUN_ROOTS or root.name != run_id:
        raise OperatorRefused("canonical_run_root_required", "run_root", "run_id")
    socket_identity = hashlib.sha256(os.fsencode(root)).hexdigest()[:32]
    socket = (
        _TMUX_SOCKET_ROOT / f"bio-neb-{os.getuid()}-{socket_identity}" / "tmux.sock"
    )
    if len(os.fsencode(socket)) > _UNIX_SOCKET_PATH_MAX_BYTES:
        raise OperatorRefused("tmux_socket_path_too_long", "tmux_socket")
    _require_no_symlink_components(_TMUX_SOCKET_ROOT, "tmux_socket")
    _require_no_symlink_components(RUN_ROOTS, "run_root")
    _require_no_symlink_components(root, "run_root", missing_tail_may_be_created=True)
    workdir = _canonical_existing_path(cwd or checkout, "cwd")
    if not workdir.is_relative_to(checkout_path):
        raise OperatorRefused("command_cwd_outside_checkout", "cwd")
    if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise OperatorRefused("command_argv_invalid", "argv")
    _require_ext4(root, "run_root")
    _require_ext4(checkout_path, "checkout")
    _checkout_identity(checkout_path, sha)
    _run(
        (
            str(UV_EXECUTABLE),
            "lock",
            "--check",
            "--project",
            str(checkout_path),
        )
    )

    credential: dict[str, Any] | None = None
    if credential_env is not None:
        credential = validate_credential_environment(credential_env, credential_names)
        _validate_credential_argv(argv, Path(credential_env))
    elif credential_names:
        raise OperatorRefused("credential_environment_missing", "credential_env")

    identity = _operator_identity()
    try:
        root.mkdir(mode=0o700)
        os.chmod(root, 0o700)
        operator = root / "operator"
        operator.mkdir(mode=0o700)
        os.chmod(operator, 0o700)
        for name in ("stdout.log", "stderr.log"):
            descriptor = os.open(
                operator / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            os.close(descriptor)
    except FileExistsError as error:
        raise OperatorRefused("run_root_collision", "run_root") from error
    except OSError as error:
        raise OperatorRefused("run_root_reservation_failed", "run_root") from error

    try:
        socket.parent.mkdir(mode=0o700)
        os.chmod(socket.parent, 0o700)
    except FileExistsError as error:
        raise OperatorRefused(
            "tmux_socket_directory_collision", "tmux_socket"
        ) from error
    except OSError as error:
        raise OperatorRefused(
            "tmux_socket_reservation_failed", "tmux_socket"
        ) from error

    clean_environment = allowlisted_environment(
        os.environ if environ is None else environ
    )
    # tmux normalizes periods in session names, but Run IDs retain their spelling.
    session = f"bio-neb-{socket_identity}"
    command = {
        "schema_version": "bio-neb.native-operator-command/1",
        "run_id": run_id,
        "run_root": str(root),
        "operator_directory": str(operator),
        "candidate_sha": sha,
        "checkout": str(checkout_path),
        "cwd": str(workdir),
        "argv": list(argv),
        "argv_preserved": True,
        "shell": False,
        "tmux_socket": str(socket),
        "tmux_session": session,
        "environment_names": sorted(clean_environment),
        "credential_environment": credential,
        "ubuntu": identity["ubuntu"],
        "helper": identity["helper"],
        "tools": identity["tools"],
        "created_at": _utc_now(),
        "authority": "operator_only_non_authoritative",
    }
    _atomic_publish(operator / "command.json", command)
    internal_argv = (
        sys.executable,
        "-m",
        "bio_neb.devtools.native_ubuntu_operator",
        "_run-child",
        "--run-root",
        str(root),
    )
    tmux_executable = str(identity["tools"]["tmux"]["path"])
    base = (tmux_executable, "-f", "/dev/null", "-S", str(socket))
    try:
        _run(
            (
                *base,
                "new-session",
                "-d",
                "-s",
                session,
                "-x",
                "160",
                "-y",
                "48",
                *internal_argv,
            ),
            cwd=checkout_path,
            env=clean_environment,
        )
        _run(
            (*base, "set-option", "-t", session, "remain-on-exit", "on"),
            env=clean_environment,
        )
        _run(
            (*base, "set-option", "-t", session, "history-limit", "50000"),
            env=clean_environment,
        )
        _atomic_publish(operator / "launch-ready.json", {"ready_at": _utc_now()})
    except (OSError, subprocess.SubprocessError, OperatorRefused) as error:
        raise OperatorRefused("tmux_start_failed", "run_root") from error

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        started = _read_json(operator / "started.json")
        if started is not None:
            return {"command": command, "started": started}
        if not _tmux_facts(command)["session_present"]:
            break
        time.sleep(0.05)
    raise OperatorRefused("child_start_not_observed", "run_root")


class _SecretOutputRefused(RuntimeError):
    pass


class _StreamGuard:
    """Hold only a possible credential prefix across writes, never safe prompts."""

    def __init__(self, patterns: Sequence[bytes]) -> None:
        self._patterns = tuple(patterns)
        self._pending = b""
        self._hold = max((len(pattern) - 1 for pattern in self._patterns), default=0)

    def feed(self, block: bytes, *, final: bool = False) -> bytes:
        combined = self._pending + block
        if any(pattern in combined for pattern in self._patterns):
            self._pending = b""
            raise _SecretOutputRefused
        hold = 0
        if not final:
            for length in range(min(self._hold, len(combined)), 0, -1):
                if any(
                    pattern.startswith(combined[-length:]) for pattern in self._patterns
                ):
                    hold = length
                    break
        emitted = combined[:-hold] if hold else combined
        self._pending = combined[-hold:] if hold else b""
        return emitted


def _tee(
    source: Any,
    targets: Sequence[Any],
    guard: _StreamGuard,
    errors: list[str],
    refused: threading.Event,
) -> None:
    while True:
        block = source.read(64 * 1024)
        if not block:
            try:
                safe = guard.feed(b"", final=True)
            except _SecretOutputRefused:
                errors.append("credential_value_in_child_output")
                refused.set()
                return
            if safe:
                for target in targets:
                    target.write(safe)
                    target.flush()
            return
        try:
            safe = guard.feed(block)
        except _SecretOutputRefused:
            errors.append("credential_value_in_child_output")
            refused.set()
            return
        if safe:
            for target in targets:
                target.write(safe)
                target.flush()


def _process_group_members(process_group: int) -> list[dict[str, int]]:
    members: list[dict[str, int]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise OperatorRefused("process_group_inventory_unavailable", "child_pgid")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            document = (entry / "stat").read_text(encoding="ascii")
        except (OSError, UnicodeError):
            continue
        try:
            identity = _parse_proc_stat(document)
        except ValueError as error:
            raise OperatorRefused(
                "process_group_inventory_malformed", "child_pgid"
            ) from error
        if identity["pgid"] == process_group:
            members.append(identity)
    return members


def _frozen_payload(
    *,
    command: Mapping[str, Any],
    operator_directory: Path,
    reason: str,
    residue: Sequence[Mapping[str, int]] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "bio-neb.native-operator-frozen/1",
        "frozen_at": _utc_now(),
        "reason": reason,
        "candidate_sha": command["candidate_sha"],
        "checkout": command["checkout"],
        "residue": [dict(item) for item in residue],
        "stdout": {
            "path": str(operator_directory / "stdout.log"),
            "byte_count": (operator_directory / "stdout.log").stat().st_size,
            "sha256": _sha256(operator_directory / "stdout.log"),
        },
        "stderr": {
            "path": str(operator_directory / "stderr.log"),
            "byte_count": (operator_directory / "stderr.log").stat().st_size,
            "sha256": _sha256(operator_directory / "stderr.log"),
        },
        "command_sha256": _sha256(operator_directory / "command.json"),
        "started_sha256": _sha256(operator_directory / "started.json"),
        "automatic_cleanup_performed": False,
        "provider_cancellation_inferred": False,
    }
    payload["integrity_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _started_record(
    *,
    command: Mapping[str, Any],
    operator_directory: Path,
    identity: Mapping[str, int],
    pane_tty: Mapping[str, Any],
) -> dict[str, Any]:
    argv = command["argv"]
    return {
        "schema_version": "bio-neb.native-operator-started/1",
        "started_at": _utc_now(),
        "candidate_sha": command["candidate_sha"],
        "checkout": command["checkout"],
        "cwd": command["cwd"],
        "argv_sha256": hashlib.sha256(_canonical({"argv": argv})).hexdigest(),
        "supervisor_pid": os.getpid(),
        "child_pid": identity["pid"],
        "child_pgid": identity["pgid"],
        "child_start_ticks": identity["start_ticks"],
        "tmux_socket": command["tmux_socket"],
        "tmux_session": command["tmux_session"],
        "credential_values_recorded": False,
        "child_stdin": dict(pane_tty),
        "child_stdout": {
            "transport": "pipe_to_distinct_log_and_pane_stdout",
            "path": str(operator_directory / "stdout.log"),
        },
        "child_stderr": {
            "transport": "pipe_to_distinct_log_and_pane_stderr",
            "path": str(operator_directory / "stderr.log"),
        },
    }


def _set_foreground_process_group(process_group: int) -> None:
    if not sys.stdin.isatty():
        return
    previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    try:
        os.tcsetpgrp(sys.stdin.fileno(), process_group)
    except OSError:
        pass
    finally:
        signal.signal(signal.SIGTTOU, previous)


def _run_child(run_root: str) -> int:
    """Internal tmux pane process.  Never invoke directly as an operator action."""

    root, operator = _operator_paths(run_root)
    command = _read_json(operator / "command.json")
    if command is None:
        raise OperatorRefused("operator_record_invalid", "command.json")
    ready = operator / "launch-ready.json"
    for _ in range(600):
        if ready.is_file():
            break
        time.sleep(0.05)
    else:
        raise OperatorRefused("tmux_launch_not_released", "run_root")

    argv = command.get("argv")
    cwd = command.get("cwd")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise OperatorRefused("operator_record_invalid", "command.json")
    if not isinstance(cwd, str):
        raise OperatorRefused("operator_record_invalid", "command.json")
    pane_tty = _pane_tty_identity()
    credential = command.get("credential_environment")
    patterns: tuple[bytes, ...] = ()
    if credential is not None:
        if not isinstance(credential, Mapping) or not isinstance(
            credential.get("path"), str
        ):
            raise OperatorRefused("operator_record_invalid", "command.json")
        patterns = _credential_value_patterns(Path(str(credential["path"])))

    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    gate_read, gate_write = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(stdout_read)
            os.close(stderr_read)
            os.close(gate_write)
            os.setpgid(0, 0)
            os.dup2(stdout_write, 1)
            os.dup2(stderr_write, 2)
            os.close(stdout_write)
            os.close(stderr_write)
            if os.read(gate_read, 1) != b"1":
                os._exit(126)
            os.close(gate_read)
            os.execvpe(argv[0], argv, dict(os.environ))
        except BaseException:
            os._exit(127)

    os.close(stdout_write)
    os.close(stderr_write)
    os.close(gate_read)
    os.setpgid(child, child)
    identity = _process_identity(child)
    if identity is None:
        os.close(gate_write)
        raise OperatorRefused("child_identity_unavailable", "run_root")
    _set_foreground_process_group(child)
    started = _started_record(
        command=command,
        operator_directory=operator,
        identity=identity,
        pane_tty=pane_tty,
    )
    _atomic_publish(operator / "started.json", started)

    with (
        os.fdopen(stdout_read, "rb", buffering=0) as child_stdout,
        os.fdopen(stderr_read, "rb", buffering=0) as child_stderr,
        (operator / "stdout.log").open("ab", buffering=0) as stdout_log,
        (operator / "stderr.log").open("ab", buffering=0) as stderr_log,
    ):
        visible_stdout = getattr(sys.stdout, "buffer", sys.stdout)
        visible_stderr = getattr(sys.stderr, "buffer", sys.stderr)
        stream_errors: list[str] = []
        output_refused = threading.Event()
        stdout_thread = threading.Thread(
            target=_tee,
            args=(
                child_stdout,
                (stdout_log, visible_stdout),
                _StreamGuard(patterns),
                stream_errors,
                output_refused,
            ),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_tee,
            args=(
                child_stderr,
                (stderr_log, visible_stderr),
                _StreamGuard(patterns),
                stream_errors,
                output_refused,
            ),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        os.write(gate_write, b"1")
        os.close(gate_write)
        while True:
            reaped, wait_status = os.waitpid(child, os.WNOHANG)
            if reaped == child:
                break
            if output_refused.is_set():
                try:
                    os.killpg(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, wait_status = os.waitpid(child, 0)
                break
            time.sleep(0.01)
        stdout_thread.join()
        stderr_thread.join()
        stdout_log.flush()
        stderr_log.flush()
        os.fsync(stdout_log.fileno())
        os.fsync(stderr_log.fileno())
    _set_foreground_process_group(os.getpgrp())

    if stream_errors:
        frozen = _frozen_payload(
            command=command,
            operator_directory=operator,
            reason="credential_value_in_child_output",
        )
        _atomic_publish(operator / "frozen.json", frozen)
        return 125

    residue = _process_group_members(identity["pgid"])
    if residue:
        frozen = _frozen_payload(
            command=command,
            operator_directory=operator,
            reason="owned_process_group_residue",
            residue=residue,
        )
        _atomic_publish(operator / "frozen.json", frozen)
        return 125

    exit_code = os.waitstatus_to_exitcode(wait_status)
    terminal_payload: dict[str, Any] = {
        "schema_version": "bio-neb.native-operator-finished/1",
        "finished_at": _utc_now(),
        "candidate_sha": command["candidate_sha"],
        "child_pid": child,
        "child_pgid": identity["pgid"],
        "exit_code": exit_code if exit_code >= 0 else None,
        "signal": -exit_code if exit_code < 0 else None,
        "stdout": {
            "path": str(operator / "stdout.log"),
            "byte_count": (operator / "stdout.log").stat().st_size,
            "sha256": _sha256(operator / "stdout.log"),
        },
        "stderr": {
            "path": str(operator / "stderr.log"),
            "byte_count": (operator / "stderr.log").stat().st_size,
            "sha256": _sha256(operator / "stderr.log"),
        },
        "command_sha256": _sha256(operator / "command.json"),
        "started_sha256": _sha256(operator / "started.json"),
        "application_or_provider_success_interpreted": False,
    }
    terminal_payload["integrity_sha256"] = hashlib.sha256(
        _canonical(terminal_payload)
    ).hexdigest()
    _atomic_publish(operator / "finished.json", terminal_payload)
    return exit_code if exit_code >= 0 else 128 + (-exit_code)


def status(run_root: str) -> dict[str, Any]:
    """Observe local lifecycle facts without interpreting application success."""

    root, operator = _operator_paths(run_root)
    command = _read_json(operator / "command.json")
    if command is None:
        raise OperatorRefused("operator_record_invalid", "command.json")
    started = _read_json(operator / "started.json")
    finished = _read_json(operator / "finished.json")
    frozen = _read_json(operator / "frozen.json")
    tmux = _tmux_facts(command)
    identity_matches = started is not None and _process_identity_matches(started)
    candidate_sha = command.get("candidate_sha")
    finished_valid = isinstance(candidate_sha, str) and _local_terminal_record_valid(
        finished,
        schema="bio-neb.native-operator-finished/1",
        operator=operator,
        candidate_sha=candidate_sha,
    )
    frozen_valid = isinstance(candidate_sha, str) and _local_terminal_record_valid(
        frozen,
        schema="bio-neb.native-operator-frozen/1",
        operator=operator,
        candidate_sha=candidate_sha,
    )
    if frozen_valid:
        lifecycle = "frozen_abnormal"
    elif finished_valid:
        lifecycle = "finished"
    elif started is None:
        lifecycle = "reserved_not_started"
    elif tmux["session_present"] and tmux["pane_dead"] is False and identity_matches:
        lifecycle = "running"
    else:
        lifecycle = "abnormal_local_termination"
    pointer = _read_json(operator / "receipt-pointer.json")
    return {
        "schema_version": "bio-neb.native-operator-status/1",
        "run_root": str(root),
        "checkout": command.get("checkout"),
        "candidate_sha": command.get("candidate_sha"),
        "viewer": {
            "attached": tmux["viewer_attached"],
            "server_present": tmux["server_present"],
            "session_present": tmux["session_present"],
        },
        "local_command": {
            "lifecycle": lifecycle,
            "process_identity_matches": identity_matches,
            "pane_dead": tmux["pane_dead"],
            "pane_pid": tmux["pane_pid"],
            "pane_current_command": tmux["pane_current_command"],
            "started": started,
            "finished": finished,
            "finished_valid": finished_valid,
            "frozen": frozen,
            "frozen_valid": frozen_valid,
        },
        "application_remote_evidence": {
            "interpreted": False,
            "receipt_pointer_present": pointer is not None,
            "receipt_pointer": pointer,
        },
    }


def attach(run_root: str) -> int:
    """Attach a viewer; this does not start or resume a command."""

    _, operator = _operator_paths(run_root)
    command = _read_json(operator / "command.json")
    if command is None or not _tmux_facts(command)["session_present"]:
        raise OperatorRefused("tmux_session_missing", "run_root")
    completed = subprocess.run(
        [*_tmux_base(command), "attach-session", "-t", str(command["tmux_session"])],
        env=allowlisted_environment(os.environ),
        shell=False,
        check=False,
    )
    return completed.returncode


def interrupt(run_root: str) -> dict[str, Any]:
    """Record and deliver SIGINT only to the exact still-live child process group."""

    _, operator = _operator_paths(run_root)
    command = _read_json(operator / "command.json")
    started = _read_json(operator / "started.json")
    finished = _read_json(operator / "finished.json")
    if command is None or started is None:
        raise OperatorRefused("interrupt_not_started", "run_root")
    if finished is not None:
        raise OperatorRefused("interrupt_already_finished", "run_root")
    if not _process_identity_matches(started):
        raise OperatorRefused("interrupt_stale_process_identity", "run_root")
    if started.get("child_pgid") != started.get("child_pid"):
        raise OperatorRefused("interrupt_process_group_mismatch", "run_root")
    request = {
        "schema_version": "bio-neb.native-operator-interruption/1",
        "requested_at": _utc_now(),
        "signal": "SIGINT",
        "signal_number": signal.SIGINT,
        "child_pid": started["child_pid"],
        "child_pgid": started["child_pgid"],
        "child_start_ticks": started["child_start_ticks"],
        "provider_cancellation_inferred": False,
        "tmux_server_killed": False,
    }
    _atomic_publish(operator / "interruption.json", request)
    try:
        os.killpg(int(started["child_pgid"]), signal.SIGINT)
    except (OSError, ValueError) as error:
        raise OperatorRefused("interrupt_delivery_failed", "run_root") from error
    return request


def _commit_reachable(repository: Path, sha: str) -> bool:
    return (
        _run(
            ("git", "-C", str(repository), "cat-file", "-e", f"{sha}^{{commit}}"),
            check=False,
        ).returncode
        == 0
    )


def _commit_ancestor(repository: Path, candidate_sha: str, current_sha: str) -> bool:
    return (
        _run(
            (
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                candidate_sha,
                current_sha,
            ),
            check=False,
        ).returncode
        == 0
    )


def _receipt_pointer_valid(operator: Path, candidate_sha: str) -> bool:
    pointer = _read_json(operator / "receipt-pointer.json")
    if (
        pointer is None
        or pointer.get("schema_version") != "bio-neb.native-operator-receipt-pointer/1"
        or pointer.get("candidate_sha") != candidate_sha
    ):
        return False
    path = pointer.get("receipt_path")
    digest = pointer.get("receipt_sha256")
    if (
        not isinstance(path, str)
        or not isinstance(digest, str)
        or _DIGEST_RE.fullmatch(digest) is None
    ):
        return False
    run_root = operator.parent
    try:
        receipt = _canonical_existing_path(path, "receipt_path")
        receipt_mode = receipt.lstat().st_mode
        protected = receipt.resolve(strict=True).is_relative_to(
            run_root.resolve(strict=True)
        )
        receipt_document = _read_json(receipt)
    except (OSError, RuntimeError, OperatorRefused):
        return False
    if not (
        receipt.is_absolute()
        and stat.S_ISREG(receipt_mode)
        and not stat.S_ISLNK(receipt_mode)
        and protected
        and _sha256(receipt) == digest
        and receipt_document is not None
        and receipt_document.get("schema_version")
        == "bio-neb.native-operator-receipt/1"
        and receipt_document.get("candidate_sha") == candidate_sha
        and receipt_document.get("run_root") == str(run_root)
    ):
        return False
    terminal_path = receipt_document.get("local_terminal_path")
    terminal_digest = receipt_document.get("local_terminal_sha256")
    evidence_path = receipt_document.get("evidence_path")
    evidence_digest = receipt_document.get("evidence_sha256")
    if not all(
        isinstance(value, str)
        for value in (terminal_path, terminal_digest, evidence_path, evidence_digest)
    ):
        return False
    expected_terminal_paths = {
        str(operator / "finished.json"),
        str(operator / "frozen.json"),
    }
    try:
        evidence = _canonical_existing_path(str(evidence_path), "evidence_path")
        evidence_protected = evidence.resolve(strict=True).is_relative_to(
            run_root.resolve(strict=True)
        )
    except (OSError, RuntimeError, OperatorRefused):
        return False
    return (
        terminal_path in expected_terminal_paths
        and _DIGEST_RE.fullmatch(str(terminal_digest)) is not None
        and Path(str(terminal_path)).is_file()
        and _sha256(Path(str(terminal_path))) == terminal_digest
        and evidence_protected
        and _DIGEST_RE.fullmatch(str(evidence_digest)) is not None
        and _sha256(evidence) == evidence_digest
    )


def _integrity_valid(record: Mapping[str, Any]) -> bool:
    integrity = record.get("integrity_sha256")
    if not isinstance(integrity, str) or _DIGEST_RE.fullmatch(integrity) is None:
        return False
    body = dict(record)
    body.pop("integrity_sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest() == integrity


def _local_terminal_record_valid(
    record: Mapping[str, Any] | None,
    *,
    schema: str,
    operator: Path,
    candidate_sha: str,
) -> bool:
    try:
        command_sha256 = _sha256(operator / "command.json")
        started_sha256 = _sha256(operator / "started.json")
    except OSError:
        return False
    if (
        record is None
        or record.get("schema_version") != schema
        or record.get("candidate_sha") != candidate_sha
        or not _integrity_valid(record)
        or record.get("command_sha256") != command_sha256
        or record.get("started_sha256") != started_sha256
    ):
        return False
    for name in ("stdout", "stderr"):
        stream = record.get(name)
        path = operator / f"{name}.log"
        if not isinstance(stream, Mapping) or stream.get("path") != str(path):
            return False
        try:
            mode = path.lstat().st_mode
            byte_count = path.stat().st_size
        except OSError:
            return False
        if (
            not stat.S_ISREG(mode)
            or stat.S_ISLNK(mode)
            or stream.get("byte_count") != byte_count
            or stream.get("sha256") != _sha256(path)
        ):
            return False
    if schema.endswith("finished/1"):
        exit_code = record.get("exit_code")
        signal_number = record.get("signal")
        if (isinstance(exit_code, int)) == (isinstance(signal_number, int)):
            return False
    elif not isinstance(record.get("reason"), str):
        return False
    return True


def _started_record_valid(
    record: Mapping[str, Any] | None, *, candidate_sha: str, checkout: Path
) -> bool:
    if (
        record is None
        or record.get("schema_version") != "bio-neb.native-operator-started/1"
        or record.get("candidate_sha") != candidate_sha
        or record.get("checkout") != str(checkout)
    ):
        return False
    for name in ("child_pid", "child_pgid", "child_start_ticks"):
        if not isinstance(record.get(name), int):
            return False
    return record.get("child_pid") == record.get("child_pgid")


def _processes_referencing(checkout: Path) -> list[int]:
    references: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return references
    target = checkout.resolve()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        referenced = False
        for locator in ("cwd", "exe", "root"):
            try:
                resolved = (entry / locator).resolve(strict=True)
                if resolved == target or resolved.is_relative_to(target):
                    referenced = True
            except (OSError, RuntimeError):
                pass
        try:
            for descriptor in (entry / "fd").iterdir():
                resolved = descriptor.resolve(strict=True)
                if resolved == target or resolved.is_relative_to(target):
                    referenced = True
                    break
        except (OSError, RuntimeError):
            pass
        try:
            marker = str(target).encode("utf-8")
            if (
                marker in (entry / "cmdline").read_bytes()
                or marker in (entry / "maps").read_bytes()
            ):
                referenced = True
        except OSError:
            pass
        if referenced:
            references.append(int(entry.name))
    return references


def require_exact_cleanup_target(
    *, target: str, checkouts_root: str, expected_sha: str
) -> Path:
    """Validate a literal candidate target.  This function never removes it."""

    sha = _full_sha(expected_sha)
    if any(character in target for character in _GLOB_MARKERS):
        raise OperatorRefused("bulk_cleanup_refused", "target")
    root = _native_path(checkouts_root, "checkouts_root")
    candidate = _native_path(target, "target")
    if candidate != root / sha or candidate.name != sha:
        raise OperatorRefused("cleanup_exact_target_required", "target", "expected_sha")
    if candidate.is_symlink():
        raise OperatorRefused("cleanup_symlink_refused", "target")
    return candidate


def cleanup_eligibility(
    *,
    target: str,
    expected_sha: str,
    preserved_checkout: str | None = None,
    preserved_sha: str | None = None,
) -> dict[str, Any]:
    """Report eligibility only against the fixed native Controller roots."""

    return _cleanup_eligibility_at(
        target=target,
        expected_sha=expected_sha,
        preserved_checkout=preserved_checkout,
        preserved_sha=preserved_sha,
        bare_cache=BARE_CACHE,
        checkouts_root=CHECKOUTS_ROOT,
        run_roots=RUN_ROOTS,
    )


def _cleanup_eligibility_at(
    *,
    target: str,
    expected_sha: str,
    preserved_checkout: str | None = None,
    preserved_sha: str | None = None,
    bare_cache: str = str(BARE_CACHE),
    checkouts_root: str = str(CHECKOUTS_ROOT),
    run_roots: str = str(RUN_ROOTS),
) -> dict[str, Any]:
    """Report every fixed native Controller cleanup prerequisite, read-only."""

    sha = _full_sha(expected_sha)
    candidate = require_exact_cleanup_target(
        target=target, checkouts_root=checkouts_root, expected_sha=sha
    )
    reasons: list[str] = []
    if preserved_checkout is None or preserved_sha is None:
        reasons.append("cleanup_preservation_boundary_unknown")
    else:
        preserved_candidate = require_exact_cleanup_target(
            target=preserved_checkout,
            checkouts_root=checkouts_root,
            expected_sha=preserved_sha,
        )
        if candidate == preserved_candidate or sha == preserved_sha:
            reasons.append("current_checkout_preserved")
    if not candidate.is_dir():
        reasons.append("checkout_missing")
    else:
        try:
            _checkout_identity(candidate, sha)
        except OperatorRefused as error:
            reasons.append(error.code)
    cache = _native_path(bare_cache, "bare_cache")
    if not cache.is_dir() or not _commit_reachable(cache, sha):
        reasons.append("commit_unreachable_native_bare_cache")
    if preserved_sha is not None and not _commit_ancestor(cache, sha, preserved_sha):
        reasons.append("candidate_not_ancestor_of_current_commit")

    run_references: list[dict[str, Any]] = []
    references = _referencing_operator_records(
        candidate, _native_path(run_roots, "run_roots")
    )
    if not references:
        reasons.append("receipt_bound_attempt_reference_missing")
    for root in references:
        operator = root / "operator"
        command = _read_json(operator / "command.json") or {}
        started = _read_json(operator / "started.json")
        finished = _read_json(operator / "finished.json")
        frozen = _read_json(operator / "frozen.json")
        command_valid = (
            command.get("schema_version") == "bio-neb.native-operator-command/1"
            and command.get("candidate_sha") == sha
            and command.get("checkout") == str(candidate)
        )
        started_valid = _started_record_valid(
            started, candidate_sha=sha, checkout=candidate
        )
        finished_valid = _local_terminal_record_valid(
            finished,
            schema="bio-neb.native-operator-finished/1",
            operator=operator,
            candidate_sha=sha,
        )
        frozen_valid = _local_terminal_record_valid(
            frozen,
            schema="bio-neb.native-operator-frozen/1",
            operator=operator,
            candidate_sha=sha,
        )
        tmux = _tmux_facts(command) if command_valid else {"session_present": False}
        live_process = started_valid and _process_identity_matches(started or {})
        receipt = _receipt_pointer_valid(operator, sha)
        run_references.append(
            {
                "run_root": str(root),
                "live_process": live_process,
                "session_present": tmux["session_present"],
                "command_valid": command_valid,
                "started_valid": started_valid,
                "terminal": finished_valid,
                "frozen": frozen_valid,
                "receipt_pointer_valid": receipt,
            }
        )
        if live_process or tmux["session_present"]:
            reasons.append("checkout_has_live_reference")
        if not command_valid or not started_valid:
            reasons.append("operator_identity_evidence_invalid")
        if not finished_valid and not frozen_valid:
            reasons.append("terminal_or_frozen_attempt_evidence_missing")
        if not receipt:
            reasons.append("durable_receipt_pointer_missing")
    process_references = _processes_referencing(candidate) if candidate.exists() else []
    if process_references:
        reasons.append("checkout_referenced_by_process")
    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": "bio-neb.native-operator-cleanup-eligibility/1",
        "target": str(candidate),
        "candidate_sha": sha,
        "eligible": not reasons,
        "reasons": reasons,
        "run_references": run_references,
        "process_reference_pids": process_references,
        "deletion_performed": False,
        "deletion_authority": "none",
        "protected_surfaces": [
            "run_roots",
            "journey_and_evidence",
            "credentials",
            "tmux_processes",
            "native_bare_cache_objects",
            "provider_resources",
        ],
    }


def inspect(
    run_root: str,
    *,
    preserved_checkout: str | None = None,
    preserved_sha: str | None = None,
) -> dict[str, Any]:
    """Capture retained pane text and report local facts and cleanup eligibility."""

    _, operator = _operator_paths(run_root)
    observed = status(run_root)
    command = _read_json(operator / "command.json") or {}
    pane_capture: dict[str, Any] | None = None
    if observed["viewer"]["session_present"]:
        result = _run(
            (
                *_tmux_base(command),
                "capture-pane",
                "-p",
                "-S",
                "-",
                "-t",
                str(command["tmux_session"]),
            )
        )
        capture = result.stdout.encode("utf-8")
        credential = command.get("credential_environment")
        if credential is not None:
            if not isinstance(credential, Mapping) or not isinstance(
                credential.get("path"), str
            ):
                raise OperatorRefused("operator_record_invalid", "command.json")
            patterns = _credential_value_patterns(Path(str(credential["path"])))
            try:
                _StreamGuard(patterns).feed(capture, final=True)
            except _SecretOutputRefused as error:
                raise OperatorRefused(
                    "pane_capture_credential_value_refused", "pane_capture"
                ) from error
        capture_path = operator / "pane-capture.txt"
        staging = operator / ".pane-capture.txt.new"
        staging.write_bytes(capture)
        os.chmod(staging, 0o600)
        os.replace(staging, capture_path)
        pane_capture = {
            "path": str(capture_path),
            "byte_count": len(capture),
            "sha256": hashlib.sha256(capture).hexdigest(),
        }
    eligibility = None
    if preserved_checkout is not None or preserved_sha is not None:
        eligibility = cleanup_eligibility(
            target=str(command["checkout"]),
            expected_sha=str(command["candidate_sha"]),
            preserved_checkout=preserved_checkout,
            preserved_sha=preserved_sha,
        )
    return {"status": observed, "pane_capture": pane_capture, "cleanup": eligibility}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    provision = commands.add_parser("provision")
    provision.add_argument("--bundle", required=True)
    provision.add_argument("--expected-sha", required=True)
    provision.add_argument("--named-run-root", action="append", default=[])
    start_parser = commands.add_parser("start")
    start_parser.add_argument("--run-root", required=True)
    start_parser.add_argument("--run-id", required=True)
    start_parser.add_argument("--checkout", required=True)
    start_parser.add_argument("--expected-sha", required=True)
    start_parser.add_argument("--cwd")
    start_parser.add_argument("--credential-env")
    start_parser.add_argument("--credential-name", action="append", default=[])
    start_parser.add_argument("argv", nargs=argparse.REMAINDER)
    for name in ("status", "attach", "interrupt"):
        child = commands.add_parser(name)
        child.add_argument("--run-root", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--run-root", required=True)
    inspect_parser.add_argument("--preserved-checkout")
    inspect_parser.add_argument("--preserved-sha")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    selected_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if selected_argv[:1] == ["_run-child"]:
            if len(selected_argv) != 3 or selected_argv[1] != "--run-root":
                raise OperatorRefused("internal_child_argv_invalid", "argv")
            return _run_child(selected_argv[2])
        arguments = _parser().parse_args(selected_argv)
        if arguments.operation == "provision":
            result = provision_candidate(
                bundle=arguments.bundle,
                expected_sha=arguments.expected_sha,
                named_run_roots=tuple(arguments.named_run_root),
            )
        elif arguments.operation == "start":
            exact_argv = (
                arguments.argv[1:] if arguments.argv[:1] == ["--"] else arguments.argv
            )
            result = start(
                run_root=arguments.run_root,
                run_id=arguments.run_id,
                checkout=arguments.checkout,
                expected_sha=arguments.expected_sha,
                cwd=arguments.cwd,
                credential_env=arguments.credential_env,
                credential_names=arguments.credential_name,
                argv=exact_argv,
            )
        elif arguments.operation == "status":
            result = status(arguments.run_root)
        elif arguments.operation == "attach":
            return attach(arguments.run_root)
        elif arguments.operation == "interrupt":
            result = interrupt(arguments.run_root)
        elif arguments.operation == "inspect":
            result = inspect(
                arguments.run_root,
                preserved_checkout=arguments.preserved_checkout,
                preserved_sha=arguments.preserved_sha,
            )
        else:
            raise AssertionError("unhandled public operator action")
    except OperatorRefused as error:
        print(
            json.dumps(
                {"status": "refused", "code": error.code, "fields": list(error.fields)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except (OSError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "code": "operator_local_failure",
                    "type": type(error).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
