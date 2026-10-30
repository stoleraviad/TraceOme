"""Local-process adapters: the runner port over tools installed in user space, and over
the Run's own probed tools (main's `ObservationToolsProbe`)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from bio_neb.application.local_file_observation import stable_file_observation
from bio_neb.investigation.observation_tools import ObservationToolsProbe

from .runner import Binding, Invocation, RunResult, pinned_binding, probe_binding
from .table import ToolEntry

DEFAULT_PREFIX = Path.home() / ".local" / "opt" / "bio-neb-run-health"


class LocalProcessRunner:
    """Run one invocation as a local process from the package's own environment.

    The executable is `<prefix>/envs/<package>/bin/<argv[0]>`, called by absolute path
    (nothing is put on PATH). As `LocalSamtoolsAdapter` does, the run observes that
    executable's digest and the first line of its `--version` before the tool runs; the
    version line is held to the pin and the digest is recorded.
    Inputs are staged as symlinks in a private temporary directory, so they sit side by
    side as under Docker's `/inputs` (an index beside its data file); the output
    directory is the working directory. Container paths in argv map to those places.
    """

    name = "local-process"
    identity_fields = ("executable_sha256", "version_line")
    # Only the version is held to the pin: an executable's digest differs by install
    # (conda writes its prefix into some), and any install of the version may run;
    # the record keeps the digest observed.
    checked_fields = ("version_line",)

    def __init__(self, *, prefix: Path = DEFAULT_PREFIX) -> None:
        self._prefix = prefix.absolute()

    def bind(self, entry: ToolEntry) -> Binding:
        return pinned_binding(entry, runner=self.name, fields=self.checked_fields)

    def run(self, invocation: Invocation) -> RunResult:
        executable = (
            self._prefix / "envs" / invocation.pin.package / "bin" / invocation.argv[0]
        )
        _, observed = stable_file_observation(str(executable))
        if not observed.exists:
            return RunResult(
                exit_code=127,
                identity={field: None for field in self.identity_fields},
                executed_argv=(),
                stderr=f"{executable} is not installed".encode(),
            )
        probe = subprocess.run(
            (str(executable), "--version"),
            capture_output=True,
            check=False,
            timeout=invocation.timeout_seconds,
        )
        lines = probe.stdout.decode(errors="replace").splitlines()
        identity = {
            "executable_sha256": f"sha256:{observed.sha256}",
            # The first stdout line, whatever the exit code: goleft 0.2.6 prints its
            # version line, then its usage, and exits 1.
            "version_line": lines[0] if lines else None,
        }
        argv, completed = _execute(executable, invocation)
        return RunResult(
            exit_code=completed.returncode,
            identity=identity,
            executed_argv=argv,
            stderr=completed.stderr,
        )


class ProbedProcessRunner:
    """Run one invocation with the Run's probed samtools or bcftools.

    The executable is the one the probe receipt names (a `bio-neb-<tool>` entry point
    in the Run's environment). Before the tool runs, the run observes that file's digest
    and the version its `--version` prints, for `run_tool` to hold against the probe.
    Inputs and outputs are placed as for `LocalProcessRunner`.
    """

    name = "probed-process"

    def __init__(self, probe: ObservationToolsProbe) -> None:
        self._probe = probe

    def bind(self, entry: ToolEntry) -> Binding:
        return probe_binding(entry, self._probe, runner=self.name)

    def run(self, invocation: Invocation) -> RunResult:
        tool = self._probe.receipt.tool(invocation.probed)
        executable = Path(tool.executable)
        _, observed = stable_file_observation(str(executable))
        if not observed.exists:
            return RunResult(
                exit_code=127,
                identity={"executable_sha256": None, "tool_version": None},
                executed_argv=(),
                stderr=f"{executable} is not available".encode(),
            )
        probe = subprocess.run(
            (str(executable), "--version"),
            capture_output=True,
            check=False,
            timeout=invocation.timeout_seconds,
        )
        lines = probe.stdout.decode(errors="replace").splitlines()
        name, _, version = lines[0].partition(" ") if lines else ("", "", "")
        identity = {
            "executable_sha256": observed.sha256,
            "tool_version": version
            if probe.returncode == 0 and name == tool.name
            else None,
        }
        argv, completed = _execute(executable, invocation)
        return RunResult(
            exit_code=completed.returncode,
            identity=identity,
            executed_argv=argv,
            stderr=completed.stderr,
        )


def _execute(
    executable: Path, invocation: Invocation
) -> tuple[tuple[str, ...], subprocess.CompletedProcess[bytes]]:
    """Stage the inputs side by side, map container paths and run the executable."""
    with tempfile.TemporaryDirectory(prefix="run-health-inputs-") as staged:
        places = []
        for mount in invocation.mounts:
            if mount.read_only:
                place = Path(staged) / Path(mount.container_path).name
                place.symlink_to(mount.host_path)
            else:
                place = mount.host_path
            places.append((mount.container_path, str(place)))
        places.sort(key=lambda mapped: len(mapped[0]), reverse=True)
        argv = (str(executable), *(_place(a, places) for a in invocation.argv[1:]))
        return argv, _run(argv, _place(invocation.workdir, places), invocation)


def _run(
    argv: tuple[str, ...], cwd: str, invocation: Invocation
) -> subprocess.CompletedProcess[bytes]:
    if invocation.stdout_path is None:
        return subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=invocation.timeout_seconds,
        )
    with invocation.stdout_path.open("xb") as stdout:
        return subprocess.run(
            argv,
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.PIPE,
            check=False,
            timeout=invocation.timeout_seconds,
        )


def _place(argument: str, places: list[tuple[str, str]]) -> str:
    for container_path, place in places:
        if argument == container_path:
            return place
        if argument.startswith(container_path + "/"):
            return place + argument[len(container_path) :]
    return argument
