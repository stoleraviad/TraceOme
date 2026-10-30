"""Run one tool-table entry against named local files through one runner port."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from bio_neb.application.local_file_observation import stable_file_observation
from bio_neb.investigation.observation_tools import ObservationToolsProbe, ToolName

from .table import TOOLS, Pin, StrictModel, ToolEntry

RECORD_SCHEMA = "bio-neb.run-health-record/1"
_INPUTS = "/inputs"
_OUTPUTS = "/outputs"
_STDERR_TAIL = 2000


class Mount(StrictModel):
    host_path: Path
    container_path: str
    read_only: bool


class Invocation(StrictModel):
    """One tool run as a container sees it: argv over `/inputs` and `/outputs` mounts.

    `pin` or `probed` is the entry's: the package or probed tool the runner executes.
    """

    pin: Pin | None = None
    probed: ToolName | None = None
    argv: tuple[str, ...]
    mounts: tuple[Mount, ...]
    workdir: str
    stdout_path: Path | None
    # Seconds each process of the run may take; None leaves it unbounded.
    timeout_seconds: int | None = None


class RunResult(StrictModel):
    """What ran: `identity` holds the runner's observed facts (None when not observed),
    `executed_argv` the command the adapter executed (empty when nothing was executed)."""

    exit_code: int
    identity: dict[str, str | None]
    executed_argv: tuple[str, ...]
    stderr: bytes


class RunRefused(Exception):
    """A run that yields no record; `code` names the reason, `detail` its evidence."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class Binding(StrictModel):
    """What a runner holds one entry to before it runs.

    `checked` are the facts the run must observe, `recorded` what the record keeps as its
    identity and `held_to` the word for where `checked` came from ("pinned", "probed").
    """

    checked: dict[str, str]
    recorded: dict[str, str]
    held_to: str
    image_reference: str | None


class ToolRunner(Protocol):
    """The one runner port: bind an entry, execute one invocation, report what ran.

    `bind` refuses an entry the runner cannot hold to its identity; `run_tool` records a
    run only when every fact in the binding's `checked` equals what the runner observed,
    and records the other observed facts beside the binding's `recorded`.
    """

    name: str

    def bind(self, entry: ToolEntry) -> Binding: ...

    def run(self, invocation: Invocation) -> RunResult: ...


def pinned_binding(
    entry: ToolEntry, *, runner: str, fields: tuple[str, ...]
) -> Binding:
    """Hold an entry to the `Pin` facts a runner observes; refused while one is unset."""
    pin = entry.pin
    if pin is None:
        raise RunRefused(
            "not_pinned",
            f"{entry.probed} is the Run's probed tool; the {runner} runner runs "
            "pinned packages",
        )
    pinned = {field: getattr(pin, field) for field in fields}
    unpinned = [field for field, value in pinned.items() if value is None]
    if unpinned:
        raise RunRefused(
            "not_pinned",
            f"{pin.package} {pin.version} has no {', '.join(unpinned)} "
            f"for the {runner} runner",
        )
    return Binding(
        checked=pinned,
        recorded=pinned,
        held_to="pinned",
        image_reference=pin.image_reference,
    )


def probe_binding(
    entry: ToolEntry, probe: ObservationToolsProbe, *, runner: str
) -> Binding:
    """Hold an entry to the Run's probe: the probed executable's digest and version.

    The record keeps the probe receipt's place and digest, not the probe's facts.
    """
    if entry.probed is None:
        raise RunRefused(
            "not_probed",
            f"{entry.argv[0]} is a pinned package; the {runner} runner runs the "
            "Run's probed tools",
        )
    tool = probe.receipt.tool(entry.probed)
    return Binding(
        checked={
            "executable_sha256": tool.executable_sha256,
            "tool_version": tool.tool_version,
        },
        recorded={
            "probe_receipt": probe.receipt_path,
            "probe_receipt_sha256": probe.receipt_sha256,
        },
        held_to="probed",
        image_reference=None,
    )


class OutputFile(StrictModel):
    sha256: str
    byte_count: int


class RunRecord(StrictModel):
    schema_version: Literal["bio-neb.run-health-record/1"] = RECORD_SCHEMA
    tool: str
    runner: str
    image_reference: str | None
    identity: dict[str, str]
    argv: tuple[str, ...]
    executed_argv: tuple[str, ...]
    inputs: dict[str, str]
    outputs: dict[str, OutputFile]


def run_tool(
    tool: str,
    inputs: Mapping[str, Path],
    output_dir: Path,
    *,
    runner: ToolRunner,
    tools: Mapping[str, ToolEntry] = TOOLS,
    parameters: Mapping[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> RunRecord:
    """Run one table entry against named local files; the record says what ran.

    `parameters` gives the entry's named values; they appear in the record's `argv`.
    `timeout_seconds` bounds each process the runner starts; one that outlasts it is
    refused (`tool_timed_out`).
    """
    entry = tools[tool]
    inputs = {name: path.absolute() for name, path in inputs.items()}
    output_dir = output_dir.absolute()
    binding = runner.bind(entry)
    optional = entry.optional_inputs
    if not set(entry.inputs) <= set(inputs) <= set(entry.inputs) | set(optional):
        takes = f"{sorted(entry.inputs)}" + (
            f" and optionally {sorted(optional)}" if optional else ""
        )
        raise RunRefused(
            "inputs_mismatch", f"{tool} takes {takes}, got {sorted(inputs)}"
        )
    parameters = dict(parameters or {})
    if set(parameters) != set(entry.parameters):
        raise RunRefused(
            "parameters_mismatch",
            f"{tool} takes parameters {sorted(entry.parameters)}, got {sorted(parameters)}",
        )
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        raise RunRefused(
            "output_dir_exists",
            f"{output_dir} exists; each run writes a fresh directory",
        ) from None
    mounts = [
        Mount(host_path=path, container_path=f"{_INPUTS}/{path.name}", read_only=True)
        for path in inputs.values()
    ]
    placeholders = (
        dict(optional)
        | parameters
        | {name: f"{_INPUTS}/{path.name}" for name, path in inputs.items()}
        | {"outputs": _OUTPUTS}
    )
    argv = tuple(argument.format_map(placeholders) for argument in entry.argv)
    mounts.append(Mount(host_path=output_dir, container_path=_OUTPUTS, read_only=False))
    try:
        result = runner.run(
            Invocation(
                pin=entry.pin,
                probed=entry.probed,
                argv=argv,
                mounts=tuple(mounts),
                workdir=_OUTPUTS,
                stdout_path=None if entry.stdout is None else output_dir / entry.stdout,
                timeout_seconds=timeout_seconds,
            )
        )
    except subprocess.TimeoutExpired as expired:
        raise RunRefused(
            "tool_timed_out",
            f"{expired.cmd[0]} ran past {timeout_seconds} s; "
            f"outputs in {output_dir} are not recorded",
        ) from None
    if result.exit_code != 0:
        stderr = result.stderr[-_STDERR_TAIL:].decode(errors="replace")
        raise RunRefused("tool_failed", f"exit {result.exit_code}; stderr: {stderr}")
    checked = binding.checked
    differences = "; ".join(
        f"{field}: {binding.held_to} {checked[field]}, ran {result.identity[field]}"
        for field in checked
        if result.identity[field] != checked[field]
    )
    if differences:
        raise RunRefused(
            "identity_mismatch",
            f"{differences}; outputs in {output_dir} are not recorded",
        )
    outputs = _observe_outputs(output_dir)
    missing = [name for name in entry.outputs if name not in outputs]
    if missing:
        raise RunRefused("output_missing", f"required outputs not written: {missing}")
    return RunRecord(
        tool=tool,
        runner=runner.name,
        image_reference=binding.image_reference,
        # What the binding holds, and what the run observed beyond it (a pinned
        # tool's executable digest, which differs by install).
        identity=binding.recorded
        | {
            field: value
            for field, value in result.identity.items()
            if field not in checked
        },
        argv=argv,
        executed_argv=result.executed_argv,
        inputs={name: str(path) for name, path in inputs.items()},
        outputs=outputs,
    )


def _observe_outputs(output_dir: Path) -> dict[str, OutputFile]:
    outputs = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        _, observed = stable_file_observation(str(path))
        outputs[path.relative_to(output_dir).as_posix()] = OutputFile(
            sha256=observed.sha256, byte_count=observed.byte_count
        )
    return outputs
