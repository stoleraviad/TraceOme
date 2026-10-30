"""The observation tools a Run uses, from its own locked environment (owner, #107).

pysam bundles htslib, bcftools and samtools. Its dispatchers run in-process, so
each tool is a console entry point (``bio-neb-bcftools``, ``bio-neb-samtools``)
that the Run executes like any other command. Two pysam behaviours shape it:
``--version`` prints nothing through the dispatcher, so the entry point answers
from the bundled versions; and its captured mode redirects a named ``--output``
to stdout, so only commands without one are captured. pysam's own stdout file
redirection aborts on this build and is never used.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import Field, StrictStr, computed_field, model_validator

from bio_neb.agentic_journey.runtime_config import (
    InvestigationTools,
    RuntimeConfigError,
)
from bio_neb.content import canonical_json
from bio_neb.contracts import FrozenContract

ToolName = Literal["bcftools", "samtools"]
_TOOLS: tuple[ToolName, ...] = ("bcftools", "samtools")
RECEIPT_NAME = "observation-tools-probe.json"


class ProbedTool(FrozenContract):
    name: ToolName
    executable: StrictStr
    executable_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    package: StrictStr
    package_version: StrictStr
    tool_version: StrictStr
    htslib_version: StrictStr


class ObservationToolsReceipt(FrozenContract):
    schema_version: Literal["bio-neb.observation-tools-probe/1"] = (
        "bio-neb.observation-tools-probe/1"
    )
    tools: tuple[ProbedTool, ...]

    @model_validator(mode="after")
    def each_named_tool_once(self) -> "ObservationToolsReceipt":
        if tuple(tool.name for tool in self.tools) != _TOOLS:
            raise ValueError("a tool probe names each observation tool once, in order")
        return self

    def tool(self, name: ToolName) -> ProbedTool:
        return next(tool for tool in self.tools if tool.name == name)


class ObservationToolsProbe(FrozenContract):
    """A Run's tool probe receipt and where it is kept."""

    receipt: ObservationToolsReceipt
    receipt_path: StrictStr
    receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class ToolPackage(FrozenContract):
    """One named tool's package as this environment provides it."""

    name: ToolName
    package: StrictStr
    package_version: StrictStr
    installed_version: StrictStr | None

    @computed_field(return_type=bool)
    @property
    def provided(self) -> bool:
        return self.installed_version == self.package_version


def tool_packages(tools: InvestigationTools) -> tuple[ToolPackage, ...]:
    """Each named tool's package, from this environment's package metadata.

    Contact-free: no tool runs. The Run's probe and live readiness read this one
    check, so readiness reports exactly what the probe would refuse.
    """
    packages: list[ToolPackage] = []
    for name in _TOOLS:
        named = getattr(tools, name)
        try:
            installed = importlib.metadata.version(named.package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        packages.append(
            ToolPackage(
                name=name,
                package=named.package,
                package_version=named.package_version,
                installed_version=installed,
            )
        )
    return tuple(packages)


def _versions(name: ToolName, stdout: bytes) -> tuple[str, str] | None:
    patterns = (
        re.compile(rf"{name} ([0-9]+(?:\.[0-9]+)+)"),
        re.compile(r"Using htslib ([0-9]+(?:\.[0-9]+)+)"),
    )
    lines = stdout.decode("utf-8", errors="replace").splitlines()
    found = [
        next((m.group(1) for line in lines if (m := p.fullmatch(line))), None)
        for p in patterns
    ]
    tool_version, htslib_version = found
    if tool_version is None or htslib_version is None:
        return None
    return tool_version, htslib_version


def _probe(name: ToolName, package: str, bin_directory: Path) -> ProbedTool | None:
    executable = bin_directory / f"bio-neb-{name}"
    try:
        completed = subprocess.run(
            (str(executable), "--version"),
            capture_output=True,
            check=False,
            shell=False,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        content = executable.read_bytes()
    except (OSError, subprocess.TimeoutExpired):
        return None
    versions = _versions(name, completed.stdout)
    if completed.returncode != 0 or versions is None:
        return None
    return ProbedTool(
        name=name,
        executable=str(executable),
        executable_sha256=hashlib.sha256(content).hexdigest(),
        package=package,
        package_version=importlib.metadata.version(package),
        tool_version=versions[0],
        htslib_version=versions[1],
    )


def probe_observation_tools(
    tools: InvestigationTools | None,
    directory: Path,
    *,
    bin_directory: Path | None = None,
) -> ObservationToolsProbe:
    """Probe the Run's named tools once and keep the receipt in ``directory``.

    A later need reuses the kept receipt without running a tool again. A tool
    the section omits, or the environment cannot provide at its named package
    version, is a configuration fault (``runtime_config_invalid``) naming its
    section path; nothing is written.
    """
    receipt_path = directory / RECEIPT_NAME
    try:
        kept = receipt_path.read_bytes()
    except FileNotFoundError:
        kept = None
    if kept is not None:
        return ObservationToolsProbe(
            receipt=ObservationToolsReceipt.model_validate_json(kept),
            receipt_path=str(receipt_path),
            receipt_sha256=hashlib.sha256(kept).hexdigest(),
        )
    if tools is None:
        raise RuntimeConfigError("runtime_config_invalid", ("investigation_tools",))
    bin_directory = bin_directory or Path(sys.prefix) / "bin"
    violations: list[str] = []
    probed: list[ProbedTool] = []
    for package in tool_packages(tools):
        if not package.provided:
            violations.append(f"investigation_tools.{package.name}.package_version")
            continue
        tool = _probe(package.name, package.package, bin_directory)
        if tool is None:
            violations.append(f"investigation_tools.{package.name}")
        else:
            probed.append(tool)
    if violations:
        raise RuntimeConfigError("runtime_config_invalid", tuple(violations))
    receipt = ObservationToolsReceipt(tools=tuple(probed))
    content = canonical_json(receipt.model_dump(mode="json")).encode("utf-8")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(content)
    os.chmod(receipt_path, 0o444)
    return ObservationToolsProbe(
        receipt=receipt,
        receipt_path=str(receipt_path),
        receipt_sha256=hashlib.sha256(content).hexdigest(),
    )


def _run(tool: str) -> NoReturn:
    import pysam.version
    from pysam.libcutils import _pysam_dispatch

    argv = sys.argv[1:]
    if argv == ["--version"]:
        version = getattr(pysam.version, f"__{tool}_version__")
        sys.stdout.write(
            f"{tool} {version}\nUsing htslib {pysam.version.__htslib_version__}\n"
        )
        raise SystemExit(0)
    if not argv:
        sys.stderr.write(f"usage: bio-neb-{tool} <command> [arguments]\n")
        raise SystemExit(2)
    # Captured mode sends a named --output to stdout instead of the file, so a
    # command that names its output file runs uncaptured.
    writes_file = "--output" in argv or "-o" in argv
    status, stderr, stdout = _pysam_dispatch(
        tool, argv[0], argv[1:], catch_stdout=not writes_file
    )
    # The dispatcher returns text for decodable output and bytes otherwise.
    for stream, captured in ((sys.stdout, stdout), (sys.stderr, stderr)):
        if captured:
            stream.buffer.write(
                captured if isinstance(captured, bytes) else captured.encode()
            )
    raise SystemExit(status)


def bcftools() -> NoReturn:
    _run("bcftools")


def samtools() -> NoReturn:
    _run("samtools")
