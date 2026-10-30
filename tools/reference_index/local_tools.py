"""Pinned `bwa` and `samtools` from the user-space install, run by absolute path.

The install (run plan) puts one environment per package under the prefix, as lane H's
install does: `<prefix>/envs/<package>/bin/<package>`. Nothing is looked up on `PATH`.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .contracts import ToolIdentity

DEFAULT_PREFIX = Path.home() / ".local" / "opt" / "bio-neb-reference-index"


class ToolFailed(Exception):
    """A local tool exited non-zero; the message carries the end of its stderr."""


class LocalTools:
    def __init__(self, prefix: Path = DEFAULT_PREFIX) -> None:
        self.prefix = prefix

    def executable(self, package: str) -> Path:
        return self.prefix / "envs" / package / "bin" / package

    def faidx(self, fasta: Path) -> None:
        """`samtools faidx`: the `.fai` beside the FASTA."""
        self.run("samtools", "faidx", str(fasta))

    def mem(self, index: Path, query: Path) -> bytes:
        """`bwa mem <index> <query>`: the SAM of a FASTA query against an indexed FASTA."""
        return self.output("bwa", "mem", str(index), str(query))

    def bases(self, fasta: Path, sequence: str, start: int, end: int) -> str:
        """`samtools faidx <fasta> <sequence>:<start>-<end>`: the bases, upper-cased."""
        text = self.output("samtools", "faidx", str(fasta), f"{sequence}:{start}-{end}")
        return "".join(text.decode().splitlines()[1:]).upper()

    def run(self, package: str, *arguments: str) -> None:
        self.output(package, *arguments)

    def output(self, package: str, *arguments: str) -> bytes:
        completed = self._call(package, *arguments)
        if completed.returncode != 0:
            raise ToolFailed(
                f"{package} {' '.join(arguments)} exited {completed.returncode}: "
                f"{completed.stderr.decode(errors='replace').strip()[-500:]}"
            )
        return completed.stdout

    def identity(self, package: str) -> ToolIdentity:
        """The executable's digest and version line (bwa prints it in its usage, on
        stderr, exiting 1; samtools on `--version`)."""
        completed = (
            self._call(package)
            if package == "bwa"
            else self._call(package, "--version")
        )
        text = (completed.stdout + completed.stderr).decode(errors="replace")
        line = next(
            (
                line.strip()
                for line in text.splitlines()
                if line.startswith("Version:") or line.startswith(f"{package} ")
            ),
            "",
        )
        if not line:
            raise ToolFailed(f"{package} printed no version line")
        executable = self.executable(package)
        return ToolIdentity(
            executable=str(executable),
            sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
            version_line=line,
        )

    def _call(self, package: str, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.executable(package)), *arguments],
            capture_output=True,
            check=False,
        )
