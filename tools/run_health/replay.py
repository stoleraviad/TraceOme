"""Replay: the runner port answered from an earlier run's recorded outputs (#406, Q5).

An earlier `health` or `items` run left each tool run's outputs in its own directory and
its record in its result. `ReplayRunner` answers the same command run again: for each
invocation it finds the recorded run in the same place under the earlier directory,
refuses a run whose argv or inputs differ from the recorded one, and copies the recorded
outputs instead of running anything. The callables run unchanged, so their results can be
compared with the earlier ones without the tools, the Run's tool install or contact.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .health import MANIFEST_NAME, HealthManifest
from .items import ITEMS_NAME, ItemsCoverage
from .runner import Binding, Invocation, RunRecord, RunRefused, RunResult
from .table import ToolEntry


class ReplayRunner:
    """One runner port for every entry, pinned or probed, over recorded runs.

    `records` maps each run's directory, relative to the output directory being
    replayed into (`output_root`), to its record; the outputs are copied from the same
    relative directory under `recorded`.
    """

    name = "replay"

    def __init__(
        self, recorded: Path, output_root: Path, records: dict[str, RunRecord]
    ) -> None:
        self._recorded = recorded.absolute()
        self._output_root = output_root.absolute()
        self._records = records

    @classmethod
    def of_health(cls, recorded: Path, output_root: Path) -> ReplayRunner:
        """Replay of a health report (`recorded`'s manifest records, by entry)."""
        manifest = _read(recorded / MANIFEST_NAME, HealthManifest)
        return cls(recorded, output_root, dict(manifest.records))

    @classmethod
    def of_items(cls, recorded: Path, output_root: Path) -> ReplayRunner:
        """Replay of a batch (`recorded`'s three records, by run directory)."""
        coverage = _read(recorded / ITEMS_NAME, ItemsCoverage)
        records = {
            "slice": coverage.slice_record,
            "depth": coverage.depth_record,
            "reference": coverage.reference_record,
        }
        return cls(
            recorded,
            output_root,
            {name: record for name, record in records.items() if record is not None},
        )

    def bind(self, entry: ToolEntry) -> Binding:
        return Binding(
            checked={},
            recorded={"replayed_from": str(self._recorded)},
            held_to="replayed",
            image_reference=None,
        )

    def run(self, invocation: Invocation) -> RunResult:
        (outputs,) = [mount for mount in invocation.mounts if not mount.read_only]
        place = outputs.host_path.relative_to(self._output_root).as_posix()
        record = self._records.get(place)
        if record is None:
            return _refused(invocation, f"no recorded run at {place}")
        if record.argv != invocation.argv:
            return _refused(
                invocation, f"{place} ran {record.argv}; this run is {invocation.argv}"
            )
        differs = self._inputs_differ(invocation, record)
        if differs is not None:
            return _refused(invocation, f"{place}: {differs}")
        try:
            shutil.copytree(
                self._recorded / place, outputs.host_path, dirs_exist_ok=True
            )
        except OSError as error:
            return _refused(
                invocation, f"{self._recorded / place} cannot be copied: {error}"
            )
        return RunResult(
            exit_code=0, identity={}, executed_argv=record.executed_argv, stderr=b""
        )

    def _inputs_differ(self, invocation: Invocation, record: RunRecord) -> str | None:
        """Why this run's inputs are not the recorded run's, if they are not.

        Inputs are matched by their place under `/inputs`. A source input (a Run file,
        the reference, another report's file) must be the very file the recorded run
        read; an input the earlier command generated (under the recorded directory)
        must be in the same place under this run's directory, with the same bytes.
        """
        current = {
            mount.container_path: mount.host_path
            for mount in invocation.mounts
            if mount.read_only
        }
        recorded = {
            f"/inputs/{Path(path).name}": Path(path) for path in record.inputs.values()
        }
        if set(current) != set(recorded):
            return f"reads {sorted(current)}; the recorded run read {sorted(recorded)}"
        for place, path in current.items():
            was = recorded[place]
            if path == was:
                continue
            try:
                generated = path.relative_to(self._output_root) == was.relative_to(
                    self._recorded
                )
            except ValueError:
                generated = False
            # A source file is settled by its path: no byte of another file is read.
            if not generated:
                return f"{place} is {path}; the recorded run read {was}"
            digest = _digest(path)
            if digest is None or digest != _digest(was):
                return f"{place} is {path}; the recorded run read {was}"
        return None


def _digest(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def _refused(invocation: Invocation, why: str) -> RunResult:
    return RunResult(
        exit_code=1,
        identity={},
        executed_argv=invocation.argv,
        stderr=f"replay: {why}".encode(),
    )


def _read(path: Path, contract):
    try:
        return contract.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise RunRefused(
            "replay_unreadable", f"{path} is not a readable recorded result: {error}"
        ) from error
