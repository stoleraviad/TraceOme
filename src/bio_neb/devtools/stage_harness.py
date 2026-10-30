"""Focused offline development checks of one execution stage, with synthetic setup.

The three stage probes (``input-storage``, ``admit-candidate`` and
``cpu-preparation``) run the shared stage engine and accept offline mode only; a
live probe needs a milestone's authority, never this harness. The earlier
reference-placement diagnostic, which copied an account's or a local prepared
candidate, was removed with those routes: a Run's index is the Execution Review's
bound prebuilt index or one CPU preparation job (owner, 2026-10-29, #464).

Usage::

    uv run python -m bio_neb.devtools.stage_harness --stage cpu-preparation \\
        --input <input.json> --mode offline
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import StrictStr, ValidationError, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.persistence import write_once


class StageHarnessRefused(RuntimeError):
    """A fail-stop refusal in the bounded stage harness."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class LiveStageBinding(FrozenContract):
    """The explicit, secret-free control-plane selection of one live harness run."""

    nebius_cli_profile: StrictStr
    tenant_id: StrictStr
    project_id: StrictStr
    journal_file: StrictStr

    @model_validator(mode="after")
    def absolute_journal(self) -> LiveStageBinding:
        if not Path(self.journal_file).is_absolute():
            raise ValueError("journal_file must be absolute")
        return self


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=("input-storage", "admit-candidate", "cpu-preparation"),
        required=True,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", choices=("offline", "live"), required=True)
    live = parser.add_argument_group(
        "live binding", "required with --mode live; refused with --mode offline"
    )
    live.add_argument("--nebius-profile")
    live.add_argument("--tenant-id")
    live.add_argument("--project-id")
    live.add_argument("--journal-file")
    parser.add_argument(
        "--result-file", help="also write the result JSON here (must not exist)"
    )
    return parser


def _live_binding(args: argparse.Namespace) -> LiveStageBinding | None:
    values = {
        "nebius_cli_profile": args.nebius_profile,
        "tenant_id": args.tenant_id,
        "project_id": args.project_id,
        "journal_file": args.journal_file,
    }
    if args.mode == "offline":
        if any(value is not None for value in values.values()):
            raise StageHarnessRefused("stage_harness_live_binding_unexpected")
        return None
    if any(value is None for value in values.values()):
        raise StageHarnessRefused("stage_harness_live_binding_required")
    try:
        return LiveStageBinding(**values)
    except ValidationError as error:
        raise StageHarnessRefused("stage_harness_live_binding_invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _live_binding(args)
        result_path = None if args.result_file is None else Path(args.result_file)
        if result_path is not None and result_path.exists():
            raise StageHarnessRefused("stage_harness_result_file_exists")
        if args.stage in {"admit-candidate", "cpu-preparation"}:
            if args.mode != "offline":
                raise StageHarnessRefused("reference_live_requires_milestone_authority")
            from bio_neb.devtools.parabricks.reference_stage_harness import (
                run_reference_stage,
            )

            result = run_reference_stage(Path(args.input), args.stage)
        else:
            if args.mode != "offline":
                raise StageHarnessRefused(
                    "input_storage_live_requires_milestone_authority"
                )
            from bio_neb.devtools.parabricks.input_storage_harness import (
                run_input_storage,
            )

            result = run_input_storage(Path(args.input))
    except StageHarnessRefused as error:
        refused = {"result": "refused", "reason": error.reason}
        if error.detail is not None:
            refused["detail"] = error.detail
        print(json.dumps(refused, sort_keys=True))
        return 2
    rendered = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    if result_path is not None:
        write_once(result_path, (rendered + "\n").encode("utf-8"))
    print(rendered)
    if args.stage in {"admit-candidate", "cpu-preparation"} and (
        result.disposition == "failed" or result.unresolved_failure_ids
    ):
        return 2
    if args.stage == "input-storage" and result.status != "execution_inputs_verified":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LiveStageBinding", "StageHarnessRefused", "main"]
