"""Write one eval run's inputs: the case's prompt, its runtime and its begin arguments.

Contact-free and credential-free. The reads and the reference come from the scenario's core
files (`evals/core_files.py`), checked against their recorded sha256 before use; nothing is
prefilled from another Run. The runtime is a donor runtime that keeps only the operational
settings: its reference settings are replaced by the scenario's core reference, no prepared
index is selected (the product builds it), the per-Run selections are this Run's, the
planning Agent's request cap is set, the global `parallel_tool_calls` is dropped (each model's
endpoint policy sends its own, owner #305 D2), and a model other than the donor's gets one
roster entry from the product's endpoint policy (`model_endpoint_policy`). A donor naming no
`investigation_tools` is refused: its Run would hold at the first Investigation grant. Run

    python -m evals.prepare --case <case_id> --model <qwen|nemotron> \
        --donor-runtime <runtime.json> --run-id <run_id> --run-root <capsule root> \
        --inputs-dir <fresh directory> --local-state <local state root>

It writes `prompt.txt`, `runtime.json` and `plan.json` (the begin arguments, the prepared
answers and every input's identity) into the inputs directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from bio_neb.agentic_journey.model_settings import model_endpoint_policy
from evals.case import CompiledCase, load_compiled
from evals.core_files import LEDGER, ScenarioCoreFiles, load_ledger, scenario_files

MODELS = {
    "qwen": "Qwen/Qwen3.5-397B-A17B",
    # The model both passing live Runs used.
    "nemotron": "nvidia/Nemotron-3.5-Super-VL",
}
# A model served away from the donor's endpoint; Super-VL is on us-north1 only.
ENDPOINTS = {"nemotron": "https://api.tokenfactory.us-north1.nebius.com/v1"}
# The run-ledger shape `<ticket>-<purpose>-<UTC>-<short-sha>-a<N>` (docs/runs/README.md).
_RUN_ID = re.compile(
    r"(?P<ticket>[0-9]+)-(?P<purpose>[a-z0-9]+(?:-[a-z0-9]+)*?)"
    r"-[0-9]{8}T[0-9]{4,6}Z-[0-9a-f]{7,40}-(?P<attempt>a[0-9]+)"
)
# The spot cap on one planning Attempt's model requests until a no-progress guard
# exists; exhausting it raises the typed Agent effort Review.
PLANNING_REQUEST_CAP = 30


def runtime_for(
    donor: dict[str, Any],
    *,
    model: str,
    run_id: str,
    run_root: Path,
    core: ScenarioCoreFiles,
    local_state: Path,
    planning_request_cap: int = PLANNING_REQUEST_CAP,
) -> dict:
    """The donor's operational settings with this Run's selections and core reference."""
    match = _RUN_ID.fullmatch(run_id)
    if match is None:
        raise ValueError(
            f"run id {run_id!r} is not <ticket>-<purpose>-<UTC>-<sha>-a<N>"
        )
    token = f"{match['purpose']}-{match['attempt']}"
    if not donor.get("investigation_tools"):
        raise ValueError(
            "donor runtime names no investigation_tools; the Run would hold at its "
            "first Investigation grant"
        )
    runtime = copy.deepcopy(donor)
    launch = runtime["parabricks"]["launch"]
    launch["object_storage"]["bucket"] = f"bio-neb-s{match['ticket']}-{token}"
    launch["object_scope"] = f"issue{match['ticket']}/{token}"
    launch["work_directory"] = str(run_root / "runs" / "execution" / "work")
    # The approved source identity comes from planning; the core file is only the
    # local replica that upload checks against it. An older donor may still name a
    # reference identity, which the runtime refuses.
    launch.pop("accepted_plain_fasta", None)
    runtime["parabricks"].pop("reference_source", None)
    launch["local_reference_source"] = {"path": str(local_state / core.reference.path)}
    # The launch names no prepared candidate; a donor that still names one is
    # refused by the runtime, as above.
    roster = runtime["model_roster"]
    if model not in roster:
        template = next(iter(roster.values()))
        selection = model_endpoint_policy(MODELS[model]).settings_selection
        roster[model] = {
            **template,
            "endpoint": ENDPOINTS.get(model, template["endpoint"]),
            "model": MODELS[model],
            "settings_selection": selection.model_dump(mode="json"),
        }
    if roster[model]["model"] != MODELS[model]:
        raise ValueError(f"roster entry {model} names {roster[model]['model']}")
    invocation = runtime.setdefault("agent_invocation", {})
    invocation.get("model_settings", {}).pop("parallel_tool_calls", None)
    invocation["model_reference"] = model
    roles = invocation.setdefault("scopes", {}).setdefault("role", {})
    roles.setdefault("planning_agent", {})["model_requests"] = planning_request_cap
    return runtime


def _verified(core: ScenarioCoreFiles, local_state: Path) -> None:
    """Each core file is present with its recorded size and sha256."""
    for row in (core.reference, *core.reads):
        path = local_state / row.path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        if (path.stat().st_size, digest.hexdigest()) != (row.bytes, row.sha256):
            raise ValueError(f"core file {row.id} changed at {path}")


def _write(path: Path, content: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def prepare(
    case: CompiledCase,
    *,
    model: str,
    donor_runtime: Path,
    run_id: str,
    run_root: Path,
    inputs_dir: Path,
    local_state: Path,
    ledger: Path = LEDGER,
) -> dict[str, Any]:
    if not all(p.is_absolute() for p in (run_root, inputs_dir, local_state)):
        raise ValueError("run root, inputs directory and local state must be absolute")
    core = scenario_files(load_ledger(ledger), case.scenario)
    roles = {row.role: row for row in core.reads}
    if set(roles) != set(case.public.fastq_roles):
        raise ValueError(f"core reads {sorted(roles)} differ from the case's roles")
    donor_bytes = donor_runtime.read_bytes()
    runtime = runtime_for(
        json.loads(donor_bytes),
        model=model,
        run_id=run_id,
        run_root=run_root,
        core=core,
        local_state=local_state,
    )
    _verified(core, local_state)
    files = {
        "prompt": _write(
            inputs_dir / "prompt.txt", case.public.research_request.encode()
        ),
        "runtime": _write(
            inputs_dir / "runtime.json", (json.dumps(runtime, indent=2) + "\n").encode()
        ),
    }
    for row in (core.reference, *core.reads):
        files[row.role or "reference"] = {
            "path": str(local_state / row.path),
            "bytes": row.bytes,
            "sha256": row.sha256,
            "core_file": row.id,
        }
    begin = [
        "begin",
        "--interactive",
        "--run-root",
        str(run_root / "runs"),
        "--prompt-file",
        files["prompt"]["path"],
        "--runtime-config",
        files["runtime"]["path"],
    ]
    for role in case.public.fastq_roles:
        begin += ["--fastq", f"{role}={files[role]['path']}"]
    plan = {
        "schema_version": "bio-neb.eval-run-plan/2",
        "case_id": case.case_id,
        "model": MODELS[model],
        "run_id": run_id,
        "run_root": str(run_root),
        "donor_runtime": {
            "path": str(donor_runtime),
            "bytes": len(donor_bytes),
            "sha256": hashlib.sha256(donor_bytes).hexdigest(),
        },
        "planning_request_cap": PLANNING_REQUEST_CAP,
        "files": files,
        "begin_arguments": begin,
        "prepared_answers": [a.model_dump() for a in case.planning.prepared_answers],
    }
    _write(inputs_dir / "plan.json", (json.dumps(plan, indent=2) + "\n").encode())
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.prepare")
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--donor-runtime", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument("--local-state", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = prepare(
        load_compiled(args.case),
        model=args.model,
        donor_runtime=args.donor_runtime,
        run_id=args.run_id,
        run_root=args.run_root,
        inputs_dir=args.inputs_dir,
        local_state=args.local_state,
    )
    print(json.dumps(plan["begin_arguments"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
