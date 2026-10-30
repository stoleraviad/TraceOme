"""Record what `observe` said at every pass boundary of a finished Run, for replay.

    uv run python make_snapshots.py <run folder> [<out dir>]   # CLI from this repo, or BIONEB_CHECKOUT

Works on copies only: for each PassClosed line N it copies artifacts/ and contacts/ plus the first N
journey lines into a temp folder named by the Journey id (the backend requires that name), runs
`cli observe` there and saves the trimmed result as <out>/<N>.json, or <N>.refused.json. Then it runs
`cli run-artifacts` once on a whole copy and saves <out>/artifacts.json, or artifacts.refused.json (a copied Run is
refused when its result names the original Journey path; the Results page then says so).
The original run folder is only read.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cli_bridge


async def main(run: Path, out: Path) -> None:
    lines = run.joinpath("journey.jsonl").read_text(encoding="utf-8").splitlines()
    journey_id = json.loads(lines[0])["journey_id"]
    ends = [n for n, line in enumerate(lines, 1) if json.loads(line)["event_type"] == "PassClosed"]
    out.mkdir(parents=True, exist_ok=True)
    for n in ends:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, refusal = await cli_bridge.observe(copy(run, journey_id, lines[:n], Path(tmp)))
        if snapshot is None:
            out.joinpath(f"{n}.refused.json").write_text(json.dumps({"line": n, "code": refusal}))
            print(n, "refused", refusal)
        else:
            snapshot["journey_file"] = f"<run>/journey.jsonl (first {n} lines)"
            out.joinpath(f"{n}.json").write_text(json.dumps(snapshot))
            print(n, snapshot["status"])
    with tempfile.TemporaryDirectory() as tmp:
        document, refused = await cli_bridge.run_artifacts(copy(run, journey_id, lines, Path(tmp)))
    for name in ("artifacts.json", "artifacts.refused.json"):
        out.joinpath(name).unlink(missing_ok=True)
    if document is None:
        out.joinpath("artifacts.refused.json").write_text(json.dumps(refused))
        print("run-artifacts refused", refused)
    else:
        out.joinpath("artifacts.json").write_text(json.dumps(document))
        print("run-artifacts", sum(len(document[g]) for g in ("outputs", "inputs", "reference")), "verified files")


def copy(run: Path, journey_id: str, lines: list[str], tmp: Path) -> Path:
    """The Run's artifacts/ and contacts/ and these Journey lines, in a folder named by the Journey id; its journey file."""
    folder = tmp / journey_id
    folder.mkdir()
    for part in ("artifacts", "contacts"):
        if run.joinpath(part).exists():
            shutil.copytree(run / part, folder / part)
    folder.joinpath("journey.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder / "journey.jsonl"


if __name__ == "__main__":
    here = Path(__file__).parent
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else here / "data" / "snapshots"))
