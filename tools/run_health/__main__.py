"""Command line: `python -m tools.run_health run <tool> --runner local|docker|probed ...`,
`python -m tools.run_health cache ...` (a Run's reference into the local cache),
`python -m tools.run_health range <contig:start-end> ...` (coverage and GC of one slice),
`python -m tools.run_health items ...` (every Work item's coverage in one batch),
`python -m tools.run_health check ...` (each unsure item of a batch checked by itself) and
`python -m tools.run_health health ...` (one health report and manifest for a Run).
A Run measures its own health and coverage as a product step (#414); these are
the lane's development extras."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bio_neb.agentic_journey.runtime_config import (
    RuntimeConfigError,
    read_runtime_config,
)
from bio_neb.investigation.observation_tools import probe_observation_tools

from .cache import cache_run_files
from .check import ITEM_REQUESTS, TOTAL_REQUESTS, check_run_items, replay_model
from .docker_host import DockerHostRunner
from .health import run_health
from .items import ItemTarget, measure_run_items
from .local_process import DEFAULT_PREFIX, LocalProcessRunner, ProbedProcessRunner
from .region import RegionFiles, measure_range
from .replay import ReplayRunner
from .runner import RunRefused, ToolRunner, run_tool
from .table import TOOLS, ToolEntry


def main(
    argv: Sequence[str] | None = None,
    *,
    tools: Mapping[str, ToolEntry] = TOOLS,
    object_read_factory: Callable[[Any], Any] | None = None,
) -> int:
    """Print the command's result as JSON (exit 0), or the refusal's code and detail
    (exit 2). `object_read_factory` stands in for object storage in tests."""
    parser = argparse.ArgumentParser(prog="python -m tools.run_health")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one tool-table entry")
    run.add_argument("tool", choices=sorted(tools))
    run.add_argument(
        "--input", action="append", default=[], metavar="NAME=PATH", type=_named_path
    )
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--runner", choices=("local", "docker", "probed"), required=True)
    run.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        type=Path,
        help="local tool install (envs/<package>/bin)",
    )
    run.add_argument("--docker", default="docker", help="Docker CLI executable")
    run.add_argument(
        "--probe-dir",
        type=Path,
        help="the Run's tool probe directory (probed runner); a kept receipt is reused",
    )
    run.add_argument(
        "--runtime-config",
        type=Path,
        help="the Run's runtime file, whose tool section a fresh probe reads",
    )
    cache = commands.add_parser(
        "cache", help="copy a Run's reference FASTA and .fai into the local cache"
    )
    cache.add_argument("--journey", required=True, type=Path, help="the Run's Journey")
    cache.add_argument("--cache-dir", required=True, type=Path)
    cache.add_argument(
        "--resume",
        action="store_true",
        help="keep the files an earlier attempt finished (reread) and copy the rest",
    )
    area = commands.add_parser(
        "range", help="coverage and GC of one typed range, read through the indexes"
    )
    area.add_argument("range", help="contig:start-end, 1-based and inclusive")
    for name in ("bam", "bam-index", "reference", "reference-index"):
        area.add_argument(f"--{name}", required=True, type=Path)
    area.add_argument("--output-dir", required=True, type=Path)
    area.add_argument("--probe-dir", required=True, type=Path)
    area.add_argument("--runtime-config", type=Path)
    batch = commands.add_parser(
        "items", help="coverage and GC of every Work item of a Run in one batch"
    )
    batch.add_argument("--journey", required=True, type=Path, help="the Run's Journey")
    batch.add_argument("--cache-dir", required=True, type=Path)
    batch.add_argument(
        "--health-dir",
        required=True,
        type=Path,
        help="the Run's health report, whose BAM index the batch reads",
    )
    batch.add_argument("--flank-bases", default=100, type=int)
    batch.add_argument("--output-dir", required=True, type=Path)
    batch.add_argument(
        "--replay",
        type=Path,
        help="an earlier batch directory: answer each tool run with its recorded "
        "outputs, running no tool",
    )
    batch.add_argument(
        "--probe",
        action="append",
        default=[],
        metavar="CONTIG:POSITION",
        type=_probe,
        help="also measure this one-base position, as `probe:<contig>:<position>`, "
        "not a Work item",
    )
    check = commands.add_parser(
        "check", help="each unsure item of a Run's batch checked by its own agent loop"
    )
    check.add_argument("--journey", required=True, type=Path, help="the Run's Journey")
    check.add_argument("--cache-dir", required=True, type=Path)
    check.add_argument("--health-dir", required=True, type=Path)
    check.add_argument(
        "--items-dir", required=True, type=Path, help="the Run's batch (`items`)"
    )
    check.add_argument("--output-dir", required=True, type=Path)
    check.add_argument(
        "--request-cap",
        default=TOTAL_REQUESTS,
        type=int,
        help="the model requests all checks together may use",
    )
    check.add_argument(
        "--replay",
        type=Path,
        help="an earlier checks directory: answer with its recorded model exchanges, "
        "no model contact",
    )
    health = commands.add_parser(
        "health", help="one health report for a Run: every health tool, then MultiQC"
    )
    health.add_argument("--journey", required=True, type=Path, help="the Run's Journey")
    health.add_argument("--cache-dir", required=True, type=Path)
    health.add_argument("--output-dir", required=True, type=Path)
    health.add_argument("--prefix", default=DEFAULT_PREFIX, type=Path)
    health.add_argument(
        "--timeout-seconds", type=int, help="bound on each process a run starts"
    )
    health.add_argument(
        "--replay",
        type=Path,
        help="an earlier health report: answer each tool run with its recorded "
        "outputs, running no tool",
    )
    arguments = parser.parse_args(argv)
    if (
        arguments.command == "run"
        and arguments.runner == "probed"
        and arguments.probe_dir is None
    ):
        parser.error("--runner probed needs --probe-dir")
    try:
        printed = _command(arguments, tools, object_read_factory)
    except RunRefused as refused:
        print(json.dumps({"refused": refused.code, "detail": refused.detail}, indent=2))
        return 2
    print(printed)
    return 0


def _command(
    arguments: argparse.Namespace,
    tools: Mapping[str, ToolEntry],
    object_read_factory: Callable[[Any], Any] | None,
) -> str:
    """The chosen command's result as JSON; a refusal raises `RunRefused`."""
    if arguments.command == "items":
        return measure_run_items(
            arguments.journey,
            arguments.cache_dir,
            arguments.health_dir,
            arguments.output_dir,
            flank_bases=arguments.flank_bases,
            probes=arguments.probe,
            runner=None
            if arguments.replay is None
            else ReplayRunner.of_items(arguments.replay, arguments.output_dir),
        ).model_dump_json(indent=2)
    if arguments.command == "check":
        return check_run_items(
            arguments.journey,
            arguments.cache_dir,
            arguments.health_dir,
            arguments.items_dir,
            arguments.output_dir,
            model=None if arguments.replay is None else replay_model(arguments.replay),
            # A replay makes no contact: no cap, so no check's replay depends on another's.
            item_requests=None if arguments.replay else ITEM_REQUESTS,
            total_requests=None if arguments.replay else arguments.request_cap,
        ).model_dump_json(indent=2)
    if arguments.command == "health":
        replay = (
            None
            if arguments.replay is None
            else ReplayRunner.of_health(arguments.replay, arguments.output_dir)
        )
        return run_health(
            arguments.journey,
            arguments.cache_dir,
            arguments.output_dir,
            pinned=replay or LocalProcessRunner(prefix=arguments.prefix),
            probed=replay,
            timeout_seconds=arguments.timeout_seconds,
        ).model_dump_json(indent=2)
    if arguments.command == "range":
        return measure_range(
            RegionFiles(
                bam=arguments.bam,
                bam_index=arguments.bam_index,
                reference=arguments.reference,
                reference_index=arguments.reference_index,
            ),
            arguments.range,
            runner=_probed(arguments),
            output_dir=arguments.output_dir,
        ).model_dump_json(indent=2)
    if arguments.command == "cache":
        return cache_run_files(
            arguments.journey,
            arguments.cache_dir,
            object_read_factory=object_read_factory,
            resume=arguments.resume,
        ).model_dump_json(indent=2)
    return run_tool(
        arguments.tool,
        dict(arguments.input),
        arguments.output_dir,
        runner=_runner(arguments),
        tools=tools,
    ).model_dump_json(indent=2)


def _runner(arguments: argparse.Namespace) -> ToolRunner:
    if arguments.runner == "local":
        return LocalProcessRunner(prefix=arguments.prefix)
    if arguments.runner == "docker":
        return DockerHostRunner(docker=arguments.docker)
    return _probed(arguments)


def _probed(arguments: argparse.Namespace) -> ProbedProcessRunner:
    try:
        tools = (
            None
            if arguments.runtime_config is None
            else read_runtime_config(arguments.runtime_config).investigation_tools
        )
        probe = probe_observation_tools(tools, arguments.probe_dir.absolute())
    except RuntimeConfigError as error:
        raise RunRefused(
            "not_probed", f"{error.code}: {', '.join(error.violations)}"
        ) from error
    return ProbedProcessRunner(probe)


def _probe(value: str) -> ItemTarget:
    contig, separator, position = value.rpartition(":")
    if not separator or not contig or not position.isdigit() or int(position) < 1:
        raise argparse.ArgumentTypeError(f"expected CONTIG:POSITION, got {value!r}")
    # One base: the span is the position itself (`items.span` reads only the length).
    return ItemTarget(
        work_item_id=f"probe:{value}",
        contig=contig,
        one_based_position=int(position),
        ref="N",
    )


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
