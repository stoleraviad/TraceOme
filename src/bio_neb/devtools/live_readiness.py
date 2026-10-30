"""Contact-free readiness before a native live step; it prints, never runs, the next commands.

DEV-owned and thin (ADR 0037): it calls only APP read seams (`read_parent_journey_identity`,
`observe`, `read_fork_point`, `inherited_runtime_selections`, `describe_runtime_drift`,
`read_runtime_config`, `tool_packages`, `default_configuration`), the local locations and the clock check's record, then prints the exact
native `bio_neb.cli` argv the operator runs next inside the capsule. It reads no credential
value, writes nothing, runs no tool and contacts nothing. An unreadable parent refuses before
any command prints; a readable parent the Fork does not admit (for example one whose head is
an open Pass) is `parent_ineligible` with the Fork's code.
Outputs go under `<live-runs>/<run-id>`, the fresh root the capsule's `start` reserves.
A fresh begin takes the build's defaults unless `--runtime-config` names a file that
overrides some (#440). It lists `researcher_decisions` for per-Run launch selections that
may belong to another Run (#209); they never block. It requires the credential file to name
each selected model's credential variable (names only, never values). A Fork's child runs
on its parent's registered selections, so a Fork takes no runtime file. Either entry
reports the observation tools its configuration names (`investigation_tools`) and whether
this checkout's environment provides each package at its named version (#107); for a fresh
begin a missing section or package blocks. A fresh begin whose configuration has no
`parabricks` section reports the account file `begin` composes one from
(`parabricks_account`, #454): its identity, validity and the three per-Run leaves it would
fill; a missing or invalid file is a researcher decision, never a blocker, since the Run
then stops at its preparation Review. Either entry also reads the clock check's record
(`clock_check.readiness`, #392) and blocks, printing the check's command, when the record
is missing, stale or beyond the bound.

    uv run --locked --offline python -m bio_neb.devtools.live_readiness --run-id <fresh-run-id> \\
      (--parent-journey-file <journey.jsonl; a relative path resolves> [--point-event-id <id>]
       | --prompt-file <absolute> --fastq ROLE=ABSOLUTE_PATH|s3://BUCKET/KEY ...
         [--request-file <absolute>]...
         [--runtime-config <absolute runtime.json>])
"""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
import sys
from pathlib import Path

from bio_neb import locations
from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.agentic_journey.runtime_config import (
    PER_RUN_LAUNCH_SELECTIONS,
    RuntimeConfigDrift,
    RuntimeConfigError,
    RuntimeConfiguration,
    RuntimeSelections,
    describe_runtime_drift,
    read_runtime_config,
)
from bio_neb.application.commands import registered_fastq_role
from bio_neb.application.fork import (
    ForkRefused,
    ParentJourneyUnreadable,
    inherited_runtime_selections,
    read_fork_point,
    read_parent_journey_identity,
)
from bio_neb.application.input_journey import paired_fastq_roles
from bio_neb.application.queries import ObserveRun, observe
from bio_neb.devtools import clock_check
from bio_neb.input_inspection.tool import is_staged_object
from bio_neb.investigation.observation_tools import tool_packages
from bio_neb.journey import canonical_json, resolve_journey_file
from bio_neb.parabricks.configuration import (
    ParabricksAccountRefused,
    default_configuration,
)

_OFFLINE_CLI = ("uv", "run", "--locked", "--offline", "python", "-m", "bio_neb.cli")
_STOP = (
    "stop before investigation-begin, begin or any other model-backed command; "
    "each needs its own go-ahead and credential file"
)

# native-ubuntu-operator.md#execution-and-retention-boundary:
# `<ticket>-<purpose>-<UTC>-<short-sha>-a<N>`; the token is `<ticket>` and `a<N>`.
_RUN_ID = re.compile(
    r"(?P<ticket>[0-9]+)-[a-z0-9]+(?:-[a-z0-9]+)*?"
    r"-[0-9]{8}T[0-9]{4,6}Z-[0-9a-f]{7,40}-(?P<attempt>a[0-9]+)"
)
_ATTEMPT = re.compile(r"a[0-9]+")


def _carries_run_token(value: str, run_root: Path, parsed: re.Match | None) -> bool:
    """The ticket as a whole digit run outside any `a<N>` segment, and `a<N>` itself.

    An unparsed run ID has no token to check; it is its own decision.
    """
    if parsed is None:
        return True
    segments = re.split(r"[^a-z0-9]+", value.lower())
    digit_runs = {
        run
        for segment in segments
        if not _ATTEMPT.fullmatch(segment)
        for run in re.findall(r"[0-9]+", segment)
    }
    return parsed["ticket"] in digit_runs and parsed["attempt"] in segments


def _inside_run_root(value: str, run_root: Path, parsed: re.Match | None) -> bool:
    return Path(value).resolve().is_relative_to(run_root)


# Each per-Run launch selection, its reason and the check it must pass; every check
# takes the value, the Run root and the parsed run ID, using what it needs.
_SELECTION_CHECKS = {
    "parabricks.launch.object_scope": (
        "object_scope_lacks_run_token",
        _carries_run_token,
    ),
    "parabricks.launch.object_storage.bucket": (
        "bucket_lacks_run_token",
        _carries_run_token,
    ),
    "parabricks.launch.work_directory": (
        "work_directory_outside_run_root",
        _inside_run_root,
    ),
}


def _researcher_decisions(
    run_id: str, run_root: Path, runtime: RuntimeConfigDrift
) -> list[dict[str, object]]:
    """Per-Run launch selections that may belong to another Run: the researcher decides.

    Never a blocker. Only the labelled per-Run selections are read; a prepared
    candidate's bucket is shared on purpose and never examined.
    """
    parsed = _RUN_ID.fullmatch(run_id)
    decisions: list[dict[str, object]] = []
    if parsed is None:
        decisions.append(
            {"path": "run_id", "value": run_id, "reason": "run_id_unparsed"}
        )
    for dotted in PER_RUN_LAUNCH_SELECTIONS:
        reason, passes = _SELECTION_CHECKS[dotted]
        value = runtime.per_run_values.get(dotted)
        if isinstance(value, str) and not passes(value, run_root, parsed):
            decisions.append({"path": dotted, "value": value, "reason": reason})
    return decisions


def _parabricks_account(run_root: Path) -> dict[str, object]:
    """The account file a Run with no section registers from, as `begin` reads it."""
    path = locations.parabricks_account()
    try:
        content = path.read_bytes()
    except OSError:
        content = None
    report: dict[str, object] = {
        "path": str(path),
        "present": path.is_file(),
        "byte_count": None if content is None else len(content),
        "sha256": None if content is None else hashlib.sha256(content).hexdigest(),
    }
    try:
        launch = default_configuration(path, str(run_root)).launch
    except ParabricksAccountRefused as refused:
        return {
            **report,
            "valid": False,
            "code": refused.code,
            "refused": list(refused.fields),
            "per_run": {},
        }
    return {
        **report,
        "valid": True,
        "code": None,
        "refused": [],
        "per_run": {
            "parabricks.launch.object_scope": launch.object_scope,
            "parabricks.launch.object_storage.bucket": launch.object_storage.bucket,
            "parabricks.launch.work_directory": launch.work_directory,
        },
    }


def _configuration(runtime_config: str | None) -> RuntimeConfiguration | None:
    """A fresh begin's selections, or None when its file does not validate."""
    try:
        return read_runtime_config(
            None if runtime_config is None else Path(runtime_config)
        )
    except RuntimeConfigError:
        return None


def _investigation_tools(
    selections: RuntimeSelections | None,
) -> tuple[dict[str, object], list[str]]:
    """The observation tools the selections name, as this checkout's environment has them.

    The Run probes these tools at its first Investigation grant and holds on a tool it
    cannot provide (#107). Readiness reads only package metadata, the probe's own check
    (`tool_packages`); no tool runs. A runtime that does not validate is the drift
    report's finding, not this one's.
    """
    if selections is None:
        return {"section": "unreadable", "tools": []}, []
    tools = selections.investigation_tools
    if tools is None:
        return {"section": "absent", "tools": []}, ["investigation_tools_missing"]
    packages = tool_packages(tools)
    return (
        {
            "section": "named",
            "tools": [package.model_dump(mode="json") for package in packages],
        },
        []
        if all(package.provided for package in packages)
        else ["investigation_tools_unavailable"],
    )


_ASSIGNED_NAME = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", re.MULTILINE
)


def _credential_names_missing(
    credential_file: Path, selections: RuntimeSelections | None
) -> list[str]:
    """Selected models' credential variables the file does not assign; names only."""
    if selections is None:
        return []
    try:
        assigned = set(
            _ASSIGNED_NAME.findall(credential_file.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError):
        assigned = set()
    names = set(selections.selected_credential_envs().values())
    return [] if names <= assigned else ["model_credential_name_missing"]


def _mode(path: Path) -> dict[str, object]:
    try:
        return {
            "path": str(path),
            "present": True,
            "mode": oct(stat.S_IMODE(path.stat().st_mode)),
        }
    except FileNotFoundError:
        return {"path": str(path), "present": False, "mode": None}


def _folders() -> dict[str, dict[str, object]]:
    state = locations.local_state_root()
    return {
        "local_state_root": _mode(state),
        "controller_home": _mode(locations.controller_home()),
        "live_runs": _mode(locations.run_roots()),
        "checkouts": _mode(locations.checkouts_root()),
        "fixtures": _mode(state / "fixtures"),
        "inputs": _mode(state / "inputs"),
    }


def _fork(parent: Path, run_root: Path, point_event_id: str | None):
    return [
        {
            "command": "fork",
            "contact": "none",
            "argv": [
                *_OFFLINE_CLI,
                "fork",
                "--parent-journey-file",
                str(parent),
                "--child-run-root",
                str(run_root / "child"),
                *(["--point-event-id", point_event_id] if point_event_id else []),
            ],
        },
        {
            "command": "observe",
            "contact": "none",
            "argv": [
                *_OFFLINE_CLI,
                "observe",
                "--journey-file",
                "<fork output: child_journey_file>",
            ],
        },
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bio_neb.devtools.live_readiness")
    parser.add_argument("--run-id", required=True)
    # A fresh begin's partial runtime file; omitted, the build's defaults.
    parser.add_argument("--runtime-config")
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--parent-journey-file")
    entry.add_argument("--prompt-file")
    parser.add_argument(
        "--point-event-id", help="a closed Pass of the parent; omitted, its head"
    )
    parser.add_argument(
        "--fastq",
        action="append",
        default=[],
        metavar="ROLE=ABSOLUTE_PATH|s3://BUCKET/KEY",
    )
    parser.add_argument("--request-file", action="append", default=[])
    parser.add_argument("--paper-file", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.parent_journey_file is not None and arguments.runtime_config:
        parser.error(
            "a Fork runs on its parent's registered selections; omit --runtime-config"
        )
    run_root = locations.run_roots() / arguments.run_id
    runtime = describe_runtime_drift(
        None if arguments.runtime_config is None else Path(arguments.runtime_config)
    )
    credentials = {
        "model_env": _mode(locations.model_env()),
        "controller_env": _mode(locations.controller_env()),
    }
    blockers: list[str] = []
    if run_root.exists():
        blockers.append("run_root_exists")
    clock, clock_findings = clock_check.readiness()
    blockers.extend(clock_findings)
    decisions: list[dict[str, object]] = []
    readiness: dict[str, object] = {
        "schema_version": "bio-neb.live-readiness/1",
        "run_id": arguments.run_id,
        "run_root": str(run_root),
    }
    if arguments.parent_journey_file is not None:
        try:
            parent = Path(resolve_journey_file(arguments.parent_journey_file))
            identity = read_parent_journey_identity(parent)
            # The Fork's own admission decides first: a head inside an open Pass is
            # ineligible even when its unclosed journal makes the whole-Run read refuse.
            try:
                point, prefix = read_fork_point(str(parent), arguments.point_event_id)
            except ForkRefused as refusal:
                if refusal.code == ParentJourneyUnreadable.code:
                    raise ParentJourneyUnreadable(refusal.detail) from refusal
                point, prefix, ineligible = None, None, refusal.code
            try:
                # An admitted point is read as its child inherits it: its prefix
                # only, so a later parent fact current code refuses is no blocker.
                observed = observe(ObserveRun(journey_file=str(parent)), journey=prefix)
                observed_view = {
                    "view": type(observed).__name__,
                    "status": getattr(observed, "status", None),
                }
            except ValueError as error:
                if point is not None:
                    raise
                observed_view = {
                    "view": None,
                    "status": None,
                    "cause": cause_identity(error),
                }
            # The child resumes on its parent's registered selections; a
            # refused point has no prefix to read them from.
            inherited = None if prefix is None else inherited_runtime_selections(prefix)
        except (ParentJourneyUnreadable, OSError, ValueError) as error:
            fields = getattr(error, "fields", ("journey_file",))
            print(
                canonical_json(
                    {
                        "status": "refused",
                        "code": ParentJourneyUnreadable.code,
                        "fields": list(fields),
                    }
                ),
                file=sys.stderr,
            )
            return 2
        tools, _ = _investigation_tools(inherited)
        if point is None:
            # A refused point gets no child-creation command.
            blockers.append(ineligible)
            readiness["entry"] = "parent_ineligible"
            commands = []
        else:
            readiness["entry"] = "fork_point"
            readiness["fork_point"] = point.model_dump(mode="json")
            commands = _fork(parent, run_root, arguments.point_event_id)
        readiness.update(
            parent={
                "identity": identity.model_dump(mode="json"),
                "observed": observed_view,
            },
        )
    else:
        prompt = Path(arguments.prompt_file)
        # A staged s3:// eval/demo object is inspected by the Run; only local files here.
        inputs = [
            Path(locator)
            for value in arguments.fastq
            if not is_staged_object(locator := value.partition("=")[2])
        ]
        requests = [Path(value) for value in arguments.request_file]
        papers = [Path(value) for value in arguments.paper_file]
        if not arguments.fastq or not all(
            path.is_file() for path in (prompt, *inputs, *requests, *papers)
        ):
            blockers.append("entry_input_missing")
        # begin refuses any role set but one forward/reverse pair before a
        # Journey exists; readiness names it without running begin.
        if arguments.fastq and not paired_fastq_roles(
            registered_fastq_role(value.partition("=")[0]) for value in arguments.fastq
        ):
            blockers.append("fastq_roles_unsupported")
        if not runtime.valid:
            blockers.append("runtime_config_refused")
        configuration = _configuration(arguments.runtime_config)
        tools, tool_findings = _investigation_tools(configuration)
        blockers.extend(tool_findings)
        credential = next(
            (item for item in credentials.values() if item["present"]), None
        )
        if credential is None:
            blockers.append("credential_file_missing")
        else:
            blockers.extend(
                _credential_names_missing(Path(credential["path"]), configuration)
            )
        commands = []
        if not blockers:
            commands.append(
                {
                    "command": "begin",
                    "contact": "model",
                    "credential_file": credential["path"],
                    "argv": [
                        "uv",
                        "run",
                        # The capsule's credential form: the file right after `run`.
                        "--env-file",
                        credential["path"],
                        "--locked",
                        "python",
                        "-m",
                        "bio_neb.cli",
                        "begin",
                        "--interactive",
                        "--run-root",
                        str(run_root),
                        "--prompt-file",
                        str(prompt),
                        *(
                            ("--runtime-config", arguments.runtime_config)
                            if arguments.runtime_config
                            else ()
                        ),
                        *(
                            part
                            for request in requests
                            for part in ("--request-file", str(request))
                        ),
                        *(
                            part
                            for paper in papers
                            for part in ("--paper-file", str(paper))
                        ),
                        *(
                            part
                            for value in arguments.fastq
                            for part in ("--fastq", value)
                        ),
                    ],
                }
            )
        readiness["entry"] = "fresh_begin"
        account = None
        if configuration is not None and configuration.parabricks is None:
            account = _parabricks_account(run_root)
        readiness["parabricks_account"] = account
        # Only a file's selections may be another Run's; those `begin` derives from
        # this Run are reported, not checked.
        decisions = _researcher_decisions(arguments.run_id, run_root, runtime)
        if account is not None and account["code"] == "runtime_not_this_build":
            # Not a researcher choice: the build's worker refuses another's request.
            blockers.append("runtime_not_this_build")
        elif account is not None and not account["valid"]:
            decisions.append(
                {
                    "path": "parabricks_account",
                    "value": account["path"],
                    "reason": account["code"],
                }
            )
    if blockers and readiness["entry"] == "fork_point":
        commands = []
    readiness.update(
        clock=clock,
        runtime=runtime.model_dump(mode="json"),
        investigation_tools=tools,
        folders=_folders(),
        credentials=credentials,
        ready=not blockers,
        blockers=blockers,
        researcher_decisions=decisions,
        next_commands=commands,
        stop=_STOP,
    )
    print(canonical_json(readiness))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
