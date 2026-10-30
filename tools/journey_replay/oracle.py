"""Capture and compare Journey meaning across exact source revisions."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

CAPTURE_SCHEMA = "bio-neb.journey-replay-capture/1"
COMPARISON_SCHEMA = "bio-neb.journey-replay-comparison/1"
NORMALIZATION_SCHEMA = "bio-neb.journey-normalization/1"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_MISSING = object()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NormalizationRules(_StrictModel):
    schema_version: Literal["bio-neb.journey-normalization/1"]
    journey_identities: str = (
        "Values in journey_id, *_journey_id and *_journey_ids fields are mapped "
        "by first schema-aware occurrence."
    )
    event_identities: str = (
        "Values in event_id, parent_ids, *_event_id and *_event_ids fields are mapped "
        "by event order; references outside the retained stream use a separate mapping "
        "by first occurrence."
    )
    attempt_identities: str = (
        "Values in attempt_id, *_attempt_id and *_attempt_ids fields are mapped "
        "by first schema-aware occurrence."
    )
    occurrence_times: str = (
        "Only occurred_at on dictionaries with the Journey event envelope is replaced."
    )
    preserved: str = (
        "List order, parent relationships, event types, payload values, content-addressed "
        "identities, artifact digests, byte counts and all other timestamps remain exact."
    )


class SourceIdentity(_StrictModel):
    checkout: str
    sha: str


class FileIdentity(_StrictModel):
    byte_count: int
    sha256: str


class JourneyInput(_StrictModel):
    journey_file: str
    journey: FileIdentity
    bundle_file_count: int


class ContactPolicy(_StrictModel):
    network: Literal["blocked"] = "blocked"
    subprocess: Literal["blocked"] = "blocked"
    filesystem_writes: Literal["blocked"] = "blocked"


class CapturedMeaning(_StrictModel):
    raw_events: list[dict[str, Any]]
    observe: dict[str, Any]
    review_journey: dict[str, Any]


class ReplayCapture(_StrictModel):
    schema_version: Literal["bio-neb.journey-replay-capture/1"]
    source: SourceIdentity
    input: JourneyInput
    contact: ContactPolicy
    normalization: NormalizationRules
    meaning: CapturedMeaning
    limits: list[str]


class CaptureRequest(_StrictModel):
    source_checkout: Path
    expected_sha: str
    journey_file: Path


class Difference(_StrictModel):
    classification: Literal[
        "added_field", "removed_field", "changed_value", "changed_structure"
    ]
    path: str
    baseline: Any = None
    candidate: Any = None


class Investigation(_StrictModel):
    status: Literal["not_required", "required"]
    allowed_dispositions: list[
        Literal["regression", "ticket_authorized_change", "oracle_limitation"]
    ]
    comparison_truth_is_immutable: Literal[True] = True


class ReplayComparison(_StrictModel):
    schema_version: Literal["bio-neb.journey-replay-comparison/1"]
    equivalent: bool
    outcome: Literal["identical", "difference_requires_investigation"]
    baseline_source: SourceIdentity
    candidate_source: SourceIdentity
    baseline_input: JourneyInput
    candidate_input: JourneyInput
    normalization: NormalizationRules
    differences: list[Difference]
    investigation: Investigation
    limits: list[str]


NORMALIZATION_RULES = NormalizationRules(schema_version=NORMALIZATION_SCHEMA)

PROOF_LIMITS = [
    "Equality covers only the retained Journey prefix supplied to both revisions.",
    "Capture exercises Journey.read, observe and review_journey reader/projection grammar; "
    "it does not rerun workflow generation or consequential commands.",
    "It does not prove live Run success, paths absent from the retained input, or a change "
    "that both revisions make consistently.",
    "Every difference remains non-equivalent until its evidence-backed disposition is "
    "retained as a regression, ticket-authorized change, or oracle limitation.",
]


class ReplayError(ValueError):
    """An input cannot support an exact replay-equivalence claim."""


def _canonical_json(value: object) -> str:
    from bio_neb.journey import canonical_json

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_unset=True)
    return canonical_json(value)


def _identity_kind(field: str | None) -> str | None:
    if field is None:
        return None
    if field in {"event_id", "parent_ids"} or field.endswith(
        ("_event_id", "_event_ids")
    ):
        return "event"
    if field == "journey_id" or field.endswith(("_journey_id", "_journey_ids")):
        return "journey"
    if field == "attempt_id" or field.endswith(("_attempt_id", "_attempt_ids")):
        return "attempt"
    return None


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, str)]
    return []


def _remember(mapping: dict[str, str], value: str, prefix: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return
    if str(parsed) != value:
        return
    if value not in mapping:
        prefix_count = sum(
            existing.startswith(f"{prefix}:") for existing in mapping.values()
        )
        mapping[value] = f"{prefix}:{prefix_count:06d}"


def _collect_schema_identities(
    value: object, mappings: dict[str, dict[str, str]]
) -> None:
    if isinstance(value, Mapping):
        for field, child in value.items():
            kind = _identity_kind(str(field))
            if kind is not None:
                for identity in _strings(child):
                    prefix = "external_event" if kind == "event" else kind
                    _remember(mappings[kind], identity, prefix)
            _collect_schema_identities(child, mappings)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _collect_schema_identities(child, mappings)


def _identity_mappings(surfaces: Mapping[str, object]) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {
        "journey": {},
        "event": {},
        "attempt": {},
    }
    events = surfaces.get("raw_events")
    if not isinstance(events, list):
        raise ReplayError("raw_events must be a list")
    for event in events:
        if not isinstance(event, Mapping):
            raise ReplayError("each raw event must be an object")
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            raise ReplayError("each raw event must have a string event_id")
        _remember(mappings["event"], event_id, "event")
    _collect_schema_identities(surfaces, mappings)
    return mappings


def _event_envelope(value: Mapping[str, object]) -> bool:
    return {"event_id", "event_type", "parent_ids", "occurred_at"}.issubset(value)


def _normalize(
    value: object,
    mappings: Mapping[str, Mapping[str, str]],
    *,
    field: str | None = None,
) -> object:
    kind = _identity_kind(field)
    if isinstance(value, str):
        if kind is not None:
            return mappings[kind].get(value, value)
        return value
    if isinstance(value, Mapping):
        event_envelope = _event_envelope(value)
        normalized: dict[str, object] = {}
        for child_field, child in value.items():
            if child_field == "occurred_at" and event_envelope:
                normalized[str(child_field)] = "<occurred_at>"
            else:
                normalized[str(child_field)] = _normalize(
                    child, mappings, field=str(child_field)
                )
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(child, mappings, field=field) for child in value]
    return value


def normalize_meaning(
    raw_events: list[dict[str, Any]],
    observe_projection: dict[str, Any],
    review_projection: dict[str, Any],
) -> dict[str, object]:
    """Normalize only replay-variant identities and event occurrence times."""

    surfaces: dict[str, object] = {
        "raw_events": raw_events,
        "observe": observe_projection,
        "review_journey": review_projection,
    }
    return _normalize(surfaces, _identity_mappings(surfaces))  # type: ignore[return-value]


def _pointer(path: tuple[str | int, ...]) -> str:
    if not path:
        return ""
    return "/" + "/".join(
        str(item).replace("~", "~0").replace("/", "~1") for item in path
    )


def _difference(
    classification: str,
    path: tuple[str | int, ...],
    baseline: object,
    candidate: object,
) -> Difference:
    result: dict[str, object] = {
        "classification": classification,
        "path": _pointer(path),
    }
    if baseline is not _MISSING:
        result["baseline"] = baseline
    if candidate is not _MISSING:
        result["candidate"] = candidate
    return Difference.model_validate(result)


def _diff(
    baseline: object,
    candidate: object,
    path: tuple[str | int, ...] = (),
) -> list[Difference]:
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        differences: list[dict[str, object]] = []
        for field in sorted(set(baseline) | set(candidate)):
            child_path = (*path, str(field))
            if field not in baseline:
                differences.append(
                    _difference("added_field", child_path, _MISSING, candidate[field])
                )
            elif field not in candidate:
                differences.append(
                    _difference("removed_field", child_path, baseline[field], _MISSING)
                )
            else:
                differences.extend(_diff(baseline[field], candidate[field], child_path))
        return differences
    baseline_sequence = isinstance(baseline, Sequence) and not isinstance(
        baseline, (str, bytes, bytearray)
    )
    candidate_sequence = isinstance(candidate, Sequence) and not isinstance(
        candidate, (str, bytes, bytearray)
    )
    if baseline_sequence and candidate_sequence:
        differences: list[Difference] = []
        common_length = min(len(baseline), len(candidate))  # type: ignore[arg-type]
        for index, (left, right) in enumerate(
            zip(baseline[:common_length], candidate[:common_length], strict=True)  # type: ignore[index]
        ):
            differences.extend(_diff(left, right, (*path, index)))
        for index in range(common_length, len(baseline)):  # type: ignore[arg-type]
            differences.append(
                _difference(
                    "changed_structure",
                    (*path, index),
                    baseline[index],
                    _MISSING,  # type: ignore[index]
                )
            )
        for index in range(common_length, len(candidate)):  # type: ignore[arg-type]
            differences.append(
                _difference(
                    "changed_structure",
                    (*path, index),
                    _MISSING,
                    candidate[index],  # type: ignore[index]
                )
            )
        return differences
    if type(baseline) is not type(candidate):
        return [_difference("changed_structure", path, baseline, candidate)]
    if baseline != candidate:
        return [_difference("changed_value", path, baseline, candidate)]
    return []


def _validated_capture(value: object, label: str) -> ReplayCapture:
    try:
        return ReplayCapture.model_validate(value, strict=True)
    except ValidationError as error:
        raise ReplayError(f"{label} is not a valid {CAPTURE_SCHEMA} capture") from error


def compare_captures(baseline: object, candidate: object) -> dict[str, object]:
    """Compare exact normalized meaning without suppressing any difference."""

    left = _validated_capture(baseline, "baseline")
    right = _validated_capture(candidate, "candidate")
    if left.normalization != right.normalization:
        raise ReplayError("captures use different normalization rules")
    differences = _diff(
        left.meaning.model_dump(mode="json"), right.meaning.model_dump(mode="json")
    )
    equivalent = not differences
    comparison = ReplayComparison(
        schema_version=COMPARISON_SCHEMA,
        equivalent=equivalent,
        outcome="identical" if equivalent else "difference_requires_investigation",
        baseline_source=left.source,
        candidate_source=right.source,
        baseline_input=left.input,
        candidate_input=right.input,
        normalization=left.normalization,
        differences=differences,
        investigation=Investigation(
            status="not_required" if equivalent else "required",
            allowed_dispositions=[
                "regression",
                "ticket_authorized_change",
                "oracle_limitation",
            ],
        ),
        limits=PROOF_LIMITS,
    )
    return comparison.model_dump(mode="json", exclude_unset=True)


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
    return {"byte_count": count, "sha256": digest.hexdigest()}


def _bundle_inventory(journey_file: Path) -> dict[str, dict[str, object]]:
    root = journey_file.parent
    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReplayError(f"Journey bundle contains a symlink: {path}")
        if path.is_file():
            inventory[str(path.relative_to(root))] = _file_identity(path)
    return inventory


def _source_identity(request: CaptureRequest) -> SourceIdentity:
    checkout = request.source_checkout.resolve()
    if not request.source_checkout.is_absolute() or not checkout.is_dir():
        raise ReplayError("source checkout must be an existing absolute directory")
    if _FULL_SHA.fullmatch(request.expected_sha) is None:
        raise ReplayError(
            "expected source SHA must be 40 lowercase hexadecimal characters"
        )
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if completed.returncode or actual != request.expected_sha:
        raise ReplayError(
            "source checkout HEAD is "
            f"{actual or 'unavailable'}, expected {request.expected_sha}"
        )
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode or status.stdout:
        raise ReplayError("source checkout must be clean")
    return SourceIdentity(checkout=str(checkout), sha=actual)


def _worker_environment(source_checkout: Path) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in ("HOME", "LANG", "LC_ALL", "PATH", "SYSTEMROOT", "WINDIR")
        if name in os.environ
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(source_checkout / "src")
    return environment


def _replicate_bundle(
    journey_file: Path, replica_workspace: Path
) -> tuple[Path, dict[str, dict[str, object]]]:
    journey = journey_file.resolve()
    workspace = replica_workspace.resolve()
    if workspace.is_relative_to(journey.parent):
        raise ReplayError("Journey replica must be outside the retained bundle")
    if workspace.exists():
        raise ReplayError(f"Journey replica destination already exists: {workspace}")
    replica = workspace / journey.parent.name
    before = _bundle_inventory(journey)
    try:
        workspace.mkdir(parents=False, exist_ok=False)
        shutil.copytree(journey.parent, replica)
    except Exception:
        if workspace.exists():
            shutil.rmtree(workspace)
        raise
    after = _bundle_inventory(journey)
    if before != after:
        shutil.rmtree(workspace)
        raise ReplayError("retained Journey bundle changed while creating its replica")
    return replica / journey.name, before


def capture_revision(request: CaptureRequest, replica_root: Path) -> dict[str, object]:
    """Capture one immutable Journey replica through one exact clean revision."""

    source = _source_identity(request)
    journey = request.journey_file.resolve()
    if not request.journey_file.is_absolute() or not journey.is_file():
        raise ReplayError("journey file must be an existing absolute file")
    resolved_replica_root = replica_root.resolve()
    replica_file, retained_inventory = _replicate_bundle(journey, resolved_replica_root)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_capture_worker",
                "--source-checkout",
                source.checkout,
                "--expected-sha",
                request.expected_sha,
                "--journey-file",
                str(replica_file),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_worker_environment(Path(source.checkout)),
        )
    finally:
        shutil.rmtree(resolved_replica_root)
    if retained_inventory != _bundle_inventory(journey):
        raise ReplayError("retained Journey bundle changed during replay capture")
    if completed.returncode:
        detail = completed.stderr.strip() or "capture worker failed without stderr"
        raise ReplayError(f"capture failed for {request.expected_sha}: {detail}")
    try:
        capture = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReplayError("capture worker returned invalid JSON") from error
    validated = _validated_capture(capture, "worker result")
    if validated.source != source:
        raise ReplayError("capture worker source identity disagrees with the parent")
    retained_journey = retained_inventory.get(journey.name)
    if retained_journey is None:
        raise ReplayError("retained bundle inventory has no Journey file")
    corrected_input = JourneyInput(
        journey_file=str(journey),
        journey=FileIdentity.model_validate(retained_journey, strict=True),
        bundle_file_count=len(retained_inventory),
    )
    return validated.model_copy(update={"input": corrected_input}).model_dump(
        mode="json"
    )


def _deny_contact_or_write(event: str, arguments: tuple[object, ...]) -> None:
    if event in {"socket.connect", "socket.bind", "subprocess.Popen", "os.system"}:
        raise RuntimeError(f"replay capture forbids {event}")
    if event == "open":
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        writes_by_mode = isinstance(mode, str) and any(
            marker in mode for marker in ("w", "a", "x", "+")
        )
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        writes_by_flags = isinstance(flags, int) and bool(flags & write_flags)
        if writes_by_mode or writes_by_flags:
            raise RuntimeError("replay capture forbids filesystem writes")
    if event in {
        "os.remove",
        "os.rename",
        "os.mkdir",
        "os.rmdir",
        "os.symlink",
        "os.link",
        "os.truncate",
    }:
        raise RuntimeError(f"replay capture forbids {event}")


def _capture_worker(
    source_checkout: Path, expected_sha: str, journey_file: Path
) -> ReplayCapture:
    source_root = source_checkout.resolve()
    source_path = source_root / "src"
    if Path(os.environ.get("PYTHONPATH", "")).resolve() != source_path:
        raise ReplayError(
            "capture worker PYTHONPATH does not match the source checkout"
        )
    sys.path.insert(0, str(source_path))
    sys.addaudithook(_deny_contact_or_write)

    from bio_neb import __file__ as package_file
    from bio_neb.application.queries import (
        ObserveRun,
        ReviewJourneyQuery,
        observe,
        review_journey,
    )
    from bio_neb.journey import Journey

    imported_package = Path(package_file or "").resolve()
    if not imported_package.is_relative_to(source_path):
        raise ReplayError(
            "capture imported bio_neb outside the selected source checkout"
        )
    before = _bundle_inventory(journey_file)
    journey = Journey.open(journey_file)
    raw_events = journey.read()
    observed = observe(
        ObserveRun(journey_file=str(journey_file)), journey=journey
    ).model_dump(mode="json")
    reviewed = review_journey(
        ReviewJourneyQuery(journey_file=str(journey_file)), journey=journey
    ).model_dump(mode="json")
    after = _bundle_inventory(journey_file)
    if before != after:
        raise ReplayError("Journey bundle changed during replay capture")
    journey_identity = before.get(journey_file.name)
    if journey_identity is None:
        raise ReplayError("replica inventory has no Journey file")
    meaning = normalize_meaning(raw_events, observed, reviewed)
    return ReplayCapture(
        schema_version=CAPTURE_SCHEMA,
        source=SourceIdentity(checkout=str(source_root), sha=expected_sha),
        input=JourneyInput(
            journey_file=str(journey_file),
            journey=FileIdentity.model_validate(journey_identity, strict=True),
            bundle_file_count=len(before),
        ),
        contact=ContactPolicy(),
        normalization=NORMALIZATION_RULES,
        meaning=CapturedMeaning.model_validate(meaning, strict=True),
        limits=PROOF_LIMITS,
    )


def _write_json(
    path: Path, value: object, *, forbidden_root: Path | None = None
) -> None:
    output = path.resolve()
    if not path.is_absolute():
        raise ReplayError("output path must be absolute")
    if forbidden_root is not None and output.is_relative_to(forbidden_root.resolve()):
        raise ReplayError(
            "output must not be placed inside the inspected Journey bundle"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(value) + "\n")


def _validate_output_path(path: Path, forbidden_root: Path) -> None:
    if not path.is_absolute():
        raise ReplayError("output path must be absolute")
    if path.resolve().is_relative_to(forbidden_root.resolve()):
        raise ReplayError(
            "output must not be placed inside the inspected Journey bundle"
        )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read JSON from {path}") from error


def _capture_command(args: argparse.Namespace) -> int:
    requested_output = Path(args.output)
    journey_file = Path(args.journey_file).resolve()
    _validate_output_path(requested_output, journey_file.parent)
    output = requested_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = capture_revision(
        CaptureRequest(
            source_checkout=Path(args.source_checkout),
            expected_sha=args.expected_sha,
            journey_file=journey_file,
        ),
        output.parent / ".journey-replay-input",
    )
    _write_json(output, capture, forbidden_root=journey_file.parent)
    print(_canonical_json(capture))
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    comparison = compare_captures(
        _read_json(Path(args.baseline)), _read_json(Path(args.candidate))
    )
    _write_json(Path(args.output), comparison)
    print(_canonical_json(comparison))
    return 0 if comparison["equivalent"] else 1


def _replay_command(args: argparse.Namespace) -> int:
    output = Path(args.output_directory).resolve()
    if not Path(args.output_directory).is_absolute():
        raise ReplayError("output directory must be absolute")
    journey_file = Path(args.journey_file).resolve()
    if output.is_relative_to(journey_file.parent):
        raise ReplayError(
            "output must not be placed inside the inspected Journey bundle"
        )
    output.mkdir(parents=True, exist_ok=False)
    baseline = capture_revision(
        CaptureRequest(
            source_checkout=Path(args.baseline_checkout),
            expected_sha=args.baseline_sha,
            journey_file=journey_file,
        ),
        output / ".journey-replay-input",
    )
    candidate = capture_revision(
        CaptureRequest(
            source_checkout=Path(args.candidate_checkout),
            expected_sha=args.candidate_sha,
            journey_file=journey_file,
        ),
        output / ".journey-replay-input",
    )
    comparison = compare_captures(baseline, candidate)
    _write_json(output / "baseline.json", baseline, forbidden_root=journey_file.parent)
    _write_json(
        output / "candidate.json", candidate, forbidden_root=journey_file.parent
    )
    _write_json(
        output / "comparison.json", comparison, forbidden_root=journey_file.parent
    )
    print(_canonical_json(comparison))
    return 0 if comparison["equivalent"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser(
        "capture", help="capture one revision's Journey meaning"
    )
    capture.add_argument("--source-checkout", required=True)
    capture.add_argument("--expected-sha", required=True)
    capture.add_argument("--journey-file", required=True)
    capture.add_argument("--output", required=True)
    capture.set_defaults(handler=_capture_command)

    compare = commands.add_parser("compare", help="compare two retained captures")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=_compare_command)

    replay = commands.add_parser(
        "replay", help="capture and compare two exact revisions"
    )
    replay.add_argument("--baseline-checkout", required=True)
    replay.add_argument("--baseline-sha", required=True)
    replay.add_argument("--candidate-checkout", required=True)
    replay.add_argument("--candidate-sha", required=True)
    replay.add_argument("--journey-file", required=True)
    replay.add_argument("--output-directory", required=True)
    replay.set_defaults(handler=_replay_command)
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("_worker")
    parser.add_argument("--source-checkout", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--journey-file", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments[:1] == ["_capture_worker"]:
            worker = _worker_parser().parse_args(arguments)
            result = _capture_worker(
                Path(worker.source_checkout),
                worker.expected_sha,
                Path(worker.journey_file),
            )
            print(_canonical_json(result))
            return 0
        args = _parser().parse_args(arguments)
        return args.handler(args)
    except ReplayError as error:
        print(
            _canonical_json(
                {
                    "schema_version": "bio-neb.journey-replay-error/1",
                    "status": "refused",
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
