"""One Fork: an idle child Run from a parent's retained prefix at a closed Pass.

The child's first fact, ``RunForked``, retains the exact parent prefix as its
artifact and names it by identity. The Journey reads inherited prefix, origin
and the child's own facts as one timeline, so the ordinary coordinator continues
the child. Creation reads the parent, writes nothing to it, contacts nothing and
starts no work; inherited approvals explain history and authorize no child action.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import Field, StrictStr, field_validator, model_validator

from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.agentic_journey.refusal_disposition import RefusalDisposition
from bio_neb.agentic_journey.runtime_config import RuntimeSelections
from bio_neb.contracts import StrictFrozenContract
from bio_neb.journey import (
    FORK_ORIGIN_EVENT_TYPE,
    Journey,
    JourneyBundleIdentity,
    JourneyFile,
    JourneyFileIdentity,
    JourneyValidationError,
)
from bio_neb.recording import PassDisposition

_PREFIX_MEDIA_TYPE = "application/vnd.bio-neb.journey-prefix+jsonl"

ForkRefusalCode = Literal[
    "parent_journey_unreadable",
    "fork_point_unknown",
    "fork_point_not_closed",
    "fork_child_root_not_fresh",
    "fork_child_conflict",
    "fork_parent_changed",
]


class ForkRefused(ValueError):
    """Refuse a Fork before any child is published; the parent stays unchanged."""

    def __init__(self, code: ForkRefusalCode, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.disposition = RefusalDisposition(kind="caller_correction_and_reissue")


class ForkRun(StrictFrozenContract):
    """Paths and an optional point only; the Fork reads every identity itself.

    Omitted, the point is the parent's head: its last valid point.
    """

    parent_journey_file: JourneyFile
    child_run_root: StrictStr
    point_event_id: StrictStr | None = None

    @field_validator("child_run_root")
    @classmethod
    def absolute_child_root(cls, value: str) -> str:
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("the child Run root must be absolute and normalized")
        return value


class RunForked(StrictFrozenContract):
    """The child's origin: the exact parent prefix and the closed Pass it ends at."""

    schema_version: Literal["bio-neb.run-fork/1"] = "bio-neb.run-fork/1"
    parent: JourneyFileIdentity
    point_event_id: StrictStr
    point_disposition: PassDisposition
    run_provenance: Literal["causal_child_or_fork"] = "causal_child_or_fork"
    grants_authority: Literal[False] = False

    @model_validator(mode="after")
    def point_is_the_parent_head(self) -> "RunForked":
        if (
            self.parent.head_event_id != self.point_event_id
            or self.parent.head_event_type != "PassClosed"
        ):
            raise ValueError("a Fork point is the closed Pass heading its prefix")
        return self


class ForkPoint(StrictFrozenContract):
    """A contact-free admission read: the exact prefix a Fork would retain."""

    parent: JourneyFileIdentity
    point_event_id: StrictStr
    point_disposition: PassDisposition


class ForkReceipt(StrictFrozenContract):
    """Derived from the child's retained origin, also after the child progresses."""

    schema_version: Literal["bio-neb.fork-receipt/1"] = "bio-neb.fork-receipt/1"
    status: Literal["fork_created", "fork_replayed"]
    child_journey_id: StrictStr
    child_journey_file: StrictStr
    origin_event_id: StrictStr
    origin: RunForked
    run_state: Literal["idle"] = "idle"
    contact_count: Literal[0] = 0
    parent_unchanged: Literal[True] = True
    inherited_event_count: int = Field(ge=1)


class ParentJourneyUnreadable(ValueError):
    """The named Journey does not read under current rules; nothing else is implied."""

    code = "parent_journey_unreadable"

    def __init__(self, cause: str) -> None:
        super().__init__(self.code)
        self.fields = ("journey_file", f"cause:{cause}")


class ParentJourneyIdentity(StrictFrozenContract):
    """One contact-free read of a retained Journey file's whole-file identity."""

    schema_version: Literal["bio-neb.parent-journey-identity/2"] = (
        "bio-neb.parent-journey-identity/2"
    )
    journey_file: StrictStr
    identity: JourneyBundleIdentity


def read_parent_journey_identity(path: Path) -> ParentJourneyIdentity:
    """Read the identity a Fork names its parent by, without writing anything."""
    try:
        identity = Journey.open(Path(path)).bundle_identity()
    except (OSError, ValueError) as error:
        raise ParentJourneyUnreadable(cause_identity(error)) from error
    return ParentJourneyIdentity(journey_file=str(path), identity=identity)


def fork_lineage(events: Sequence[Mapping[str, Any]]) -> tuple[RunForked, ...]:
    """Every Fork origin in a timeline, oldest first, each resting on its point."""
    lineage = []
    for event in events:
        if event["event_type"] != FORK_ORIGIN_EVENT_TYPE:
            continue
        origin = RunForked.model_validate(event["payload"])
        if list(event["parent_ids"]) != [origin.point_event_id]:
            raise ValueError("Fork origin does not rest on its point")
        lineage.append(origin)
    return tuple(lineage)


def own_fork_origin(journey: Journey) -> tuple[str, RunForked] | None:
    """This Run's own origin, when a Fork created it."""
    for event in journey.events_view():
        if (
            event["event_type"] == FORK_ORIGIN_EVENT_TYPE
            and event["journey_id"] == journey.journey_id
        ):
            return event["event_id"], RunForked.model_validate(event["payload"])
    return None


def _prefix_at(parent: Journey, point_event_id: str | None) -> Journey:
    if point_event_id is None:
        return parent
    own = parent.identity().event_count
    first_own = parent.revision - own
    position = parent.event_position(point_event_id)
    if position is None or position < first_own:
        raise ForkRefused("fork_point_unknown", point_event_id)
    lines = parent.retained_prefix.splitlines(keepends=True)
    return parent.retained_view(b"".join(lines[: position - first_own + 1]))


class _Admitted(NamedTuple):
    parent: Journey
    prefix: Journey
    point: ForkPoint
    local_content_ids: tuple[str, ...]


def _admit(parent_journey_file: str, point_event_id: str | None) -> _Admitted:
    from bio_neb.journey_path import JourneyPath
    from bio_neb.recording import PassClosed, read_passes

    try:
        parent = Journey.open(Path(parent_journey_file))
    except (JourneyValidationError, OSError, ValueError) as error:
        raise ForkRefused("parent_journey_unreadable", type(error).__name__) from error
    prefix = _prefix_at(parent, point_event_id)
    head = prefix.latest_event()
    assert head is not None  # An admitted Journey is never empty.
    if head["event_type"] != "PassClosed":
        # An open Pass's contact journal cannot be cut at its head.
        raise ForkRefused("fork_point_not_closed", head["event_type"])
    try:
        # Each closed Pass journal still binds its contact evidence.
        read_passes(prefix)
        local = tuple(
            pointer.content_id
            for _, pointer in JourneyPath.from_journey(prefix).local_artifact_pointers()
        )
    except (JourneyValidationError, OSError, ValueError) as error:
        raise ForkRefused("parent_journey_unreadable", type(error).__name__) from error
    point = ForkPoint(
        parent=prefix.identity(),
        point_event_id=head["event_id"],
        point_disposition=PassClosed.model_validate(head["payload"]).disposition,
    )
    return _Admitted(parent, prefix, point, local)


def read_fork_point(
    parent_journey_file: str, point_event_id: str | None = None
) -> tuple[ForkPoint, Journey]:
    """The same admission a Fork applies, as a contact-free read, with its prefix.

    The admitted point and the exact prefix a child at it would inherit: a child
    reads only its prefix, so a parent fact past the point that current
    code refuses does not make the point unreadable (#105).
    """
    admitted = _admit(parent_journey_file, point_event_id)
    return admitted.point, admitted.prefix


def inherited_runtime_selections(prefix: Journey) -> RuntimeSelections:
    """The selections a child at this prefix resolves its Attempts from: its parent's
    one registration, never a runtime file (#440)."""
    from bio_neb.application.input_journey import RunInputsRegistered, validate_fact
    from bio_neb.recording import read_evidence

    registrations = [
        validate_fact(event)
        for event in read_evidence(prefix)
        if event.get("event_type") == "RunInputsRegistered"
    ]
    if len(registrations) != 1 or not isinstance(registrations[0], RunInputsRegistered):
        raise ValueError("a Fork prefix holds exactly one Run registration")
    return registrations[0].runtime_selections


def _origin(point: ForkPoint) -> RunForked:
    return RunForked(
        parent=point.parent,
        point_event_id=point.point_event_id,
        point_disposition=point.point_disposition,
    )


def _receipt(child: Journey, status: str) -> ForkReceipt:
    found = own_fork_origin(child)
    if found is None:
        raise ForkRefused("fork_child_root_not_fresh", child.journey_id)
    origin_event_id, origin = found
    return ForkReceipt(
        status=status,
        child_journey_id=child.journey_id,
        child_journey_file=str(child.path),
        origin_event_id=origin_event_id,
        origin=origin,
        inherited_event_count=child.event_position(origin_event_id),
    )


def _replay(root: Path, origin: RunForked) -> ForkReceipt:
    """Return the child this exact request already created there, or refuse."""
    entries = list(root.iterdir()) if root.is_dir() else []
    if len(entries) != 1 or not (entries[0] / "journey.jsonl").is_file():
        raise ForkRefused("fork_child_root_not_fresh", str(root))
    try:
        child = Journey.open(entries[0] / "journey.jsonl")
    except (JourneyValidationError, OSError, ValueError) as error:
        raise ForkRefused("fork_child_root_not_fresh", str(root)) from error
    found = own_fork_origin(child)
    if found is None:
        raise ForkRefused("fork_child_root_not_fresh", str(root))
    if found[1] != origin:
        raise ForkRefused("fork_child_conflict", found[0])
    return _receipt(child, "fork_replayed")


def fork_run(command: ForkRun) -> ForkReceipt:
    """Create one idle child Run from a closed point, or return the same child again."""
    parent, prefix, point, local = _admit(
        command.parent_journey_file, command.point_event_id
    )
    origin = _origin(point)
    root = Path(command.child_run_root)
    if os.path.lexists(root):
        return _replay(root, origin)
    staging = root.with_name(f".{root.name}.staging-{uuid.uuid4()}")
    try:
        child = Journey.create_fork(
            staging,
            prefix,
            origin.model_dump(mode="json"),
            media_type=_PREFIX_MEDIA_TYPE,
            content_ids=local,
        )
        if not parent.is_current():
            raise ForkRefused("fork_parent_changed", parent.journey_id)
        if os.path.lexists(root):
            raise ForkRefused("fork_child_root_not_fresh", str(root))
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _receipt(
        Journey.open(root / child.journey_id / "journey.jsonl"), "fork_created"
    )


__all__ = [
    "ForkPoint",
    "ForkReceipt",
    "ForkRefused",
    "ForkRun",
    "ParentJourneyIdentity",
    "ParentJourneyUnreadable",
    "RunForked",
    "fork_lineage",
    "fork_run",
    "own_fork_origin",
    "read_fork_point",
    "read_parent_journey_identity",
]
