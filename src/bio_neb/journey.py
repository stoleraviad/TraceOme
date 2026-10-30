"""Typed, append-only Journey bundle authority."""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from bio_neb.content import ContentId, sha256_bytes
from bio_neb.content import canonical_json as canonical_json
from bio_neb.content import canonical_json_digest as canonical_json_digest
from bio_neb.journey_contracts import semantic_families

CONTRACT_VERSION = "1.0"
FORK_ORIGIN_EVENT_TYPE = "RunForked"
"""The one fact that opens a Fork's child Journey; its artifact is the parent prefix."""


def _journey_file(value: str) -> str:
    if not value:
        raise PydanticCustomError("journey_file_empty", "a Journey file is required")
    if not os.path.isabs(value):
        raise PydanticCustomError(
            "journey_file_not_absolute", "a Journey file must be absolute"
        )
    return value


JourneyFile = Annotated[StrictStr, AfterValidator(_journey_file)]
"""The one boundary type for a Journey-path argument: an absolute path.

A registered Run keeps its lexical locations, so its Journey path need not be
normalized. Each refusal's error type names its reason. Operator entries resolve
a relative path first with :func:`resolve_journey_file`; commands only see the
result.
"""


def resolve_journey_file(value: str) -> str:
    """Resolve an operator's Journey path against the working directory.

    An empty value keeps its own refusal instead of naming the working directory.
    """
    return TypeAdapter(JourneyFile).validate_python(
        os.path.abspath(value) if value else value
    )


_ARTIFACT_ALGORITHM = "sha256"
_processing_writers: dict[Path, "Journey"] = {}
_writers_lock = threading.RLock()


def _serialized_write(method):
    @wraps(method)
    def write(self, *args, **kwargs):
        with _writers_lock, self._lock:
            if self.read_only:
                raise JourneyValidationError("retained Journey view is read-only")
            owner = _processing_writers.get(self.path.resolve())
            if owner is not None and owner is not self:
                raise JourneyValidationError(
                    "another Journey writer owns this interval"
                )
            if owner is None and not self.is_current():
                raise JourneyValidationError("Journey writer is stale or unusable")
            try:
                return method(self, *args, **kwargs)
            except OSError:
                self._write_failed = True
                raise

    return write


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JourneyValidationError(ValueError):
    pass


def _refuse_shared_mutation(*args, **kwargs):
    raise TypeError("shared Journey evidence is read-only; copy it before changing it")


class ReadOnlyDict(dict):
    """An admitted JSON object shared by owner readers; copies are ordinary dicts."""

    __slots__ = ()
    __setitem__ = __delitem__ = __ior__ = _refuse_shared_mutation
    clear = pop = popitem = setdefault = update = _refuse_shared_mutation

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return {key: deepcopy(value, memo) for key, value in self.items()}

    def __reduce__(self):
        return dict, (dict(self),)


class ReadOnlyList(list):
    """An admitted JSON array shared by owner readers; copies are ordinary lists."""

    __slots__ = ()
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse_shared_mutation
    append = extend = insert = pop = remove = _refuse_shared_mutation
    clear = sort = reverse = _refuse_shared_mutation

    def __copy__(self):
        return list(self)

    def __deepcopy__(self, memo):
        return [deepcopy(value, memo) for value in self]

    def __reduce__(self):
        return list, (list(self),)


def read_only_json(value: Any) -> Any:
    """Share a JSON value without copies per reader; mutation raises ``TypeError``."""
    if isinstance(value, (ReadOnlyDict, ReadOnlyList)):
        return value
    if isinstance(value, dict):
        return ReadOnlyDict({key: read_only_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return ReadOnlyList(read_only_json(item) for item in value)
    return value


# Families whose admitted meaning is frozen or read-only all the way down, so
# readers may share the held value. Add one only with that proof.
_SHARED_CHECKED_FAMILIES = frozenset({"planning_review"})


# A JSON object field of a frozen read model, held read-only so a shared read
# refuses mutation all the way down; its dumped JSON is unchanged.
ReadOnlyJsonObject = Annotated[dict[str, JsonValue], AfterValidator(read_only_json)]


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UUID")
    return value


class ArtifactReference(BaseModel):
    """Identity and Journey lineage for one retained immutable byte sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    artifact_id: str
    byte_count: int = Field(ge=0)
    journey_id: str
    media_type: str
    source_event_id: str

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        ContentId.parse(value)
        return value

    @field_validator("journey_id", "source_event_id")
    @classmethod
    def validate_uuid_fields(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("media_type must be a non-empty canonical string")
        return value


class ArtifactEnrichmentRetained(BaseModel):
    """Bind optional local evidence to an already durable semantic fact."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_type: Literal["ArtifactEnrichmentRetained"] = "ArtifactEnrichmentRetained"
    source_event_id: str
    role: str
    byte_count: int = Field(ge=0)
    sha256: str

    @field_validator("source_event_id")
    @classmethod
    def validate_source_event_id(cls, value: str) -> str:
        return _canonical_uuid(value, "source_event_id")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("role must be non-empty and normalized")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("sha256 must be lowercase hexadecimal")
        return value


class JourneyEvent(BaseModel):
    """One semantic fact in a Journey's append-only lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    artifact_references: List[ArtifactReference] = Field(default_factory=list)
    event_id: str
    event_type: str
    journey_id: str
    occurred_at: str
    parent_ids: List[str] = Field(default_factory=list)
    payload: Dict[str, JsonValue]

    @field_validator("event_id", "journey_id")
    @classmethod
    def validate_uuid_fields(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)

    @field_validator("parent_ids")
    @classmethod
    def validate_parent_ids(cls, values: List[str]) -> List[str]:
        for value in values:
            _canonical_uuid(value, "parent_id")
        return values

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        if not value:
            raise ValueError("event_type must be a non-empty string")
        return value

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("occurred_at must be an ISO timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value


_SHA256_DIGEST = r"^[0-9a-f]{64}$"


class JourneyFileIdentity(BaseModel):
    """Whole-file identity of one admitted Journey prefix, Pass mechanics included."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    journey_id: str
    event_count: int = Field(ge=1)
    byte_count: int = Field(ge=1)
    journey_sha256: str = Field(pattern=_SHA256_DIGEST)
    head_event_id: str
    head_event_type: str
    head_sha256: str = Field(pattern=_SHA256_DIGEST)

    @field_validator("journey_id", "head_event_id")
    @classmethod
    def validate_uuid_fields(cls, value: str, info: Any) -> str:
        return _canonical_uuid(value, info.field_name)


class JourneyBundleIdentity(BaseModel):
    """The whole file plus the artifact references it retains; nothing else."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    journey_contract_version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    journey: JourneyFileIdentity
    artifact_manifest_sha256: str = Field(pattern=_SHA256_DIGEST)
    bundle_sha256: str = Field(pattern=_SHA256_DIGEST)

    @model_validator(mode="after")
    def exact_bundle(self) -> "JourneyBundleIdentity":
        if self.bundle_sha256 != _bundle_sha256(
            self.journey, self.artifact_manifest_sha256
        ):
            raise ValueError("Journey bundle identity changed")
        return self


def _bundle_sha256(journey: JourneyFileIdentity, artifact_manifest_sha256: str) -> str:
    return canonical_json_digest(
        {
            "journey_id": journey.journey_id,
            "journey_sha256": journey.journey_sha256,
            "artifact_manifest_sha256": artifact_manifest_sha256,
        }
    )


def artifact_manifest(events: Sequence[JourneyEvent]) -> list[Dict[str, JsonValue]]:
    """Every retained artifact reference with the event naming it, canonically sorted."""
    return sorted(
        [
            {
                "referencing_event_id": event.event_id,
                "reference": reference.model_dump(mode="json"),
            }
            for event in events
            for reference in event.artifact_references
        ],
        key=canonical_json,
    )


class JourneyProjection(BaseModel):
    """Smallest contact-free read model: the validated semantic timeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    journey_id: str
    events: Tuple[JourneyEvent, ...]

    @field_validator("journey_id")
    @classmethod
    def validate_journey_id(cls, value: str) -> str:
        return _canonical_uuid(value, "journey_id")


class _Candidate(NamedTuple):
    """One admitted, not yet persisted event and the family meaning it produces."""

    supplied: JourneyEvent
    event: JourneyEvent
    line: bytes
    views: dict[str, Any]


def _parse_journey_lines(content: bytes) -> List[JourneyEvent]:
    """Parse canonical Journey JSONL lines; envelope and lineage are checked on admit."""
    try:
        retained = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise JourneyValidationError("Journey JSONL is not UTF-8") from error
    if not retained:
        raise JourneyValidationError("Journey JSONL is empty")

    events: List[JourneyEvent] = []
    for line_number, retained_line in enumerate(
        retained.splitlines(keepends=True), start=1
    ):
        if not retained_line.endswith("\n") or retained_line.endswith("\r\n"):
            raise JourneyValidationError(
                f"non-canonical JSONL framing on Journey line {line_number}"
            )
        line = retained_line[:-1]
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise JourneyValidationError(
                f"invalid JSON on Journey line {line_number}"
            ) from error
        try:
            canonical = canonical_json(value)
        except (TypeError, ValueError) as error:
            raise JourneyValidationError(
                f"invalid JSON value on Journey line {line_number}"
            ) from error
        if canonical != line:
            raise JourneyValidationError(
                f"non-canonical JSON on Journey line {line_number}"
            )
        try:
            events.append(JourneyEvent.model_validate(value))
        except ValidationError as error:
            raise JourneyValidationError(
                f"invalid Journey event on line {line_number}"
            ) from error

    return events


class Journey:
    """Own one validated prefix and immutable copies of its admitted artifact bytes.

    Reads use this owner's snapshot, never a permanently trusted mutable path.
    An independent open admits current disk contents afresh. Appends advance this
    owner's prefix; callers must check current history before consequential work.
    Memory holds admitted bytes until this owner is released.
    """

    def __init__(
        self, path: Path, journey_id: str, events: Iterable[JourneyEvent] = ()
    ):
        self._admit(path, journey_id, deepcopy(list(events)), None)

    @classmethod
    def _admitted(
        cls,
        path: Path,
        journey_id: str,
        events: List[JourneyEvent],
        retained: bytes,
        inherited: int = 0,
        verified: Optional[Mapping[str, bytes]] = None,
    ) -> "Journey":
        """Admit freshly parsed events whose exact retained bytes are known."""
        owner = cls.__new__(cls)
        owner._admit(path, journey_id, events, retained, verified, inherited=inherited)
        return owner

    def _admit(
        self,
        path: Path,
        journey_id: str,
        events: List[JourneyEvent],
        retained: Optional[bytes],
        verified_artifacts: Optional[Mapping[str, bytes]] = None,
        *,
        inherited: int = 0,
    ) -> None:
        """Admit one timeline: a Fork's inherited parent prefix, then this file."""
        self.path = Path(path)
        self.journey_id = journey_id
        self._events = events
        # Events before this index are the parent prefix a Fork origin retains.
        self._inherited = inherited
        self._lineage_ids = {event.journey_id for event in events} | {journey_id}
        self._artifact_contents: dict[str, bytes] = dict(verified_artifacts or {})
        self._write_failed = False
        self._read_only = False
        self._lock = threading.RLock()
        self.before_write = None
        self._checked_views = {}
        self._read_views = {}
        self._read_only_events: tuple[Mapping[str, Any], ...] = ()
        self._derived: dict[Any, Any] = {}
        self._validate_sequence(self._events, verify_artifacts=True)
        self._checked_views = self._validate_shared_semantics(self.events_view())
        self._index_events()
        self._retained_bytes = (
            retained
            if retained is not None
            else b"".join(
                (canonical_json(event.model_dump(mode="json")) + "\n").encode("utf-8")
                for event in self._events[inherited:]
            )
        )

    def _index_events(self) -> None:
        """Hold the admitted prefix's identities and latest position per event type."""
        self._by_id = {event.event_id: event for event in self._events}
        self._positions = {
            event.event_id: index for index, event in enumerate(self._events)
        }
        self._latest_by_type = {
            event.event_type: index for index, event in enumerate(self._events)
        }
        self._lineage_ids = {event.journey_id for event in self._events} | {
            self.journey_id
        }

    @property
    def read_only(self) -> bool:
        return self._read_only

    def retained_view(self, content: bytes) -> "Journey":
        """Admit an exact reached prefix for reads at its original evidence location.

        This view cannot become a writer, even when it selects the complete file.
        Its bytes are evidence, never a replacement for the source's current head.
        It reuses the artifact bytes this owner already proved; an independent
        open remains the boundary that re-reads them from disk.
        """
        if (
            not content
            or not content.endswith(b"\n")
            or not self.retained_prefix.startswith(content)
        ):
            raise JourneyValidationError(
                "selected bytes are not a complete reached prefix"
            )
        events = self._events[: self._inherited + len(content.splitlines())]
        # The view inherits this owner's proof of the artifact bytes it admitted;
        # it re-checks the envelope and re-derives family meaning for its prefix.
        verified = {
            reference.artifact_id: self._artifact_contents[reference.artifact_id]
            for event in events
            for reference in event.artifact_references
            if reference.artifact_id in self._artifact_contents
        }
        view = type(self).__new__(type(self))
        view._admit(
            self.path,
            self.journey_id,
            list(events),
            content,
            verified,
            inherited=self._inherited,
        )
        view._read_only = True
        return view

    @property
    def prefix_event_ids(self) -> tuple[str, ...]:
        """Identity of the exact immutable event prefix held by this owner."""
        return tuple(event.event_id for event in self._events)

    @property
    def head_event_id(self) -> Optional[str]:
        """The newest admitted event's identity, without re-reading history."""
        with self._lock:
            return self._events[-1].event_id if self._events else None

    @property
    def revision(self) -> int:
        """Admitted event count; it grows with every accepted append or refresh."""
        with self._lock:
            return len(self._events)

    def event_position(self, event_id: str) -> Optional[int]:
        """Index of an admitted event in this prefix, without reading history."""
        with self._lock:
            return self._positions.get(event_id)

    def has_event_type(self, event_type: str) -> bool:
        """Whether the admitted prefix contains an event of this type."""
        with self._lock:
            return event_type in self._latest_by_type

    def latest_event(
        self, event_types: Optional[Iterable[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Isolated copy of the newest admitted event, optionally of these types."""
        with self._lock:
            if event_types is None:
                index = len(self._events) - 1
            else:
                index = max(
                    (
                        self._latest_by_type[event_type]
                        for event_type in set(event_types)
                        if event_type in self._latest_by_type
                    ),
                    default=-1,
                )
            return self._events[index].model_dump(mode="json") if index >= 0 else None

    @property
    def retained_prefix(self) -> bytes:
        """The immutable bytes admitted by this single semantic owner."""
        return self._retained_bytes

    def is_current(self) -> bool:
        """Compare exact retained history without readmitting owned artifact bytes.

        This is a single-writer freshness guard, not a transaction or file lock.
        """
        with _writers_lock:
            if _processing_writers.get(self.path.resolve()) is self:
                return not self._write_failed
        return self.path.read_bytes() == self._retained_bytes

    @contextmanager
    def processing(self):
        """Bind nested application commands to one serialized, admitted writer.

        Check disk once at entry. During this interval all local opens share the
        owner; external mutation and distributed writers are not supported.
        """
        if self.read_only:
            raise JourneyValidationError("retained Journey view is read-only")
        key = self.path.resolve()
        with _writers_lock:
            owner = _processing_writers.get(key)
            if owner is not None:
                raise JourneyValidationError(
                    "another Journey writer owns this interval"
                )
            if self._write_failed or not self.is_current():
                raise JourneyValidationError("Journey writer is stale or unusable")
            _processing_writers[key] = self
        try:
            yield self
        finally:
            with _writers_lock:
                del _processing_writers[key]

    @property
    def run_directory(self) -> Path:
        return self.path.parent

    @property
    def artifacts_directory(self) -> Path:
        return self.run_directory / "artifacts" / _ARTIFACT_ALGORITHM

    @classmethod
    def create(cls, run_root: Path) -> "Journey":
        root = Path(run_root)
        if not root.is_absolute():
            raise JourneyValidationError("run_root must be absolute")
        root.mkdir(parents=True, exist_ok=True)
        journey_id = str(uuid.uuid4())
        journey_directory = root / journey_id
        journey_directory.mkdir(exist_ok=False)
        path = journey_directory / "journey.jsonl"
        path.touch(exist_ok=False)
        return cls(path, journey_id)

    @classmethod
    def create_fork(
        cls,
        run_root: Path,
        parent: "Journey",
        payload: Dict[str, Any],
        *,
        media_type: str,
        content_ids: Iterable[str] = (),
    ) -> "Journey":
        """Create a child whose first fact retains ``parent``'s exact admitted prefix.

        The parent is only read. Every artifact its timeline references, and each
        named local content id, is copied into the child's content-addressed store,
        so the child reads its whole timeline from its own run directory.
        """
        child = cls.create(run_root)
        for reference in {
            reference.artifact_id: reference
            for event in parent._events
            for reference in event.artifact_references
        }.values():
            child._store(parent._verified_artifact_bytes(reference))
        for content_id in content_ids:
            digest = ContentId.parse(content_id).digest
            content = (parent.artifacts_directory / digest).read_bytes()
            if ContentId.of(content) != ContentId.parse(content_id):
                raise JourneyValidationError(
                    "referenced artifact digest does not match"
                )
            child._store(content)
        child._admit(
            child.path,
            child.journey_id,
            list(parent._events),
            b"",
            inherited=len(parent._events),
        )
        child.record_with_artifact(
            FORK_ORIGIN_EVENT_TYPE,
            payload,
            parent.retained_prefix,
            media_type,
            parent_ids=[parent.head_event_id],
        )
        return child

    @classmethod
    def processing_owner(cls, path: Path) -> "Journey | None":
        """Return the current local interval owner, without opening storage."""
        with _writers_lock:
            return _processing_writers.get(Path(path).resolve())

    @classmethod
    def open(cls, path: Path) -> "Journey":
        owner = cls.processing_owner(path)
        if owner is not None:
            return owner
        events, retained, inherited, verified = cls._load(Path(path))
        return cls._admitted(
            Path(path),
            events[inherited].journey_id,
            events,
            retained,
            inherited,
            verified,
        )

    def refresh(self) -> None:
        """Admit an append-only extension, retaining proof of already owned bytes.

        Replacement or truncated history refuses without changing the owned prefix.
        Independently reopening remains the boundary for current artifact disk bytes.
        """
        if self.read_only:
            raise JourneyValidationError("retained Journey view is read-only")
        with _writers_lock:
            if _processing_writers.get(self.path.resolve()) is self:
                return
        events, retained, inherited, _ = self._load(
            self.path, known=self._artifact_contents
        )
        if not retained.startswith(self._retained_bytes):
            raise JourneyValidationError("retained Journey prefix changed")
        self._validate_sequence(events, verify_artifacts=True, inherited=inherited)
        checked_review = self._validate_shared_semantics(
            [e.model_dump(mode="json") for e in events]
        )
        self._events = events
        self._inherited = inherited
        self._checked_views = checked_review
        self._retained_bytes = retained
        self._read_only_events = ()
        self._derived = {}
        self._index_events()

    @classmethod
    def _load(
        cls, path: Path, *, known: Optional[Mapping[str, bytes]] = None
    ) -> tuple[List[JourneyEvent], bytes, int, dict[str, bytes]]:
        """This file's events after any inherited Fork prefix, that prefix's size and
        the prefix artifacts already verified while expanding it."""
        own, retained = cls._load_events(path)
        verified: dict[str, bytes] = dict(known or {})
        inherited = cls._inherited_events(
            own, Path(path).parent / "artifacts" / _ARTIFACT_ALGORITHM, verified
        )
        return [*inherited, *own], retained, len(inherited), verified

    @classmethod
    def _inherited_events(
        cls,
        own: List[JourneyEvent],
        artifacts_directory: Path,
        verified: dict[str, bytes],
    ) -> List[JourneyEvent]:
        """The exact parent prefix a Fork origin retains, through its own origins."""
        origin = own[0]
        if origin.event_type != FORK_ORIGIN_EVENT_TYPE:
            return []
        if len(origin.artifact_references) != 1:
            raise JourneyValidationError("Fork origin must retain its parent prefix")
        reference = origin.artifact_references[0]
        content = verified.get(reference.artifact_id)
        try:
            if content is None:
                content = (
                    artifacts_directory / ContentId.parse(reference.artifact_id).digest
                ).read_bytes()
        except FileNotFoundError as error:
            raise JourneyValidationError("Fork parent prefix is missing") from error
        if (
            ContentId.of(content) != ContentId.parse(reference.artifact_id)
            or len(content) != reference.byte_count
        ):
            raise JourneyValidationError("Fork parent prefix digest does not match")
        verified[reference.artifact_id] = content
        parent = _parse_journey_lines(content)
        return [*cls._inherited_events(parent, artifacts_directory, verified), *parent]

    @staticmethod
    def _load_events(path: Path) -> tuple[List[JourneyEvent], bytes]:
        event_path = Path(path)
        if not event_path.is_file():
            raise JourneyValidationError("Journey JSONL does not exist")
        content = event_path.read_bytes()
        events = _parse_journey_lines(content)
        journey_id = events[0].journey_id
        if event_path.parent.name != journey_id:
            raise JourneyValidationError(
                "Journey directory does not match Journey identity"
            )
        return events, content

    def read(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [event.model_dump(mode="json") for event in self._events]

    def events_view(self) -> tuple[Mapping[str, Any], ...]:
        """The admitted prefix as shared read-only JSON, for owner-internal readers.

        Each event is materialized once per owner and the view extends with the
        prefix; mutating it raises. ``read()`` keeps returning private copies.
        """
        with self._lock:
            held = self._read_only_events
            if len(held) != len(self._events):
                held = self._read_only_events = (
                    *held,
                    *(
                        read_only_json(event.model_dump(mode="json"))
                        for event in self._events[len(held) :]
                    ),
                )
            return held

    def project(self) -> JourneyProjection:
        with self._lock:
            return JourneyProjection(
                journey_id=self.journey_id,
                events=tuple(
                    JourneyEvent.model_validate(event.model_dump(mode="json"))
                    for event in self._events
                ),
            )

    def identity(self) -> JourneyFileIdentity:
        """Identity of exactly the bytes this owner admitted; nothing is reread."""
        with self._lock:
            if not self._events:
                raise JourneyValidationError("Journey is empty")
            head = self._events[-1]
            return JourneyFileIdentity(
                journey_id=self.journey_id,
                event_count=len(self._events) - self._inherited,
                byte_count=len(self._retained_bytes),
                journey_sha256=sha256_bytes(self._retained_bytes),
                head_event_id=head.event_id,
                head_event_type=head.event_type,
                head_sha256=canonical_json_digest(head.model_dump(mode="json")),
            )

    def bundle_identity(self) -> JourneyBundleIdentity:
        """Whole-file identity bound to the artifact references it retains."""
        with self._lock:
            journey = self.identity()
            manifest_sha256 = canonical_json_digest(artifact_manifest(self._events))
            return JourneyBundleIdentity(
                journey=journey,
                artifact_manifest_sha256=manifest_sha256,
                bundle_sha256=_bundle_sha256(journey, manifest_sha256),
            )

    def checked_view(self, family: str):
        """Isolated meaning admitted by the registered domain definition."""
        with self._lock:
            return deepcopy(self._checked_views[family])

    def shared_checked_view(self, family: str):
        """The held admitted meaning itself, for a family whose meaning refuses
        mutation; an append replaces it rather than changing it."""
        if family not in _SHARED_CHECKED_FAMILIES:
            raise ValueError(
                f"{family} meaning is not shared read-only; use checked_view"
            )
        with self._lock:
            return self._checked_views[family]

    def read_view(self, reader, *, revision=None):
        """Cache a checked interpretation of this owner's exact admitted prefix.

        Readers remain domain-owned, receive this owner for artifact validation,
        and return isolated views. An append invalidates the prefix identity;
        neither a caller nor a second process can update this cache as authority.
        """
        with self._lock:
            return deepcopy(self._read_context(reader, revision=revision))

    def derived_once(self, key, derive):
        """Meaning fixed by immutable inputs, derived once for this owner.

        ``key`` names those inputs: an admitted event or a content address.
        Neither changes while the owner lives, so the result stays exact as the
        prefix grows. Callers keep it immutable; an independent open, refresh or
        retained view derives afresh.
        """
        with self._lock:
            if key not in self._derived:
                self._derived[key] = derive()
            return self._derived[key]

    def _read_context(self, reader, *, revision=None):
        """Share an owner-private interpretation; public readers copy its views."""
        with self._lock:
            head = (self._events[-1].event_id if self._events else None, revision)
            cached = self._read_views.get(reader)
            if cached is None or cached[0] != head:
                cached = (head, reader(self))
                self._read_views[reader] = cached
            return cached[1]

    def _validate_shared_semantics(self, events):
        return {
            definition.name: definition.admit(events)
            for definition in semantic_families(
                frozenset(e["event_type"] for e in events)
            )
        }

    def _require_state(self, state: JourneyProjection) -> None:
        # Equal to comparing with ``project()``, without re-materializing history.
        if (
            not isinstance(state, JourneyProjection)
            or state.journey_id != self.journey_id
            or len(state.events) != len(self._events)
            or state.events != tuple(self._events)
        ):
            raise JourneyValidationError(
                "Journey state does not match the writer prefix"
            )

    @_serialized_write
    def preview(
        self,
        state: JourneyProjection,
        event_type: str,
        payload: dict,
        parent_ids=None,
        *,
        event_id=None,
    ) -> JourneyProjection:
        """Validate a complete candidate without persisting facts or attachments."""
        self._require_state(state)
        candidate = self._build_event(
            event_type, payload, parent_ids, event_id, None, (), verify_artifacts=True
        )
        return JourneyProjection(
            journey_id=self.journey_id, events=(*state.events, candidate.supplied)
        )

    @_serialized_write
    def _preview_candidate(
        self, event_type: str, payload: dict, parent_ids=None, *, event_id=None
    ) -> None:
        """Validate a complete candidate against this writer's own held prefix."""
        self._build_event(
            event_type, payload, parent_ids, event_id, None, (), verify_artifacts=True
        )

    @_serialized_write
    def record_against(
        self,
        state: JourneyProjection,
        event_type: str,
        payload: dict,
        parent_ids=None,
        *,
        event_id=None,
    ):
        """Admit against the caller's checked prefix and return the new snapshot."""
        self._require_state(state)
        event = self.record(event_type, payload, parent_ids, event_id=event_id)
        return event, self.project()

    @_serialized_write
    def record(
        self,
        event_type: str,
        payload: Dict[str, Any],
        parent_ids: Optional[List[str]] = None,
        event_id: Optional[str] = None,
        occurred_at: Optional[str] = None,
        artifact_references: Optional[
            Sequence[ArtifactReference | Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        candidate = self._build_event(
            event_type,
            payload,
            parent_ids,
            event_id,
            occurred_at,
            artifact_references or (),
            verify_artifacts=True,
        )
        self._append(candidate)
        return candidate.supplied.model_dump(mode="json")

    @_serialized_write
    def record_with_artifact(
        self,
        event_type: str,
        payload: Dict[str, Any],
        artifact_bytes: bytes,
        media_type: str,
        parent_ids: Optional[List[str]] = None,
        event_id: Optional[str] = None,
        occurred_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retain bytes and append the semantic event that created their lineage."""

        if not isinstance(artifact_bytes, bytes):
            raise JourneyValidationError("artifact_bytes must be bytes")
        selected_event_id = event_id or str(uuid.uuid4())
        reference = self._artifact_reference(
            artifact_bytes,
            media_type=media_type,
            source_event_id=selected_event_id,
        )
        candidate = self._build_event(
            event_type,
            payload,
            parent_ids,
            selected_event_id,
            occurred_at,
            (reference,),
            verify_artifacts=False,
        )
        self._write_artifact(reference, artifact_bytes)
        # The reference was derived from exactly these bytes, now durably retained.
        self._artifact_contents[reference.artifact_id] = artifact_bytes
        self._append(candidate)
        return candidate.supplied.model_dump(mode="json")

    def read_artifact(self, reference: ArtifactReference | Dict[str, Any]) -> bytes:
        validated = self._validate_reference(reference)
        if not any(validated in event.artifact_references for event in self._events):
            raise JourneyValidationError(
                "artifact reference is not retained in the Journey"
            )
        return self._verified_artifact_bytes(validated)

    def artifact_path(self, reference: ArtifactReference | Dict[str, Any]) -> Path:
        validated = self._validate_reference(reference)
        return self.artifacts_directory / ContentId.parse(validated.artifact_id).digest

    def _artifact_reference(
        self, content: bytes, *, media_type: str, source_event_id: str
    ) -> ArtifactReference:
        try:
            return ArtifactReference(
                artifact_id=str(ContentId.of(content)),
                byte_count=len(content),
                journey_id=self.journey_id,
                media_type=media_type,
                source_event_id=source_event_id,
            )
        except ValidationError as error:
            raise JourneyValidationError("invalid artifact reference") from error

    def _build_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        parent_ids: Optional[List[str]],
        event_id: Optional[str],
        occurred_at: Optional[str],
        artifact_references: Sequence[ArtifactReference | Dict[str, Any]],
        *,
        verify_artifacts: bool,
    ) -> _Candidate:
        if self._write_failed:
            raise JourneyValidationError(
                "Journey writer is unusable after an I/O failure"
            )
        value = {
            "contract_version": CONTRACT_VERSION,
            "artifact_references": list(artifact_references),
            "event_id": event_id or str(uuid.uuid4()),
            "event_type": event_type,
            "journey_id": self.journey_id,
            "occurred_at": occurred_at or utc_now(),
            "parent_ids": list(parent_ids or []),
            "payload": payload,
        }
        try:
            supplied = JourneyEvent.model_validate(value)
            line = (canonical_json(supplied.model_dump(mode="json")) + "\n").encode(
                "utf-8"
            )
            # In-memory consumers and a fresh reader see the same canonical mapping
            # order, including nested payloads used in deterministic evidence closure.
            event = JourneyEvent.model_validate_json(line)
        except (ValidationError, TypeError, UnicodeEncodeError, ValueError) as error:
            raise JourneyValidationError("invalid Journey event") from error
        # The held prefix was admitted at open or by earlier appends; only the new
        # event's identity, parents and lineage need checking against it.
        self._check_envelope(
            len(self._events),
            event,
            self._by_id,
            verify_artifacts=False,
            previous=self._events[-1] if self._events else None,
        )
        views = self._admit_families(event)
        if verify_artifacts:
            for reference in event.artifact_references:
                self._verified_artifact_bytes(reference)
        return _Candidate(supplied, event, line, views)

    def _admit_families(self, event: JourneyEvent) -> dict[str, Any]:
        """Admit the new event in each family containing it, from held state.

        A family with a step folds only this event into its checked state; one
        without a step, or not yet held, refolds the read-only prefix plus it.
        """
        views = {}
        prefix = self.events_view()
        candidate = read_only_json(event.model_dump(mode="json"))
        for definition in semantic_families(frozenset({event.event_type})):
            if event.event_type not in definition.payloads:
                continue
            if definition.name in self._checked_views:
                state = self._checked_views[definition.name]
                views[definition.name] = definition.admit_next(state, candidate, prefix)
            else:
                views[definition.name] = definition.admit([*prefix, candidate])
        return views

    def _append(self, candidate: _Candidate) -> None:
        # The candidate was admitted against the held prefix. For
        # record_with_artifact, the content-derived reference is validated before
        # the exact bytes are durably written. Revalidating here would immediately
        # reread every retained artifact and turn later event appends into unrelated
        # full-byte passes.
        revision = len(self._events)
        if self.before_write is not None:
            self.before_write()
        event, retained_line, views = candidate.event, candidate.line, candidate.views
        if len(self._events) != revision:
            # The lazy Pass start extended the prefix the candidate was folded over.
            views = self._admit_families(event)
        try:
            with self.path.open("ab") as stream:
                if stream.write(retained_line) != len(retained_line):
                    raise OSError("short Journey write")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._write_failed = True
            raise
        self._events.append(event)
        self._retained_bytes += retained_line
        self._by_id[event.event_id] = event
        self._positions[event.event_id] = len(self._events) - 1
        self._latest_by_type[event.event_type] = len(self._events) - 1
        self._checked_views.update(views)

    def _write_artifact(self, reference: ArtifactReference, content: bytes) -> None:
        self._store(content, self.artifact_path(reference))

    def _store(self, content: bytes, path: Optional[Path] = None) -> None:
        """Write content-addressed bytes once; existing equal bytes are kept."""
        if path is None:
            path = self.artifacts_directory / ContentId.of(content).digest
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                if stream.write(content) != len(content):
                    raise OSError("short Journey artifact write")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes() != content:
                # Preserve the existing implicit FileExistsError context.
                raise JourneyValidationError(  # noqa: B904
                    "content-addressed artifact path contains different bytes"
                )

    def _validate_reference(
        self, reference: ArtifactReference | Dict[str, Any]
    ) -> ArtifactReference:
        try:
            validated = (
                reference
                if isinstance(reference, ArtifactReference)
                else ArtifactReference.model_validate(reference)
            )
        except ValidationError as error:
            raise JourneyValidationError("invalid artifact reference") from error
        if validated.journey_id not in self._lineage_ids:
            raise JourneyValidationError(
                "artifact reference belongs to another Journey"
            )
        return validated

    def _verified_artifact_bytes(self, reference: ArtifactReference) -> bytes:
        content = self._artifact_contents.get(reference.artifact_id)
        if content is None:
            path = self.artifact_path(reference)
            try:
                content = path.read_bytes()
            except FileNotFoundError as error:
                raise JourneyValidationError(
                    "referenced artifact is missing"
                ) from error
            if ContentId.parse(reference.artifact_id) != ContentId.of(content):
                raise JourneyValidationError(
                    "referenced artifact digest does not match"
                )
        if reference.byte_count != len(content):
            raise JourneyValidationError(
                "referenced artifact byte count does not match"
            )
        self._artifact_contents[reference.artifact_id] = content
        return content

    def _validate_sequence(
        self,
        events: Iterable[JourneyEvent],
        *,
        verify_artifacts: bool,
        inherited: Optional[int] = None,
    ) -> None:
        seen: Dict[str, JourneyEvent] = {}
        previous = None
        for index, raw_event in enumerate(events):
            try:
                event = (
                    raw_event
                    if isinstance(raw_event, JourneyEvent)
                    else JourneyEvent.model_validate(raw_event)
                )
            except ValidationError as error:
                raise JourneyValidationError("invalid Journey event") from error
            self._check_envelope(
                index,
                event,
                seen,
                verify_artifacts=verify_artifacts,
                previous=previous,
                inherited=inherited,
            )
            seen[event.event_id] = event
            previous = event

    def _check_envelope(
        self,
        index: int,
        event: JourneyEvent,
        seen: Mapping[str, JourneyEvent],
        *,
        verify_artifacts: bool,
        previous: Optional[JourneyEvent] = None,
        inherited: Optional[int] = None,
    ) -> None:
        """Check one event against the admitted events before it.

        A Fork's inherited prefix keeps each ancestor's identity: the identity
        changes only at a Fork origin, and this file's events carry its own.
        """
        inherited = self._inherited if inherited is None else inherited
        forked = event.event_type == FORK_ORIGIN_EVENT_TYPE
        if index >= inherited:
            if event.journey_id != self.journey_id or forked != (
                inherited > 0 and index == inherited
            ):
                raise JourneyValidationError(
                    "Journey identity changed inside event stream"
                )
        elif (
            previous is not None
            and not forked
            and (event.journey_id != previous.journey_id)
        ):
            raise JourneyValidationError("Journey identity changed inside event stream")
        if forked and (
            previous is None
            or event.parent_ids != [previous.event_id]
            or len(event.artifact_references) != 1
        ):
            raise JourneyValidationError("Fork origin must rest on its parent head")
        if index == 0 and event.parent_ids:
            raise JourneyValidationError("first Journey event cannot have parents")
        for parent_id in event.parent_ids:
            if parent_id not in seen:
                raise JourneyValidationError(
                    "event parent is not earlier in the Journey"
                )
        if event.event_id in seen:
            raise JourneyValidationError("event ID was reused")
        for reference in event.artifact_references:
            source_event = (
                event
                if reference.source_event_id == event.event_id
                else seen.get(reference.source_event_id)
            )
            if source_event is not None and reference.journey_id != (
                source_event.journey_id
            ):
                raise JourneyValidationError(
                    "artifact reference belongs to another Journey"
                )
            if source_event is not event:
                if source_event is None:
                    raise JourneyValidationError(
                        "artifact source event is not in the Journey lineage"
                    )
                if reference not in source_event.artifact_references:
                    raise JourneyValidationError(
                        "artifact source event did not retain this reference"
                    )
            if verify_artifacts:
                self._verified_artifact_bytes(reference)


def has_ancestor(
    events: list[dict[str, Any]], *, ancestor_id: str, descendant_id: str
) -> bool:
    by_id = {event["event_id"]: event for event in events}
    pending = [descendant_id]
    seen = set()
    while pending:
        selected = pending.pop()
        if selected == ancestor_id:
            return True
        if selected in seen:
            continue
        seen.add(selected)
        event = by_id.get(selected)
        if event is not None:
            pending.extend(event["parent_ids"])
    return False
