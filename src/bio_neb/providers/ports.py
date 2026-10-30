"""Layout-blind APP capabilities; providers never receive the Journey writer.

Mutations take deterministic names/keys supplied by the plan. Implementations
look up that name before creating and report the resulting resource count,
including when an earlier response was lost. No port grants deletion or cancel.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, JsonValue, StrictStr

from bio_neb.contracts import FrozenContract, StrictFrozenContract
from bio_neb.verified_artifacts import ArtifactProof

PortName = Literal["store", "jobs", "identity"]
VerificationMode = Literal["reuse_creation_proof", "independent_full_read"]


class TransformedStream(Protocol):
    """A reader of a source rewritten in flight, and the source's measured identity."""

    def read(self, size: int = -1) -> bytes: ...

    def source_identity(self) -> tuple[int, str]: ...


class StreamTransform(Protocol):
    """A consumer-owned rewrite of a streamed source (ADR 0020: the operation owns
    what it rewrites; the transport owns upload and integrity). ``placed_byte_count``
    is the placed size when it is known in advance, else ``None`` (the transport then
    places ``whole`` of the source)."""

    placed_byte_count: int | None

    def open(self, source: Any) -> TransformedStream: ...

    def whole(self, data: bytes) -> bytes: ...


@dataclass(frozen=True)
class TransferSource:
    """A selected locator, expected size and optional independent checksum.

    ``sha256`` is an independent SHA-256 expectation. ``md5`` is a publisher's md5,
    checked while streaming; the SHA-256 is then derived. With
    ``measure_gzip_content`` the upload also decompresses the stream and reports
    the decompressed size and SHA-256 as acquired values: those of the FASTA without
    the records named in ``excluded_sequences`` (the Run's reference sequences, #464).
    With ``transform`` the upload is the source rewritten in flight (a
    `StreamTransform`; #464's contig renames): ``byte_count`` is the source's, and
    the receipt names the source's measured size and SHA-256 (``source``) beside the
    placed object's.
    """

    kind: Literal["local", "url"]
    locator: str
    byte_count: int
    sha256: str | None
    md5: str | None = None
    measure_gzip_content: bool = False
    excluded_sequences: tuple[str, ...] = ()
    transform: StreamTransform | None = None

    def __post_init__(self):
        if (
            self.kind not in {"local", "url"}
            or not self.locator
            or self.byte_count <= 0
            or not _optional_hex(self.sha256, 64)
            or not _optional_hex(self.md5, 32)
            or (self.sha256 is not None and self.md5 is not None)
        ):
            raise ValueError(
                "transfer source requires a locator, positive size and at most one valid checksum"
            )
        if self.transform is not None and (
            self.sha256 is not None or self.md5 is not None or self.measure_gzip_content
        ):
            raise ValueError("a transformed source carries no checksum and no measure")


def _optional_hex(value: str | None, length: int) -> bool:
    return value is None or (
        len(value) == length and all(c in "0123456789abcdef" for c in value)
    )


class ContactReceipt(StrictFrozenContract):
    """A small receipt locator; provider bodies and credentials stay outside rows."""

    kind: Literal["transfer_proof", "job_submission", "manifest_publication"]
    ref: StrictStr = Field(min_length=1)


class Receipt(FrozenContract):
    """Secret-free result; the engine binds its journal proof locator.

    fresh_bucket/submit must attest resource_count=1, including named reuse.
    Reads and object transfers may omit this bucket/job resource count.
    """

    values: dict[str, JsonValue] = Field(default_factory=dict)
    provider_ids: tuple[str, ...] = ()
    resource_count: int | None = Field(default=None, ge=0)
    contact: ContactReceipt | None = None
    content: bytes | None = None
    proof_ref: str | None = None


@runtime_checkable
class ObjectStore(Protocol):
    def fresh_bucket(self, name: str) -> Receipt: ...
    def put(
        self, bucket: str, key: str, content: bytes | TransferSource
    ) -> Receipt: ...
    def copy(self, source: ArtifactProof, bucket: str, key: str) -> Receipt: ...
    def head(
        self,
        bucket: str,
        key: str,
        *,
        verification: VerificationMode = "reuse_creation_proof",
    ) -> Receipt: ...
    def read_small(self, bucket: str, key: str) -> Receipt: ...


@runtime_checkable
class JobRunner(Protocol):
    def submit(self, name: str, request: dict[str, JsonValue]) -> Receipt: ...
    def observe(self, job_id: str, *, window: int) -> Receipt: ...


@runtime_checkable
class Identity(Protocol):
    def whoami(self) -> Receipt: ...


@dataclass(frozen=True)
class Ports:
    store: ObjectStore
    jobs: JobRunner
    identity: Identity


class Provider(Protocol):
    store: ObjectStore
    jobs: JobRunner
    identity: Identity

    def for_step(self, stage: str, step: str) -> Ports:
        """Bind contact attribution; all returned ports share provider state."""
        ...
