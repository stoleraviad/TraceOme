"""Consequence-aware Nebius Object Storage mechanics for host adapters."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)
from botocore.response import StreamingBody
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError

from bio_neb.failure import (
    ResolvedStageLimits,
    UnboundStageLimits,
    bound_operation_policy,
    bound_part_limits,
    operation_retry_decision,
)
from bio_neb.providers.nebius_common import (
    AwsCredentialMaterial,
    ExecutionContactObserver,
    observe_execution_contact_finish,
    observe_execution_contact_start,
)
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ABSENT_CODES = {
    "404",
    "NoSuchKey",
    "NotFound",
    "NoSuchBucket",
    "NoSuchLifecycleConfiguration",
}
_ACCESS_CODES = {
    "AccessDenied",
    "ExpiredToken",
    "InvalidAccessKeyId",
    "InvalidToken",
    "SignatureDoesNotMatch",
    "Unauthorized",
}
_COLLISION_CODES = {
    "409",
    "412",
    "ConditionalRequestConflict",
    "OperationAborted",
    "PreconditionFailed",
}
_TRANSIENT_CODES = {
    "408",
    "429",
    "500",
    "502",
    "503",
    "504",
    "InternalError",
    "RequestTimeout",
    "RequestTimeoutException",
    "ServiceUnavailable",
    "SlowDown",
    "Throttling",
    "ThrottlingException",
}

_S3_EXECUTION_CONTACT_OPERATIONS = {
    "head_object": "s3.head-object",
    "get_object": "s3.get-object",
    "list_objects_v2": "s3.list-objects-v2",
    "list_multipart_uploads": "s3.list-multipart-uploads",
    "put_object": "s3.put-object",
    "get_bucket_lifecycle_configuration": ("s3.get-bucket-lifecycle-configuration"),
    "put_bucket_lifecycle_configuration": ("s3.put-bucket-lifecycle-configuration"),
    "create_multipart_upload": "s3.create-multipart-upload",
    "upload_part": "s3.upload-part",
    "upload_part_copy": "s3.upload-part-copy",
    "complete_multipart_upload": "s3.complete-multipart-upload",
    "copy_object": "s3.copy-object",
}

# Part policy. Long single connections are reset by the host path, so an object
# larger than one part moves as parts, uploaded from memory or downloaded as ranged
# GETs. The SDK client performs one attempt per request (`total_max_attempts: 1`);
# each part is retried only for a transient fault, by the failure policy of the
# stage it runs in (`_part_attempts`). Each part attempt has its stage's part
# deadline (`failure.StageLimits`): it ends within the deadline plus at most one
# socket operation the client's timeouts bound, and never succeeds after it.
MULTIPART_PART_BYTES = 32 * 1024 * 1024
_CONNECT_TIMEOUT_SECONDS = 30
_READ_TIMEOUT_SECONDS = 300

ReadOperation = Literal[
    "head",
    "get",
    "list_objects",
    "list_multipart",
    "get_lifecycle",
]
MutationOperation = Literal[
    "conditional_put",
    "conditional_copy",
    "put_lifecycle",
    "multipart_create",
    "multipart_part",
    "multipart_complete",
]
Operation = ReadOperation | MutationOperation
ReadClassification = Literal[
    "success", "transient", "absent", "access", "malformed", "transfer"
]
SafeMutationFailureCategory = Literal[
    "transport_interrupted",
    "provider_rejected",
    "checksum_mismatch",
]


# Failures a source stream can raise mid-part: local files and plain streams
# raise OSError; a fetched SDK response body raises botocore's stream errors.
_SOURCE_STREAM_ERRORS = (
    OSError,
    IncompleteReadError,
    ResponseStreamingError,
    ReadTimeoutError,
)


@dataclass(frozen=True, slots=True)
class SafeProviderDetail:
    """Bounded provider status/code only; response text and headers never escape."""

    status: int | None = None
    code: str | None = None


@dataclass(frozen=True, slots=True)
class ReadContact:
    operation: ReadOperation
    ordinal: int
    classification: ReadClassification
    detail: SafeProviderDetail = SafeProviderDetail()


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ReadResult(Generic[T]):
    value: T
    contacts: tuple[ReadContact, ...]


@dataclass(frozen=True, slots=True)
class ObjectIdentity:
    byte_count: int
    sha256: str


class PartsDownload(BaseModel):
    """One object downloaded as ranged parts, proved by its streamed digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    credential_receipt: ObjectStorageCredentialReuseReceipt
    part_bytes: StrictInt = Field(gt=0)
    part_count: StrictInt = Field(gt=0)


class PartEndCause(BaseModel):
    """Why one part attempt ended: success, its deadline, or its failure's safe detail.

    ``status`` and ``code`` are the provider's bounded answer; ``error_type`` is
    only the failure's class name, since a socket fault carries no status.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["success", "deadline", "provider", "transport", "malformed"]
    status: StrictInt | None = None
    code: StrictStr | None = None
    error_type: StrictStr | None = None


class PartObserver(Protocol):
    """Sees a download's part limits, then each part attempt start and end.

    ``opened`` precedes any contact; each attempt is seen before its request and
    its end, with its cause, before any retry.
    """

    def opened(self, limits: ResolvedStageLimits | UnboundStageLimits) -> None: ...

    def started(
        self, part: int, attempt: int, offset: int, byte_count: int
    ) -> None: ...

    def ended(
        self,
        part: int,
        attempt: int,
        classification: str,
        byte_count: int | None,
        cause: PartEndCause,
    ) -> None: ...


class _Unobserved:
    def opened(self, limits: ResolvedStageLimits | UnboundStageLimits) -> None:
        return None

    def started(self, part: int, attempt: int, offset: int, byte_count: int) -> None:
        return None

    def ended(
        self,
        part: int,
        attempt: int,
        classification: str,
        byte_count: int | None,
        cause: PartEndCause,
    ) -> None:
        return None


class PartDeadlinePassed(TimeoutError):
    """A part attempt reached its total deadline; like a timeout, it is transient."""


class _PartDeadline:
    """One part attempt's total deadline, on the mechanics' monotonic clock.

    Each phase of the attempt is checked against it, and a body is read or sent
    under it, so the attempt ends within the deadline plus at most one bounded
    socket operation (30 s connect, 300 s read), and never succeeds after it
    (owner, #418 a). A response whose headers trickle stays inside the SDK's
    parsing until it returns, then is refused: a known limit (owner, #418 i).
    """

    def __init__(self, seconds: int, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._ends = clock() + seconds

    def remaining(self) -> float:
        """Seconds left; a part with none left raises `PartDeadlinePassed`."""
        left = self._ends - self._clock()
        if left <= 0:
            raise PartDeadlinePassed("object part deadline passed")
        return left

    def passed(self) -> bool:
        return self._ends - self._clock() <= 0


class _DeadlineBody(io.BytesIO):
    """An upload part's bytes, refused to the sender once its deadline passes.

    The SDK sends a part by reading its body in blocks; a send that trickles
    is ended at its next read past the deadline.
    """

    def __init__(self, chunk: bytes, deadline: _PartDeadline) -> None:
        super().__init__(chunk)
        self._deadline = deadline

    def read(self, size: int | None = -1) -> bytes:
        self._deadline.remaining()
        return super().read(size)


class _PartTransient(Exception):
    """A part attempt's transient fault, for the shared part retry to decide."""

    def __init__(self, error: NebiusObjectStorageError | Exception) -> None:
        super().__init__(str(error))
        self.error = error


def _part_attempts(
    attempt_once: Callable[[int, _PartDeadline], T],
    *,
    part_seconds: int,
    clock: Callable[[], float],
) -> T:
    """Run one part, retried only for a transient fault by the bound stage policy.

    Upload and download share this rule (owner, #363): the failure module decides
    each retry and its backoff; an exhausted part raises its last fault. Each
    attempt has its own deadline of ``part_seconds`` (owner, #418); one that
    passes it ends transient, so the same policy retries it.
    """
    policy = bound_operation_policy()
    ordinal = 0
    while True:
        ordinal += 1
        try:
            return attempt_once(ordinal, _PartDeadline(part_seconds, clock))
        except _PartTransient as transient:
            _, delay = operation_retry_decision("transient", ordinal, policy)
            if delay is None:
                raise transient.error from transient.error.__cause__
            time.sleep(random.uniform(0, delay))


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    identity: ObjectIdentity
    custom_metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ObservedObject:
    """An object's size and version facts, without a controlled digest."""

    byte_count: int
    etag: str
    modified_at_ns: int


@dataclass(frozen=True, slots=True)
class ListedObject:
    object_key: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class MultipartResidue:
    """UploadId is opaque state and never content or version authority."""

    object_key: str
    upload_id: str


@dataclass(frozen=True, slots=True)
class StorageClosure:
    objects: tuple[ListedObject, ...]
    multipart: tuple[MultipartResidue, ...]
    contacts: tuple[ReadContact, ...]


@dataclass(frozen=True, slots=True)
class _LostCompletion:
    """A multipart completion whose response was lost, with its head reconciliation."""

    error: BaseException
    detail: SafeProviderDetail
    matched: bool
    contacts: tuple[ReadContact, ...]


@dataclass(frozen=True, slots=True)
class ConditionalWriteResult:
    identity: ObjectIdentity
    reconciled_after_response_loss: bool
    reconciliation_contacts: tuple[ReadContact, ...] = ()


@dataclass(frozen=True, slots=True)
class ReuseOrWriteResult:
    identity: ObjectIdentity
    disposition: Literal["reused_exact", "written", "reconciled_after_response_loss"]
    contacts: tuple[ReadContact, ...] = ()


@dataclass(frozen=True, slots=True)
class LifecycleWriteResult:
    configuration: dict[str, Any]
    reconciled_after_response_loss: bool
    reconciliation_contacts: tuple[ReadContact, ...] = ()


@dataclass(frozen=True, slots=True)
class ProducerStreamWriteResult:
    byte_count: int
    approved_algorithm: str
    approved_checksum: str
    sha256: str
    reconciled_after_response_loss: bool
    reconciliation_contacts: tuple[ReadContact, ...] = ()


class NebiusObjectStorageError(RuntimeError):
    kind = "storage_error"

    def __init__(
        self,
        operation: Operation,
        *,
        detail: SafeProviderDetail = SafeProviderDetail(),  # noqa: B008 - Frozen detail; preserve the existing shared default.
        contacts: tuple[ReadContact, ...] = (),
        residue: tuple[MultipartResidue, ...] = (),
        bytes_read: int | None = None,
        safe_error_category: SafeMutationFailureCategory | None = None,
    ) -> None:
        super().__init__(f"{operation}:{self.kind}")
        self.operation = operation
        self.detail = detail
        self.contacts = contacts
        self.residue = residue
        self.bytes_read = bytes_read
        self.safe_error_category = safe_error_category


class StorageReadAbsent(NebiusObjectStorageError):
    kind = "read_absent"


class StorageConditionalCollision(NebiusObjectStorageError):
    kind = "conditional_collision"


class StorageAccessError(NebiusObjectStorageError):
    kind = "access"


class StorageMalformedResponse(NebiusObjectStorageError):
    kind = "malformed_response"


class StorageTransferError(NebiusObjectStorageError):
    kind = "transfer_failure"


class StorageMutationRejected(NebiusObjectStorageError):
    kind = "mutation_rejected"


class StorageMutationAmbiguous(NebiusObjectStorageError):
    kind = "mutation_ambiguous"


class StorageMultipartResidue(NebiusObjectStorageError):
    kind = "multipart_residue"


class ClientFactory(Protocol):
    def __call__(
        self,
        endpoint_url: str,
        region: str,
        credentials: AwsCredentialMaterial,
    ) -> Any: ...


def nebius_object_storage_client(
    endpoint_url: str,
    region: str,
    credentials: AwsCredentialMaterial,
) -> Any:
    # boto3/S3/AWS fields below are Nebius-compatible wire grammar, not product ownership.
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key,
        aws_session_token=credentials.session_token,
        config=Config(
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            read_timeout=_READ_TIMEOUT_SECONDS,
            # One SDK attempt per request: the mechanics decide what may be
            # repeated (multipart parts) and what must be reconciled instead.
            retries={"total_max_attempts": 1, "mode": "standard"},
        ),
    )


def controlled_sha256(
    metadata: Mapping[object, object], *, expected: str | None = None
) -> str:
    """Extract the sole controlled digest while ignoring unrelated custom metadata."""

    matches = [
        value
        for key, value in metadata.items()
        if isinstance(key, str) and key.casefold() == "checksum-sha256"
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError("controlled checksum metadata must occur exactly once")
    digest = matches[0]
    if _SHA256.fullmatch(digest) is None:
        raise ValueError("controlled checksum metadata must be lowercase SHA-256")
    if expected is not None and digest != _validated_sha256(expected):
        raise ValueError("controlled checksum metadata mismatch")
    return digest


def _validated_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def _controlled_metadata(
    digest: str, custom_metadata: Mapping[str, str] | None
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in (custom_metadata or {}).items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError(
                "custom metadata must contain non-empty string keys and values"
            )
        if key.casefold() == "checksum-sha256":
            raise ValueError("custom metadata must not set the controlled checksum")
        metadata[key] = value
    metadata["checksum-sha256"] = _validated_sha256(digest)
    return metadata


def _uncontrolled_custom_metadata(
    custom_metadata: Mapping[str, str] | None,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in (custom_metadata or {}).items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError(
                "custom metadata must contain non-empty string keys and values"
            )
        if key.casefold() == "checksum-sha256":
            raise ValueError("custom metadata must not set the controlled checksum")
        metadata[key] = value
    return metadata


@dataclass(frozen=True, slots=True)
class _ApprovedStream:
    """What a streamed upload must turn out to be, in the authority's algorithm.

    ``identity`` is the controlled SHA-256 identity when the authority algorithm is
    SHA-256, and ``None`` otherwise: then SHA-256 is derived while streaming and is
    known only after the last byte, so the stored object carries no controlled
    checksum metadata and a lost response cannot be reconciled by a head.
    """

    byte_count: int
    algorithm: str
    checksum: str | None

    @classmethod
    def from_identity(cls, identity: ObjectIdentity) -> _ApprovedStream:
        return cls(identity.byte_count, "sha256", identity.sha256)

    @property
    def identity(self) -> ObjectIdentity | None:
        if self.algorithm != "sha256" or self.checksum is None:
            return None
        return ObjectIdentity(self.byte_count, self.checksum)

    def metadata(self, custom_metadata: Mapping[str, str] | None) -> dict[str, str]:
        if self.identity is None:
            return _uncontrolled_custom_metadata(custom_metadata)
        return _controlled_metadata(self.checksum, custom_metadata)


class _StreamDigest:
    """Byte count, SHA-256 and the approved digest over the bytes actually consumed."""

    def __init__(self, approved_algorithm: str = "sha256") -> None:
        self.byte_count = 0
        self.digest = hashlib.sha256()
        self.approved = (
            self.digest
            if approved_algorithm == "sha256"
            else hashlib.new(approved_algorithm)
        )

    def update(self, chunk: bytes) -> None:
        self.byte_count += len(chunk)
        self.digest.update(chunk)
        if self.approved is not self.digest:
            self.approved.update(chunk)

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()

    def matches(self, approved: _ApprovedStream) -> bool:
        return self.byte_count == approved.byte_count and (
            approved.checksum is None or self.approved.hexdigest() == approved.checksum
        )


class _HashingReader:
    """Digest the bytes the client actually consumes from a single-put body."""

    def __init__(self, source: Any, digest: _StreamDigest) -> None:
        self._source = source
        self.digest = digest

    def read(self, size: int = -1) -> bytes:
        chunk = self._source.read(size)
        if not isinstance(chunk, bytes):
            raise OSError("source returned non-bytes")
        self.digest.update(chunk)
        return chunk


@dataclass(frozen=True, slots=True)
class _StreamedUpload:
    """One stream placed once, with the digest over every byte that was sent."""

    digest: _StreamDigest
    reconciled_after_response_loss: bool = False
    reconciliation_contacts: tuple[ReadContact, ...] = ()


@dataclass(frozen=True, slots=True)
class _CopySourcePart:
    """One multipart part the provider copies itself from a source byte range.

    A part source variant of the client-sent ``bytes`` chunk: the same part
    loop uploads it through ``UploadPartCopy`` instead of ``UploadPart``, so no
    byte of it crosses the client. Its identity is the source object's
    controlled identity checked before the copy, not a digest over bytes.
    """

    bucket: str
    key: str
    first_byte: int
    last_byte: int

    @property
    def byte_count(self) -> int:
        return self.last_byte - self.first_byte + 1

    @property
    def copy_source(self) -> dict[str, str]:
        return {"Bucket": self.bucket, "Key": self.key}

    @property
    def copy_source_range(self) -> str:
        return f"bytes={self.first_byte}-{self.last_byte}"


def _provider_detail(error: BaseException) -> SafeProviderDetail:
    if not isinstance(error, ClientError):
        return SafeProviderDetail()
    response = error.response if isinstance(error.response, dict) else {}
    metadata = response.get("ResponseMetadata", {})
    error_data = response.get("Error", {})
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    code = error_data.get("Code") if isinstance(error_data, dict) else None
    return SafeProviderDetail(
        status=status
        if (
            isinstance(status, int)
            and not isinstance(status, bool)
            and 100 <= status <= 599
        )
        else None,
        code=str(code)[:80] if code is not None else None,
    )


def _is_absent(detail: SafeProviderDetail) -> bool:
    return detail.status == 404 or detail.code in _ABSENT_CODES


def _is_access(detail: SafeProviderDetail) -> bool:
    return detail.status in {401, 403} or detail.code in _ACCESS_CODES


def _is_collision(detail: SafeProviderDetail) -> bool:
    return detail.status in {409, 412} or detail.code in _COLLISION_CODES


def _is_transient(error: BaseException, detail: SafeProviderDetail) -> bool:
    return (
        isinstance(error, (BotoCoreError, OSError, TimeoutError))
        or (
            detail.status is not None
            and (detail.status in {408, 429} or detail.status >= 500)
        )
        or detail.code in _TRANSIENT_CODES
    )


def _read_within(stream: Any, size: int, deadline: _PartDeadline) -> bytes:
    """At most one socket receive, waiting no longer than the part's time left.

    The SDK body's ``read(n)`` waits for all ``n`` bytes, so a trickle would
    outlive any per-read timeout; its urllib3 stream's ``read1`` receives once.
    """
    remaining = deadline.remaining()
    if not isinstance(stream, StreamingBody):
        return stream.read1(size)
    raw = stream._raw_stream
    # A response read to its length has released its socket; only EOF is left.
    if not raw.closed:
        stream.set_socket_timeout(min(_READ_TIMEOUT_SECONDS, remaining))
    try:
        return raw.read1(size)
    except URLLib3ReadTimeoutError as error:
        raise ReadTimeoutError(endpoint_url=error.url, error=error) from error
    except URLLib3ProtocolError as error:
        raise ResponseStreamingError(error=error) from error


def _part_end_cause(
    error: NebiusObjectStorageError, deadline: _PartDeadline
) -> PartEndCause:
    """A failed part attempt's cause: its deadline, or its failure's safe detail."""
    detail = error.contacts[-1].detail if error.contacts else SafeProviderDetail()
    original = error.__cause__
    error_type = None if original is None else type(original).__name__
    if (
        error.contacts
        and error.contacts[-1].classification == "transient"
        and (deadline.passed())
    ):
        return PartEndCause(
            kind="deadline",
            error_type=None if isinstance(original, PartDeadlinePassed) else error_type,
        )
    kind = (
        "provider"
        if isinstance(original, ClientError)
        else "malformed"
        if isinstance(error, StorageMalformedResponse)
        else "transport"
    )
    return PartEndCause(
        kind=kind, status=detail.status, code=detail.code, error_type=error_type
    )


class _ObservedS3Client:
    """Wrap only actual SDK method dispatch; client construction is not a request."""

    def __init__(
        self,
        client: Any,
        *,
        observer: ExecutionContactObserver,
    ) -> None:
        self._client = client
        self._observer = observer

    def __getattr__(self, name: str) -> Any:
        selected = getattr(self._client, name)
        if not callable(selected):
            return selected

        def observed(*args: Any, **kwargs: Any) -> Any:
            operation = _S3_EXECUTION_CONTACT_OPERATIONS.get(
                name, "s3.allowlisted-other"
            )
            unit_id = observe_execution_contact_start(
                self._observer,
                category="storage",
                operation=operation,
                executable_identity="s3-sdk",
            )
            try:
                result = selected(*args, **kwargs)
            except BaseException as error:
                detail = _provider_detail(error)
                classification = (
                    "absent"
                    if _is_absent(detail)
                    else "access"
                    if _is_access(detail)
                    else "transient"
                    if _is_transient(error, detail)
                    else "provider_error"
                )
                observe_execution_contact_finish(
                    self._observer,
                    unit_id,
                    disposition="failed",
                    exit_status=detail.status,
                    parse_classification=classification,
                )
                raise
            observe_execution_contact_finish(
                self._observer,
                unit_id,
                disposition="succeeded",
                exit_status=0,
                parse_classification="response_received",
            )
            return result

        return observed


class NebiusObjectStorageMechanics:
    """One bucket's client policy, reads, pagination, and no-repeat mutations."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        bucket: str,
        credential_resolver: Callable[[], AwsCredentialMaterial],
        client_factory: ClientFactory = nebius_object_storage_client,
        contact_observer: Callable[[ReadContact], None] | None = None,
        execution_contact_observer: ExecutionContactObserver | None = None,
        multipart_part_bytes: int = MULTIPART_PART_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not endpoint_url or not region or not bucket:
            raise ValueError("endpoint_url, region, and bucket are required")
        if not isinstance(multipart_part_bytes, int) or multipart_part_bytes <= 0:
            raise ValueError("multipart_part_bytes must be a positive byte count")
        self._multipart_part_bytes = multipart_part_bytes
        self._endpoint_url = endpoint_url
        self._region = region
        self._bucket = bucket
        self._credential_resolver = credential_resolver
        self._client_factory = client_factory
        self._contact_observer = contact_observer
        self._execution_contact_observer = execution_contact_observer
        self._clock = clock

    def _client(self, operation: Operation) -> Any:
        try:
            credentials = self._credential_resolver()
            client = self._client_factory(self._endpoint_url, self._region, credentials)
            return (
                _ObservedS3Client(
                    client,
                    observer=self._execution_contact_observer,
                )
                if self._execution_contact_observer is not None
                else client
            )
        except NebiusObjectStorageError:
            raise
        except Exception as error:
            raise StorageAccessError(operation) from error

    def _read(
        self,
        operation: ReadOperation,
        contact: Callable[[], T],
        *,
        ordinal_offset: int = 0,
    ) -> ReadResult[T]:
        observations: list[ReadContact] = []
        for attempt in (1, 2):
            ordinal = ordinal_offset + attempt
            try:
                value = contact()
            except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
                detail = _provider_detail(error)
                if _is_absent(detail):
                    observed = ReadContact(operation, ordinal, "absent", detail)
                    observations.append(observed)
                    self._observe(observed)
                    raise StorageReadAbsent(
                        operation, detail=detail, contacts=tuple(observations)
                    ) from error
                if _is_access(detail):
                    observed = ReadContact(operation, ordinal, "access", detail)
                    observations.append(observed)
                    self._observe(observed)
                    raise StorageAccessError(
                        operation, detail=detail, contacts=tuple(observations)
                    ) from error
                if _is_transient(error, detail):
                    observed = ReadContact(operation, ordinal, "transient", detail)
                    observations.append(observed)
                    self._observe(observed)
                    if attempt == 1:
                        continue
                else:
                    observed = ReadContact(operation, ordinal, "transfer", detail)
                    observations.append(observed)
                    self._observe(observed)
                raise StorageTransferError(
                    operation, detail=detail, contacts=tuple(observations)
                ) from error
            except (KeyError, TypeError, ValueError) as error:
                observed = ReadContact(operation, ordinal, "malformed")
                observations.append(observed)
                self._observe(observed)
                raise StorageMalformedResponse(
                    operation, contacts=tuple(observations)
                ) from error
            observed = ReadContact(operation, ordinal, "success")
            observations.append(observed)
            self._observe(observed)
            return ReadResult(value=value, contacts=tuple(observations))
        raise AssertionError("read attempts exhausted without a result")

    def _read_once(
        self, operation: ReadOperation, contact: Callable[[], T]
    ) -> ReadResult[T]:
        try:
            value = contact()
        except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
            detail = _provider_detail(error)
            classification: ReadClassification = (
                "absent"
                if _is_absent(detail)
                else "access"
                if _is_access(detail)
                else "transient"
                if _is_transient(error, detail)
                else "transfer"
            )
            observed = ReadContact(operation, 1, classification, detail)
            self._observe(observed)
            error_type: type[NebiusObjectStorageError] = (
                StorageReadAbsent
                if classification == "absent"
                else StorageAccessError
                if classification == "access"
                else StorageTransferError
            )
            raise error_type(operation, detail=detail, contacts=(observed,)) from error
        except (KeyError, TypeError, ValueError) as error:
            observed = ReadContact(operation, 1, "malformed")
            self._observe(observed)
            raise StorageMalformedResponse(operation, contacts=(observed,)) from error
        observed = ReadContact(operation, 1, "success")
        self._observe(observed)
        return ReadResult(value, (observed,))

    def _observe(self, contact: ReadContact) -> None:
        if self._contact_observer is not None:
            self._contact_observer(contact)

    @staticmethod
    def _identity(response: Any, expected_sha256: str | None = None) -> ObjectIdentity:
        if not isinstance(response, dict):
            raise ValueError("head response must be an object")
        byte_count = response.get("ContentLength")
        metadata = response.get("Metadata")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
            or not isinstance(metadata, Mapping)
        ):
            raise ValueError("head response identity is malformed")
        return ObjectIdentity(
            byte_count=byte_count,
            sha256=controlled_sha256(metadata, expected=expected_sha256),
        )

    def head(
        self, object_key: str, *, expected_sha256: str | None = None
    ) -> ReadResult[ObjectIdentity]:
        client = self._client("head")
        try:
            response = self._read(
                "head",
                lambda: client.head_object(Bucket=self._bucket, Key=object_key),
            )
            try:
                identity = self._identity(response.value, expected_sha256)
            except ValueError as error:
                raise StorageMalformedResponse(
                    "head", contacts=response.contacts
                ) from error
            return ReadResult(identity, response.contacts)
        except ValueError as error:
            raise StorageMalformedResponse("head") from error

    def observe(self, object_key: str) -> ReadResult[ObservedObject]:
        """One HEAD of an object Bio Neb did not write; no digest is claimed."""
        client = self._client("head")

        def contact() -> ObservedObject:
            response = client.head_object(Bucket=self._bucket, Key=object_key)
            byte_count, etag = response["ContentLength"], response["ETag"]
            if not isinstance(byte_count, int) or not isinstance(etag, str):
                raise ValueError("head response identity is malformed")
            return ObservedObject(
                byte_count=byte_count,
                etag=etag.strip('"'),
                modified_at_ns=int(response["LastModified"].timestamp()) * 10**9,
            )

        return self._read("head", contact)

    def head_metadata(
        self, object_key: str, *, expected_sha256: str | None = None
    ) -> ReadResult[ObjectMetadata]:
        client = self._client("head")
        response = self._read(
            "head", lambda: client.head_object(Bucket=self._bucket, Key=object_key)
        )
        try:
            identity = self._identity(response.value, expected_sha256)
            metadata = response.value.get("Metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("head metadata is malformed")
            custom: list[tuple[str, str]] = []
            for key, value in metadata.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("head metadata is malformed")
                if key.casefold() != "checksum-sha256":
                    custom.append((key.casefold(), value))
            if len(custom) != len({key for key, _ in custom}):
                raise ValueError("custom metadata has duplicate case variants")
        except ValueError as error:
            raise StorageMalformedResponse(
                "head", contacts=response.contacts
            ) from error
        return ReadResult(
            ObjectMetadata(identity, tuple(sorted(custom))), response.contacts
        )

    def get_bytes(self, object_key: str, *, maximum_bytes: int) -> ReadResult[bytes]:
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive")
        client = self._client("get")

        def contact() -> bytes:
            response = client.get_object(Bucket=self._bucket, Key=object_key)
            if not isinstance(response, dict) or "Body" not in response:
                raise ValueError("get response is malformed")
            stream = response["Body"]
            try:
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = stream.read(maximum_bytes + 1 - total)
                    if not isinstance(chunk, bytes):
                        raise ValueError("get body returned non-bytes")
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise ValueError("get body is outside its bound")
                body = b"".join(chunks)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if not isinstance(body, bytes) or not body or len(body) > maximum_bytes:
                raise ValueError("get body is malformed or outside its bound")
            return body

        try:
            return self._read("get", contact)
        except ValueError as error:
            raise StorageMalformedResponse("get") from error

    def read_range(
        self, object_key: str, *, offset: int, byte_count: int
    ) -> ReadResult[bytes]:
        if offset < 0 or byte_count <= 0:
            raise ValueError(
                "range must have a non-negative offset and positive byte count"
            )
        client = self._client("get")
        return self._read(
            "get",
            lambda: self._ranged_body(
                client, object_key, offset=offset, byte_count=byte_count
            ),
        )

    def download(
        self,
        object_key: str,
        destination: Any,
        *,
        expected_byte_count: int,
        expected_sha256: str,
        observe_part: PartObserver | None = None,
    ) -> PartsDownload:
        """Download one object as the upload sends it: ranged parts, in order.

        Each part is fetched whole before it is written, so a failed attempt
        leaves no bytes behind; only that part is fetched again. One SHA-256
        streamed over every part must equal the known digest (no reread).
        """

        if expected_byte_count <= 0:
            raise ValueError("expected_byte_count must be positive")
        expected_sha256 = _validated_sha256(expected_sha256)
        observe = observe_part or _Unobserved()
        limits = bound_part_limits()
        observe.opened(limits)
        client = self._client("get")
        try:
            credential_receipt = ObjectStorageCredentialReuseReceipt.model_validate(
                self._credential_resolver.receipt
            )
        except Exception as error:
            raise StorageAccessError("get") from error
        digest = hashlib.sha256()
        offset = 0
        part = 0
        while offset < expected_byte_count:
            part += 1
            size = min(self._multipart_part_bytes, expected_byte_count - offset)

            def attempt_once(
                attempt: int,
                deadline: _PartDeadline,
                part: int = part,
                offset: int = offset,
                size: int = size,
            ) -> bytes:
                observe.started(part, attempt, offset, size)
                try:
                    body = self._read_once(
                        "get",
                        lambda: self._ranged_body(
                            client,
                            object_key,
                            offset=offset,
                            byte_count=size,
                            total=expected_byte_count,
                            deadline=deadline,
                        ),
                    ).value
                except NebiusObjectStorageError as error:
                    classification = (
                        error.contacts[-1].classification
                        if error.contacts
                        else "failed"
                    )
                    observe.ended(
                        part,
                        attempt,
                        classification,
                        None,
                        _part_end_cause(error, deadline),
                    )
                    if classification == "transient":
                        raise _PartTransient(error) from error
                    raise
                observe.ended(
                    part, attempt, "success", size, PartEndCause(kind="success")
                )
                return body

            body = _part_attempts(
                attempt_once,
                part_seconds=limits.effective.part_seconds,
                clock=self._clock,
            )
            digest.update(body)
            if destination.write(body) != size:
                raise OSError("short object destination write")
            offset += size
        if digest.hexdigest() != expected_sha256:
            raise StorageMalformedResponse(
                "get", bytes_read=offset, safe_error_category="checksum_mismatch"
            )
        return PartsDownload(
            byte_count=expected_byte_count,
            sha256=expected_sha256,
            credential_receipt=credential_receipt,
            part_bytes=self._multipart_part_bytes,
            part_count=part,
        )

    def _ranged_body(
        self,
        client: Any,
        object_key: str,
        *,
        offset: int,
        byte_count: int,
        total: int | None = None,
        deadline: _PartDeadline | None = None,
    ) -> bytes:
        """One ranged GET's exact body; ``total`` also binds the object's size.

        A part's GET reads under its ``deadline``: each read first checks it, so
        a response whose headers came after it is not read, then receives once
        and waits no longer than the time left, so a trickling body ends with it.
        """
        end = offset + byte_count - 1
        response = client.get_object(
            Bucket=self._bucket, Key=object_key, Range=f"bytes={offset}-{end}"
        )
        if not isinstance(response, dict) or "Body" not in response:
            raise ValueError("ranged get response is malformed")
        content_range = response.get("ContentRange")
        prefix = f"bytes {offset}-{end}/"
        if content_range is not None and not str(content_range).startswith(prefix):
            raise ValueError("ranged get response does not match the requested range")
        if total is not None and content_range != f"{prefix}{total}":
            raise ValueError("ranged get response names another object size")
        stream = response["Body"]
        read = (
            stream.read
            if deadline is None
            else lambda size: _read_within(stream, size, deadline)
        )
        try:
            chunks: list[bytes] = []
            received = 0
            while received < byte_count:
                chunk = read(byte_count - received)
                if not isinstance(chunk, bytes):
                    raise ValueError("ranged get body returned non-bytes")
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
            trailing = read(1)
            if not isinstance(trailing, bytes) or trailing:
                raise ValueError("ranged get body exceeds the requested range")
            body = b"".join(chunks)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        if len(body) != byte_count:
            raise ValueError("ranged get body has the wrong size")
        return body

    def full_hash(self, object_key: str) -> ReadResult[ObjectIdentity]:
        client = self._client("get")

        def contact() -> ObjectIdentity:
            response = client.get_object(Bucket=self._bucket, Key=object_key)
            if not isinstance(response, dict) or "Body" not in response:
                raise ValueError("get response is malformed")
            stream = response["Body"]
            digest = hashlib.sha256()
            byte_count = 0
            try:
                while chunk := stream.read(8 * 1024 * 1024):
                    if not isinstance(chunk, bytes):
                        raise ValueError("object stream returned non-bytes")
                    byte_count += len(chunk)
                    digest.update(chunk)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if byte_count <= 0:
                raise ValueError("object stream is empty")
            return ObjectIdentity(byte_count, digest.hexdigest())

        return self._read("get", contact)

    def list_objects(self, prefix: str) -> ReadResult[tuple[ListedObject, ...]]:
        client = self._client("list_objects")
        objects: list[ListedObject] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        contacts: list[ReadContact] = []
        while True:
            request: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if token is not None:
                request["ContinuationToken"] = token
            page_result = self._read(
                "list_objects",
                lambda request=request: client.list_objects_v2(**request),
                ordinal_offset=len(contacts),
            )
            contacts.extend(page_result.contacts)
            page = page_result.value
            try:
                if not isinstance(page, dict):
                    raise ValueError("object inventory page is malformed")
                contents = page.get("Contents", ())
                if not isinstance(contents, (list, tuple)):
                    raise ValueError("object inventory contents are malformed")
                for item in contents:
                    if not isinstance(item, dict):
                        raise ValueError("object inventory item is malformed")
                    key, size = item.get("Key"), item.get("Size")
                    if (
                        not isinstance(key, str)
                        or not key.startswith(prefix)
                        or not isinstance(size, int)
                        or isinstance(size, bool)
                        or size < 0
                    ):
                        raise ValueError("object inventory item is malformed")
                    objects.append(ListedObject(key, size))
                truncated = page.get("IsTruncated", False)
                if not isinstance(truncated, bool):
                    raise ValueError("object inventory truncation flag is malformed")
                if not truncated:
                    if len(objects) != len({item.object_key for item in objects}):
                        raise ValueError("object inventory contains duplicate keys")
                    return ReadResult(
                        tuple(sorted(objects, key=lambda item: item.object_key)),
                        tuple(contacts),
                    )
                token = page.get("NextContinuationToken")
                if not isinstance(token, str) or not token or token in seen_tokens:
                    raise ValueError("object inventory pagination is malformed")
                seen_tokens.add(token)
            except (KeyError, TypeError, ValueError) as error:
                raise StorageMalformedResponse(
                    "list_objects", contacts=tuple(contacts)
                ) from error

    def list_multipart(self, prefix: str) -> ReadResult[tuple[MultipartResidue, ...]]:
        client = self._client("list_multipart")
        residue: list[MultipartResidue] = []
        markers: tuple[str, str] | None = None
        seen_markers: set[tuple[str, str]] = set()
        contacts: list[ReadContact] = []
        while True:
            request: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if markers is not None:
                request["KeyMarker"], request["UploadIdMarker"] = markers
            page_result = self._read(
                "list_multipart",
                lambda request=request: client.list_multipart_uploads(**request),
                ordinal_offset=len(contacts),
            )
            contacts.extend(page_result.contacts)
            page = page_result.value
            try:
                if not isinstance(page, dict):
                    raise ValueError("multipart inventory page is malformed")
                uploads = page.get("Uploads", ())
                if not isinstance(uploads, (list, tuple)):
                    raise ValueError("multipart inventory is malformed")
                for item in uploads:
                    if not isinstance(item, dict):
                        raise ValueError("multipart item is malformed")
                    key, upload_id = item.get("Key"), item.get("UploadId")
                    if (
                        not isinstance(key, str)
                        or not key.startswith(prefix)
                        or not isinstance(upload_id, str)
                        or not upload_id
                    ):
                        raise ValueError("multipart item is malformed")
                    residue.append(MultipartResidue(key, upload_id))
                truncated = page.get("IsTruncated", False)
                if not isinstance(truncated, bool):
                    raise ValueError("multipart truncation flag is malformed")
                if not truncated:
                    result = tuple(
                        sorted(
                            residue, key=lambda item: (item.object_key, item.upload_id)
                        )
                    )
                    if len(result) != len(set(result)):
                        raise ValueError("multipart inventory contains duplicates")
                    return ReadResult(result, tuple(contacts))
                next_markers = (
                    page.get("NextKeyMarker"),
                    page.get("NextUploadIdMarker"),
                )
                if (
                    not all(isinstance(item, str) and item for item in next_markers)
                    or next_markers in seen_markers
                ):
                    raise ValueError("multipart pagination is malformed")
                markers = next_markers  # type: ignore[assignment]
                seen_markers.add(markers)
            except (KeyError, TypeError, ValueError) as error:
                raise StorageMalformedResponse(
                    "list_multipart", contacts=tuple(contacts)
                ) from error

    def closure(self, prefix: str) -> StorageClosure:
        objects = self.list_objects(prefix)
        multipart = self.list_multipart(prefix)
        return StorageClosure(
            objects=objects.value,
            multipart=multipart.value,
            contacts=objects.contacts + multipart.contacts,
        )

    def _reconcile_identity(
        self, object_key: str, expected: ObjectIdentity
    ) -> tuple[bool, tuple[ReadContact, ...]]:
        try:
            result = self.head(object_key, expected_sha256=expected.sha256)
        except NebiusObjectStorageError as error:
            return False, error.contacts
        return result.value == expected, result.contacts

    @staticmethod
    def _raise_mutation_error(
        operation: MutationOperation,
        error: BaseException,
        *,
        residue: tuple[MultipartResidue, ...] = (),
    ) -> None:
        detail = _provider_detail(error)
        if _is_access(detail):
            raise StorageAccessError(
                operation, detail=detail, residue=residue
            ) from error
        if _is_collision(detail):
            raise StorageConditionalCollision(
                operation, detail=detail, residue=residue
            ) from error
        if _is_transient(error, detail):
            raise StorageMutationAmbiguous(
                operation, detail=detail, residue=residue
            ) from error
        raise StorageMutationRejected(
            operation, detail=detail, residue=residue
        ) from error

    def put_conditional_bytes(
        self,
        object_key: str,
        body: bytes,
        *,
        content_type: str = "application/json",
        custom_metadata: Mapping[str, str] | None = None,
    ) -> ConditionalWriteResult:
        if not isinstance(body, bytes) or not body:
            raise ValueError("body must contain exact bytes")
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        client = self._client("conditional_put")
        try:
            client.put_object(
                Bucket=self._bucket,
                Key=object_key,
                Body=body,
                ContentLength=expected.byte_count,
                ContentType=content_type,
                Metadata=_controlled_metadata(expected.sha256, custom_metadata),
                IfNoneMatch="*",
            )
        except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
            detail = _provider_detail(error)
            if _is_collision(detail):
                raise StorageConditionalCollision(
                    "conditional_put", detail=detail
                ) from error
            if _is_access(detail):
                raise StorageAccessError("conditional_put", detail=detail) from error
            if not _is_transient(error, detail):
                raise StorageMutationRejected(
                    "conditional_put", detail=detail
                ) from error
            matched, contacts = self._reconcile_identity(object_key, expected)
            if matched:
                return ConditionalWriteResult(expected, True, contacts)
            raise StorageMutationAmbiguous(
                "conditional_put", detail=detail, contacts=contacts
            ) from error
        return ConditionalWriteResult(expected, False)

    def put_conditional_stream(
        self,
        object_key: str,
        source: Any,
        *,
        expected: ObjectIdentity | None,
        expected_byte_count: int | None = None,
        content_type: str = "application/octet-stream",
        custom_metadata: Mapping[str, str] | None = None,
    ) -> ConditionalWriteResult:
        """Establish SHA-256 in the necessary upload, comparing it when known."""
        if expected is not None:
            expected = ObjectIdentity(
                expected.byte_count, _validated_sha256(expected.sha256)
            )
            approved = _ApprovedStream.from_identity(expected)
            if (
                expected_byte_count is not None
                and expected_byte_count != expected.byte_count
            ):
                raise ValueError("conflicting expected byte counts")
        else:
            if expected_byte_count is None or expected_byte_count <= 0:
                raise ValueError("stream requires a positive expected byte count")
            approved = _ApprovedStream(expected_byte_count, "sha256", None)
        placed = self._put_stream(
            object_key,
            source,
            approved=approved,
            content_type=content_type,
            custom_metadata=custom_metadata,
        )
        return ConditionalWriteResult(
            ObjectIdentity(placed.digest.byte_count, placed.digest.sha256),
            placed.reconciled_after_response_loss,
            placed.reconciliation_contacts,
        )

    def put_conditional_authority_stream(
        self,
        object_key: str,
        source: Any,
        *,
        expected_byte_count: int,
        approved_algorithm: str,
        approved_checksum: str,
        content_type: str = "application/octet-stream",
        custom_metadata: Mapping[str, str] | None = None,
    ) -> ProducerStreamWriteResult:
        """Upload once while validating an authority checksum and deriving SHA-256.

        The authority may approve a non-SHA-256 digest (NCBI publishes md5). The
        stream is then hashed with that algorithm and with SHA-256 at the same time;
        the SHA-256 is known only after the last byte, so the stored object carries
        no controlled checksum metadata (as the single put always did) and the
        derived digest is returned for the caller's receipt.
        """

        if expected_byte_count <= 0:
            raise ValueError("expected_byte_count must be positive")
        try:
            expected_approved = hashlib.new(approved_algorithm)
        except ValueError as error:
            raise ValueError("approved_algorithm is unsupported") from error
        if (
            not isinstance(approved_checksum, str)
            or len(approved_checksum) != expected_approved.digest_size * 2
            or any(
                character not in "0123456789abcdef" for character in approved_checksum
            )
        ):
            raise ValueError("approved_checksum is malformed")
        placed = self._put_stream(
            object_key,
            source,
            approved=_ApprovedStream(
                expected_byte_count, approved_algorithm, approved_checksum
            ),
            content_type=content_type,
            custom_metadata=custom_metadata,
        )
        return ProducerStreamWriteResult(
            expected_byte_count,
            approved_algorithm,
            approved_checksum,
            placed.digest.sha256,
            placed.reconciled_after_response_loss,
            placed.reconciliation_contacts,
        )

    def _put_stream(
        self,
        object_key: str,
        source: Any,
        *,
        approved: _ApprovedStream,
        content_type: str,
        custom_metadata: Mapping[str, str] | None,
    ) -> _StreamedUpload:
        """Place one stream once: a single put up to one part, multipart above it.

        Every stream (local file, HTTPS response, anything with ``read``) enters
        here, so size alone decides the mechanics and part retry, residue,
        conditional completion and the identity check over the streamed bytes are
        not duplicated per caller. A mismatch after the whole stream was consumed
        is ``checksum_mismatch`` evidence: from a single put as
        ``StorageMutationAmbiguous``, from multipart as ``StorageMultipartResidue``
        with the uncompleted upload attached.
        """

        metadata = approved.metadata(custom_metadata)
        if approved.byte_count > self._multipart_part_bytes:
            return self._put_multipart_stream(
                object_key,
                self._stream_parts(source),
                approved=approved,
                content_type=content_type,
                metadata=metadata,
            )
        hashing_source = _HashingReader(source, _StreamDigest(approved.algorithm))
        client = self._client("conditional_put")
        try:
            client.put_object(
                Bucket=self._bucket,
                Key=object_key,
                Body=hashing_source,
                ContentLength=approved.byte_count,
                ContentType=content_type,
                Metadata=metadata,
                IfNoneMatch="*",
            )
        except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
            detail = _provider_detail(error)
            bytes_read = hashing_source.digest.byte_count
            if _is_collision(detail):
                raise StorageConditionalCollision(
                    "conditional_put", detail=detail
                ) from error
            if _is_access(detail):
                raise StorageAccessError("conditional_put", detail=detail) from error
            if not _is_transient(error, detail):
                raise StorageMutationRejected(
                    "conditional_put",
                    detail=detail,
                    bytes_read=bytes_read,
                    safe_error_category="provider_rejected",
                ) from error
            # Only a fully consumed stream with a controlled identity can be
            # reconciled by one head; anything else stays ambiguous evidence.
            identity = approved.identity
            if identity is None or not hashing_source.digest.matches(approved):
                raise StorageMutationAmbiguous(
                    "conditional_put",
                    detail=detail,
                    bytes_read=bytes_read,
                    safe_error_category="transport_interrupted",
                ) from error
            matched, contacts = self._reconcile_identity(object_key, identity)
            if matched:
                return _StreamedUpload(hashing_source.digest, True, contacts)
            raise StorageMutationAmbiguous(
                "conditional_put",
                detail=detail,
                contacts=contacts,
                bytes_read=bytes_read,
                safe_error_category="transport_interrupted",
            ) from error
        if not hashing_source.digest.matches(approved):
            raise StorageMutationAmbiguous(
                "conditional_put",
                bytes_read=hashing_source.digest.byte_count,
                safe_error_category="checksum_mismatch",
            )
        return _StreamedUpload(hashing_source.digest)

    def reuse_exact_or_put_bytes(
        self,
        object_key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        custom_metadata: Mapping[str, str] | None = None,
    ) -> ReuseOrWriteResult:
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        try:
            observed = self.head(object_key)
        except StorageReadAbsent as absent:
            pre_contacts = absent.contacts
        else:
            if observed.value != expected:
                raise StorageConditionalCollision(
                    "conditional_put", contacts=observed.contacts
                )
            return ReuseOrWriteResult(expected, "reused_exact", observed.contacts)
        written = self.put_conditional_bytes(
            object_key,
            body,
            content_type=content_type,
            custom_metadata=custom_metadata,
        )
        return ReuseOrWriteResult(
            expected,
            "reconciled_after_response_loss"
            if written.reconciled_after_response_loss
            else "written",
            pre_contacts + written.reconciliation_contacts,
        )

    def copy_conditional(
        self,
        *,
        source_bucket: str,
        source_key: str,
        destination_key: str,
        expected: ObjectIdentity,
        custom_metadata: Mapping[str, str] | None = None,
    ) -> ConditionalWriteResult:
        """Copy one proved immutable object using its retained identity.

        No bundle byte crosses the client: above one part the provider copies
        byte ranges as multipart parts (``UploadPartCopy``) through the shared
        part loop, so per-part retry, residue and the conditional completion are
        the streamed path's; up to one part it is one ``CopyObject``. The
        provider accepts but does not enforce ``If-None-Match`` on ``CopyObject``
        (issue 31 probe, 2026-09-20), so that variant first confirms the
        destination is absent and refuses a present one as a collision.
        """

        expected = ObjectIdentity(
            expected.byte_count, _validated_sha256(expected.sha256)
        )
        client = self._client("conditional_copy")
        approved = _ApprovedStream.from_identity(expected)
        metadata = approved.metadata(custom_metadata)
        content_type = "application/octet-stream"
        prechecks = ()
        if expected.byte_count > self._multipart_part_bytes:
            try:
                placed = self._put_multipart_stream(
                    destination_key,
                    self._copy_source_parts(
                        source_bucket, source_key, expected.byte_count
                    ),
                    approved=approved,
                    content_type=content_type,
                    metadata=metadata,
                )
            except NebiusObjectStorageError as error:
                error.contacts = prechecks + error.contacts
                raise
            return ConditionalWriteResult(
                expected,
                placed.reconciled_after_response_loss,
                prechecks + placed.reconciliation_contacts,
            )
        try:
            self._read(
                "head",
                lambda: client.head_object(Bucket=self._bucket, Key=destination_key),
            )
        except StorageReadAbsent as absent:
            prechecks += absent.contacts
        except NebiusObjectStorageError as error:
            error.contacts = prechecks + error.contacts
            raise
        else:
            raise StorageConditionalCollision("conditional_copy", contacts=prechecks)
        try:
            client.copy_object(
                Bucket=self._bucket,
                Key=destination_key,
                CopySource={"Bucket": source_bucket, "Key": source_key},
                MetadataDirective="REPLACE",
                Metadata=metadata,
                ContentType=content_type,
                IfNoneMatch="*",
            )
        except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
            detail = _provider_detail(error)
            if _is_collision(detail):
                raise StorageConditionalCollision(
                    "conditional_copy", detail=detail, contacts=prechecks
                ) from error
            if _is_access(detail):
                raise StorageAccessError(
                    "conditional_copy", detail=detail, contacts=prechecks
                ) from error
            if not _is_transient(error, detail):
                raise StorageMutationRejected(
                    "conditional_copy",
                    detail=detail,
                    contacts=prechecks,
                    bytes_read=0,
                    safe_error_category="provider_rejected",
                ) from error
            matched, contacts = self._reconcile_identity(destination_key, expected)
            if matched:
                return ConditionalWriteResult(expected, True, prechecks + contacts)
            raise StorageMutationAmbiguous(
                "conditional_copy",
                detail=detail,
                contacts=prechecks + contacts,
                bytes_read=0,
                safe_error_category="transport_interrupted",
            ) from error
        return ConditionalWriteResult(expected, False, prechecks)

    def _copy_source_parts(
        self, source_bucket: str, source_key: str, byte_count: int
    ) -> Iterator[_CopySourcePart]:
        """Yield the source's byte ranges as full-size parts, the last one short."""

        first_byte = 0
        while first_byte < byte_count:
            last_byte = min(first_byte + self._multipart_part_bytes, byte_count) - 1
            yield _CopySourcePart(source_bucket, source_key, first_byte, last_byte)
            first_byte = last_byte + 1

    @staticmethod
    def _canonical_lifecycle(configuration: Mapping[str, Any]) -> dict[str, Any]:
        try:
            serialized = json.dumps(
                configuration, sort_keys=True, separators=(",", ":")
            )
            result = json.loads(serialized)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "lifecycle configuration must be canonical JSON data"
            ) from error
        if not isinstance(result, dict) or not isinstance(result.get("Rules"), list):
            raise ValueError("lifecycle configuration must contain Rules")
        return {"Rules": result["Rules"]}

    def get_lifecycle(self) -> ReadResult[dict[str, Any] | None]:
        client = self._client("get_lifecycle")
        try:
            response = self._read_once(
                "get_lifecycle",
                lambda: client.get_bucket_lifecycle_configuration(Bucket=self._bucket),
            )
        except StorageReadAbsent as absent:
            if absent.detail.code != "NoSuchLifecycleConfiguration":
                raise
            return ReadResult(None, absent.contacts)
        if not isinstance(response.value, dict):
            raise StorageMalformedResponse("get_lifecycle", contacts=response.contacts)
        try:
            value = self._canonical_lifecycle(response.value)
        except ValueError as error:
            raise StorageMalformedResponse(
                "get_lifecycle", contacts=response.contacts
            ) from error
        return ReadResult(value, response.contacts)

    def put_lifecycle_once(
        self, configuration: Mapping[str, Any]
    ) -> LifecycleWriteResult:
        expected = self._canonical_lifecycle(configuration)
        client = self._client("put_lifecycle")
        try:
            client.put_bucket_lifecycle_configuration(
                Bucket=self._bucket, LifecycleConfiguration=expected
            )
        except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
            detail = _provider_detail(error)
            if _is_access(detail):
                raise StorageAccessError("put_lifecycle", detail=detail) from error
            if not _is_transient(error, detail):
                raise StorageMutationRejected("put_lifecycle", detail=detail) from error
            try:
                observed = self.get_lifecycle()
            except NebiusObjectStorageError as read_error:
                raise StorageMutationAmbiguous(
                    "put_lifecycle", detail=detail, contacts=read_error.contacts
                ) from error
            if observed.value == expected:
                return LifecycleWriteResult(expected, True, observed.contacts)
            raise StorageMutationAmbiguous(
                "put_lifecycle", detail=detail, contacts=observed.contacts
            ) from error
        return LifecycleWriteResult(expected, False)

    def publish_manifest_last(
        self,
        *,
        prefix: str,
        manifest_key: str,
        body: bytes,
        expected_pre_inventory: tuple[ListedObject, ...],
    ) -> ConditionalWriteResult:
        pre = self.closure(prefix)
        if pre.objects != tuple(
            sorted(expected_pre_inventory, key=lambda item: item.object_key)
        ):
            raise StorageMalformedResponse("list_objects", contacts=pre.contacts)
        if pre.multipart:
            raise StorageMultipartResidue(
                "list_multipart", contacts=pre.contacts, residue=pre.multipart
            )
        try:
            self.head(manifest_key)
        except StorageReadAbsent:
            pass
        else:
            raise StorageConditionalCollision("conditional_put")
        result = self.put_conditional_bytes(manifest_key, body)
        exact = self.get_bytes(manifest_key, maximum_bytes=len(body))
        if exact.value != body:
            raise StorageMalformedResponse("get", contacts=exact.contacts)
        final = self.closure(prefix)
        expected_final = tuple(
            sorted(
                (*expected_pre_inventory, ListedObject(manifest_key, len(body))),
                key=lambda item: item.object_key,
            )
        )
        if final.objects != expected_final:
            raise StorageMalformedResponse("list_objects", contacts=final.contacts)
        if final.multipart:
            raise StorageMultipartResidue(
                "list_multipart", contacts=final.contacts, residue=final.multipart
            )
        return result

    def multipart_upload(
        self,
        *,
        object_key: str,
        chunks: Iterable[bytes],
        expected: ObjectIdentity,
        prefix: str,
        expected_final_inventory: tuple[ListedObject, ...],
        content_type: str = "application/octet-stream",
    ) -> ConditionalWriteResult:
        expected = ObjectIdentity(
            expected.byte_count, _validated_sha256(expected.sha256)
        )
        approved = _ApprovedStream.from_identity(expected)
        client = self._client("multipart_create")
        upload_id = self._create_multipart(
            client,
            object_key,
            residue_prefix=prefix,
            content_type=content_type,
            metadata={"checksum-sha256": expected.sha256},
        )
        residue = (MultipartResidue(object_key, upload_id),)
        parts = self._upload_parts(
            client,
            object_key,
            upload_id,
            chunks,
            digest=_StreamDigest(),
            approved=approved,
            residue=residue,
        )
        lost = self._complete_multipart(
            client, object_key, upload_id, parts, approved=approved, residue=residue
        )
        if lost is not None:
            try:
                closure = self.closure(prefix)
            except NebiusObjectStorageError as read_error:
                raise StorageMutationAmbiguous(
                    "multipart_complete",
                    detail=lost.detail,
                    contacts=lost.contacts + read_error.contacts,
                    residue=residue,
                ) from lost.error
            expected_inventory = tuple(
                sorted(expected_final_inventory, key=lambda item: item.object_key)
            )
            if (
                lost.matched
                and closure.objects == expected_inventory
                and not closure.multipart
            ):
                return ConditionalWriteResult(
                    expected, True, lost.contacts + closure.contacts
                )
            raise StorageMutationAmbiguous(
                "multipart_complete",
                detail=lost.detail,
                contacts=lost.contacts + closure.contacts,
                residue=closure.multipart or residue,
            ) from lost.error
        closure = self.closure(prefix)
        if closure.objects != tuple(
            sorted(expected_final_inventory, key=lambda item: item.object_key)
        ):
            raise StorageMalformedResponse("list_objects", contacts=closure.contacts)
        if closure.multipart:
            raise StorageMultipartResidue(
                "list_multipart", contacts=closure.contacts, residue=closure.multipart
            )
        return ConditionalWriteResult(expected, False)

    def _put_multipart_stream(
        self,
        object_key: str,
        chunks: Iterable[bytes | _CopySourcePart],
        *,
        approved: _ApprovedStream,
        content_type: str,
        metadata: Mapping[str, str],
    ) -> _StreamedUpload:
        """Place one object as re-sendable parts; identity and no-repeat as single put.

        ``chunks`` are client-sent ``bytes`` from a stream or provider-copied
        ``_CopySourcePart`` ranges; the part loop treats both alike.
        """

        client = self._client("multipart_create")
        upload_id = self._create_multipart(
            client,
            object_key,
            residue_prefix=object_key,
            content_type=content_type,
            metadata=metadata,
        )
        residue = (MultipartResidue(object_key, upload_id),)
        digest = _StreamDigest(approved.algorithm)
        parts = self._upload_parts(
            client,
            object_key,
            upload_id,
            chunks,
            digest=digest,
            approved=approved,
            residue=residue,
        )
        lost = self._complete_multipart(
            client, object_key, upload_id, parts, approved=approved, residue=residue
        )
        if lost is None:
            return _StreamedUpload(digest)
        if lost.matched:
            return _StreamedUpload(digest, True, lost.contacts)
        raise StorageMutationAmbiguous(
            "multipart_complete",
            detail=lost.detail,
            contacts=lost.contacts,
            residue=residue,
            bytes_read=approved.byte_count,
            safe_error_category="transport_interrupted",
        ) from lost.error

    def _stream_parts(self, source: Any) -> Iterator[bytes]:
        """Yield full-size parts from a stream that may return short reads."""

        while True:
            pieces: list[bytes] = []
            remaining = self._multipart_part_bytes
            while remaining > 0:
                piece = source.read(remaining)
                if not isinstance(piece, bytes):
                    raise OSError("source returned non-bytes")
                if not piece:
                    break
                pieces.append(piece)
                remaining -= len(piece)
            part = b"".join(pieces)
            if not part:
                return
            yield part
            if remaining > 0:
                return

    def _create_multipart(
        self,
        client: Any,
        object_key: str,
        *,
        residue_prefix: str,
        content_type: str,
        metadata: Mapping[str, str],
    ) -> str:
        try:
            created = client.create_multipart_upload(
                Bucket=self._bucket,
                Key=object_key,
                ContentType=content_type,
                Metadata=dict(metadata),
            )
        except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
            detail = _provider_detail(error)
            if _is_transient(error, detail):
                try:
                    observed = self.list_multipart(residue_prefix)
                except NebiusObjectStorageError as read_error:
                    raise StorageMutationAmbiguous(
                        "multipart_create",
                        detail=detail,
                        contacts=read_error.contacts,
                    ) from error
                residue = tuple(
                    item for item in observed.value if item.object_key == object_key
                )
                raise StorageMutationAmbiguous(
                    "multipart_create",
                    detail=detail,
                    contacts=observed.contacts,
                    residue=residue,
                ) from error
            self._raise_mutation_error("multipart_create", error)
        if (
            not isinstance(created, dict)
            or not isinstance(created.get("UploadId"), str)
            or not created["UploadId"]
        ):
            raise StorageMalformedResponse("multipart_create")
        return created["UploadId"]

    def _upload_parts(
        self,
        client: Any,
        object_key: str,
        upload_id: str,
        chunks: Iterable[bytes | _CopySourcePart],
        *,
        digest: _StreamDigest,
        approved: _ApprovedStream,
        residue: tuple[MultipartResidue, ...],
    ) -> list[dict[str, Any]]:
        """Upload each part with bounded transient retry; refuse an unexpected identity.

        ``digest`` accumulates every client-sent part, so the caller can read the
        derived SHA-256 afterwards when the approved algorithm is another one. A
        copy-source part contributes no bytes to it: its ``bytes_read`` evidence
        is the extent the provider has confirmed copied through completed parts,
        and the identity check is that those extents cover the approved size.
        """

        parts: list[dict[str, Any]] = []
        part_number = 0
        copied = 0
        pending = iter(chunks)
        while True:
            # A source read failure is classified like the single-put path (the
            # SDK surfaces it as a transport interruption there), with the open
            # upload attached so stage diagnostics can see the residue. A source
            # that is itself an SDK response body fails with the SDK's own
            # stream errors rather than OSError.
            try:
                chunk = next(pending, None)
            except _SOURCE_STREAM_ERRORS as error:
                raise StorageMutationAmbiguous(
                    "multipart_part",
                    residue=residue,
                    bytes_read=digest.byte_count,
                    safe_error_category="transport_interrupted",
                ) from error
            if chunk is None:
                break
            part_number += 1
            if isinstance(chunk, _CopySourcePart):
                bytes_read = copied
            elif isinstance(chunk, bytes) and chunk:
                digest.update(chunk)
                bytes_read = digest.byte_count
            else:
                raise StorageMultipartResidue(
                    "multipart_part", residue=residue, bytes_read=digest.byte_count
                )
            uploaded = self._upload_part_with_retry(
                client,
                object_key,
                upload_id,
                part_number,
                chunk,
                residue=residue,
                bytes_read=bytes_read,
            )
            if (
                not isinstance(uploaded, dict)
                or not isinstance(uploaded.get("ETag"), str)
                or not uploaded["ETag"]
            ):
                raise StorageMultipartResidue(
                    "multipart_part", residue=residue, bytes_read=bytes_read
                )
            if isinstance(chunk, _CopySourcePart):
                copied += chunk.byte_count
            parts.append({"PartNumber": part_number, "ETag": uploaded["ETag"]})
        complete = (
            digest.byte_count == 0 and copied == approved.byte_count
            if copied
            else digest.matches(approved)
        )
        if not parts or not complete:
            raise StorageMultipartResidue(
                "multipart_part",
                residue=residue,
                bytes_read=digest.byte_count + copied,
                safe_error_category="checksum_mismatch",
            )
        return parts

    def _upload_part_with_retry(
        self,
        client: Any,
        object_key: str,
        upload_id: str,
        part_number: int,
        chunk: bytes | _CopySourcePart,
        *,
        residue: tuple[MultipartResidue, ...],
        bytes_read: int,
    ) -> Any:
        def attempt_once(_attempt: int, deadline: _PartDeadline) -> Any:
            try:
                if isinstance(chunk, _CopySourcePart):
                    copied = client.upload_part_copy(
                        Bucket=self._bucket,
                        Key=object_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        CopySource=chunk.copy_source,
                        CopySourceRange=chunk.copy_source_range,
                    )
                    # The part's ETag arrives inside CopyPartResult; the caller
                    # checks it exactly as an UploadPart response.
                    uploaded = (
                        copied.get("CopyPartResult")
                        if isinstance(copied, dict)
                        else None
                    )
                else:
                    uploaded = client.upload_part(
                        Bucket=self._bucket,
                        Key=object_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=_DeadlineBody(chunk, deadline),
                    )
                # A response that arrives after the deadline is not a
                # success: the part ends transient and is sent again.
                deadline.remaining()
                return uploaded
            except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
                detail = _provider_detail(error)
                if not _is_transient(error, detail):
                    try:
                        self._raise_mutation_error(
                            "multipart_part", error, residue=residue
                        )
                    except StorageMutationRejected as rejected:
                        raise StorageMultipartResidue(
                            "multipart_part",
                            detail=rejected.detail,
                            residue=residue,
                            bytes_read=bytes_read,
                            safe_error_category="provider_rejected",
                        ) from error
                # Re-uploading the same PartNumber replaces it: idempotent.
                ambiguous = StorageMutationAmbiguous(
                    "multipart_part",
                    detail=detail,
                    residue=residue,
                    bytes_read=bytes_read,
                    safe_error_category="transport_interrupted",
                )
                ambiguous.__cause__ = error
                raise _PartTransient(ambiguous) from error

        return _part_attempts(
            attempt_once,
            part_seconds=bound_part_limits().effective.part_seconds,
            clock=self._clock,
        )

    def _complete_multipart(
        self,
        client: Any,
        object_key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
        *,
        approved: _ApprovedStream,
        residue: tuple[MultipartResidue, ...],
    ) -> _LostCompletion | None:
        """Complete once; after a lost response, reconcile a controlled identity by one head.

        Without a controlled SHA-256 (non-SHA-256 authority) the object carries no
        checksum metadata a head could confirm, so the loss stays unreconciled.
        """

        try:
            client.complete_multipart_upload(
                Bucket=self._bucket,
                Key=object_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
        except (ClientError, BotoCoreError, OSError, TimeoutError) as error:
            detail = _provider_detail(error)
            if not _is_transient(error, detail):
                try:
                    self._raise_mutation_error(
                        "multipart_complete", error, residue=residue
                    )
                except StorageMutationRejected as rejected:
                    raise StorageMultipartResidue(
                        "multipart_complete",
                        detail=rejected.detail,
                        residue=residue,
                        bytes_read=approved.byte_count,
                        safe_error_category="provider_rejected",
                    ) from error
            identity = approved.identity
            if identity is None:
                return _LostCompletion(error, detail, False, ())
            matched, contacts = self._reconcile_identity(object_key, identity)
            return _LostCompletion(error, detail, matched, contacts)
        return None


__all__ = [
    "MULTIPART_PART_BYTES",
    "ConditionalWriteResult",
    "LifecycleWriteResult",
    "ListedObject",
    "MultipartResidue",
    "NebiusObjectStorageError",
    "NebiusObjectStorageMechanics",
    "ObjectIdentity",
    "ObjectMetadata",
    "PartDeadlinePassed",
    "PartEndCause",
    "PartObserver",
    "PartsDownload",
    "ProducerStreamWriteResult",
    "ReadContact",
    "ReadResult",
    "ReuseOrWriteResult",
    "SafeProviderDetail",
    "StorageAccessError",
    "StorageClosure",
    "StorageConditionalCollision",
    "StorageMalformedResponse",
    "StorageMultipartResidue",
    "StorageMutationAmbiguous",
    "StorageMutationRejected",
    "StorageReadAbsent",
    "StorageTransferError",
    "controlled_sha256",
    "nebius_object_storage_client",
]
