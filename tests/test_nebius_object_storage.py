from __future__ import annotations

import dataclasses
import hashlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)

from bio_neb.providers.nebius_common import AwsCredentialMaterial
from bio_neb.providers.nebius_object_storage import (
    MULTIPART_PART_BYTES,
    ConditionalWriteResult,
    ListedObject,
    MultipartResidue,
    NebiusObjectStorageMechanics,
    ObjectIdentity,
    ProducerStreamWriteResult,
    StorageAccessError,
    StorageConditionalCollision,
    StorageMalformedResponse,
    StorageMultipartResidue,
    StorageMutationAmbiguous,
    StorageMutationRejected,
    StorageReadAbsent,
    StorageTransferError,
    _CopySourcePart,
    controlled_sha256,
    nebius_object_storage_client,
)
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt
from tests.issue98_execution_fixture import load_issue98_execution_fixture


def _credentials() -> AwsCredentialMaterial:
    return AwsCredentialMaterial(
        access_key_id="contact-only-access",
        secret_access_key="contact-only-secret",
        session_token=None,
        credentials_profile="object-storage-profile",
        worker_secret_selector="secret@version",
    )


def _error(status: int, code: str, operation: str = "HeadObject") -> ClientError:
    return ClientError(
        {
            "ResponseMetadata": {"HTTPStatusCode": status},
            "Error": {"Code": code, "Message": "provider text must not escape"},
        },
        operation,
    )


def _head(body: bytes, **metadata: str) -> dict[str, object]:
    return {
        "ContentLength": len(body),
        "Metadata": {"checksum-sha256": hashlib.sha256(body).hexdigest(), **metadata},
        "ETag": "opaque-etag",
        "VersionId": "opaque-version",
        "ChecksumSHA256": "provider-header-is-not-authority",
    }


class _QueuedClient:
    def __init__(self, **responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _take(self, operation: str, request: dict[str, object]) -> object:
        self.calls.append((operation, request))
        response = self.responses[operation].pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def head_object(self, **request: object) -> object:
        return self._take("head", request)

    def get_object(self, **request: object) -> object:
        return self._take("get", request)

    def list_objects_v2(self, **request: object) -> object:
        return self._take("objects", request)

    def list_multipart_uploads(self, **request: object) -> object:
        return self._take("multipart", request)

    def put_object(self, **request: object) -> object:
        return self._take("put", request)


class _ShortRead(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 2) if size >= 0 else 2)


class _ReceiptCredentials:
    def __init__(self) -> None:
        self.receipt = ObjectStorageCredentialReuseReceipt(
            service_account_id="service-account-exact",
            access_key_resource_id="access-key-resource-exact",
            access_key_id="contact-only-access",
            exact_topology_match_count=1,
            access_key_state="ACTIVE",
            secret_state="ACTIVE",
            secret_delivery_mode="mystery_box",
            host_secret_resolved=True,
            worker_secret_version_bound=True,
        )

    def __call__(self):
        return _credentials()


class _ExecutionContacts:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.finished: list[dict[str, object]] = []

    def start(self, **values: object) -> str:
        self.started.append(dict(values))
        return f"unit-{len(self.started)}"

    def finish(self, _: str | None, **values: object) -> None:
        self.finished.append(dict(values))


def _mechanics(
    client: object,
    *,
    observed: list[object] | None = None,
    execution_contacts: _ExecutionContacts | None = None,
    credential_resolver=None,
):
    return NebiusObjectStorageMechanics(
        endpoint_url="https://storage.eu-north1.nebius.cloud",
        region="eu-north1",
        bucket="run-bucket",
        credential_resolver=credential_resolver or _credentials,
        client_factory=lambda endpoint, region, credentials: client,
        contact_observer=None if observed is None else observed.append,
        execution_contact_observer=execution_contacts,
    )


class _Parts:
    """A part observer recording each call, with each end's cause kind."""

    def __init__(self):
        self.rows = []
        self.causes = []

    def opened(self, limits):
        surface = getattr(limits, "surface", None)
        self.rows.append(
            ("opened", surface and surface.name, limits.effective.part_seconds)
        )

    def started(self, part, attempt, offset, byte_count):
        self.rows.append(("started", part, attempt, offset, byte_count))

    def ended(self, part, attempt, classification, byte_count, cause):
        self.rows.append(
            ("ended", part, attempt, classification, byte_count, cause.kind)
        )
        self.causes.append(cause)


class _Clock:
    """The mechanics' monotonic clock, advanced only by the scripted network."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _TrickleRaw:
    """A urllib3 response body that hands out ``step`` bytes per receive.

    Each receive takes ``seconds`` on the injected clock; the socket under it
    records each timeout the reader sets, through botocore's own accessor.
    """

    def __init__(self, body: bytes, *, step: int, seconds: float, clock: _Clock):
        self._body = io.BytesIO(body)
        self.step = step
        self.seconds = seconds
        self.clock = clock
        self.timeouts: list[float] = []
        self.receives = 0
        self.closed = False
        socket = SimpleNamespace(settimeout=self.timeouts.append)
        self._fp = SimpleNamespace(
            fp=SimpleNamespace(raw=SimpleNamespace(_sock=socket))
        )

    def read1(self, size: int) -> bytes:
        self.receives += 1
        self.clock.now += self.seconds
        return self._body.read(min(size, self.step))

    def read(self, size: int | None = None) -> bytes:
        raise AssertionError("a part's body is read one receive at a time")

    def close(self) -> None:
        self.closed = True


class _TrickleClient:
    """Ranged GETs whose bodies are real SDK bodies over scripted streams.

    ``paces`` gives each GET's (bytes per receive, seconds per receive), in order.
    """

    def __init__(self, body: bytes, paces: list[tuple[int, float]], clock: _Clock):
        self.body = body
        self.paces = list(paces)
        self.clock = clock
        self.streams: list[_TrickleRaw] = []
        self.get_ranges: list[str] = []

    def get_object(self, *, Range: str, **kwargs: object):
        from botocore.response import StreamingBody

        self.get_ranges.append(Range)
        start, end = (int(bound) for bound in Range.removeprefix("bytes=").split("-"))
        step, seconds = self.paces.pop(0)
        raw = _TrickleRaw(
            self.body[start : end + 1], step=step, seconds=seconds, clock=self.clock
        )
        self.streams.append(raw)
        return {
            "Body": StreamingBody(raw, end + 1 - start),
            "ContentRange": f"bytes {start}-{end}/{len(self.body)}",
        }


class ClientPolicyTests(unittest.TestCase):
    def test_client_has_explicit_nebius_policy_and_no_default_chain(self) -> None:
        with patch("bio_neb.providers.nebius_object_storage.boto3.client") as client:
            nebius_object_storage_client(
                "https://storage.eu-north1.nebius.cloud", "eu-north1", _credentials()
            )

        kwargs = client.call_args.kwargs
        self.assertEqual(client.call_args.args, ("s3",))
        self.assertEqual(
            kwargs["endpoint_url"], "https://storage.eu-north1.nebius.cloud"
        )
        self.assertEqual(kwargs["region_name"], "eu-north1")
        self.assertEqual(kwargs["aws_access_key_id"], "contact-only-access")
        self.assertEqual(kwargs["aws_secret_access_key"], "contact-only-secret")
        self.assertIsNone(kwargs["aws_session_token"])
        self.assertEqual(kwargs["config"].connect_timeout, 30)
        self.assertEqual(kwargs["config"].read_timeout, 300)
        self.assertEqual(kwargs["config"].retries["total_max_attempts"], 1)
        self.assertEqual(kwargs["config"].retries["mode"], "standard")

    def test_factory_failure_is_pre_invocation_access_error(self) -> None:
        mechanics = NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=_credentials,
            client_factory=lambda *args: (_ for _ in ()).throw(
                RuntimeError("no client")
            ),
        )
        with self.assertRaises(StorageAccessError) as raised:
            mechanics.put_conditional_bytes("manifest.json", b"{}")
        self.assertEqual(raised.exception.operation, "conditional_put")


class ControlledMetadataTests(unittest.TestCase):
    def test_accepts_casing_and_unrelated_metadata(self) -> None:
        digest = hashlib.sha256(b"exact").hexdigest()
        self.assertEqual(
            controlled_sha256(
                {"owner": "researcher", "Checksum-SHA256": digest}, expected=digest
            ),
            digest,
        )

    def test_rejects_every_non_authoritative_or_ambiguous_shape(self) -> None:
        digest = hashlib.sha256(b"exact").hexdigest()
        bad_cases = {
            "missing": {"owner": "researcher"},
            "malformed": {"checksum-sha256": "not-a-digest"},
            "uppercase_digest": {"checksum-sha256": digest.upper()},
            "wrong": {"checksum-sha256": "0" * 64},
            "duplicate_case": {
                "checksum-sha256": digest,
                "CHECKSUM-SHA256": digest,
            },
        }
        for name, metadata in bad_cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                controlled_sha256(metadata, expected=digest)

    def test_provider_headers_etag_and_version_do_not_substitute(self) -> None:
        body = b"exact"
        client = _QueuedClient(
            head=[{k: v for k, v in _head(body).items() if k != "Metadata"}]
        )
        with self.assertRaises(StorageMalformedResponse):
            _mechanics(client).head("object")

    def test_head_metadata_exposes_custom_fields_but_not_provider_headers(self) -> None:
        body = b"exact"
        response = _head(body, **{"Tracer-Kind": "small-live"})
        client = _QueuedClient(head=[response])
        result = _mechanics(client).head_metadata("object")
        self.assertEqual(result.value.custom_metadata, (("tracer-kind", "small-live"),))
        self.assertEqual(result.value.identity.sha256, hashlib.sha256(body).hexdigest())


class ReadPolicyTests(unittest.TestCase):
    def test_transient_head_retries_once_and_retains_both_contacts(self) -> None:
        body = b"exact"
        observed: list[object] = []
        client = _QueuedClient(head=[_error(503, "SlowDown"), _head(body)])
        result = _mechanics(client, observed=observed).head("object")
        self.assertEqual(result.value.byte_count, len(body))
        self.assertEqual(
            [contact.classification for contact in result.contacts],
            ["transient", "success"],
        )
        self.assertEqual(tuple(observed), result.contacts)
        self.assertEqual(len(client.calls), 2)

    def test_transient_get_and_list_are_allowlisted(self) -> None:
        get_client = _QueuedClient(
            get=[OSError("response lost"), {"Body": io.BytesIO(b"control")}]
        )
        get_result = _mechanics(get_client).get_bytes("control.json", maximum_bytes=32)
        self.assertEqual(get_result.value, b"control")
        self.assertEqual(
            [item.classification for item in get_result.contacts],
            ["transient", "success"],
        )

        list_client = _QueuedClient(
            objects=[
                _error(500, "InternalError", "ListObjectsV2"),
                {"Contents": [], "IsTruncated": False},
            ]
        )
        listed = _mechanics(list_client).list_objects("run/")
        self.assertEqual(listed.value, ())
        self.assertEqual(len(listed.contacts), 2)

    def test_get_accumulates_short_reads_in_one_physical_contact(self) -> None:
        client = _QueuedClient(get=[{"Body": _ShortRead(b"control")}])
        exact = _mechanics(client).get_bytes("control", maximum_bytes=7)
        self.assertEqual(exact.value, b"control")
        self.assertEqual(len(exact.contacts), 1)
        self.assertEqual([name for name, _ in client.calls], ["get"])

    def test_range_accumulates_short_reads_in_one_physical_contact(self) -> None:
        client = _QueuedClient(
            get=[{"Body": _ShortRead(b"range"), "ContentRange": "bytes 3-7/10"}]
        )
        ranged = _mechanics(client).read_range("object", offset=3, byte_count=5)
        self.assertEqual(ranged.value, b"range")
        self.assertEqual(len(ranged.contacts), 1)
        self.assertEqual([name for name, _ in client.calls], ["get"])

    def test_transient_exhaustion_has_exactly_two_contacts(self) -> None:
        client = _QueuedClient(
            head=[_error(503, "SlowDown"), EndpointConnectionError(endpoint_url="x")]
        )
        with self.assertRaises(StorageTransferError) as raised:
            _mechanics(client).head("object")
        self.assertEqual(
            [contact.classification for contact in raised.exception.contacts],
            ["transient", "transient"],
        )
        self.assertEqual(len(client.calls), 2)

    def test_absence_access_and_malformed_are_not_retried(self) -> None:
        cases = (
            (_error(404, "NoSuchKey"), StorageReadAbsent, "absent"),
            (_error(403, "AccessDenied"), StorageAccessError, "access"),
            (
                {"ContentLength": "five", "Metadata": {}},
                StorageMalformedResponse,
                "success",
            ),
        )
        for response, error_type, classification in cases:
            client = _QueuedClient(head=[response])
            with (
                self.subTest(error=error_type.__name__),
                self.assertRaises(error_type) as raised,
            ):
                _mechanics(client).head("object")
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(
                raised.exception.contacts[0].classification, classification
            )

    def test_nontransient_read_failure_is_not_retried(self) -> None:
        client = _QueuedClient(head=[_error(400, "InvalidRequest")])
        with self.assertRaises(StorageTransferError) as raised:
            _mechanics(client).head("object")
        self.assertEqual(raised.exception.contacts[0].classification, "transfer")
        self.assertEqual(len(client.calls), 1)

    def test_transient_lifecycle_get_is_observed_but_not_retried(self) -> None:
        client = _StorageClient()
        client.lifecycle = None
        client.lifecycle_error = None
        client.get_bucket_lifecycle_configuration = lambda **kwargs: (
            _ for _ in ()
        ).throw(_error(503, "ServiceUnavailable", "GetBucketLifecycle"))
        with self.assertRaises(StorageTransferError) as raised:
            _mechanics(client).get_lifecycle()
        self.assertEqual(len(raised.exception.contacts), 1)
        self.assertEqual(raised.exception.contacts[0].classification, "transient")


class PaginationTests(unittest.TestCase):
    def test_object_pagination_is_complete_sorted_and_opaque(self) -> None:
        execution_contacts = _ExecutionContacts()
        client = _QueuedClient(
            objects=[
                {
                    "Contents": [{"Key": "run/b", "Size": 2}],
                    "IsTruncated": True,
                    "NextContinuationToken": "opaque-object-token",
                },
                {"Contents": [{"Key": "run/a", "Size": 1}], "IsTruncated": False},
            ]
        )
        result = _mechanics(client, execution_contacts=execution_contacts).list_objects(
            "run/"
        )
        self.assertEqual(
            result.value,
            (ListedObject("run/a", 1), ListedObject("run/b", 2)),
        )
        self.assertEqual(client.calls[1][1]["ContinuationToken"], "opaque-object-token")
        self.assertEqual([item.ordinal for item in result.contacts], [1, 2])
        self.assertEqual(
            [item["operation"] for item in execution_contacts.started],
            ["s3.list-objects-v2", "s3.list-objects-v2"],
        )

    def test_multipart_pagination_is_complete_sorted_and_opaque(self) -> None:
        client = _QueuedClient(
            multipart=[
                {
                    "Uploads": [{"Key": "run/b", "UploadId": "opaque-upload-2"}],
                    "IsTruncated": True,
                    "NextKeyMarker": "opaque-key-marker",
                    "NextUploadIdMarker": "opaque-upload-marker",
                },
                {
                    "Uploads": [{"Key": "run/a", "UploadId": "opaque-upload-1"}],
                    "IsTruncated": False,
                },
            ]
        )
        result = _mechanics(client).list_multipart("run/")
        self.assertEqual(
            result.value,
            (
                MultipartResidue("run/a", "opaque-upload-1"),
                MultipartResidue("run/b", "opaque-upload-2"),
            ),
        )
        self.assertEqual(client.calls[1][1]["KeyMarker"], "opaque-key-marker")
        self.assertEqual(client.calls[1][1]["UploadIdMarker"], "opaque-upload-marker")
        self.assertEqual([item.ordinal for item in result.contacts], [1, 2])

    def test_object_pagination_rejects_missing_or_repeated_tokens(self) -> None:
        clients = {
            "missing": _QueuedClient(objects=[{"Contents": [], "IsTruncated": True}]),
            "repeated": _QueuedClient(
                objects=[
                    {
                        "Contents": [],
                        "IsTruncated": True,
                        "NextContinuationToken": "same-token",
                    },
                    {
                        "Contents": [],
                        "IsTruncated": True,
                        "NextContinuationToken": "same-token",
                    },
                ]
            ),
        }
        for name, client in clients.items():
            with self.subTest(name=name), self.assertRaises(StorageMalformedResponse):
                _mechanics(client).list_objects("run/")

    def test_multipart_pagination_rejects_incomplete_or_repeated_markers(self) -> None:
        clients = {
            "incomplete": _QueuedClient(
                multipart=[
                    {
                        "Uploads": [],
                        "IsTruncated": True,
                        "NextKeyMarker": "key-only",
                    }
                ]
            ),
            "repeated": _QueuedClient(
                multipart=[
                    {
                        "Uploads": [],
                        "IsTruncated": True,
                        "NextKeyMarker": "same-key",
                        "NextUploadIdMarker": "same-upload",
                    },
                    {
                        "Uploads": [],
                        "IsTruncated": True,
                        "NextKeyMarker": "same-key",
                        "NextUploadIdMarker": "same-upload",
                    },
                ]
            ),
        }
        for name, client in clients.items():
            with self.subTest(name=name), self.assertRaises(StorageMalformedResponse):
                _mechanics(client).list_multipart("run/")


class ConditionalMutationTests(unittest.TestCase):
    def test_conditional_put_invokes_once_and_reconciles_response_loss(self) -> None:
        body = b'{"authority":"application"}'
        execution_contacts = _ExecutionContacts()
        client = _QueuedClient(
            put=[EndpointConnectionError(endpoint_url="x")], head=[_head(body)]
        )
        result = _mechanics(
            client, execution_contacts=execution_contacts
        ).put_conditional_bytes("run/manifest.json", body)
        self.assertTrue(result.reconciled_after_response_loss)
        self.assertEqual([name for name, _ in client.calls], ["put", "head"])
        put = client.calls[0][1]
        self.assertEqual(put["IfNoneMatch"], "*")
        self.assertEqual(
            put["Metadata"], {"checksum-sha256": hashlib.sha256(body).hexdigest()}
        )
        self.assertEqual(
            [item["operation"] for item in execution_contacts.started],
            ["s3.put-object", "s3.head-object"],
        )

    def test_collision_is_not_success_or_retried(self) -> None:
        client = _QueuedClient(put=[_error(412, "PreconditionFailed", "PutObject")])
        with self.assertRaises(StorageConditionalCollision):
            _mechanics(client).put_conditional_bytes("object", b"exact")
        self.assertEqual([name for name, _ in client.calls], ["put"])

    def test_known_rejection_is_not_retried(self) -> None:
        rejected = _QueuedClient(put=[_error(400, "InvalidRequest", "PutObject")])
        with self.assertRaises(StorageMutationRejected):
            _mechanics(rejected).put_conditional_bytes("object", b"exact")
        self.assertEqual([name for name, _ in rejected.calls], ["put"])

    def test_unresolved_response_loss_is_ambiguous_after_one_reconciliation(
        self,
    ) -> None:
        ambiguous = _QueuedClient(
            put=[EndpointConnectionError(endpoint_url="x")],
            head=[_error(404, "NoSuchKey")],
        )
        with self.assertRaises(StorageMutationAmbiguous) as raised:
            _mechanics(ambiguous).put_conditional_bytes("object", b"exact")
        self.assertEqual(raised.exception.contacts[0].classification, "absent")
        self.assertEqual([name for name, _ in ambiguous.calls], ["put", "head"])

    def test_unrelated_custom_metadata_is_preserved_with_controlled_checksum(
        self,
    ) -> None:
        body = b"tracer"
        client = _QueuedClient(put=[{}])
        _mechanics(client).put_conditional_bytes(
            "object", body, custom_metadata={"tracer-purpose": "ticket-76"}
        )
        self.assertEqual(
            client.calls[0][1]["Metadata"],
            {
                "tracer-purpose": "ticket-76",
                "checksum-sha256": hashlib.sha256(body).hexdigest(),
            },
        )

    def test_custom_metadata_cannot_replace_controlled_checksum(self) -> None:
        body = b"tracer"
        with self.assertRaises(ValueError):
            _mechanics(_QueuedClient()).put_conditional_bytes(
                "object", body, custom_metadata={"CHECKSUM-SHA256": "0" * 64}
            )


class _StorageClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {"run/data": b"data"}
        self.metadata: dict[str, dict[str, str]] = {
            "run/data": {"checksum-sha256": hashlib.sha256(b"data").hexdigest()}
        }
        self.uploads: dict[str, tuple[str, dict[int, bytes]]] = {}
        self.calls: list[str] = []
        self.complete_error: BaseException | None = None
        self.create_error: BaseException | None = None
        self.part_error: BaseException | None = None
        self.part_error_queue: list[BaseException | None] = []
        self.part_numbers: list[int] = []
        self.copy_error: BaseException | None = None
        self.copy_sources: list[tuple[str, str]] = []
        self.copy_ranges: list[str] = []
        self.get_ranges: list[str] = []
        self.get_errors: list[BaseException | None] = []
        self.lifecycle: dict[str, object] | None = None
        self.lifecycle_error: BaseException | None = None

    def head_object(self, *, Key: str, **kwargs: object):
        self.calls.append("head")
        if Key not in self.objects:
            raise _error(404, "NoSuchKey")
        return {
            "ContentLength": len(self.objects[Key]),
            "Metadata": self.metadata[Key],
        }

    def get_object(self, *, Key: str, **kwargs: object):
        self.calls.append("get")
        if Key not in self.objects:
            raise _error(404, "NoSuchKey")
        body = self.objects[Key]
        byte_range = kwargs.get("Range")
        if isinstance(byte_range, str):
            self.get_ranges.append(byte_range)
            error = self.get_errors.pop(0) if self.get_errors else None
            if error is not None:
                raise error
            bounds = byte_range.removeprefix("bytes=").split("-")
            start, end = int(bounds[0]), int(bounds[1])
            return {
                "Body": io.BytesIO(body[start : end + 1]),
                "ContentRange": f"bytes {start}-{end}/{len(body)}",
            }
        return {"Body": io.BytesIO(body)}

    def list_objects_v2(self, *, Prefix: str, **kwargs: object):
        self.calls.append("list_objects")
        return {
            "Contents": [
                {"Key": key, "Size": len(body)}
                for key, body in sorted(self.objects.items())
                if key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }

    def list_multipart_uploads(self, *, Prefix: str, **kwargs: object):
        self.calls.append("list_multipart")
        return {
            "Uploads": [
                {"Key": key, "UploadId": upload_id}
                for upload_id, (key, _) in sorted(self.uploads.items())
                if key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }

    def put_object(
        self, *, Key: str, Body: bytes, Metadata: dict[str, str], **kwargs: object
    ):
        self.calls.append("put")
        if Key in self.objects:
            raise _error(412, "PreconditionFailed", "PutObject")
        self.objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.metadata[Key] = dict(Metadata)
        return {"ETag": "opaque"}

    def copy_object(
        self,
        *,
        Key: str,
        CopySource: dict[str, str],
        Metadata: dict[str, str],
        **kwargs: object,
    ):
        # Nebius accepts IfNoneMatch on CopyObject but does not enforce it (a
        # live probe): an existing key is silently replaced.
        self.calls.append("copy_object")
        self.copy_sources.append((CopySource["Bucket"], CopySource["Key"]))
        if self.copy_error is not None:
            raise self.copy_error
        self.objects[Key] = self.objects[CopySource["Key"]]
        self.metadata[Key] = dict(Metadata)
        return {"CopyObjectResult": {"ETag": "opaque-copy-etag"}}

    def upload_part_copy(
        self,
        *,
        UploadId: str,
        PartNumber: int,
        CopySource: dict[str, str],
        CopySourceRange: str,
        **kwargs: object,
    ):
        self.calls.append("multipart_part_copy")
        self.part_numbers.append(PartNumber)
        self.copy_sources.append((CopySource["Bucket"], CopySource["Key"]))
        self.copy_ranges.append(CopySourceRange)
        if self.part_error_queue:
            error = self.part_error_queue.pop(0)
            if error is not None:
                raise error
        elif self.part_error is not None:
            raise self.part_error
        first, last = CopySourceRange.removeprefix("bytes=").split("-")
        body = self.objects[CopySource["Key"]][int(first) : int(last) + 1]
        self.uploads[UploadId][1][PartNumber] = body
        return {"CopyPartResult": {"ETag": f"opaque-copy-etag-{PartNumber}"}}

    def get_bucket_lifecycle_configuration(self, **kwargs: object):
        self.calls.append("get_lifecycle")
        if self.lifecycle is None:
            raise _error(404, "NoSuchLifecycleConfiguration", "GetBucketLifecycle")
        return self.lifecycle

    def put_bucket_lifecycle_configuration(
        self, *, LifecycleConfiguration: dict[str, object], **kwargs: object
    ):
        self.calls.append("put_lifecycle")
        self.lifecycle = LifecycleConfiguration
        if self.lifecycle_error is not None:
            raise self.lifecycle_error
        return {}

    def create_multipart_upload(self, *, Key: str, **kwargs: object):
        self.calls.append("multipart_create")
        if self.create_error is not None:
            raise self.create_error
        self.uploads["opaque-upload"] = (Key, {})
        self.upload_metadata = dict(kwargs.get("Metadata", {}))
        self.upload_content_type = kwargs.get("ContentType")
        return {"UploadId": "opaque-upload"}

    def upload_part(
        self, *, UploadId: str, PartNumber: int, Body: io.BytesIO, **kwargs: object
    ):
        self.calls.append("multipart_part")
        self.part_numbers.append(PartNumber)
        if self.part_error_queue:
            error = self.part_error_queue.pop(0)
            if error is not None:
                raise error
        elif self.part_error is not None:
            raise self.part_error
        # The SDK sends a part's body by reading it.
        self.uploads[UploadId][1][PartNumber] = Body.read()
        return {"ETag": f"opaque-etag-{PartNumber}"}

    def complete_multipart_upload(self, *, Key: str, UploadId: str, **kwargs: object):
        self.calls.append("multipart_complete")
        parts = self.uploads[UploadId][1]
        body = b"".join(parts[number] for number in sorted(parts))
        self.objects[Key] = body
        self.metadata[Key] = {
            **getattr(self, "upload_metadata", {}),
            "checksum-sha256": hashlib.sha256(body).hexdigest(),
        }
        del self.uploads[UploadId]
        if self.complete_error is not None:
            raise self.complete_error
        return {"ETag": "opaque-complete-etag"}


class ManifestAndMultipartTests(unittest.TestCase):
    def test_manifest_is_conditional_read_back_and_last(self) -> None:
        client = _StorageClient()
        body = b'{"manifest":"exact"}'
        result = _mechanics(client).publish_manifest_last(
            prefix="run/",
            manifest_key="run/manifest.json",
            body=body,
            expected_pre_inventory=(ListedObject("run/data", 4),),
        )
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(
            client.calls,
            [
                "list_objects",
                "list_multipart",
                "head",
                "put",
                "get",
                "list_objects",
                "list_multipart",
            ],
        )
        self.assertEqual(client.objects["run/manifest.json"], body)

    def test_manifest_refuses_residue_without_mutation_or_cleanup(self) -> None:
        client = _StorageClient()
        client.uploads["retained-upload"] = ("run/partial", {})
        with self.assertRaises(StorageMultipartResidue) as raised:
            _mechanics(client).publish_manifest_last(
                prefix="run/",
                manifest_key="run/manifest.json",
                body=b"{}",
                expected_pre_inventory=(ListedObject("run/data", 4),),
            )
        self.assertEqual(
            raised.exception.residue,
            (MultipartResidue("run/partial", "retained-upload"),),
        )
        self.assertNotIn("put", client.calls)
        self.assertFalse(any(call in client.calls for call in ("abort", "delete")))

    def test_multipart_complete_response_loss_reconciles_without_repeating(
        self,
    ) -> None:
        execution_contacts = _ExecutionContacts()
        client = _StorageClient()
        client.complete_error = EndpointConnectionError(endpoint_url="x")
        body = b"prepared-object"
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        expected_inventory = (
            ListedObject("run/data", 4),
            ListedObject("run/prepared", len(body)),
        )
        result = _mechanics(
            client, execution_contacts=execution_contacts
        ).multipart_upload(
            object_key="run/prepared",
            chunks=(body[:5], body[5:]),
            expected=expected,
            prefix="run/",
            expected_final_inventory=expected_inventory,
        )
        self.assertTrue(result.reconciled_after_response_loss)
        self.assertEqual(client.calls.count("multipart_create"), 1)
        self.assertEqual(client.calls.count("multipart_part"), 2)
        self.assertEqual(client.calls.count("multipart_complete"), 1)
        self.assertFalse(client.uploads)
        self.assertEqual(
            [item["operation"] for item in execution_contacts.started],
            [
                "s3.create-multipart-upload",
                "s3.upload-part",
                "s3.upload-part",
                "s3.complete-multipart-upload",
                "s3.head-object",
                "s3.list-objects-v2",
                "s3.list-multipart-uploads",
            ],
        )

    def test_server_side_copy_requests_are_named_execution_contacts(self) -> None:
        execution_contacts = _ExecutionContacts()
        client = _StorageClient()
        body = b"prepared-object"
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        client.objects["donor/reference"] = body
        client.metadata["donor/reference"] = {"checksum-sha256": expected.sha256}
        objects = NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=_credentials,
            client_factory=lambda endpoint, region, credentials: client,
            execution_contact_observer=execution_contacts,
            multipart_part_bytes=8,
        )
        objects.copy_conditional(
            source_bucket="donor-bucket",
            source_key="donor/reference",
            destination_key="run/parts",
            expected=expected,
        )
        self.assertEqual(
            [item["operation"] for item in execution_contacts.started],
            [
                "s3.create-multipart-upload",
                "s3.upload-part-copy",
                "s3.upload-part-copy",
                "s3.complete-multipart-upload",
            ],
        )
        small = NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=_credentials,
            client_factory=lambda endpoint, region, credentials: client,
            execution_contact_observer=execution_contacts,
        )
        small.copy_conditional(
            source_bucket="donor-bucket",
            source_key="donor/reference",
            destination_key="run/whole",
            expected=expected,
        )
        self.assertEqual(
            [item["operation"] for item in execution_contacts.started[4:]],
            ["s3.head-object", "s3.copy-object"],
        )
        self.assertEqual(
            {item["parse_classification"] for item in execution_contacts.finished}
            - {"response_received"},
            {"absent"},
        )

    def test_multipart_part_transient_exhaustion_retains_upload_without_cleanup(
        self,
    ) -> None:
        from bio_neb.failure import (
            FailureOverrides,
            FailureScopes,
            FailureSurface,
            FailureTuning,
            OperationEntry,
            operation_failure_policy,
            policy_for,
        )

        client = _StorageClient()
        client.part_error = EndpointConnectionError(endpoint_url="x")
        body = b"prepared-object"
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        # The upload's stage policy decides each part's retry: this Run
        # allows two attempts at `input_storage`.
        policy = policy_for(
            FailureSurface(scope="stage", name="input_storage"),
            FailureTuning(
                scopes=FailureScopes(
                    stage={
                        "input_storage": OperationEntry(
                            operation=FailureOverrides(retry_attempts=2)
                        )
                    }
                )
            ),
        )
        with (
            patch("time.sleep") as sleep,
            operation_failure_policy(policy),
            self.assertRaises(StorageMutationAmbiguous) as raised,
        ):
            _mechanics(client).multipart_upload(
                object_key="run/prepared",
                chunks=(body,),
                expected=expected,
                prefix="run/",
                expected_final_inventory=(ListedObject("run/prepared", len(body)),),
            )
        self.assertEqual(
            raised.exception.residue,
            (MultipartResidue("run/prepared", "opaque-upload"),),
        )
        self.assertEqual(raised.exception.safe_error_category, "transport_interrupted")
        self.assertEqual(client.part_numbers, [1, 1])
        self.assertEqual(sleep.call_count, 1)
        self.assertNotIn("multipart_complete", client.calls)
        self.assertIn("opaque-upload", client.uploads)

    def test_multipart_part_transient_failure_retries_the_same_part_only(
        self,
    ) -> None:
        execution_contacts = _ExecutionContacts()
        client = _StorageClient()
        client.part_error_queue = [None, EndpointConnectionError(endpoint_url="x")]
        body = b"prepared-object"
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        with patch("time.sleep") as sleep:
            result = _mechanics(
                client, execution_contacts=execution_contacts
            ).multipart_upload(
                object_key="run/prepared",
                chunks=(body[:5], body[5:10], body[10:]),
                expected=expected,
                prefix="run/",
                expected_final_inventory=(
                    ListedObject("run/data", 4),
                    ListedObject("run/prepared", len(body)),
                ),
            )
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(client.objects["run/prepared"], body)
        self.assertEqual(client.part_numbers, [1, 2, 2, 3])
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(client.calls.count("multipart_create"), 1)
        self.assertEqual(client.calls.count("multipart_complete"), 1)
        self.assertEqual(
            [item["parse_classification"] for item in execution_contacts.finished][:4],
            [
                "response_received",
                "response_received",
                "transient",
                "response_received",
            ],
        )

    def test_an_upload_part_sent_past_its_deadline_is_retried_as_transient(
        self,
    ) -> None:
        # Upload parts share the part deadline. The SDK reads a part's body
        # block by block as it sends; part 2's first send takes 200 s per 2-byte
        # block, so its third block is refused at the deadline.
        clock = _Clock()

        class SlowSend(_StorageClient):
            def __init__(self):
                super().__init__()
                self.sent: list[bytes] = []
                self.reads: list[float] = []

            def upload_part(self, *, Body, PartNumber, **request):
                if PartNumber == 2 and self.part_numbers.count(2) == 0:
                    self.calls.append("multipart_part")
                    self.part_numbers.append(PartNumber)
                    self.began = clock.now
                    while True:
                        self.reads.append(clock.now - self.began)
                        block = Body.read(2)
                        if not block:
                            break
                        self.sent.append(block)
                        clock.now += 200.0
                    raise AssertionError("a send past the deadline continued")
                return super().upload_part(Body=Body, PartNumber=PartNumber, **request)

        execution_contacts = _ExecutionContacts()
        client = SlowSend()
        body = b"prepared-object"
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        with patch("time.sleep") as sleep:
            result = NebiusObjectStorageMechanics(
                endpoint_url="https://storage.eu-north1.nebius.cloud",
                region="eu-north1",
                bucket="run-bucket",
                credential_resolver=_credentials,
                client_factory=lambda endpoint, region, credentials: client,
                execution_contact_observer=execution_contacts,
                clock=clock,
            ).multipart_upload(
                object_key="run/prepared",
                chunks=(body[:5], body[5:10], body[10:]),
                expected=expected,
                prefix="run/",
                expected_final_inventory=(
                    ListedObject("run/data", 4),
                    ListedObject("run/prepared", len(body)),
                ),
            )
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(client.objects["run/prepared"], body)
        # The attempt ends within its deadline plus the one send that crossed
        # it. Blocks went out at 0 s and 200 s; the send begun at 200 s ended at
        # 400 s, and the next read, the first after the deadline, was refused:
        # nothing was sent after it.
        self.assertEqual(client.sent, [b"re", b"d-"])
        self.assertEqual(client.reads, [0.0, 200.0, 400.0])
        self.assertLessEqual(client.reads[-1], 300.0 + 200.0)
        self.assertEqual(client.part_numbers, [1, 2, 2, 3])
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(
            [item["parse_classification"] for item in execution_contacts.finished][:4],
            [
                "response_received",
                "response_received",
                "transient",
                "response_received",
            ],
        )

    def test_an_upload_or_copy_part_answered_after_its_deadline_is_sent_again(
        self,
    ) -> None:
        # A part whose response arrives after its deadline is not a success; it
        # ends transient and the stage policy sends it again.
        clock = _Clock()
        late = {"answers": 1}

        def answered_late(respond):
            def call(**request):
                answer = respond(**request)
                if late["answers"]:
                    late["answers"] -= 1
                    clock.now += 310.0
                return answer

            return call

        mechanics = NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=_credentials,
            client_factory=lambda endpoint, region, credentials: client,
            clock=clock,
        )
        for name, chunk, method in (
            ("upload", b"part", "upload_part"),
            (
                "copy",
                _CopySourcePart("source-bucket", "source-key", 0, 3),
                "upload_part_copy",
            ),
        ):
            with self.subTest(name):
                late["answers"] = 1
                client = _StorageClient()
                client.objects["source-key"] = b"data"
                client.uploads["upload-id"] = ("run/artifact", {})
                setattr(client, method, answered_late(getattr(client, method)))
                with patch("time.sleep") as sleep:
                    uploaded = mechanics._upload_part_with_retry(
                        client,
                        "run/artifact",
                        "upload-id",
                        1,
                        chunk,
                        residue=(),
                        bytes_read=4,
                    )
                self.assertEqual(
                    uploaded["ETag"],
                    "opaque-etag-1" if name == "upload" else "opaque-copy-etag-1",
                )
                self.assertEqual(sleep.call_count, 1)
                self.assertEqual(client.part_numbers, [1, 1])

    def test_multipart_part_rejection_and_access_loss_are_not_retried(self) -> None:
        body = b"prepared-object"
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        for error, kind in (
            (_error(400, "InvalidPart", "UploadPart"), StorageMultipartResidue),
            (_error(403, "AccessDenied", "UploadPart"), StorageAccessError),
        ):
            with self.subTest(kind=kind.kind):
                client = _StorageClient()
                client.part_error = error
                with (
                    patch("time.sleep") as sleep,
                    self.assertRaises(kind) as raised,
                ):
                    _mechanics(client).multipart_upload(
                        object_key="run/prepared",
                        chunks=(body,),
                        expected=expected,
                        prefix="run/",
                        expected_final_inventory=(),
                    )
                self.assertEqual(client.part_numbers, [1])
                self.assertEqual(sleep.call_count, 0)
                self.assertEqual(
                    raised.exception.residue,
                    (MultipartResidue("run/prepared", "opaque-upload"),),
                )
                self.assertNotIn("multipart_complete", client.calls)

    def test_multipart_create_ambiguity_lists_exact_residue_without_second_create(
        self,
    ) -> None:
        client = _StorageClient()
        client.uploads["provider-retained"] = ("run/prepared", {})
        client.create_error = EndpointConnectionError(endpoint_url="x")
        body = b"prepared-object"
        with self.assertRaises(StorageMutationAmbiguous) as raised:
            _mechanics(client).multipart_upload(
                object_key="run/prepared",
                chunks=(body,),
                expected=ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
                prefix="run/",
                expected_final_inventory=(),
            )
        self.assertEqual(client.calls.count("multipart_create"), 1)
        self.assertEqual(
            raised.exception.residue,
            (MultipartResidue("run/prepared", "provider-retained"),),
        )
        self.assertNotIn("multipart_part", client.calls)


class StreamMultipartRoutingTests(unittest.TestCase):
    """Stream uploads larger than one part are multipart; callers do not choose."""

    PART = 8

    def _mechanics(self, client, **kwargs):
        return NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=_credentials,
            client_factory=lambda endpoint, region, credentials: client,
            **{"multipart_part_bytes": self.PART, **kwargs},
        )

    def test_default_part_policy_is_within_the_recommended_range(self) -> None:
        self.assertEqual(MULTIPART_PART_BYTES, 32 * 1024 * 1024)
        with self.assertRaises(ValueError):
            self._mechanics(_StorageClient(), multipart_part_bytes=0)

    def test_unknown_checksum_is_established_during_the_single_necessary_upload(self):
        for body in (b"abc", b"abc" * self.PART):
            with self.subTest(size=len(body)):
                client = _StorageClient()
                stream = io.BytesIO(body)
                result = self._mechanics(client).put_conditional_stream(
                    "run/acquired", stream, expected=None, expected_byte_count=len(body)
                )
                self.assertEqual(
                    result.identity,
                    ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
                )
                self.assertEqual(client.objects["run/acquired"], body)
                self.assertEqual(stream.tell(), len(body))
                self.assertNotIn("get", client.calls)
                self.assertNotIn("head", client.calls)

    def test_unknown_checksum_still_rejects_short_or_incomplete_transfer(self):
        for expected_size in (4, self.PART + 2):
            with self.subTest(expected_size=expected_size):
                client = _StorageClient()
                with self.assertRaises(
                    (StorageMutationAmbiguous, StorageMultipartResidue)
                ):
                    self._mechanics(client).put_conditional_stream(
                        "run/short",
                        io.BytesIO(b"abc"),
                        expected=None,
                        expected_byte_count=expected_size,
                    )
                self.assertNotIn("multipart_complete", client.calls)

    def test_stream_up_to_one_part_stays_a_single_conditional_put(self) -> None:
        client = _StorageClient()
        body = b"x" * self.PART
        result = self._mechanics(client).put_conditional_stream(
            "run/small",
            io.BytesIO(body),
            expected=ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
        )
        self.assertEqual(result.identity.byte_count, len(body))
        self.assertEqual(client.calls, ["put"])
        self.assertEqual(client.objects["run/small"], body)

    def test_stream_above_one_part_is_multipart_with_full_metadata(self) -> None:
        execution_contacts = _ExecutionContacts()
        client = _StorageClient()
        body = bytes(range(256)) * 2 + b"tail"  # 516 bytes -> 64 full parts + 4
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        result = self._mechanics(
            client, execution_contact_observer=execution_contacts
        ).put_conditional_stream(
            "run/large",
            _ShortRead(body),
            expected=expected,
            content_type="application/gzip",
            custom_metadata={"origin": "registered-input"},
        )
        self.assertEqual(result, ConditionalWriteResult(expected, False))
        self.assertNotIn("put", client.calls)
        self.assertEqual(client.calls.count("multipart_create"), 1)
        self.assertEqual(client.part_numbers, list(range(1, 66)))
        self.assertEqual(client.calls.count("multipart_complete"), 1)
        self.assertEqual(client.objects["run/large"], body)
        self.assertEqual(client.upload_content_type, "application/gzip")
        self.assertEqual(
            client.metadata["run/large"],
            {"origin": "registered-input", "checksum-sha256": expected.sha256},
        )
        self.assertFalse(client.uploads)
        self.assertEqual(
            [item["operation"] for item in execution_contacts.started][-1],
            "s3.complete-multipart-upload",
        )

    def test_stream_multipart_identity_mismatch_leaves_residue_uncompleted(
        self,
    ) -> None:
        client = _StorageClient()
        body = b"y" * (self.PART * 2 + 1)
        wrong = ObjectIdentity(len(body), hashlib.sha256(b"other").hexdigest())
        with self.assertRaises(StorageMultipartResidue) as raised:
            self._mechanics(client).put_conditional_stream(
                "run/large", io.BytesIO(body), expected=wrong
            )
        self.assertEqual(raised.exception.safe_error_category, "checksum_mismatch")
        self.assertEqual(raised.exception.bytes_read, len(body))
        self.assertEqual(
            raised.exception.residue, (MultipartResidue("run/large", "opaque-upload"),)
        )
        self.assertNotIn("multipart_complete", client.calls)
        self.assertNotIn("run/large", client.objects)

        short = _StorageClient()
        with self.assertRaises(StorageMultipartResidue) as raised:
            self._mechanics(short).put_conditional_stream(
                "run/large",
                io.BytesIO(body[:-1]),
                expected=ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
            )
        self.assertEqual(raised.exception.bytes_read, len(body) - 1)
        self.assertNotIn("multipart_complete", short.calls)

    def test_stream_multipart_part_exhaustion_is_ambiguous_with_evidence(
        self,
    ) -> None:
        client = _StorageClient()
        client.part_error_queue = [None, None]
        client.part_error = EndpointConnectionError(endpoint_url="x")
        body = b"z" * (self.PART * 3)
        with (
            patch("time.sleep"),
            self.assertRaises(StorageMutationAmbiguous) as raised,
        ):
            self._mechanics(client).put_conditional_stream(
                "run/large",
                io.BytesIO(body),
                expected=ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
            )
        self.assertEqual(raised.exception.operation, "multipart_part")
        self.assertEqual(raised.exception.safe_error_category, "transport_interrupted")
        self.assertEqual(raised.exception.bytes_read, self.PART * 3)
        self.assertEqual(
            raised.exception.residue, (MultipartResidue("run/large", "opaque-upload"),)
        )
        self.assertEqual(client.part_numbers, [1, 2, 3, 3, 3])
        self.assertNotIn("multipart_complete", client.calls)

    def test_stream_source_read_failure_is_transport_evidence_with_residue(
        self,
    ) -> None:
        class FailingSource(io.BytesIO):
            def read(self, size: int = -1) -> bytes:
                if self.tell() >= 8:
                    raise OSError("source vanished")
                return super().read(size)

        client = _StorageClient()
        body = b"v" * (self.PART * 3)
        with self.assertRaises(StorageMutationAmbiguous) as raised:
            self._mechanics(client).put_conditional_stream(
                "run/large",
                FailingSource(body),
                expected=ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
            )
        self.assertEqual(raised.exception.operation, "multipart_part")
        self.assertEqual(raised.exception.safe_error_category, "transport_interrupted")
        self.assertEqual(raised.exception.bytes_read, self.PART)
        self.assertEqual(
            raised.exception.residue, (MultipartResidue("run/large", "opaque-upload"),)
        )
        self.assertEqual(client.part_numbers, [1])
        self.assertNotIn("multipart_complete", client.calls)

    def test_stream_sdk_source_errors_are_transport_evidence_with_residue(
        self,
    ) -> None:
        """A fetched source body fails with SDK stream errors, not only OSError."""

        body = b"w" * (self.PART * 3)
        errors = (
            IncompleteReadError(actual_bytes=self.PART * 2, expected_bytes=len(body)),
            ResponseStreamingError(error=RuntimeError("connection reset")),
            ReadTimeoutError(endpoint_url="https://storage.example.test"),
        )
        for stream_error in errors:
            with self.subTest(error=type(stream_error).__name__):

                class FailingSource(io.BytesIO):
                    failure = stream_error

                    def read(self, size: int = -1) -> bytes:
                        if self.tell() >= 16:
                            raise self.failure
                        return super().read(size)

                client = _StorageClient()
                with self.assertRaises(StorageMutationAmbiguous) as raised:
                    self._mechanics(client).put_conditional_stream(
                        "run/large",
                        FailingSource(body),
                        expected=ObjectIdentity(
                            len(body), hashlib.sha256(body).hexdigest()
                        ),
                    )
                self.assertIs(raised.exception.__cause__, stream_error)
                self.assertEqual(raised.exception.operation, "multipart_part")
                self.assertEqual(
                    raised.exception.safe_error_category, "transport_interrupted"
                )
                self.assertEqual(raised.exception.bytes_read, self.PART * 2)
                self.assertEqual(
                    raised.exception.residue,
                    (MultipartResidue("run/large", "opaque-upload"),),
                )
                self.assertEqual(client.part_numbers, [1, 2])
                self.assertNotIn("multipart_complete", client.calls)

    def test_stream_multipart_complete_rejection_is_provider_evidence(self) -> None:
        client = _StorageClient()
        client.complete_error = _error(400, "InvalidPart", "CompleteMultipartUpload")
        body = b"u" * (self.PART + 1)
        with self.assertRaises(StorageMultipartResidue) as raised:
            self._mechanics(client).put_conditional_stream(
                "run/large",
                io.BytesIO(body),
                expected=ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
            )
        self.assertEqual(raised.exception.operation, "multipart_complete")
        self.assertEqual(raised.exception.safe_error_category, "provider_rejected")
        self.assertEqual(raised.exception.bytes_read, len(body))
        self.assertEqual(raised.exception.detail.status, 400)
        self.assertEqual(
            raised.exception.residue, (MultipartResidue("run/large", "opaque-upload"),)
        )

    def test_stream_multipart_complete_collision_and_response_loss(self) -> None:
        body = b"w" * (self.PART + 1)
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())

        collided = _StorageClient()
        collided.complete_error = _error(
            412, "PreconditionFailed", "CompleteMultipartUpload"
        )
        with self.assertRaises(StorageConditionalCollision) as raised:
            self._mechanics(collided).put_conditional_stream(
                "run/large", io.BytesIO(body), expected=expected
            )
        self.assertEqual(raised.exception.operation, "multipart_complete")
        self.assertEqual(collided.calls.count("multipart_complete"), 1)

        lost = _StorageClient()
        lost.complete_error = EndpointConnectionError(endpoint_url="x")
        result = self._mechanics(lost).put_conditional_stream(
            "run/large", io.BytesIO(body), expected=expected
        )
        self.assertTrue(result.reconciled_after_response_loss)
        self.assertEqual(result.identity, expected)
        self.assertEqual(lost.calls[-2:], ["multipart_complete", "head"])

        drifted = _StorageClient()
        drifted.complete_error = EndpointConnectionError(endpoint_url="x")
        drifted.metadata_after_complete = {"checksum-sha256": "0" * 64}
        original = drifted.complete_multipart_upload

        def complete_then_drift(**kwargs):
            try:
                return original(**kwargs)
            finally:
                drifted.metadata[kwargs["Key"]] = drifted.metadata_after_complete

        drifted.complete_multipart_upload = complete_then_drift
        with self.assertRaises(StorageMutationAmbiguous) as raised:
            self._mechanics(drifted).put_conditional_stream(
                "run/large", io.BytesIO(body), expected=expected
            )
        self.assertEqual(raised.exception.operation, "multipart_complete")
        self.assertEqual(
            raised.exception.residue, (MultipartResidue("run/large", "opaque-upload"),)
        )

    def test_md5_authority_stream_above_one_part_is_multipart_without_sha256_metadata(
        self,
    ) -> None:
        # A SHA-256 authority streams exactly as put_conditional_stream does, which
        # test_stream_above_one_part_is_multipart_with_full_metadata owns.
        body = bytes(range(256)) * 2 + b"tail"  # 516 bytes -> 64 full parts + 4
        sha256 = hashlib.sha256(body).hexdigest()
        approved = hashlib.md5(body).hexdigest()
        execution_contacts = _ExecutionContacts()
        client = _StorageClient()
        result = self._mechanics(
            client, execution_contact_observer=execution_contacts
        ).put_conditional_authority_stream(
            "run/reference",
            _ShortRead(body),
            expected_byte_count=len(body),
            approved_algorithm="md5",
            approved_checksum=approved,
            content_type="application/gzip",
            custom_metadata={"origin": "ncbi"},
        )
        self.assertEqual(
            result,
            ProducerStreamWriteResult(len(body), "md5", approved, sha256, False),
        )
        self.assertNotIn("put", client.calls)
        self.assertEqual(client.calls.count("multipart_create"), 1)
        self.assertEqual(client.part_numbers, list(range(1, 66)))
        self.assertEqual(client.calls.count("multipart_complete"), 1)
        self.assertEqual(client.objects["run/reference"], body)
        self.assertEqual(client.upload_content_type, "application/gzip")
        # A non-SHA-256 authority cannot claim a controlled digest at
        # create time; the derived SHA-256 travels in the result only.
        self.assertEqual(client.upload_metadata, {"origin": "ncbi"})
        self.assertFalse(client.uploads)
        self.assertEqual(
            [item["operation"] for item in execution_contacts.started],
            ["s3.create-multipart-upload"]
            + ["s3.upload-part"] * 65
            + ["s3.complete-multipart-upload"],
        )

    def test_authority_stream_multipart_part_transient_failure_retries_same_part(
        self,
    ) -> None:
        client = _StorageClient()
        client.part_error_queue = [None, EndpointConnectionError(endpoint_url="x")]
        body = b"q" * (self.PART * 2 + 3)
        with patch("time.sleep") as sleep:
            result = self._mechanics(client).put_conditional_authority_stream(
                "run/reference",
                io.BytesIO(body),
                expected_byte_count=len(body),
                approved_algorithm="md5",
                approved_checksum=hashlib.md5(body).hexdigest(),
            )
        self.assertEqual(result.sha256, hashlib.sha256(body).hexdigest())
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(client.part_numbers, [1, 2, 2, 3])
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(client.objects["run/reference"], body)
        self.assertEqual(client.calls.count("multipart_complete"), 1)

    def test_md5_authority_stream_multipart_mismatch_leaves_residue_uncompleted(
        self,
    ) -> None:
        # SHA-256 and short-source mismatches stream exactly as put_conditional_stream
        # does; test_stream_multipart_identity_mismatch_leaves_residue_uncompleted owns them.
        body = b"o" * (self.PART * 2 + 1)
        client = _StorageClient()
        with self.assertRaises(StorageMultipartResidue) as raised:
            self._mechanics(client).put_conditional_authority_stream(
                "run/reference",
                io.BytesIO(body),
                expected_byte_count=len(body),
                approved_algorithm="md5",
                approved_checksum=hashlib.md5(b"other").hexdigest(),
            )
        self.assertEqual(raised.exception.safe_error_category, "checksum_mismatch")
        self.assertEqual(raised.exception.bytes_read, len(body))
        self.assertEqual(
            raised.exception.residue,
            (MultipartResidue("run/reference", "opaque-upload"),),
        )
        self.assertNotIn("multipart_complete", client.calls)
        self.assertNotIn("run/reference", client.objects)

    def _copy_client(self, body: bytes) -> tuple[_StorageClient, ObjectIdentity]:
        client = _StorageClient()
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        client.objects["donor/reference"] = body
        client.metadata["donor/reference"] = {"checksum-sha256": expected.sha256}
        return client, expected

    def test_cross_bucket_copy_above_one_part_is_copy_source_parts(self) -> None:
        body = b"c" * (self.PART * 2 + 1)
        client, expected = self._copy_client(body)
        result = self._mechanics(client).copy_conditional(
            source_bucket="donor-bucket",
            source_key="donor/reference",
            destination_key="run/reference",
            expected=expected,
            custom_metadata={"copy-purpose": "prepared-admission"},
        )
        self.assertEqual(result.identity, expected)
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(len(result.reconciliation_contacts), 0)
        self.assertEqual(
            client.calls,
            ["multipart_create"] + ["multipart_part_copy"] * 3 + ["multipart_complete"],
        )
        self.assertNotIn("get", client.calls)
        self.assertEqual(client.copy_sources, [("donor-bucket", "donor/reference")] * 3)
        self.assertEqual(
            client.copy_ranges,
            [
                f"bytes=0-{self.PART - 1}",
                f"bytes={self.PART}-{self.PART * 2 - 1}",
                f"bytes={self.PART * 2}-{self.PART * 2}",
            ],
        )
        self.assertEqual(client.objects["run/reference"], body)
        self.assertEqual(
            client.upload_metadata,
            {"copy-purpose": "prepared-admission", "checksum-sha256": expected.sha256},
        )
        self.assertFalse(client.uploads)

    def test_copy_source_part_failure_after_n_parts_retains_evidence(self) -> None:
        body = b"c" * (self.PART * 3 + 1)
        client, expected = self._copy_client(body)
        client.part_error_queue = [None, None] + [
            EndpointConnectionError(endpoint_url="x")
        ] * 3
        with (
            patch("time.sleep") as sleep,
            self.assertRaises(StorageMutationAmbiguous) as raised,
        ):
            self._mechanics(client).copy_conditional(
                source_bucket="donor-bucket",
                source_key="donor/reference",
                destination_key="run/reference",
                expected=expected,
            )
        self.assertEqual(raised.exception.operation, "multipart_part")
        self.assertEqual(raised.exception.safe_error_category, "transport_interrupted")
        self.assertEqual(raised.exception.bytes_read, self.PART * 2)
        self.assertIsNone(raised.exception.detail.status)
        self.assertEqual(len(raised.exception.contacts), 0)
        self.assertEqual(
            raised.exception.residue,
            (MultipartResidue("run/reference", "opaque-upload"),),
        )
        self.assertIsInstance(raised.exception.__cause__, EndpointConnectionError)
        # Outside a bound stage the failure module's defaults apply: 3 attempts.
        self.assertEqual(client.part_numbers, [1, 2] + [3] * 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("multipart_complete", client.calls)
        self.assertNotIn("get", client.calls)
        self.assertIn("opaque-upload", client.uploads)

        rejected_client, expected = self._copy_client(body)
        rejected_client.part_error_queue = [None, _error(400, "InvalidRequest")]
        with self.assertRaises(StorageMultipartResidue) as rejected:
            self._mechanics(rejected_client).copy_conditional(
                source_bucket="donor-bucket",
                source_key="donor/reference",
                destination_key="run/reference",
                expected=expected,
            )
        self.assertEqual(rejected.exception.safe_error_category, "provider_rejected")
        self.assertEqual(rejected.exception.bytes_read, self.PART)
        self.assertEqual(rejected.exception.detail.status, 400)
        self.assertEqual(
            rejected.exception.residue,
            (MultipartResidue("run/reference", "opaque-upload"),),
        )
        self.assertEqual(rejected_client.part_numbers, [1, 2])

    def test_copy_source_part_transient_then_success_replaces_the_part(self) -> None:
        body = b"c" * (self.PART * 2 + 1)
        client, expected = self._copy_client(body)
        client.part_error_queue = [None, ReadTimeoutError(endpoint_url="x"), None]
        with patch("time.sleep"):
            result = self._mechanics(client).copy_conditional(
                source_bucket="donor-bucket",
                source_key="donor/reference",
                destination_key="run/reference",
                expected=expected,
            )
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(client.part_numbers, [1, 2, 2, 3])
        self.assertEqual(client.objects["run/reference"], body)

    def test_copy_source_completion_is_conditional_and_reconciled_after_loss(
        self,
    ) -> None:
        body = b"c" * (self.PART + 1)
        client, expected = self._copy_client(body)
        client.complete_error = EndpointConnectionError(endpoint_url="x")
        result = self._mechanics(client).copy_conditional(
            source_bucket="donor-bucket",
            source_key="donor/reference",
            destination_key="run/reference",
            expected=expected,
        )
        self.assertTrue(result.reconciled_after_response_loss)
        self.assertEqual(
            client.calls,
            ["multipart_create", "multipart_part_copy", "multipart_part_copy"]
            + ["multipart_complete", "head"],
        )

        collided, expected = self._copy_client(body)
        collided.complete_error = _error(412, "KeyAlreadyExists", "Complete")
        with self.assertRaises(StorageConditionalCollision) as raised:
            self._mechanics(collided).copy_conditional(
                source_bucket="donor-bucket",
                source_key="donor/reference",
                destination_key="run/reference",
                expected=expected,
            )
        self.assertEqual(raised.exception.operation, "multipart_complete")

    def test_copy_source_part_without_etag_is_residue(self) -> None:
        class NoEtag(_StorageClient):
            def upload_part_copy(self, **request):
                super().upload_part_copy(**request)
                return {"CopyPartResult": {}}

        body = b"c" * (self.PART + 1)
        client = NoEtag()
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        client.objects["donor/reference"] = body
        client.metadata["donor/reference"] = {"checksum-sha256": expected.sha256}
        with self.assertRaises(StorageMultipartResidue) as raised:
            self._mechanics(client).copy_conditional(
                source_bucket="donor-bucket",
                source_key="donor/reference",
                destination_key="run/reference",
                expected=expected,
            )
        self.assertEqual(raised.exception.bytes_read, 0)
        self.assertEqual(
            raised.exception.residue,
            (MultipartResidue("run/reference", "opaque-upload"),),
        )

    def test_authority_stream_single_put_mismatch_is_ambiguous_checksum_evidence(
        self,
    ) -> None:
        body = b"n" * self.PART
        with self.assertRaises(StorageMutationAmbiguous) as raised:
            self._mechanics(_StorageClient()).put_conditional_authority_stream(
                "run/reference",
                io.BytesIO(body),
                expected_byte_count=len(body),
                approved_algorithm="md5",
                approved_checksum=hashlib.md5(b"other").hexdigest(),
            )
        self.assertEqual(raised.exception.operation, "conditional_put")
        self.assertEqual(raised.exception.safe_error_category, "checksum_mismatch")
        self.assertEqual(raised.exception.bytes_read, len(body))

    def test_authority_stream_multipart_completion_loss_reconciles_only_sha256(
        self,
    ) -> None:
        body = b"m" * (self.PART + 1)
        sha256 = hashlib.sha256(body).hexdigest()

        known = _StorageClient()
        known.complete_error = EndpointConnectionError(endpoint_url="x")
        result = self._mechanics(known).put_conditional_authority_stream(
            "run/reference",
            io.BytesIO(body),
            expected_byte_count=len(body),
            approved_algorithm="sha256",
            approved_checksum=sha256,
        )
        self.assertEqual(
            (
                result.byte_count,
                result.approved_algorithm,
                result.approved_checksum,
                result.sha256,
                result.reconciled_after_response_loss,
            ),
            (len(body), "sha256", sha256, sha256, True),
        )
        self.assertEqual(len(result.reconciliation_contacts), 1)
        self.assertEqual(known.calls[-2:], ["multipart_complete", "head"])

        unknown = _StorageClient()
        unknown.complete_error = EndpointConnectionError(endpoint_url="x")
        with self.assertRaises(StorageMutationAmbiguous) as raised:
            self._mechanics(unknown).put_conditional_authority_stream(
                "run/reference",
                io.BytesIO(body),
                expected_byte_count=len(body),
                approved_algorithm="md5",
                approved_checksum=hashlib.md5(body).hexdigest(),
            )
        self.assertEqual(raised.exception.operation, "multipart_complete")
        self.assertEqual(raised.exception.safe_error_category, "transport_interrupted")
        self.assertEqual(raised.exception.contacts, ())
        self.assertEqual(raised.exception.bytes_read, len(body))
        self.assertEqual(
            raised.exception.residue,
            (MultipartResidue("run/reference", "opaque-upload"),),
        )
        self.assertNotIn("head", unknown.calls)


class ExpandedHostMechanicsTests(unittest.TestCase):
    def test_authority_stream_derives_sha256_from_one_md5_bound_upload(self) -> None:
        client = _StorageClient()
        body = b"remote-reference"
        result = _mechanics(client).put_conditional_authority_stream(
            "run/reference",
            io.BytesIO(body),
            expected_byte_count=len(body),
            approved_algorithm="md5",
            approved_checksum=hashlib.md5(body).hexdigest(),
        )
        self.assertEqual(result.sha256, hashlib.sha256(body).hexdigest())
        self.assertEqual(client.metadata["run/reference"], {})
        self.assertEqual(client.calls.count("put"), 1)

    def test_non_sha_authority_stream_response_loss_is_not_reconciled_or_retried(
        self,
    ) -> None:
        body = b"remote-reference"
        client = _QueuedClient(put=[EndpointConnectionError(endpoint_url="x")])
        with self.assertRaises(StorageMutationAmbiguous) as raised:
            _mechanics(client).put_conditional_authority_stream(
                "run/reference",
                io.BytesIO(body),
                expected_byte_count=len(body),
                approved_algorithm="md5",
                approved_checksum=hashlib.md5(body).hexdigest(),
            )
        self.assertEqual(raised.exception.contacts, ())
        self.assertEqual([name for name, _ in client.calls], ["put"])

    def test_stream_put_hashes_the_bytes_consumed_by_the_client(self) -> None:
        client = _StorageClient()
        body = b"streamed-input"
        result = _mechanics(client).put_conditional_stream(
            "run/streamed",
            io.BytesIO(body),
            expected=ObjectIdentity(len(body), hashlib.sha256(body).hexdigest()),
            custom_metadata={"origin": "registered-input"},
        )
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(client.objects["run/streamed"], body)
        self.assertEqual(client.metadata["run/streamed"]["origin"], "registered-input")
        self.assertEqual(client.calls.count("put"), 1)

        class InterruptedStream(_StorageClient):
            def put_object(self, **request):
                self.calls.append("put")
                request["Body"].read(1)
                self.objects[request["Key"]] = body
                self.metadata[request["Key"]] = dict(request["Metadata"])
                raise TimeoutError("response lost after partial consumption")

        # A cross-bucket copy streams nothing through the client, so only the two
        # client-streamed entry points share this interruption path.
        for operation in ("upload", "authority"):
            with self.subTest(operation=operation):
                interrupted = InterruptedStream()
                objects = _mechanics(interrupted)
                expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
                with self.assertRaises(StorageMutationAmbiguous):
                    if operation == "upload":
                        objects.put_conditional_stream(
                            "destination", io.BytesIO(body), expected=expected
                        )
                    else:
                        objects.put_conditional_authority_stream(
                            "destination",
                            io.BytesIO(body),
                            expected_byte_count=len(body),
                            approved_algorithm="sha256",
                            approved_checksum=expected.sha256,
                        )
                self.assertEqual(interrupted.calls.count("put"), 1)

        # Meter digest input independently from the producer's retained counters.
        hashed_bytes = []
        sha256_factory, digest_factory = hashlib.sha256, hashlib.new

        class MeteredDigest:
            def __init__(self, digest):
                self.digest = digest

            def update(self, chunk):
                hashed_bytes.append(len(chunk))
                self.digest.update(chunk)

            def hexdigest(self):
                return self.digest.hexdigest()

        client = _StorageClient()
        expected = ObjectIdentity(len(body), hashlib.sha256(body).hexdigest())
        with (
            patch(
                "hashlib.sha256",
                side_effect=lambda *args: MeteredDigest(sha256_factory(*args)),
            ),
            patch(
                "hashlib.new",
                side_effect=lambda *args: MeteredDigest(digest_factory(*args)),
            ),
        ):
            _mechanics(client).put_conditional_stream(
                "metered", io.BytesIO(body), expected=expected
            )
        self.assertEqual(sum(hashed_bytes), len(body))

    def test_full_hash_reads_all_bytes_and_surfaces_contact(self) -> None:
        client = _StorageClient()
        hashed = _mechanics(client).full_hash("run/data")
        self.assertEqual(
            hashed.value, ObjectIdentity(4, hashlib.sha256(b"data").hexdigest())
        )
        self.assertEqual(len(hashed.contacts), 1)
        self.assertEqual(client.calls, ["get"])

    def test_download_reads_ranged_parts_in_order_and_proves_the_known_digest(
        self,
    ) -> None:
        client = _StorageClient()
        body = b"generated-artifact-of-twenty-six"
        client.objects["run/aligned.bam"] = body
        resolver = _ReceiptCredentials()
        destination = io.BytesIO()

        result = NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=resolver,
            client_factory=lambda endpoint, region, credentials: client,
            multipart_part_bytes=12,
        ).download(
            "run/aligned.bam",
            destination,
            expected_byte_count=len(body),
            expected_sha256=hashlib.sha256(body).hexdigest(),
        )

        # The upload's part size: 12 + 12 + 8 bytes, each one ranged GET.
        self.assertEqual(
            client.get_ranges, ["bytes=0-11", "bytes=12-23", "bytes=24-31"]
        )
        self.assertEqual(destination.getvalue(), body)
        self.assertEqual(
            (result.byte_count, result.sha256, result.part_bytes, result.part_count),
            (32, hashlib.sha256(body).hexdigest(), 12, 3),
        )
        self.assertEqual(result.credential_receipt, resolver.receipt)

    def _download(self, client, body, *, observer=None):
        return NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=_ReceiptCredentials(),
            client_factory=lambda endpoint, region, credentials: client,
            multipart_part_bytes=12,
        ).download(
            "run/aligned.bam",
            destination := io.BytesIO(),
            expected_byte_count=len(body),
            expected_sha256=hashlib.sha256(body).hexdigest(),
            observe_part=observer,
        ), destination

    def test_download_retries_a_transient_part_alone_by_the_bound_stage_policy(
        self,
    ) -> None:
        from bio_neb.failure import (
            FailureOverrides,
            FailureScopes,
            FailureSurface,
            FailureTuning,
            OperationEntry,
            operation_failure_policy,
            policy_for,
        )

        body = b"generated-artifact-of-twenty-six"
        # The Run's section allows two attempts at this stage, not the default 3.
        policy = policy_for(
            FailureSurface(scope="stage", name="bam_materialization"),
            FailureTuning(
                scopes=FailureScopes(
                    stage={
                        "bam_materialization": OperationEntry(
                            operation=FailureOverrides(retry_attempts=2)
                        )
                    }
                )
            ),
        )
        client = _StorageClient()
        client.objects["run/aligned.bam"] = body
        client.get_errors = [None, EndpointConnectionError(endpoint_url="x")]
        parts = _Parts()
        with patch("time.sleep") as sleep, operation_failure_policy(policy):
            result, destination = self._download(client, body, observer=parts)
        # Only the faulted second part is fetched again; the first stays written.
        self.assertEqual(
            client.get_ranges,
            ["bytes=0-11", "bytes=12-23", "bytes=12-23", "bytes=24-31"],
        )
        self.assertEqual(destination.getvalue(), body)
        self.assertEqual(result.sha256, hashlib.sha256(body).hexdigest())
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(
            parts.rows,
            [
                ("opened", "bam_materialization", 300),
                ("started", 1, 1, 0, 12),
                ("ended", 1, 1, "success", 12, "success"),
                ("started", 2, 1, 12, 12),
                ("ended", 2, 1, "transient", None, "transport"),
                ("started", 2, 2, 12, 12),
                ("ended", 2, 2, "success", 12, "success"),
                ("started", 3, 1, 24, 8),
                ("ended", 3, 1, "success", 8, "success"),
            ],
        )

        # The bound policy's second transient fault exhausts the part: its own
        # fault is raised, nothing after it is fetched, no partial part written.
        client = _StorageClient()
        client.objects["run/aligned.bam"] = body
        client.get_errors = [
            None,
            EndpointConnectionError(endpoint_url="x"),
            EndpointConnectionError(endpoint_url="x"),
        ]
        with (
            patch("time.sleep") as sleep,
            operation_failure_policy(policy),
            self.assertRaises(StorageTransferError) as raised,
        ):
            self._download(client, body)
        self.assertEqual(
            client.get_ranges, ["bytes=0-11", "bytes=12-23", "bytes=12-23"]
        )
        self.assertEqual(raised.exception.contacts[-1].classification, "transient")
        self.assertEqual(sleep.call_count, 1)

    def _paced_download(self, client, body, clock, *, observer=None):
        return NebiusObjectStorageMechanics(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            region="eu-north1",
            bucket="run-bucket",
            credential_resolver=_ReceiptCredentials(),
            client_factory=lambda endpoint, region, credentials: client,
            multipart_part_bytes=12,
            clock=clock,
        ).download(
            "run/aligned.bam",
            destination := io.BytesIO(),
            expected_byte_count=len(body),
            expected_sha256=hashlib.sha256(body).hexdigest(),
            observe_part=observer,
        ), destination

    def test_a_trickling_part_ends_at_its_deadline_and_is_retried(self) -> None:
        # 300 s per part attempt in total. A recorded part kept receiving a
        # few bytes for 20 minutes; here 3 bytes arrive per 100 s.
        from bio_neb.failure import FailureSurface, operation_failure_policy, policy_for

        body = b"generated-artifact-of-twenty-six"
        clock = _Clock()
        client = _TrickleClient(body, [(3, 100.0)] + [(12, 1.0)] * 3, clock)
        parts = _Parts()
        policy = policy_for(FailureSurface(scope="stage", name="bam_materialization"))
        with patch("time.sleep") as sleep, operation_failure_policy(policy):
            result, destination = self._paced_download(
                client, body, clock, observer=parts
            )

        self.assertEqual(destination.getvalue(), body)
        self.assertEqual(result.sha256, hashlib.sha256(body).hexdigest())
        # Part 1's first attempt ends at its deadline: after 300 s and 9 of its
        # 12 bytes, before a fourth receive; the stage policy retries it.
        trickle = client.streams[0]
        self.assertEqual((trickle.receives, clock.now > 1000.0), (3, True))
        # Each receive waited no longer than the time the part had left.
        self.assertEqual(trickle.timeouts, [300.0, 200.0, 100.0])
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(
            parts.rows,
            [
                ("opened", "bam_materialization", 300),
                ("started", 1, 1, 0, 12),
                ("ended", 1, 1, "transient", None, "deadline"),
                ("started", 1, 2, 0, 12),
                ("ended", 1, 2, "success", 12, "success"),
                ("started", 2, 1, 12, 12),
                ("ended", 2, 1, "success", 12, "success"),
                ("started", 3, 1, 24, 8),
                ("ended", 3, 1, "success", 8, "success"),
            ],
        )
        self.assertEqual(
            parts.causes[0].model_dump(),
            {"kind": "deadline", "status": None, "code": None, "error_type": None},
        )

        # A part that trickles on every attempt exhausts the policy: its last
        # fault is raised as transient and nothing after it is fetched.
        clock = _Clock()
        client = _TrickleClient(body, [(3, 100.0)] * 3, clock)
        parts = _Parts()
        with (
            patch("time.sleep"),
            operation_failure_policy(policy),
            self.assertRaises(StorageTransferError) as raised,
        ):
            self._paced_download(client, body, clock, observer=parts)
        self.assertEqual(client.get_ranges, ["bytes=0-11"] * 3)
        self.assertEqual(raised.exception.contacts[-1].classification, "transient")
        self.assertEqual(
            [row[-1] for row in parts.rows if row[0] == "ended"], ["deadline"] * 3
        )

    def test_a_part_read_that_times_out_past_its_deadline_ends_by_the_deadline(
        self,
    ) -> None:
        # A stalled receive waits only the time left; its timeout is the deadline.
        from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError

        body = b"generated-artifact-of-twenty-six"
        clock = _Clock()
        client = _TrickleClient(body, [(12, 1.0)] * 4, clock)
        stalled = client.get_object

        def stall_first(**request):
            response = stalled(**request)
            if len(client.streams) == 1:
                raw = client.streams[0]

                def timed_out(size):
                    clock.now += raw.timeouts[-1]
                    raise URLLib3ReadTimeoutError(None, None, "Read timed out.")

                raw.read1 = timed_out
            return response

        client.get_object = stall_first
        parts = _Parts()
        with patch("time.sleep"):
            result, _ = self._paced_download(client, body, clock, observer=parts)
        self.assertEqual(result.part_count, 3)
        self.assertEqual(client.streams[0].timeouts, [300.0])
        self.assertEqual(
            parts.causes[0].model_dump(),
            {
                "kind": "deadline",
                "status": None,
                "code": None,
                "error_type": "ReadTimeoutError",
            },
        )

    def test_a_response_whose_headers_come_after_the_deadline_is_not_read(
        self,
    ) -> None:
        # A phase the body reads cannot cut short (connect and header wait, here
        # one 30 s connect and one 290 s read) ends the attempt at its first
        # check after the deadline; its late body is never read.
        body = b"generated-artifact-of-twenty-six"
        clock = _Clock()
        client = _TrickleClient(body, [(12, 1.0)] * 4, clock)
        respond = client.get_object
        started_at: list[float] = []

        def late_headers(**request):
            started_at.append(clock.now)
            if len(started_at) == 1:
                clock.now += 320.0
            return respond(**request)

        client.get_object = late_headers
        parts = _Parts()
        with patch("time.sleep"):
            result, destination = self._paced_download(
                client, body, clock, observer=parts
            )
        self.assertEqual(destination.getvalue(), body)
        self.assertEqual(result.part_count, 3)
        # The late response was refused before any receive, then part 1 retried.
        self.assertEqual(client.streams[0].receives, 0)
        self.assertEqual(
            parts.rows[1:5],
            [
                ("started", 1, 1, 0, 12),
                ("ended", 1, 1, "transient", None, "deadline"),
                ("started", 1, 2, 0, 12),
                ("ended", 1, 2, "success", 12, "success"),
            ],
        )
        # It ended with the one socket operation that crossed the deadline.
        self.assertEqual(started_at[1] - started_at[0], 320.0)

    def test_a_part_read_to_its_length_on_the_real_sdk_stack_finishes(self) -> None:
        # The SDK's own response stack, offline: http.client parses the wire,
        # urllib3 streams it and botocore wraps it. Reading a body to its length
        # releases its socket, so the end-of-body check must not touch it.
        import http.client

        from botocore.response import StreamingBody
        from urllib3.response import HTTPResponse

        socket = SimpleNamespace(timeouts=[])
        socket.settimeout = socket.timeouts.append

        def makefile(mode):
            wire = io.BytesIO(
                b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\n\r\npart"
            )
            wire.raw = SimpleNamespace(_sock=socket)
            return wire

        socket.makefile = makefile
        parsed = http.client.HTTPResponse(socket)
        parsed.begin()
        body = StreamingBody(
            HTTPResponse(
                body=parsed,
                headers=dict(parsed.getheaders()),
                original_response=parsed,
                status=206,
                preload_content=False,
                decode_content=False,
            ),
            4,
        )
        client = SimpleNamespace(
            get_object=lambda **request: {"Body": body, "ContentRange": "bytes 0-3/4"}
        )
        parts = _Parts()
        result, destination = self._paced_download(
            client, b"part", _Clock(), observer=parts
        )
        self.assertEqual((result.byte_count, destination.getvalue()), (4, b"part"))
        self.assertEqual(parts.rows[-1], ("ended", 1, 1, "success", 4, "success"))
        # The one receive waited no longer than the part's whole deadline.
        self.assertEqual(socket.timeouts, [300.0])

    def test_every_part_end_names_its_cause(self) -> None:
        body = b"generated-artifact-of-twenty-six"
        cases = (
            (
                EndpointConnectionError(endpoint_url="x"),
                "transient",
                {
                    "kind": "transport",
                    "status": None,
                    "code": None,
                    "error_type": "EndpointConnectionError",
                },
            ),
            (
                _error(503, "SlowDown", "GetObject"),
                "transient",
                {
                    "kind": "provider",
                    "status": 503,
                    "code": "SlowDown",
                    "error_type": "ClientError",
                },
            ),
            (
                _error(403, "AccessDenied", "GetObject"),
                "access",
                {
                    "kind": "provider",
                    "status": 403,
                    "code": "AccessDenied",
                    "error_type": "ClientError",
                },
            ),
        )
        for fault, classification, cause in cases:
            with self.subTest(
                fault=type(fault).__name__, classification=classification
            ):
                client = _StorageClient()
                client.objects["run/aligned.bam"] = body
                client.get_errors = [fault]
                parts = _Parts()
                with patch("time.sleep"):
                    try:
                        self._download(client, body, observer=parts)
                    except StorageAccessError:
                        pass
                self.assertEqual(parts.rows[2][3], classification)
                self.assertEqual(parts.causes[0].model_dump(), cause)
                # A success ends with its own cause, never a missing one.
                self.assertTrue(
                    all(
                        item.kind == "success"
                        for row, item in zip(
                            [r for r in parts.rows if r[0] == "ended"],
                            parts.causes,
                            strict=False,
                        )
                        if row[3] == "success"
                    )
                )
        # A body that does not match its range ends malformed.
        client = _StorageClient()
        client.objects["run/aligned.bam"] = body + b"!"
        parts = _Parts()
        with self.assertRaises(StorageMalformedResponse):
            self._download(client, body, observer=parts)
        self.assertEqual(
            parts.causes[0].model_dump(),
            {
                "kind": "malformed",
                "status": None,
                "code": None,
                "error_type": "ValueError",
            },
        )

    def test_download_refuses_a_non_transient_part_and_a_digest_mismatch(
        self,
    ) -> None:
        body = b"generated-artifact-of-twenty-six"
        client = _StorageClient()
        client.objects["run/aligned.bam"] = body
        client.get_errors = [None, _error(403, "AccessDenied")]
        with patch("time.sleep") as sleep, self.assertRaises(StorageAccessError):
            self._download(client, body)
        # An access refusal is never retried.
        self.assertEqual(client.get_ranges, ["bytes=0-11", "bytes=12-23"])
        sleep.assert_not_called()

        client = _StorageClient()
        client.objects["run/aligned.bam"] = body.replace(b"twenty", b"thirty")
        with self.assertRaises(StorageMalformedResponse) as raised:
            self._download(client, body)
        # Every part arrived; the streamed digest is not the known one.
        self.assertEqual(len(client.get_ranges), 3)
        self.assertEqual(raised.exception.safe_error_category, "checksum_mismatch")

        # An object of another size fails its first part's range binding.
        client = _StorageClient()
        client.objects["run/aligned.bam"] = body + b"!"
        with self.assertRaises(StorageMalformedResponse):
            self._download(client, body)
        self.assertEqual(client.get_ranges, ["bytes=0-11"])

    def test_bounded_get_returns_exact_bytes_and_surfaces_contact(self) -> None:
        client = _StorageClient()
        exact = _mechanics(client).get_bytes("run/data", maximum_bytes=4)
        self.assertEqual(exact.value, b"data")
        self.assertEqual(len(exact.contacts), 1)
        self.assertEqual(client.calls, ["get"])

    def test_range_read_returns_exact_bytes_and_surfaces_contact(self) -> None:
        client = _StorageClient()
        ranged = _mechanics(client).read_range("run/data", offset=1, byte_count=2)
        self.assertEqual(ranged.value, b"at")
        self.assertEqual(len(ranged.contacts), 1)
        self.assertEqual(client.calls, ["get"])

    def test_exact_reuse_is_distinct_from_collision_and_write(self) -> None:
        client = _StorageClient()
        mechanics = _mechanics(client)
        reused = mechanics.reuse_exact_or_put_bytes("run/data", b"data")
        self.assertEqual(reused.disposition, "reused_exact")
        self.assertNotIn("put", client.calls)
        with self.assertRaises(StorageConditionalCollision):
            mechanics.reuse_exact_or_put_bytes("run/data", b"different")
        written = mechanics.reuse_exact_or_put_bytes("run/new", b"new")
        self.assertEqual(written.disposition, "written")
        self.assertEqual(client.calls.count("put"), 1)

    def test_lifecycle_absence_is_a_single_read(self) -> None:
        client = _StorageClient()
        absent = _mechanics(client).get_lifecycle()
        self.assertIsNone(absent.value)
        self.assertEqual(absent.contacts[0].classification, "absent")
        self.assertEqual(client.calls, ["get_lifecycle"])

    def test_ambiguous_lifecycle_write_reconciles_with_one_get_and_no_repeat(
        self,
    ) -> None:
        client = _StorageClient()
        client.lifecycle_error = EndpointConnectionError(endpoint_url="x")
        configuration = {
            "Rules": [
                {
                    "ID": "readiness",
                    "Status": "Disabled",
                    "Filter": {},
                    "Expiration": {"Days": 1},
                }
            ]
        }
        result = _mechanics(client).put_lifecycle_once(configuration)
        self.assertTrue(result.reconciled_after_response_loss)
        self.assertEqual(client.calls.count("put_lifecycle"), 1)
        self.assertEqual(client.calls.count("get_lifecycle"), 1)

    def test_cross_bucket_copy_up_to_one_part_is_one_server_side_copy(self) -> None:
        client = _StorageClient()
        expected = ObjectIdentity(4, hashlib.sha256(b"data").hexdigest())
        observed: list[object] = []
        result = _mechanics(client, observed=observed).copy_conditional(
            source_bucket="donor-bucket",
            source_key="run/data",
            destination_key="run/copied",
            expected=expected,
            custom_metadata={"copy-purpose": "prepared-admission"},
        )
        self.assertFalse(result.reconciled_after_response_loss)
        self.assertEqual(client.objects["run/copied"], b"data")
        self.assertEqual(
            client.metadata["run/copied"],
            {"copy-purpose": "prepared-admission", "checksum-sha256": expected.sha256},
        )
        # Source identity, then destination absence (the provider does not enforce
        # If-None-Match on CopyObject), then the one server-side copy.
        self.assertEqual(client.calls, ["head", "copy_object"])
        self.assertEqual(client.copy_sources, [("donor-bucket", "run/data")])
        self.assertEqual(
            [(item.operation, item.classification) for item in observed],
            [("head", "absent")],
        )
        self.assertEqual(result.reconciliation_contacts, tuple(observed))

    def test_cross_bucket_copy_refuses_an_existing_destination_before_copying(
        self,
    ) -> None:
        client = _StorageClient()
        expected = ObjectIdentity(4, hashlib.sha256(b"data").hexdigest())
        client.objects["run/copied"] = b"stale"
        client.metadata["run/copied"] = {"checksum-sha256": "0" * 64}
        with self.assertRaises(StorageConditionalCollision) as raised:
            _mechanics(client).copy_conditional(
                source_bucket="donor-bucket",
                source_key="run/data",
                destination_key="run/copied",
                expected=expected,
            )
        self.assertEqual(raised.exception.operation, "conditional_copy")
        self.assertEqual(client.calls, ["head"])
        self.assertEqual(client.objects["run/copied"], b"stale")

    def test_cross_bucket_copy_failures_classify_like_the_single_put(self) -> None:
        expected = ObjectIdentity(4, hashlib.sha256(b"data").hexdigest())

        class LostResponse(_StorageClient):
            def copy_object(self, **request):
                super().copy_object(**request)
                raise EndpointConnectionError(endpoint_url="x")

        lost = LostResponse()
        result = _mechanics(lost).copy_conditional(
            source_bucket="donor-bucket",
            source_key="run/data",
            destination_key="run/copied",
            expected=expected,
        )
        self.assertTrue(result.reconciled_after_response_loss)
        self.assertEqual(lost.calls, ["head", "copy_object", "head"])

        for error, error_type, category in (
            (
                _error(400, "InvalidRequest", "CopyObject"),
                StorageMutationRejected,
                "provider_rejected",
            ),
            (_error(403, "AccessDenied", "CopyObject"), StorageAccessError, None),
            (
                EndpointConnectionError(endpoint_url="x"),
                StorageMutationAmbiguous,
                "transport_interrupted",
            ),
        ):
            with self.subTest(error=error):
                client = _StorageClient()
                client.copy_error = error
                with self.assertRaises(error_type) as raised:
                    _mechanics(client).copy_conditional(
                        source_bucket="donor-bucket",
                        source_key="run/data",
                        destination_key="run/copied",
                        expected=expected,
                    )
                self.assertEqual(raised.exception.operation, "conditional_copy")
                self.assertEqual(raised.exception.safe_error_category, category)
                self.assertNotIn("run/copied", client.objects)
                self.assertNotIn("get", client.calls)
                self.assertNotIn("put", client.calls)


class NebiusProviderTests(unittest.TestCase):
    """Real shared multipart core with only CLI, HTTP and S3 contacts replaced."""

    def test_local_source_without_checksum_returns_acquired_proof(self):
        import tempfile
        from pathlib import Path

        from bio_neb.providers.ports import TransferSource
        from bio_neb.verified_artifacts import ArtifactProof

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reads.fastq"
            path.write_bytes(b"abc")
            receipt = self.provider.store.put(
                "bucket", "reads", TransferSource("local", str(path), 3, None)
            )
        proof = ArtifactProof.model_validate(receipt.values["artifact_proof"])
        self.assertEqual(
            proof.identity.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertIsNone(proof.expected_identity)

    def test_unknown_checksum_never_adopts_a_colliding_destination(self):
        from bio_neb.failure import ProviderError
        from bio_neb.providers.ports import TransferSource

        self.client.objects["occupied"] = b"abc"
        self.client.metadata["occupied"] = {
            "checksum-sha256": hashlib.sha256(b"abc").hexdigest()
        }
        with self.assertRaisesRegex(ProviderError, "object_content_conflict"):
            self.provider.store.put(
                "bucket", "occupied", TransferSource("local", "/absent", 3, None)
            )
        self.assertEqual(self.client.calls, ["head"])

    def test_transient_http_errors_are_retryable_and_source_journal_is_complete(self):
        from urllib.error import HTTPError

        from bio_neb.failure import ProviderError
        from bio_neb.parabricks.execution_contact_accounting import (
            ExecutionContactJournal,
            _journal_complete,
        )
        from bio_neb.providers.ports import TransferSource

        journal = ExecutionContactJournal(
            self.root / "contacts.jsonl", "parabricks-execution:" + "a" * 64
        )
        self.provider._journal = journal
        source = TransferSource(
            "url",
            "https://source.invalid/reference",
            10,
            hashlib.sha256(b"abcdefghij").hexdigest(),
        )

        def unavailable(request):
            raise HTTPError(request.full_url, 503, "unavailable", {}, None)

        self.provider._url_opener = unavailable
        with self.assertRaises(ProviderError) as raised:
            self.provider.store.put("bucket", "remote", source)
        self.assertEqual(raised.exception.failure_class, "transient")
        self.provider._url_opener = lambda _: io.BytesIO(b"abcdefghij")
        self.provider.store.put("bucket", "remote", source)
        self.assertFalse(journal.degraded)
        binding = journal.finalize()
        self.assertTrue(_journal_complete(journal.path.read_bytes(), binding))

    def setUp(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock

        from bio_neb.providers.nebius import NebiusProvider

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        from tests.approved_execution_facts import _context

        selection = _context(
            {
                "nebius": {
                    "cli_version": "0.12.263",
                    "executable_byte_count": 83259554,
                    "executable_sha256": "a" * 64,
                }
            }
        )
        self.scope = selection.topology
        self.runner = Mock()
        self.client = _StorageClient()
        self.provider = NebiusProvider(
            selection=selection,
            endpoint_url="https://storage.invalid",
            region="region",
            journal=None,
            cli_runner=self.runner,
            credential_resolver=_credentials,
            s3_client_factory=lambda *_: self.client,
            url_opener=lambda _: io.BytesIO(b"abcdefghij"),
        )
        original = NebiusObjectStorageMechanics
        self.patch = patch(
            "bio_neb.providers.nebius.NebiusObjectStorageMechanics",
            side_effect=lambda **kw: original(**kw, multipart_part_bytes=4),
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_local_and_url_uploads_use_shared_multipart_and_attested_head(self):
        from bio_neb.providers.ports import TransferSource

        content = b"abcdefghij"
        path = self.root / "reads"
        path.write_bytes(content)
        for kind, locator in [
            ("local", str(path)),
            ("url", "https://source.invalid/reference"),
        ]:
            with self.subTest(kind=kind):
                receipt = self.provider.store.put(
                    "bucket",
                    kind,
                    TransferSource(
                        kind, locator, len(content), hashlib.sha256(content).hexdigest()
                    ),
                )
                self.assertEqual(self.client.objects[kind], content)
                self.assertEqual(receipt.values["verification_mode"], "streamed_sha256")
                self.assertEqual(
                    self.provider.store.head("bucket", kind).values["sha256"],
                    receipt.values["sha256"],
                )
        self.assertEqual(self.client.calls.count("multipart_part"), 6)
        self.assertNotIn("get", self.client.calls)

    def test_publisher_md5_source_streams_once_and_measures_its_gzip_content(self):
        """Acquisition checks the approved md5 and size, derives the .gz SHA-256 and
        measures the plain content; a changed source refuses with a typed cause."""
        import gzip

        from bio_neb.failure import ProviderError
        from bio_neb.providers.ports import TransferSource

        plain = b">chr1\nACGTACGTAC\n>chr2\nTTTTGGGG\n"
        # Two gzip members, as concatenated publisher files are; parts are 4 bytes.
        content = gzip.compress(plain[:12], mtime=0) + gzip.compress(
            plain[12:], mtime=0
        )
        self.provider._url_opener = lambda _: io.BytesIO(content)
        source = TransferSource(
            "url",
            "https://source.invalid/reference.fna.gz",
            len(content),
            None,
            md5=hashlib.md5(content).hexdigest(),
            measure_gzip_content=True,
        )
        receipt = self.provider.store.put("bucket", "raw/reference.fna.gz", source)
        self.assertEqual(self.client.objects["raw/reference.fna.gz"], content)
        self.assertEqual(
            (
                receipt.values["bytes"],
                receipt.values["sha256"],
                receipt.values["md5"],
                receipt.values["decompressed"],
            ),
            (
                len(content),
                hashlib.sha256(content).hexdigest(),
                hashlib.md5(content).hexdigest(),
                {"bytes": 32, "sha256": hashlib.sha256(plain).hexdigest()},
            ),
        )
        # No independent SHA-256 was compared; the md5 was.
        self.assertIsNone(receipt.values["artifact_proof"]["expected_identity"])

        changed = bytearray(content)
        changed[-9] ^= 1
        self.provider._url_opener = lambda _: io.BytesIO(bytes(changed))
        with self.assertRaises(ProviderError) as raised:
            self.provider.store.put("bucket", "raw/changed.fna.gz", source)
        self.assertEqual(
            (raised.exception.failure_class, raised.exception.detail),
            ("semantic", "source_identity_mismatch"),
        )
        self.assertNotIn("raw/changed.fna.gz", self.client.objects)

        # The measured plain FASTA is the Run's reference sequences; the excluded
        # record (chr2 here) is left out of it, the placed.gz is whole.
        self.provider._url_opener = lambda _: io.BytesIO(content)
        receipt = self.provider.store.put(
            "bucket",
            "raw/selected.fna.gz",
            dataclasses.replace(source, excluded_sequences=("chr2",)),
        )
        self.assertEqual(self.client.objects["raw/selected.fna.gz"], content)
        selected = b">chr1\nACGTACGTAC\n"
        self.assertEqual(
            receipt.values["decompressed"],
            {"bytes": len(selected), "sha256": hashlib.sha256(selected).hexdigest()},
        )

    def test_a_renamed_source_streams_once_with_ncbis_names_and_its_own_identity(self):
        """A bound Galaxy file streams URL to bucket with its contig names rewritten
        (owner I7, #464): the placed object is the renamed bytes, the receipt names
        the source's measured size and SHA-256, and a source of another size than
        the bound one refuses with a typed cause."""
        from bio_neb.failure import ProviderError
        from bio_neb.prebuilt_index.rename import ContigRename, renamed_byte_count
        from bio_neb.providers.ports import TransferSource

        names = (("chr1", "NC_000001.11"),)
        fasta = b">chr1\nACGTACGTAC\n>chrM\nTTTT\n"
        renamed = b">NC_000001.11\nACGTACGTAC\n>chrM\nTTTT\n"
        fai = b"chr1\t10\t6\t10\t11\nchrM\t4\t23\t4\t5\n"
        renamed_fai = b"NC_000001.11\t10\t14\t10\t11\nchrM\t4\t31\t4\t5\n"
        cases = {
            "fasta": (
                fasta,
                renamed,
                ContigRename("fasta", names, renamed_byte_count(len(fasta), names)),
            ),
            "fai": (fai, renamed_fai, ContigRename("fai", names)),
        }
        for name, (source, placed, rename) in cases.items():
            with self.subTest(name):
                self.provider._url_opener = lambda _, body=source: io.BytesIO(body)
                receipt = self.provider.store.put(
                    "bucket",
                    f"candidate/{name}",
                    TransferSource(
                        "url",
                        f"https://galaxy.invalid/x.{name}",
                        len(source),
                        None,
                        transform=rename,
                    ),
                )
                self.assertEqual(self.client.objects[f"candidate/{name}"], placed)
                self.assertEqual(
                    (
                        receipt.values["bytes"],
                        receipt.values["sha256"],
                        receipt.values["source"],
                    ),
                    (
                        len(placed),
                        hashlib.sha256(placed).hexdigest(),
                        {
                            "bytes": len(source),
                            "sha256": hashlib.sha256(source).hexdigest(),
                        },
                    ),
                )
                grown = source + b"\n"
                self.provider._url_opener = lambda _, body=grown: io.BytesIO(body)
                with self.assertRaises(ProviderError) as raised:
                    self.provider.store.put(
                        "bucket",
                        f"candidate/{name}-changed",
                        TransferSource(
                            "url",
                            f"https://galaxy.invalid/x.{name}",
                            len(source),
                            None,
                            transform=rename,
                        ),
                    )
                self.assertEqual(
                    (raised.exception.failure_class, raised.exception.detail),
                    ("semantic", "source_identity_mismatch"),
                )
                self.assertNotIn(f"candidate/{name}-changed", self.client.objects)

    def test_copy_source_parts_and_explicit_full_read_use_existing_core(self):
        content = b"abcdefghij"
        self.client.objects["source"] = content
        self.client.metadata["source"] = {
            "checksum-sha256": hashlib.sha256(content).hexdigest()
        }
        from bio_neb.verified_artifacts import ArtifactProof

        source = ArtifactProof(
            identity={
                "byte_count": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            location="s3://origin/source",
            producer="test source",
            expected_identity=None,
        )
        receipt = self.provider.store.copy(source, "target", "replica")
        self.assertEqual(
            self.client.copy_ranges, ["bytes=0-3", "bytes=4-7", "bytes=8-9"]
        )
        self.assertEqual(self.client.objects["replica"], content)
        self.assertEqual(receipt.values["verification_mode"], "controlled_copy")
        self.assertNotIn("get", self.client.calls)
        reused = self.provider.store.copy(source, "target", "replica")
        self.assertEqual(reused.values["verification_mode"], "controlled_copy")
        self.assertEqual(len(self.client.copy_ranges), 3)
        checked = self.provider.store.head(
            "target", "replica", verification="independent_full_read"
        )
        self.assertEqual(checked.values["sha256"], receipt.values["sha256"])
        self.assertIn("get", self.client.calls)

    def test_reuse_receipt_attests_metadata_without_reading_the_local_source(self):
        from bio_neb.providers.ports import TransferSource

        content = b"abcdefghij"
        digest = hashlib.sha256(content).hexdigest()
        self.client.objects["retained"] = content
        self.client.metadata["retained"] = {"checksum-sha256": digest}
        receipt = self.provider.store.put(
            "bucket",
            "retained",
            TransferSource("local", "/absent", len(content), digest),
        )
        self.assertEqual(receipt.values["verification_mode"], "named_creation_reuse")
        self.assertEqual(receipt.values["sha256"], digest)
        self.assertEqual(self.client.calls, ["head"])
        read = self.provider.store.read_small("bucket", "retained")
        self.assertEqual(read.content, content)
        self.assertEqual(read.values["verification_mode"], "independent_full_read")

    def run_failed_stage(self, step, *, port="store", failure=None):
        """Current approved stage test with a live positive-flow absence baseline.

        ``failure`` is the stage's entry in the Run's failure section (#223).
        """
        import json
        from pathlib import Path

        from bio_neb.failure import FailureTuning, run_failure_tuning
        from bio_neb.journey import Journey
        from bio_neb.journey_path import (
            ExecutionBound,
            JourneyPath,
            PathFact,
            project_execution,
        )
        from bio_neb.parabricks.execution_contact_accounting import (
            ExecutionContactJournal,
        )
        from bio_neb.parabricks.plan import RetainedExecution, plan_execution
        from bio_neb.parabricks.runner import StageDef, run_stage
        from bio_neb.recording import processing_pass
        from tests.parabricks_operation_fakes import approved_input

        fixture = load_issue98_execution_fixture()
        self.assertEqual(len(fixture.scenarios), 1)
        baseline = fixture.scenarios[0]
        positive_boundaries = tuple(
            name
            for name in baseline.path_events
            if name in {"InputsVerified", "ReferencePrepared", "H100ArtifactsVerified"}
        )
        self.assertEqual(
            positive_boundaries,
            ("InputsVerified", "ReferencePrepared", "H100ArtifactsVerified"),
        )
        approved = approved_input(self.root / "failure-run")
        owner = Journey.open(Path(approved.journey_file))

        def unexpected_success(receipts):
            self.fail("failed provider operation must not publish a successful stage")

        with owner.processing(), processing_pass(owner) as interval:
            try:
                interval.start()
                journey = JourneyPath.open(Path(approved.journey_file))
                plan = plan_execution(
                    approved,
                    RetainedExecution(
                        journey.state, approved.authority, approved.proposal
                    ),
                )
                _, state = journey.path(
                    journey.state,
                    PathFact(
                        event_type="ExecutionBound",
                        kind="authority",
                        grounding="human",
                        **{"pass": 1},
                        payload=ExecutionBound(
                            execution_id=plan.identity.execution_id,
                            authority_id=plan.identity.authority_id,
                            proposal_id=plan.identity.proposal_id,
                            run_id=plan.identity.journey_id,
                            approved_input=approved.model_dump(mode="json"),
                        ),
                    ),
                )
                journal = ExecutionContactJournal(
                    interval.journal.path,
                    approved.execution_id,
                    processing_interval=interval,
                    owner_event_id=state.latest_event_ids["ExecutionBound"],
                )
                self.provider._journal = journal
                tuning = FailureTuning.model_validate(
                    {"scopes": {"stage": {"input_storage": failure or {}}}}
                )
                with run_failure_tuning(tuning):
                    state = run_stage(
                        state,
                        plan,
                        StageDef(
                            name="input_storage",
                            steps=(step,),
                            ports={step.__name__: port},
                            path_fact=unexpected_success,
                        ),
                        self.provider,
                        journey=journey,
                        journal=journal,
                        live=lambda _: None,
                    )
                journal.finalize()
                interval.close(
                    "waiting" if state.failure_review is not None else "completed"
                )
            except BaseException:
                interval.close("interrupted")
                raise
        reader = JourneyPath.open(journey.file)
        self.assertEqual(project_execution(reader.read()), state)
        for boundary in positive_boundaries:
            self.assertNotIn(boundary, state.latest_event_ids)
        verified_roles = {p.role for p in (*baseline.inputs, *baseline.reference)} | {
            item.role for item in baseline.verified_objects
        }
        published_roles = {
            p.role
            for event in reader.read()
            if event.kind.value == "verified_artifact"
            for p in getattr(event.payload, "pointers", ())
        }
        self.assertTrue(verified_roles.isdisjoint(published_roles))
        failed = state.failures[0]
        self.assertEqual(
            (failed.module, failed.stage, failed.step),
            (__name__, "input_storage", step.__name__),
        )
        evidence = json.loads(reader.read_artifact(failed.evidence[0]))
        rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
        attempts = [
            row["domain"]["payload"]
            for row in rows
            if row.get("domain") is not None
            and row["domain"]["owner_event_id"]
            == state.latest_event_ids["ExecutionBound"]
            and row["domain"]["operation_id"] == approved.execution_id
            and row["domain"]["payload"].get("phase") == "claimed"
        ]
        return state, evidence, attempts

    def test_same_size_local_checksum_failure_retains_cause_without_url_fallback(self):
        from unittest.mock import Mock

        from bio_neb.providers.ports import TransferSource

        path = self.root / "wrong-local"
        path.write_bytes(b"abcdefghij")
        self.provider._url_opener = Mock(side_effect=AssertionError("no URL fallback"))

        def upload(ctx, store):
            return store.put(
                "bucket", "bad-local", TransferSource("local", str(path), 10, "0" * 64)
            )

        state, retained, attempts = self.run_failed_stage(upload)
        # A semantic failure asks its Review on default policy.
        self.assertEqual(state.status, "execution_review_pending")
        self.assertEqual(
            state.failure_review.failure_event_id,
            state.latest_event_ids["StageFailed"],
        )
        self.assertEqual(state.failures[0].failure_class, "semantic")
        self.assertEqual(retained["detail"], "source_identity_mismatch")
        self.assertEqual(
            retained["evidence"],
            {
                "kind": "multipart_residue",
                "operation": "multipart_part",
                "safe_error_category": "checksum_mismatch",
                "bytes_read": 10,
                "residue": [{"object_key": "bad-local", "upload_id": "opaque-upload"}],
            },
        )
        self.assertEqual(len(attempts), 1)
        self.assertNotIn("bad-local", self.client.objects)
        self.assertNotIn("multipart_complete", self.client.calls)
        self.assertEqual(self.client.calls.count("multipart_create"), 1)
        self.provider._url_opener.assert_not_called()

    def test_cli_diagnosis_survives_actual_runner_to_journey_without_raw_streams(self):
        import subprocess
        from contextlib import redirect_stderr
        from types import SimpleNamespace

        from bio_neb.providers.nebius_common import (
            NebiusCliInstallIdentity,
            NebiusCliRunner,
        )

        # Synthetic output follows the supported CLI diagnosis format; no provider contact.
        stderr = (
            "Error: rpc error: code = NotFound desc = secret-sentinel\n"
            "Trace ID: ba6fd61a3f05c84c639836764059c44e\n"
            "https://auth.invalid/?token=secret-sentinel\n"
        )
        calls = []

        def process(argv, **kwargs):
            calls.append(argv)
            if argv[-1] == "version":
                return subprocess.CompletedProcess(argv, 0, "0.12.263\n", "")
            return subprocess.CompletedProcess(argv, 13, "raw-stdout-sentinel", stderr)

        self.provider._runner = NebiusCliRunner(
            expected_identity=NebiusCliInstallIdentity(
                executable="/offline/nebius",
                resolved_executable="/offline/nebius",
                version="0.12.263",
                byte_count=1,
                sha256="a" * 64,
            ),
            process_runner=process,
            executable_inspector=lambda _: SimpleNamespace(
                resolved_executable="/offline/nebius",
                byte_count=1,
                sha256="a" * 64,
            ),
        )

        def identity(ctx, identity):
            return identity.whoami()

        with redirect_stderr(io.StringIO()):
            state, retained, attempts = self.run_failed_stage(identity, port="identity")
        self.assertEqual(state.failures[0].failure_class, "rejected")
        self.assertEqual(retained["detail"], "nebius_command_failed")
        self.assertEqual(retained["evidence"]["code"], "provider_command_failed")
        self.assertEqual(
            retained["evidence"]["journey_reason"],
            "provider_command_failed;exit=13;rpc=NotFound;trace=ba6fd61a3f05c84c639836764059c44e",
        )
        fields = retained["evidence"]["failure_evidence"]
        self.assertIn("child_exit_code:13", fields)
        self.assertIn("rpc_code:NotFound", fields)
        self.assertIn("stderr_classification:not_found", fields)
        self.assertIn(
            "stderr_sha256:" + hashlib.sha256(stderr.encode()).hexdigest(), fields
        )
        for unsafe in (
            "secret-sentinel",
            "auth.invalid",
            "raw-stdout-sentinel",
            "desc =",
        ):
            self.assertNotIn(unsafe, str(retained))
        self.assertEqual(len(attempts), 1)
        self.assertEqual(
            len(calls), 2
        )  # Pinned preflight + one failed command, no replay.

    def test_url_checksum_uses_explicit_semantic_review_policy_without_replay(self):
        from bio_neb.failure import Ask
        from bio_neb.providers.ports import TransferSource

        def upload(ctx, store):
            return store.put(
                "bucket",
                "bad-url",
                TransferSource("url", "https://source.invalid/bad", 10, "0" * 64),
            )

        state, retained, attempts = self.run_failed_stage(
            upload, failure={"operation": {"semantic": "ask"}}
        )
        self.assertIsInstance(state.failure_review, Ask)
        self.assertEqual(
            state.failures[0].context["failure_policy"]["surface_overrides"],
            {"semantic": "ask"},
        )
        self.assertEqual(state.failures[0].failure_class, "semantic")
        self.assertEqual(
            state.failures[0].context["original_failure_class"], "semantic"
        )
        self.assertEqual(retained["detail"], "source_identity_mismatch")
        self.assertEqual(
            retained["evidence"]["safe_error_category"], "checksum_mismatch"
        )
        self.assertEqual(len(attempts), 1)
        self.assertNotIn("bad-url", self.client.objects)
        self.assertNotIn("multipart_complete", self.client.calls)

    def test_pretransfer_http_failure_retries_boundedly_without_starting_upload(self):
        from unittest.mock import Mock
        from urllib.error import HTTPError

        from bio_neb.providers.ports import TransferSource

        self.provider._url_opener = Mock(
            side_effect=HTTPError(
                "https://source.invalid/reference",
                503,
                "unsafe-provider-text",
                {},
                None,
            )
        )

        def upload(ctx, store):
            return store.put(
                "bucket",
                "unavailable",
                TransferSource("url", "https://source.invalid/reference", 10, "0" * 64),
            )

        with patch("bio_neb.parabricks.runner.time.sleep") as sleep:
            state, retained, attempts = self.run_failed_stage(upload)
        self.assertEqual(state.failures[0].failure_class, "exhausted")
        self.assertEqual(
            state.failures[0].context["original_failure_class"], "transient"
        )
        self.assertEqual(
            retained,
            {"detail": "source_http_unavailable", "evidence": {"http_status": 503}},
        )
        self.assertEqual(len(attempts), 3)
        self.assertEqual(self.provider._url_opener.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("multipart_create", self.client.calls)
        self.assertEqual(self.client.objects, {"run/data": b"data"})

    def test_interrupted_upload_retains_residue_and_stops_for_review_without_replay(
        self,
    ):
        from bio_neb.failure import Ask
        from bio_neb.providers.ports import TransferSource

        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell() >= 4:
                    raise TimeoutError("secret-stream-sentinel")
                return super().read(min(size, 4))

        self.provider._url_opener = lambda _: Interrupted(b"abcdefghij")

        def upload(ctx, store):
            return store.put(
                "bucket",
                "interrupted",
                TransferSource(
                    "url",
                    "https://source.invalid/reference",
                    10,
                    hashlib.sha256(b"abcdefghij").hexdigest(),
                ),
            )

        state, retained, attempts = self.run_failed_stage(upload)
        self.assertIsInstance(state.failure_review, Ask)
        self.assertEqual(state.failures[0].failure_class, "exhausted")
        self.assertEqual(
            retained,
            {
                "detail": "storage_transfer_incomplete",
                "evidence": {
                    "kind": "mutation_ambiguous",
                    "operation": "multipart_part",
                    "safe_error_category": "transport_interrupted",
                    "bytes_read": 4,
                    "residue": [
                        {"object_key": "interrupted", "upload_id": "opaque-upload"}
                    ],
                },
            },
        )
        self.assertEqual(len(attempts), 1)
        self.assertEqual(self.client.calls.count("multipart_create"), 1)
        self.assertEqual(self.client.calls.count("multipart_part"), 1)
        self.assertNotIn("multipart_complete", self.client.calls)
        self.assertNotIn("interrupted", self.client.objects)

    def test_storage_access_retains_safe_status_code_and_contacts(self):
        self.client = _QueuedClient(head=[_error(403, "AccessDenied")])

        def inspect(ctx, store):
            return store.head("bucket", "refused")

        state, retained, attempts = self.run_failed_stage(inspect)
        self.assertEqual(state.failures[0].failure_class, "rejected")
        self.assertEqual(
            retained,
            {
                "detail": "storage_access_refused",
                "evidence": {
                    "kind": "access",
                    "operation": "head",
                    "detail": {"status": 403, "code": "AccessDenied"},
                    "contacts": [
                        {
                            "operation": "head",
                            "ordinal": 1,
                            "classification": "access",
                            "detail": {"status": 403, "code": "AccessDenied"},
                        }
                    ],
                },
            },
        )
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(self.client.calls), 1)

    def test_hash_drift_never_publishes_successful_transfer_receipt(self):
        from bio_neb.failure import ProviderError
        from bio_neb.parabricks.execution_contact_accounting import (
            ExecutionContactJournal,
            read_complete_execution_contacts,
        )
        from bio_neb.providers.ports import TransferSource

        journal = ExecutionContactJournal(
            self.root / "contacts.jsonl", "parabricks-execution:" + "a" * 64
        )
        self.provider._journal = journal
        with self.assertRaises(ProviderError) as raised:
            self.provider.store.put(
                "bucket",
                "bad",
                TransferSource("url", "https://source.invalid/bad", 10, "0" * 64),
            )
        self.assertEqual(raised.exception.failure_class, "semantic")
        self.assertEqual(raised.exception.detail, "source_identity_mismatch")
        self.assertNotIn("bad", self.client.objects)
        journal.finalize()
        rows = read_complete_execution_contacts(
            journal.path.read_bytes(), journal.execution_id
        )
        source = [
            row
            for row in rows
            if row.category == "source" and row.row_kind == "terminal"
        ]
        self.assertEqual(
            [(row.disposition, row.parse_classification) for row in source],
            [("succeeded", "response_received")],
        )
        self.assertEqual(
            raised.exception.evidence["safe_error_category"], "checksum_mismatch"
        )
        self.assertEqual(raised.exception.evidence["bytes_read"], 10)

    def test_http_200_with_interrupted_body_retains_failed_get(self):
        from bio_neb.failure import ProviderError
        from bio_neb.parabricks.execution_contact_accounting import (
            ExecutionContactJournal,
            read_complete_execution_contacts,
        )
        from bio_neb.providers.ports import TransferSource

        class Interrupted(io.BytesIO):
            status = 200

            def read(self, size=-1):
                if self.tell() >= 4:
                    raise ConnectionResetError("offline interrupted response")
                return super().read(min(size, 4))

        self.provider._url_opener = lambda _: Interrupted(b"abcdefghij")
        journal = ExecutionContactJournal(
            self.root / "contacts.jsonl", "parabricks-execution:" + "a" * 64
        )
        self.provider._journal = journal
        with self.assertRaises(ProviderError) as raised:
            self.provider.store.put(
                "bucket",
                "interrupted",
                TransferSource(
                    "url",
                    "https://source.invalid/reference",
                    10,
                    hashlib.sha256(b"abcdefghij").hexdigest(),
                ),
            )
        journal.finalize()
        rows = read_complete_execution_contacts(
            journal.path.read_bytes(), journal.execution_id
        )
        self.assertEqual(
            [
                row.disposition
                for row in rows
                if row.category == "source" and row.row_kind == "terminal"
            ],
            ["failed"],
        )
        self.assertEqual(
            raised.exception.evidence["safe_error_category"], "transport_interrupted"
        )
        self.assertNotIn("interrupted", self.client.objects)

    def test_lost_or_malformed_create_reconciles_stored_bucket_once(self):
        from bio_neb.providers.nebius_common import (
            AmbiguousNebiusCreate,
            NebiusCliError,
        )

        for response in (
            AmbiguousNebiusCreate(),
            NebiusCliError("provider_json_invalid"),
            {},
            {"metadata": None},
        ):
            with self.subTest(response=response):
                buckets = []
                operations = []
                self.client.calls.clear()

                def contact(
                    *args, operations=operations, buckets=buckets, response=response
                ):
                    operation = args[2]
                    operations.append(operation)
                    self.assertEqual(
                        args[args.index("--parent-id") + 1], self.scope.project_id
                    )
                    if operation == "list":
                        return {"items": list(buckets)}
                    self.assertEqual(operation, "create")
                    buckets.append(
                        {
                            "metadata": {
                                "id": "created-id",
                                "name": "fresh",
                                "parent_id": self.scope.project_id,
                            },
                            "status": {
                                "state": "ACTIVE",
                                "region": "region",
                                "domain_name": "storage.invalid",
                            },
                        }
                    )
                    if isinstance(response, Exception):
                        raise response
                    return response

                self.runner.json.side_effect = contact
                result = self.provider.store.fresh_bucket("fresh")
                self.assertEqual(result.provider_ids, ("created-id",))
                self.assertEqual(result.resource_count, 1)
                self.assertFalse(result.values["reused"])
                self.assertEqual(operations, ["list", "create", "list"])
                self.assertEqual(len(buckets), 1)
                self.assertEqual(self.client.calls.count("put_lifecycle"), 1)

    def test_undecodable_create_output_reconciles_through_actual_cli_runner(self):
        import json
        import subprocess
        from types import SimpleNamespace

        from bio_neb.failure import ProviderError
        from bio_neb.providers.nebius_common import (
            NebiusCliInstallIdentity,
            NebiusCliRunner,
        )

        for visible in (True, False):
            with self.subTest(reconciliation_visible=visible):
                buckets, operations = [], []
                self.client.calls.clear()

                def process(
                    argv,
                    *,
                    visible=visible,
                    buckets=buckets,
                    operations=operations,
                    **kwargs,
                ):
                    if argv[-1] == "version":
                        return subprocess.CompletedProcess(argv, 0, "0.12.263\n", "")
                    operation = argv[3]
                    operations.append(operation)
                    if operation == "create":
                        buckets.append(
                            {
                                "metadata": {
                                    "id": "created-id",
                                    "name": "fresh",
                                    "parent_id": self.scope.project_id,
                                },
                                "status": {
                                    "state": "ACTIVE",
                                    "region": "region",
                                    "domain_name": "storage.invalid",
                                },
                            }
                        )
                        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
                    self.assertEqual(operation, "list")
                    return subprocess.CompletedProcess(
                        argv, 0, json.dumps({"items": buckets if visible else []}), ""
                    )

                self.provider._runner = NebiusCliRunner(
                    expected_identity=NebiusCliInstallIdentity(
                        executable="/offline/nebius",
                        resolved_executable="/offline/nebius",
                        version="0.12.263",
                        byte_count=1,
                        sha256="a" * 64,
                    ),
                    process_runner=process,
                    executable_inspector=lambda _: SimpleNamespace(
                        resolved_executable="/offline/nebius",
                        byte_count=1,
                        sha256="a" * 64,
                    ),
                )
                if visible:
                    receipt = self.provider.store.fresh_bucket("fresh")
                    self.assertEqual(receipt.provider_ids, ("created-id",))
                    self.assertEqual(receipt.resource_count, 1)
                    self.assertEqual(self.client.calls.count("put_lifecycle"), 1)
                else:
                    with self.assertRaises(ProviderError) as raised:
                        self.provider.store.fresh_bucket("fresh")
                    self.assertEqual(raised.exception.detail, "bucket_create_ambiguous")
                    self.assertEqual(
                        raised.exception.evidence["create"]["code"],
                        "provider_output_invalid_utf8",
                    )
                    self.assertNotIn("put_lifecycle", self.client.calls)
                self.assertEqual(operations, ["list", "create", "list"])
                self.assertEqual(len(buckets), 1)

    def test_uncertain_bucket_identity_never_replays_create_or_configures_lifecycle(
        self,
    ):
        from bio_neb.failure import ProviderError
        from bio_neb.providers.nebius_common import (
            AmbiguousNebiusCreate,
            NebiusCliError,
        )

        valid = {
            "metadata": {
                "id": "created-id",
                "name": "fresh",
                "parent_id": self.scope.project_id,
            },
            "status": {
                "state": "ACTIVE",
                "region": "region",
                "domain_name": "storage.invalid",
            },
        }
        import copy

        observations = [
            {"items": []},
            {"items": [valid, valid]},
            {"items": None},
            NebiusCliError("provider_command_failed"),
        ]
        for group, field, value in (
            ("metadata", "parent_id", "other-project"),
            ("metadata", "name", "other-name"),
            ("metadata", "id", None),
            ("status", "state", "CREATING"),
            ("status", "region", "other-region"),
            ("status", "domain_name", "other.invalid"),
            ("status", "suspension_state", []),
        ):
            bucket = copy.deepcopy(valid)
            bucket[group][field] = value
            observations.append({"items": [bucket]})
        for observation in observations:
            with self.subTest(observation=observation):
                operations = []
                stored = []

                def contact(
                    *args, operations=operations, stored=stored, observation=observation
                ):
                    operations.append(args[2])
                    if args[2] == "create":
                        stored.append(valid)
                        raise AmbiguousNebiusCreate()
                    if not stored:
                        return {"items": []}
                    if isinstance(observation, Exception):
                        raise observation
                    return observation

                self.runner.json.side_effect = contact
                with self.assertRaises(ProviderError) as raised:
                    self.provider.store.fresh_bucket("fresh")
                self.assertEqual(raised.exception.detail, "bucket_create_ambiguous")
                self.assertEqual(raised.exception.failure_class, "exhausted")
                self.assertIn("create", raised.exception.evidence)
                self.assertIn("reconciliation", raised.exception.evidence)
                self.assertEqual(operations, ["list", "create", "list"])
                self.assertEqual(len(stored), 1)
                self.assertNotIn("put_lifecycle", self.client.calls)

    def test_provider_authentication_reuses_scope_across_identity_and_job_preflight(
        self,
    ):
        from types import SimpleNamespace

        from tests.nebius_serverless_job_contact_fakes import _target

        target = _target().model_copy(
            update={
                "profile": self.scope.profile,
                "tenant_id": self.scope.tenant_id,
                "project_id": self.scope.project_id,
            }
        )
        self.runner.preflight.return_value = SimpleNamespace(
            executable=target.executable,
            resolved_executable=target.resolved_executable,
            version=target.cli_version,
            byte_count=target.executable_byte_count,
            sha256=target.executable_sha256,
        )
        operations = []

        def contact(*args):
            operations.append(args[:2] if args[:2] == ("iam", "whoami") else args[:3])
            self.assertEqual(args[args.index("--profile") + 1], self.scope.profile)
            self.assertIn("--auth-timeout", args)
            self.assertEqual(args[args.index("--auth-timeout") + 1], "15m0s")
            if args[:2] == ("iam", "whoami"):
                return {
                    "service_account_profile": {
                        "info": {
                            "status": {"active": True},
                            "metadata": {"parent_id": self.scope.project_id},
                        }
                    }
                }
            if args[:3] == ("vpc", "subnet", "get"):
                return {
                    "metadata": {
                        "id": target.subnet_id,
                        "parent_id": self.scope.project_id,
                    },
                    "status": {"state": "READY"},
                }
            self.assertEqual(args[:3], ("iam", "project", "get"))
            return {
                "metadata": {
                    "id": self.scope.project_id,
                    "parent_id": self.scope.tenant_id,
                }
            }

        self.runner.json.side_effect = contact
        first = self.provider.identity.whoami()
        self.assertEqual(self.provider.identity.whoami(), first)
        self.assertEqual(len(operations), 2)
        self.provider.jobs.contact.preflight(target)
        self.provider.jobs.contact.preflight(target)
        self.assertEqual(
            operations,
            [
                ("iam", "whoami"),
                ("iam", "project", "get"),
                ("iam", "project", "get"),
                ("vpc", "subnet", "get"),
            ],
        )
        self.assertEqual(first.values["project_id"], self.scope.project_id)

    def test_bucket_precheck_reuses_exact_name_and_refuses_ambiguity(self):
        from bio_neb.failure import ProviderError

        bucket = {
            "metadata": {
                "id": "bucket-id",
                "name": "fresh",
                "parent_id": self.scope.project_id,
            },
            "status": {
                "state": "ACTIVE",
                "region": "region",
                "domain_name": "storage.invalid",
            },
        }
        self.runner.json.return_value = {"items": [bucket]}
        self.assertEqual(self.provider.store.fresh_bucket("fresh").resource_count, 1)
        self.assertEqual(self.runner.json.call_count, 1)
        self.assertNotIn("put_lifecycle", self.client.calls)
        self.runner.json.return_value = {"items": [bucket, bucket]}
        with self.assertRaisesRegex(ProviderError, "ambiguous"):
            self.provider.store.fresh_bucket("fresh")

    def test_new_bucket_uses_intelligent_class_and_only_incomplete_upload_lifecycle(
        self,
    ):
        bucket = {
            "metadata": {
                "id": "bucket-id",
                "name": "fresh",
                "parent_id": self.scope.project_id,
            },
            "status": {
                "state": "ACTIVE",
                "region": "region",
                "domain_name": "storage.invalid",
            },
        }
        self.runner.json.side_effect = [{"items": []}, bucket]
        self.assertEqual(
            self.provider.store.fresh_bucket("fresh").provider_ids, ("bucket-id",)
        )
        self.assertIn("intelligent", self.runner.json.call_args.args)
        self.assertEqual(
            self.client.lifecycle["Rules"][0]["AbortIncompleteMultipartUpload"],
            {"DaysAfterInitiation": 7},
        )

    def test_missing_credentials_retains_safe_cause_through_identity_and_journey(self):
        from bio_neb.providers.nebius_common import EnvironmentAwsCredentialResolver

        self.provider._credentials = EnvironmentAwsCredentialResolver({})
        self.runner.json.side_effect = [
            {
                "service_account_profile": {
                    "info": {
                        "status": {"active": True},
                        "metadata": {"parent_id": self.scope.project_id},
                    }
                }
            },
            {
                "metadata": {
                    "id": self.scope.project_id,
                    "parent_id": self.scope.tenant_id,
                }
            },
        ]

        def identity(ctx, identity):
            return identity.whoami()

        state, retained, attempts = self.run_failed_stage(identity, port="identity")
        self.assertEqual(state.failures[0].failure_class, "rejected")
        self.assertEqual(
            retained,
            {
                "detail": "storage_access_refused",
                "evidence": {"kind": "credentials_unavailable"},
            },
        )
        self.assertEqual(len(attempts), 1)
        self.assertEqual(self.client.calls, [])

    def test_identity_checks_project_and_returns_no_credentials(self):
        self.runner.json.side_effect = [
            {
                "service_account_profile": {
                    "info": {
                        "status": {"active": True},
                        "metadata": {"parent_id": self.scope.project_id},
                    }
                }
            },
            {
                "metadata": {
                    "id": self.scope.project_id,
                    "parent_id": self.scope.tenant_id,
                }
            },
        ]
        receipt = self.provider.identity.whoami()
        self.assertEqual(
            receipt.values,
            {
                "profile_kind": "service_account_profile",
                "project_id": self.scope.project_id,
                "tenant_id": self.scope.tenant_id,
            },
        )


if __name__ == "__main__":
    unittest.main()


class StoreEdgeClassificationTests(unittest.TestCase):
    """Each store method classifies native storage errors at the provider edge (ADR
    0020); #365's `download` beside `read_small` keeps one classifier each."""

    def test_read_small_of_an_absent_object_is_object_missing(self):
        from types import SimpleNamespace

        from bio_neb.failure import ProviderError
        from bio_neb.providers.nebius import _Store
        from bio_neb.providers.nebius_object_storage import StorageReadAbsent

        def absent(*args, **kwargs):
            raise StorageReadAbsent("get")

        objects = SimpleNamespace(get_bytes=absent)
        store = _Store(SimpleNamespace(_objects=lambda bucket: objects))
        with self.assertRaises(ProviderError) as raised:
            store.read_small("bucket", "manifest")
        self.assertEqual(
            (raised.exception.failure_class, raised.exception.detail),
            ("rejected", "object_missing"),
        )
