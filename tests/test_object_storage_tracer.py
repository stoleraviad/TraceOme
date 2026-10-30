import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from bio_neb.cli import main
from bio_neb.journey import canonical_json
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
    NebiusObjectStorageTopologyCandidate,
)
from bio_neb.providers.nebius_controller_runtime import ACCEPTED_NEBIUS_EXECUTABLE
from bio_neb.providers.nebius_object_storage import (
    ConditionalWriteResult,
    ListedObject,
    ObjectIdentity,
    ObjectMetadata,
    ReadContact,
    ReadResult,
    StorageClosure,
)
from bio_neb.storage_contracts import (
    ExecutionInputStorageAdapterError,
    FreshBucketNameAvailable,
    FreshBucketReceipt,
    InputStorageAuthenticationReceipt,
    InputStorageInventory,
    InputStorageReadinessWriteReceipt,
    ObjectStorageCredentialReuseReceipt,
)

_EXECUTABLE = ACCEPTED_NEBIUS_EXECUTABLE
_EXECUTABLE_SHA256 = "8712903f152120e40ec64e96d242e754496e46dffde3b491a8f1a0d4cf67896d"
_DATA_KEY = "object-storage-tracer/deterministic-bytes.txt"
_MANIFEST_KEY = "object-storage-tracer/manifest.json"


def _selection() -> NebiusExecutionContextSelection:
    return NebiusExecutionContextSelection(
        controller_runtime="native_ubuntu",
        nebius_executable=_EXECUTABLE,
        resolved_nebius_executable=_EXECUTABLE,
        nebius_cli_version="0.12.263",
        nebius_executable_byte_count=83_259_554,
        nebius_executable_sha256=_EXECUTABLE_SHA256,
        topology=NebiusObjectStorageTopologyCandidate.bind(
            profile="selected-profile",
            tenant_id="tenant-selected",
            project_id="project-selected",
            service_account_id="serviceaccount-selected",
            group_id="group-selected",
            group_membership_id="groupmembership-selected",
            access_permit_id="accesspermit-selected",
            access_key_resource_id="accesskey-selected",
            access_key_id="key-selected",
            secret_delivery_mode="mystery_box",
        ),
    )


class _OfflineReadinessAdapter:
    def __init__(self, failure: str | None = None) -> None:
        self.calls: list[str] = []
        self.failure = failure

    def authenticate(self, config):
        self.calls.append("authenticate")
        return InputStorageAuthenticationReceipt(
            nebius_cli_profile=config.nebius.nebius_cli_profile,
            tenant_id=config.nebius.tenant_id,
            project_id=config.nebius.project_id,
            object_storage_credential=ObjectStorageCredentialReuseReceipt(
                service_account_id="serviceaccount-selected",
                access_key_resource_id="accesskey-selected",
                access_key_id="key-selected",
                exact_topology_match_count=1,
                access_key_state="ACTIVE",
                secret_state="ACTIVE",
                secret_delivery_mode="mystery_box",
                host_secret_resolved=True,
                worker_secret_version_bound=True,
            ),
        )

    def precheck_fresh_bucket(self, config):
        self.calls.append("precheck_fresh_bucket")
        return FreshBucketNameAvailable(bucket_name=config.bucket_name)

    def create_fresh_bucket(self, config, invoked):
        self.calls.append("create_fresh_bucket")
        invoked()
        return FreshBucketReceipt(
            bucket_id="storagebucket-selected",
            parent_id=config.nebius.project_id,
            name=config.bucket_name,
            state="ACTIVE",
            suspended=False,
            domain=config.bucket_domain,
            region=config.object_storage.region,
            reconciled_after_ambiguous_create=0,
        )

    def inspect_inventory(self, config, bucket):
        self.calls.append("inspect_inventory")
        return InputStorageInventory(contact_complete=1)

    def verify_readiness_write(self, config, bucket, claimed, invoked):
        self.calls.append("verify_readiness_write")
        claimed()
        invoked()
        if self.failure == "lifecycle":
            raise ExecutionInputStorageAdapterError(
                "readiness_write_ambiguous", "bucket_readiness"
            )
        return InputStorageReadinessWriteReceipt(
            lifecycle_rule_id="bio-neb-readiness-tracer",
            status="Disabled",
            exact_readback=True,
            reconciled_after_ambiguous_write=0,
        )


class _OfflineObjectStorageAdapter:
    def __init__(self, observe) -> None:
        self.observe = observe
        self.calls: list[str] = []
        self.data = b"bio-neb-object-storage-tracer/v1\n"
        self.data_identity = ObjectIdentity(
            len(self.data), hashlib.sha256(self.data).hexdigest()
        )
        self.manifest_identity = None

    def _seen(self, operation, ordinal=1):
        contact = ReadContact(operation, ordinal, "success")
        self.observe(contact)
        return (contact,)

    def put_conditional_stream(
        self,
        key,
        source,
        *,
        expected,
        content_type="application/octet-stream",
        custom_metadata=None,
    ):
        self.calls.append(f"put:{key}")
        assert key == _DATA_KEY
        body = source.read()
        assert body == self.data
        assert expected == self.data_identity
        assert custom_metadata == {"tracer-kind": "bounded-live-acceptance"}
        return ConditionalWriteResult(self.data_identity, False)

    def head_metadata(self, key, *, expected_sha256=None):
        self.calls.append(f"head:{key}")
        assert key == _DATA_KEY and expected_sha256 == self.data_identity.sha256
        return ReadResult(
            ObjectMetadata(
                self.data_identity,
                (("tracer-kind", "bounded-live-acceptance"),),
            ),
            self._seen("head"),
        )

    def get_bytes(self, key, *, maximum_bytes):
        self.calls.append(f"get:{key}")
        assert key == _DATA_KEY and maximum_bytes == len(self.data)
        return ReadResult(self.data, self._seen("get"))

    def closure(self, prefix):
        self.calls.append("closure")
        contacts = (
            self._seen("list_objects")
            + self._seen("list_objects", 2)
            + self._seen("list_objects", 3)
            + self._seen("list_multipart")
        )
        objects = [ListedObject(_DATA_KEY, len(self.data))]
        if self.manifest_identity is not None:
            objects.append(
                ListedObject(_MANIFEST_KEY, self.manifest_identity.byte_count)
            )
        return StorageClosure(
            tuple(sorted(objects, key=lambda item: item.object_key)), (), contacts
        )

    def publish_manifest_last(
        self, *, prefix, manifest_key, body, expected_pre_inventory
    ):
        self.calls.append(f"publish:{manifest_key}")
        assert expected_pre_inventory == (ListedObject(_DATA_KEY, len(self.data)),)
        pre = self.closure(prefix)
        assert pre.objects == expected_pre_inventory and not pre.multipart
        self.manifest_identity = ObjectIdentity(
            len(body), hashlib.sha256(body).hexdigest()
        )
        self.calls.append(f"get:{manifest_key}")
        self._seen("get")
        final = self.closure(prefix)
        assert len(final.objects) == 2 and not final.multipart
        return ConditionalWriteResult(self.manifest_identity, False)


class ObjectStorageTracerTests(unittest.TestCase):
    @staticmethod
    def _offline_runner(_):
        identity = SimpleNamespace(
            executable=_EXECUTABLE,
            resolved_executable=_EXECUTABLE,
            version="0.12.263",
            byte_count=83_259_554,
            sha256=_EXECUTABLE_SHA256,
        )
        return SimpleNamespace(preflight=lambda: identity)

    def _run_offline_success_tracer(self, root: Path):
        readiness = _OfflineReadinessAdapter()
        constructed: list[_OfflineObjectStorageAdapter] = []

        def offline_adapter_factory(config, selection, observe):
            objects = _OfflineObjectStorageAdapter(observe)
            constructed.append(objects)
            return SimpleNamespace(readiness=readiness, objects=objects)

        selection_path = root / "selection.json"
        receipt_path = root / "receipt.json"
        selection_path.write_text(
            canonical_json(_selection().model_dump(mode="json")), encoding="utf-8"
        )
        with redirect_stdout(io.StringIO()):
            code = main(
                [
                    "object-storage-tracer",
                    "--selection-file",
                    str(selection_path),
                    "--bucket-name",
                    "bio-neb-t76-fresh-tracer",
                    "--receipt-file",
                    str(receipt_path),
                ],
                nebius_runner_factory=self._offline_runner,
                object_storage_tracer_adapter_factory=offline_adapter_factory,
            )
        self.assertEqual(code, 0)
        document = __import__("json").loads(receipt_path.read_text(encoding="utf-8"))
        return receipt_path, document, readiness, constructed[0]

    def test_offline_fake_public_tracer_receipt_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, document, _, objects = self._run_offline_success_tracer(Path(directory))

            self.assertEqual(document["status"], "object_storage_tracer_succeeded")
            self.assertEqual(
                document["schema_version"],
                "bio-neb.object-storage-tracer-receipt/2",
            )
            self.assertEqual(
                document["verification_policy"],
                {
                    "policy": "transition_based",
                    "data_upload_stream_full_byte_pass_count": 1,
                    "routine_host_data_full_read_count": 0,
                    "audit_host_data_full_read_count": 1,
                    "manifest_exact_readback_count": 1,
                },
            )
            self.assertFalse(document["live_transient_retry_exercised"])
            self.assertEqual(document["counts"]["serverless_job_create_count"], 0)
            self.assertEqual(document["counts"]["cleanup_count"], 0)
            self.assertEqual(document["client_policy"]["connect_timeout_seconds"], 30)
            self.assertEqual(document["client_policy"]["read_timeout_seconds"], 300)
            self.assertEqual(document["client_policy"]["sdk_total_max_attempts"], 1)
            self.assertIn(
                3,
                [
                    item["physical_contact_ordinal"]
                    for item in document["read_contacts"]
                    if item["operation"] == "list_objects"
                ],
            )
            self.assertEqual(len(document["final_closure"]["objects"]), 2)
            self.assertEqual(objects.calls.count("closure"), 2)

    def test_hostile_reopen_is_byte_exact_and_constructs_no_contacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path, _, readiness, objects = self._run_offline_success_tracer(
                Path(directory)
            )
            calls = tuple(readiness.calls), tuple(objects.calls)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "object-storage-tracer-review",
                        "--receipt-file",
                        str(receipt_path),
                    ],
                    nebius_runner_factory=lambda _: self.fail("review constructed CLI"),
                    object_storage_tracer_adapter_factory=lambda *_: self.fail(
                        "review constructed Adapter"
                    ),
                )
            self.assertEqual(code, 0)
            review = __import__("json").loads(output.getvalue())
            self.assertTrue(review["original_receipt_bytes_unchanged"])
            self.assertEqual(review["reopen_adapter_contact_count"], 0)
            self.assertEqual(review["reopen_nebius_cli_contact_count"], 0)
            self.assertEqual(review["reopen_object_storage_contact_count"], 0)
            self.assertEqual(calls, (tuple(readiness.calls), tuple(objects.calls)))

    def test_offline_fake_first_failure_retains_counts_and_makes_no_object_calls(
        self,
    ) -> None:
        readiness = _OfflineReadinessAdapter("lifecycle")
        objects: list[_OfflineObjectStorageAdapter] = []

        def offline_adapter_factory(config, selection, observe):
            selected = _OfflineObjectStorageAdapter(observe)
            objects.append(selected)
            return SimpleNamespace(readiness=readiness, objects=selected)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection.json"
            receipt_path = root / "receipt.json"
            selection_path.write_text(
                canonical_json(_selection().model_dump(mode="json")), encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "object-storage-tracer",
                        "--selection-file",
                        str(selection_path),
                        "--bucket-name",
                        "bio-neb-t76-failed-tracer",
                        "--receipt-file",
                        str(receipt_path),
                    ],
                    nebius_runner_factory=self._offline_runner,
                    object_storage_tracer_adapter_factory=offline_adapter_factory,
                )
            self.assertEqual(code, 2)
            document = __import__("json").loads(
                receipt_path.read_text(encoding="utf-8")
            )
            self.assertEqual(document["failure_reason"], "readiness_write_ambiguous")
            self.assertEqual(document["counts"]["lifecycle_write_invocation_count"], 1)
            self.assertEqual(objects[0].calls, [])


if __name__ == "__main__":
    unittest.main()
