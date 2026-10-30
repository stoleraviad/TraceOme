import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from bio_neb.cli import main
from bio_neb.journey import canonical_json
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
    NebiusObjectStorageTopologyCandidate,
)
from bio_neb.providers.nebius_controller_runtime import ACCEPTED_NEBIUS_EXECUTABLE
from bio_neb.storage_contracts import (
    ExecutionInputStorageAdapterError,
    FreshBucketNameAvailable,
    FreshBucketReceipt,
    InputStorageAuthenticationReceipt,
    InputStorageInventory,
    InputStorageReadinessWriteReceipt,
    InventoryObject,
    MultipartUpload,
    ObjectStorageCredentialReuseReceipt,
)
from tests.content_identity_vectors import assert_json_identity

_EXECUTABLE = ACCEPTED_NEBIUS_EXECUTABLE
_EXECUTABLE_SHA256 = "8712903f152120e40ec64e96d242e754496e46dffde3b491a8f1a0d4cf67896d"


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
            access_key_id="aws-key-selected",
            secret_delivery_mode="mystery_box",
        ),
    )


class _ReadinessAdapter:
    """Offline protocol fake; its calls are not live S3/provider proof."""

    def __init__(self, *, failure: str | None = None) -> None:
        self.failure = failure
        self.calls: list[str] = []

    @staticmethod
    def _authentication() -> InputStorageAuthenticationReceipt:
        return InputStorageAuthenticationReceipt(
            nebius_cli_profile="selected-profile",
            tenant_id="tenant-selected",
            project_id="project-selected",
            object_storage_credential=ObjectStorageCredentialReuseReceipt(
                service_account_id="serviceaccount-selected",
                access_key_resource_id="accesskey-selected",
                access_key_id="aws-key-selected",
                exact_topology_match_count=1,
                access_key_state="ACTIVE",
                secret_state="ACTIVE",
                secret_delivery_mode="mystery_box",
                host_secret_resolved=True,
                worker_secret_version_bound=True,
            ),
        )

    def authenticate(self, config):
        self.calls.append("authenticate")
        return self._authentication()

    def precheck_fresh_bucket(self, config):
        self.calls.append("precheck_fresh_bucket")
        if self.failure == "freshness":
            raise ExecutionInputStorageAdapterError(
                "bucket_freshness_unknown", "bucket_freshness"
            )
        return FreshBucketNameAvailable(bucket_name=config.bucket_name)

    def create_fresh_bucket(self, config, record_create_invoked):
        self.calls.append("create_fresh_bucket")
        record_create_invoked()
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

    def verify_readiness_write(
        self, config, bucket, record_write_claimed, record_write_invoked
    ):
        self.calls.append("verify_readiness_write")
        record_write_claimed()
        record_write_invoked()
        return InputStorageReadinessWriteReceipt(
            lifecycle_rule_id="bio-neb-readiness-test",
            status="Disabled",
            exact_readback=True,
            reconciled_after_ambiguous_write=0,
        )


class _CredentialDriftAdapter(_ReadinessAdapter):
    def authenticate(self, config):
        self.calls.append("authenticate")
        return self._authentication().model_copy(
            update={
                "object_storage_credential": self._authentication().object_storage_credential.model_copy(
                    update={"service_account_id": "serviceaccount-drifted"}
                )
            }
        )


class _BucketDriftAdapter(_ReadinessAdapter):
    def create_fresh_bucket(self, config, record_create_invoked):
        receipt = super().create_fresh_bucket(config, record_create_invoked)
        return receipt.model_copy(update={"region": "region-drifted"})


class _CreateFailureWithInventoryAdapter(_ReadinessAdapter):
    def create_fresh_bucket(self, config, record_create_invoked):
        self.calls.append("create_fresh_bucket")
        record_create_invoked()
        raise ExecutionInputStorageAdapterError(
            "bucket_create_ambiguous",
            "bucket_create",
            inventory=InputStorageInventory(
                contact_complete=1,
                objects=(InventoryObject(object_key="residue", byte_count=7),),
            ),
        )


class _AuthenticationFailureAdapter(_ReadinessAdapter):
    def authenticate(self, config):
        self.calls.append("authenticate")
        raise ExecutionInputStorageAdapterError(
            "authentication_unavailable", "authentication"
        )


class _ReadinessWriteFailureAdapter(_ReadinessAdapter):
    def verify_readiness_write(
        self, config, bucket, record_write_claimed, record_write_invoked
    ):
        self.calls.append("verify_readiness_write")
        record_write_claimed()
        record_write_invoked()
        raise ExecutionInputStorageAdapterError(
            "readiness_write_mismatch", "bucket_readiness"
        )


class _FinalInventoryFailureAdapter(_ReadinessAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.inventory_calls = 0

    def inspect_inventory(self, config, bucket):
        self.calls.append("inspect_inventory")
        self.inventory_calls += 1
        if self.inventory_calls == 1:
            return InputStorageInventory(contact_complete=1)
        return InputStorageInventory(
            contact_complete=1,
            multipart_uploads=(
                MultipartUpload(object_key="residue", upload_id="upload-residue"),
            ),
        )


class StorageReadinessTests(unittest.TestCase):
    @staticmethod
    def _native_runner_factory(executable):
        assert executable == _EXECUTABLE
        identity = SimpleNamespace(
            executable=_EXECUTABLE,
            resolved_executable=_EXECUTABLE,
            version="0.12.263",
            byte_count=83_259_554,
            sha256=_EXECUTABLE_SHA256,
        )
        return SimpleNamespace(preflight=lambda: identity)

    def _main(self, arguments, **kwargs):
        return main(
            arguments,
            nebius_runner_factory=self._native_runner_factory,
            **kwargs,
        )

    @staticmethod
    def _arguments(selection_path: Path, receipt_path: Path) -> list[str]:
        return [
            "storage-readiness",
            "--selection-file",
            str(selection_path),
            "--bucket-name",
            "bio-neb-fresh-readiness",
            "--receipt-file",
            str(receipt_path),
        ]

    def _failed_readiness(self, adapter):
        """Exercise one offline Adapter fake and return its immutable failure receipt."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selection_path = root / "selection.json"
            receipt_path = root / "receipt.json"
            selection_path.write_text(
                canonical_json(_selection().model_dump(mode="json")), encoding="utf-8"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = self._main(
                    self._arguments(selection_path, receipt_path),
                    storage_readiness_adapter_factory=lambda *_: adapter,
                )
            self.assertEqual(exit_code, 2)
            terminal = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(terminal["status"], "execution_storage_readiness_failed")
            return terminal

    def test_public_command_reaches_storage_ready_with_exact_mutation_ceilings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selection_path = root / "selection.json"
            receipt_path = root / "readiness.json"
            selection_path.write_text(
                canonical_json(_selection().model_dump(mode="json")), encoding="utf-8"
            )
            adapter = _ReadinessAdapter()
            output = io.StringIO()

            def factory(config, selection):
                first = json.loads(output.getvalue().splitlines()[0])
                self.assertEqual(first["stage"], "precontact_readiness_summary")
                self.assertEqual(first["plan"]["ceilings"]["bucket_create_ceiling"], 1)
                self.assertEqual(first["plan"]["ceilings"]["h100_create_ceiling"], 0)
                self.assertFalse(hasattr(config, "fastqs"))
                self.assertFalse(hasattr(config, "source_object_key"))
                self.assertEqual(selection, _selection())
                assert_json_identity(
                    self,
                    config.configuration_id,
                    config,
                    "schema_version nebius object_storage bucket_name bucket_domain",
                    "execution-storage-readiness-configuration:",
                )
                return adapter

            with redirect_stdout(output):
                exit_code = self._main(
                    [
                        "storage-readiness",
                        "--selection-file",
                        str(selection_path),
                        "--bucket-name",
                        "bio-neb-fresh-readiness",
                        "--receipt-file",
                        str(receipt_path),
                    ],
                    storage_readiness_adapter_factory=factory,
                )
            self.assertEqual(exit_code, 0)
            stages = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(
                [item.get("stage") for item in stages[:-1]],
                [
                    "precontact_readiness_summary",
                    "authenticated_context",
                    "bucket_name_absent",
                    "bucket_create_claimed",
                    "bucket_create_invoked",
                    "bucket_active",
                    "empty_inventory_before_readiness_write",
                    "readiness_write_claimed",
                    "readiness_write_invoked",
                    "readiness_lifecycle_readback",
                    "empty_object_inventory",
                    "zero_multipart_inventory",
                ],
            )
            terminal = stages[-1]
            self.assertEqual(terminal["status"], "execution_storage_ready")
            self.assertEqual(
                terminal["counts"]["adapter_operation_invocation_count"], 6
            )
            self.assertEqual(terminal["counts"]["bucket_create_invocation_count"], 1)
            self.assertEqual(terminal["counts"]["readiness_write_invocation_count"], 1)
            for zero in (
                "reference_put_count",
                "fastq_put_count",
                "prepared_candidate_contact_count",
                "reference_preparation_create_count",
                "h100_create_count",
                "retry_count",
                "resume_count",
                "recreate_count",
                "cleanup_count",
            ):
                self.assertEqual(terminal["counts"][zero], 0)
            self.assertEqual(
                adapter.calls,
                [
                    "authenticate",
                    "precheck_fresh_bucket",
                    "create_fresh_bucket",
                    "inspect_inventory",
                    "verify_readiness_write",
                    "inspect_inventory",
                ],
            )
            terminal_bytes = output.getvalue().splitlines()[-1]
            self.assertEqual(receipt_path.read_text(encoding="utf-8"), terminal_bytes)

    def test_review_reopens_immutable_success_receipt_without_contact(self):
        """Provider-shaped fakes create the receipt; review itself is contact-free."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selection_path = root / "selection.json"
            receipt_path = root / "readiness.json"
            selection_path.write_text(
                canonical_json(_selection().model_dump(mode="json")), encoding="utf-8"
            )
            initial = io.StringIO()
            with redirect_stdout(initial):
                initial_exit = self._main(
                    self._arguments(selection_path, receipt_path),
                    storage_readiness_adapter_factory=lambda *_: _ReadinessAdapter(),
                )
            self.assertEqual(initial_exit, 0)
            terminal = json.loads(initial.getvalue().splitlines()[-1])
            terminal_bytes = receipt_path.read_bytes()

            reopened = io.StringIO()
            hostile_factory_calls = []
            with redirect_stdout(reopened):
                reopen_exit = self._main(
                    ["storage-readiness-review", "--receipt-file", str(receipt_path)],
                    storage_readiness_adapter_factory=lambda *_: (
                        hostile_factory_calls.append("contact")
                    ),
                )
            self.assertEqual(reopen_exit, 0)
            review = json.loads(reopened.getvalue())
            self.assertEqual(review["receipt"], terminal)
            self.assertEqual(review["receipt_byte_count"], len(terminal_bytes))
            self.assertTrue(review["original_receipt_bytes_unchanged"])
            for zero in (
                "reopen_adapter_contact_count",
                "reopen_nebius_cli_contact_count",
                "reopen_object_storage_contact_count",
                "reopen_write_count",
                "reopen_cleanup_count",
            ):
                self.assertEqual(review[zero], 0)
            self.assertEqual(hostile_factory_calls, [])

    def test_first_adapter_error_freezes_failure_without_later_contact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selection_path = root / "selection.json"
            receipt_path = root / "failed-readiness.json"
            selection_path.write_text(
                canonical_json(_selection().model_dump(mode="json")), encoding="utf-8"
            )
            adapter = _ReadinessAdapter(failure="freshness")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = self._main(
                    [
                        "storage-readiness",
                        "--selection-file",
                        str(selection_path),
                        "--bucket-name",
                        "bio-neb-failed-readiness",
                        "--receipt-file",
                        str(receipt_path),
                    ],
                    storage_readiness_adapter_factory=lambda *_: adapter,
                )
            self.assertEqual(exit_code, 2)
            terminal = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(terminal["status"], "execution_storage_readiness_failed")
            self.assertEqual(terminal["failure_phase"], "bucket_freshness")
            self.assertEqual(terminal["failure_reason"], "bucket_freshness_unknown")
            self.assertEqual(
                terminal["counts"]["adapter_operation_invocation_count"], 2
            )
            self.assertEqual(terminal["counts"]["bucket_create_claim_count"], 0)
            self.assertEqual(adapter.calls, ["authenticate", "precheck_fresh_bucket"])

            later_factory_calls = []
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                repeated_exit = self._main(
                    [
                        "storage-readiness",
                        "--selection-file",
                        str(selection_path),
                        "--bucket-name",
                        "bio-neb-failed-readiness",
                        "--receipt-file",
                        str(receipt_path),
                    ],
                    storage_readiness_adapter_factory=lambda *_: (
                        later_factory_calls.append("contact")
                    ),
                )
            self.assertEqual(repeated_exit, 2)
            self.assertEqual(later_factory_calls, [])
            self.assertEqual(
                json.loads(stderr.getvalue())["code"],
                "direct_storage_readiness_receipt_invalid",
            )

    def test_credential_identity_drift_fails_at_authentication_without_later_contact(
        self,
    ):
        adapter = _CredentialDriftAdapter()
        terminal = self._failed_readiness(adapter)
        self.assertEqual(terminal["failure_phase"], "authentication")
        self.assertEqual(terminal["failure_reason"], "adapter_output_invalid")
        self.assertIsNone(terminal["partial_inventory"])
        self.assertEqual(adapter.calls, ["authenticate"])

    def test_bucket_placement_drift_fails_after_single_create_without_later_contact(
        self,
    ):
        adapter = _BucketDriftAdapter()
        terminal = self._failed_readiness(adapter)
        self.assertEqual(terminal["failure_phase"], "bucket_create")
        self.assertEqual(terminal["failure_reason"], "adapter_output_invalid")
        self.assertIsNone(terminal["partial_inventory"])
        self.assertEqual(
            adapter.calls,
            ["authenticate", "precheck_fresh_bucket", "create_fresh_bucket"],
        )
        self.assertEqual(terminal["counts"]["bucket_create_invocation_count"], 1)

    def test_ambiguous_bucket_create_preserves_supplied_partial_inventory(self):
        adapter = _CreateFailureWithInventoryAdapter()
        terminal = self._failed_readiness(adapter)
        self.assertEqual(terminal["failure_phase"], "bucket_create")
        self.assertEqual(terminal["failure_reason"], "bucket_create_ambiguous")
        self.assertEqual(
            terminal["partial_inventory"]["objects"],
            [{"object_key": "residue", "byte_count": 7}],
        )
        self.assertEqual(
            adapter.calls,
            ["authenticate", "precheck_fresh_bucket", "create_fresh_bucket"],
        )

    def test_authentication_failure_stops_before_bucket_contact(self):
        adapter = _AuthenticationFailureAdapter()
        terminal = self._failed_readiness(adapter)
        self.assertEqual(terminal["failure_phase"], "authentication")
        self.assertEqual(terminal["failure_reason"], "authentication_unavailable")
        self.assertEqual(adapter.calls, ["authenticate"])
        self.assertEqual(terminal["counts"]["adapter_operation_invocation_count"], 1)

    def test_readiness_write_failure_stops_after_single_claim_and_invocation(self):
        adapter = _ReadinessWriteFailureAdapter()
        terminal = self._failed_readiness(adapter)
        self.assertEqual(terminal["failure_phase"], "bucket_readiness")
        self.assertEqual(terminal["failure_reason"], "readiness_write_mismatch")
        self.assertEqual(
            adapter.calls,
            [
                "authenticate",
                "precheck_fresh_bucket",
                "create_fresh_bucket",
                "inspect_inventory",
                "verify_readiness_write",
            ],
        )
        self.assertEqual(terminal["counts"]["readiness_write_claim_count"], 1)
        self.assertEqual(terminal["counts"]["readiness_write_invocation_count"], 1)

    def test_final_inventory_multipart_residue_is_preserved_and_terminal(self):
        adapter = _FinalInventoryFailureAdapter()
        terminal = self._failed_readiness(adapter)
        self.assertEqual(terminal["failure_phase"], "bucket_readiness")
        self.assertEqual(terminal["failure_reason"], "multipart_residue")
        self.assertEqual(
            terminal["partial_inventory"]["multipart_uploads"],
            [{"object_key": "residue", "upload_id": "upload-residue"}],
        )
        self.assertEqual(
            adapter.calls,
            [
                "authenticate",
                "precheck_fresh_bucket",
                "create_fresh_bucket",
                "inspect_inventory",
                "verify_readiness_write",
                "inspect_inventory",
            ],
        )
        self.assertEqual(terminal["counts"]["adapter_operation_invocation_count"], 6)

    def _public_refusal(
        self,
        *,
        selection_document: str,
        bucket_name: str = "bio-neb-fresh-readiness",
        precreate_receipt: bool = False,
    ) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selection_path = root / "selection.json"
            receipt_path = root / "receipt.json"
            selection_path.write_text(selection_document, encoding="utf-8")
            if precreate_receipt:
                receipt_path.write_text("retained", encoding="utf-8")
            factory_calls = []
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = self._main(
                    [
                        "storage-readiness",
                        "--selection-file",
                        str(selection_path),
                        "--bucket-name",
                        bucket_name,
                        "--receipt-file",
                        str(receipt_path),
                    ],
                    storage_readiness_adapter_factory=lambda *_: factory_calls.append(
                        "contact"
                    ),
                )
            self.assertEqual(exit_code, 2)
            self.assertEqual(factory_calls, [])
            failure = json.loads(stderr.getvalue())
            self.assertEqual(failure["status"], "failed")
            return failure

    def test_invalid_selection_refuses_before_adapter_factory(self):
        failure = self._public_refusal(selection_document="{}")
        self.assertEqual(failure["code"], "direct_storage_readiness_selection_invalid")
        self.assertEqual(failure["fields"], ["selection_file"])

    def test_invalid_bucket_name_refuses_before_adapter_factory(self):
        failure = self._public_refusal(
            selection_document=canonical_json(_selection().model_dump(mode="json")),
            bucket_name="Not_A_Bucket",
        )
        self.assertEqual(failure["code"], "direct_storage_readiness_selection_invalid")
        self.assertEqual(failure["fields"], ["selection_file", "bucket_name"])

    def test_existing_receipt_refuses_before_adapter_factory(self):
        failure = self._public_refusal(
            selection_document=canonical_json(_selection().model_dump(mode="json")),
            precreate_receipt=True,
        )
        self.assertEqual(failure["code"], "direct_storage_readiness_receipt_invalid")
        self.assertEqual(failure["fields"], ["receipt_file"])


if __name__ == "__main__":
    unittest.main()
