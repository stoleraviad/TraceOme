import unittest

from botocore.exceptions import ClientError

from bio_neb.providers.nebius_common import AwsCredentialMaterial
from bio_neb.providers.nebius_storage_readiness import NebiusStorageReadinessAdapter
from bio_neb.storage_contracts import (
    ExecutionInputStorageAdapterError,
    ExecutionStorageReadinessConfig,
    FreshBucketReceipt,
    ObjectStorageCredentialReuseReceipt,
)


def _config() -> ExecutionStorageReadinessConfig:
    return ExecutionStorageReadinessConfig.model_validate(
        {
            "schema_version": "nebius-execution-storage-readiness/1",
            "nebius": {
                "controller_runtime": "native_ubuntu",
                "executable": "/opt/nebius/bin/nebius",
                "resolved_executable": "/opt/nebius/bin/nebius",
                "cli_version": "0.12.263",
                "executable_byte_count": 83259554,
                "executable_sha256": "a" * 64,
                "nebius_cli_profile": "control-profile",
                "tenant_id": "tenant-exact",
                "project_id": "project-exact",
            },
            "object_storage": {
                "endpoint_url": "https://storage.eu-north1.example.test",
                "region": "eu-north1",
            },
            "bucket_name": "fresh-run-bucket",
            "bucket_domain": "storage.eu-north1.example.test",
        }
    )


class _ReconciledCredentials:
    """Offline credential fake for lifecycle mechanics."""

    def __init__(self) -> None:
        self.receipt = ObjectStorageCredentialReuseReceipt(
            service_account_id="service-account-exact",
            access_key_resource_id="access-resource-exact",
            access_key_id="aws-key-exact",
            exact_topology_match_count=1,
            access_key_state="ACTIVE",
            secret_state="ACTIVE",
            secret_delivery_mode="mystery_box",
            host_secret_resolved=True,
            worker_secret_version_bound=True,
        )

    def __call__(self):
        return AwsCredentialMaterial(
            access_key_id="aws-key-exact",
            secret_access_key="raw-secret-value",
            credentials_profile="object-storage-profile",
            worker_secret_selector="secret-selector-id@secret-version-id",
        )


class _OfflineLifecycleClient:
    """Provider-shaped fake; its operations are not live S3/provider proof."""

    def __init__(self, *, response_lost: bool = False) -> None:
        self.response_lost = response_lost
        self.lifecycle = None
        self.operations = []

    def put_bucket_lifecycle_configuration(self, **request):
        self.operations.append("put_bucket_lifecycle_configuration")
        self.lifecycle = request["LifecycleConfiguration"]
        if self.response_lost:
            raise ClientError(
                {
                    "Error": {"Code": "ServiceUnavailable"},
                    "ResponseMetadata": {"HTTPStatusCode": 503},
                },
                "PutBucketLifecycleConfiguration",
            )

    def get_bucket_lifecycle_configuration(self, **request):
        self.operations.append("get_bucket_lifecycle_configuration")
        if self.lifecycle is None:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchLifecycleConfiguration"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetBucketLifecycleConfiguration",
            )
        return self.lifecycle


class NebiusStorageReadinessAdapterTests(unittest.TestCase):
    @staticmethod
    def _bucket(config):
        return FreshBucketReceipt(
            bucket_id="bucket-exact",
            parent_id="project-exact",
            name=config.bucket_name,
            state="ACTIVE",
            suspended=False,
            domain=config.bucket_domain,
            region=config.object_storage.region,
            reconciled_after_ambiguous_create=0,
        )

    def test_persists_disabled_rule_and_reads_it_without_object_mutation(self):
        client = _OfflineLifecycleClient()
        adapter = NebiusStorageReadinessAdapter(
            credential_resolver=_ReconciledCredentials(),
            s3_client_factory=lambda *_: client,
        )
        config = _config()
        bucket = self._bucket(config)
        claimed = []
        invoked = []

        receipt = adapter.verify_readiness_write(
            config,
            bucket,
            lambda: claimed.append("claimed"),
            lambda: invoked.append("invoked"),
        )

        self.assertEqual(claimed, ["claimed"])
        self.assertEqual(invoked, ["invoked"])
        self.assertEqual(receipt.status, "Disabled")
        self.assertEqual(receipt.reconciled_after_ambiguous_write, 0)
        rules = client.lifecycle["Rules"]
        self.assertEqual(
            [rule["ID"] for rule in rules],
            [receipt.lifecycle_rule_id, receipt.lifecycle_rule_id + "-abort-multipart"],
        )
        self.assertEqual(rules[0]["Status"], "Disabled")
        self.assertEqual(
            rules[1],
            {
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                "Filter": {},
                "ID": receipt.lifecycle_rule_id + "-abort-multipart",
                "Status": "Enabled",
            },
        )
        self.assertTrue(all("Expiration" not in rule for rule in rules[1:]))
        self.assertEqual(
            client.operations,
            [
                "get_bucket_lifecycle_configuration",
                "put_bucket_lifecycle_configuration",
                "get_bucket_lifecycle_configuration",
            ],
        )
        self.assertFalse(any("object" in operation for operation in client.operations))

    def test_response_loss_performs_one_get_reconciliation_and_no_second_put(self):
        client = _OfflineLifecycleClient(response_lost=True)
        adapter = NebiusStorageReadinessAdapter(
            credential_resolver=_ReconciledCredentials(),
            s3_client_factory=lambda *_: client,
        )
        config = _config()
        bucket = self._bucket(config)

        receipt = adapter.verify_readiness_write(
            config, bucket, lambda: None, lambda: None
        )

        self.assertEqual(receipt.reconciled_after_ambiguous_write, 1)
        self.assertEqual(
            client.operations.count("put_bucket_lifecycle_configuration"), 1
        )
        self.assertEqual(
            client.operations.count("get_bucket_lifecycle_configuration"), 2
        )

    def test_existing_lifecycle_refuses_before_claim_or_put(self):
        client = _OfflineLifecycleClient()
        client.lifecycle = {"Rules": [{"ID": "preexisting"}]}
        adapter = NebiusStorageReadinessAdapter(
            credential_resolver=_ReconciledCredentials(),
            s3_client_factory=lambda *_: client,
        )
        config = _config()
        claimed = []

        with self.assertRaises(ExecutionInputStorageAdapterError) as failure:
            adapter.verify_readiness_write(
                config,
                self._bucket(config),
                lambda: claimed.append("claimed"),
                lambda: self.fail("write invoked for pre-existing lifecycle"),
            )

        self.assertEqual(failure.exception.reason, "readiness_write_mismatch")
        self.assertEqual(claimed, [])
        self.assertEqual(client.operations, ["get_bucket_lifecycle_configuration"])


if __name__ == "__main__":
    unittest.main()
