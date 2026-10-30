import unittest

from bio_neb.parabricks.nebius_execution_composition import (
    make_nebius_storage_readiness_adapter,
)
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
    NebiusObjectStorageTopologyCandidate,
)
from bio_neb.providers.nebius_controller_runtime import ACCEPTED_NEBIUS_EXECUTABLE
from bio_neb.providers.nebius_object_storage_credentials import (
    SelectedNebiusCredentialResolver,
)
from bio_neb.providers.nebius_storage_readiness import NebiusStorageReadinessAdapter
from bio_neb.storage_contracts import (
    ExecutionStorageReadinessConfig,
    InputStorageNebiusTarget,
    InputStorageObjectStorageTarget,
)

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


class NebiusExecutionCompositionTests(unittest.TestCase):
    def test_production_readiness_factory_uses_selected_resolver_and_storage_adapter(
        self,
    ):
        """Construction proof only: no resolver, CLI, or Object Storage contact."""

        selection = _selection()
        config = ExecutionStorageReadinessConfig(
            nebius=InputStorageNebiusTarget(
                controller_runtime="native_ubuntu",
                executable=selection.nebius_executable,
                resolved_executable=selection.resolved_nebius_executable,
                cli_version=selection.nebius_cli_version,
                executable_byte_count=selection.nebius_executable_byte_count,
                executable_sha256=selection.nebius_executable_sha256,
                nebius_cli_profile=selection.topology.profile,
                tenant_id=selection.topology.tenant_id,
                project_id=selection.topology.project_id,
            ),
            object_storage=InputStorageObjectStorageTarget(
                endpoint_url="https://storage.eu-north1.nebius.cloud",
                region="eu-north1",
            ),
            bucket_name="bio-neb-factory-readiness",
            bucket_domain="storage.eu-north1.nebius.cloud",
        )
        adapter = make_nebius_storage_readiness_adapter(config, selection)
        self.assertIsInstance(adapter, NebiusStorageReadinessAdapter)
        self.assertIsInstance(adapter._credentials, SelectedNebiusCredentialResolver)
        self.assertIs(adapter._credentials._target, config.nebius)
        self.assertIs(adapter._credentials._selection, selection)


if __name__ == "__main__":
    unittest.main()
