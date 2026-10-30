"""Concrete JobRunner with fresh shared planning and isolated effect substitutes."""

import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from bio_neb.failure import ProviderError
from bio_neb.journey_path import ExecutionState
from bio_neb.parabricks import reference_recipe as contracts
from bio_neb.parabricks.plan import RetainedExecution, plan_execution
from bio_neb.parabricks.reference_stages import submit_cpu
from bio_neb.providers.nebius import NebiusProvider
from bio_neb.providers.ports import Receipt
from tests.parabricks_operation_fakes import approved_input
from tests.reference_fakes import source_acquisition
from tests.reference_stage_fakes import CpuCli as _Cli
from tests.reference_stage_fakes import cpu_observation_inputs, credentials


class ReferenceJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate with the current builders even for a single selected test.
        root = Path(cls.enterClassContext(tempfile.TemporaryDirectory()))
        approved = approved_input(root)
        cls._prepared_plan = plan_execution(
            approved,
            RetainedExecution(ExecutionState(), approved.authority, approved.proposal),
        )

    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        prepared = deepcopy(self._prepared_plan)
        # These JobRunner cases supply private request-placement storage at the
        # stage seam; they do not claim approval of a changed launch configuration.
        self.plan = replace(
            prepared,
            launch=prepared.launch.model_copy(update={"work_directory": str(root)}),
        )
        self.config = cpu_observation_inputs(self.plan)
        self.request = contracts.compile_cpu_request(
            self.plan, source_acquisition()
        ).model_copy(update={"deterministic_job_name": self.plan.layout.cpu_job_name})
        self.cli = _Cli(self.config, self.request)
        self.lineage = SimpleNamespace(
            index_digest=self.config.runtime.provider_image_index_digest,
            linux_amd64_child_digest=self.config.runtime.worker_runtime_image.rsplit(
                "@", 1
            )[1],
        )
        self.provider = self.make_provider()
        self.ctx = SimpleNamespace(
            plan=self.plan,
            receipts=(
                Receipt(),
                Receipt(
                    values={
                        "request": self.request.model_dump(mode="json"),
                        "source": contracts.VerifiedReferenceSource(
                            source_mode="existing_object",
                            reference_asset_id=self.plan.reference.reference_asset_id,
                            source_locator="object://bucket/source",
                            object=contracts.VerifiedObject(
                                role="reference_source",
                                bucket=self.plan.storage.bucket,
                                object_key=self.plan.layout.source_object_key,
                                byte_count=source_acquisition().source_byte_count,
                                sha256=source_acquisition().source_sha256,
                            ),
                            manifest_object=contracts.VerifiedObject(
                                role="source_manifest",
                                bucket=self.plan.storage.bucket,
                                object_key=self.plan.layout.source_manifest_object_key,
                                byte_count=1,
                                sha256="0" * 64,
                            ),
                            exact_inventory=tuple(
                                sorted(
                                    (
                                        self.plan.layout.source_object_key,
                                        self.plan.layout.source_manifest_object_key,
                                    )
                                )
                            ),
                            verification_mode="reuse_creation_proof",
                            independently_full_read=False,
                        ).model_dump(mode="json"),
                    }
                ),
            ),
        )

    def make_provider(self):
        return NebiusProvider(
            selection=self.plan.launch.context,
            endpoint_url=self.plan.storage.endpoint_url,
            region=self.plan.storage.region,
            journal=None,
            cli_runner=self.cli,
            credential_resolver=credentials,
            registry_resolver=lambda _: self.lineage,
        )

    def test_repeat_and_fresh_provider_reuse_exact_job_without_second_create(self):
        first = submit_cpu(self.ctx, self.provider.jobs)
        second = submit_cpu(self.ctx, self.provider.jobs)
        third = submit_cpu(self.ctx, self.make_provider().jobs)
        self.assertEqual(first.provider_ids, third.provider_ids)
        self.assertFalse(first.values["reused"])
        self.assertTrue(second.values["reused"])
        self.assertEqual(third.values["create"]["invocation_count"], 0)
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in self.cli.calls), 1
        )
        sample = self.provider.jobs.observe(first.provider_ids[0], window=1000)
        self.assertEqual(sample.values["state"], "RUNNING")
        self.assertEqual(sample.values["window"], 1000)

    def test_changed_existing_envelope_refuses_without_create(self):
        submit_cpu(self.ctx, self.provider.jobs)
        self.cli.spec["platform"] = "different"
        with self.assertRaises(ProviderError):
            submit_cpu(self.ctx, self.make_provider().jobs)
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in self.cli.calls), 1
        )

    def test_registry_drift_refuses_before_create(self):
        self.lineage.index_digest = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ProviderError, "job_image_drift"):
            submit_cpu(self.ctx, self.provider.jobs)
        self.assertFalse(self.cli.created)

    def test_completed_named_job_can_be_observed_repeatedly_after_reuse(self):
        submit_cpu(self.ctx, self.provider.jobs)
        self.cli.initial_state = "COMPLETED"
        self.cli.states = iter(("COMPLETED", "COMPLETED"))
        reused_provider = self.make_provider()
        reused = submit_cpu(self.ctx, reused_provider.jobs)
        self.assertTrue(reused.values["reused"])
        first = reused_provider.jobs.observe(reused.provider_ids[0], window=1000)
        second = reused_provider.jobs.observe(reused.provider_ids[0], window=1000)
        self.assertEqual(first.values["state"], "COMPLETED")
        self.assertFalse(second.values["state_changed"])
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in self.cli.calls), 1
        )

    def test_terminal_state_drift_is_a_classified_provider_failure(self):
        submit_cpu(self.ctx, self.provider.jobs)
        self.cli.initial_state = "COMPLETED"
        self.cli.states = iter(("RUNNING",))
        reused_provider = self.make_provider()
        reused = submit_cpu(self.ctx, reused_provider.jobs)
        with self.assertRaisesRegex(
            ProviderError, "terminal provider state changed"
        ) as raised:
            reused_provider.jobs.observe(reused.provider_ids[0], window=1000)
        self.assertEqual(
            raised.exception.evidence["provider_job_id"], reused.provider_ids[0]
        )
