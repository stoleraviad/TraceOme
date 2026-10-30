"""Contact-free composition proof from the end-of-dev-cleanup Run's approved input."""

import unittest

from bio_neb.journey_path import ExecutionBound, ExecutionState
from bio_neb.parabricks.plan import (
    ApprovedParabricksInput,
    ExecutionCompositionRefused,
    RetainedExecution,
    plan_execution,
)
from tests.reference_junction_fakes import approved_with_runtimes
from tests.retained_execution_path import SUPPLIED_SOURCE_MD5, approved_input


def reviewed(approved, *, launch=None, runtime_selection=None):
    """Synthetic: a changed launch or runtime as if reviewed, identities not recomputed.

    The reviewed configuration changes with it, so composition reaches the changed
    setting's own gate rather than refusing the unreviewed change first.
    """
    launch = launch or approved.launch
    runtime_selection = runtime_selection or approved.proposal.runtime_selection
    proposal = approved.proposal.model_copy(
        update={
            "runtime_selection": runtime_selection,
            "configuration": approved.proposal.configuration.model_copy(
                update={"launch": launch, "runtime_selection": runtime_selection}
            ),
        }
    )
    return approved.model_copy(update={"launch": launch, "proposal": proposal})


def retained(approved, execution=None):
    return RetainedExecution(
        execution or ExecutionState(), approved.authority, approved.proposal
    )


class ExecutionPlanTests(unittest.TestCase):
    def test_launch_without_a_mode_resolves_every_slot_to_creation_proof_with_a_stable_policy_id(
        self,
    ):
        """Ordinary execution retains creation proof without verification contacts."""

        from bio_neb.application.verification_policy import (
            resolve_object_verification_policy,
        )
        from bio_neb.journey import canonical_json_digest
        from bio_neb.parabricks.plan import ParabricksLaunch

        approved = approved_input()
        raw = approved.launch.model_dump(mode="json")
        # The Run recorded the default explicitly; omitting it must mean the same.
        self.assertEqual(raw["verification_mode"], "reuse_creation_proof")
        del raw["verification_mode"]
        del raw["verification_overrides"]
        launch = ParabricksLaunch.model_validate(raw)
        self.assertEqual(launch, approved.launch)
        self.assertEqual(launch.verification_mode, "reuse_creation_proof")
        self.assertEqual(launch.verification_overrides.model_dump(), {})
        policy = resolve_object_verification_policy(
            launch.verification_mode, launch.verification_overrides
        )
        self.assertEqual(
            policy.effective.model_dump(),
            {
                "input_storage": "reuse_creation_proof",
                "prepared_candidate": "reuse_creation_proof",
                "reference_preparation": "reuse_creation_proof",
                "h100_outputs": "reuse_creation_proof",
            },
        )
        again = resolve_object_verification_policy(
            launch.verification_mode, launch.verification_overrides
        )
        self.assertEqual(policy.policy_id, again.policy_id)
        self.assertEqual(
            policy.policy_id,
            "object-verification-policy:"
            + canonical_json_digest(
                policy.model_dump(mode="json", exclude={"policy_id"})
            ),
        )
        self.assertEqual(policy.model_dump(mode="json")["overrides"], {})

    def test_capability_only_gpu_amendment_does_not_add_automatic_fallbacks(self):
        approved = approved_with_runtimes(profile_ids=("alternate",))
        plan = plan_execution(
            approved,
            RetainedExecution(
                ExecutionState(amendments={"gpu_class": "fallback"}),
                approved.authority,
                approved.proposal,
            ),
        )
        self.assertEqual(plan.runtimes.h100.platform, "gpu-fallback")
        self.assertEqual(plan.runtimes.fallbacks, ())

    def test_gpu_amendment_preserves_the_remaining_approved_fallback_order(self):
        approved = approved_with_runtimes()
        plan = plan_execution(
            approved,
            RetainedExecution(
                ExecutionState(amendments={"gpu_class": "alternate"}),
                approved.authority,
                approved.proposal,
            ),
        )
        self.assertEqual(plan.runtimes.h100.platform, "gpu-alternative")
        self.assertEqual(
            tuple(runtime.platform for runtime in plan.runtimes.fallbacks),
            ("gpu-fallback",),
        )

    def test_human_gpu_amendment_uses_the_capability_profile_without_changing_identity(
        self,
    ):
        approved = approved_with_runtimes()
        retained = RetainedExecution(
            ExecutionState(), approved.authority, approved.proposal
        )
        original = plan_execution(approved, retained)
        amended = ExecutionState(amendments={"gpu_class": "fallback"})
        result = plan_execution(
            approved, RetainedExecution(amended, approved.authority, approved.proposal)
        )
        self.assertEqual(result.runtimes.h100.platform, "gpu-fallback")
        self.assertEqual(result.identity, original.identity)
        with self.assertRaisesRegex(ExecutionCompositionRefused, "lineage"):
            plan_execution(
                approved,
                RetainedExecution(
                    ExecutionState(amendments={"gpu_class": "unapproved"}),
                    approved.authority,
                    approved.proposal,
                ),
            )

    def test_a_location_amendment_is_an_undeclared_amendment(self):
        """Owner decision 2026-10-29 (#464): the location amendment is gone, so a
        retained one composes no plan; only the GPU class is amendable."""
        approved = approved_with_runtimes()
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(
                approved,
                RetainedExecution(
                    ExecutionState(
                        amendments={"location": {"bucket": "prepared"}},
                        amendment_event_ids={"location": "answer"},
                    ),
                    approved.authority,
                    approved.proposal,
                ),
            )
        self.assertEqual(refusal.exception.details, ("undeclared_amendment",))
        plan = plan_execution(
            approved,
            RetainedExecution(ExecutionState(), approved.authority, approved.proposal),
        )
        self.assertEqual(plan.amendable, frozenset({"gpu_class"}))

    def test_malformed_manual_gpu_choice_is_refused_locally(self):
        approved = approved_with_runtimes()
        for value in (
            "",
            None,
            False,
            0,
            [],
            {},
            ["alternate"],
            {"profile": "alternate"},
            1,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ExecutionCompositionRefused) as refusal:
                    plan_execution(
                        approved,
                        RetainedExecution(
                            ExecutionState(amendments={"gpu_class": value}),
                            approved.authority,
                            approved.proposal,
                        ),
                    )
                self.assertEqual(
                    refusal.exception.details, ("invalid_gpu_class_amendment",)
                )

    def test_manual_gpu_choice_cannot_change_cpu_preparation(self):
        approved = approved_with_runtimes(
            profile_ids=("alternate",), fallback_cpu_disk="200Gi"
        )
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(
                approved,
                RetainedExecution(
                    ExecutionState(amendments={"gpu_class": "fallback"}),
                    approved.authority,
                    approved.proposal,
                ),
            )
        self.assertEqual(refusal.exception.details, ("invalid_gpu_class_amendment",))

    def test_capability_only_choice_has_no_automatic_substitution_authority(self):
        approved = approved_with_runtimes(profile_ids=("alternate",))
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(
                approved,
                RetainedExecution(
                    ExecutionState(runtime_profile_id="fallback"),
                    approved.authority,
                    approved.proposal,
                ),
            )
        self.assertEqual(
            refusal.exception.details, ("unapproved_runtime_substitution",)
        )

    def test_launch_and_plan_policies_refuse_an_unbounded_heartbeat(self):
        from pydantic import ValidationError

        from bio_neb.parabricks.plan import ParabricksLaunch, Policies

        launch = approved_input().launch.model_dump()
        for interval in (float("inf"), float("nan"), -1, 0):
            with self.subTest(interval=interval):
                with self.assertRaises(ValidationError):
                    ParabricksLaunch.model_validate(
                        {**launch, "heartbeat_interval_seconds": interval}
                    )
                with self.assertRaises(ValidationError):
                    Policies(heartbeat_interval_seconds=interval)
        self.assertEqual(
            Policies(heartbeat_interval_seconds=0.25).heartbeat_interval_seconds, 0.25
        )

    def test_admitted_source_cannot_alias_the_reserved_source_manifest(self):
        from bio_neb.journey import canonical_json_digest
        from bio_neb.parabricks.execution_package import ExecutionPackage
        from bio_neb.parabricks.execution_proposal import (
            ExecutionAuthority,
            ExecutionProposal,
            execution_authority_id,
        )
        from bio_neb.reference_discovery.reference_source import (
            reference_source_admission_id,
        )

        original = approved_input()
        # Recompile synthetic approval identities so this reaches the layout gate,
        # not an upstream source/approval validation refusal.
        for basename in ("source.fna.gz", "manifest.json"):
            with self.subTest(basename=basename):
                proposal = original.proposal.model_dump(
                    mode="json", exclude={"proposal_id"}
                )
                locator = "https://tests.invalid/reference/" + basename
                admission = proposal["reference_source_admission"]
                admission.pop("admission_id")
                admission["source_locator"] = locator
                admission["admission_id"] = reference_source_admission_id(admission)
                package = proposal["execution_package"]
                package.pop("package_id")
                package["reference"]["source_locator"] = locator
                package["reference"]["source_admission_id"] = admission["admission_id"]
                package = ExecutionPackage(
                    package_id="execution-package:" + canonical_json_digest(package),
                    **package,
                )
                proposal["execution_package"] = package.model_dump(mode="json")
                proposal = ExecutionProposal(
                    proposal_id="execution-proposal:" + canonical_json_digest(proposal),
                    **proposal,
                )
                authority = original.authority.model_dump(
                    exclude={"proposal_id", "authority_id"}
                )
                authority = ExecutionAuthority(
                    **authority,
                    proposal_id=proposal.proposal_id,
                    authority_id=execution_authority_id(
                        proposal.proposal_id,
                        authority["request_event_id"],
                        authority["approval_event_id"],
                    ),
                )
                approved = ApprovedParabricksInput(
                    journey_file=original.journey_file,
                    launch=original.launch,
                    proposal=proposal,
                    authority=authority,
                )
                retained = RetainedExecution(ExecutionState(), authority, proposal)
                if basename == "manifest.json":
                    with self.assertRaises(ExecutionCompositionRefused) as refusal:
                        plan_execution(approved, retained)
                    self.assertEqual(refusal.exception.code, "lineage")
                    self.assertEqual(
                        refusal.exception.details,
                        ("reference_source_layout_collision",),
                    )
                else:
                    plan = plan_execution(approved, retained)
                    self.assertTrue(
                        plan.layout.source_object_key.endswith("/source.fna.gz")
                    )
                    self.assertTrue(
                        plan.layout.source_manifest_object_key.endswith(
                            "/manifest.json"
                        )
                    )
                    self.assertNotEqual(
                        plan.layout.source_object_key,
                        plan.layout.source_manifest_object_key,
                    )

    def test_local_reference_replica_selects_approved_bytes_without_url_contact(self):
        # The Run's reviewed launch selected a local replica of the approved public
        # reference; composition carries the approved source identity (size and
        # publisher md5, checked while streaming), not a URL.
        approved = approved_input()
        plan = plan_execution(approved, retained(approved))
        self.assertEqual(plan.input_source.kind, "local")
        self.assertEqual(
            plan.input_source.locator,
            "/workspaces/bio-neb-work-issue29-retry/.bio-neb-local/fixtures/"
            "bio-neb-t11-h100-20260917-a018d029-a2/raw/t11/h100/a018d029/a2/"
            "GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
        )
        self.assertEqual(
            (
                plan.input_source.byte_count,
                plan.input_source.sha256,
                plan.input_source.md5,
                plan.input_source.measure_gzip_content,
            ),
            (972898531, None, SUPPLIED_SOURCE_MD5, True),
        )

    def test_the_packages_excluded_sequences_reach_the_measure_and_the_cpu_build(self):
        """Owner, #464 ("primey for the ref we are running"): the names the package
        drops reach input storage's measure and the CPU worker's request, and name a
        bundle of their own."""
        import json

        from bio_neb.journey_path import ReferenceAcquisition
        from bio_neb.parabricks import reference_recipe

        excluded = ("NT_187515.1", "NW_025791820.1")
        approved = approved_input(excluded_sequences=excluded)
        plan = plan_execution(approved, retained(approved))
        whole = approved_input()
        whole_plan = plan_execution(whole, retained(whole))

        self.assertEqual(plan.reference.excluded_sequences, excluded)
        self.assertEqual(plan.input_source.excluded_sequences, excluded)
        self.assertEqual(whole_plan.input_source.excluded_sequences, ())
        self.assertNotEqual(
            plan.layout.reference_bundle_id, whole_plan.layout.reference_bundle_id
        )
        acquisition = ReferenceAcquisition(
            source_md5=SUPPLIED_SOURCE_MD5,
            source_byte_count=972898531,
            source_sha256="a" * 64,
            plain_fasta_byte_count=3099750718,
            plain_fasta_sha256="b" * 64,
        )
        for cpu in (
            reference_recipe.compile_cpu_request(plan, acquisition),
            reference_recipe.compile_cpu_request(plan, None),
        ):
            with self.subTest(source=cpu.source.mode):
                wire = json.loads(
                    reference_recipe._compile_worker_request(cpu).request_bytes
                )
                self.assertEqual(
                    wire["reference"]["excluded_sequences"], list(excluded)
                )

    def test_without_a_supplied_fasta_or_bound_index_the_controller_streams_no_source(
        self,
    ):
        approved = approved_input()
        reference = approved.proposal.execution_package.reference
        neither = reviewed(
            approved,
            launch=approved.launch.model_copy(update={"local_reference_source": None}),
        )
        plan = plan_execution(neither, retained(neither))
        # The CPU preparation job fetches the approved URL itself.
        self.assertIsNone(plan.input_source)
        self.assertEqual(
            (
                plan.reference.source_url,
                plan.reference.expected_size_bytes,
                plan.reference.expected_md5,
            ),
            (
                reference.source_locator,
                reference.expected_size_bytes,
                reference.expected_checksum,
            ),
        )

    def test_local_reference_replica_refuses_relative_path(self):
        from bio_neb.parabricks.plan import LocalReferenceSource

        with self.assertRaisesRegex(ValueError, "absolute path"):
            LocalReferenceSource(path="reference.gz")

    def test_snapshot_approved_runtime_override_reaches_plan(self):
        changed = approved_with_runtimes()
        plan = plan_execution(
            changed,
            RetainedExecution(ExecutionState(), changed.authority, changed.proposal),
        )
        self.assertEqual(
            (plan.runtimes.h100.platform, plan.runtimes.h100.preset),
            ("gpu-alternative", "alternative-preset"),
        )
        self.assertEqual(
            tuple(item.platform for item in plan.runtimes.fallbacks), ("gpu-fallback",)
        )
        self.assertNotEqual(
            changed.proposal.proposal_id, approved_input().proposal.proposal_id
        )

    def test_missing_capability_selection_refuses_in_lineage_family(self):
        approved = approved_with_runtimes()
        selection = approved.proposal.runtime_selection.model_copy(
            update={"profile_ids": ("missing",)}
        )
        changed = reviewed(approved, runtime_selection=selection)
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(changed, retained(changed))
        self.assertEqual(
            (refusal.exception.code, refusal.exception.details),
            ("lineage", ("runtime_capability",)),
        )

    def test_capability_resource_drift_refuses_in_lineage_family(self):
        approved = approved_with_runtimes()
        selection = approved.proposal.runtime_selection
        snapshot = selection.capability
        drifted = snapshot.runtime_profiles[0].model_copy(
            update={
                "h100": snapshot.runtime_profiles[0].h100.model_copy(
                    update={"gpu_count": 8}
                )
            }
        )
        selection = selection.model_copy(
            update={
                "capability": snapshot.model_copy(
                    update={
                        "runtime_profiles": (drifted, *snapshot.runtime_profiles[1:])
                    }
                )
            }
        )
        changed = reviewed(approved, runtime_selection=selection)
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(changed, retained(changed))
        self.assertEqual(
            (refusal.exception.code, refusal.exception.details),
            ("lineage", ("runtime_capability",)),
        )

    def test_plain_runtime_types_admit_alternatives(self):
        from bio_neb.parabricks.execution_runtime import (
            CpuReferenceRuntime,
            H100Runtime,
        )

        cpu = CpuReferenceRuntime(
            platform="cpu-next", preset="8vcpu-32gb", disk_size="200Gi"
        )
        gpu = H100Runtime(
            recipe_id="recipe",
            resource_profile_id="alternative",
            platform="gpu-b300",
            preset="1gpu-32vcpu-400gb",
            gpu_count=2,
            cpu_count=32,
            memory_gib=400,
            accelerator_memory_gib=160,
        )
        self.assertEqual(
            (cpu.platform, gpu.platform, gpu.gpu_count), ("cpu-next", "gpu-b300", 2)
        )

    def test_fastqs_require_one_complete_pair(self):
        approved = approved_input()
        package = approved.proposal.execution_package
        changed_proposal = approved.proposal.model_copy(
            update={
                "execution_package": package.model_copy(
                    update={"inputs": package.inputs[:1]}
                )
            }
        )
        changed = approved.model_copy(update={"proposal": changed_proposal})
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(
                changed,
                RetainedExecution(
                    ExecutionState(), approved.authority, changed_proposal
                ),
            )
        self.assertEqual(refusal.exception.code, "fastq_roles")

    def test_invalid_verification_policy_is_a_typed_refusal(self):
        approved = approved_input()
        changed = approved.model_copy(
            update={
                "launch": approved.launch.model_copy(
                    update={"verification_mode": "invalid"}
                )
            }
        )
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(
                changed,
                RetainedExecution(
                    ExecutionState(), approved.authority, approved.proposal
                ),
            )
        self.assertEqual(refusal.exception.code, "verification_policy_invalid")

    def test_retained_bound_facts_cannot_drift(self):
        approved = approved_input()
        bound = ExecutionBound(
            execution_id=approved.execution_id,
            authority_id=approved.authority.authority_id,
            proposal_id=approved.proposal.proposal_id,
            run_id=approved.proposal.journey_id,
            approved_input=approved.model_dump(mode="json"),
        )
        changed = approved.model_copy(
            update={
                "launch": approved.launch.model_copy(
                    update={"object_scope": "changed/scope"}
                )
            }
        )
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(
                changed,
                RetainedExecution(
                    ExecutionState(bound=bound), approved.authority, approved.proposal
                ),
            )
        self.assertEqual(
            (refusal.exception.code, refusal.exception.details),
            ("lineage", ("approved_facts_drift",)),
        )

    def test_selected_nebius_context_cannot_drift_from_scope(self):
        approved = approved_input()
        launch = approved.launch.model_copy(
            update={
                "nebius": approved.launch.nebius.model_copy(
                    update={"profile": "another-profile"}
                )
            }
        )
        changed = reviewed(approved, launch=launch)
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(changed, retained(changed))
        self.assertEqual(
            (refusal.exception.code, refusal.exception.details),
            ("nebius_context_drift", ("selected_context",)),
        )

    def test_authority_must_be_current(self):
        approved = approved_input()
        with self.assertRaises(ExecutionCompositionRefused) as refusal:
            plan_execution(approved, RetainedExecution(ExecutionState(), None, None))
        self.assertEqual(
            (refusal.exception.code, refusal.exception.details),
            ("lineage", ("authority_not_current",)),
        )

    def test_retained_approved_input_yields_the_runs_plan(self):
        # Expected values are the Run's recorded bucket and object keys, not a second
        # composition. The proposal is derived to the source-identity shape, so its
        # identities are no longer the Run's recorded ones.
        approved = approved_input()
        plan = plan_execution(approved, retained(approved))
        self.assertNotEqual(
            plan.identity.proposal_id,
            "execution-proposal:c95cf7862285df7fa8db99774c6d8466188e3b20a1f1ff494af53b02135ac173",
        )
        self.assertEqual(plan.storage.bucket, "bio-neb-s266-20261013-a1")
        self.assertEqual(
            tuple(fastq.role for fastq in plan.fastqs), ("read_1", "read_2")
        )
        self.assertEqual(
            plan.layout.source_object_key,
            "raw/issue266/20261013-a1/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
        )
        self.assertEqual(
            plan.layout.h100_manifest_object_key,
            "execution/issue266/20261013-a1/manifest.json",
        )
        self.assertEqual(plan.runtimes.cpu.platform, "cpu-d3")
        self.assertEqual(plan.runtimes.h100.platform, "gpu-h100-sxm")
        self.assertEqual(
            plan.verification.effective.h100_outputs,
            "reuse_creation_proof",
        )

    def test_full_read_overrides_require_a_reason_and_keep_selected_slots(self):
        from pydantic import ValidationError

        from bio_neb.application.verification_policy import ObjectVerificationOverrides

        with self.assertRaises(ValidationError):
            ObjectVerificationOverrides(h100_outputs="independent_full_read")
        with self.assertRaises(ValidationError):
            ObjectVerificationOverrides(
                h100_outputs="independent_full_read", reason="   "
            )
        with self.assertRaises(ValidationError):
            ObjectVerificationOverrides(
                h100_outputs="reuse_creation_proof",
                reason="no full read is selected",
            )
        overrides = ObjectVerificationOverrides(
            input_storage="independent_full_read",
            h100_outputs="independent_full_read",
            reason="issue 43 offline proof: audit the controller's and the worker's uploads",
        )
        self.assertEqual(
            overrides.model_dump(),
            {
                "input_storage": "independent_full_read",
                "h100_outputs": "independent_full_read",
                "reason": overrides.reason,
            },
        )
        self.assertEqual(
            ObjectVerificationOverrides.model_validate(overrides.model_dump()),
            overrides,
        )
