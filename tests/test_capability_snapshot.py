from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from bio_neb.journey import canonical_json
from bio_neb.parabricks.capability_policy import (
    CapabilityPolicyImage,
    CapabilityPolicyRecipe,
    CapabilityPolicyResource,
    CapabilityPolicyTool,
    CapabilitySnapshot,
    CapabilitySnapshotAuthoring,
    CapabilitySnapshotError,
    LocalCapabilitySourceInput,
    RetainedCapabilitySourceInput,
    compile_capability_snapshot,
    compile_capability_snapshot_value,
    read_capability_snapshot,
)
from bio_neb.parabricks.execution_runtime import (
    ATTESTED_BWA_VERSION,
    ATTESTED_IMAGE_CHILD,
    ATTESTED_IMAGE_INDEX,
    ATTESTED_PBRUN_VERSION,
    ATTESTED_PROVIDER_IMAGE,
    ATTESTED_SAMTOOLS_VERSION,
    ATTESTED_WORKER_SHA256,
)
from tests.planning_fakes import policy


def _authoring(source: Path, *, reverse: bool = False) -> CapabilitySnapshotAuthoring:
    sources = (
        LocalCapabilitySourceInput(
            kind="local_bytes",
            locator="file:///selected/capability-source.txt",
            retrieved_at="2026-08-12T18:00:00Z",
            path=str(source),
        ),
        RetainedCapabilitySourceInput(
            kind="retained_nonlocal",
            locator="https://docs.neutral.invalid/capability-release",
            retrieved_at="2026-08-12T17:00:00Z",
            content_digest="9" * 64,
        ),
    )
    tools = (
        CapabilityPolicyTool(
            tool_id="bwa",
            execution_scope="worker_image",
            absolute_path="/usr/bin/bwa",
            version=ATTESTED_BWA_VERSION,
            image_id="parabricks-image",
        ),
        CapabilityPolicyTool(
            tool_id="pbrun",
            execution_scope="worker_image",
            absolute_path="/usr/local/parabricks/pbrun",
            version=ATTESTED_PBRUN_VERSION,
            image_id="parabricks-image",
        ),
        CapabilityPolicyTool(
            tool_id="samtools",
            execution_scope="worker_image",
            absolute_path="/usr/bin/samtools",
            version=ATTESTED_SAMTOOLS_VERSION,
            image_id="parabricks-image",
        ),
        CapabilityPolicyTool(
            tool_id="nebius-cli",
            execution_scope="controller",
            absolute_path="/home/bhalu/.nebius/bin/nebius",
            version="0.12.263",
            executable_sha256=(
                "8712903f152120e40ec64e96d242e754496e46dffde3b491a8f1a0d4cf67896d"
            ),
            executable_byte_count=83_259_554,
        ),
    )
    tool_ids = tuple(item.tool_id for item in reversed(tools) if reverse) or tuple(
        item.tool_id for item in tools
    )
    return CapabilitySnapshotAuthoring(
        schema_version="bio-neb.execution-capability-authoring/1",
        capability_version="accepted-parabricks-germline/1",
        sources=tuple(reversed(sources)) if reverse else sources,
        recipes=(
            CapabilityPolicyRecipe(
                recipe_id="germline-approved",
                assay="gDNA",
                read_layout="paired",
                output_kind="VCF",
                requires_prepared_reference=True,
                requires_companion_roles=("genomic_fasta",),
            ),
        ),
        resources=(
            CapabilityPolicyResource(
                profile_id="h100-approved",
                recipe_id="germline-approved",
                image_id="parabricks-image",
                tool_ids=tool_ids,
                accelerator_count=1,
                cpu_count=16,
                memory_gib=200,
                accelerator_memory_gib=80,
            ),
        ),
        images=(
            CapabilityPolicyImage(
                image_id="parabricks-image",
                provider_image=ATTESTED_PROVIDER_IMAGE,
                image_index_digest=ATTESTED_IMAGE_INDEX,
                platform="linux/amd64",
                platform_child_digest=ATTESTED_IMAGE_CHILD,
                worker_sha256=ATTESTED_WORKER_SHA256,
                worker_module=("bio_neb.parabricks.reusable_object_storage_parabricks"),
                worker_request_schema="bio-neb.parabricks-worker-request/7",
                worker_receipt_schema="bio-neb.parabricks-worker-receipt/3",
                worker_entrypoint=(
                    "/opt/bio-neb/venv/bin/python",
                    "/opt/bio-neb/reusable_object_storage_parabricks.py",
                ),
            ),
        ),
        tools=tuple(reversed(tools)) if reverse else tools,
    )


class CapabilitySnapshotCompilerTests(unittest.TestCase):
    def test_snapshot_identity_covers_exact_canonical_capability_claims(self):
        retained = policy()
        changed_recipe = retained.recipes[0].model_copy(update={"assay": "RNA"})
        with self.assertRaisesRegex(ValidationError, "canonical capability claims"):
            CapabilitySnapshot(
                **{
                    **retained.model_dump(mode="python"),
                    "recipes": (changed_recipe,),
                }
            )

    def test_retained_source_locators_refuse_credentials(self):
        retained = policy()
        with self.assertRaisesRegex(ValidationError, "credentials"):
            CapabilitySnapshot.model_validate(
                {
                    **retained.model_dump(mode="python"),
                    "source_manifest": (
                        {
                            **retained.source_manifest[0].model_dump(mode="python"),
                            "locator": "https://user:secret@example.test/policy?token=x",
                        },
                    ),
                }
            )

    def test_equivalent_authoring_is_byte_identical_and_local_paths_are_not_retained(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.txt"
            source.write_bytes("selected π capability bytes".encode())
            first = root / "first.json"
            second = root / "second.json"
            first_input = root / "first-authoring.json"
            second_input = root / "second-authoring.json"
            first_input.write_text(
                canonical_json(_authoring(source).model_dump(mode="json")),
                encoding="utf-8",
            )
            second_input.write_text(
                canonical_json(
                    _authoring(source, reverse=True).model_dump(mode="json")
                ),
                encoding="utf-8",
            )

            first_audit = compile_capability_snapshot(first_input, first)
            second_audit = compile_capability_snapshot(second_input, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_audit, second_audit)
            self.assertNotIn(str(source).encode(), first.read_bytes())
            self.assertEqual(
                next(
                    item.content_assurance
                    for item in first_audit.source_manifest
                    if item.locator.startswith("file:")
                ),
                "application_hashed_local_bytes",
            )
            self.assertEqual(
                read_capability_snapshot(first).snapshot_id, first_audit.snapshot_id
            )

    def test_each_claim_family_changes_the_derived_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.txt"
            source.write_bytes(b"source one")
            original = _authoring(source)
            original_id = compile_capability_snapshot_value(original).snapshot_id
            changed = (
                original.model_copy(
                    update={
                        "recipes": (
                            original.recipes[0].model_copy(update={"assay": "RNA"}),
                        )
                    }
                ),
                original.model_copy(
                    update={
                        "resources": (
                            original.resources[0].model_copy(
                                update={"memory_gib": 201}
                            ),
                        )
                    }
                ),
                original.model_copy(
                    update={
                        "images": (
                            original.images[0].model_copy(
                                update={"worker_sha256": "8" * 64}
                            ),
                        )
                    }
                ),
                original.model_copy(
                    update={
                        "tools": tuple(
                            item.model_copy(update={"version": "pbrun: changed"})
                            if item.tool_id == "pbrun"
                            else item
                            for item in original.tools
                        )
                    }
                ),
            )
            self.assertEqual(
                len(
                    {
                        compile_capability_snapshot_value(item).snapshot_id
                        for item in changed
                    }
                ),
                len(changed),
            )
            self.assertTrue(
                all(
                    compile_capability_snapshot_value(item).snapshot_id != original_id
                    for item in changed
                )
            )
            source.write_bytes(b"source two")
            self.assertNotEqual(
                compile_capability_snapshot_value(original).snapshot_id, original_id
            )
            image = original.images[0]
            pin_baseline_id = compile_capability_snapshot_value(original).snapshot_id
            pin_changes = {
                "worker_module": {"worker_module": "neutral.changed.worker"},
                "worker_request_schema": {
                    "worker_request_schema": (
                        "bio-neb.parabricks-worker-request/changed"
                    )
                },
                "worker_receipt_schema": {
                    "worker_receipt_schema": (
                        "bio-neb.parabricks-worker-receipt/changed"
                    )
                },
                "worker_entrypoint": {
                    "worker_entrypoint": (
                        "/opt/bio-neb/venv/bin/python",
                        "/opt/bio-neb/changed_worker.py",
                    )
                },
            }
            changed_ids = set()
            for name, change in pin_changes.items():
                with self.subTest(worker_release_pin=name):
                    changed_id = compile_capability_snapshot_value(
                        original.model_copy(
                            update={"images": (image.model_copy(update=change),)}
                        )
                    ).snapshot_id
                    self.assertNotEqual(changed_id, pin_baseline_id)
                    changed_ids.add(changed_id)
            self.assertEqual(len(changed_ids), len(pin_changes))

    def test_snapshot_refuses_missing_cross_reference_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.txt"
            source.write_bytes(b"source")
            authoring = _authoring(source)
            with self.assertRaisesRegex(ValueError, "recipe identity does not exist"):
                compile_capability_snapshot_value(
                    authoring.model_copy(
                        update={
                            "resources": (
                                authoring.resources[0].model_copy(
                                    update={"recipe_id": "missing-recipe"}
                                ),
                            )
                        }
                    )
                )
            with self.assertRaisesRegex(ValueError, "tool identities must be unique"):
                compile_capability_snapshot_value(
                    authoring.model_copy(
                        update={"tools": authoring.tools + (authoring.tools[0],)}
                    )
                )

    def test_local_source_locator_refuses_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.txt"
            source.write_bytes(b"source")
            with self.assertRaises(ValidationError):
                LocalCapabilitySourceInput(
                    kind="local_bytes",
                    locator="https://docs.invalid/source?token=secret",
                    retrieved_at="2026-08-12T18:00:00Z",
                    path=str(source),
                )

    def test_worker_tool_paths_must_be_absolute(self):
        with self.assertRaises(ValidationError):
            CapabilityPolicyTool(
                tool_id="bad",
                execution_scope="worker_image",
                absolute_path="relative/tool",
                version="1",
                image_id="parabricks-image",
            )

    def test_worker_release_requires_complete_schema_and_absolute_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.txt"
            source.write_bytes(b"source")
            authoring = _authoring(source)
            incomplete_image = authoring.images[0].model_dump(mode="json")
            incomplete_image.pop("worker_receipt_schema")
            with self.assertRaises(ValidationError):
                CapabilityPolicyImage.model_validate(incomplete_image)
            with self.assertRaises(ValidationError):
                CapabilityPolicyImage.model_validate(
                    {
                        **authoring.images[0].model_dump(mode="json"),
                        "worker_entrypoint": ["python", "/opt/bio-neb/worker.py"],
                    }
                )

    def test_compile_target_collision_refuses_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.txt"
            source.write_bytes(b"source")
            authoring_path = root / "authoring.json"
            target = root / "snapshot.json"
            authoring = _authoring(source)
            authoring_path.write_text(
                canonical_json(authoring.model_dump(mode="json")), encoding="utf-8"
            )
            compile_capability_snapshot(authoring_path, target)
            retained = target.read_bytes()
            with self.assertRaisesRegex(CapabilitySnapshotError, "target_exists"):
                compile_capability_snapshot(authoring_path, target)
            self.assertEqual(target.read_bytes(), retained)

    def test_read_snapshot_requires_canonical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.txt"
            source.write_bytes(b"source")
            authoring_path = root / "authoring.json"
            target = root / "snapshot.json"
            authoring = _authoring(source)
            authoring_path.write_text(
                canonical_json(authoring.model_dump(mode="json")), encoding="utf-8"
            )
            compile_capability_snapshot(authoring_path, target)
            retained = target.read_bytes()
            target.write_bytes(retained + b"\n")
            with self.assertRaisesRegex(CapabilitySnapshotError, "not_canonical"):
                read_capability_snapshot(target)

    def test_compile_refuses_malformed_authoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            authoring_path = root / "authoring.json"
            authoring_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(CapabilitySnapshotError, "authoring_invalid"):
                compile_capability_snapshot(authoring_path, root / "malformed.json")

    def test_authoring_refuses_supplied_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.txt"
            source.write_bytes(b"source")
            authoring = _authoring(source)
            invalid = authoring.model_dump(mode="json")
            invalid["snapshot_id"] = "0" * 64
            with self.assertRaises(ValidationError):
                CapabilitySnapshotAuthoring.model_validate(invalid)

    def test_authoring_refuses_scientific_evidence_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.txt"
            source.write_bytes(b"source")
            authoring = _authoring(source)
            scientific = authoring.model_dump(mode="json")
            scientific["retrieved_title"] = "scientific evidence is a separate model"
            with self.assertRaises(ValidationError):
                CapabilitySnapshotAuthoring.model_validate(scientific)

    def test_compile_refuses_unavailable_local_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "source.txt"
            source.write_bytes(b"source")
            authoring = _authoring(source)
            source.unlink()
            with self.assertRaisesRegex(CapabilitySnapshotError, "source_unavailable"):
                compile_capability_snapshot_value(authoring)


if __name__ == "__main__":
    unittest.main()
