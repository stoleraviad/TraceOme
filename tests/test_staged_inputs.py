"""A FASTQ registered as a staged ``s3://`` object: inspected by HEAD, copied in the
provider instead of uploaded, and trusted on its staging manifest's digest."""

import hashlib
import json
import tempfile
import unittest
from collections import deque
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    RegistrationError,
    register_run,
)
from bio_neb.input_inspection.staged_objects import RegisteredInputsAdapter
from bio_neb.input_inspection.tool import (
    InputMetadataInspectionTool,
    InspectInputMetadataInput,
    RegisteredInputPath,
)
from bio_neb.journey_path import JourneyPath
from bio_neb.parabricks.operation import execute_parabricks
from bio_neb.providers.fake import FakeProvider
from tests.parabricks_operation_fakes import input_storage_provider
from tests.proposal_information_fakes import runtime_configuration

STAGED = "s3://bio-neb-evals/inputs/s5-scope/{role}.fastq.gz"
SET_ID = "00000000-0000-4000-8000-000000000001"


def _clock():
    return "2026-10-30T10:00:00Z"


class StagedRegistrationTests(unittest.TestCase):
    """The ticket's additive S3 form (#488 Carried decisions, "Additive only"): a
    registered FASTQ is an absolute path or one exact ``s3://bucket/key``. Staged
    objects are inspected and copied through the Run's object storage, so ``begin``
    refuses them without a Parabricks section (docs/design/journey-path.md, Run
    defaults)."""

    def test_a_staged_object_registers_beside_a_local_path(self):
        staged = RegisterFastqInput(role="read_1", registered_path=STAGED)
        self.assertEqual(staged.registered_path, STAGED)
        for locator in ("s3://bio-neb-evals", "s3://bio-neb-evals/", "relative.fq"):
            with self.subTest(locator=locator), self.assertRaises(ValidationError):
                RegisterFastqInput(role="read_1", registered_path=locator)

    def test_staged_inputs_without_object_storage_are_refused_before_a_journey(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prompt, runtime = root / "prompt.txt", root / "runtime.json"
            prompt.write_text("Analyze the staged pair.", encoding="utf-8")
            configuration = runtime_configuration()
            configuration.pop("parabricks", None)
            runtime.write_text(json.dumps(configuration), encoding="utf-8")
            run_root = root / "run"
            with self.assertRaises(RegistrationError) as refused:
                register_run(
                    RegisterRunCommand(
                        run_root=str(run_root),
                        prompt_file=str(prompt),
                        runtime_config_file=str(runtime),
                        inputs=tuple(
                            RegisterFastqInput(
                                role=role, registered_path=STAGED.format(role=role)
                            )
                            for role in ("read_1", "read_2")
                        ),
                    ),
                    repository_root=root / "elsewhere",
                )
            self.assertEqual(
                refused.exception.code, "staged_inputs_need_object_storage"
            )
            self.assertFalse(run_root.exists())


class StagedInspectionTests(unittest.TestCase):
    """Owner, 2026-10-30, Q2 "whatever is simplest" (readiness.md): a staged object's
    HEAD fills the existing observation, and an absent object is the existing
    ``input_missing`` with no new failure code (docs/design/journey-path.md,
    Supplied input metadata)."""

    def _inspect(self, requests, objects):
        heads = []

        def head(location):
            heads.append(location)
            return objects.get(location)

        adapter = RegisteredInputsAdapter(lambda: head, clock=_clock)
        result = InputMetadataInspectionTool().inspect(
            InspectInputMetadataInput(input_set_id=SET_ID, inputs=requests), adapter
        )
        return result, heads

    def test_a_staged_object_is_observed_by_head_beside_a_stat_local_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary).resolve() / "read_2.fastq.gz"
            local.write_bytes(b"local-reads")
            staged = STAGED.format(role="read_1")
            result, heads = self._inspect(
                (
                    RegisteredInputPath(role="read_1", registered_path=staged),
                    RegisteredInputPath(role="read_2", registered_path=str(local)),
                ),
                {staged: (572201032, "b44bf473-69", 1761818053000000000)},
            )
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(heads, [staged])
        first, second = result.inputs
        self.assertEqual(
            (
                first.role,
                first.size_bytes,
                first.filesystem_identity.device,
                first.filesystem_identity.inode,
                first.modified_at_ns,
            ),
            (
                "read_1",
                572201032,
                "s3://bio-neb-evals",
                "b44bf473-69",
                1761818053000000000,
            ),
        )
        self.assertEqual((second.role, second.size_bytes), ("read_2", 11))

    def test_an_absent_staged_object_fails_inspection_as_missing(self):
        staged = STAGED.format(role="read_1")
        result, _ = self._inspect(
            (RegisteredInputPath(role="read_1", registered_path=staged),), {}
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(
            [(f.code, f.registered_path) for f in result.failures],
            [("input_missing", staged)],
        )

    def test_local_inputs_make_no_storage_contact(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary).resolve() / "read_1.fastq.gz"
            local.write_bytes(b"reads")
            adapter = RegisteredInputsAdapter(
                lambda: self.fail("a local input built a storage client"), clock=_clock
            )
            observation = adapter.observe(
                (RegisteredInputPath(role="read_1", registered_path=str(local)),)
            )
        self.assertEqual(observation.inputs[0].size_bytes, 5)


class StagedPlacementTests(unittest.TestCase):
    """A Run registered, inspected and approved with staged FASTQ objects.

    Owner, 2026-10-30 (readiness.md, Owner answers): the staged object is trusted;
    a manifest sha256 or md5, or none, are all valid. Its copy, digest basis and
    the one size check follow docs/design/parabricks-execution.md (Engine seam).
    Only storage effects are substituted: each staged object's HEAD answers the
    builder's local FASTQ bytes, and the staging bucket is in memory.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _approved_staged_run(self):
        from bio_neb.input_inspection import staged_objects
        from bio_neb.journey import Journey
        from bio_neb.parabricks.plan import approved_execution_input
        from bio_neb.parabricks.planning_commands import (
            RespondExecutionHitlCommand,
            respond_execution_hitl,
        )
        from tests.execution_approval import approve_execution
        from tests.planning_assignment_fakes import prepare_configured_run

        root = Path(self.temp.name).resolve() / "run"
        local = {"read_1": root / "r1.fastq", "read_2": root / "r2.fastq"}

        def head(location):
            role = location.rpartition("/")[2].removesuffix(".fastq.gz")
            return len(local[role].read_bytes()), f"etag-{role}", 1761818053000000000

        with patch.object(staged_objects, "_object_storage_head", return_value=head):
            registration, review, _ = prepare_configured_run(
                root, staged_inputs={role: STAGED.format(role=role) for role in local}
            )
        request = review.execution_hitl.pending_request
        respond_execution_hitl(
            RespondExecutionHitlCommand(
                journey_file=registration.journey_file,
                request_event_id=request.request_event_id,
                response=approve_execution(request.proposal),
            )
        )
        self.file = Path(registration.journey_file)
        return approved_execution_input(Journey.open(self.file)), local

    def _execute(self, *, manifest=None):
        approved, local = self._approved_staged_run()
        self.assertEqual(
            [
                item.registered_path
                for item in approved.proposal.execution_package.inputs
            ],
            [STAGED.format(role=role) for role in ("read_1", "read_2")],
        )
        staging = FakeProvider()
        contacts = []

        def factory(plan, journal):
            for fastq in plan.fastqs:
                key = fastq.registered_path.removeprefix("s3://bio-neb-evals/")
                staging.objects["bio-neb-evals", key] = local[fastq.role].read_bytes()
            if manifest is not None:
                staging.objects[
                    "bio-neb-evals", "inputs/s5-scope/staging-manifest.json"
                ] = json.dumps(manifest(staging)).encode()
            # The fake scripts one receipt per upload from local bytes; no FASTQ is
            # uploaded here, so only a streamed reference source's receipt is kept.
            uploaded = tuple(
                fastq.model_copy(update={"registered_path": str(local[fastq.role])})
                for fastq in plan.fastqs
            )
            provider = input_storage_provider(replace(plan, fastqs=uploaded), journal)
            script = provider.scripts["input_storage", "place_inputs"]
            provider.scripts["input_storage", "place_inputs"] = deque(
                [script[-1]] if plan.input_source is not None else []
            )
            original = provider.for_step

            def for_step(stage, step):
                ports = original(stage, step)
                if (stage, step) == ("input_storage", "place_inputs"):
                    put = ports.store.put

                    def upload(bucket, key, source):
                        self.assertNotIn(".fastq", key, "a staged FASTQ was uploaded")
                        return put(bucket, key, source)

                    def head(bucket, key, *, verification):
                        contacts.append(("read", bucket, key, verification))
                        return staging.store.head(
                            bucket, key, verification=verification
                        )

                    def copy(source, bucket, key):
                        contacts.append(("copy", source.location, bucket, key))
                        return staging.store.copy(source, bucket, key)

                    ports.store.put = upload
                    ports.store.head = head
                    ports.store.copy = copy
                    ports.store.read_small = staging.store.read_small
                return ports

            provider.for_step = for_step
            return provider

        result = execute_parabricks(approved, infrastructure_factory=factory)
        return result, staging, contacts

    def _manifest(self, *, sha256=True, md5=False, sizes=None):
        def build(staging):
            files = []
            for role in ("read_1", "read_2"):
                key = f"inputs/s5-scope/{role}.fastq.gz"
                content = staging.objects["bio-neb-evals", key]
                entry = {
                    "role": role,
                    "object_key": key,
                    "bytes": (sizes or {}).get(role, len(content)),
                    "seconds": 3,
                }
                if sha256:
                    entry["sha256"] = hashlib.sha256(content).hexdigest()
                if md5:
                    entry["md5"] = hashlib.md5(content).hexdigest()
                files.append(entry)
            return {"schema": "bio-neb.eval-staged-inputs/1", "files": files}

        return build

    def _staged_inputs(self):
        return JourneyPath.open(self.file).state.inputs

    def _assert_inputs_verified(self):
        """Input storage verified the inputs; later stages are not this claim's."""
        from bio_neb.journey_path import Verified

        state = JourneyPath.open(self.file).state
        self.assertIsInstance(state.inputs, Verified)
        self.assertEqual(
            [f.detail for f in state.failures if f.stage == "input_storage"], []
        )

    def test_a_listed_digest_is_trusted_and_the_object_copied_without_a_read(self):
        result, staging, contacts = self._execute(
            manifest=self._manifest(sha256=True, md5=True)
        )
        self._assert_inputs_verified()
        self.assertEqual([c[0] for c in contacts], ["copy", "copy"])
        payload = self._staged_inputs()
        bucket = contacts[0][2]
        for basis, contact in zip(payload.staged_inputs, contacts, strict=True):
            content = staging.objects["bio-neb-evals", contact[1].split("/", 3)[3]]
            self.assertEqual(staging.objects[bucket, contact[3]], content)
            self.assertEqual(
                (
                    basis.source_location,
                    basis.manifest_location,
                    basis.sha256,
                    basis.sha256_basis,
                    basis.md5,
                ),
                (
                    contact[1],
                    "s3://bio-neb-evals/inputs/s5-scope/staging-manifest.json",
                    hashlib.sha256(content).hexdigest(),
                    "staging_manifest",
                    hashlib.md5(content).hexdigest(),
                ),
            )
            pointer = next(p for p in payload.pointers if p.role == basis.role)
            self.assertEqual(pointer.content_id, "sha256:" + basis.sha256)
            self.assertEqual(pointer.location, f"s3://{bucket}/{contact[3]}")

    def test_without_a_manifest_the_staged_object_is_read_once_to_hash_it(self):
        result, staging, contacts = self._execute(manifest=None)
        self._assert_inputs_verified()
        self.assertEqual(
            [(c[0], c[-1]) for c in contacts if c[0] == "read"],
            [("read", "independent_full_read")] * 2,
        )
        for basis in self._staged_inputs().staged_inputs:
            key = basis.source_location.removeprefix("s3://bio-neb-evals/")
            self.assertEqual(
                (basis.manifest_location, basis.sha256_basis, basis.md5),
                (None, "source_read", None),
            )
            self.assertEqual(
                basis.sha256,
                hashlib.sha256(staging.objects["bio-neb-evals", key]).hexdigest(),
            )

    def test_a_manifest_without_a_digest_is_kept_and_the_object_read(self):
        result, _staging, contacts = self._execute(
            manifest=self._manifest(sha256=False)
        )
        self._assert_inputs_verified()
        self.assertEqual(sum(c[0] == "read" for c in contacts), 2)
        for basis in self._staged_inputs().staged_inputs:
            self.assertEqual(
                (basis.manifest_location is not None, basis.sha256_basis),
                (True, "source_read"),
            )

    def test_a_staged_size_differing_from_the_inspected_size_is_refused(self):
        result, staging, contacts = self._execute(
            manifest=self._manifest(sizes={"read_1": 1})
        )
        self.assertEqual(result.status, "execution_review_pending")
        self.assertNotIn("copy", [c[0] for c in contacts])
        state = JourneyPath.open(self.file).state
        self.assertEqual(state.failures[-1].detail, "staged_input_size_mismatch")


class RootObjectManifestTests(unittest.TestCase):
    """A staged object at a bucket's root finds the manifest beside it, at the root
    (Spec r1 SP2; the accepted locator is any exact ``s3://bucket/key``)."""

    def test_a_root_object_uses_the_root_manifest_without_a_read(self):
        from bio_neb.parabricks.input_storage import _copy_staged

        content = b"@r\nACGT\n+\n!!!!\n"
        sha256 = hashlib.sha256(content).hexdigest()
        staging = FakeProvider()
        staging.objects["staged-bucket", "read_1.fastq.gz"] = content
        staging.objects["staged-bucket", "staging-manifest.json"] = json.dumps(
            {
                "files": [
                    {
                        "object_key": "read_1.fastq.gz",
                        "bytes": len(content),
                        "sha256": sha256,
                    }
                ]
            }
        ).encode()
        store = staging.store
        store.head = lambda *a, **kw: self.fail("a listed digest was re-read")
        receipt, basis = _copy_staged(
            store,
            "run-bucket",
            "inputs/scope/read_1.fastq.gz",
            "read_1",
            "s3://staged-bucket/read_1.fastq.gz",
            len(content),
        )
        self.assertEqual(
            (basis["manifest_location"], basis["sha256_basis"], basis["sha256"]),
            ("s3://staged-bucket/staging-manifest.json", "staging_manifest", sha256),
        )
        self.assertEqual(
            staging.objects["run-bucket", "inputs/scope/read_1.fastq.gz"], content
        )


if __name__ == "__main__":
    unittest.main()
