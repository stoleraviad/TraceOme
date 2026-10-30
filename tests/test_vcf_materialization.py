from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bio_neb.application.artifact_materialization import (
    MaterializationFiles,
)
from bio_neb.cli import main
from bio_neb.failure import ProviderError
from bio_neb.investigation.bam_materialization import (
    BamMaterializationRefused,
    BamMaterializedFile,
)
from bio_neb.investigation.vcf_materialization import (
    VcfMaterializationFile,
    VcfMaterializationReceipt,
    VcfMaterializationRefused,
)
from bio_neb.journey import canonical_json
from bio_neb.journey_path import JourneyPath
from bio_neb.parabricks.operation import execute_parabricks
from bio_neb.parabricks.result import read_completed_parabricks_result
from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt
from bio_neb.verified_artifacts import VerifiedObject
from tests.content_identity_vectors import assert_json_identity
from tests.parabricks_operation_fakes import approved_input, input_storage_provider
from tests.run_prerequisites import RunPrerequisites, prerequisite

VCF_BYTES = b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
BAM_BYTES = b"bounded synthetic BAM acquisition bytes"


def measured_rebinding(
    argv: list[str],
    *,
    observation: bool = False,
    sensitive: bool = False,
    fetch: bool = False,
) -> dict:
    """Child-process OS boundary meter, independent of production accounting."""
    reads: dict[str, int] = {}
    stats: dict[str, int] = {}
    hashed = 0
    original_fdopen = os.fdopen
    original_fstat, original_lstat, original_stat = os.fstat, os.lstat, os.stat

    def counted(path) -> None:
        path = str(path)
        if path.endswith("/variants.vcf"):
            stats[path] = stats.get(path, 0) + 1

    def fstat(descriptor):
        counted(os.readlink(f"/proc/self/fd/{descriptor}"))
        return original_fstat(descriptor)

    def lstat(path, *args, **kwargs):
        counted(path)
        return original_lstat(path, *args, **kwargs)

    def stat_(path, *args, **kwargs):
        counted(path)
        return original_stat(path, *args, **kwargs)

    original_sha256 = hashlib.sha256

    class ReadMeter:
        def __init__(self, stream, path):
            self.stream, self.path = stream, path

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, *args):
            data = self.stream.read(*args)
            reads[self.path] = reads.get(self.path, 0) + len(data)
            return data

    def fdopen(descriptor, *args, **kwargs):
        path = os.readlink(f"/proc/self/fd/{descriptor}")
        stream = original_fdopen(descriptor, *args, **kwargs)
        return ReadMeter(stream, path) if path.endswith("/variants.vcf") else stream

    class HashMeter:
        def __init__(self, *args, **kwargs):
            self.digest = original_sha256(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.digest, name)

        def update(self, chunk):
            nonlocal hashed
            if chunk == VCF_BYTES:
                hashed += len(chunk)
            self.digest.update(chunk)

    def forbidden(_packet):
        raise AssertionError("local rebinding attempted provider contact")

    with (
        patch.object(os, "fdopen", fdopen),
        patch.object(hashlib, "sha256", HashMeter),
        patch.object(os, "fstat", fstat),
        patch.object(os, "lstat", lstat),
        patch.object(os, "stat", stat_),
    ):
        if observation:
            from bio_neb.application.local_file_observation import (
                ObservationFiles,
            )
            from bio_neb.application.verification_policy import (
                ArtifactVerificationDefaults,
                ArtifactVerificationOverrides,
                resolve_artifact_verification_policy,
            )
            from bio_neb.investigation.investigation_observation_adapter import (
                DirectInvestigationObservationEvidence,
            )
            from bio_neb.investigation.local_bcftools import (
                LocalBcftoolsAdapter,
            )

            policy = resolve_artifact_verification_policy(
                ArtifactVerificationDefaults(),
                ArtifactVerificationOverrides(
                    mode="independent_full_read", reason="explicit observation audit"
                )
                if sensitive
                else ArtifactVerificationOverrides(),
            )
            with ObservationFiles(verification_policy=policy) as files:
                DirectInvestigationObservationEvidence(files=files).materialization(
                    receipt_file=argv[0],
                    vcf_file=argv[1],
                )
                # Preparation trusts the replica and reads none of it; only the
                # selected audit inspects it, once per session (0029).
                first = None
                if files.verification_policy.effective.mode == "independent_full_read":
                    adapter = LocalBcftoolsAdapter(files=files)
                    first = adapter.inspect_file(argv[1])
                    assert adapter.inspect_file(argv[1]) == first
                code, stdout, stderr = (
                    0,
                    json.dumps(
                        {
                            "observation": None
                            if first is None
                            else first.model_dump(mode="json")
                        }
                    ),
                    "",
                )
        else:
            code, stdout, stderr = _run(
                argv, source_factory=(lambda _packet: _OneGet()) if fetch else forbidden
            )
    return {
        "pid": os.getpid(),
        "code": code,
        "stdout": stdout,
        "stderr": stderr,
        "reads": reads,
        "stats": stats,
        "hashed_bytes": hashed,
    }


def _credential_receipt() -> ObjectStorageCredentialReuseReceipt:
    return ObjectStorageCredentialReuseReceipt(
        service_account_id="service-account-exact",
        access_key_resource_id="access-key-resource-exact",
        access_key_id="aws-access-key-exact",
        exact_topology_match_count=1,
        access_key_state="ACTIVE",
        secret_state="ACTIVE",
        secret_delivery_mode="mystery_box",
        host_secret_resolved=True,
        worker_secret_version_bound=True,
    )


class _Parts:
    """Records the download's part limits, then each attempt's start and end."""

    def __init__(self) -> None:
        self.limits = []
        self.starts: list[tuple[int, int]] = []
        self.ends: list[tuple[int, int, str, int | None]] = []

    def opened(self, limits):
        self.limits.append(limits)

    def started(self, part, attempt, offset, byte_count):
        self.starts.append((part, attempt))

    def ended(self, part, attempt, classification, byte_count, cause):
        self.ends.append((part, attempt, classification, byte_count))


def _proved(path: Path, descriptor: VerifiedObject):
    from bio_neb.verified_artifacts import ArtifactIdentity, ArtifactProof

    identity = ArtifactIdentity(
        byte_count=descriptor.byte_count, sha256=descriptor.sha256
    )
    return ArtifactProof(
        identity=identity,
        location=str(path),
        producer="vcf_object_get",
        expected_identity=identity,
        source_proof="h100-execution-receipt:" + "0" * 64,
    )


_DONOR_RECEIPT_ID = "vcf-materialization-receipt:" + "1" * 64


class _OneGet:
    def __init__(self, body: bytes = VCF_BYTES) -> None:
        self.body = body
        self.calls = []

    def download(
        self,
        object_key,
        destination,
        *,
        expected_byte_count,
        expected_sha256,
        observe_part=None,
    ):
        self.calls.append(object_key)
        if (
            expected_byte_count != len(self.body)
            or expected_sha256 != hashlib.sha256(self.body).hexdigest()
        ):
            raise AssertionError("application changed the exact descriptor")
        destination.write(self.body)
        return SimpleNamespace(
            byte_count=len(self.body),
            sha256=hashlib.sha256(self.body).hexdigest(),
            credential_receipt=_credential_receipt(),
            part_bytes=32 * 1024 * 1024,
            part_count=1,
        )


class _SourceFactory:
    def __init__(self, source: _OneGet) -> None:
        self.source = source
        self.calls = []

    def __call__(self, packet):
        self.calls.append(packet)
        return self.source


def _terminal_packet(root: Path) -> str:
    return str(
        prerequisite(
            "executed", output_payloads={"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES}
        ).journey_file
    )


def _run(arguments: list[str], *, source_factory=None, bam_source_factory=None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(
            arguments,
            vcf_object_read_factory=source_factory,
            bam_object_read_factory=bam_source_factory,
        )
    return code, stdout.getvalue(), stderr.getvalue()


class PublicVcfMaterializationTests(unittest.TestCase):
    def test_retained_live_fixture_exposes_verified_vcf_without_authority(self):
        from tests.execution_fixture import load_current_execution_fixture

        fixture = load_current_execution_fixture()
        scenario = fixture.scenarios[0]
        descriptor = next(
            item for item in scenario.verified_objects if item.role == "VCF"
        )
        self.assertEqual(scenario.evidence_kind, "retained_live")
        self.assertEqual(scenario.disposition, "succeeded")
        self.assertFalse(fixture.fixture_grants_live_authority)
        self.assertEqual(descriptor.byte_count, 48_808_433)
        self.assertEqual(
            descriptor.sha256,
            "04fc0cd2720aeb7062be122430889c2e12a6a850e41da5ac8c9d90a2cc409ede",
        )
        self.assertEqual(scenario.contacts.network_contact.status, "unmeasurable")
        self.assertIsNone(scenario.contacts.network_contact.count)

    def test_local_vcf_admission_accepts_matching_small_bytes_and_rejects_drift(self):
        from bio_neb.investigation.vcf_materialization import copy_verified_local_vcf
        from tests.execution_fixture import load_current_execution_fixture

        fixture = load_current_execution_fixture()
        scenario = fixture.scenarios[0]
        live_descriptor = next(
            item for item in scenario.verified_objects if item.role == "VCF"
        )
        # The live 48 MB descriptor cannot be satisfied by a small synthetic VCF.
        # This local vector proves the copy mechanics without fetching it.
        body = b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        descriptor = VerifiedObject(
            role="VCF",
            bucket="synthetic-bucket",
            object_key="execution/synthetic/variants.vcf",
            byte_count=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.vcf"
            source.write_bytes(body)
            source.chmod(0o444)
            copied = copy_verified_local_vcf(
                source_path=str(source),
                output_directory=str(root / "replica"),
                descriptor=descriptor,
                source_receipt_id=_DONOR_RECEIPT_ID,
                source_proof=_proved(source, descriptor),
            )
            self.assertEqual(Path(copied.materialized_vcf.path).read_bytes(), body)
            self.assertEqual(copied.proof.identity.sha256, descriptor.sha256)
            self.assertEqual(copied.proof.location, copied.materialized_vcf.path)
            self.assertEqual(copied.proof.source_proof, _DONOR_RECEIPT_ID)
            changed = root / "changed.vcf"
            changed.write_bytes(b"changed bytes")
            changed.chmod(0o444)
            with self.assertRaisesRegex(
                VcfMaterializationRefused, "candidate_identity_mismatch"
            ):
                copy_verified_local_vcf(
                    source_path=str(changed),
                    output_directory=str(root / "refused"),
                    descriptor=descriptor,
                    source_receipt_id=_DONOR_RECEIPT_ID,
                    source_proof=_proved(changed, descriptor),
                )
            with self.assertRaisesRegex(VcfMaterializationRefused, "proof_drift"):
                copy_verified_local_vcf(
                    source_path=str(source),
                    output_directory=str(root / "live-refused"),
                    descriptor=live_descriptor,
                    source_receipt_id=_DONOR_RECEIPT_ID,
                    source_proof=_proved(source, descriptor),
                )
            self.assertFalse((root / "live-refused").exists())

    def test_current_bam_acquisition_uses_terminal_run_without_coverage_authority(self):
        from tests.observation_input_fakes import RunOutputStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            packet = read_completed_parabricks_result(command)
            target = next(
                item for item in packet.verified_objects if item.role == "aligned_bam"
            )
            # 39 bytes in 16-byte parts: three ranged GETs.
            store = RunOutputStore(vcf=VCF_BYTES, bam=BAM_BYTES, part_bytes=16)
            output = root / "current-bam"
            # The BAM's identity comes only from the Run's verified Execution
            # result: the command names the Run and a destination (Q-C).
            arguments = [
                "bam-materialize-current",
                "--execution-journey-file",
                command,
                "--output-directory",
                str(output),
            ]
            code, stdout, stderr = _run(arguments, bam_source_factory=store)
            self.assertEqual(code, 0, stderr)
            result = json.loads(stdout)
            receipt = result["receipt"]
            self.assertEqual(
                store.bucket.gets,
                [
                    (target.object_key, "bytes=0-15"),
                    (target.object_key, "bytes=16-31"),
                    (target.object_key, "bytes=32-38"),
                ],
            )
            self.assertEqual((output / "aligned.bam").read_bytes(), BAM_BYTES)
            self.assertEqual(
                receipt["schema_version"], "bio-neb.current-bam-acquisition/2"
            )
            self.assertEqual(receipt["lineage"]["journey_id"], packet.journey_id)
            self.assertEqual(receipt["lineage"]["authority_id"], packet.authority_id)
            self.assertEqual((receipt["part_bytes"], receipt["part_count"]), (16, 3))
            # The streamed digest over every part is the proof; nothing rereads
            # the written file (ADR 0029).
            identity = {
                "byte_count": len(BAM_BYTES),
                "sha256": hashlib.sha256(BAM_BYTES).hexdigest(),
            }
            self.assertEqual(
                receipt["proof"],
                {
                    "identity": identity,
                    "location": str(output / "aligned.bam"),
                    "producer": "bam_object_get",
                    "expected_identity": identity,
                    "source_proof": packet.h100_receipt_id,
                },
            )
            self.assertFalse(receipt["materialized_bam"]["independently_full_read"])
            self.assertNotIn("contact_ledger", receipt)
            self.assertEqual(
                stat.S_IMODE((output / "aligned.bam").stat().st_mode), 0o444
            )
            for name in (
                "assembly_evidence",
                "sample_id",
                "sort_order",
                "required_contigs",
            ):
                self.assertNotIn(name, receipt)
            code, _, _ = _run(arguments, bam_source_factory=store)
            self.assertEqual(code, 2)
            self.assertEqual(len(store.gets), 3)

            # An operator cannot supply an identity: the retired flag is a usage
            # error before any read or contact.
            forbidden = RunOutputStore(vcf=VCF_BYTES, bam=BAM_BYTES)
            refused = [
                *arguments[:-1],
                str(root / "refused"),
                "--expected-sha256",
                hashlib.sha256(BAM_BYTES).hexdigest(),
            ]
            with self.assertRaises(SystemExit) as usage:
                _run(refused, bam_source_factory=forbidden)
            self.assertEqual(usage.exception.code, 2)
            self.assertEqual(forbidden.packets, [])
            self.assertFalse((root / "refused").exists())

    def test_current_bam_transport_failure_never_publishes_acquisition(self):
        from tests.observation_input_fakes import (
            RunOutputStore,
            access_refused,
            transient,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            for failure, store, code_name in (
                (
                    "corrupt_bytes",
                    RunOutputStore(vcf=VCF_BYTES, bam=BAM_BYTES[:-1] + b"!"),
                    "bam_materialization_object_read_mismatch",
                ),
                (
                    "access",
                    RunOutputStore([access_refused()], vcf=VCF_BYTES, bam=BAM_BYTES),
                    "bam_materialization_object_read_failed",
                ),
                (
                    # The default policy's three attempts all fault.
                    "exhausted",
                    RunOutputStore([transient()] * 3, vcf=VCF_BYTES, bam=BAM_BYTES),
                    "bam_materialization_object_read_failed",
                ),
            ):
                with self.subTest(failure=failure), patch("time.sleep"):
                    output = root / failure
                    code, stdout, stderr = _run(
                        [
                            "bam-materialize-current",
                            "--execution-journey-file",
                            command,
                            "--output-directory",
                            str(output),
                        ],
                        bam_source_factory=store,
                    )
                    self.assertEqual((code, stdout), (2, ""), stderr)
                    self.assertEqual(json.loads(stderr)["code"], code_name)
                    # No receipt, no partial bytes, no directory remain: a retry
                    # starts from scratch.
                    self.assertFalse(output.exists())

    def test_a_download_never_rereads_its_written_vcf_or_bam(self):
        # ADR 0029: the streamed digest over every part is the proof; no
        # destination reread. Every read-only open of the output is metered.
        from tests.observation_input_fakes import RunOutputStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            for kind, name in (("vcf", "variants.vcf"), ("bam", "aligned.bam")):
                with self.subTest(kind=kind):
                    output = root / kind
                    reads = []
                    original = os.open

                    def metered(
                        path,
                        flags,
                        *args,
                        target=str(output / name),
                        reads=reads,
                        original=original,
                        **kwargs,
                    ):
                        if str(path) == target and flags & os.O_ACCMODE == os.O_RDONLY:
                            reads.append(path)
                        return original(path, flags, *args, **kwargs)

                    store = RunOutputStore(vcf=VCF_BYTES, bam=BAM_BYTES, part_bytes=8)
                    with patch.object(os, "open", metered):
                        code, _, stderr = _run(
                            [
                                f"{kind}-materialize"
                                if kind == "vcf"
                                else "bam-materialize-current",
                                "--execution-journey-file",
                                command,
                                "--output-directory",
                                str(output),
                            ],
                            source_factory=store,
                            bam_source_factory=store,
                        )
                    self.assertEqual(code, 0, stderr)
                    self.assertGreater(len(store.gets), 1)
                    self.assertEqual(reads, [])

    def test_bam_shared_read_adapts_file_identity_without_weakening_vcf_type(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory).resolve() / "artifact.bin"
            artifact.write_bytes(BAM_BYTES)

            bam_file, _ = MaterializationFiles(
                BamMaterializedFile, BamMaterializationRefused
            ).stable_full_read(artifact, require_read_only=False)
            vcf_file, _ = MaterializationFiles(
                VcfMaterializationFile, VcfMaterializationRefused
            ).stable_full_read(artifact, require_read_only=False)

            self.assertIs(type(bam_file), BamMaterializedFile)
            self.assertIs(type(vcf_file), VcfMaterializationFile)

    def test_get_materializes_one_immutable_current_vcf_and_reopens_without_contact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            output = root / "materialized"
            source = _OneGet()
            factory = _SourceFactory(source)
            code, stdout, stderr = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--output-directory",
                    str(output),
                ],
                source_factory=factory,
            )

            self.assertEqual((code, stderr), (0, ""))
            result = json.loads(stdout)
            assert_json_identity(
                self,
                result["receipt"]["receipt_id"],
                result["receipt"],
                "schema_version lineage descriptor materialized_vcf proof source_candidate source_candidate_receipt "
                "source_candidate_receipt_id credential_receipt verification_policy verification_checks",
                "vcf-materialization-receipt:",
            )
            self.assertEqual(len(factory.calls), 1)
            packet = factory.calls[0]
            journey = JourneyPath.open(Path(packet.journey_file))
            manifest_pointer = next(
                p for p in journey.state.h100.pointers if p.role == "execution_manifest"
            )
            proof = json.loads(journey.read_artifact(manifest_pointer))[
                "transfer_proof"
            ]
            self.assertEqual(proof["verification_mode"], "reuse_creation_proof")
            self.assertEqual(proof["host_audit_bytes"], 0)
            self.assertEqual(
                result["receipt"]["lineage"]["h100_receipt_id"],
                journey.state.h100.receipt_id,
            )
            self.assertEqual(
                result["receipt"]["lineage"]["composition_binding_id"],
                packet.composition_binding_id,
            )
            self.assertEqual(
                source.calls, [result["receipt"]["descriptor"]["object_key"]]
            )
            self.assertEqual(
                result["receipt"]["proof"],
                {
                    "identity": {
                        "byte_count": len(VCF_BYTES),
                        "sha256": hashlib.sha256(VCF_BYTES).hexdigest(),
                    },
                    "location": result["attested_vcf"]["local_path"],
                    "producer": "vcf_object_get",
                    "expected_identity": {
                        "byte_count": len(VCF_BYTES),
                        "sha256": hashlib.sha256(VCF_BYTES).hexdigest(),
                    },
                    "source_proof": journey.state.h100.receipt_id,
                },
            )
            # The streamed GET established the identity; nothing reread it.
            self.assertEqual(result["receipt"]["verification_checks"], [])
            self.assertFalse(
                result["receipt"]["materialized_vcf"]["independently_full_read"]
            )
            credential = result["receipt"]["credential_receipt"]
            topology = result["receipt"]["lineage"]["selected_context"]["topology"]
            self.assertEqual(
                (
                    credential["service_account_id"],
                    credential["access_key_resource_id"],
                    credential["access_key_id"],
                ),
                (
                    topology["service_account_id"],
                    topology["access_key_resource_id"],
                    topology["access_key_id"],
                ),
            )
            self.assertEqual(
                (
                    credential["bootstrap_claim_counts"],
                    credential["bootstrap_invocation_counts"],
                ),
                (
                    {
                        "service_account": 0,
                        "group": 0,
                        "group_membership": 0,
                        "access_permit": 0,
                        "access_key": 0,
                    },
                )
                * 2,
            )
            vcf = Path(result["attested_vcf"]["local_path"])
            receipt = Path(result["attested_vcf"]["acquisition_receipt_path"])
            self.assertEqual(vcf.read_bytes(), VCF_BYTES)
            self.assertEqual(stat.S_IMODE(vcf.stat().st_mode) & 0o222, 0)
            self.assertEqual(stat.S_IMODE(receipt.stat().st_mode) & 0o222, 0)

            refused = _OneGet()
            refused_factory = _SourceFactory(refused)
            second, _, second_error = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--output-directory",
                    str(output),
                ],
                source_factory=refused_factory,
            )
            self.assertEqual(second, 2)
            self.assertEqual(refused.calls, [])
            self.assertEqual(refused_factory.calls, [])
            self.assertEqual(
                json.loads(second_error)["code"], "vcf_materialization_output_exists"
            )
            self.assertEqual(vcf.read_bytes(), VCF_BYTES)

    def test_a_transient_part_fault_retries_by_the_registered_policy(self) -> None:
        from bio_neb.failure import FailureTuning, run_failure_tuning
        from bio_neb.investigation.vcf_materialization import (
            MaterializeCurrentVcfCommand,
            materialize_current_vcf,
        )
        from tests.observation_input_fakes import (
            RunOutputStore,
            access_refused,
            transient,
        )

        tuning = FailureTuning.model_validate(
            {
                "scopes": {
                    "stage": {
                        "vcf_materialization": {
                            "operation": {"retry_attempts": 2, "backoff_seconds": 0}
                        }
                    }
                }
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            journey_file = _terminal_packet(root / "case")
            cases = (
                ("recovered", [transient()], None),
                ("exhausted", [transient(), transient()], "transient"),
                ("access", [access_refused()], "access"),
            )
            for name, faults, refused_class in cases:
                with self.subTest(name):
                    store = RunOutputStore(faults, vcf=VCF_BYTES, bam=BAM_BYTES)
                    parts = _Parts()
                    command = MaterializeCurrentVcfCommand(
                        execution_journey_file=journey_file,
                        output_directory=str(root / name),
                    )
                    with run_failure_tuning(tuning):
                        if refused_class is None:
                            result = materialize_current_vcf(
                                command,
                                object_read_factory=store,
                                observe_part=parts,
                            )
                        else:
                            with self.assertRaises(VcfMaterializationRefused) as error:
                                materialize_current_vcf(
                                    command,
                                    object_read_factory=store,
                                    observe_part=parts,
                                )
                    if refused_class is None:
                        # One fresh replica; the failed attempt wrote nothing.
                        self.assertEqual(
                            parts.ends,
                            [
                                (1, 1, "transient", None),
                                (1, 2, "success", len(VCF_BYTES)),
                            ],
                        )
                        self.assertEqual(
                            Path(result.attested_vcf.local_path).read_bytes(),
                            VCF_BYTES,
                        )
                    else:
                        self.assertEqual(
                            (error.exception.code, error.exception.details[-1]),
                            ("vcf_materialization_object_read_failed", refused_class),
                        )
                        # The Run's policy allows two attempts; access is never retried.
                        self.assertEqual(
                            [end[2] for end in parts.ends],
                            ["transient", "transient"]
                            if name == "exhausted"
                            else ["access"],
                        )
                        # Nothing remains, so a retry starts from scratch.
                        self.assertFalse((root / name).exists())
                    # Each attempt started before its request and ended after it,
                    # under the part deadline the stage resolved.
                    self.assertEqual(
                        parts.starts, [(part, n) for part, n, _, _ in parts.ends]
                    )
                    self.assertEqual(
                        [
                            (item.surface.name, item.effective.part_seconds)
                            for item in parts.limits
                        ],
                        [("vcf_materialization", 300)],
                    )
                    self.assertEqual(len(store.gets), len(parts.starts))

    def test_execution_storage_reports_its_credential_lookups_to_the_observer(
        self,
    ) -> None:
        # The product path gives the selected context's resolver the caller's
        # observer, so a grant's journal counts each lookup.
        from bio_neb.application.artifact_materialization import (
            execution_object_storage,
        )

        built = []

        class Resolver:
            def __init__(self, target, selection, *, execution_contact_observer=None):
                built.append((target, selection, execution_contact_observer))

        observer = object()
        with tempfile.TemporaryDirectory() as directory:
            packet = read_completed_parabricks_result(
                _terminal_packet(Path(directory).resolve() / "case")
            )
            with patch(
                "bio_neb.providers.nebius_object_storage_credentials."
                "SelectedNebiusCredentialResolver",
                Resolver,
            ):
                storage = execution_object_storage(
                    packet, RuntimeError("context missing"), observer
                )
        self.assertEqual(built, [(None, packet.selected_context, observer)])
        self.assertIsNotNone(storage)

    def test_equal_candidate_and_receipt_select_zero_contact_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            first = root / "first"
            source = _OneGet()
            source_factory = _SourceFactory(source)
            code, stdout, _ = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--output-directory",
                    str(first),
                ],
                source_factory=source_factory,
            )
            self.assertEqual(code, 0)
            first_result = json.loads(stdout)

            second = root / "second"
            argv = [
                "vcf-materialize",
                "--execution-journey-file",
                command,
                "--candidate-file",
                first_result["attested_vcf"]["local_path"],
                "--candidate-receipt-file",
                first_result["attested_vcf"]["acquisition_receipt_path"],
                "--output-directory",
                str(second),
            ]
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,sys; from tests.test_vcf_materialization import "
                        "measured_rebinding; print(json.dumps(measured_rebinding(json.loads(sys.argv[1]))))"
                    ),
                    json.dumps(argv),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            measured = json.loads(child.stdout)
            self.assertNotEqual(measured["pid"], os.getpid())
            code, stdout, stderr = (
                measured[key] for key in ("code", "stdout", "stderr")
            )
            self.assertEqual((code, stderr), (0, ""))
            rebound = json.loads(stdout)
            source_path = first_result["attested_vcf"]["local_path"]
            destination_path = rebound["attested_vcf"]["local_path"]
            # Proved bytes are copied once, unhashed, never reread and with no
            # custody stat of the source (0029); only the fresh replica's own
            # initial admission stats it.
            self.assertEqual(measured["reads"], {source_path: len(VCF_BYTES)})
            self.assertEqual(measured["hashed_bytes"], 0)
            self.assertNotIn(source_path, measured["stats"])
            self.assertNotEqual(
                Path(source_path).stat().st_ino, Path(destination_path).stat().st_ino
            )
            self.assertEqual(
                rebound["receipt"]["proof"]["source_proof"],
                first_result["receipt"]["receipt_id"],
            )
            assert_json_identity(
                self,
                rebound["receipt"]["receipt_id"],
                rebound["receipt"],
                "schema_version lineage descriptor materialized_vcf proof source_candidate source_candidate_receipt "
                "source_candidate_receipt_id credential_receipt verification_policy verification_checks",
                "vcf-materialization-receipt:",
            )
            self.assertEqual(rebound["receipt"]["proof"]["producer"], "vcf_local_copy")
            self.assertIsNone(rebound["receipt"]["credential_receipt"])
            self.assertEqual(
                rebound["receipt"]["source_candidate_receipt_id"],
                first_result["receipt"]["receipt_id"],
            )
            self.assertEqual(
                Path(rebound["attested_vcf"]["local_path"]).read_bytes(),
                VCF_BYTES,
            )

    def test_ordinary_observation_reads_no_replica_byte_and_an_audit_hashes_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            code, stdout, stderr = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--output-directory",
                    str(root / "first"),
                ],
                source_factory=lambda _packet: _OneGet(),
            )
            self.assertEqual((code, stderr), (0, ""))
            attested = json.loads(stdout)["attested_vcf"]
            pids = set()
            for sensitive in (False, False, True):
                child = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import json,sys; from tests.test_vcf_materialization import measured_rebinding; "
                            "print(json.dumps(measured_rebinding(json.loads(sys.argv[1]), observation=True, "
                            "sensitive=json.loads(sys.argv[2]))))"
                        ),
                        json.dumps(
                            [
                                attested["acquisition_receipt_path"],
                                attested["local_path"],
                            ]
                        ),
                        json.dumps(sensitive),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                measured = json.loads(child.stdout)
                pids.add(measured["pid"])
                self.assertEqual(
                    measured["reads"],
                    {attested["local_path"]: len(VCF_BYTES)} if sensitive else {},
                )
                self.assertEqual(
                    measured["hashed_bytes"], len(VCF_BYTES) if sensitive else 0
                )
                result = json.loads(measured["stdout"])
                if sensitive:
                    self.assertEqual(
                        result["observation"]["sha256"],
                        hashlib.sha256(VCF_BYTES).hexdigest(),
                    )
                else:
                    self.assertIsNone(result["observation"])
            self.assertEqual(len(pids), 3)

    def test_invocation_policy_controls_copy_without_rewriting_source_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            code, stdout, stderr = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--output-directory",
                    str(root / "first"),
                ],
                source_factory=lambda _packet: _OneGet(),
            )
            self.assertEqual((code, stderr), (0, ""))
            original = json.loads(stdout)
            self.assertEqual(
                original["receipt"]["verification_policy"]["effective"]["mode"],
                "reuse_creation_proof",
            )
            attested = original["attested_vcf"]
            receipt_path = Path(attested["acquisition_receipt_path"])
            retained = receipt_path.read_bytes()
            config = root / "runtime.json"
            config.write_text(
                json.dumps(
                    {
                        "contract_version": "1.0",
                        "heartbeat_interval_seconds": 1,
                        "model_roster": {
                            "qwen": {
                                "provider": "nebius",
                                "endpoint": "http://127.0.0.1:8080/v1",
                                "model": "Qwen/Qwen3.5-397B-A17B",
                                "credential_env": "UNUSED_KEY",
                            }
                        },
                        "agent_invocation": {"model_reference": "qwen"},
                        "artifact_verification": {
                            "mode": "independent_full_read",
                            "reason": "explicit runtime audit",
                        },
                    }
                ),
                encoding="utf-8",
            )
            for index, override in enumerate(
                ([], ["--artifact-verification", "reuse_creation_proof"])
            ):
                output = root / f"copy-{index}"
                measured = measured_rebinding(
                    [
                        "vcf-materialize",
                        "--execution-journey-file",
                        command,
                        "--candidate-file",
                        attested["local_path"],
                        "--candidate-receipt-file",
                        str(receipt_path),
                        "--output-directory",
                        str(output),
                        "--runtime-config",
                        str(config),
                        *override,
                    ]
                )
                self.assertEqual((measured["code"], measured["stderr"]), (0, ""))
                rebound = json.loads(measured["stdout"])
                policy = rebound["receipt"]["verification_policy"]
                self.assertEqual(policy["defaults"]["mode"], "independent_full_read")
                self.assertEqual(
                    policy["overrides"],
                    {"mode": "reuse_creation_proof", "reason": None}
                    if override
                    else {},
                )
                # The audit hashes the copy stream and rereads the replica once;
                # the trusted override copies once and hashes nothing.
                destination = rebound["attested_vcf"]["local_path"]
                self.assertEqual(
                    measured["hashed_bytes"], (0 if override else 2) * len(VCF_BYTES)
                )
                self.assertEqual(
                    measured["reads"],
                    {attested["local_path"]: len(VCF_BYTES)}
                    if override
                    else {
                        attested["local_path"]: len(VCF_BYTES),
                        destination: len(VCF_BYTES),
                    },
                )
                checks = rebound["receipt"]["verification_checks"]
                self.assertEqual(
                    [(check["reason"], check["path"]) for check in checks],
                    []
                    if override
                    else [
                        ("policy_full_hash", attested["local_path"]),
                        ("policy_full_hash", destination),
                    ],
                )
            self.assertEqual(receipt_path.read_bytes(), retained)

    def test_an_explicit_audit_detects_a_changed_proved_replica(self) -> None:
        # Trusted reuse rechecks nothing (0029); a selected full read still
        # detects bytes that changed under an applicable-looking proof.
        from bio_neb.application.verification_policy import (
            ArtifactVerificationDefaults,
            ArtifactVerificationOverrides,
            resolve_artifact_verification_policy,
        )
        from bio_neb.investigation.vcf_materialization import (
            VcfMaterializationRefused,
            copy_verified_local_vcf,
        )

        audit = resolve_artifact_verification_policy(
            ArtifactVerificationDefaults(),
            ArtifactVerificationOverrides(
                mode="independent_full_read", reason="explicit replica audit"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected = VerifiedObject(
                role="VCF",
                bucket="retained",
                object_key="variants.vcf",
                byte_count=len(VCF_BYTES),
                sha256=hashlib.sha256(VCF_BYTES).hexdigest(),
            )
            changed = root / "changed.vcf"
            changed.write_bytes(b"!" + VCF_BYTES[1:])
            changed.chmod(0o444)
            with self.assertRaisesRegex(
                VcfMaterializationRefused, "candidate_identity_mismatch"
            ):
                copy_verified_local_vcf(
                    source_path=str(changed),
                    output_directory=str(root / "audited"),
                    descriptor=expected,
                    source_receipt_id=_DONOR_RECEIPT_ID,
                    source_proof=_proved(changed, expected),
                    verification_policy=audit,
                )
            link = root / "link.vcf"
            link.symlink_to(changed)
            with self.assertRaises(VcfMaterializationRefused):
                copy_verified_local_vcf(
                    source_path=str(link),
                    output_directory=str(root / "linked"),
                    descriptor=expected,
                    source_receipt_id=_DONOR_RECEIPT_ID,
                    source_proof=_proved(link, expected),
                )

    def test_a_reason_without_a_verification_selection_refuses_before_contact(
        self,
    ) -> None:
        # Migrated from the retired investigation-observe CLI: a reason amends
        # only an explicit selection, so alone it is an invalid command.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            forbidden = _SourceFactory(_OneGet())
            code, stdout, stderr = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--output-directory",
                    str(root / "refused"),
                    "--artifact-verification-reason",
                    "explicit audit",
                ],
                source_factory=forbidden,
            )
            self.assertEqual((code, stdout), (2, ""))
            self.assertEqual(json.loads(stderr)["code"], "command_invalid")
            self.assertEqual(forbidden.calls, [])
            self.assertFalse((root / "refused").exists())

    def test_candidate_refusals_never_fall_through_to_object_contact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = _terminal_packet(root / "case")
            first = root / "first"
            code, stdout, _ = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--output-directory",
                    str(first),
                ],
                source_factory=lambda packet: _OneGet(),
            )
            self.assertEqual(code, 0)
            retained = json.loads(stdout)["attested_vcf"]
            candidate = Path(retained["local_path"])
            receipt = Path(retained["acquisition_receipt_path"])

            prior = VcfMaterializationReceipt.model_validate_json(receipt.read_bytes())
            stale = VcfMaterializationReceipt.bind(
                lineage=prior.lineage,
                descriptor=prior.descriptor.model_copy(
                    update={"bucket": "stale-run-bucket"}
                ),
                materialized_vcf=prior.materialized_vcf,
                proof=prior.proof,
                source_candidate=prior.source_candidate,
                source_candidate_receipt=prior.source_candidate_receipt,
                source_candidate_receipt_id=prior.source_candidate_receipt_id,
                credential_receipt=prior.credential_receipt,
                verification_policy=prior.verification_policy,
            )
            stale_receipt = root / "stale-receipt.json"
            stale_receipt.write_text(
                canonical_json(stale.model_dump(mode="json")), encoding="utf-8"
            )
            stale_receipt.chmod(0o444)

            mismatch = root / "mismatch.vcf"
            mismatch.write_bytes(b"different-vcf")
            mismatch.chmod(0o444)
            malformed_receipt = root / "malformed-receipt.json"
            malformed_receipt.write_bytes(b"{}")
            malformed_receipt.chmod(0o444)
            writable_receipt = root / "writable-receipt.json"
            writable_receipt.write_bytes(receipt.read_bytes())
            writable_receipt.chmod(0o644)
            cases = (
                (
                    root / "missing.vcf",
                    receipt,
                    "vcf_materialization_file_unavailable",
                ),
                (
                    mismatch,
                    receipt,
                    "vcf_materialization_candidate_identity_mismatch",
                ),
                (
                    candidate,
                    malformed_receipt,
                    "vcf_materialization_candidate_receipt_invalid",
                ),
                (
                    candidate,
                    stale_receipt,
                    "vcf_materialization_candidate_identity_mismatch",
                ),
                (
                    candidate,
                    writable_receipt,
                    "vcf_materialization_candidate_mutable",
                ),
                (
                    receipt,
                    receipt,
                    "vcf_materialization_candidate_collision",
                ),
            )
            for index, (selected, selected_receipt, expected) in enumerate(cases):
                forbidden = _OneGet()
                forbidden_factory = _SourceFactory(forbidden)
                code, _, stderr = _run(
                    [
                        "vcf-materialize",
                        "--execution-journey-file",
                        command,
                        "--candidate-file",
                        str(selected),
                        "--candidate-receipt-file",
                        str(selected_receipt),
                        "--output-directory",
                        str(root / f"refused-{index}"),
                    ],
                    source_factory=forbidden_factory,
                )
                self.assertEqual(code, 2)
                self.assertEqual(forbidden.calls, [])
                self.assertEqual(forbidden_factory.calls, [])
                self.assertEqual(json.loads(stderr)["code"], expected)
                self.assertFalse((root / f"refused-{index}").exists())

            # Trusted reuse does not recheck its source's mode (0029); a selected
            # audit reads it in full and refuses a source that is not read-only.
            candidate.chmod(0o644)
            forbidden = _OneGet()
            forbidden_factory = _SourceFactory(forbidden)

            code, _, stderr = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    command,
                    "--candidate-file",
                    str(candidate),
                    "--candidate-receipt-file",
                    retained["acquisition_receipt_path"],
                    "--output-directory",
                    str(root / "refused"),
                    "--artifact-verification",
                    "independent_full_read",
                    "--artifact-verification-reason",
                    "explicit candidate audit",
                ],
                source_factory=forbidden_factory,
            )
            self.assertEqual(code, 2)
            self.assertEqual(forbidden.calls, [])
            self.assertEqual(forbidden_factory.calls, [])
            self.assertEqual(
                json.loads(stderr)["code"], "vcf_materialization_candidate_mutable"
            )
            self.assertEqual(list((root / "refused").iterdir()), [])

    def test_incomplete_execution_and_unpaired_candidate_refuse_before_contact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            approved = approved_input(root / "case")

            def collided(plan, journal):
                provider = input_storage_provider(plan, journal)
                provider.scripts["input_storage", "fresh_bucket"] = deque(
                    [ProviderError("semantic", "bucket_collision")]
                )
                return provider

            state = execute_parabricks(approved, infrastructure_factory=collided)
            self.assertEqual(state.failures[-1].detail, "bucket_collision")
            incomplete = approved.journey_file
            forbidden = _OneGet()
            forbidden_factory = _SourceFactory(forbidden)
            code, _, stderr = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    incomplete,
                    "--output-directory",
                    str(root / "output"),
                ],
                source_factory=forbidden_factory,
            )
            self.assertEqual(code, 2)
            self.assertEqual(forbidden.calls, [])
            self.assertEqual(forbidden_factory.calls, [])
            self.assertEqual(
                json.loads(stderr)["code"], "vcf_materialization_execution_incomplete"
            )

            code, _, stderr = _run(
                [
                    "bam-materialize-current",
                    "--execution-journey-file",
                    incomplete,
                    "--output-directory",
                    str(root / "bam-output"),
                ],
                bam_source_factory=forbidden_factory,
            )
            self.assertEqual(code, 2)
            self.assertEqual(
                json.loads(stderr)["code"], "bam_materialization_execution_incomplete"
            )
            self.assertEqual(forbidden_factory.calls, [])
            self.assertFalse((root / "bam-output").exists())

            code, _, stderr = _run(
                [
                    "vcf-materialize",
                    "--execution-journey-file",
                    incomplete,
                    "--candidate-file",
                    str(root / "candidate.vcf"),
                    "--output-directory",
                    str(root / "unpaired"),
                ],
                source_factory=forbidden_factory,
            )
            self.assertEqual(code, 2)
            self.assertEqual(
                json.loads(stderr)["code"], "vcf_materialization_candidate_unpaired"
            )


class InitialReplicaDirectoryTests(RunPrerequisites, unittest.TestCase):
    """VCF and BAM acquisition check only the file and its own directory."""

    def _parent(self) -> str:
        return str(
            self.prerequisite(
                "executed", output_payloads={"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES}
            ).journey_file
        )

    def _commands(self, journey_file: str, output: Path) -> dict:
        return {
            "vcf": [
                "vcf-materialize",
                "--execution-journey-file",
                journey_file,
                "--output-directory",
                str(output / "vcf"),
            ],
            "bam": [
                "bam-materialize-current",
                "--execution-journey-file",
                journey_file,
                "--output-directory",
                str(output / "bam"),
            ],
        }

    @staticmethod
    def _run_kind(kind: str, arguments: list[str], factory):
        if kind == "vcf":
            return _run(arguments, source_factory=factory)
        return _run(arguments, bam_source_factory=factory)

    @staticmethod
    def _copy(output: Path, root: Path):
        from bio_neb.investigation.vcf_materialization import copy_verified_local_vcf

        donor = root / "donor.vcf"
        if not donor.exists():
            donor.write_bytes(VCF_BYTES)
            donor.chmod(0o444)
        descriptor = VerifiedObject(
            role="VCF",
            bucket="retained",
            object_key="variants.vcf",
            byte_count=len(VCF_BYTES),
            sha256=hashlib.sha256(VCF_BYTES).hexdigest(),
        )
        return copy_verified_local_vcf(
            source_path=str(donor),
            output_directory=str(output),
            descriptor=descriptor,
            source_receipt_id=_DONOR_RECEIPT_ID,
            source_proof=_proved(donor, descriptor),
        )

    def test_writable_non_root_ancestors_admit_vcf_and_bam_with_one_get(self) -> None:
        from bio_neb.application.local_artifact_proof_adapter import (
            controlled_replica_metadata,
        )

        journey_file = self._parent()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            # User-owned, group/world-writable and not sticky: the retired
            # ancestor walk refused this chain after the GET.
            shared = root / "world" / "live-runs"
            shared.mkdir(parents=True)
            (root / "world").chmod(0o777)
            shared.chmod(0o775)
            commands = self._commands(journey_file, shared)
            for kind, body, name in (
                ("vcf", VCF_BYTES, "variants.vcf"),
                ("bam", BAM_BYTES, "aligned.bam"),
            ):
                with self.subTest(kind=kind):
                    factory = _SourceFactory(_OneGet(body))
                    code, stdout, stderr = self._run_kind(kind, commands[kind], factory)
                    self.assertEqual((code, stderr), (0, ""))
                    self.assertEqual(len(factory.calls), 1)
                    self.assertEqual(len(factory.source.calls), 1)
                    own = shared / kind
                    materialized = own / name
                    self.assertEqual(materialized.read_bytes(), body)
                    self.assertEqual(stat.S_IMODE(own.stat().st_mode), 0o700)
                    self.assertEqual(stat.S_IMODE(materialized.stat().st_mode), 0o444)
                    controlled_replica_metadata(str(materialized))
            with self.subTest(kind="local_copy"):
                copied = self._copy(shared / "copy", root)
                controlled_replica_metadata(copied.materialized_vcf.path)
                self.assertEqual(
                    Path(copied.materialized_vcf.path).read_bytes(), VCF_BYTES
                )

    def test_bad_own_directory_refuses_before_get_with_typed_code(self) -> None:
        journey_file = self._parent()
        for case, observed_mode in (("setgid_parent", "0o2700"), ("umask", "0o600")):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                parent = root / "parent"
                parent.mkdir(mode=0o700)
                if case == "setgid_parent":
                    # Linux gives a directory created here the setgid bit: 02700.
                    parent.chmod(0o2700)
                    self.assertTrue(parent.stat().st_mode & stat.S_ISGID)
                commands = self._commands(journey_file, parent)
                for kind in ("vcf", "bam", "local_copy"):
                    with self.subTest(kind=kind):
                        factory = _SourceFactory(_OneGet())
                        own = parent / kind
                        previous = os.umask(0o177) if case == "umask" else None
                        try:
                            if kind == "local_copy":
                                with self.assertRaises(
                                    VcfMaterializationRefused
                                ) as raised:
                                    self._copy(own, root)
                                failure = {
                                    "code": raised.exception.code,
                                    "fields": list(raised.exception.details),
                                }
                            else:
                                code, stdout, stderr = self._run_kind(
                                    kind, commands[kind], factory
                                )
                                self.assertEqual((code, stdout), (2, ""), stderr)
                                failure = json.loads(stderr)
                        finally:
                            if previous is not None:
                                os.umask(previous)
                        prefix = "bam" if kind == "bam" else "vcf"
                        self.assertEqual(
                            failure["code"],
                            f"{prefix}_materialization_output_directory_invalid",
                        )
                        self.assertEqual(failure["fields"][0], str(own))
                        self.assertIn(f"({observed_mode})", failure["fields"][1])
                        self.assertEqual(factory.calls, [])
                        self.assertEqual(factory.source.calls, [])
                        own.chmod(0o700)
                        self.assertEqual(list(own.iterdir()), [])

    def test_output_link_added_during_transfer_refuses_vcf_and_bam_alike(
        self,
    ) -> None:
        journey_file = self._parent()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            commands = self._commands(journey_file, root)
            for kind, body, name, receipt in (
                ("vcf", VCF_BYTES, "variants.vcf", "vcf-materialization-receipt.json"),
                (
                    "bam",
                    BAM_BYTES,
                    "aligned.bam",
                    "current-bam-acquisition-receipt.json",
                ),
            ):
                with self.subTest(kind=kind):
                    output = root / kind / name
                    second = root / f"{kind}-second-link"

                    class LinkingGet(_OneGet):
                        def download(
                            self,
                            object_key,
                            destination,
                            output=output,
                            second=second,
                            **expected,
                        ):
                            result = super().download(
                                object_key, destination, **expected
                            )
                            os.link(output, second)
                            return result

                    factory = _SourceFactory(LinkingGet(body))
                    code, stdout, stderr = self._run_kind(kind, commands[kind], factory)
                    self.assertEqual((code, stdout), (2, ""), stderr)
                    failure = json.loads(stderr)
                    self.assertEqual(
                        failure["code"],
                        f"{kind}_materialization_output_replica_invalid",
                    )
                    self.assertEqual(failure["fields"][0], str(output))
                    self.assertEqual(len(factory.source.calls), 1)
                    self.assertFalse((root / kind / receipt).exists())


if __name__ == "__main__":
    unittest.main()
