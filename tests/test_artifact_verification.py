"""Shared artifact identity and purpose-bound audits, without synthetic pass counts."""

import unittest

from bio_neb.application.artifact_verification import use_produced_object
from bio_neb.failure import ProviderError
from bio_neb.providers.fake import FakeProvider
from bio_neb.verified_artifacts import ArtifactIdentity, ArtifactProof, VerifiedObject


class ArtifactVerificationTests(unittest.TestCase):
    def test_admission_distinguishes_acquired_identity_and_independent_expectation(
        self,
    ):
        identity = ArtifactIdentity(
            byte_count=3,
            sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        for location in ("/controlled/abc", "s3://bucket/abc"):
            proof = ArtifactProof(
                identity=identity,
                location=location,
                producer="import",
                expected_identity=None,
            )
            self.assertIsNone(proof.expected_identity)
            self.assertTrue(proof.proof_id.startswith("artifact-proof:"))
            self.assertEqual(
                ArtifactProof(
                    identity=identity,
                    location=location,
                    producer="import",
                    expected_identity=identity,
                ).identity,
                identity,
            )
            with self.assertRaisesRegex(ValueError, "independent expectation"):
                ArtifactProof(
                    identity=identity,
                    location=location,
                    producer="import",
                    expected_identity=ArtifactIdentity(byte_count=4, sha256="0" * 64),
                )

    def test_explicit_audit_keeps_origin_and_refuses_actual_mismatch(self):
        store = FakeProvider().store
        created = store.put("bucket", "abc", b"abc")
        item = VerifiedObject(
            role="test",
            bucket="bucket",
            object_key="abc",
            byte_count=3,
            sha256=created.values["sha256"],
        )
        checked = use_produced_object(
            store, item, proof_ref="original-producer", mode="independent_full_read"
        )
        self.assertEqual(checked.proof_ref, "original-producer")
        store.provider.objects["bucket", "abc"] = b"changed"
        with self.assertRaisesRegex(ProviderError, "artifact_identity_mismatch"):
            use_produced_object(
                store, item, proof_ref="original-producer", mode="independent_full_read"
            )
        # Ordinary reuse makes the deliberately chosen trust promise explicit.
        trusted = use_produced_object(
            store, item, proof_ref="original-producer", mode="reuse_creation_proof"
        )
        self.assertEqual(trusted.values["bytes"], 3)
        with self.assertRaisesRegex(ProviderError, "artifact_producer_proof_missing"):
            use_produced_object(
                store, item, proof_ref=None, mode="reuse_creation_proof"
            )
