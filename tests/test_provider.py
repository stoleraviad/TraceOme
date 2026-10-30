"""Offline port contract: named mutations return one retained resource."""

import hashlib
import unittest

from bio_neb.failure import ProviderError
from bio_neb.providers.fake import FakeProvider
from bio_neb.providers.ports import (
    Identity,
    JobRunner,
    ObjectStore,
    Receipt,
    TransferSource,
)
from bio_neb.verified_artifacts import ArtifactProof


class ProviderTests(unittest.TestCase):
    def test_created_and_copied_artifacts_retain_identity_and_original_proof(self):
        from bio_neb.verified_artifacts import ArtifactProof

        store = FakeProvider().store
        created = store.put("bucket", "key", b"abc")
        proof = ArtifactProof.model_validate(created.values["artifact_proof"])
        self.assertEqual(
            proof.identity.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertIsNone(proof.expected_identity)
        copied = store.copy(proof, "bucket", "replica")
        replica = ArtifactProof.model_validate(copied.values["artifact_proof"])
        self.assertEqual(replica.source_proof, proof.proof_id)
        self.assertEqual(replica.identity, proof.identity)
        self.assertEqual(replica.location, "s3://bucket/replica")

    def test_copy_reuses_proof_and_explicit_audit_alone_detects_silent_mutation(self):
        provider = FakeProvider()
        created = provider.store.put("bucket", "key", b"abc")
        proof = ArtifactProof.model_validate(created.values["artifact_proof"])
        provider.objects["bucket", "key"] = b"changed"
        copied = provider.store.copy(proof, "bucket", "replica")
        self.assertEqual(copied.values["sha256"], proof.identity.sha256)
        audit = provider.store.head(
            "bucket", "replica", verification="independent_full_read"
        )
        self.assertNotEqual(audit.values["sha256"], proof.identity.sha256)

    def test_stream_sources_are_explicitly_seeded_and_hash_checked_offline(self):
        source = TransferSource(
            "url",
            "https://example.invalid/source",
            4,
            hashlib.sha256(b"data").hexdigest(),
        )
        provider = FakeProvider(sources={source.locator: b"data"})
        receipt = provider.store.put("bucket", "source", source)
        self.assertEqual(receipt.values["sha256"], source.sha256)
        self.assertEqual(provider.store.read_small("bucket", "source").content, b"data")
        with self.assertRaisesRegex(ProviderError, "source_identity_mismatch"):
            FakeProvider(sources={source.locator: b"drift"}).store.put(
                "bucket", "source", source
            )

    def test_scripts_are_scoped_to_step_and_ports_have_no_destructive_capability(self):
        provider = FakeProvider(
            {("cpu", "submit"): [ProviderError("transient", "busy")]}
        )
        ports = provider.for_step("cpu", "submit")
        with self.assertRaisesRegex(ProviderError, "busy"):
            ports.jobs.submit("cpu", {})
        self.assertEqual(ports.jobs.submit("cpu", {}).resource_count, 1)
        self.assertEqual(
            provider.for_step("input", "auth").identity.whoami().values,
            {"identity": "offline-fake"},
        )
        for port, protocol in (
            (ports.store, ObjectStore),
            (ports.jobs, JobRunner),
            (ports.identity, Identity),
        ):
            self.assertIsInstance(port, protocol)
            self.assertFalse(hasattr(port, "delete"))
            self.assertFalse(hasattr(port, "cancel"))

    def test_precheck_reuses_an_already_created_resource_without_new_mutation(self):
        provider = FakeProvider(
            {("cpu", "submit"): [ProviderError("rejected", "no capacity")]}
        )
        retained = Receipt(provider_ids=("retained-job",), resource_count=1)
        provider.jobs_by_name["run-cpu"] = retained
        self.assertEqual(
            provider.for_step("cpu", "submit").jobs.submit("run-cpu", {}), retained
        )

    def test_object_mutations_are_idempotent_and_never_replace_different_content(self):
        store = FakeProvider().for_step("inputs", "upload").store
        first = store.put("bucket", "key", b"data")
        self.assertEqual(first, store.put("bucket", "key", b"data"))
        with self.assertRaisesRegex(ProviderError, "object_content_conflict"):
            store.put("bucket", "key", b"changed")
        store.copy(
            ArtifactProof.model_validate(
                store.put("bucket", "key", b"data").values["artifact_proof"]
            ),
            "bucket",
            "replica",
        )
        self.assertEqual(store.read_small("bucket", "replica").content, b"data")
        self.assertEqual(store.head("bucket", "replica").values["bytes"], 4)

    def test_named_resources_survive_reinvoke(self):
        provider = FakeProvider()
        first = provider.for_step("inputs", "bucket").store.fresh_bucket("run-bucket")
        second = provider.for_step("inputs", "bucket").store.fresh_bucket("run-bucket")
        self.assertEqual(first.provider_ids, second.provider_ids)
        self.assertEqual(second.resource_count, 1)
        jobs = provider.for_step("cpu", "submit").jobs
        first_job = jobs.submit("run-cpu", {"image": "approved-image"})
        self.assertEqual(first_job, jobs.submit("run-cpu", {"image": "approved-image"}))
        self.assertEqual(len(provider.jobs_by_name), 1)


class GzipContentMeasureTests(unittest.TestCase):
    """The controller's acquisition measure accepts what the CPU worker's gzip reader
    accepts, over any read size, and refuses an incomplete or corrupt stream."""

    PLAIN = b">chr1\nACGTACGT\n>chr2\nTTTT\n"

    @staticmethod
    def _drain(content, size):
        import io

        from bio_neb.providers.gzip_content import GzipContentMeasure

        measure = GzipContentMeasure(io.BytesIO(content))
        passed = b""
        while chunk := measure.read(size):
            passed += chunk
        return passed, measure

    def test_members_and_zero_padding_measure_the_plain_content(self):
        import gzip

        first = gzip.compress(self.PLAIN[:12], mtime=0)
        second = gzip.compress(self.PLAIN[12:], mtime=0)
        expected = (len(self.PLAIN), hashlib.sha256(self.PLAIN).hexdigest())
        streams = {
            "one_member": gzip.compress(self.PLAIN, mtime=0),
            "two_members": first + second,
            "padding_after": first + second + b"\0" * 9,
            "padding_between": first + b"\0" * 5 + second,
        }
        for name, content in streams.items():
            for size in (1, 4, 1 << 20):
                with self.subTest(stream=name, read_size=size):
                    passed, measure = self._drain(content, size)
                    self.assertEqual(passed, content)
                    self.assertEqual(measure.identity(), expected)

    def test_excluded_records_are_left_out_of_the_measured_content(self):
        """Owner, #464 ("primey for the ref we are running"): the measured FASTA is the
        Run's reference sequences, as the CPU worker derives it; the stream passed on
        is the whole source."""
        import gzip
        import io

        from bio_neb.providers.gzip_content import GzipContentMeasure

        content = gzip.compress(self.PLAIN, mtime=0)
        selected = b">chr1\nACGTACGT\n"
        for size in (1, 7, 1 << 20):
            with self.subTest(read_size=size):
                measure = GzipContentMeasure(io.BytesIO(content), ("chr2",))
                passed = b""
                while chunk := measure.read(size):
                    passed += chunk
                self.assertEqual(passed, content)
                self.assertEqual(
                    measure.identity(),
                    (len(selected), hashlib.sha256(selected).hexdigest()),
                )

    def test_the_fake_store_reports_the_selected_content_of_a_streamed_source(self):
        """The fake store, like the Nebius one, measures the placed source's plain FASTA
        without the source's excluded records (owner, #464); it places the whole .gz."""
        import gzip

        content = gzip.compress(self.PLAIN, mtime=0)
        provider = FakeProvider()
        provider.sources["https://example.test/r.fa.gz"] = content
        receipt = provider.store.put(
            "bucket",
            "raw/r.fa.gz",
            TransferSource(
                "url",
                "https://example.test/r.fa.gz",
                len(content),
                None,
                md5=hashlib.md5(content).hexdigest(),
                measure_gzip_content=True,
                excluded_sequences=("chr2",),
            ),
        )
        selected = b">chr1\nACGTACGT\n"
        self.assertEqual(provider.objects["bucket", "raw/r.fa.gz"], content)
        self.assertEqual(
            receipt.values["decompressed"],
            {"bytes": len(selected), "sha256": hashlib.sha256(selected).hexdigest()},
        )

    def test_an_incomplete_or_corrupt_stream_is_not_gzip(self):
        import gzip

        member = gzip.compress(self.PLAIN, mtime=0)
        corrupt_crc = bytearray(member)
        corrupt_crc[-6] ^= 1
        streams = {
            "truncated_trailer": member[:-3],
            "crc_mismatch": bytes(corrupt_crc),
            "trailing_garbage": member + b"garbage",
            "leading_zeros": b"\0" * 4 + member,
            "not_gzip": self.PLAIN,
        }
        for name, content in streams.items():
            for size in (1, 1 << 20):
                with self.subTest(stream=name, read_size=size):
                    passed, measure = self._drain(content, size)
                    self.assertEqual(passed, content)
                    with self.assertRaises(ProviderError) as raised:
                        measure.identity()
                    self.assertEqual(
                        (raised.exception.failure_class, raised.exception.detail),
                        ("semantic", "source_not_gzip"),
                    )
