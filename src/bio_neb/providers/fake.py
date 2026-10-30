"""One in-memory provider with three narrow ports and contact-level scripts.

Scripts are consumed in order for each (stage, step). Once exhausted, normal
in-memory behavior resumes. Seeded named resources model an earlier successful
create whose response was lost; prechecks reuse them before any new mutation.
"""

import hashlib
import io
from collections import deque
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from bio_neb.failure import ProviderError
from bio_neb.providers.gzip_content import GzipContentMeasure
from bio_neb.providers.ports import (
    ContactReceipt,
    Ports,
    Receipt,
    TransferSource,
    VerificationMode,
)


class FakeProvider:
    def __init__(
        self,
        scripts: Mapping[tuple[str, str], Sequence[Receipt | ProviderError]]
        | None = None,
        *,
        sources: Mapping[str, bytes] | None = None,
    ):
        self.scripts = {key: deque(items) for key, items in (scripts or {}).items()}
        self.buckets_by_name: dict[str, Receipt] = {}
        self.jobs_by_name: dict[str, Receipt] = {}
        self.objects: dict[tuple[str, str], bytes] = {}
        self.sources = dict(sources or {})
        ports = self.for_step("", "")
        self.store, self.jobs, self.identity = ports.store, ports.jobs, ports.identity

    def for_step(self, stage: str, step: str) -> Ports:
        script = self.scripts.setdefault((stage, step), deque())
        return Ports(_Store(self, script), _Jobs(self, script), _Identity(script))


class _Scripted:
    def __init__(self, script):
        self.script = script

    def scripted(self) -> Receipt | None:
        if not self.script:
            return None
        result = self.script.popleft()
        if isinstance(result, ProviderError):
            raise result
        return result


class _Store(_Scripted):
    def __init__(self, provider, script):
        super().__init__(script)
        self.provider = provider

    def fresh_bucket(self, name: str) -> Receipt:
        if name not in self.provider.buckets_by_name:
            receipt = self.scripted() or Receipt(
                values={"name": name}, provider_ids=(name,), resource_count=1
            )
            self.provider.buckets_by_name[name] = receipt
        return self.provider.buckets_by_name[name]

    def put(self, bucket: str, key: str, content: bytes | TransferSource) -> Receipt:
        expected = source = None
        receipt = self.scripted()
        if receipt is not None:
            return receipt
        if isinstance(content, TransferSource):
            source = content
            if source.sha256 is None and (bucket, key) in self.provider.objects:
                raise ProviderError("rejected", "object_content_conflict", {"key": key})
            expected = (
                {"byte_count": source.byte_count, "sha256": source.sha256}
                if source.sha256 is not None
                else None
            )
            content = self.provider.sources.get(source.locator)
            if content is None and source.kind == "local":
                # A local file the Run registered, such as a supplied FASTA.
                from pathlib import Path

                if Path(source.locator).is_file():
                    content = Path(source.locator).read_bytes()
            if content is None and source.kind == "url":
                # As the real adapter answers a URL that serves nothing (HTTP 404).
                raise ProviderError(
                    "rejected", "source_http_unavailable", {"http_status": 404}
                )
            if content is None:
                raise ProviderError("rejected", "offline_source_missing")
            if (
                len(content) != source.byte_count
                or (
                    source.sha256 is not None
                    and hashlib.sha256(content).hexdigest() != source.sha256
                )
                or (
                    source.md5 is not None
                    and hashlib.md5(content).hexdigest() != source.md5
                )
            ):
                raise ProviderError("semantic", "source_identity_mismatch")
        renamed_from = None
        if isinstance(source, TransferSource) and source.transform is not None:
            # The source's names rewritten in flight.
            renamed_from = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            content = source.transform.whole(content)
            placed = source.transform.placed_byte_count
            if placed is not None and len(content) != placed:
                raise ProviderError("semantic", "source_identity_mismatch")
        if (bucket, key) in self.provider.objects:
            if self.provider.objects[bucket, key] != content:
                raise ProviderError("rejected", "object_content_conflict", {"key": key})
            return self._metadata(bucket, key, expected=expected)
        measured = None
        if isinstance(source, TransferSource) and source.measure_gzip_content:
            measure = GzipContentMeasure(io.BytesIO(content), source.excluded_sequences)
            while measure.read(1 << 20):
                pass
            measured = measure.identity()
        self.provider.objects[bucket, key] = content
        receipt = self._metadata(bucket, key, expected=expected)
        if renamed_from is not None:
            return receipt.model_copy(
                update={"values": {**receipt.values, "source": renamed_from}}
            )
        if measured is None:
            return receipt
        return receipt.model_copy(
            update={
                "values": {
                    **receipt.values,
                    "md5": source.md5,
                    "decompressed": {"bytes": measured[0], "sha256": measured[1]},
                }
            }
        )

    def copy(self, source, bucket: str, key: str) -> Receipt:
        from bio_neb.providers.object_locations import object_address

        receipt = self.scripted()
        if receipt is not None:
            return receipt
        source_bucket, source_key = object_address(source.location)
        content = self._content(source_bucket, source_key)
        placed = self.put(bucket, key, content)
        proof = source.model_copy(
            update={
                "location": f"s3://{bucket}/{key}",
                "producer": "controlled_copy",
                "expected_identity": None,
                "source_proof": source.proof_id,
            }
        )
        return placed.model_copy(
            update={
                "values": {
                    "bytes": source.identity.byte_count,
                    "sha256": source.identity.sha256,
                    "artifact_proof": proof.model_dump(mode="json"),
                    "source_proof": source.model_dump(mode="json"),
                }
            }
        )

    def _content(self, bucket: str, key: str) -> bytes:
        try:
            return self.provider.objects[bucket, key]
        except KeyError:
            raise ProviderError("rejected", "object_missing", {"key": key}) from None

    def _metadata(self, bucket: str, key: str, *, expected=None) -> Receipt:
        from bio_neb.verified_artifacts import ArtifactIdentity, ArtifactProof

        content = self._content(bucket, key)
        proof = ArtifactProof(
            identity=ArtifactIdentity(
                byte_count=len(content), sha256=hashlib.sha256(content).hexdigest()
            ),
            location=f"s3://{bucket}/{key}",
            producer="offline_storage",
            expected_identity=expected,
        )
        return Receipt(
            values={
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "artifact_proof": proof.model_dump(mode="json"),
            },
            provider_ids=(f"s3://{bucket}/{key}",),
            contact=ContactReceipt(kind="transfer_proof", ref=f"s3://{bucket}/{key}"),
        )

    def head(
        self,
        bucket: str,
        key: str,
        *,
        verification: VerificationMode = "reuse_creation_proof",
    ) -> Receipt:
        receipt = self.scripted() or self._metadata(bucket, key)
        return receipt.model_copy(
            update={"values": {**receipt.values, "verification_mode": verification}}
        )

    def read_small(self, bucket: str, key: str) -> Receipt:
        return self.scripted() or self._metadata(bucket, key).model_copy(
            update={"content": self._content(bucket, key)}
        )


class _Jobs(_Scripted):
    def __init__(self, provider, script):
        super().__init__(script)
        self.provider = provider

    def submit(self, name: str, request: dict[str, JsonValue]) -> Receipt:
        if name not in self.provider.jobs_by_name:
            receipt = self.scripted() or Receipt(
                values={"name": name},
                provider_ids=(f"job:{name}",),
                resource_count=1,
                contact=ContactReceipt(kind="job_submission", ref=f"job:{name}"),
            )
            self.provider.jobs_by_name[name] = receipt
        return self.provider.jobs_by_name[name]

    def observe(self, job_id: str, *, window: int) -> Receipt:
        return self.scripted() or Receipt(
            values={"state": "COMPLETED", "lines": []}, provider_ids=(job_id,)
        )


class _Identity(_Scripted):
    def whoami(self) -> Receipt:
        return self.scripted() or Receipt(values={"identity": "offline-fake"})
