"""The Run's own outputs behind a controlled object store, for automatic supply.

An executed parent built with ``output_payloads={"VCF": RUN_VCF}`` holds this
verified VCF and the default BAM; ``RunOutputStore`` serves both through the real
part transfer (`NebiusObjectStorageMechanics.download`), over an in-memory bucket
that answers ranged GETs (#107, #363).
"""

from __future__ import annotations

import io

from tests.h100_stage_fakes import OUTPUTS

SAMPLE = "sample-001"
RUN_VCF = (
    b"##fileformat=VCFv4.2\n"
    b"##contig=<ID=NC_000019.10,length=58617616>\n"
    b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + SAMPLE.encode() + b"\n"
    b"NC_000019.10\t44908684\t.\tT\tC\t50\tPASS\t.\tGT\t0/1\n"
)
TABLE_READING_ENVIRON = {"LOCAL_MODEL_KEY": "secret-value-not-retained"}


def _credential_receipt(topology):
    from bio_neb.storage_contracts import ObjectStorageCredentialReuseReceipt

    return ObjectStorageCredentialReuseReceipt(
        service_account_id=topology.service_account_id,
        access_key_resource_id=topology.access_key_resource_id,
        access_key_id=topology.access_key_id,
        exact_topology_match_count=1,
        access_key_state="ACTIVE",
        secret_state="ACTIVE",
        secret_delivery_mode="mystery_box",
        host_secret_resolved=True,
        worker_secret_version_bound=True,
    )


# The `aligned_bam` an executed Run holds unless its `output_payloads` names one.
RUN_BAM = OUTPUTS["aligned_bam"]


class _ReceiptResolver:
    """The selected context's credential, as the real resolver reports it."""

    def __init__(self, topology) -> None:
        self.receipt = _credential_receipt(topology)

    def __call__(self):
        from bio_neb.providers.nebius_common import AwsCredentialMaterial

        return AwsCredentialMaterial(
            access_key_id="contact-only-access",
            secret_access_key="contact-only-secret",
            session_token=None,
            credentials_profile="object-storage-profile",
            worker_secret_selector="secret@version",
        )


class _ReportingResolver(_ReceiptResolver):
    """The resolver as it reports each call (`SelectedNebiusCredentialResolver`).

    Each call is one credential lookup through the shared fail-soft observer
    calls: the first resolves the selected context, later ones are cached.
    """

    def __init__(self, topology, observer) -> None:
        super().__init__(topology)
        self._observer = observer
        self._resolved = False

    def __call__(self):
        from bio_neb.providers.nebius_common import (
            observe_execution_contact_finish,
            observe_execution_contact_start,
        )

        unit_id = observe_execution_contact_start(
            self._observer,
            category="credential",
            operation="selected-object-storage-credential",
            executable_identity="credential-resolver",
            selected_identity="candidate",
        )
        outcome = "credential_cached" if self._resolved else "credential_resolved"
        self._resolved = True
        observe_execution_contact_finish(
            self._observer,
            unit_id,
            disposition="succeeded",
            parse_classification=outcome,
        )
        return super().__call__()


class ExecutionBucket:
    """An S3 client over exact objects; ``faults`` fail ranged GETs, in order.

    A fault is an exception instance the GET raises (a transport error is
    transient, a 403 is an access refusal), ``KeyboardInterrupt`` for the
    operator's interrupt, or ``None`` for a GET that succeeds.
    """

    def __init__(self, objects: dict[str, bytes], faults=()) -> None:
        self.objects = dict(objects)
        self.faults = list(faults)
        self.gets: list[tuple[str, str]] = []

    def get_object(self, *, Bucket, Key, Range):
        self.gets.append((Key, Range))
        fault = self.faults.pop(0) if self.faults else None
        if fault is not None:
            raise fault() if isinstance(fault, type) else fault
        body = self.objects[Key]
        start, end = (int(bound) for bound in Range.removeprefix("bytes=").split("-"))
        return {
            "Body": io.BytesIO(body[start : end + 1]),
            "ContentRange": f"bytes {start}-{end}/{len(body)}",
        }


def transient():
    from botocore.exceptions import EndpointConnectionError

    return EndpointConnectionError(endpoint_url="https://storage.example")


def access_refused():
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "ResponseMetadata": {"HTTPStatusCode": 403},
            "Error": {"Code": "AccessDenied"},
        },
        "GetObject",
    )


class RunOutputStore:
    """The Execution's bucket: its verified outputs' bytes, served in parts.

    ``vcf`` and ``bam`` are the objects' bytes (the parent's ``output_payloads``);
    ``gets`` lists each GET's object key, one per part attempt.
    """

    def __init__(
        self,
        faults=(),
        *,
        vcf: bytes = RUN_VCF,
        bam: bytes = RUN_BAM,
        part_bytes: int | None = None,
    ) -> None:
        self.faults = list(faults)
        self.payloads = {"VCF": vcf, "aligned_bam": bam}
        self.part_bytes = part_bytes
        self.bucket: ExecutionBucket | None = None
        self.packets = []

    def __call__(self, packet):
        return self._mechanics(
            packet, _ReceiptResolver(packet.selected_context.topology)
        )

    def execution_storage(self, packet, context_missing, credential_observer=None):
        """`execution_object_storage` over this bucket, reporting its lookups."""
        return self._mechanics(
            packet,
            _ReportingResolver(packet.selected_context.topology, credential_observer),
        )

    def _mechanics(self, packet, resolver):
        from bio_neb.providers.nebius_object_storage import (
            MULTIPART_PART_BYTES,
            NebiusObjectStorageMechanics,
        )

        self.packets.append(packet)
        if self.bucket is None:
            self.bucket = ExecutionBucket(
                {
                    item.object_key: self.payloads[item.role]
                    for item in packet.verified_objects
                    if item.role in self.payloads
                },
                self.faults,
            )
        bucket = self.bucket
        return NebiusObjectStorageMechanics(
            endpoint_url="https://storage.example",
            region="eu-north1",
            bucket=packet.storage.bucket,
            credential_resolver=resolver,
            client_factory=lambda endpoint, region, credentials: bucket,
            multipart_part_bytes=self.part_bytes or MULTIPART_PART_BYTES,
        )

    @property
    def gets(self) -> list[str]:
        return [] if self.bucket is None else [key for key, _ in self.bucket.gets]
