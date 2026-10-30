"""Deterministic preparation and one-target observation for an attested VCF."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from subprocess import CalledProcessError, TimeoutExpired
from typing import (
    Annotated,
    Any,
    Literal,
    Mapping,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from bio_neb.application.local_artifact_proof import ArtifactVerificationCheck
from bio_neb.contracts import FrozenContract
from bio_neb.failure import (
    FailureSurface,
    ProviderError,
    call_with_failure_policy,
    policy_for,
)
from bio_neb.investigation.observation_defaults import VcfLookupDefaults
from bio_neb.journey import canonical_json_digest
from bio_neb.verified_artifacts import VerifiedObject


def command_safe_contig(name: str) -> bool:
    """A contig name a bcftools regions file can carry as one field: non-empty, with no
    whitespace or control character (VCF's CHROM has none). Command safety only; the
    name's shape never judges a sequence (#467, ADR 0049)."""
    return bool(name) and not any(
        character.isspace() or not character.isprintable() for character in name
    )


# A lookup target's allele, as a Work item's target admits it: N bases, and an
# empty side for an insertion or deletion.
_TARGET_ALLELE = re.compile(r"^[ACGTN]*$")
# Only called bases compare: ``*``, symbolic and N-bearing VCF alleles never match.
_CALLED_ALLELE = re.compile(r"^[ACGT]+$")
_QUERY_FORMAT = (
    r"%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%QUAL[\t%SAMPLE\t%GT\t%GQ\t%DP\t%AD]\n"
)

VcfObservationAuthorityKind = Literal[
    "scientific_continuation",
    "investigation_authority",
    "non_authoritative_mechanics",
]


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class LocalFileObservation(FrozenContract):
    path: StrictStr
    exists: StrictBool
    is_regular: StrictBool
    is_symlink: StrictBool
    is_writable: StrictBool
    is_executable: StrictBool
    byte_count: StrictInt | None
    sha256: StrictStr | None
    device: StrictStr | None
    inode: StrictStr | None
    modified_at_ns: StrictInt | None
    verification: ArtifactVerificationCheck | None = None

    @model_serializer(mode="wrap")
    def optional_verification(self, handler: Any) -> dict[str, Any]:
        document = handler(self)
        if self.verification is None:
            document.pop("verification", None)
        return document

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not os.path.isabs(value):
            raise ValueError("observed path must be absolute")
        return value

    @model_validator(mode="after")
    def coherent_state(self) -> "LocalFileObservation":
        if self.verification is not None and (
            not self.exists
            or self.verification.bytes_read != self.byte_count
            or self.verification.path != self.path
            or (
                self.verification.reason == "controlled_proof_reuse"
                and self.verification.policy.effective.mode != "reuse_creation_proof"
            )
        ):
            raise ValueError("verification measurement changed consumed byte count")
        retained = (
            self.byte_count,
            self.sha256,
            self.device,
            self.inode,
            self.modified_at_ns,
        )
        if not self.exists:
            if (
                self.is_regular
                or self.is_symlink
                or self.is_writable
                or self.is_executable
                or any(item is not None for item in retained)
            ):
                raise ValueError("missing file contains fabricated identity")
        elif (
            not self.is_regular
            or self.is_symlink
            or any(item is None for item in retained)
            or self.byte_count < 0
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256 or "")
        ):
            raise ValueError("existing file lacks a safe complete identity")
        return self


@dataclass(frozen=True)
class CompletedProcess:
    return_code: int
    stdout: bytes
    stderr: bytes


@runtime_checkable
class VcfProcessAdapter(Protocol):
    """Perform only application-compiled local file and process mechanics."""

    def inspect_file(self, path: str) -> LocalFileObservation: ...

    def path_exists(self, path: str) -> bool: ...

    def create_fresh_directory(self, path: str) -> None: ...

    def write_fresh_file(self, path: str, data: bytes) -> None:
        """Write ``data`` to a file at ``path`` that does not yet exist, visible at
        ``path`` only once whole: a failed write leaves nothing there, so an existing
        path is this adapter's complete creation (#467 Standards r1 ST1, ADR 0029)."""
        ...

    def run(
        self,
        argv: Tuple[str, ...],
        *,
        timeout_seconds: int,
        stdout_path: str | None = None,
    ) -> CompletedProcess:
        """Run argv; with ``stdout_path``, write stdout to that fresh file instead."""
        ...


@runtime_checkable
class VcfObservationReference(Protocol):
    assembly_accession: str


@runtime_checkable
class VcfObservationSample(Protocol):
    sample_id: str


@runtime_checkable
class VcfObservationAuthority(Protocol):
    """Minimum read-only identity surface consumed by the deep VCF Tool."""

    continuation_id: str
    journey_id: str | None
    reference: VcfObservationReference
    sample: VcfObservationSample
    vcf: VerifiedObject


class AttestedPlainVcf(FrozenContract):
    local_path: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    acquisition_receipt_path: StrictStr
    acquisition_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("local_path", "acquisition_receipt_path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not os.path.isabs(value):
            raise ValueError("attested VCF path must be absolute")
        return value


class BcftoolsRuntimeBinding(FrozenContract):
    executable_path: StrictStr
    executable_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    bcftools_version: StrictStr
    htslib_version: StrictStr
    probe_receipt_path: StrictStr
    probe_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("executable_path", "probe_receipt_path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not os.path.isabs(value):
            raise ValueError("bcftools executable path must be absolute")
        return value

    @field_validator("bcftools_version", "htslib_version")
    @classmethod
    def canonical_version(cls, value: str) -> str:
        if not value or value.strip() != value or any(ch.isspace() for ch in value):
            raise ValueError("runtime version must be one canonical token")
        return value


class VcfMaterializationBinding(FrozenContract):
    attested_vcf: AttestedPlainVcf
    runtime: BcftoolsRuntimeBinding
    derivation_root: StrictStr
    continuation_sample_id: StrictStr
    expected_vcf_sample_names: Tuple[StrictStr, ...] = Field(min_length=1)

    @field_validator("derivation_root")
    @classmethod
    def absolute_root(cls, value: str) -> str:
        if not os.path.isabs(value):
            raise ValueError("VCF derivation root must be absolute")
        return value

    @field_validator("continuation_sample_id")
    @classmethod
    def canonical_sample(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("continuation sample identity must be canonical")
        return value

    @field_validator("expected_vcf_sample_names")
    @classmethod
    def unique_samples(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if any(not item or item.strip() != item for item in value):
            raise ValueError("VCF sample names must be canonical")
        if len(value) != len(set(value)):
            raise ValueError("VCF sample names must be unique")
        return value

    @model_validator(mode="after")
    def exact_continuation_sample(self) -> "VcfMaterializationBinding":
        if self.expected_vcf_sample_names != (self.continuation_sample_id,):
            raise ValueError("VCF sample must exactly match the continuation sample")
        return self


InvocationPurpose = Literal[
    "bgzf_derivation",
    "csi_derivation",
    "header_inventory",
    "variant_query",
]


class RegionsFile(FrozenContract):
    """The regions file a bcftools query reads (#467, owner Q3 a): its path, its text
    (``CHROM<TAB>BEG<TAB>END``, one-based and inclusive) and that text's sha256, so a
    reopen checks the region as the command read it."""

    path: StrictStr = Field(min_length=1)
    text: StrictStr = Field(min_length=1)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_text(self) -> "RegionsFile":
        if not os.path.isabs(self.path) or self.sha256 != _bytes_sha256(
            self.text.encode()
        ):
            raise ValueError("a regions file names its absolute path and its text")
        return self


class ProcessInvocationReceipt(FrozenContract):
    purpose: InvocationPurpose
    argv: Tuple[StrictStr, ...] = Field(min_length=2)
    return_code: StrictInt
    stdout_byte_count: StrictInt = Field(ge=0)
    stdout_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_byte_count: StrictInt = Field(ge=0)
    stderr_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    # A variant query's regions file; no other command reads one.
    regions_file: RegionsFile | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class VcfContigInventoryEntry(FrozenContract):
    """One usable ``##contig`` header line: a safe exact name and its length."""

    name: StrictStr
    length: StrictInt = Field(gt=0)

    @field_validator("name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        if not command_safe_contig(value):
            raise ValueError("indexed VCF contig identity is not safe")
        return value


class SkippedVcfContig(FrozenContract):
    """A ``##contig`` header line the lookup cannot use; other contigs still run."""

    contig_id: StrictStr = Field(min_length=1)
    reason: Literal[
        "missing_id", "unsafe_name", "missing_length", "invalid_length", "duplicate"
    ]


def _structured_header_fields(body: str) -> dict[str, str]:
    """Split a ``##key=<...>`` body on commas outside double quotes."""
    fields: dict[str, str] = {}
    key, value, quoted, in_value = "", "", False, False
    for character in body + ",":
        if character == '"':
            quoted = not quoted
            value += character
        elif character == "," and not quoted:
            if key:
                fields.setdefault(key, value)
            key, value, in_value = "", "", False
        elif character == "=" and not in_value:
            in_value = True
        elif in_value:
            value += character
        else:
            key += character
    return fields


def _header_inventory(
    header: str,
) -> tuple[
    Tuple[str, ...], Tuple[VcfContigInventoryEntry, ...], Tuple[SkippedVcfContig, ...]
]:
    """Read the sample columns and contig inventory a VCF header declares."""
    samples: Tuple[str, ...] = ()
    contigs: list[VcfContigInventoryEntry] = []
    skipped: list[SkippedVcfContig] = []
    seen: set[str] = set()
    for line in header.splitlines():
        if line.startswith("#CHROM\t"):
            columns = line.split("\t")
            samples = tuple(columns[9:]) if columns[8:9] == ["FORMAT"] else ()
            continue
        if not (line.startswith("##contig=<") and line.endswith(">")):
            continue
        fields = _structured_header_fields(line[len("##contig=<") : -1])
        name = fields.get("ID", "")
        raw_length = fields.get("length")
        reason: str | None = None
        if not name:
            name, reason = line, "missing_id"
        elif name in seen:
            reason = "duplicate"
        elif not command_safe_contig(name):
            reason = "unsafe_name"
        elif raw_length is None:
            reason = "missing_length"
        elif (
            not raw_length.isascii() or not raw_length.isdigit() or int(raw_length) < 1
        ):
            reason = "invalid_length"
        seen.add(name)
        if reason is None:
            contigs.append(VcfContigInventoryEntry(name=name, length=int(raw_length)))
        else:
            skipped.append(SkippedVcfContig(contig_id=name, reason=reason))
    return samples, tuple(contigs), tuple(skipped)


def _validate_authority_id_kind(
    authority_kind: VcfObservationAuthorityKind,
    continuation_id: str,
) -> None:
    scientific_id = re.fullmatch(
        r"scientific-continuation:[0-9a-f]{64}", continuation_id
    )
    investigation_id = re.fullmatch(
        r"investigation-authority:[0-9a-f]{64}", continuation_id
    )
    mechanics_id = re.fullmatch(
        r"vcf-observation-mechanics:[0-9a-f]{64}", continuation_id
    )
    if authority_kind == "scientific_continuation":
        if scientific_id is None:
            raise ValueError("scientific authority lacks continuation identity")
    elif authority_kind == "investigation_authority":
        if investigation_id is None:
            raise ValueError("Investigation authority lacks authority identity")
    elif mechanics_id is None:
        raise ValueError("mechanics authority fabricated continuation identity")


def _validate_authority_identity(
    authority_kind: VcfObservationAuthorityKind,
    continuation_id: str,
    journey_id: str | None,
) -> None:
    _validate_authority_id_kind(authority_kind, continuation_id)
    journey_identity = (
        isinstance(journey_id, str)
        and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            journey_id,
        )
        is not None
    )
    if authority_kind in {"scientific_continuation", "investigation_authority"}:
        if not journey_identity:
            raise ValueError("scientific authority lacks Journey identity")
    elif journey_id is not None:
        raise ValueError("mechanics authority fabricated continuation/Journey identity")


class PreparedVcfObservationReceipt(FrozenContract):
    """One trusted preparation: the derived BGZF/CSI and the header inventory.

    The attested VCF and the executable are trusted under 0029; the attested VCF is
    read only when an independent full read is selected (``original``).
    """

    status: Literal["succeeded"] = "succeeded"
    schema_version: Literal["bio-neb.vcf-observation-preparation/2"] = (
        "bio-neb.vcf-observation-preparation/2"
    )
    receipt_id: StrictStr = Field(pattern=r"^vcf-observation-preparation:[0-9a-f]{64}$")
    authority_kind: VcfObservationAuthorityKind
    continuation_id: StrictStr = Field(
        pattern=(
            r"^(?:scientific-continuation|investigation-authority|vcf-observation-mechanics):[0-9a-f]{64}$"
        )
    )
    journey_id: StrictStr | None = None
    assembly: StrictStr
    continuation_sample_id: StrictStr
    vcf_descriptor: VerifiedObject
    acquisition_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    acquisition_receipt_file: LocalFileObservation
    original: LocalFileObservation | None = None
    runtime: BcftoolsRuntimeBinding
    probe_receipt_file: LocalFileObservation
    derived_bgzf: LocalFileObservation
    derived_csi: LocalFileObservation
    vcf_sample_names: Tuple[StrictStr, ...] = Field(min_length=1)
    vcf_contigs: Tuple[VcfContigInventoryEntry, ...] = Field(min_length=1)
    skipped_contigs: Tuple[SkippedVcfContig, ...] = ()
    invocations: Tuple[
        ProcessInvocationReceipt,
        ProcessInvocationReceipt,
        ProcessInvocationReceipt,
    ]

    @model_serializer(mode="wrap")
    def optional_evidence(self, handler: Any) -> dict[str, Any]:
        document = handler(self)
        if self.original is None:
            document.pop("original", None)
        if not self.skipped_contigs:
            document.pop("skipped_contigs", None)
        return document

    @model_validator(mode="after")
    def exact_receipt(self) -> "PreparedVcfObservationReceipt":
        _validate_authority_identity(
            self.authority_kind, self.continuation_id, self.journey_id
        )
        if self.vcf_descriptor.role != "VCF":
            raise ValueError("preparation receipt lacks the VCF descriptor")
        derivation, index, header = self.invocations
        attested_path = derivation.argv[-1]
        if self.original is not None and (
            not self.original.exists
            or self.original.path != attested_path
            or self.original.byte_count != self.vcf_descriptor.byte_count
            or self.original.sha256 != self.vcf_descriptor.sha256
        ):
            raise ValueError("preparation receipt changed the attested VCF read")
        if (
            self.acquisition_receipt_file.sha256 != self.acquisition_receipt_sha256
            or self.probe_receipt_file.path != self.runtime.probe_receipt_path
            or self.probe_receipt_file.sha256 != self.runtime.probe_receipt_sha256
            or not self.derived_bgzf.exists
            or not self.derived_bgzf.byte_count
            or not self.derived_csi.exists
            or not self.derived_csi.byte_count
            or self.derived_csi.path != f"{self.derived_bgzf.path}.csi"
            or os.path.basename(self.derived_bgzf.path) != "variants.vcf.gz"
            or derivation.stdout_byte_count != self.derived_bgzf.byte_count
            or derivation.stdout_sha256 != self.derived_bgzf.sha256
        ):
            raise ValueError("preparation receipt contains incoherent file identity")
        usable = {item.name for item in self.vcf_contigs}
        if (
            self.vcf_sample_names != (self.continuation_sample_id,)
            or len(self.vcf_contigs) != len(usable)
            or any(
                item.contig_id in usable and item.reason != "duplicate"
                for item in self.skipped_contigs
            )
        ):
            raise ValueError("preparation receipt contains invalid VCF inventory")
        if tuple(item.purpose for item in self.invocations) != (
            "bgzf_derivation",
            "csi_derivation",
            "header_inventory",
        ):
            raise ValueError("preparation invocation order changed")
        executable = self.runtime.executable_path
        if (
            derivation.argv
            != (executable, "view", "--no-version", "--output-type", "z", attested_path)
            or not os.path.isabs(attested_path)
            or index.argv != (executable, "index", "--csi", self.derived_bgzf.path)
            or header.argv
            != (
                executable,
                "view",
                "--no-version",
                "--header-only",
                self.derived_bgzf.path,
            )
            or any(item.return_code != 0 for item in self.invocations)
        ):
            raise ValueError("preparation receipt command identity changed")
        material = self.model_dump(mode="json", exclude={"receipt_id"})
        if (
            self.receipt_id
            != f"vcf-observation-preparation:{canonical_json_digest(material)}"
        ):
            raise ValueError("preparation receipt identity changed")
        return self


VcfPreparationFailureCode = Literal[
    "binding_invalid",
    "input_unavailable",
    "original_identity_mismatch",
    "runtime_unavailable",
    "runtime_identity_mismatch",
    "derivation_target_exists",
    "tool_failure",
    "derived_output_invalid",
]


class VcfPreparationFailed(FrozenContract):
    status: Literal["failed"] = "failed"
    code: VcfPreparationFailureCode
    details: Tuple[StrictStr, ...] = ()
    invocations: Tuple[ProcessInvocationReceipt, ...] = ()


class ObserveVariantInput(FrozenContract):
    assembly: StrictStr
    contig: StrictStr
    one_based_position: StrictInt = Field(gt=0)
    ref: StrictStr
    alt: StrictStr
    display_label: StrictStr | None = None

    @field_validator("assembly")
    @classmethod
    def canonical_assembly(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("assembly must be a canonical non-empty identity")
        return value

    @field_validator("contig")
    @classmethod
    def safe_contig(cls, value: str) -> str:
        if not command_safe_contig(value):
            raise ValueError("contig is not a safe exact VCF identity")
        return value

    @field_validator("ref", "alt")
    @classmethod
    def lexical_allele(cls, value: str) -> str:
        if not _TARGET_ALLELE.fullmatch(value):
            raise ValueError("allele must contain uppercase A/C/G/T/N only")
        return value

    @field_validator("display_label")
    @classmethod
    def bounded_label(cls, value: str | None) -> str | None:
        if value is not None and (
            not value or value.strip() != value or len(value) > 200
        ):
            raise ValueError("display label must be canonical and bounded")
        return value

    @model_validator(mode="after")
    def distinct_alleles(self) -> "ObserveVariantInput":
        if self.ref == self.alt:
            raise ValueError("REF and ALT must differ")
        return self


class ObserveRegionInput(FrozenContract):
    """One region Work item's stretch of the Run's sequence and the Run's reference
    bases over it (#367): every record in it is read, and each REF is checked."""

    assembly: StrictStr
    contig: StrictStr
    start: StrictInt = Field(gt=0)
    end: StrictInt = Field(gt=0)
    reference_bases: StrictStr = Field(pattern=r"^[ACGTN]+$")

    @field_validator("contig")
    @classmethod
    def safe_contig(cls, value: str) -> str:
        if not command_safe_contig(value):
            raise ValueError("contig is not a safe exact VCF identity")
        return value

    @model_validator(mode="after")
    def spans_the_region(self) -> "ObserveRegionInput":
        if self.start > self.end or len(self.reference_bases) != (
            self.end - self.start + 1
        ):
            raise ValueError("a region's reference bases span it")
        return self


class SampleGenotype(FrozenContract):
    """One sample's call as the VCF wrote it; quality is recorded, never gated."""

    sample_name: StrictStr
    genotype: StrictStr
    gq: StrictInt | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    dp: StrictInt | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    ad: Tuple[StrictInt | None, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def canonical_genotype(self) -> "SampleGenotype":
        if (
            not self.sample_name
            or self.sample_name.strip() != self.sample_name
            or not self.genotype
            or self.genotype.strip() != self.genotype
            or any(item is not None and item < 0 for item in self.ad or ())
        ):
            raise ValueError("sample genotype must be canonical")
        return self


GenotypeState = Literal[
    "target_alt_carried",
    "target_alt_not_carried",
    "no_call",
    "invalid",
]


def _states_no_phase(raw_genotype: str) -> bool:
    """A one-allele call, or one whose alleles carry both separators (VCF 4.3)."""
    return ("/" in raw_genotype) == ("|" in raw_genotype)


class TargetGenotypeAssessment(FrozenContract):
    """The sample's genotype read against the target ALT's index in its record.

    ``target_alt_index`` is ``None`` when the record at the target lists other
    ALTs only; a fully called genotype then does not carry the target ALT.
    """

    sample_name: StrictStr
    raw_genotype: StrictStr
    target_alt_index: StrictInt | None = Field(gt=0)
    state: GenotypeState
    phased: StrictBool | None
    allele_indexes: Tuple[StrictInt | None, ...] = Field(min_length=1)
    reason: Literal["missing_allele", "malformed", "out_of_range"] | None = None

    @model_validator(mode="after")
    def coherent_assessment(self) -> "TargetGenotypeAssessment":
        if self.state in {"target_alt_carried", "target_alt_not_carried"}:
            if (
                self.reason is not None
                or (self.phased is None and not _states_no_phase(self.raw_genotype))
                or any(item is None for item in self.allele_indexes)
            ):
                raise ValueError("resolved genotype has unresolved evidence")
            carried = (
                self.target_alt_index is not None
                and self.target_alt_index in self.allele_indexes
            )
            if carried != (self.state == "target_alt_carried"):
                raise ValueError("genotype state contradicts target ALT index")
        elif self.state == "no_call":
            if self.reason != "missing_allele" or all(
                item is not None for item in self.allele_indexes
            ):
                raise ValueError("no-call genotype lacks a missing allele")
        elif self.reason not in {"malformed", "out_of_range"}:
            raise ValueError("invalid genotype lacks its closed reason")
        return self


class ObservedVcfRecord(FrozenContract):
    """One VCF record as written, with FILTER and QUAL recorded, never gated.

    A record may start upstream of the target when it spans it.
    """

    contig: StrictStr
    one_based_position: StrictInt = Field(gt=0)
    ref: StrictStr
    alts: Tuple[StrictStr, ...] = Field(min_length=1)
    filters: Tuple[StrictStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    qual: StrictStr | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    genotypes: Tuple[SampleGenotype, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_record(self) -> "ObservedVcfRecord":
        if (
            not self.contig
            or not self.ref
            or len(self.alts) != len(set(self.alts))
            or any(not item for item in self.alts)
            or any(not item or ";" in item for item in self.filters)
            or len(self.genotypes) != len({item.sample_name for item in self.genotypes})
        ):
            raise ValueError("VCF record identity is incoherent")
        return self


EMPTY_ALLELE = "-"


def allele_text(ref: str, alt: str) -> str:
    """``REF>ALT`` as a researcher reads it, with ``-`` for an empty side."""
    return f"{ref or EMPTY_ALLELE}>{alt or EMPTY_ALLELE}"


def trimmed_allele(position: int, ref: str, alt: str) -> tuple[int, str, str]:
    """One allele without its shared bases: the suffix first, then the prefix.

    The result starts at the first changed base; an insertion has an empty REF and
    a deletion an empty ALT. Trimming the suffix first left-aligns a
    reference-expanded allele (dbSNP SPDI) and drops the anchor base of an
    anchored, left-aligned VCF allele, so both reach the same form (#368).
    """
    while ref and alt and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while ref and alt and ref[0] == alt[0]:
        ref, alt, position = ref[1:], alt[1:], position + 1
    return position, ref, alt


def _normalized_allele(
    position: int, ref: str, alt: str
) -> tuple[int, str, str] | None:
    """One record ALT in a target's form; ``*``, symbolic and N-bearing never match."""
    if not _CALLED_ALLELE.fullmatch(ref) or not _CALLED_ALLELE.fullmatch(alt):
        return None
    return trimmed_allele(position, ref, alt)


def _called_target(target: ObserveVariantInput) -> bool:
    """Only a target of called bases can be read as carried or not carried."""
    return "N" not in target.ref + target.alt


def _target_span(target: ObserveVariantInput) -> tuple[int, int]:
    """The one-based positions a target's VCF record must reach.

    A substitution or deletion spans its REF; an insertion (empty REF) sits after
    the anchor base the VCF writes before it (the first base at position 1).
    """
    position = target.one_based_position
    if not target.ref:
        anchor = max(1, position - 1)
        return anchor, anchor
    return position, position + len(target.ref) - 1


def _target_alt_index(
    record: ObservedVcfRecord, target: ObserveVariantInput
) -> int | None:
    """The 1-based ALT of this record that is the target once split and trimmed."""
    wanted = (target.one_based_position, target.ref, target.alt)
    return next(
        (
            index
            for index, alt in enumerate(record.alts, start=1)
            if _normalized_allele(record.one_based_position, record.ref, alt) == wanted
        ),
        None,
    )


def _covers_locus(record: ObservedVcfRecord, target: ObserveVariantInput) -> bool:
    """A record at the target: it starts inside the target's span, or one of its
    trimmed ALTs starts at the target or inside its REF.

    Any other returned record starts upstream and only reaches the span.
    """
    first, last = _target_span(target)
    position = target.one_based_position
    end = max(position, position + len(target.ref) - 1)
    return first <= record.one_based_position <= last or any(
        normalized is not None and position <= normalized[0] <= end
        for normalized in (
            _normalized_allele(record.one_based_position, record.ref, alt)
            for alt in record.alts
        )
    )


def _reference_at_target(
    record: ObservedVcfRecord, target: ObserveVariantInput
) -> str | None:
    """The record's REF bases where the target's REF sits, if it holds them all."""
    offset = target.one_based_position - record.one_based_position
    end = offset + len(target.ref)
    return record.ref[offset:end] if 0 <= offset and end <= len(record.ref) else None


def _answers_absence(record: ObservedVcfRecord, target: ObserveVariantInput) -> bool:
    """The record can say a fully called sample lacks the target's ALT.

    It holds the target's REF bases where the target sits, and every ALT has a
    known sequence (called bases, or ``*`` beside a substitution or deletion), so
    none could be the target's. For an insertion, whose REF is empty, it also lists
    another insertion at the same point.
    """
    if not _called_target(target) or _reference_at_target(record, target) != target.ref:
        return False
    known = {"*"} if target.ref else set()
    if any(
        alt not in known and not _CALLED_ALLELE.fullmatch(alt) for alt in record.alts
    ):
        return False
    return bool(target.ref) or any(
        normalized is not None
        and normalized[0] == target.one_based_position
        and not normalized[1]
        for normalized in (
            _normalized_allele(record.one_based_position, record.ref, alt)
            for alt in record.alts
        )
    )


def _reaches(record: ObservedVcfRecord, first: int, last: int) -> bool:
    """The record overlaps the one-based positions ``first`` to ``last``.

    A REF states its own span. A symbolic ALT (``<DEL>``) states its span in
    ``INFO/END``, which the lookup does not read; its upstream record stands on
    bcftools' region selection and never matches a target.
    """
    return record.one_based_position <= last and (
        record.one_based_position + len(record.ref) - 1 >= first
        or any(alt.startswith("<") for alt in record.alts)
    )


def _validate_record_locus(
    record: ObservedVcfRecord, target: ObserveVariantInput
) -> None:
    # A record starting upstream counts only when its REF spans the target.
    if record.contig != target.contig or not _reaches(record, *_target_span(target)):
        raise ValueError("observation record changed requested locus")


def _regions(contig: str, start: int, end: int) -> str:
    """One region as a bcftools regions file writes it: tab-separated, one-based and
    inclusive (the file's name does not end ``.bed``). A file carries any contig name,
    which a ``--regions`` string's own ``:`` and ``,`` syntax cannot (#467)."""
    return f"{contig}\t{start}\t{end}\n"


def _point_region(target: "ObserveVariantInput | ObserveRegionInput") -> str:
    if isinstance(target, ObserveRegionInput):
        return _regions(target.contig, target.start, target.end)
    first, last = _target_span(target)
    return _regions(target.contig, first, last)


def _query_argv(executable: str, regions_path: str, bgzf_path: str) -> tuple[str, ...]:
    return (
        executable,
        "query",
        "--allow-undef-tags",
        "--regions-file",
        regions_path,
        "--format",
        _QUERY_FORMAT,
        bgzf_path,
    )


def _regions_file(
    adapter: "VcfProcessAdapter", bgzf_path: str, text: str
) -> RegionsFile:
    """The query's regions file beside the derived VCF, named by its text's digest, so
    the same region is one file however often it is queried. The adapter publishes a
    file only once whole, so an existing one is that text's complete creation."""
    digest = _bytes_sha256(text.encode())
    path = os.path.join(os.path.dirname(bgzf_path), f"regions-{digest[:16]}.tsv")
    if not adapter.path_exists(path):
        adapter.write_fresh_file(path, text.encode())
    return RegionsFile(path=path, text=text, sha256=digest)


def _validate_query_invocation(
    invocation: ProcessInvocationReceipt,
    region: str,
    *,
    require_zero: bool,
) -> None:
    argv = invocation.argv
    regions = invocation.regions_file
    if (
        invocation.purpose != "variant_query"
        or regions is None
        or regions.text != region
        or len(argv) != 8
        or not os.path.isabs(argv[0])
        or argv != _query_argv(argv[0], regions.path, argv[7])
        or not os.path.isabs(argv[7])
        or (require_zero and invocation.return_code != 0)
    ):
        raise ValueError("variant query invocation changed fixed command identity")


def _validate_query_output(
    invocation: ProcessInvocationReceipt, *, expect_empty: bool
) -> None:
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    is_empty = (
        invocation.stdout_byte_count == 0 and invocation.stdout_sha256 == empty_sha256
    )
    if is_empty != expect_empty:
        expected = "empty" if expect_empty else "non-empty"
        raise ValueError(f"variant query receipt does not bind {expected} output")


def _validate_records(
    records: Tuple[ObservedVcfRecord, ...], target: ObserveVariantInput
) -> None:
    for record in records:
        _validate_record_locus(record, target)


class _ObservationBase(FrozenContract):
    schema_version: Literal["bio-neb.variant-observation/2"] = (
        "bio-neb.variant-observation/2"
    )
    authority_kind: VcfObservationAuthorityKind
    continuation_id: StrictStr = Field(
        pattern=(
            r"^(?:scientific-continuation|investigation-authority|vcf-observation-mechanics):[0-9a-f]{64}$"
        )
    )
    preparation_receipt_id: StrictStr = Field(
        pattern=r"^vcf-observation-preparation:[0-9a-f]{64}$"
    )
    target: ObserveVariantInput
    query_invocation: ProcessInvocationReceipt

    @model_validator(mode="after")
    def exact_query(self) -> "_ObservationBase":
        _validate_authority_id_kind(self.authority_kind, self.continuation_id)
        _validate_query_invocation(
            self.query_invocation, _point_region(self.target), require_zero=True
        )
        return self


class _RecordObservation(_ObservationBase):
    """One record answers the target; the other overlapping records stay beside it."""

    record: ObservedVcfRecord
    genotype: TargetGenotypeAssessment
    other_records: Tuple[ObservedVcfRecord, ...] = ()

    @model_serializer(mode="wrap")
    def optional_other_records(self, handler: Any) -> dict[str, Any]:
        document = handler(self)
        if not self.other_records:
            document.pop("other_records", None)
        return document

    def _validate_answer(self, states: set[str]) -> int | None:
        _validate_query_output(self.query_invocation, expect_empty=False)
        _validate_records((self.record, *self.other_records), self.target)
        index = _target_alt_index(self.record, self.target)
        if index is None or any(
            _target_alt_index(item, self.target) is not None
            for item in self.other_records
        ):
            raise ValueError("observation record changed requested REF/ALT")
        if (
            self.genotype
            != DeterministicVcfObservationTool._assess_genotype(self.record, index)
            or self.genotype.state not in states
        ):
            raise ValueError("observation contradicts the sample genotype")
        return index


class ExactAlleleObserved(_RecordObservation):
    outcome: Literal["exact_allele_observed"] = "exact_allele_observed"
    claim_scope: Literal["Exact allele and genotype observation in this VCF only."] = (
        "Exact allele and genotype observation in this VCF only."
    )

    @model_validator(mode="after")
    def exact_semantics(self) -> "ExactAlleleObserved":
        self._validate_answer({"target_alt_carried"})
        return self


class TargetAlleleNotCarried(_RecordObservation):
    """The sample's call at the requested REF does not carry the target ALT.

    Either the record lists the target ALT and the genotype does not carry it, or
    the one record at the target lists other ALTs only and the genotype is fully
    called (``genotype.target_alt_index`` is ``None``). Either way the record
    answers absence (``_answers_absence``): no ALT's sequence is unknown.
    """

    outcome: Literal["target_allele_not_carried"] = "target_allele_not_carried"
    claim_scope: Literal[
        "This sample's genotype at the requested REF does not carry the target ALT."
    ] = "This sample's genotype at the requested REF does not carry the target ALT."

    @model_validator(mode="after")
    def absent_semantics(self) -> "TargetAlleleNotCarried":
        if self.genotype.target_alt_index is not None:
            self._validate_answer({"target_alt_not_carried"})
            if not _answers_absence(self.record, self.target):
                raise ValueError("not-carried result rests on an unknown ALT")
            return self
        _validate_query_output(self.query_invocation, expect_empty=False)
        records = (self.record, *self.other_records)
        _validate_records(records, self.target)
        if (
            any(_target_alt_index(item, self.target) is not None for item in records)
            or [item for item in records if _covers_locus(item, self.target)]
            != [self.record]
            or not _answers_absence(self.record, self.target)
            or self.genotype
            != DeterministicVcfObservationTool._assess_genotype(self.record, None)
            or self.genotype.state != "target_alt_not_carried"
        ):
            raise ValueError("not-carried result contradicts the record at the target")
        return self


class TargetGenotypeUnresolved(_RecordObservation):
    outcome: Literal["target_genotype_unresolved"] = "target_genotype_unresolved"
    claim_scope: Literal[
        "The record contains the requested REF/ALT, but the sample genotype does not "
        "support a resolved target-allele observation."
    ] = (
        "The record contains the requested REF/ALT, but the sample genotype does not "
        "support a resolved target-allele observation."
    )

    @model_validator(mode="after")
    def unresolved_semantics(self) -> "TargetGenotypeUnresolved":
        """A no-call, an invalid call, or a call that leaves the target ALT out but
        carries an ALT of unknown sequence, which could be the target's."""
        self._validate_answer({"no_call", "invalid", "target_alt_not_carried"})
        if self.genotype.state == "target_alt_not_carried" and _answers_absence(
            self.record, self.target
        ):
            raise ValueError("unresolved result hides a not-carried answer")
        return self


class LocusIdentityMismatch(_ObservationBase):
    outcome: Literal["locus_present_identity_mismatch"] = (
        "locus_present_identity_mismatch"
    )
    records: Tuple[ObservedVcfRecord, ...] = Field(min_length=1)
    claim_scope: Literal[
        "The locus is present, but the requested REF/ALT identity was not observed."
    ] = "The locus is present, but the requested REF/ALT identity was not observed."

    @model_validator(mode="after")
    def mismatch_semantics(self) -> "LocusIdentityMismatch":
        _validate_query_output(self.query_invocation, expect_empty=False)
        _validate_records(self.records, self.target)
        if any(
            _target_alt_index(record, self.target) is not None
            for record in self.records
        ):
            raise ValueError("mismatch result contains an exact record")
        if not any(_covers_locus(record, self.target) for record in self.records):
            raise ValueError("mismatch result lacks a record at the locus")
        return self


class LocusOverlapped(_ObservationBase):
    """Only records that start upstream and span the target were returned."""

    outcome: Literal["locus_overlapped"] = "locus_overlapped"
    records: Tuple[ObservedVcfRecord, ...] = Field(min_length=1)
    claim_scope: Literal[
        "A record starting upstream spans this locus; no record with the target "
        "identity was observed."
    ] = (
        "A record starting upstream spans this locus; no record with the target "
        "identity was observed."
    )

    @model_validator(mode="after")
    def overlap_semantics(self) -> "LocusOverlapped":
        _validate_query_output(self.query_invocation, expect_empty=False)
        _validate_records(self.records, self.target)
        if any(_covers_locus(record, self.target) for record in self.records):
            raise ValueError("overlap result contains a record at the locus")
        return self


class NoRecordObserved(_ObservationBase):
    outcome: Literal["no_record"] = "no_record"
    claim_scope: Literal[
        "No record was observed at this locus in this variant-only VCF; "
        "reference genotype is not established."
    ] = (
        "No record was observed at this locus in this variant-only VCF; "
        "reference genotype is not established."
    )

    @model_validator(mode="after")
    def empty_semantics(self) -> "NoRecordObserved":
        _validate_query_output(self.query_invocation, expect_empty=True)
        return self


class AmbiguousRecordsObserved(_ObservationBase):
    outcome: Literal["ambiguous_records"] = "ambiguous_records"
    matching_records: Tuple[ObservedVcfRecord, ...] = Field(min_length=2)
    claim_scope: Literal[
        "Multiple records match the requested identity; "
        "no unique observation is selected."
    ] = (
        "Multiple records match the requested identity; "
        "no unique observation is selected."
    )

    @model_validator(mode="after")
    def ambiguous_semantics(self) -> "AmbiguousRecordsObserved":
        _validate_query_output(self.query_invocation, expect_empty=False)
        _validate_records(self.matching_records, self.target)
        if any(
            _target_alt_index(record, self.target) is None
            for record in self.matching_records
        ):
            raise ValueError("ambiguous result contains a non-matching record")
        return self


VariantObservationRefusalCode = Literal[
    "invalid_target",
    "target_unresolved",
    "tool_failure",
]


class VariantObservationRefused(FrozenContract):
    schema_version: Literal["bio-neb.variant-observation/2"] = (
        "bio-neb.variant-observation/2"
    )
    outcome: Literal["refused"] = "refused"
    code: VariantObservationRefusalCode
    violations: Tuple[StrictStr, ...] = ()
    authority_kind: VcfObservationAuthorityKind | None = None
    continuation_id: StrictStr | None = Field(
        default=None,
        pattern=(
            r"^(?:scientific-continuation|investigation-authority|vcf-observation-mechanics):[0-9a-f]{64}$"
        ),
    )
    preparation_receipt_id: StrictStr | None = Field(
        default=None,
        pattern=r"^vcf-observation-preparation(?:-failure)?:[0-9a-f]{64}$",
    )
    preparation_failure: VcfPreparationFailed | None = None
    target: ObserveVariantInput | ObserveRegionInput | None = None
    query_invocation: ProcessInvocationReceipt | None = None
    claim_scope: Literal[
        "No biological absence, function, causality, diagnosis, or reference genotype "
        "is established."
    ] = (
        "No biological absence, function, causality, diagnosis, or reference genotype "
        "is established."
    )

    @model_validator(mode="after")
    def coherent_refusal(self) -> "VariantObservationRefused":
        if self.preparation_failure is not None:
            expected_id = (
                "vcf-observation-preparation-failure:"
                + canonical_json_digest(
                    self.preparation_failure.model_dump(mode="json")
                )
            )
            if (
                self.code != "tool_failure"
                or self.authority_kind != "investigation_authority"
                or self.continuation_id is None
                or self.preparation_receipt_id != expected_id
                or self.target is None
                or self.query_invocation is not None
                or not self.violations
                or self.violations[0] != f"preparation:{self.preparation_failure.code}"
            ):
                raise ValueError("preparation refusal changed exact failure evidence")
            _validate_authority_id_kind(self.authority_kind, self.continuation_id)
            return self
        if self.code == "invalid_target":
            if any(
                item is not None
                for item in (
                    self.authority_kind,
                    self.continuation_id,
                    self.preparation_receipt_id,
                    self.target,
                    self.query_invocation,
                    self.preparation_failure,
                )
            ):
                raise ValueError(
                    "invalid target refusal fabricated validated authority"
                )
            return self
        if (
            self.authority_kind is None
            or self.continuation_id is None
            or self.preparation_receipt_id is None
            or self.target is None
        ):
            raise ValueError("validated refusal lacks exact authority and target")
        _validate_authority_id_kind(self.authority_kind, self.continuation_id)
        if self.query_invocation is not None:
            _validate_query_invocation(
                self.query_invocation,
                _point_region(self.target),
                require_zero=False,
            )
            if self.code != "tool_failure":
                raise ValueError("non-Tool refusal retained a query invocation")
        return self


RegionReference = Literal["matches", "differs", "outside"]
"""A region record's REF against the Run's reference bases: equal, different, or a
record reaching outside the region, which is not compared."""


class RegionRecord(FrozenContract):
    """One VCF record in a region, as written, with its REF checked (#367)."""

    record: ObservedVcfRecord
    reference: RegionReference


class VcfRegionObserved(FrozenContract):
    """Every record a region's window of the prepared VCF holds (#367, owner option
    b), up to ``record_limit`` rows; the rest are counted in ``not_reported``. Each
    row is a record the VCF holds there, never a finding."""

    schema_version: Literal["bio-neb.vcf-region-observation/1"] = (
        "bio-neb.vcf-region-observation/1"
    )
    outcome: Literal["region_observed"] = "region_observed"
    authority_kind: VcfObservationAuthorityKind
    continuation_id: StrictStr
    preparation_receipt_id: StrictStr = Field(
        pattern=r"^vcf-observation-preparation:[0-9a-f]{64}$"
    )
    target: ObserveRegionInput
    query_invocation: ProcessInvocationReceipt
    record_limit: StrictInt = Field(gt=0)
    records: Tuple[RegionRecord, ...] = ()
    not_reported: StrictInt = Field(default=0, ge=0)
    claim_scope: Literal[
        "Records this VCF holds in the region only; no coverage, reference genotype "
        "or absence is established, and no finding is admitted."
    ] = (
        "Records this VCF holds in the region only; no coverage, reference genotype "
        "or absence is established, and no finding is admitted."
    )

    @model_validator(mode="after")
    def exact_region(self) -> "VcfRegionObserved":
        _validate_authority_id_kind(self.authority_kind, self.continuation_id)
        _validate_query_invocation(
            self.query_invocation,
            _regions(self.target.contig, self.target.start, self.target.end),
            require_zero=True,
        )
        if len(self.records) > self.record_limit or (
            self.not_reported and len(self.records) != self.record_limit
        ):
            raise ValueError("a region reports up to its record limit, then counts")
        if any(
            row.record.contig != self.target.contig
            or not _reaches(row.record, self.target.start, self.target.end)
            or row.reference != region_reference(row.record, self.target)
            for row in self.records
        ):
            raise ValueError("a region record lies outside it or changed its check")
        return self


def region_reference(record: ObservedVcfRecord, region: ObserveRegionInput) -> str:
    """A record's REF against the region's reference bases (#367)."""
    first = record.one_based_position
    last = first + len(record.ref) - 1
    if first < region.start or last > region.end:
        return "outside"
    held = region.reference_bases[first - region.start : last - region.start + 1]
    return "matches" if held == record.ref else "differs"


VariantObservationResult = Annotated[
    Union[
        ExactAlleleObserved,
        TargetAlleleNotCarried,
        TargetGenotypeUnresolved,
        LocusIdentityMismatch,
        LocusOverlapped,
        NoRecordObserved,
        AmbiguousRecordsObserved,
        VariantObservationRefused,
        VcfRegionObserved,
    ],
    Field(discriminator="outcome"),
]


class VcfWindowQuery(FrozenContract):
    """An item's window: the bases read on each side of its target (the Run's flank)."""

    flank_bases: StrictInt = Field(ge=0)


_WINDOW_CLAIM_SCOPE = (
    "Records in this VCF window only; no coverage, reference genotype or absence is "
    "established."
)


class VcfWindowObserved(FrozenContract):
    """The records one item's window of the prepared VCF returned (ADR 0047)."""

    schema_version: Literal["bio-neb.vcf-window-query/1"] = "bio-neb.vcf-window-query/1"
    outcome: Literal["window_observed"] = "window_observed"
    authority_kind: VcfObservationAuthorityKind
    continuation_id: StrictStr
    preparation_receipt_id: StrictStr = Field(
        pattern=r"^vcf-observation-preparation:[0-9a-f]{64}$"
    )
    target: ObserveVariantInput
    flank_bases: StrictInt = Field(ge=0)
    window_start: StrictInt = Field(gt=0)
    window_end: StrictInt = Field(gt=0)
    query_invocation: ProcessInvocationReceipt
    records: Tuple[ObservedVcfRecord, ...] = ()
    claim_scope: Literal[
        "Records in this VCF window only; no coverage, reference genotype or absence "
        "is established."
    ] = _WINDOW_CLAIM_SCOPE

    @model_validator(mode="after")
    def exact_window(self) -> "VcfWindowObserved":
        _validate_authority_id_kind(self.authority_kind, self.continuation_id)
        first, last = _target_span(self.target)
        if (
            self.window_start != max(1, first - self.flank_bases)
            or not last <= self.window_end <= last + self.flank_bases
        ):
            raise ValueError("window changed its target and flank")
        _validate_query_invocation(
            self.query_invocation,
            _regions(self.target.contig, self.window_start, self.window_end),
            require_zero=True,
        )
        _validate_query_output(self.query_invocation, expect_empty=not self.records)
        if any(
            record.contig != self.target.contig
            or not _reaches(record, self.window_start, self.window_end)
            for record in self.records
        ):
            raise ValueError("window record lies outside its window")
        return self


class VcfWindowQueryRefused(FrozenContract):
    schema_version: Literal["bio-neb.vcf-window-query/1"] = "bio-neb.vcf-window-query/1"
    outcome: Literal["refused"] = "refused"
    code: Literal["invalid_window", "target_unresolved", "tool_failure"]
    violations: Tuple[StrictStr, ...] = ()
    authority_kind: VcfObservationAuthorityKind
    continuation_id: StrictStr
    preparation_receipt_id: StrictStr = Field(
        pattern=r"^vcf-observation-preparation:[0-9a-f]{64}$"
    )
    target: ObserveVariantInput
    flank_bases: StrictInt = Field(ge=0)
    query_invocation: ProcessInvocationReceipt | None = None
    claim_scope: Literal[
        "Records in this VCF window only; no coverage, reference genotype or absence "
        "is established."
    ] = _WINDOW_CLAIM_SCOPE

    @model_validator(mode="after")
    def coherent_refusal(self) -> "VcfWindowQueryRefused":
        _validate_authority_id_kind(self.authority_kind, self.continuation_id)
        if self.query_invocation is not None:
            # The query's own window around the target, at its fixed grammar.
            regions = self.query_invocation.regions_file
            region = "" if regions is None else regions.text
            contig, start, end = (region.rstrip("\n").split("\t") + ["", "", ""])[:3]
            position = self.target.one_based_position
            if (
                self.code != "tool_failure"
                or contig != self.target.contig
                or start != str(max(1, position - self.flank_bases))
                or not end.isdigit()
                or not position <= int(end) <= position + self.flank_bases
            ):
                raise ValueError("window refusal changed its query evidence")
            _validate_query_invocation(
                self.query_invocation, region, require_zero=False
            )
        elif self.code == "tool_failure" and not self.violations:
            # A fault that left no process result names itself instead.
            raise ValueError("window refusal changed its query evidence")
        return self


VcfWindowQueryResult = Annotated[
    Union[VcfWindowObserved, VcfWindowQueryRefused],
    Field(discriminator="outcome"),
]


def _invocation(
    purpose: InvocationPurpose,
    argv: Tuple[str, ...],
    result: CompletedProcess,
    *,
    stdout_file: LocalFileObservation | None = None,
    regions_file: RegionsFile | None = None,
) -> ProcessInvocationReceipt:
    """Bind one process result; a redirected stdout is described by its file."""
    return ProcessInvocationReceipt(
        purpose=purpose,
        argv=argv,
        regions_file=regions_file,
        return_code=result.return_code,
        stdout_byte_count=(
            len(result.stdout) if stdout_file is None else stdout_file.byte_count or 0
        ),
        stdout_sha256=(
            _bytes_sha256(result.stdout)
            if stdout_file is None
            else stdout_file.sha256 or _bytes_sha256(b"")
        ),
        stderr_byte_count=len(result.stderr),
        stderr_sha256=_bytes_sha256(result.stderr),
    )


def _validated_file(value: Any) -> LocalFileObservation:
    if not isinstance(value, LocalFileObservation):
        raise TypeError("Adapter file observation changed contract")
    return LocalFileObservation.model_validate(value.model_dump(mode="json"))


def _run_query(
    adapter: "VcfProcessAdapter", argv: Tuple[str, ...], *, timeout_seconds: int
) -> CompletedProcess:
    """One VCF query (a lookup or a window) under the ``vcf_observation`` stage's failure policy.

    A timeout is transient and retried by the Run's recorded policy; the last one
    is re-raised for the typed ``tool_failure``. Every other fault is raised at
    once. The shared preparation runs once: its commands write derived files that
    a retry would first need removed.
    """

    def operation() -> CompletedProcess:
        try:
            return _validated_process(
                adapter.run(argv, timeout_seconds=timeout_seconds)
            )
        except TimeoutExpired as error:
            raise ProviderError("transient", type(error).__name__) from error

    try:
        return call_with_failure_policy(
            operation,
            policy_for(FailureSurface(scope="stage", name="vcf_observation")),
            lambda _error, _ordinal, _delay: None,
        )
    except ProviderError as error:
        raise error.__cause__ from None


def _validated_process(value: Any) -> CompletedProcess:
    if (
        not isinstance(value, CompletedProcess)
        or type(value.return_code) is not int
        or not isinstance(value.stdout, bytes)
        or not isinstance(value.stderr, bytes)
    ):
        raise TypeError("Adapter process observation changed contract")
    return value


def _violations(error: ValidationError) -> Tuple[str, ...]:
    return tuple(
        ".".join(str(item) for item in violation["loc"]) or "$"
        for violation in error.errors(
            include_url=False, include_context=False, include_input=False
        )
    )


def _named_receipt(material: dict[str, Any]) -> PreparedVcfObservationReceipt:
    """Name a preparation by the canonical digest of its public document."""
    draft = PreparedVcfObservationReceipt.model_construct(
        receipt_id="", **material
    ).model_dump(mode="json", exclude={"receipt_id"})
    return PreparedVcfObservationReceipt(
        receipt_id=f"vcf-observation-preparation:{canonical_json_digest(draft)}",
        **material,
    )


class DeterministicVcfObservationTool:
    """Own all VCF command grammar, parsing, and result derivation."""

    name = "observe_variant"

    def __init__(self, defaults: VcfLookupDefaults | None = None) -> None:
        self.defaults = defaults or VcfLookupDefaults()

    def prepare(  # noqa: C901 - one ordered preparation with its typed failures.
        self,
        continuation: VcfObservationAuthority,
        binding: VcfMaterializationBinding,
        adapter: VcfProcessAdapter,
        *,
        authority_kind: VcfObservationAuthorityKind = "scientific_continuation",
        read_original: bool = False,
    ) -> PreparedVcfObservationReceipt | VcfPreparationFailed:
        """Derive the queryable BGZF/CSI and read the header inventory once.

        The attested VCF and the executable are trusted (0029). ``read_original``
        is the explicit independent full read, recorded as ``original``.
        """
        descriptor = continuation.vcf
        try:
            _validate_authority_identity(
                authority_kind,
                continuation.continuation_id,
                continuation.journey_id,
            )
        except ValueError:
            return VcfPreparationFailed(code="binding_invalid")
        if (
            descriptor.role != "VCF"
            or binding.attested_vcf.byte_count != descriptor.byte_count
            or binding.attested_vcf.sha256 != descriptor.sha256
            or binding.continuation_sample_id != continuation.sample.sample_id
        ):
            return VcfPreparationFailed(code="binding_invalid")

        original: LocalFileObservation | None = None
        if read_original:
            try:
                original = _validated_file(
                    adapter.inspect_file(binding.attested_vcf.local_path)
                )
            except (OSError, TimeoutExpired) as error:
                return VcfPreparationFailed(
                    code="input_unavailable", details=(type(error).__name__,)
                )
            except ValueError as error:
                return VcfPreparationFailed(
                    code="original_identity_mismatch", details=(str(error),)
                )
            if not original.exists:
                return VcfPreparationFailed(code="input_unavailable")
            if (
                original.byte_count != binding.attested_vcf.byte_count
                or original.sha256 != binding.attested_vcf.sha256
            ):
                return VcfPreparationFailed(code="original_identity_mismatch")

        try:
            acquisition_receipt_file = _validated_file(
                adapter.inspect_file(binding.attested_vcf.acquisition_receipt_path)
            )
        except (OSError, TimeoutExpired) as error:
            return VcfPreparationFailed(
                code="binding_invalid", details=(type(error).__name__,)
            )
        if (
            not acquisition_receipt_file.exists
            or acquisition_receipt_file.sha256
            != binding.attested_vcf.acquisition_receipt_sha256
        ):
            return VcfPreparationFailed(
                code="binding_invalid", details=("acquisition_receipt",)
            )

        try:
            probe_receipt_file = _validated_file(
                adapter.inspect_file(binding.runtime.probe_receipt_path)
            )
        except (OSError, TimeoutExpired) as error:
            return VcfPreparationFailed(
                code="runtime_unavailable", details=(type(error).__name__,)
            )
        if (
            not probe_receipt_file.exists
            or probe_receipt_file.sha256 != binding.runtime.probe_receipt_sha256
        ):
            return VcfPreparationFailed(code="runtime_identity_mismatch")

        derived_bgzf_path = os.path.join(binding.derivation_root, "variants.vcf.gz")
        derived_csi_path = f"{derived_bgzf_path}.csi"
        try:
            root_exists = adapter.path_exists(binding.derivation_root)
            if type(root_exists) is not bool:
                raise TypeError("Adapter path observation changed contract")
        except (OSError, TimeoutExpired) as error:
            return VcfPreparationFailed(
                code="tool_failure", details=(type(error).__name__,)
            )
        if root_exists:
            return VcfPreparationFailed(code="derivation_target_exists")
        try:
            adapter.create_fresh_directory(binding.derivation_root)
        except FileExistsError:
            return VcfPreparationFailed(code="derivation_target_exists")
        except (OSError, TimeoutExpired) as error:
            return VcfPreparationFailed(
                code="tool_failure", details=(type(error).__name__,)
            )

        # No command names an output file and none relies on --version or
        # --list-samples: the pysam dispatcher prints nothing for those and
        # redirects a named --output to stdout.
        executable = binding.runtime.executable_path
        steps: tuple[tuple[InvocationPurpose, Tuple[str, ...], str | None], ...] = (
            (
                "bgzf_derivation",
                (
                    executable,
                    "view",
                    "--no-version",
                    "--output-type",
                    "z",
                    binding.attested_vcf.local_path,
                ),
                derived_bgzf_path,
            ),
            ("csi_derivation", (executable, "index", "--csi", derived_bgzf_path), None),
            (
                "header_inventory",
                (
                    executable,
                    "view",
                    "--no-version",
                    "--header-only",
                    derived_bgzf_path,
                ),
                None,
            ),
        )
        invocations: list[ProcessInvocationReceipt] = []
        derived_bgzf: LocalFileObservation | None = None
        header_output = b""
        for purpose, argv, stdout_path in steps:
            try:
                result = _validated_process(
                    adapter.run(
                        argv,
                        timeout_seconds=self.defaults.preparation_timeout_seconds,
                        stdout_path=stdout_path,
                    )
                )
                if stdout_path is not None:
                    derived_bgzf = _validated_file(adapter.inspect_file(stdout_path))
                invocation = _invocation(
                    purpose,
                    argv,
                    result,
                    stdout_file=derived_bgzf if stdout_path else None,
                )
            except (OSError, TimeoutExpired, ValueError) as error:
                return VcfPreparationFailed(
                    code="tool_failure",
                    details=(type(error).__name__,),
                    invocations=tuple(invocations),
                )
            invocations.append(invocation)
            if result.return_code != 0:
                return VcfPreparationFailed(
                    code="tool_failure", invocations=tuple(invocations)
                )
            if purpose == "header_inventory":
                header_output = result.stdout

        try:
            sample_names, contigs, skipped_contigs = _header_inventory(
                header_output.decode("utf-8")
            )
        except (UnicodeDecodeError, ValueError):
            sample_names, contigs, skipped_contigs = (), (), ()
        try:
            derived_csi = _validated_file(adapter.inspect_file(derived_csi_path))
        except (OSError, TimeoutExpired, ValueError) as error:
            return VcfPreparationFailed(
                code="derived_output_invalid",
                details=(type(error).__name__,),
                invocations=tuple(invocations),
            )
        if (
            derived_bgzf is None
            or not derived_bgzf.exists
            or not derived_csi.exists
            or not derived_bgzf.byte_count
            or not derived_csi.byte_count
            or sample_names != binding.expected_vcf_sample_names
            or not contigs
        ):
            return VcfPreparationFailed(
                code="derived_output_invalid", invocations=tuple(invocations)
            )

        material: dict[str, Any] = {
            "authority_kind": authority_kind,
            "continuation_id": continuation.continuation_id,
            "journey_id": continuation.journey_id,
            "assembly": continuation.reference.assembly_accession,
            "continuation_sample_id": continuation.sample.sample_id,
            "vcf_descriptor": descriptor,
            "acquisition_receipt_sha256": (
                binding.attested_vcf.acquisition_receipt_sha256
            ),
            "acquisition_receipt_file": acquisition_receipt_file,
            "original": original,
            "runtime": binding.runtime,
            "probe_receipt_file": probe_receipt_file,
            "derived_bgzf": derived_bgzf,
            "derived_csi": derived_csi,
            "vcf_sample_names": sample_names,
            "vcf_contigs": contigs,
            "skipped_contigs": skipped_contigs,
            "invocations": tuple(invocations),
        }
        return _named_receipt(material)

    def rebind_preparation(
        self,
        preparation: PreparedVcfObservationReceipt,
        authority: VcfObservationAuthority,
    ) -> PreparedVcfObservationReceipt | VcfPreparationFailed:
        """Carry an owned preparation into a new exact Investigation grant.

        The application session supplies ownership; the derived files are trusted
        (0029), so this checks the scientific binding only and repeats nothing.
        Earlier immutable receipts and invocation evidence remain unchanged.
        """
        if (
            preparation.authority_kind != "investigation_authority"
            or preparation.journey_id != authority.journey_id
            or preparation.vcf_descriptor != authority.vcf
            or preparation.assembly != authority.reference.assembly_accession
            or preparation.continuation_sample_id != authority.sample.sample_id
        ):
            return VcfPreparationFailed(code="binding_invalid")
        material = {
            name: getattr(preparation, name)
            for name in type(preparation).model_fields
            if name not in {"receipt_id", "status", "schema_version"}
        }
        material["continuation_id"] = authority.continuation_id
        return _named_receipt(material)

    def transition_transversion(
        self,
        preparation: PreparedVcfObservationReceipt,
        adapter: VcfProcessAdapter,
        *,
        timeout_seconds: int,
    ) -> float | None:
        """The prepared VCF's whole-file Ti/Tv, copied from ``bcftools stats``'
        ``TSTV`` line; None when the VCF holds no transversion (#414).

        The Run's measurement reads it here, the one VCF reader (ADR 0047). The
        process's timeout or start failure raises as the adapter raised it, a
        nonzero exit as ``CalledProcessError``, for the caller to route; an output
        this parser cannot read raises ``ValueError``.
        """
        argv = (
            preparation.runtime.executable_path,
            "stats",
            preparation.derived_bgzf.path,
        )
        process = _validated_process(adapter.run(argv, timeout_seconds=timeout_seconds))
        if process.return_code != 0:
            raise CalledProcessError(process.return_code, argv)
        rows = [
            line.split("\t")
            for line in process.stdout.decode("utf-8").splitlines()
            if line.startswith("TSTV\t")
        ]
        if len(rows) != 1:
            raise ValueError("bcftools stats wrote no single TSTV line")
        transversions, ratio = int(rows[0][3]), float(rows[0][4])
        return None if transversions == 0 else ratio

    def observe(  # noqa: C901 - one ordered lookup with its typed refusals.
        self,
        tool_input: ObserveVariantInput | Mapping[str, Any],
        continuation: VcfObservationAuthority,
        preparation: PreparedVcfObservationReceipt,
        adapter: VcfProcessAdapter,
        *,
        authority_kind: VcfObservationAuthorityKind = "scientific_continuation",
    ) -> VariantObservationResult:
        """Query one target in the prepared VCF and derive its typed outcome.

        The prepared files are trusted (0029): nothing is inspected or hashed again.
        """
        try:
            target = (
                tool_input
                if isinstance(tool_input, ObserveVariantInput)
                else ObserveVariantInput.model_validate(tool_input)
            )
        except ValidationError as error:
            return VariantObservationRefused(
                code="invalid_target", violations=_violations(error)
            )

        def unresolved(violation: str) -> VariantObservationRefused:
            return VariantObservationRefused(
                code="target_unresolved",
                violations=(violation,),
                authority_kind=authority_kind,
                continuation_id=continuation.continuation_id,
                preparation_receipt_id=preparation.receipt_id,
                target=target,
            )

        if target.assembly != continuation.reference.assembly_accession:
            return unresolved("assembly")
        if (
            preparation.authority_kind != authority_kind
            or preparation.continuation_id != continuation.continuation_id
            or preparation.journey_id != continuation.journey_id
            or preparation.assembly != target.assembly
            or preparation.vcf_descriptor != continuation.vcf
            or preparation.continuation_sample_id != continuation.sample.sample_id
        ):
            return unresolved("continuation_binding")
        contig = next(
            (item for item in preparation.vcf_contigs if item.name == target.contig),
            None,
        )
        if contig is None:
            return unresolved(
                "contig_unusable"
                if any(
                    item.contig_id == target.contig
                    for item in preparation.skipped_contigs
                )
                else "contig"
            )
        if _target_span(target)[1] > contig.length:
            return unresolved("one_based_position")

        region = _point_region(target)
        try:
            # The regions file is written with the query: a local fault is the Tool's.
            regions = _regions_file(adapter, preparation.derived_bgzf.path, region)
            argv = _query_argv(
                preparation.runtime.executable_path,
                regions.path,
                preparation.derived_bgzf.path,
            )
            process = _run_query(
                adapter, argv, timeout_seconds=self.defaults.query_timeout_seconds
            )
        except (OSError, TimeoutExpired) as error:
            return self._refused(
                "tool_failure",
                continuation,
                preparation,
                target,
                type(error).__name__,
            )
        query_invocation = _invocation(
            "variant_query", argv, process, regions_file=regions
        )
        if process.return_code != 0:
            return self._refused(
                "tool_failure",
                continuation,
                preparation,
                target,
                f"exit_{process.return_code}",
                query_invocation=query_invocation,
            )
        if len(process.stdout) > self.defaults.query_output_byte_limit:
            return self._refused(
                "tool_failure",
                continuation,
                preparation,
                target,
                "output_too_large",
                query_invocation=query_invocation,
            )
        try:
            records = self._parse_records(process.stdout, target)
        except (UnicodeDecodeError, ValueError):
            return self._refused(
                "tool_failure",
                continuation,
                preparation,
                target,
                "output_invalid",
                query_invocation=query_invocation,
            )

        expected_samples = preparation.vcf_sample_names
        if any(
            tuple(item.sample_name for item in record.genotypes) != expected_samples
            for record in records
        ):
            return self._refused(
                "tool_failure",
                continuation,
                preparation,
                target,
                "sample_identity",
                query_invocation=query_invocation,
            )

        base = {
            "authority_kind": authority_kind,
            "continuation_id": continuation.continuation_id,
            "preparation_receipt_id": preparation.receipt_id,
            "target": target,
            "query_invocation": query_invocation,
        }
        return self._outcome(records, target, base)

    def observe_region(
        self,
        target: ObserveRegionInput,
        continuation: VcfObservationAuthority,
        preparation: PreparedVcfObservationReceipt,
        adapter: VcfProcessAdapter,
        *,
        record_limit: int,
        authority_kind: VcfObservationAuthorityKind = "investigation_authority",
    ) -> VcfRegionObserved | VariantObservationRefused:
        """Read every record in one region (#367): the query's region is the
        region itself, with the same grammar, parser and trust as ``query_window``;
        each record's REF is checked against the Run's reference bases."""

        def refused(code: str, *violations: str, **evidence: Any):
            return VariantObservationRefused(
                code=code,
                violations=violations,
                authority_kind=authority_kind,
                continuation_id=continuation.continuation_id,
                preparation_receipt_id=preparation.receipt_id,
                target=target,
                **evidence,
            )

        if (
            preparation.authority_kind != authority_kind
            or preparation.continuation_id != continuation.continuation_id
            or preparation.journey_id != continuation.journey_id
            or preparation.assembly != target.assembly
            or preparation.vcf_descriptor != continuation.vcf
            or preparation.continuation_sample_id != continuation.sample.sample_id
        ):
            return refused("target_unresolved", "continuation_binding")
        contig = next(
            (item for item in preparation.vcf_contigs if item.name == target.contig),
            None,
        )
        if contig is None or target.end > contig.length:
            return refused("target_unresolved", "contig")
        try:
            # The regions file is written with the query: a local fault is the Tool's.
            regions = _regions_file(
                adapter,
                preparation.derived_bgzf.path,
                _regions(target.contig, target.start, target.end),
            )
            argv = _query_argv(
                preparation.runtime.executable_path,
                regions.path,
                preparation.derived_bgzf.path,
            )
            process = _run_query(
                adapter, argv, timeout_seconds=self.defaults.query_timeout_seconds
            )
        except (OSError, TimeoutExpired) as error:
            return refused("tool_failure", type(error).__name__)
        invocation = _invocation("variant_query", argv, process, regions_file=regions)
        if process.return_code != 0:
            return refused(
                "tool_failure",
                f"exit_{process.return_code}",
                query_invocation=invocation,
            )
        if len(process.stdout) > self.defaults.query_output_byte_limit:
            return refused(
                "tool_failure", "output_too_large", query_invocation=invocation
            )
        try:
            records = self._parse_records(
                process.stdout, target, window=(target.start, target.end)
            )
        except (UnicodeDecodeError, ValueError):
            return refused(
                "tool_failure", "output_invalid", query_invocation=invocation
            )
        if any(
            tuple(item.sample_name for item in record.genotypes)
            != preparation.vcf_sample_names
            for record in records
        ):
            return refused(
                "tool_failure", "sample_identity", query_invocation=invocation
            )
        return VcfRegionObserved(
            authority_kind=authority_kind,
            continuation_id=continuation.continuation_id,
            preparation_receipt_id=preparation.receipt_id,
            target=target,
            query_invocation=invocation,
            record_limit=record_limit,
            records=tuple(
                RegionRecord(record=record, reference=region_reference(record, target))
                for record in records[:record_limit]
            ),
            not_reported=max(0, len(records) - record_limit),
        )

    def query_window(
        self,
        target: ObserveVariantInput,
        window: VcfWindowQuery,
        continuation: VcfObservationAuthority,
        preparation: PreparedVcfObservationReceipt,
        adapter: VcfProcessAdapter,
        *,
        authority_kind: VcfObservationAuthorityKind = "investigation_authority",
    ) -> VcfWindowQueryResult:
        """Read the records around one bound target (an item's window, 0047).

        Same grammar, parser and trust as ``observe``; the window is clamped to the
        contig and to a named maximum flank.
        """
        flank = window.flank_bases
        base = {
            "authority_kind": authority_kind,
            "continuation_id": continuation.continuation_id,
            "preparation_receipt_id": preparation.receipt_id,
            "target": target,
            "flank_bases": flank,
        }

        def refused(
            code: str, *violations: str, **evidence: Any
        ) -> VcfWindowQueryRefused:
            return VcfWindowQueryRefused(
                code=code, violations=violations, **base, **evidence
            )

        if flank > self.defaults.window_flank_max_bases:
            return refused("invalid_window", "flank_bases")
        if (
            preparation.authority_kind != authority_kind
            or preparation.continuation_id != continuation.continuation_id
            or preparation.journey_id != continuation.journey_id
            or preparation.assembly != target.assembly
            or preparation.vcf_descriptor != continuation.vcf
            or preparation.continuation_sample_id != continuation.sample.sample_id
        ):
            return refused("target_unresolved", "continuation_binding")
        contig = next(
            (item for item in preparation.vcf_contigs if item.name == target.contig),
            None,
        )
        if contig is None or _target_span(target)[1] > contig.length:
            return refused("target_unresolved", "contig")
        first, last = _target_span(target)
        start = max(1, first - flank)
        end = min(contig.length, last + flank)
        try:
            # The regions file is written with the query: a local fault is the Tool's.
            regions = _regions_file(
                adapter,
                preparation.derived_bgzf.path,
                _regions(target.contig, start, end),
            )
            argv = _query_argv(
                preparation.runtime.executable_path,
                regions.path,
                preparation.derived_bgzf.path,
            )
            process = _run_query(
                adapter, argv, timeout_seconds=self.defaults.query_timeout_seconds
            )
        except (OSError, TimeoutExpired) as error:
            return refused("tool_failure", type(error).__name__)
        invocation = _invocation("variant_query", argv, process, regions_file=regions)
        if process.return_code != 0:
            return refused(
                "tool_failure",
                f"exit_{process.return_code}",
                query_invocation=invocation,
            )
        if len(process.stdout) > self.defaults.query_output_byte_limit:
            return refused(
                "tool_failure", "output_too_large", query_invocation=invocation
            )
        try:
            records = self._parse_records(process.stdout, target, window=(start, end))
        except (UnicodeDecodeError, ValueError):
            return refused(
                "tool_failure", "output_invalid", query_invocation=invocation
            )
        if any(
            tuple(item.sample_name for item in record.genotypes)
            != preparation.vcf_sample_names
            for record in records
        ):
            return refused(
                "tool_failure", "sample_identity", query_invocation=invocation
            )
        return VcfWindowObserved(
            **base,
            window_start=start,
            window_end=end,
            query_invocation=invocation,
            records=records,
        )

    @classmethod
    def _outcome(
        cls,
        records: Tuple[ObservedVcfRecord, ...],
        target: ObserveVariantInput,
        base: dict[str, Any],
    ) -> VariantObservationResult:
        """Split and trim each record, then answer from the records at the target."""
        if not records:
            return NoRecordObserved(**base)
        matched = [
            (position, index)
            for position, record in enumerate(records)
            if (index := _target_alt_index(record, target)) is not None
        ]
        if len(matched) > 1:
            return AmbiguousRecordsObserved(
                matching_records=tuple(records[position] for position, _ in matched),
                **base,
            )
        if matched:
            position, index = matched[0]
            record = records[position]
            others = records[:position] + records[position + 1 :]
            assessment = cls._assess_genotype(record, index)
            outcome = {
                "target_alt_carried": ExactAlleleObserved,
                "target_alt_not_carried": TargetAlleleNotCarried
                if _answers_absence(record, target)
                else TargetGenotypeUnresolved,
            }.get(assessment.state, TargetGenotypeUnresolved)
            return outcome(
                record=record, genotype=assessment, other_records=others, **base
            )
        at_locus = [
            position
            for position, record in enumerate(records)
            if _covers_locus(record, target)
        ]
        if not at_locus:
            return LocusOverlapped(records=records, **base)
        if len(at_locus) == 1:
            record = records[at_locus[0]]
            assessment = cls._assess_genotype(record, None)
            if (
                _answers_absence(record, target)
                and assessment.state == "target_alt_not_carried"
            ):
                return TargetAlleleNotCarried(
                    record=record,
                    genotype=assessment,
                    other_records=records[: at_locus[0]] + records[at_locus[0] + 1 :],
                    **base,
                )
        return LocusIdentityMismatch(records=records, **base)

    @staticmethod
    def _refused(
        code: VariantObservationRefusalCode,
        continuation: VcfObservationAuthority,
        preparation: PreparedVcfObservationReceipt,
        target: ObserveVariantInput,
        *violations: str,
        query_invocation: ProcessInvocationReceipt | None = None,
    ) -> VariantObservationRefused:
        return VariantObservationRefused(
            code=code,
            violations=tuple(violations),
            authority_kind=preparation.authority_kind,
            continuation_id=continuation.continuation_id,
            preparation_receipt_id=preparation.receipt_id,
            target=target,
            query_invocation=query_invocation,
        )

    @staticmethod
    def _parse_records(
        payload: bytes,
        target: ObserveVariantInput,
        *,
        window: tuple[int, int] | None = None,
    ) -> Tuple[ObservedVcfRecord, ...]:
        """Read ``_QUERY_FORMAT`` rows overlapping the target or the window.

        A record may start upstream of its region only when its REF reaches it.
        """
        first, last = _target_span(target) if window is None else window

        def count(text: str) -> int | None:
            if text == ".":
                return None
            if not text.isascii() or not text.isdigit():
                raise ValueError("query returned an invalid count")
            return int(text)

        text = payload.decode("utf-8")
        records: list[ObservedVcfRecord] = []
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) < 11 or (len(fields) - 6) % 5:
                raise ValueError("query row lacks genotype evidence")
            contig, raw_position, ref, raw_alts, raw_filter, raw_qual = fields[:6]
            if (
                contig != target.contig
                or not raw_position.isascii()
                or not raw_position.isdigit()
                or not 0 < int(raw_position) <= last
                or (
                    int(raw_position) + len(ref) - 1 < first
                    and not any(alt.startswith("<") for alt in raw_alts.split(","))
                )
            ):
                raise ValueError("query returned a different locus")
            alts = tuple(raw_alts.split(","))
            if (
                not ref
                or not alts
                or any(not item for item in alts)
                or len(alts) != len(set(alts))
            ):
                raise ValueError("query returned invalid allele identity")
            genotypes: list[SampleGenotype] = []
            for offset in range(6, len(fields), 5):
                sample, genotype, raw_gq, raw_dp, raw_ad = fields[offset : offset + 5]
                if not sample or not genotype:
                    raise ValueError("query returned invalid genotype evidence")
                genotypes.append(
                    SampleGenotype(
                        sample_name=sample,
                        genotype=genotype,
                        gq=count(raw_gq),
                        dp=count(raw_dp),
                        ad=None
                        if raw_ad == "."
                        else tuple(count(item) for item in raw_ad.split(",")),
                    )
                )
            if len(genotypes) != len({item.sample_name for item in genotypes}):
                raise ValueError("query returned duplicate sample identity")
            records.append(
                ObservedVcfRecord(
                    contig=contig,
                    one_based_position=int(raw_position),
                    ref=ref,
                    alts=alts,
                    filters=() if raw_filter == "." else tuple(raw_filter.split(";")),
                    qual=None if raw_qual == "." else raw_qual,
                    genotypes=tuple(genotypes),
                )
            )
        return tuple(records)

    @staticmethod
    def _assess_genotype(
        record: ObservedVcfRecord, target_alt_index: int | None
    ) -> TargetGenotypeAssessment:
        genotype = record.genotypes[0]
        raw = genotype.genotype

        def assessment(**values: Any) -> TargetGenotypeAssessment:
            return TargetGenotypeAssessment(
                sample_name=genotype.sample_name,
                raw_genotype=raw,
                target_alt_index=target_alt_index,
                **values,
            )

        # VCF 4.3 lets each allele after the first carry its own separator, so a
        # call states one phase only when every separator agrees; a one-allele call
        # states none. The raw genotype keeps the separators as written.
        tokens = raw.replace("|", "/").split("/")
        separators = {item for item in raw if item in "/|"}
        phased = separators == {"|"} if len(separators) == 1 else None
        parsed: list[int | None] = []
        malformed = not tokens
        for token in tokens:
            if token == ".":
                parsed.append(None)
            elif (
                not token
                or not token.isascii()
                or not token.isdigit()
                or (len(token) > 1 and token.startswith("0"))
            ):
                malformed = True
                parsed.append(None)
            else:
                parsed.append(int(token))
        if malformed:
            return assessment(
                state="invalid",
                phased=None,
                allele_indexes=tuple(parsed) or (None,),
                reason="malformed",
            )
        if any(item is None for item in parsed):
            return assessment(
                state="no_call",
                phased=phased,
                allele_indexes=tuple(parsed),
                reason="missing_allele",
            )
        if any(item > len(record.alts) for item in parsed if item is not None):
            return assessment(
                state="invalid",
                phased=phased,
                allele_indexes=tuple(parsed),
                reason="out_of_range",
            )
        return assessment(
            state="target_alt_carried"
            if target_alt_index is not None and target_alt_index in parsed
            else "target_alt_not_carried",
            phased=phased,
            allele_indexes=tuple(parsed),
        )


__all__ = [
    "AmbiguousRecordsObserved",
    "AttestedPlainVcf",
    "BcftoolsRuntimeBinding",
    "CompletedProcess",
    "DeterministicVcfObservationTool",
    "ExactAlleleObserved",
    "LocalFileObservation",
    "LocusIdentityMismatch",
    "LocusOverlapped",
    "NoRecordObserved",
    "ObserveVariantInput",
    "ObservedVcfRecord",
    "PreparedVcfObservationReceipt",
    "ProcessInvocationReceipt",
    "SampleGenotype",
    "SkippedVcfContig",
    "TargetAlleleNotCarried",
    "TargetGenotypeAssessment",
    "TargetGenotypeUnresolved",
    "VariantObservationRefused",
    "VariantObservationResult",
    "VcfContigInventoryEntry",
    "VcfMaterializationBinding",
    "VcfPreparationFailed",
    "VcfProcessAdapter",
    "VcfWindowObserved",
    "VcfWindowQuery",
    "VcfWindowQueryRefused",
    "VcfWindowQueryResult",
]
