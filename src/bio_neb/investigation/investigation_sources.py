"""Deterministic normalization for the two lawful Investigation source operations."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any, Literal, NamedTuple, Protocol, get_args

from pydantic import ValidationError

from bio_neb.investigation.investigation import (
    ADMISSIBLE_LOOKUP_STATUSES,
    AssemblyReportStatus,
    AssemblyReportToolResult,
    AssemblyStatus,
    AssessmentVerdict,
    BodylessAnswer,
    EvaLookupDetail,
    EvidenceAssessment,
    LookupSource,
    RefSnpStatus,
    RefSnpToolResult,
    ResolvedVariantCandidate,
    RetrievedSource,
    SequenceChoice,
    SequenceNaming,
    StatedVersionPlacement,
    VariantLocationDetail,
    VersionListing,
    VersionNaming,
    eva_record_key,
)
from bio_neb.investigation.investigation_request import (
    RSID_CANONICAL_PATTERN,
    PlacedPosition,
    RequestedVariantIdentity,
    StatedAssembly,
    StatedPosition,
    StatedRegion,
)
from bio_neb.investigation.lookup_origins import (
    LookupOrigin as LookupOriginType,
)
from bio_neb.investigation.lookup_origins import (
    Malformed,
    Merged,
    PlacementUnnamed,
    PositionWithoutAlleles,
    RecordState,
    SequenceUnnamed,
    SourceAnswered,
    VersionUnnamed,
    ending,
    source_words,
)
from bio_neb.investigation.vcf_observation import trimmed_allele
from bio_neb.providers.ncbi_assembly_report import (
    ReportUnreadable,
    SequenceRecord,
    record_of,
    report_rows,
)
from bio_neb.providers.ncbi_datasets import genome_report_page
from bio_neb.providers.variant_placements import (
    PlacementsUnreadable,
    VariantPlacement,
    eva_clustered_placements,
    eva_submitted_variants,
    refsnp_placements,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
    AssemblySourceRecord,
)


def _assessment(
    source: RetrievedSource,
    verdict: AssessmentVerdict,
    rationale: str,
) -> EvidenceAssessment:
    return EvidenceAssessment.create(
        source_id=source.source_id, verdict=verdict, rationale=rationale
    )


def _json_object(raw_response: bytes) -> Mapping[str, Any] | None:
    try:
        value = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _validate_raw_source(
    source: RetrievedSource,
    *,
    operation: Literal["fetch_refsnp", "fetch_assembly_report"],
    raw_response: bytes,
) -> None:
    if source.operation != operation:
        raise ValueError(f"{operation} normalization received the wrong operation")
    if source.raw_response_byte_count != len(raw_response):
        raise ValueError(f"{operation} raw response byte count changed")
    if source.raw_response_sha256 != hashlib.sha256(raw_response).hexdigest():
        raise ValueError(f"{operation} raw response digest changed")


def _merged_rsids(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The rsIDs a current record says dbSNP merged into it, in source order."""
    merges = document.get("dbsnp1_merges")
    if not isinstance(merges, list):
        return ()
    rsids = (
        f"rs{item.get('merged_rsid')}" for item in merges if isinstance(item, Mapping)
    )
    return tuple(
        dict.fromkeys(
            rsid for rsid in rsids if re.fullmatch(RSID_CANONICAL_PATTERN, rsid)
        )
    )


def _merged_into(document: Mapping[str, Any]) -> tuple[str, ...] | None:
    """A merged record's current rsIDs; ``()`` when its merge record is unusable.

    ``None`` means the record is current (it has primary snapshot data or no merge
    record at all).
    """
    merged = document.get("merged_snapshot_data")
    if document.get("primary_snapshot_data") is not None or not isinstance(
        merged, Mapping
    ):
        return None
    current = merged.get("merged_into")
    merged_into = (
        tuple(f"rs{item}" for item in current) if isinstance(current, list) else ()
    )
    if all(re.fullmatch(RSID_CANONICAL_PATTERN, item) for item in merged_into):
        return merged_into
    return ()


def record_state(document: Mapping[str, Any]) -> str | None:
    """The snapshot a dbSNP record carries, by dbSNP's own key (``primary_snapshot_data``,
    ``merged_snapshot_data``, ``withdrawn_snapshot_data``, ...), copied (owner #468 Q5 a)."""
    keys = sorted(key for key in document if key.endswith("_snapshot_data"))
    return ", ".join(keys) or None


def _on_run(
    placements: tuple[VariantPlacement, ...], assembly_accession: str
) -> tuple[VariantPlacement, ...]:
    """Each dbSNP placement marked on the Run's assembly when dbSNP lists that exact
    assembly for its sequence (membership, ADR 0049); its Run contig is its sequence."""
    return tuple(
        placement.model_copy(
            update={"on_run_assembly": True, "run_contig": placement.sequence}
        )
        if any(
            item.assembly_accession == assembly_accession
            for item in placement.dbsnp.assemblies
        )
        else placement
        for placement in placements
    )


class _PlacedAlleles:
    """Every allele of the placements on the Run's assembly, as their SPDI states it."""

    def __init__(
        self, placements: tuple[VariantPlacement, ...], assembly_accession: str
    ) -> None:
        self.alleles: set[tuple[str, str, str, int, str, str]] = set()
        self.malformed = False
        for placement in placements:
            if placement.on_run_assembly:
                self._read(placement, assembly_accession)

    def _read(self, placement: VariantPlacement, assembly_accession: str) -> None:
        assembly_name = next(
            item.assembly_name
            for item in placement.dbsnp.assemblies
            if item.assembly_accession == assembly_accession
        )
        spdis = placement.dbsnp.alleles
        references = [
            item for item in spdis if item.deleted_sequence == item.inserted_sequence
        ]
        if len(references) != 1:
            return
        reference = references[0]
        for alternate in spdis:
            if alternate is reference:
                continue
            if (
                alternate.seq_id != reference.seq_id
                or alternate.position != reference.position
                or alternate.deleted_sequence != reference.deleted_sequence
            ):
                self.malformed = True
            else:
                self.alleles.add(
                    (
                        assembly_accession,
                        assembly_name,
                        reference.seq_id,
                        reference.position + 1,
                        reference.deleted_sequence,
                        alternate.inserted_sequence,
                    )
                )

    def candidates(self, rsid: str) -> tuple[ResolvedVariantCandidate, ...]:
        normalized = []
        for item in sorted(self.alleles):
            try:
                normalized.append(
                    ResolvedVariantCandidate(
                        assembly_accession=item[0],
                        assembly_name=item[1],
                        contig=item[2],
                        one_based_position=item[3],
                        ref=item[4],
                        alt=item[5],
                        aliases=(rsid,),
                    )
                )
            except ValidationError:
                self.malformed = True
        return tuple(normalized)


def named_placement_outcome(
    source_name: str,
    rsid: str,
    candidates: tuple[ResolvedVariantCandidate, ...],
    placement: str | None,
    on_run: tuple[str, ...],
) -> tuple[tuple[ResolvedVariantCandidate, ...], str | None]:
    """The candidates of the placement the model named (owner #468 Q1 a), checked by
    membership: it is one of the record's placements on the Run's assembly. Returns the
    named placement's candidates, or the ending that leaves it unnamed."""
    if placement is None:
        loci = {candidate.locus for candidate in candidates}
        if len(loci) <= 1:
            return candidates, None
        return candidates, PlacementUnnamed(
            source=source_name,
            rsid=rsid,
            sequences=tuple(dict.fromkeys(c.contig for c in candidates)),
            loci=len(loci),
        )
    if placement not in on_run:
        return candidates, PlacementUnnamed(
            source=source_name, rsid=rsid, sequences=on_run, named=placement
        )
    named = tuple(c for c in candidates if c.contig == placement)
    if len({candidate.locus for candidate in named}) > 1:
        return named, PlacementUnnamed(
            source=source_name, rsid=rsid, sequences=on_run, named=placement
        )
    return named, None


def normalize_refsnp(
    *,
    rsid: str,
    assembly_accession: str,
    source: RetrievedSource,
    raw_response: bytes,
    requested_variant: RequestedVariantIdentity | None = None,
    requested_rsid: str | None = None,
    location: VariantLocationDetail | Literal["not_applicable"] = "not_applicable",
    placement: str | None = None,
) -> RefSnpToolResult:
    """Copy one RefSNP answer for the queried rsID and read it on the exact assembly.

    Every placement is copied (`providers.variant_placements`), each marked on the Run's
    assembly when dbSNP lists that assembly for its sequence (#468, ADR 0049). A merged
    record names its current rsIDs. A current record yields one candidate per alternate
    allele of its one locus on the Run's assembly, of any class (SNV, MNV, insertion,
    deletion), in dbSNP's reference-expanded SPDI form; with more than one such locus the
    model names the placement (``placement``). Any other answer, a non-2xx status among
    them, is the source's to be read (`source_answered`).
    """
    _validate_raw_source(source, operation="fetch_refsnp", raw_response=raw_response)
    document = _json_object(raw_response) if 200 <= source.status_code < 300 else None
    identity = None if document is None else document.get("refsnp_id")
    verified_rsid = f"rs{identity}"
    if re.fullmatch(RSID_CANONICAL_PATTERN, verified_rsid) is None:
        verified_rsid = None

    def ended(
        status: RefSnpStatus,
        origin: LookupOriginType,
        candidates: tuple[ResolvedVariantCandidate, ...] = (),
        **recorded: Any,
    ) -> RefSnpToolResult:
        """An ending by its typed origin: its audit sentence is the conflict."""
        text, kept = ending(origin)
        return result(status, (text,), text, candidates, **kept, **recorded)

    def result(
        status: RefSnpStatus,
        conflicts: tuple[str, ...],
        rationale: str,
        candidates: tuple[ResolvedVariantCandidate, ...] = (),
        **recorded: Any,
    ) -> RefSnpToolResult:
        verdict = (
            "accepted"
            if status == "resolved"
            else "ambiguous"
            if status == "placement_unnamed"
            else status
        )
        return RefSnpToolResult.create(
            requested_rsid=requested_rsid
            if location != "not_applicable"
            else requested_rsid or rsid,
            **({"verified_rsid": verified_rsid} if verified_rsid is not None else {}),
            expected_assembly_accession=assembly_accession,
            status=status,
            candidates=candidates,
            conflicts=conflicts,
            # The call's named placement is recorded on every reading of it.
            **{
                name: value
                for name, value in {"named_placement": placement, **recorded}.items()
                if value
            },
            source=source,
            assessment=_assessment(source, verdict, rationale),
            lookup_source="ncbi_dbsnp",
            eva="not_applicable",
            location=location,
        )

    if not 200 <= source.status_code < 300:
        # The source's own answer, whatever its status: its body is in the Tool's
        # source answers, for the model to read, and its words in the ending's
        # statement.
        return ended(
            "source_answered",
            SourceAnswered(
                source="NCBI RefSNP",
                status_code=source.status_code,
                key=rsid,
                words_sent=source_words(raw_response),
            ),
        )
    if document is None:
        return ended(
            "malformed",
            Malformed(what="NCBI RefSNP's answer", detail="it is not one JSON object"),
        )
    if str(document.get("refsnp_id")) != rsid.removeprefix("rs"):
        return result(
            "conflicted",
            ("NCBI RefSNP identity did not match the requested rsID.",),
            "The retained source disagreed with the requested rsID.",
        )
    state = record_state(document)
    merged_into = _merged_into(document)
    if merged_into is not None:
        if not merged_into:
            return ended(
                "malformed",
                Malformed(
                    what="NCBI dbSNP's merge record",
                    detail="it names no one valid current rsID",
                ),
                record_state=state,
            )
        return ended(
            "merged",
            Merged(rsid=rsid, into=merged_into),
            merged_into=merged_into,
            record_state=state,
        )
    if "primary_snapshot_data" not in document:
        return ended(
            "source_answered", RecordState(rsid=rsid, state=state), record_state=state
        )
    try:
        placements = _on_run(refsnp_placements(document), assembly_accession)
    except PlacementsUnreadable as unreadable:
        return ended(
            "malformed",
            Malformed(what="NCBI RefSNP's placements", detail=str(unreadable)),
            record_state=state,
        )
    alleles = _PlacedAlleles(placements, assembly_accession)
    normalized = alleles.candidates(rsid)
    on_run = tuple(item.sequence for item in placements if item.on_run_assembly)
    recorded = {
        "merged_rsids": _merged_rsids(document),
        "placements": placements,
        "record_state": state,
    }
    if placement is not None:
        # A named placement is checked by membership whatever else resolves the
        # target (Spec r1 SP3), and only its candidates are read.
        named, unnamed = named_placement_outcome(
            "NCBI RefSNP", rsid, normalized, placement, on_run
        )
        if placement not in on_run:
            return ended("placement_unnamed", unnamed, named, **recorded)
        normalized = named
    if requested_variant is not None and normalized:
        exact = tuple(
            candidate
            for candidate in normalized
            if candidate.target_identity == requested_variant
        )
        if exact:
            return result(
                "resolved",
                (),
                "NCBI RefSNP resolved the exact requested assembly-bound allele.",
                exact,
                **recorded,
            )
        conflict = "NCBI RefSNP did not contain the exact requested target."
        return result("conflicted", (conflict,), conflict, normalized, **recorded)
    named, unnamed = named_placement_outcome(
        "NCBI RefSNP", rsid, normalized, placement, on_run
    )
    if unnamed is not None:
        return ended("placement_unnamed", unnamed, named, **recorded)
    if alleles.malformed:
        return ended(
            "malformed",
            Malformed(what="An allele of NCBI RefSNP's placements"),
            **recorded,
        )
    if named:
        return result(
            "resolved",
            (),
            "NCBI RefSNP resolved one assembly-bound locus; each alternate "
            "allele is a candidate.",
            named,
            **recorded,
        )
    if on_run:
        conflict = (
            "NCBI RefSNP's placements on the Run's assembly hold no alternate allele."
        )
        return result("not_found", (conflict,), conflict, **recorded)
    if placements:
        conflict = "NCBI RefSNP had no placement on the exact assembly."
        return result("incompatible", (conflict,), conflict, **recorded)
    conflict = "NCBI RefSNP's record holds no placement."
    return result("not_found", (conflict,), conflict, **recorded)


class AssemblyReport(NamedTuple):
    """One assembly version as NCBI's exact report of it states it."""

    accession: str
    name: str
    synonym: str | None
    paired_accession: str | None
    status: AssemblyStatus
    taxid: int | None


AssemblyReportRefusal = Literal[
    "unread", "none", "other", "repeated", "unnamed", "unsafe"
]


def read_assembly_report(
    document: Any, accession: str
) -> AssemblyReport | AssemblyReportRefusal:
    """The one exact report of ``accession`` in an NCBI Datasets answer, its current
    report or its any-version one (`NcbiInvestigationSourceAdapter.fetch_assembly_report`),
    or why there is none: ``unread`` (no report list), ``none`` (no report), ``other``
    (only other accessions), ``repeated`` (more than one), ``unnamed`` (no plain
    assembly name) or ``unsafe`` (a synonym, paired accession, status or taxon it
    cannot hold)."""
    if not isinstance(document, Mapping):
        return "unread"
    reports = document.get("reports", [])
    if not isinstance(reports, list) or not all(
        isinstance(report, Mapping) for report in reports
    ):
        return "unread"
    exact = [report for report in reports if report.get("accession") == accession]
    if len(exact) != 1:
        return "repeated" if exact else "other" if reports else "none"
    return assembly_of(exact[0])


def assembly_of(report: Mapping[str, Any]) -> AssemblyReport | AssemblyReportRefusal:
    """One report of an NCBI Datasets answer as the assembly version it states, or
    ``unnamed`` or ``unsafe`` (`read_assembly_report`)."""
    accession = report.get("accession")
    info = report.get("assembly_info")
    info = info if isinstance(info, Mapping) else {}
    organism = report.get("organism")
    name, synonym = info.get("assembly_name"), info.get("synonym")
    paired, status = report.get("paired_accession"), info.get("assembly_status")
    taxid = organism.get("tax_id") if isinstance(organism, Mapping) else None
    if not _plain_name(name):
        return "unnamed"
    if (
        not _accession(accession)
        or not (synonym is None or _plain_name(synonym))
        or not (paired is None or _accession(paired))
        or status not in get_args(AssemblyStatus)
        or not (
            taxid is None
            or (isinstance(taxid, int) and not isinstance(taxid, bool) and taxid > 0)
        )
    ):
        return "unsafe"
    return AssemblyReport(accession, name, synonym, paired, status, taxid)


def normalize_assembly_report(
    *,
    expected_assembly_accession: str,
    expected_assembly_name: str,
    source: RetrievedSource,
    raw_response: bytes,
) -> AssemblyReportToolResult:
    _validate_raw_source(
        source, operation="fetch_assembly_report", raw_response=raw_response
    )
    status: AssemblyReportStatus
    observed_status: AssemblyStatus | None = None
    document = _json_object(raw_response)
    if source.status_code == 404:
        status = "not_found"
        observed_accession = None
        observed_name = None
        conflicts = ("NCBI returned no exact assembly report.",)
        rationale = "NCBI Datasets returned HTTP 404 for the exact request."
    elif not 200 <= source.status_code < 300:
        status = "malformed"
        observed_accession = None
        observed_name = None
        conflicts = (
            f"NCBI answered HTTP {source.status_code}, not an assembly report.",
        )
        rationale = (
            f"NCBI Datasets answered HTTP {source.status_code} for the exact request."
        )
    elif document is None:
        status = "malformed"
        observed_accession = None
        observed_name = None
        conflicts = ("NCBI assembly report was not one JSON object.",)
        rationale = "The retained NCBI assembly representation was malformed."
    else:
        report = read_assembly_report(document, expected_assembly_accession)
        observed_accession = observed_name = None
        if isinstance(report, AssemblyReport):
            observed_accession, observed_name = report.accession, report.name
            observed_status = report.status
            if report.name == expected_assembly_name:
                status, conflicts = "compatible", ()
                rationale = "NCBI Datasets confirmed the exact assembly identity."
            else:
                status = "conflicted"
                conflicts = ("NCBI assembly name disagreed with RefSNP.",)
                rationale = conflicts[0]
        else:
            status, conflict = {
                "unread": ("malformed", "NCBI assembly report held no report list."),
                "none": ("not_found", "NCBI returned no exact assembly report."),
                "other": (
                    "incompatible",
                    "NCBI returned a different assembly accession.",
                ),
                "repeated": (
                    "conflicted",
                    "NCBI returned repeated exact assembly reports.",
                ),
                "unnamed": (
                    "malformed",
                    "NCBI assembly report contained a malformed assembly name.",
                ),
                "unsafe": (
                    "malformed",
                    "NCBI assembly report named the assembly's synonym, pair, status "
                    "or taxon unsafely.",
                ),
            }[report]
            conflicts = (conflict,)
            rationale = conflict
    verdict = "accepted" if status == "compatible" else status
    return AssemblyReportToolResult.create(
        expected_assembly_accession=expected_assembly_accession,
        expected_assembly_name=expected_assembly_name,
        status=status,
        observed_assembly_accession=observed_accession,
        observed_assembly_name=observed_name,
        observed_assembly_status=observed_status,
        conflicts=conflicts,
        source=source,
        assessment=_assessment(source, verdict, rationale),
    )


EvaFetch = Callable[[str, str], tuple[RetrievedSource, bytes]]
"""One GET for an operation and its key: the retained source and its body. An EVA key is
the GET's own text: an rsID, or an INSDC contig for ``eva_contig_alias``, either with the
query EVA's own options add (``rs123?contigNamingConvention=REFSEQ``, ``CM000823.5?page=1``);
``fetch_assembly_report`` takes the Run's accession. The caller owns the contact, its
failure surface and the retained evidence."""
EVA_CONTIG_NAMINGS = (
    "GENBANK_SEQUENCE_NAME",
    "ENA_SEQUENCE_NAME",
    "INSDC",
    "REFSEQ",
    "UCSC",
    "MD5_CHECKSUM",
    "TRUNC512_CHECKSUM",
    "NO_REPLACEMENT",
)
"""EVA's `contigNamingConvention` values, as its identifiers API documents them (#468
spike): the option a lookup passes through verbatim (owner R3)."""
EvaContigNaming = Literal[EVA_CONTIG_NAMINGS]
"""One of `EVA_CONTIG_NAMINGS`, as a Tool argument's type."""
EVA_VARIANT_TYPES = (
    "SNV",
    "DEL",
    "INS",
    "INDEL",
    "TANDEM_REPEAT",
    "SEQUENCE_ALTERATION",
    "NO_SEQUENCE_ALTERATION",
    "NO_ALTERNATE",
    "MNV",
    "SV",
    "CNV",
)
"""EVA's `variantType` values for its variants at a position, as its identifiers API
documents them (#469 spike): the option a position lookup passes through verbatim, in
place of the stated allele's EVA form (owner R1 a)."""
EvaVariantType = Literal[EVA_VARIANT_TYPES]
"""One of `EVA_VARIANT_TYPES`, as a Tool argument's type."""
_SOURCE_NAMES = {"ncbi_dbsnp": "dbSNP", "eva": "EVA", "ncbi": "NCBI"}
"""A lookup source's name in what its result says (#469)."""
EVA_NAMING_COLUMNS = {
    "INSDC": "genbank",
    "REFSEQ": "refseq",
    "UCSC": "ucsc",
    "GENBANK_SEQUENCE_NAME": "sequence_name",
}
"""The column of NCBI's assembly report that names a sequence as each of EVA's
contigNamingConvention values does (EVA's by-position query reads its referenceName in
the convention it is sent with, #469 spike); EVA's ENA name, checksums and
NO_REPLACEMENT have no column there."""


def _json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


class _EvaAnswered(Exception):
    """An EVA lookup GET whose answer the lookup does not read (a non-2xx status among
    them): the lookup ends on it, for the model to read (owner #468 R4 a)."""


class _EvaLookup:
    """One EVA rsID lookup: its GETs in order, the sources they kept, and the result
    in the shared lookup shape on the Run's own contig names."""

    def __init__(
        self,
        *,
        rsid: str,
        assembly_accession: str,
        fetch: EvaFetch,
        requested_variant: RequestedVariantIdentity | None,
        requested_rsid: str | None,
        location: VariantLocationDetail | Literal["not_applicable"],
        placement: str | None = None,
        contig_naming: str | None = None,
        run_report: AssemblyReport | None = None,
    ) -> None:
        self.rsid = rsid
        # NCBI's report of the Run's assembly when the caller already read it (a
        # position lookup): its paired accession is not asked again.
        self.run_report = run_report
        self.aliases: dict[str, str | None] = {}
        self.location = location
        self.assembly_accession = assembly_accession
        self.fetch = fetch
        self.requested_variant = requested_variant
        self.requested_rsid = requested_rsid
        self.placement = placement
        self.contig_naming = contig_naming
        self.source: RetrievedSource | None = None
        self.supporting: list[RetrievedSource] = []
        self.run_insdc: str | None = None
        self.assembly_name: str | None = None
        self.placements: tuple[VariantPlacement, ...] = ()

    def get(self, operation: str, key: str) -> Any:
        """One GET, its answer read when 2xx; any other ends the lookup."""
        source, body = self.fetch(operation, key)
        _check_retained(source, operation, body)
        if operation == "eva_clustered_variant":
            self.source = source
        else:
            self.supporting.append(source)
        if not 200 <= source.status_code < 300:
            raise _EvaAnswered(
                SourceAnswered(
                    source=_EVA_GETS.get(operation, operation),
                    status_code=source.status_code,
                    key=key,
                    words_sent=source_words(body),
                )
            )
        return _json(body)

    def named(self, key: str) -> str:
        """``key`` with the naming convention the model asked for, as EVA's query."""
        return eva_record_key(key, self.contig_naming)

    def result(
        self,
        status: RefSnpStatus,
        conflict: str | LookupOriginType | None,
        rationale: str,
        *,
        candidates: tuple[ResolvedVariantCandidate, ...] = (),
        verified: bool = False,
    ) -> RefSnpToolResult:
        # A typed ending's audit sentence is its conflict and rationale.
        conflict, kept = ending(conflict)
        if kept:
            rationale = conflict
        verdict = (
            "accepted"
            if status == "resolved"
            else "ambiguous"
            if status == "placement_unnamed"
            else status
        )
        return RefSnpToolResult.create(
            requested_rsid=self.requested_rsid
            if self.location != "not_applicable"
            else self.requested_rsid or self.rsid,
            **({"verified_rsid": self.rsid} if verified else {}),
            expected_assembly_accession=self.assembly_accession,
            status=status,
            candidates=candidates,
            conflicts=(conflict,) if conflict else (),
            **({"placements": self.placements} if self.placements else {}),
            **({"named_placement": self.placement} if self.placement else {}),
            **kept,
            source=self.source,
            assessment=_assessment(self.source, verdict, rationale),
            lookup_source="eva",
            eva=EvaLookupDetail(
                run_insdc_assembly_accession=self.run_insdc,
                **({"contig_naming": self.contig_naming} if self.contig_naming else {}),
                supporting_sources=tuple(self.supporting),
            ),
            location=self.location,
        )

    def run(self) -> RefSnpToolResult:
        try:
            return self._run()
        except _EvaAnswered as answered:
            (origin,) = answered.args
            return self.result(
                "source_answered", origin, "", verified=self.placements != ()
            )
        except PlacementsUnreadable as unreadable:
            return self.result(
                "malformed",
                Malformed(what="EVA's placements", detail=str(unreadable)),
                "",
                verified=self.placements != (),
            )

    def _run(self) -> RefSnpToolResult:
        document = self.get("eva_clustered_variant", self.named(self.rsid))
        if document == []:
            conflict = f"EVA has no clustered variant {self.rsid}."
            return self.result("not_found", conflict, conflict)
        self.placements = eva_clustered_placements(document)
        if any(f"rs{item.eva.accession}" != self.rsid for item in self.placements):
            conflict = (
                "EVA's clustered-variant identity did not match the requested rsID."
            )
            self.placements = ()
            return self.result("conflicted", conflict, conflict)
        self.read_run_assembly()
        self.placements = tuple(
            placement.model_copy(update={"on_run_assembly": True})
            if self.run_insdc is not None
            and placement.eva.assembly_accession == self.run_insdc
            else placement
            for placement in self.placements
        )
        if not any(placement.on_run_assembly for placement in self.placements):
            conflict = "EVA had no placement on the exact assembly."
            return self.result(
                "incompatible",
                conflict,
                f"EVA has no placement of {self.rsid} on {self.assembly_accession}.",
                verified=True,
            )
        for index, placement in enumerate(self.placements):
            if placement.on_run_assembly:
                named = placement.model_copy(
                    update={"run_contig": self.run_name(placement.sequence)}
                )
                self.placements = (
                    *self.placements[:index],
                    named,
                    *self.placements[index + 1 :],
                )
        self.with_submitted()
        return self.typed()

    def read_run_assembly(self) -> None:
        """The Run's INSDC accession, as NCBI's report of the Run's assembly pairs it
        (owner #468 R2 a): an EVA placement names its assembly by INSDC accession."""
        report = self.run_report
        if report is None:
            document = self.get("fetch_assembly_report", self.assembly_accession)
            report = (
                read_assembly_report(document, self.assembly_accession)
                if isinstance(document, Mapping)
                else "unread"
            )
        if not isinstance(report, AssemblyReport):
            raise PlacementsUnreadable(
                "NCBI's report of the Run's assembly did not name it."
            )
        self.assembly_name = report.name
        self.run_insdc = (
            self.assembly_accession
            if self.assembly_accession.startswith("GCA_")
            else report.paired_accession
        )

    def run_name(self, sequence: str) -> str | None:
        """The Run's name for the sequence of a placement on the Run's assembly
        (membership: EVA places it on the Run's INSDC accession): the sequence itself
        when EVA already names it as the Run does, else its RefSeq name from EVA's
        contig aliases, else none."""
        naming = self.contig_naming or "INSDC"
        run_naming = "INSDC" if self.assembly_accession.startswith("GCA_") else "REFSEQ"
        if naming == run_naming:
            return sequence
        if naming == "INSDC":
            # One alias read per sequence, however many placements it holds.
            if sequence not in self.aliases:
                self.aliases[sequence] = self.refseq_name(sequence)
            return self.aliases[sequence]
        return None

    def refseq_name(self, contig: str) -> str | None:
        """The RefSeq name EVA's contig aliases give ``contig`` on the Run's assembly
        pair, read page by page as EVA pages its answer until the pair is found
        (owner #468 R2 a: no cap of the lookup's own); ``None`` when EVA keeps none."""
        page = 0
        while True:
            key = contig if page == 0 else f"{contig}?page={page}"
            document = self.get("eva_contig_alias", key)
            embedded = (
                document.get("_embedded", {}) if isinstance(document, Mapping) else None
            )
            entities = (
                embedded.get("chromosomeEntities", [])
                if isinstance(embedded, Mapping)
                else None
            )
            paging = document.get("page") if isinstance(document, Mapping) else None
            if not isinstance(entities, list):
                raise PlacementsUnreadable("EVA's contig-alias answer was malformed.")
            for entity in entities:
                pair = entity.get("assembly") if isinstance(entity, Mapping) else None
                # Each identifier is a string as EVA documents it, or no RefSeq twin;
                # anything else is an answer the lookup cannot read.
                if (
                    not isinstance(pair, Mapping)
                    or not isinstance(entity.get("insdcAccession"), str)
                    or not isinstance(pair.get("insdcAccession"), str)
                    or not (
                        pair.get("refseq") is None or isinstance(pair["refseq"], str)
                    )
                ):
                    raise PlacementsUnreadable(
                        "EVA's contig-alias answer was malformed."
                    )
                if (
                    entity.get("insdcAccession") == contig
                    and pair.get("insdcAccession") == self.run_insdc
                    and pair.get("refseq") == self.assembly_accession
                ):
                    name = entity.get("refseq")
                    if name is not None and not _plain_name(name):
                        raise PlacementsUnreadable(
                            "EVA's contig-alias answer named its sequence unsafely."
                        )
                    return name
            pages = paging.get("totalPages") if isinstance(paging, Mapping) else None
            if not _positive_int(pages) or page + 1 >= pages:
                return None
            page += 1

    def with_submitted(self) -> None:
        """Each on-assembly placement with the submitted variants EVA keeps at the same
        place, asked once. Every submitted variant names this cluster."""
        document = self.get("eva_submitted_variants", self.named(self.rsid))
        submitted = eva_submitted_variants(document, int(self.rsid[2:]))
        self.placements = tuple(
            placement.model_copy(
                update={
                    "eva": placement.eva.model_copy(
                        update={
                            "submitted": tuple(
                                variant
                                for place, variant in submitted
                                if place
                                == (
                                    placement.eva.assembly_accession,
                                    placement.sequence,
                                    placement.eva.start,
                                )
                            )
                        }
                    )
                }
            )
            if placement.on_run_assembly
            else placement
            for placement in self.placements
        )

    def typed(self) -> RefSnpToolResult:
        def result(status, conflict, rationale, kept=()):
            return self.result(
                status, conflict, rationale, candidates=kept, verified=True
            )

        try:
            candidates = tuple(
                dict.fromkeys(
                    ResolvedVariantCandidate(
                        assembly_accession=self.assembly_accession,
                        assembly_name=self.assembly_name,
                        contig=placement.run_contig,
                        one_based_position=placement.eva.start,
                        ref=variant.reference,
                        alt=variant.alternate,
                        aliases=(self.rsid,),
                    )
                    for placement in self.placements
                    if placement.run_contig is not None
                    for variant in placement.eva.submitted
                    if variant.reference != variant.alternate
                )
            )
        except ValidationError:
            return result("malformed", Malformed(what="EVA's submitted alleles"), "")
        if not any(placement.run_contig for placement in self.placements):
            naming = self.contig_naming or "INSDC"
            conflict = (
                f"EVA names the sequences of {self.rsid}'s placements on the Run's "
                f"assembly in its {naming} convention, and no name of the Run's own "
                "kind (RefSeq on a GCF_ Run, INSDC on a GCA_ Run) is held for them."
            )
            return result("incompatible", conflict, conflict)
        if not candidates:
            conflict = (
                f"EVA kept no submitted alleles for {self.rsid} on the exact assembly."
            )
            return result("not_found", conflict, conflict)
        on_run = tuple(
            placement.run_contig
            for placement in self.placements
            if placement.run_contig is not None
        )
        if self.placement is not None:
            # A named placement is checked by membership whatever else resolves the
            # target (Spec r1 SP3), and only its candidates are read.
            named, unnamed = named_placement_outcome(
                "EVA", self.rsid, candidates, self.placement, on_run
            )
            if self.placement not in on_run:
                return result("placement_unnamed", unnamed, unnamed, named)
            candidates = named
        if self.requested_variant is not None:
            exact = tuple(
                candidate
                for candidate in candidates
                if candidate.target_identity == self.requested_variant
            )
            if exact:
                return result(
                    "resolved",
                    None,
                    "EVA resolved the exact requested assembly-bound allele.",
                    exact,
                )
            conflict = "EVA did not contain the exact requested target."
            return result("conflicted", conflict, conflict, candidates)
        named, unnamed = named_placement_outcome(
            "EVA", self.rsid, candidates, self.placement, on_run
        )
        if unnamed is not None:
            return result("placement_unnamed", unnamed, unnamed, named)
        return result(
            "resolved",
            None,
            "EVA resolved one assembly-bound locus; each submitted alternate allele "
            "is a candidate.",
            named,
        )


_EVA_GETS = {
    "eva_clustered_variant": "EVA's clustered variant",
    "eva_submitted_variants": "EVA's submitted variants",
    "eva_contig_alias": "EVA's contig aliases",
    "fetch_assembly_report": "NCBI's report of the Run's assembly",
}


def _check_retained(source: RetrievedSource, operation: str, body: bytes) -> None:
    if source.operation != operation:
        raise ValueError(f"{operation} lookup received the wrong operation")
    if (
        source.raw_response_byte_count != len(body)
        or source.raw_response_sha256 != hashlib.sha256(body).hexdigest()
    ):
        raise ValueError(f"{operation} raw response changed")


def _accession(value: Any) -> bool:
    """A versioned assembly accession, as a source states it."""
    return isinstance(value, str) and bool(
        re.fullmatch(ASSEMBLY_ACCESSION_PATTERN, value)
    )


def _plain_name(value: Any) -> bool:
    """A source-stated name a target can hold: non-empty, trimmed, one line."""
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and not any(character in value for character in "\r\n\0")
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def look_up_eva_rsid(
    *,
    rsid: str,
    assembly_accession: str,
    fetch: EvaFetch,
    requested_variant: RequestedVariantIdentity | None = None,
    requested_rsid: str | None = None,
    location: VariantLocationDetail | Literal["not_applicable"] = "not_applicable",
    placement: str | None = None,
    contig_naming: str | None = None,
    run_report: AssemblyReport | None = None,
) -> RefSnpToolResult:
    """Look one rsID up in EVA for the Run's assembly, in the shared lookup shape.

    The clustered variant comes first, every placement copied (#468, ADR 0049: no cap).
    NCBI's report of the Run's assembly gives its INSDC accession, and a placement EVA
    states on it is on the Run's assembly; only those placements' sequences are named
    for the Run (EVA's contig aliases on a ``GCF_`` Run). The submitted variants then give
    the alleles at the on-assembly placements. One candidate per submitted alternate
    allele at one locus, on the Run's contig name, of any class (#368); with more than
    one locus the model names the placement (``placement``). ``contig_naming`` is EVA's
    own `contigNamingConvention`, passed verbatim (owner R3). Any GET whose answer is not
    2xx ends the lookup `source_answered`, its answer kept for the model to read.
    Every GET's source is kept: the clustered variant as the result's source, the others
    in its EVA detail.
    """
    if re.fullmatch(RSID_CANONICAL_PATTERN, rsid) is None:
        raise ValueError("an EVA lookup takes one canonical rsID")
    if contig_naming is not None and contig_naming not in EVA_CONTIG_NAMINGS:
        raise ValueError("an EVA lookup's contig naming is one EVA documents")
    return _EvaLookup(
        rsid=rsid,
        assembly_accession=assembly_accession,
        fetch=fetch,
        requested_variant=requested_variant,
        requested_rsid=requested_rsid,
        location=location,
        placement=placement,
        contig_naming=contig_naming,
        run_report=run_report,
    ).run()


SourceFetch = Callable[[str, str], tuple[RetrievedSource, bytes]]
"""One GET for an operation and its key: the retained source and its body. The caller
owns the contact, its failure surface and the retained evidence. A key is the GET's own
text: an accession, an rsID, ``sequence:start0:length`` for a reference span, an SPDI,
or ``assembly:contig:start:type`` for EVA's variants at a position."""


def eva_form(position: int, ref: str, alt: str) -> tuple[int, str, str, str]:
    """EVA's start, alleles and variant type for a VCF-form variant: the shared leading
    bases (the VCF context base) are dropped and the start moves past them; an SNV or
    MNV keeps at least one base on each side."""
    keep = 1 if len(ref) == len(alt) else 0
    shared = 0
    while shared < min(len(ref), len(alt)) - keep and ref[shared] == alt[shared]:
        shared += 1
    reference, alternate = ref[shared:], alt[shared:]
    if len(reference) == len(alternate) == 1:
        kind = "SNV"
    elif not alternate:
        kind = "DEL"
    elif not reference:
        kind = "INS"
    elif len(reference) == len(alternate):
        kind = "MNV"
    else:
        kind = "INDEL"
    return position + shared, reference, alternate, kind


_POSITION_GETS = {
    "fetch_assembly_report": "NCBI's report of the Run's assembly",
    "ncbi_sequence_names": "NCBI's sequence names",
    "ncbi_reference_span": "NCBI's reference bases at the stated position",
    "ncbi_spdi_rsids": "NCBI's rsIDs at the stated allele",
    "eva_clustered_variants_at": "EVA's variants at the stated position",
    "ncbi_taxon_assemblies": "NCBI's assemblies of the Run's organism",
    "ncbi_spdi_equivalents": "NCBI's equivalents of the stated allele",
}


class _PositionLookup:
    """One stated position's GETs, in order: the Run's assembly report and its
    sequence names (the sequence named by the model, #467), the Run's reference bases at
    the stated REF, the rsIDs the routed source lists at the stated allele, then that
    one rsID's own lookup."""

    def __init__(
        self,
        *,
        position: StatedPosition | None,
        placed: PlacedPosition | None,
        stated_assembly: StatedAssembly | None,
        region: StatedRegion | None = None,
        rsid: str | None,
        requested_rsid: str | None,
        assembly_accession: str,
        fetch: SourceFetch,
        choices: tuple[SequenceChoice, ...] = (),
        words: tuple[str, ...] = (),
        literal: bool = False,
        version: VersionNaming | None = None,
        placement: str | None = None,
        listed_none: str | BodylessAnswer | None = None,
        source: LookupSource = "ncbi",
        variant_type: str | None = None,
        contig_naming: str | None = None,
    ) -> None:
        # The variant source the model chose to list and read the position's rsIDs (both
        # are offered on every Run); a region names none and reads NCBI alone.
        # ``variant_type`` is EVA's own variantType for its listing.
        self.source = source
        self.variant_type = variant_type
        # EVA's own contigNamingConvention the model named for EVA's GETs: the
        # convention of the name the listing sends, and of the names EVA's record
        # answers (as fetch_eva_variant's).
        self.contig_naming = contig_naming
        self.sequence_record: SequenceRecord | None = None
        # The placement the model named, by its sequence, when the record read has more
        # than one on the Run's assembly.
        self.placement = placement
        # The model's quote of a listing's own answer that it lists no rsID, or the
        # status of a listing answer with no body.
        self.listed_none = listed_none
        # The body-less listing answer that decision read as empty: what the result's
        # audit says it does not establish.
        self.bodyless: tuple[str, int] | None = None
        self.last_body = b""
        # The model's named sequences and the researcher's words their quotes are
        # checked against. A caller with no model (a development tool) names by
        # literal membership instead.
        self.choices = choices
        self.words = words
        self.literal = literal
        self.namings: list[SequenceNaming] = []
        # The model's checked naming of the stated version, and the one the lookup
        # read: that naming, or the Run's own assembly when the stated version is
        # literally one of its names.
        self.version = version
        self.version_naming: VersionNaming | None = None
        # The position read: the placed one, else the stated one; a
        # region's sequence is named at its start.
        self.stated = (
            placed
            or position
            or StatedPosition(contig=region.contig, position=region.start)
        )
        self.request_stated = position
        self.placed = placed
        self.region = region
        self.stated_assembly = stated_assembly
        self.rsid = rsid
        self.requested_rsid = requested_rsid
        self.assembly_accession = assembly_accession
        self.fetch = fetch
        self.sources: list[RetrievedSource] = []
        self.assembly_name: str | None = None
        self.synonym: str | None = None
        self.paired: str | None = None
        self.assembly_status: AssemblyStatus | None = None
        self.run_contig: str | None = None
        self.refseq: str | None = None
        self.genbank: str | None = None
        self.length: int | None = None
        self.bases: str | None = None
        self.listed: tuple[str, ...] = ()
        self.stated_placements: tuple[StatedVersionPlacement, ...] = ()
        self.stated_contig: str | None = None
        self.version_listings: list[VersionListing] = []
        self.report: AssemblyReport | None = None
        self._answers: dict[tuple[str, str], tuple[int, Any]] = {}
        self._reports: dict[str, tuple[int, bytes]] = {}
        # The record the lookup read and how many GETs came before it: a lookup that
        # ends after it (a later version's sequence unnamed) keeps it as its source,
        # with the GETs after it as following ones.
        self.record_read: tuple[RetrievedSource, int] | None = None

    @property
    def version_unknown(self) -> bool:
        """No version states the position: none stated, and no placement put it on
        the Run's assembly (#367)."""
        return self.stated_assembly is None and self.placed is None

    def get(self, operation: str, key: str) -> tuple[int, Any]:
        source, body = self.fetch(operation, key)
        _check_retained(source, operation, body)
        self.sources.append(source)
        self.last_body = body
        return source.status_code, _json(body) if source.status_code != 404 else None

    def listing_answered(self, operation: str, status: int, what: str = "") -> None:
        """A listing service's answer that is not 2xx (its 404 among them): the lookup
        ends on it for the model to read, unless the model quoted that answer saying it
        lists no rsID there (``listed_none``, owner #468 R5 b), checked by membership in
        the retained answer; then the listing is read as empty."""
        quote = self.listed_none
        if isinstance(quote, BodylessAnswer):
            # Only the answer the decision cites: its status, and no body.
            # Zero bytes, not an audit sentence's trimmed words (review r1 ST1).
            if quote.status_code == status and self.last_body == b"":
                self.bodyless = (
                    f"{_POSITION_GETS.get(operation, operation)}{what}",
                    status,
                )
                return
            quote = None
        elif quote and _quoted_in(quote, self.last_body):
            return
        raise _Ended(
            self.ended(
                "source_answered",
                SourceAnswered(
                    source=f"{_POSITION_GETS.get(operation, operation)}{what}",
                    status_code=status,
                    words_sent=source_words(self.last_body),
                    unquoted=quote or None,
                ),
            )
        )

    def answered(self, operation: str, status: int, what: str = "") -> RefSnpToolResult:
        """A GET whose answer the lookup does not read (a non-2xx status): the lookup
        ends on it, its answer in the Tool's source answers for the model to read (owner
        #468 R1 b, R4 a; ADR 0049: no status has a fixed meaning)."""
        return self.ended(
            "source_answered",
            SourceAnswered(
                source=f"{_POSITION_GETS.get(operation, operation)}{what}",
                status_code=status,
                words_sent=source_words(self.last_body),
            ),
        )

    def get_once(self, operation: str, key: str) -> tuple[int, Any]:
        """A JSON GET placement may need twice (a taxon's listing, an SPDI's
        equivalents), asked once and its answer reused; sequence names have their own
        (`sequence_names`)."""
        if (operation, key) not in self._answers:
            self._answers[(operation, key)] = (self.get(operation, key), self.last_body)
        # The answer read now is the one an ending quotes.
        answer, self.last_body = self._answers[(operation, key)]
        return answer

    def sequence_names(self, accession: str) -> tuple[int, bytes]:
        """An assembly's whole assembly report (#467), asked once in this lookup and
        its retained bytes reused: placement and naming may both read it."""
        if accession not in self._reports:
            source, body = self.fetch("ncbi_sequence_names", accession)
            _check_retained(source, "ncbi_sequence_names", body)
            self.sources.append(source)
            self._reports[accession] = (source.status_code, body)
        status, self.last_body = self._reports[accession]
        return status, self.last_body

    def named_on(self, accession: str, name: str | None = None) -> SequenceRecord:
        """The record the model named on ``accession`` for the written contig, checked
        by membership only (#467, ADR 0049): its quote stands in the researcher's words
        and holds the contig, and the assembly's report has the record. Anything else
        ends the lookup ``sequence_unnamed``, naming what to name (owner D1 a)."""
        written = self.stated.contig
        status, body = self.sequence_names(accession)
        what = name or accession
        if not 200 <= status < 300:
            raise _Ended(self.answered("ncbi_sequence_names", status, f" of {what}"))
        if self.placed is not None and accession == self.assembly_accession:
            # A placement that moved the position names its Run sequence by the
            # accession NCBI's answer gave it: membership of that copied id, with no
            # quote. A kept paper contig is named like any other.
            try:
                placed = record_of(body, written)
            except ReportUnreadable as unreadable:
                raise _Ended(
                    self.ended(
                        "malformed",
                        Malformed(
                            what=f"NCBI's sequence names of {what}",
                            detail=str(unreadable),
                        ),
                    )
                ) from unreadable
            if placed is not None:
                self.namings.append(
                    SequenceNaming(
                        assembly_accession=accession,
                        written=written,
                        record=written,
                        sequence_name=placed.sequence_name,
                        quote=None,
                        match="literal",
                    )
                )
                return placed
        choice = next(
            (item for item in self.choices if item.assembly_accession == accession),
            None,
        )
        if choice is None and self.literal:
            choice = self.literal_choice(accession, body, what)
        if choice is None:
            self.readable(accession, body, what)
            raise _Ended(self.unnamed((accession,)))
        quote = choice.quote
        if not self.literal and not any(quote in words for words in self.words):
            raise _Ended(
                self.ended(
                    "sequence_unnamed",
                    SequenceUnnamed(
                        contig=written,
                        assemblies=(accession,),
                        quote=quote,
                        quote_fault="not_in_words",
                    ),
                )
            )
        if written not in quote:
            raise _Ended(
                self.ended(
                    "sequence_unnamed",
                    SequenceUnnamed(
                        contig=written,
                        assemblies=(accession,),
                        quote=quote,
                        quote_fault="without_contig",
                    ),
                )
            )
        try:
            record = record_of(body, choice.record)
        except ReportUnreadable as unreadable:
            raise _Ended(
                self.ended(
                    "malformed",
                    Malformed(
                        what=f"NCBI's sequence names of {what}", detail=str(unreadable)
                    ),
                )
            ) from unreadable
        if record is None:
            raise _Ended(
                self.ended(
                    "sequence_unnamed",
                    SequenceUnnamed(
                        contig=written,
                        assemblies=(accession,),
                        missing_record=choice.record,
                    ),
                )
            )
        self.namings.append(
            SequenceNaming(
                assembly_accession=accession,
                written=written,
                record=choice.record,
                sequence_name=record.sequence_name,
                quote=quote,
                match="literal" if written in record.names else "model_matched",
            )
        )
        return record

    def literal_choice(
        self, accession: str, body: bytes, what: str
    ) -> SequenceChoice | None:
        """The one record literally named the written contig, for a caller with no
        model (owner #467 R4 a); ``None`` when none or several are."""
        written = self.stated.contig
        try:
            held = [row for row in report_rows(body) if written in row.names]
        except ReportUnreadable as unreadable:
            raise _Ended(
                self.ended(
                    "malformed",
                    Malformed(
                        what=f"NCBI's sequence names of {what}", detail=str(unreadable)
                    ),
                )
            ) from unreadable
        if len(held) != 1 or not held[0].accessions:
            return None
        return SequenceChoice(
            assembly_accession=accession,
            record=held[0].refseq or held[0].genbank,
            quote=written,
        )

    def readable(self, accession: str, body: bytes, what: str) -> None:
        """A report that does not read ends the lookup ``malformed`` before the
        model is asked to name a sequence of it (#458: the reader's typing)."""
        try:
            for _ in report_rows(body):
                pass
        except ReportUnreadable as unreadable:
            raise _Ended(
                self.ended(
                    "malformed",
                    Malformed(
                        what=f"NCBI's sequence names of {what}", detail=str(unreadable)
                    ),
                )
            ) from unreadable

    def unnamed(self, accessions: tuple[str, ...]) -> RefSnpToolResult:
        """The lookup reached assemblies with no named sequence for the contig."""
        return self.ended(
            "sequence_unnamed",
            SequenceUnnamed(contig=self.stated.contig, assemblies=accessions),
        )

    def detail(
        self,
        sources: tuple[RetrievedSource, ...],
        following: tuple[RetrievedSource, ...] = (),
    ) -> VariantLocationDetail:
        return VariantLocationDetail(
            stated=self.request_stated,
            stated_assembly=self.stated_assembly,
            placed=self.placed,
            region=self.region,
            run_assembly_name=self.assembly_name,
            run_assembly_synonym=self.synonym,
            run_assembly_status=self.assembly_status,
            run_assembly_paired_accession=self.paired,
            run_contig=self.run_contig,
            reference_sequence=self.refseq or self.genbank,
            reference_bases=self.bases,
            listed_rsids=self.listed,
            stated_version_placements=self.stated_placements,
            stated_version_contig=self.stated_contig,
            version_listings=tuple(self.version_listings),
            sequence_namings=tuple(self.namings),
            **({"version_naming": self.version_naming} if self.version_naming else {}),
            **({"listed_none": self.listed_none} if self.listed_none else {}),
            **({"variant_type": self.variant_type} if self.variant_type else {}),
            **({"contig_naming": self.contig_naming} if self.contig_naming else {}),
            supporting_sources=sources,
            following_sources=following,
        )

    def ended(
        self,
        status: RefSnpStatus,
        conflict: str | LookupOriginType | None,
        candidates: tuple[ResolvedVariantCandidate, ...] = (),
    ) -> RefSnpToolResult:
        """A lookup that reached no record ends on its own last GET; one that read a
        record ends on that record, its later GETs following it. A typed ending's
        audit sentence is its conflict and rationale (#478)."""
        conflict, kept = ending(conflict)
        if self.record_read is not None:
            last, before = self.record_read
            supporting, following = self.sources[:before], self.sources[before:]
        else:
            *supporting, last = self.sources
            following = []
        rationale = conflict or (
            "The region's sequence is named on the Run's assembly and its "
            "reference bases are read; no rsID is listed for a region."
            if status == "region_located"
            else f"{self.bodyless[0]} answered HTTP {self.bodyless[1]} with no body, "
            "read as no rsID listed there; a body-less "
            f"{self.bodyless[1]} does not establish that no variant exists at the "
            "position. "
            + (
                "The Run's reference holds the stated REF."
                if self.bases is not None
                else "An insertion's empty REF has no reference base to read."
            )
            if self.bodyless is not None
            # The one source asked, never every source.
            else f"{_SOURCE_NAMES[self.source]} lists no variant at the stated "
            "position; "
            + (
                "the Run's reference holds the stated REF."
                if self.bases is not None
                else "an insertion's empty REF has no reference base to read."
            )
        )
        verdict = (
            "accepted"
            if status in {"unlisted", "region_located"}
            # No sequence or version is named yet: the evidence settles none.
            else "ambiguous"
            if status in {"sequence_unnamed", "version_unnamed"}
            else status
        )
        return RefSnpToolResult.create(
            **(
                {"requested_rsid": self.requested_rsid}
                if self.requested_rsid is not None
                else {}
            ),
            expected_assembly_accession=self.assembly_accession,
            status=status,
            candidates=candidates,
            conflicts=(conflict,) if conflict else (),
            **({"named_placement": self.placement} if self.placement else {}),
            **kept,
            source=last,
            assessment=_assessment(last, verdict, rationale),
            lookup_source=self.source,
            eva="not_applicable",
            location=self.detail(tuple(supporting), tuple(following)),
        )

    def run(self) -> RefSnpToolResult:
        try:
            return self._run()
        except _Ended as ended:
            return ended.result

    def _run(self) -> RefSnpToolResult:
        if (failure := self.read_assembly()) is not None:
            return failure
        if self.stated.identity is None and self.rsid is None:
            # Neither source lists rsIDs at a position without an allele: it is not
            # looked up further, and its item is deferred.
            return self.ended(
                "not_found",
                PositionWithoutAlleles(
                    contig=self.stated.contig, position=self.stated.position
                ),
            )
        stated = self.stated_assembly
        # A placed position is on the Run's assembly by its placement.
        if self.placed is None and stated is not None:
            version = self.stated_version()
            if version is None:
                if self.rsid is not None and self.source == "ncbi_dbsnp":
                    # dbSNP's record names the assemblies it places the variant on:
                    # the result copies them for the model to name the version.
                    return self.unnamed_on_record(self.rsid)
                return self.version_unnamed()
            self.version_naming = version
            if not self.on_run(version.assembly_accession):
                if self.rsid is not None and self.source == "ncbi_dbsnp":
                    return self.read_on_stated_version(self.rsid)
                if (unplaced := self.place_on_run(stated)) is not None:
                    return unplaced
        if (failure := self.name_sequence()) is not None:
            return failure
        if (failure := self.read_reference()) is not None:
            return failure
        if self.stated.identity is None:
            # A position without alleles places only the rsID the request states.
            return self.read_record(self.rsid)
        if self.bases is not None and self.bases != self.stated.ref:
            conflict = (
                f"The Run's reference holds {self.bases} at {self.run_contig}:"
                f"{self.stated.position}; the request states {self.stated.ref}."
            )
            return self.ended("conflicted", conflict)
        if (failure := self.list_rsids()) is not None:
            return failure
        if not self.listed and self.version_unknown:
            # With no version stated, nothing but the Run's reference confirms the
            # position: the item's own Review decides.
            return self.ended(
                "ambiguous",
                "The request states no genome version, and "
                f"{_SOURCE_NAMES[self.source]} lists no "
                f"variant at {self.run_contig}:{self.stated.position} on the Run's "
                f"{self.assembly_name}; the Run's reference holds its REF there.",
            )
        if not self.listed:
            identity = self.stated
            return self.ended(
                "unlisted",
                None,
                (
                    ResolvedVariantCandidate(
                        assembly_accession=self.assembly_accession,
                        assembly_name=self.assembly_name,
                        contig=self.run_contig,
                        one_based_position=identity.position,
                        ref=identity.ref,
                        alt=identity.alt,
                    ),
                ),
            )
        if len(self.listed) > 1:
            return self.ended(
                "ambiguous",
                f"The source lists {len(self.listed)} variants at the stated allele: "
                f"{', '.join(self.listed)}.",
            )
        return self.read_record(self.listed[0])

    def on_run(self, accession: str) -> bool:
        """Whether an assembly is the Run's own: its accession or its paired one."""
        return accession in {self.assembly_accession, self.paired}

    def stated_version(self) -> VersionNaming | None:
        """The assembly the request's stated version names (#471 Q3 a): the model's
        naming checked by membership, else the Run's own when the stated words are
        literally one of its names; with no model (``literal``), the one row
        NCBI's search lists literally under the stated words. ``None`` when none is
        named yet; code strips, lists and picks no version."""
        written = self.stated_assembly.name
        if self.version is not None:
            return self.version
        if written in {
            self.assembly_name,
            self.synonym,
            self.assembly_accession,
            self.paired,
        }:
            return VersionNaming(
                written=written,
                assembly_accession=self.assembly_accession,
                quote=None,
                match="literal",
            )
        if not self.literal or self.report is None or self.report.taxid is None:
            return None
        found = literal_version(written, self.report.taxid, self.listing_page)
        if isinstance(found, str):
            raise _Ended(
                self.ended(
                    "incompatible",
                    f"The request states its position on {written}: {found}.",
                )
            )
        return VersionNaming(
            written=written,
            assembly_accession=found.assembly_accession,
            quote=None,
            match="literal",
        )

    def listing_page(self, key: str) -> bytes:
        """One page of NCBI's genome reports, a non-2xx answer ending the lookup."""
        status, _document = self.get_once("ncbi_taxon_assemblies", key)
        if not 200 <= status < 300:
            raise _Ended(self.answered("ncbi_taxon_assemblies", status))
        return self.last_body

    def version_unnamed(self) -> RefSnpToolResult:
        """The lookup reached a stated version no assembly is named for yet."""
        return self.ended("version_unnamed", self.version_question())

    def version_question(self) -> VersionUnnamed:
        return VersionUnnamed(
            version=self.stated_assembly.name,
            assembly_name=self.assembly_name,
            assembly_accession=self.assembly_accession,
        )

    def unnamed_on_record(self, rsid: str) -> RefSnpToolResult:
        """``version_unnamed`` on dbSNP's record read: its placements, every
        assembly it places the variant on, are copied for the model to name one."""
        current = self.current_record(rsid)
        if isinstance(current, RefSnpToolResult):
            return current
        source, body, _document, rsid = current
        before = tuple(self.sources)
        self.record_read = (source, len(before))
        result = normalize_refsnp(
            rsid=rsid,
            assembly_accession=self.assembly_accession,
            source=source,
            raw_response=body,
            requested_rsid=self.requested_rsid,
            location=self.detail(before),
            placement=self.placement,
        )
        return _retyped(result, "version_unnamed", self.version_question())

    def place_on_run(self, stated: StatedAssembly) -> RefSnpToolResult | None:
        """Move a position stated on another version onto the Run's assembly (#462 R2):
        NCBI names its sequence on the stated version and its Variation service gives
        the stated allele one equivalent on the Run's sequences. The moved position
        then goes through this same lookup, as a researcher's amendment does; any
        other answer ends here with its reason, and the item's issue Review asks
        (R1 a). No flank move runs in a Run (#449)."""
        sources = _LookupPlacementSources(self)
        try:
            named = stated_version_sequence(
                self.version_naming.assembly_accession,
                stated.name,
                lambda accession, name: self.named_on(accession, name),
            )
            moved = (
                named
                if isinstance(named, str)
                else run_equivalents(
                    self.stated, named[0], self.assembly_accession, sources
                )
            )
        except _Unplaced as unplaced:
            moved = str(unplaced)
        status: RefSnpStatus = "incompatible"
        if not isinstance(moved, str):
            spdi, on_run = moved
            if len(on_run) == 1:
                contig, start, deleted, inserted = on_run[0].split(":")
                self.placed = PlacedPosition(
                    contig=contig, position=int(start) + 1, ref=deleted, alt=inserted
                )
                self.stated = self.placed
                return None
            status = "ambiguous"
            moved = (
                f"NCBI's Variation service gives {spdi} {len(on_run)} equivalents on "
                f"the Run's sequences: {', '.join(on_run)}"
            )
        run = self.assembly_name + (f" ({self.synonym})" if self.synonym else "")
        return self.ended(
            status,
            f"The request states its position on {stated.name}; the Run's reference "
            f"is {run}, and placement finds no position there: {moved}.",
        )

    def read_assembly(self) -> RefSnpToolResult | None:
        status, document = self.get("fetch_assembly_report", self.assembly_accession)
        if not 200 <= status < 300:
            return self.answered("fetch_assembly_report", status)
        report = read_assembly_report(document, self.assembly_accession)
        if isinstance(report, AssemblyReport):
            self.report = report
            self.assembly_name, self.synonym = report.name, report.synonym
            self.paired, self.assembly_status = report.paired_accession, report.status
            return None
        detail = {
            "unread": "it holds no report",
            "unsafe": "it names the assembly unsafely",
        }.get(report, "it does not name the assembly")
        return self.ended(
            "malformed",
            Malformed(what="NCBI's report of the Run's assembly", detail=detail),
        )

    def name_sequence(self) -> RefSnpToolResult | None:
        """The Run's sequence the model named for the written contig (#467): its
        Run name is the record's RefSeq accession on a ``GCF_`` Run, else its GenBank
        one, as the Run's reference FASTA names it."""
        record = self.named_on(self.assembly_accession, self.assembly_name)
        run_contig = (
            record.refseq
            if self.assembly_accession.startswith("GCF_")
            else record.genbank
        )
        if run_contig is None:
            conflict = (
                f"The Run's {self.assembly_name} report gives {record.sequence_name} no "
                "accession of the Run's own kind."
            )
            return self.ended("incompatible", conflict)
        self.run_contig, self.refseq, self.genbank = (
            run_contig,
            record.refseq,
            record.genbank,
        )
        self.sequence_record = record
        self.length = record.length
        return None

    def read_reference(self) -> RefSnpToolResult | None:
        stated = self.stated
        span = len(stated.ref or "") or 1
        if stated.position + span - 1 > self.length:
            conflict = (
                f"{self.run_contig} ends at {self.length}; the request states "
                f"{stated.position + span - 1}."
            )
            return self.ended("incompatible", conflict)
        if not stated.ref:
            # A trimmed insertion has no REF to read.
            return None
        sequence = self.refseq or self.genbank
        key = f"{sequence}:{stated.position - 1}:{len(stated.ref)}"
        status, document = self.get("ncbi_reference_span", key)
        if not 200 <= status < 300:
            return self.answered("ncbi_reference_span", status)
        data = document.get("data") if isinstance(document, Mapping) else None
        bases = data.get("deleted_sequence") if isinstance(data, Mapping) else None
        if (
            not isinstance(data, Mapping)
            or (data.get("seq_id"), data.get("position"))
            != (sequence, stated.position - 1)
            or not isinstance(bases, str)
            or not re.fullmatch(r"[ACGTN]+", bases.upper())
            or len(bases) != len(stated.ref)
        ):
            return self.ended(
                "malformed",
                Malformed(what="NCBI's reference bases at the stated position"),
            )
        self.bases = bases.upper()
        return None

    def list_rsids(self) -> RefSnpToolResult | None:
        stated = self.stated
        if self.source == "ncbi_dbsnp":
            sequence = self.refseq or self.genbank
            spdi = f"{sequence}:{stated.position - 1}:{stated.ref}:{stated.alt}"
            status, document = self.get("ncbi_spdi_rsids", spdi)
            if not 200 <= status < 300:
                self.listing_answered("ncbi_spdi_rsids", status)
                self.listed = ()
                return None
            listed = _spdi_rsids(status, document)
            if listed is None:
                return self.ended(
                    "malformed", Malformed(what="NCBI's rsIDs at the stated allele")
                )
            self.listed = listed
            return None
        start, _, _, kind = eva_form(stated.position, stated.ref, stated.alt)
        # The model may ask EVA under another of its own types (EVA answers a type it
        # does not file the variant under with an empty 404).
        kind = self.variant_type or kind
        insdc = (
            self.paired
            if self.assembly_accession.startswith("GCF_")
            else (self.assembly_accession)
        )
        if insdc is None:
            conflict = "NCBI names no INSDC accession for the Run's assembly."
            return self.ended("incompatible", conflict)
        # The listing names the sequence in the convention the model chose, by the
        # Run's report's own column for it (EVA's INSDC by default).
        record = self.sequence_record
        column = EVA_NAMING_COLUMNS.get(self.contig_naming or "INSDC")
        name = None if column is None or record is None else getattr(record, column)
        if name is None:
            conflict = (
                f"NCBI's report of the Run's {self.assembly_name} gives "
                f"{self.run_contig} no name under EVA's {self.contig_naming}."
            )
            return self.ended("incompatible", conflict)
        status, document = self.get(
            "eva_clustered_variants_at",
            # The copied name is percent-encoded in the key, so any text NCBI writes
            # reaches EVA whole (review r2 SP6); the provider decodes it.
            eva_record_key(
                f"{insdc}:{urllib.parse.quote(name, safe='')}:{start}:{kind}",
                self.contig_naming,
            ),
        )
        if not 200 <= status < 300:
            self.listing_answered("eva_clustered_variants_at", status)
            return None
        if document == []:
            return None
        try:
            located = eva_clustered_placements(document)
        except PlacementsUnreadable as unreadable:
            return self.ended(
                "malformed",
                Malformed(
                    what="EVA's variants at the position", detail=str(unreadable)
                ),
            )
        # Each answered variant is at the stated position: the Run's INSDC assembly, a
        # name the Run's report gives the sequence (membership), its start and type.
        if any(
            (item.eva.assembly_accession, item.eva.start, item.eva.type)
            != (insdc, start, kind)
            or item.sequence not in record.names
            for item in located
        ):
            return self.ended(
                "malformed",
                Malformed(
                    what="EVA's variants at the position",
                    detail="one is not at the stated position",
                ),
            )
        self.listed = tuple(
            dict.fromkeys(f"rs{item.eva.accession}" for item in located)
        )
        return None

    def current_record(
        self, rsid: str
    ) -> (
        tuple[RetrievedSource, bytes, Mapping[str, Any] | None, str] | RefSnpToolResult
    ):
        """``rsid``'s dbSNP record, followed through each merge to its one current
        record in this lookup; every merged record stays one of the location's GETs.
        A merge back to an rsID already read is malformed, with no further GET."""
        read: list[str] = []
        while True:
            source, body = self.fetch("fetch_refsnp", rsid)
            _check_retained(source, "fetch_refsnp", body)
            read.append(rsid)
            document = _json_object(body) if 200 <= source.status_code < 300 else None
            # The record's own reading first: only a checked merge record of the
            # asked rsID authorizes the next GET.
            checked = normalize_refsnp(
                rsid=rsid,
                assembly_accession=self.assembly_accession,
                source=source,
                raw_response=body,
                requested_rsid=self.requested_rsid,
                location=self.detail(tuple(self.sources)),
            )
            if document is not None and str(document.get("refsnp_id")) != rsid[2:]:
                # Another record answered: its identity conflict is the lookup's
                # outcome, and none of its placements is read.
                return checked
            merged = checked.merged_into if checked.status == "merged" else ()
            if len(merged) != 1:
                return source, body, document, rsid
            if merged[0] in read:
                return self.record_ended(
                    source,
                    "malformed",
                    Malformed(
                        what="NCBI dbSNP's merge records",
                        detail="they form a cycle: " + " -> ".join((*read, merged[0])),
                    ),
                )
            self.sources.append(source)
            rsid = merged[0]

    def record_ended(
        self,
        source: RetrievedSource,
        status: RefSnpStatus,
        conflict: str | LookupOriginType,
    ) -> RefSnpToolResult:
        """A record the lookup read but cannot admit: its own source, no candidate."""
        conflict, kept = ending(conflict)
        return RefSnpToolResult.create(
            **(
                {"requested_rsid": self.requested_rsid}
                if self.requested_rsid is not None
                else {}
            ),
            expected_assembly_accession=self.assembly_accession,
            status=status,
            candidates=(),
            conflicts=(conflict,),
            **({"named_placement": self.placement} if self.placement else {}),
            **kept,
            source=source,
            assessment=_assessment(source, status, conflict),
            lookup_source=self.source,
            eva="not_applicable",
            location=self.detail(tuple(self.sources)),
        )

    def read_on_stated_version(self, rsid: str) -> RefSnpToolResult:
        """An rsID stated with its position on another version than the Run's: the
        record's own placement on that version is kept beside its Run candidates,
        and the stated sequence is named on that version, for the intent check to
        compare where the request places it (#369). No position is mapped; a stated
        allele binds only the Run allele that is the same allele of the record."""
        current = self.current_record(rsid)
        if isinstance(current, RefSnpToolResult):
            return current
        source, body, document, rsid = current
        before = tuple(self.sources)
        self.record_read = (source, len(before))
        alleles = _version_alleles(document)
        # The record's placements on the assembly the stated version was named
        # as: by its accession, never by matching version names.
        version = self.version_naming.assembly_accession
        placements = tuple(
            dict.fromkeys(
                item.placement for item in alleles if item.assembly_accession == version
            )
        )
        self.stated_placements = placements
        if len({item.assembly_accession for item in placements}) == 1:
            # A sequence with no RefSeq name there is the intent check's to refuse.
            self.stated_contig = self.refseq_on(placements[0].assembly_accession)
        requested, stated_found = self.corresponding_allele(alleles)
        result = normalize_refsnp(
            rsid=rsid,
            assembly_accession=self.assembly_accession,
            source=source,
            raw_response=body,
            requested_variant=requested,
            requested_rsid=self.requested_rsid,
            location=self.detail(before, tuple(self.sources[len(before) :])),
            placement=self.placement,
        )
        on_run = tuple(
            dict.fromkeys(
                item.placement.contig
                for item in alleles
                if item.assembly_accession == self.assembly_accession
            )
        )
        if self.placement is not None and self.placement not in on_run:
            return _retyped(
                result,
                "placement_unnamed",
                PlacementUnnamed(
                    source="NCBI dbSNP",
                    rsid=result.verified_rsid or rsid,
                    sequences=on_run,
                    named=self.placement,
                ),
            )
        if self.placement is None and len(on_run) > 1:
            # Which of the record's placements on the Run's assembly the stated
            # allele corresponds to is the model's to name.
            return _retyped(
                result,
                "placement_unnamed",
                PlacementUnnamed(
                    source="NCBI dbSNP",
                    rsid=result.verified_rsid or rsid,
                    sequences=on_run,
                    loci=len(on_run),
                ),
            )
        if stated_found and requested is None:
            # A stated allele the record holds there, with no one corresponding
            # Run allele, admits none. One it does not hold is the intent check's.
            return _retyped(
                result,
                "conflicted",
                f"The allele the request states on {self.stated_assembly.name} has "
                f"no one corresponding allele of {result.verified_rsid or rsid} on "
                f"the Run's {self.assembly_name}.",
            )
        return result

    def refseq_on(self, accession: str) -> str | None:
        """The stated sequence's RefSeq name on another assembly: the record the model
        named there (#467, owner D1 a), ``None`` when that record has no RefSeq name."""
        return self.named_on(accession).refseq

    def corresponding_allele(
        self, alleles
    ) -> tuple[RequestedVariantIdentity | None, bool]:
        """The Run's allele that is the stated version's stated allele: the one at
        the same index of the record's allele order on the record's one placement on the
        Run's assembly, ``None`` when there is no one such allele (several placements
        there, #468: membership, no top-level pick); and whether the record holds the
        stated allele on the stated version at all."""
        identity = self.stated.identity
        if identity is None or self.stated_contig is None:
            return None, False
        stated = (self.stated_contig, *identity[1:])
        indexes = {
            item.index
            for item in alleles
            if item.assembly_accession == self.version_naming.assembly_accession
            and (
                item.placement.contig,
                *trimmed_allele(
                    item.placement.one_based_position,
                    item.placement.ref,
                    item.placement.alt,
                ),
            )
            == stated
        }
        run = [
            item.placement
            for item in alleles
            if item.assembly_accession == self.assembly_accession
            and item.index in indexes
            and self.placement in (None, item.placement.contig)
        ]
        if len(indexes) != 1 or len(run) != 1:
            return None, bool(indexes)
        position, ref, alt = trimmed_allele(
            run[0].one_based_position, run[0].ref, run[0].alt
        )
        return RequestedVariantIdentity(run[0].contig, position, ref, alt), True

    def read_record(self, rsid: str) -> RefSnpToolResult:
        identity = self.stated.identity
        requested = (
            None if identity is None else identity._replace(contig=self.run_contig)
        )
        if self.source == "eva":
            result = look_up_eva_rsid(
                rsid=rsid,
                assembly_accession=self.assembly_accession,
                fetch=self.fetch,
                requested_variant=requested,
                requested_rsid=self.requested_rsid,
                location=self.detail(tuple(self.sources)),
                placement=self.placement,
                run_report=self.report,
                contig_naming=self.contig_naming,
            )
            if (
                self.version_unknown
                and identity is not None
                and result.status in ADMISSIBLE_LOOKUP_STATUSES
            ):
                # EVA's answer names no other version to check the position on:
                # with none stated, the item's Review decides.
                return _retyped(
                    result,
                    "ambiguous",
                    "The request states no genome version, and EVA's lookup reads "
                    f"the Run's {self.assembly_name} only; the position is not "
                    "checked on any other version.",
                )
            return result
        current = self.current_record(rsid)
        if isinstance(current, RefSnpToolResult):
            return current
        source, body, document, rsid = current
        before = tuple(self.sources)
        self.record_read = (source, len(before))
        unread = None
        if self.version_unknown:
            # No version stated: the record's placements on every other version are
            # kept, and a stated allele is listed on each of those versions too.
            alleles = _version_alleles(document)
            self.stated_placements = tuple(
                dict.fromkeys(
                    item.placement
                    for item in alleles
                    if item.assembly_accession != self.assembly_accession
                )
            )
            if identity is not None and self.listed:
                unread = self.list_on_other_versions(alleles)
        result = normalize_refsnp(
            rsid=rsid,
            assembly_accession=self.assembly_accession,
            source=source,
            raw_response=body,
            requested_variant=requested,
            requested_rsid=self.requested_rsid,
            location=self.detail(before, tuple(self.sources[len(before) :])),
            placement=self.placement,
        )
        if unread is not None and result.status in ADMISSIBLE_LOOKUP_STATUSES:
            return _retyped(result, *unread)
        others = [
            listing
            for listing in self.version_listings
            if listing.listed_rsids and set(listing.listed_rsids) != set(self.listed)
        ]
        if others and result.status in ADMISSIBLE_LOOKUP_STATUSES:
            stated = self.stated
            return _retyped(
                result,
                "ambiguous",
                "The request states no genome version, and its position names "
                "another variant on another version: "
                + "; ".join(
                    f"{listing.assembly_name} lists {', '.join(listing.listed_rsids)} "
                    f"at {listing.contig}:{stated.position}"
                    for listing in others
                )
                + f"; the Run's {self.assembly_name} lists {', '.join(self.listed)}.",
            )
        return result

    def list_on_other_versions(
        self, alleles
    ) -> tuple[RefSnpStatus, LookupOriginType] | None:
        """The rsIDs dbSNP lists at the stated allele read on each other version
        the record places it on: a position states no version by itself. A
        version whose answer cannot be read leaves the position unsettled; it is
        never read as an empty listing."""
        versions = dict.fromkeys(
            (item.assembly_accession, item.assembly_name)
            for item in alleles
            if item.assembly_accession != self.assembly_accession
        )
        named = {choice.assembly_accession for choice in self.choices}
        missing = (
            ()
            if self.literal
            else tuple(accession for accession, _ in versions if accession not in named)
        )
        if missing:
            # Every other version's sequence is the model's to name, all at once; each
            # version's names are read first, so the model can search them.
            for accession, name in versions:
                if accession in missing:
                    status, body = self.sequence_names(accession)
                    if not 200 <= status < 300:
                        raise _Ended(
                            self.answered("ncbi_sequence_names", status, f" of {name}")
                        )
                    self.readable(accession, body, name)
            raise _Ended(self.unnamed(missing))
        for accession, name in versions:
            contig = self.refseq_on(accession)
            listed: tuple[str, ...] = ()
            if contig is not None:
                stated = self.stated
                status, document = self.get(
                    "ncbi_spdi_rsids",
                    f"{contig}:{stated.position - 1}:{stated.ref}:{stated.alt}",
                )
                if not 200 <= status < 300:
                    self.listing_answered("ncbi_spdi_rsids", status, f" on {name}")
                    status, document = 200, {"data": {"rsids": []}}
                read = _spdi_rsids(status, document)
                if read is None:
                    return "malformed", Malformed(
                        what=f"NCBI's rsIDs at the stated allele on {name}"
                    )
                listed = read
            self.version_listings.append(
                VersionListing(
                    assembly_accession=accession,
                    assembly_name=name,
                    contig=contig,
                    listed_rsids=listed,
                )
            )
        return None


def _spdi_rsids(status: int, document: Any) -> tuple[str, ...] | None:
    """The rsIDs NCBI's SPDI service lists at an allele in a 2xx answer, ``None`` when it
    is not a list of rsID numbers. A non-2xx answer is the source's to be read
    (`_PositionLookup.listing_answered`, owner #468 R5 b)."""
    del status
    data = document.get("data") if isinstance(document, Mapping) else None
    numbers = data.get("rsids") if isinstance(data, Mapping) else None
    if not isinstance(numbers, list) or not all(
        _positive_int(number) for number in numbers
    ):
        return None
    return tuple(dict.fromkeys(f"rs{number}" for number in numbers))


def _quoted_in(quote: str, body: bytes) -> bool:
    """Whether ``quote`` stands in a retained answer: in its text, or in one of its JSON
    string values as the model is shown them (membership, ADR 0049)."""
    text = body.decode("utf-8", "replace")
    if quote in text:
        return True

    def holds(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(holds(item) for item in value.values())
        if isinstance(value, list):
            return any(holds(item) for item in value)
        return isinstance(value, str) and quote in value

    return holds(_json(body))


def _retyped(
    result: RefSnpToolResult, status: RefSnpStatus, conflict: str | LookupOriginType
) -> RefSnpToolResult:
    """The same lookup, read as a non-admissible outcome: its candidates kept as the
    evidence they are, its conflict named, its assessment the new verdict's."""
    conflict, kept = ending(conflict)
    material = {
        name: getattr(result, name)
        for name in RefSnpToolResult.model_fields
        if name
        not in {
            "result_id",
            "status",
            "conflicts",
            "assessment",
            "merged_into",
            "origin",
        }
        # An omitted optional field stays omitted, as the result's dump omits it.
        and getattr(result, name) not in (None, ())
    }
    if status not in {"ambiguous", "conflicted", "placement_unnamed"}:
        # Only an ambiguous, conflicted or unnamed reading keeps candidates beside it.
        material.pop("candidates", None)
    return RefSnpToolResult.create(
        **{"candidates": (), **material},
        **kept,
        status=status,
        conflicts=(conflict,),
        assessment=_assessment(
            result.source,
            "ambiguous"
            if status in {"placement_unnamed", "version_unnamed"}
            else status,
            conflict,
        ),
    )


class _VersionAllele(NamedTuple):
    assembly_accession: str
    assembly_name: str
    index: int
    placement: StatedVersionPlacement


def _version_alleles(document: Mapping[str, Any] | None) -> tuple[_VersionAllele, ...]:
    """Each alternate allele a RefSNP record places, on every placement (#468: the one
    placement reader, no top-level filter), once for each assembly a placement names,
    with its index among the placement's alleles: dbSNP lists every placement's alleles in
    the record's own allele order (#369)."""
    try:
        placements = (
            refsnp_placements(document)
            if isinstance(document, Mapping) and "primary_snapshot_data" in document
            else ()
        )
    except PlacementsUnreadable:
        return ()
    kept = []
    for placement in placements:
        for assembly in placement.dbsnp.assemblies:
            for index, allele in enumerate(placement.dbsnp.alleles):
                if allele.deleted_sequence == allele.inserted_sequence:
                    continue
                try:
                    kept.append(
                        _VersionAllele(
                            assembly.assembly_accession,
                            assembly.assembly_name,
                            index,
                            StatedVersionPlacement(
                                assembly_accession=assembly.assembly_accession,
                                assembly_name=assembly.assembly_name,
                                contig=allele.seq_id,
                                one_based_position=allele.position + 1,
                                ref=allele.deleted_sequence,
                                alt=allele.inserted_sequence,
                            ),
                        )
                    )
                except ValidationError:
                    continue
    return tuple(dict.fromkeys(kept))


# A stated version's placement steps, one mechanism for the paper placement
# (`target_request.resolver`) and the Run's position lookup: NCBI names the stated
# sequence on the stated version, and its Variation service gives the stated allele
# its equivalents on the Run's sequences.


class StatedVersionSources(Protocol):
    """The NCBI answers a stated version's placement reads: the paper placement's
    `NcbiSources`, or the Run's lookup on its own retained GETs."""

    def run_sequences(self, accession: str) -> frozenset[str]: ...

    def equivalents(self, spdi: str) -> tuple[str, ...]: ...


def literal_version_query(stated: str, taxid: int, page_token: str | None) -> str:
    """The one query a caller with no model sends for a stated version: NCBI's own
    search for the stated words, every version, one report per GenBank/RefSeq pair
    (NCBI's ``exclude_paired_reports``), page by page (#471 R4 a)."""
    query = [
        ("filters.exclude_paired_reports", "true"),
        ("filters.assembly_version", "all_assemblies"),
        ("filters.search_text", stated),
        ("page_size", "100"),
    ]
    if page_token:
        query.append(("page_token", page_token))
    return f"{taxid}?{urllib.parse.urlencode(query)}"


def literal_version(
    stated: str, taxid: int, read_page: Callable[[str], bytes]
) -> AssemblySourceRecord | str:
    """The one assembly NCBI's search lists under the stated version's exact words
    (its name, UCSC name, accession or paired accession), reading every page; or why
    none is (#471 R4 a). A caller with no model checks literal membership only, and
    a written version no row is literally named stays unplaced until a model names it
    (#383): NCBI's search does not cover UCSC names."""
    named: list[AssemblySourceRecord] = []
    token: str | None = None
    while True:
        rows, _total, token = genome_report_page(
            read_page(literal_version_query(stated, taxid, token))
        )
        named.extend(
            row
            for row in rows
            if stated
            in {
                row.assembly_name,
                row.synonym,
                row.assembly_accession,
                row.paired_accession,
            }
        )
        if token is None:
            break
    if len(named) == 1:
        return named[0]
    listed = ", ".join(
        f"{row.assembly_name} ({row.assembly_accession})" for row in named
    )
    return (
        f"NCBI's search lists no assembly of taxon {taxid} literally named {stated}"
        if not named
        else f"NCBI lists several assemblies of taxon {taxid} named {stated}: {listed}"
    )


_SPDI_ALLELE = re.compile(r"^[ACGTN]*$")


def spdis_of(document: Any) -> tuple[str, ...]:
    """The Variation service's SPDIs as text, each field checked: a sequence that is
    not one line of text without the SPDI's own ``:``, a position that is no number or
    alleles that are not bases raise ``ValueError`` (#393 Standards r1 S1). The
    sequence name is copied, never judged by its shape (#467, ADR 0049)."""

    def text(spdi: Mapping[str, Any]) -> str:
        sequence, position = spdi["seq_id"], spdi["position"]
        deleted, inserted = spdi["deleted_sequence"], spdi["inserted_sequence"]
        if not (
            _plain_name(sequence)
            and ":" not in sequence
            and type(position) is int
            and position >= 0
            and all(
                isinstance(bases, str) and _SPDI_ALLELE.match(bases)
                for bases in (deleted, inserted)
            )
        ):
            raise ValueError(f"the SPDI {spdi} is not a sequence, a position and bases")
        return f"{sequence}:{position}:{deleted}:{inserted}"

    return tuple(text(spdi) for spdi in document["data"]["spdis"])


def stated_version_sequence(
    version: str,
    version_name: str,
    name_sequence: Callable[[str, str], SequenceRecord | str],
) -> tuple[str, str] | str:
    """The stated contig's RefSeq sequence on the named stated version ``version``,
    with that accession, or why none is named. The version is the model's naming,
    checked by membership (#471 Q3 a), or a caller's literal one (R4 a); code lists,
    strips and picks no version. The sequence there is ``name_sequence(accession,
    name)``'s: the Run's lookup takes the model's named record (#467, owner D1 a),
    paper placement a record that literally holds the written contig (owner R4 a);
    either returns a reason when it names none."""
    record = name_sequence(version, version_name)
    if isinstance(record, str):
        return record
    if record.refseq is None:
        return (
            f"NCBI's {version_name} ({version}) gives {record.sequence_name} "
            "no RefSeq name"
        )
    return record.refseq, version


def run_equivalents(
    position: StatedPosition,
    sequence: str,
    run_accession: str,
    sources: StatedVersionSources,
) -> tuple[str, tuple[str, ...]] | str:
    """The SPDI of the stated allele on its stated ``sequence`` and its equivalents on
    the Run's sequences, each a same-length substitution, or why none answers."""
    if position.ref is None:
        return "a position without its alleles has no allele to ask for"
    spdi = f"{sequence}:{position.position - 1}:{position.ref}:{position.alt}"
    on_run = tuple(
        equivalent
        for equivalent in sources.equivalents(spdi)
        if equivalent.split(":")[0] in sources.run_sequences(run_accession)
    )
    if not on_run:
        return f"NCBI's Variation service gives {spdi} no equivalent on the Run's sequences"
    unstated = [equivalent for equivalent in on_run if not _substitution(equivalent)]
    if unstated:
        # Only a same-length substitution is a position a request states: one or several
        # others are no answer of this step.
        return (
            f"NCBI's Variation service gives {spdi} no substitution on the Run's "
            f"sequences ({', '.join(unstated)})"
        )
    return spdi, on_run


def _substitution(spdi: str) -> bool:
    """A same-length change: alleles of one length that differ (an SPDI whose alleles
    are equal changes nothing, Spec r1 SP3)."""
    _, _, deleted, inserted = spdi.split(":")
    return bool(deleted) and len(deleted) == len(inserted) and deleted != inserted


class _Unplaced(Exception):
    """A placement read the Run's lookup could not use; its text is the reason."""


class _Ended(Exception):
    """The lookup ended inside a step, with its result (#467)."""

    def __init__(self, result: RefSnpToolResult) -> None:
        super().__init__(result.status)
        self.result = result


class _LookupPlacementSources:
    """A stated version's placement reads on the Run's lookup's own GETs (#462 R2):
    each answer is retained as every lookup GET is. A non-2xx answer ends the lookup
    `source_answered`, for the model to read (owner #468 R4 a); a 2xx answer that is not
    the document asked for is the placement's reason, never a stop (#458)."""

    def __init__(self, lookup: "_PositionLookup") -> None:
        self.lookup = lookup

    def _read(self, operation: str, key: str, what: str, read: Callable[[Any], Any]):
        status, document = self.lookup.get_once(operation, key)
        if not 200 <= status < 300:
            raise _Ended(self.lookup.answered(operation, status, f" for {key}"))
        try:
            return read(document)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise _Unplaced(
                f"{what}'s answer for {key} did not read as expected"
            ) from (error)

    def run_sequences(self, accession: str) -> frozenset[str]:
        status, body = self.lookup.sequence_names(accession)
        if not 200 <= status < 300:
            raise _Ended(
                self.lookup.answered("ncbi_sequence_names", status, f" for {accession}")
            )
        try:
            return frozenset(
                record.refseq for record in report_rows(body) if record.refseq
            )
        except ReportUnreadable as unreadable:
            raise _Unplaced(str(unreadable)) from unreadable

    def equivalents(self, spdi: str) -> tuple[str, ...]:
        return self._read(
            "ncbi_spdi_equivalents", spdi, "NCBI's Variation service", spdis_of
        )


def look_up_region(
    *,
    region: StatedRegion,
    stated_assembly: StatedAssembly | None = None,
    assembly_accession: str,
    fetch: SourceFetch,
    max_bases: int,
    choices: tuple[SequenceChoice, ...] = (),
    words: tuple[str, ...] = (),
    version: VersionNaming | None = None,
) -> RefSnpToolResult:
    """A region target's lookup (#367, owner option b), in the shared lookup shape:
    the position lookup's own GETs name its sequence on the Run's assembly (the
    assembly report and the sequence report) and read the Run's reference bases over
    the whole region in one span GET. No rsID is listed. ``region_located`` holds the
    Run's sequence name and the bases; a sequence the Run does not name, or a region
    past its end or longer than ``max_bases``, is ``incompatible``."""
    lookup = _PositionLookup(
        position=None,
        placed=None,
        stated_assembly=stated_assembly,
        region=region,
        rsid=None,
        requested_rsid=None,
        assembly_accession=assembly_accession,
        fetch=fetch,
        choices=choices,
        words=words,
        version=version,
    )
    try:
        return _region(lookup, region, stated_assembly, max_bases)
    except _Ended as ended:
        return ended.result


def _region(
    lookup: "_PositionLookup",
    region: StatedRegion,
    stated_assembly: StatedAssembly | None,
    max_bases: int,
) -> RefSnpToolResult:
    if (failure := lookup.read_assembly()) is not None:
        return failure
    if stated_assembly is not None:
        version = lookup.stated_version()
        if version is None:
            return lookup.version_unnamed()
        lookup.version_naming = version
    if stated_assembly is not None and not lookup.on_run(
        lookup.version_naming.assembly_accession
    ):
        # A region stated on another version keeps its numbers there.
        run = lookup.assembly_name + (f" ({lookup.synonym})" if lookup.synonym else "")
        return lookup.ended(
            "incompatible",
            f"The request states its region on {stated_assembly.name}; the Run's "
            f"reference is {run}, and no mapping between them is made.",
        )
    if (failure := lookup.name_sequence()) is not None:
        return failure
    if region.end > lookup.length:
        return lookup.ended(
            "incompatible",
            f"{lookup.run_contig} ends at {lookup.length}; the region ends at "
            f"{region.end}.",
        )
    length = region.end - region.start + 1
    if length > max_bases:
        return lookup.ended(
            "incompatible", f"The region spans more than {max_bases} bases."
        )
    sequence = lookup.refseq or lookup.genbank
    status, document = lookup.get(
        "ncbi_reference_span", f"{sequence}:{region.start - 1}:{length}"
    )
    if not 200 <= status < 300:
        # NCBI's own answer, for the model to read.
        return lookup.answered("ncbi_reference_span", status)
    data = document.get("data") if isinstance(document, Mapping) else None
    bases = data.get("deleted_sequence") if isinstance(data, Mapping) else None
    if (
        not isinstance(data, Mapping)
        or (data.get("seq_id"), data.get("position")) != (sequence, region.start - 1)
        or not isinstance(bases, str)
        or not re.fullmatch(r"[ACGTN]+", bases.upper())
        or len(bases) != length
    ):
        return lookup.ended(
            "malformed", Malformed(what="NCBI's reference bases over the region")
        )
    lookup.bases = bases.upper()
    return lookup.ended("region_located", None)


def look_up_stated_position(
    *,
    position: StatedPosition | None,
    placed: PlacedPosition | None = None,
    stated_assembly: StatedAssembly | None,
    rsid: str | None,
    requested_rsid: str | None,
    assembly_accession: str,
    fetch: SourceFetch,
    choices: tuple[SequenceChoice, ...] = (),
    words: tuple[str, ...] = (),
    literal: bool = False,
    version: VersionNaming | None = None,
    placement: str | None = None,
    listed_none: str | BodylessAnswer | None = None,
    source: LookupSource,
    variant_type: str | None = None,
    contig_naming: str | None = None,
) -> RefSnpToolResult:
    """Place a request's stated position on the Run's assembly and read what the
    variant source the model chose keeps there, in the shared lookup shape (#369,
    #469).

    The Run's assembly report gives its names; a position stated on another version
    that no step placed is moved onto the Run's by NCBI's Variation service and read
    there (#462 R2, `_PositionLookup.place_on_run`), else ends with why it was not.
    With a stated rsID and dbSNP chosen, the record's own placements on the stated
    version are read instead. The model names the stated sequence on each assembly
    the lookup reads (``choices``, #467, owner D1 a), checked by membership in NCBI's
    assembly report and the researcher's ``words`` (or, with ``literal`` and no model,
    the one record literally named the contig); NCBI's reference bases at the stated REF are kept for the intent check;
    the chosen ``source`` (dbSNP through NCBI's SPDI service, or EVA at its EVA-form
    position, under the model's ``variant_type`` when it names one) lists the rsIDs at
    the stated allele. One listed rsID is read as an rsID
    lookup bound to the stated allele; none is the stated allele, ``unlisted``.
    """
    return _PositionLookup(
        position=position,
        placed=placed,
        stated_assembly=stated_assembly,
        rsid=rsid,
        requested_rsid=requested_rsid,
        assembly_accession=assembly_accession,
        fetch=fetch,
        choices=choices,
        words=words,
        literal=literal,
        version=version,
        placement=placement,
        listed_none=listed_none,
        source=source,
        variant_type=variant_type,
        contig_naming=contig_naming,
    ).run()


__all__ = [
    "AssemblyReport",
    "assembly_of",
    "read_assembly_report",
    "eva_form",
    "look_up_stated_position",
    "look_up_eva_rsid",
    "normalize_assembly_report",
    "normalize_refsnp",
]
