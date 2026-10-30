"""G15's claim flow (#376): a gene plus protein or coding change on the Run's assembly.

For each claim: the Run's assembly report (the product's ``fetch_assembly_report``) names the
Run's version, so the paper's stated version is resolved against it, literally (assembly
name, UCSC synonym, accession, or the paired GenBank accession, #471); the change is read;
the gene's transcripts come from the assembly's own GFF3, every organism's alike (#469, owner
Q4 a), each with its translation table and tags copied; each transcript's codon is read
through the product's ``ncbi_reference_span`` and translated here with its own table
(``gene_change.py``). The change is the model's typed one (#473 Q2 a): no code reads
change text. A claim naming a transcript is also read by NCBI's HGVS service, whose answer
is copied with its status, whatever the organism (#473 Q3 a). Every GET is
a product source operation (`source_operations.SOURCE_OPERATIONS`) fetched through the
product's one seam (`eva_investigation.fetch_source`) on its own surface of the failure module;
a contact the module gives up on is the claim's answer, and the other claims run on.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from bio_neb.agentic_journey.source_answers import source_answered
from bio_neb.failure import ProviderError
from bio_neb.investigation.investigation_request import literal_version_form
from bio_neb.investigation.investigation_sources import (
    AssemblyReport,
    read_assembly_report,
    spdis_of,
)
from bio_neb.providers.eva_investigation import fetch_source
from bio_neb.providers.ncbi_assembly_report import SequenceRecord, report_rows
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceResponse,
    NcbiInvestigationSourceAdapter,
)

from .annotation import AnnotatedGene, gene_from_gff
from .gene_change import Change, CodingChange, ProteinChange, decide
from .records import (
    GeneCandidate,
    GeneChangeAnswer,
    GeneChangeClaim,
    HgvsAnswer,
    TranscriptRead,
    VersionResolution,
)
from .source_failure import SourceFailed, through_failure_module

_BASES = re.compile(r"^[ACGTN]+$")
T = TypeVar("T")
Contact = Callable[[str, str, Callable[[InvestigationSourceResponse], T]], T]
"""One operation's GET of one key, read by the callable, under its failure policy."""


class NcbiSources:
    """NCBI's answers placement reads, each a product source operation: G15's (#376) and
    the resolver's (#393). An assembly's annotation download is kept under the cache and
    read again rather than asked again; every other answer is read once per run."""

    def __init__(
        self,
        ncbi: NcbiInvestigationSourceAdapter,
        cache: Path,
        contact: Contact | None = None,
    ) -> None:
        self.ncbi = ncbi
        self.cache = cache
        # One operation's GET and the reading of its answer under the operation's
        # failure policy, applied once (ADR 0031): the failure module's here, or a
        # Run Tool's, which retains each GET and its fault in the Journey.
        self._contact = contact or self._through_failure_module
        self._assemblies: dict[str, AssemblyReport | SourceFailed] = {}
        self._listings: dict[str, bytes] = {}
        self._sequences: dict[str, tuple[SequenceRecord, ...]] = {}
        self._bases: dict[tuple[str, int, int], str] = {}

    @property
    def contact_count(self) -> int:
        return self.ncbi.contact_count

    def run_assembly(self, accession: str) -> AssemblyReport:
        """The Run's report, current or any version, read once per run: a failure the
        module gave up on is every dependent claim's or target's, with no second ask."""
        if accession not in self._assemblies:
            try:
                self._assemblies[accession] = self._call(
                    lambda body: _run_assembly(accession, body),
                    "fetch_assembly_report",
                    accession,
                )
            except SourceFailed as failed:
                self._assemblies[accession] = failed
        found = self._assemblies[accession]
        if isinstance(found, SourceFailed):
            raise found
        return found

    def run_sequences(self, accession: str) -> frozenset[str]:
        """The RefSeq names of the assembly's sequences."""
        return frozenset(
            sequence.refseq
            for sequence in self.sequences(accession)
            if sequence.refseq is not None
        )

    def sequences(self, accession: str) -> tuple[SequenceRecord, ...]:
        """Every sequence of the assembly, copied from its assembly report (#467)."""
        if accession not in self._sequences:
            self._sequences[accession] = self._call(
                lambda body: tuple(report_rows(body)),
                "ncbi_sequence_names",
                accession,
            )
        return self._sequences[accession]

    def genome_reports(self, taxon_query: str) -> bytes:
        """One page of NCBI's genome reports for ``<taxid>?<query>`` (#471 R4 a),
        asked once per run, so a paper's aliases ask no more (#411)."""
        if taxon_query not in self._listings:
            self._listings[taxon_query] = self._call(
                lambda body: body, "ncbi_taxon_assemblies", taxon_query
            )
        return self._listings[taxon_query]

    def equivalents(self, spdi: str) -> tuple[str, ...]:
        """NCBI's Variation service's equivalents of one SPDI allele, on any sequence it
        covers; none when it covers none."""
        return self._call(
            lambda body: spdis_of(json.loads(body)), "ncbi_spdi_equivalents", spdi
        )

    def assembly_gene(self, accession: str, symbol: str) -> AnnotatedGene:
        """The gene from the assembly's GFF3, downloaded once and kept; a download or a
        stored copy that does not read as one is the claim's `semantic` failure."""
        stored = self.cache / "annotations" / accession / "genome_gff.zip"
        if stored.exists():
            return through_failure_module(
                "ncbi_genome_annotation",
                lambda: _read_source(lambda: gene_from_gff(stored, symbol)),
            )

        def keep(body: bytes) -> AnnotatedGene:
            if not zipfile.is_zipfile(io.BytesIO(body)):
                raise ValueError("NCBI's annotation download is not a zip")
            stored.parent.mkdir(parents=True, exist_ok=True)
            stored.write_bytes(body)
            return gene_from_gff(stored, symbol)

        return self._call(keep, "ncbi_genome_annotation", accession)

    def read_bases(self, contig: str, start: int, length: int) -> str:
        key = (contig, start, length)
        if key not in self._bases:
            span = f"{contig}:{start - 1}:{length}"
            self._bases[key] = self._call(
                lambda body: _span_bases(body, contig, start - 1, length),
                "ncbi_reference_span",
                span,
            )
        return self._bases[key]

    def hgvs(self, name: str, assembly: str, *, coding: bool) -> HgvsAnswer:
        """NCBI's HGVS service's answer for one name, copied with its status and body
        (#473 Q3 a): a 400 is NCBI's answer, not a failure. For a ``coding`` name it
        answers nucleotide alleles: every contextual SPDI it names and each one's
        equivalents on the sequences of the Run's ``assembly``, all kept. A protein
        name's alleles are residues', kept in the body for the model to read (NCBI's
        equivalents service takes nucleotides)."""

        def answered(response):
            if not source_answered(response.status_code):
                raise ProviderError(
                    "rejected", f"NCBI answered {response.status_code}."
                )
            read = coding and 200 <= response.status_code < 300
            return response, (
                _read_source(lambda: spdis_of(json.loads(response.body)))
                if read
                else ()
            )

        response, spdis = self._contact("ncbi_hgvs_contextuals", name, answered)
        return HgvsAnswer(
            hgvs=name,
            status_code=response.status_code,
            body=response.body.decode("utf-8", errors="replace"),
            spdis=spdis,
            on_run=tuple(
                dict.fromkeys(
                    equivalent
                    for spdi in spdis
                    for equivalent in self.equivalents(spdi)
                    if equivalent.split(":")[0] in self.run_sequences(assembly)
                )
            ),
        )

    def _call(self, read: Callable[[bytes], T], operation: str, key: str) -> T:
        """One product operation's GET on its own failure surface; a non-success answer
        (a 404 among them) is rejected here, and an answer that does not read as the
        expected document is `semantic`, inside the same failure policy."""

        def checked(response: InvestigationSourceResponse) -> T:
            if not 200 <= response.status_code < 300:
                raise ProviderError(
                    "rejected", f"NCBI answered {response.status_code}."
                )
            return _read_source(lambda: read(response.body))

        return self._contact(operation, key, checked)

    def _through_failure_module(self, operation: str, key: str, read):
        """The product's contact: the GET and its reading under the operation's policy;
        a fault the module gave up on is the claim's `SourceFailed`."""
        return through_failure_module(
            operation, lambda: read(fetch_source(self.ncbi, operation, key))
        )


def _read_source(read: Callable[[], T]) -> T:
    """`read` of a source's answer; one that does not read as expected is `semantic`."""
    try:
        return read()
    except (KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ProviderError(
            "semantic", f"NCBI's answer did not read as expected: {error}"
        ) from error


def _run_assembly(accession: str, body: bytes) -> AssemblyReport:
    report = read_assembly_report(json.loads(body), accession)
    if not isinstance(report, AssemblyReport) or report.taxid is None:
        raise ValueError(f"NCBI's report of {accession} does not name it: {report}")
    return report


def _span_bases(body: bytes, contig: str, position: int, length: int) -> str:
    data = json.loads(body)["data"]
    bases = data["deleted_sequence"]
    if (data["seq_id"], data["position"]) != (contig, position) or not (
        isinstance(bases, str) and _BASES.match(bases.upper()) and len(bases) == length
    ):
        raise ValueError(f"the span is not {contig}:{position}:{length}")
    return bases.upper()


def resolve_version(stated: str | None, run: AssemblyReport) -> VersionResolution:
    """Which of the Run's names the stated version literally is (#471 Q4 a)."""
    return VersionResolution(
        stated=stated,
        run_accession=run.accession,
        run_name=run.name,
        run_synonym=run.synonym,
        run_paired_accession=run.paired_accession,
        matched=None
        if stated is None
        else literal_version_form(
            stated, run.name, run.synonym, run.accession, run.paired_accession
        ),
    )


def typed_change(claim: GeneChangeClaim) -> Change | None:
    """The change the model typed (#473 Q2 a), or None when it typed none."""
    stated = claim.change
    protein = (
        None
        if stated.protein is None
        else ProteinChange(stated.protein.ref, stated.protein.codon, stated.protein.alt)
    )
    coding = (
        None
        if stated.coding is None
        else CodingChange(
            stated.coding.cds_position, stated.coding.ref, stated.coding.alt
        )
    )
    if coding is not None and protein is not None:
        return coding, protein
    return coding or protein


def answer_claim(
    sources: NcbiSources, assembly: str, claim: GeneChangeClaim
) -> GeneChangeAnswer:
    """One claim's typed answer on the Run's assembly; a source contact the failure
    module gave up on is the answer."""
    version = None
    try:
        run = sources.run_assembly(assembly)
        stated = claim.stated_assembly.name if claim.stated_assembly else None
        version = resolve_version(stated, run)
        change = typed_change(claim)
        base = {"claim": claim, "version": version}
        gene_name = claim.change.gene
        if change is None:
            reason = (
                f"the change '{claim.change.change}' is not typed: the model types "
                "its protein or coding change from the words that state it"
            )
            return GeneChangeAnswer(**base, outcome="not_checked", reason=reason)
        gene = sources.assembly_gene(assembly, gene_name)
        if gene.outcome != "found" or not gene.transcripts:
            # NCBI is asked whenever a transcript is named (Q3 a), whatever the
            # annotation answered.
            found = gene.outcome == "found"
            return GeneChangeAnswer(
                **base,
                outcome="not_found" if found else gene.outcome,
                reason=f"the annotation gives {gene_name} no protein-coding transcript"
                if found
                else gene.reason,
                hgvs=_hgvs(sources, assembly, claim, (), ()),
            )
        decision = decide(
            change,
            gene.transcripts,
            sources.read_bases,
            transcript=claim.change.transcript,
        )
        candidates = tuple(
            GeneCandidate(
                contig=candidate.contig,
                position=candidate.position,
                ref=candidate.ref,
                alt=candidate.alt,
                bases_changed=candidate.bases_changed,
                reference_codon=candidate.reference_codon,
                alt_codon=candidate.alt_codon,
                transcripts=transcripts,
            )
            for candidate, transcripts in decision.candidates
        )
        hgvs = _hgvs(sources, assembly, claim, decision.readings, candidates)
        return GeneChangeAnswer(
            **base,
            outcome=decision.outcome,
            reason=decision.reason,
            transcripts=tuple(_read(reading) for reading in decision.readings),
            candidates=candidates,
            hgvs=hgvs,
        )
    except SourceFailed as failed:
        return GeneChangeAnswer(
            claim=claim,
            outcome=None,
            reason=None,
            version=version,
            failure=failed.failure,
        )


def _hgvs(sources, assembly, claim, readings, candidates) -> HgvsAnswer | None:
    """NCBI's answer for the claim's HGVS name, when one can be built, with the
    allele equivalents on the Run's sequences that are candidates'."""
    name = _hgvs_name(claim.change, readings)
    if name is None:
        return None
    answer = sources.hgvs(name, assembly, coding=claim.change.coding is not None)
    given = {_spdi_of(candidate) for candidate in candidates}
    return answer.model_copy(
        update={"candidates": tuple(spdi for spdi in answer.on_run if spdi in given)}
    )


def _hgvs_name(stated, readings) -> str | None:
    """The HGVS name of a claim that names its transcript (#473 Q3 a): its typed coding
    change on that transcript, else its typed protein change on the protein the
    annotation gives that transcript; None when no transcript is named, or a protein
    change's transcript has no one protein in the annotation."""
    if stated.transcript is None:
        return None
    if stated.coding is not None:
        coding = stated.coding
        return f"{stated.transcript}:c.{coding.cds_position}{coding.ref}>{coding.alt}"
    proteins = {
        reading.transcript.protein
        for reading in readings
        if reading.transcript.protein is not None
    }
    if len(proteins) != 1:
        return None
    protein = stated.protein
    return f"{proteins.pop()}:p.{protein.ref}{protein.codon}{protein.alt}"


def _spdi_of(candidate) -> str:
    return (
        f"{candidate.contig}:{candidate.position - 1}:{candidate.ref}:{candidate.alt}"
    )


def _read(reading) -> TranscriptRead:
    transcript = reading.transcript
    return TranscriptRead(
        accession=transcript.accession,
        protein=transcript.protein,
        contig=transcript.contig,
        strand=transcript.strand,
        genetic_code=transcript.genetic_code,
        genetic_code_origin=transcript.genetic_code_origin,
        tags=transcript.tags,
        cds_phase=transcript.cds_phase,
        codon_number=reading.codon_number,
        reference_codon=reading.reference_codon,
        reference_residue=reading.reference_residue,
        matches=reading.matches,
        reason=reading.reason,
        unplaced=reading.unplaced,
    )
