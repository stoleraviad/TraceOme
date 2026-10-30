"""Every public source operation: one GET each, on its own failure surface.

A definition names the adapter method that makes its GET, how its envelope is checked
(`eva_investigation.fetch_source`) and any limit of its own beside the failure module's
defaults. `source_contacts.source_tools()` is built from these registrations, the Agent
Tools' rows and the paper download's surface; a Run's lookups and the developer entries
fetch through the same table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bio_neb.failure import ContactLimitOverrides

_MIB = 1024 * 1024


@dataclass(frozen=True)
class SourceOperation:
    name: str
    method: str
    # A JSON metadata answer, EVA's answer (the JSON is the lookup's), both passing
    # the source's answer about the request, a 404 or a 410 among them
    # (`source_answers.source_answered`); downloaded bytes (2xx only); or a paper's
    # document (a 404 says the host holds none).
    answer: Literal["json", "eva", "bytes", "document"]
    limits: ContactLimitOverrides | None = None


@dataclass(frozen=True)
class SourceSurface:
    """A source Tool's contacts, declared by its owner: its own one surface, or its
    operations' surfaces, and its own per-call limits over the failure package's
    defaults (owner, #447 Q4). An Agent Tool declares it on its registry row."""

    operations: frozenset[str] = frozenset()
    limits: ContactLimitOverrides | None = None


def _registered(*operations: SourceOperation) -> dict[str, SourceOperation]:
    registered: dict[str, SourceOperation] = {}
    for operation in operations:
        if operation.name in registered:
            raise ValueError(f"source operation {operation.name} is registered twice")
        registered[operation.name] = operation
    return registered


SOURCE_OPERATIONS = _registered(
    # A reference companion's NCBI GETs: its md5checksums.txt listing and JSON report, a few
    # KB, and the assembly report beside the genomic FASTA, read once for the Run's reference
    # sequences. That report is read whole, so the surface takes the sequence names' bound
    # (wheat's is 9.1 MB, cap a).
    SourceOperation(
        "describe_companion",
        "describe_companions",
        "json",
        ContactLimitOverrides(response_bytes=256 * _MIB),
    ),
    # The genomic FASTA's index Tools on Galaxy's datacache: its catalogues are about
    # 1.1 MB each and keep the default bound.
    SourceOperation("galaxy_prebuilt_index", "get", "document"),
    # NCBI's dbSNP record and the Run's assembly (current or, for a replaced or
    # suppressed one, any version).
    SourceOperation("fetch_refsnp", "fetch_refsnp", "json"),
    SourceOperation("fetch_assembly_report", "fetch_assembly_report", "json"),
    # A stated position's placement GETs. An assembly's sequence names are its whole
    # assembly report: NCBI's `links` answer, then the report, whose envelope is a
    # metadata answer's. A fragmented assembly's report is large (wheat 9.1 MB,
    # loblolly pine 143 MB), read whole within the GFF3 download's bound.
    SourceOperation(
        "ncbi_sequence_names",
        "fetch_sequence_names",
        "json",
        ContactLimitOverrides(response_bytes=256 * _MIB),
    ),
    SourceOperation("ncbi_reference_span", "fetch_reference_span", "json"),
    SourceOperation("ncbi_spdi_rsids", "fetch_spdi_rsids", "json"),
    # The European Variation Archive's operations.
    SourceOperation("eva_clustered_variant", "fetch_eva_clustered_variant", "eva"),
    SourceOperation("eva_submitted_variants", "fetch_eva_submitted_variants", "eva"),
    SourceOperation(
        "eva_clustered_variants_at", "fetch_eva_clustered_variants_at", "eva"
    ),
    SourceOperation("eva_contig_alias", "fetch_eva_contig_alias", "eva"),
    # A taxon's RefSeq assemblies of every version, and a gene-plus-change claim's
    # GETs (lane Rf's). They keep the 16 MiB bound their developer tool gave them,
    # and an assembly's GFF3 download its 256 MiB (about 20 MB for a mammal's
    # assembly, development GET d7).
    SourceOperation(
        "ncbi_taxon_assemblies",
        "fetch_taxon_assemblies",
        "json",
        ContactLimitOverrides(response_bytes=16 * _MIB),
    ),
    SourceOperation(
        "ncbi_genome_annotation",
        "fetch_genome_annotation",
        "bytes",
        ContactLimitOverrides(response_bytes=256 * _MIB),
    ),
    SourceOperation(
        "ncbi_hgvs_contextuals",
        "fetch_hgvs_contextuals",
        "json",
        ContactLimitOverrides(response_bytes=16 * _MIB),
    ),
    SourceOperation(
        "ncbi_spdi_equivalents",
        "fetch_spdi_equivalents",
        "json",
        ContactLimitOverrides(response_bytes=16 * _MIB),
    ),
    # A paper's open copy: Europe PMC's search, full text and supplement zip, and the
    # PMC open-data bucket's listing and files. A document may be a large figure or
    # workbook (11.7 MB and 4.3 MB among the papers downloaded before), so the
    # document GETs take 64 MiB; search and listing keep the default. The search
    # passes Europe PMC's answer about the query, a 400 among them, with its body, as
    # a JSON lookup does: the model reads it.
    SourceOperation("europe_pmc_search", "fetch_europe_pmc_search", "json"),
    SourceOperation(
        "europe_pmc_full_text",
        "fetch_europe_pmc_full_text",
        "document",
        ContactLimitOverrides(response_bytes=64 * _MIB),
    ),
    SourceOperation(
        "europe_pmc_supplementary_files",
        "fetch_europe_pmc_supplementary_files",
        "document",
        ContactLimitOverrides(response_bytes=64 * _MIB),
    ),
    SourceOperation("pmc_open_data_listing", "fetch_pmc_open_data_listing", "document"),
    SourceOperation(
        "pmc_open_data_file",
        "fetch_pmc_open_data_file",
        "document",
        ContactLimitOverrides(response_bytes=64 * _MIB),
    ),
)


__all__ = ["SOURCE_OPERATIONS", "SourceOperation"]
