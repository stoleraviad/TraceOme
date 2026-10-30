"""Planning's reference facts the Investigation carries (owner, #476 alignment G; sweep
Pattern G).

Planning already read NCBI's report of the Run's assembly (its dataset report) and,
when it described the genomic FASTA, the assembly's sequence names. The Investigation's
start records once which of planning's retained GETs describe the Run's assembly
(`carried_reference_sources`), and the Investigation's lookups read those bytes in
place of a new NCBI contact; NCBI is contacted only for other assemblies, or when
planning holds no report. The start also records the Run organism's taxon at species
level (`run_species_taxid`, owner #479 Q4 A), which `search_assemblies` is bound to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bio_neb.providers.ncbi_datasets import assembly_report_url

# The Investigation operation each carried planning GET stands in for.
CARRIED_OPERATIONS = ("fetch_assembly_report", "ncbi_sequence_names")

# Ranks a Run's organism may be recorded at below its species (NCBI's ranks).
_BELOW_SPECIES = frozenset(
    {
        "subspecies",
        "varietas",
        "forma",
        "strain",
        "isolate",
        "serotype",
        "serogroup",
        "biotype",
        "pathogroup",
        "genotype",
        "morph",
        "forma specialis",
    }
)


def carried_operation(
    payload: Mapping[str, Any], assembly_accession: str
) -> str | None:
    """The Investigation operation a planning ``ReferenceSourceRetrieved`` answers for
    the Run's assembly: its dataset report, or its sequence-names report; ``None`` for
    any other GET or a non-2xx answer."""
    if not 200 <= payload["status_code"] < 300:
        return None
    identity = payload["request_identity"]
    if identity == assembly_report_url(assembly_accession):
        return "fetch_assembly_report"
    name = identity.rsplit("/", 1)[-1]
    if name.startswith(f"{assembly_accession}_") and name.endswith(
        "_assembly_report.txt"
    ):
        return "ncbi_sequence_names"
    return None


def carried_reference_sources(
    events: Sequence[Mapping[str, Any]], assembly_accession: str
) -> tuple[str, ...]:
    """Planning's latest retained GET of each carried kind for the Run's assembly, by
    event id, in operation order."""
    latest: dict[str, str] = {}
    for event in events:
        if event["event_type"] != "ReferenceSourceRetrieved":
            continue
        operation = carried_operation(event["payload"], assembly_accession)
        if operation is not None:
            latest[operation] = event["event_id"]
    return tuple(latest[name] for name in CARRIED_OPERATIONS if name in latest)


def run_species_taxid(
    events: Sequence[Mapping[str, Any]], assembly_accession: str
) -> int | None:
    """The Run organism's taxon for `search_assemblies` (owner #479 Q4 A): the Run
    reference's recorded ``ncbi_tax_id``, at species level when planning's taxon
    record ranks it below species and names its species; ``None`` when planning
    recorded no description of the Run's assembly."""
    recorded = None
    for event in events:
        if event["event_type"] != "ReferenceDescriptionRecorded":
            continue
        result = event["payload"]["result"]
        for reference in result.get("references", ()):
            if reference["assembly_accession"] == assembly_accession:
                recorded = reference["record"].get("ncbi_tax_id") or recorded
    if recorded is None:
        return None
    return _species_of(events, recorded) or recorded


def _species_of(events, tax_id: int) -> int | None:
    """The species planning's taxon answer names for ``tax_id`` when it ranks it
    below species: NCBI's own classification, as its answer writes it."""
    for event in reversed(events):
        if event["event_type"] != "TaxonResolutionRecorded":
            continue
        for answer in event["payload"].get("source_answers", ()):
            body = answer.get("body")
            reports = body.get("reports", ()) if isinstance(body, Mapping) else ()
            for report in reports if isinstance(reports, list) else ():
                taxonomy = (
                    report.get("taxonomy") if isinstance(report, Mapping) else None
                )
                if (
                    not isinstance(taxonomy, Mapping)
                    or taxonomy.get("tax_id") != tax_id
                ):
                    continue
                if str(taxonomy.get("rank", "")).lower() not in _BELOW_SPECIES:
                    return None
                species = (taxonomy.get("classification") or {}).get("species") or {}
                identity = species.get("id") if isinstance(species, Mapping) else None
                return identity if isinstance(identity, int) and identity > 0 else None
    return None
