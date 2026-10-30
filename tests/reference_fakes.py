"""Reusable reference and companion providers for Journey builders."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import uuid
from datetime import UTC, datetime

from bio_neb.parabricks.planning_hitl import (
    ClarifyPlanningHitlResponse,
    PlanningHitlCandidate,
    PlanningHitlRequested,
    SelectPlanningHitlResponse,
    make_planning_hitl_responded,
)
from bio_neb.parabricks.proposal_information import (
    ContextEvidence,
    ProposalInformationPlan,
    ProposalInformationRequirement,
    RequestAssertionEvidence,
    SlotVerification,
    ToolEvidenceValidation,
)
from bio_neb.parabricks.proposal_journey import ProposalRequirementsIdentified
from bio_neb.providers.ncbi_datasets import NcbiDatasetsReferenceAdapter
from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionAdapterObservation,
    CompanionRequirementGrounding,
    CompanionSourceRecord,
    DescribeCompanionInput,
    DescribeCompanionTool,
    VerifiedReferenceBinding,
    derive_companion_work_items,
)
from bio_neb.reference_discovery.reference_facts_journey import (
    CompanionObservationRecorded,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyAdapterObservation,
    AssemblyQuery,
    AssemblySourceRecord,
    DescribeReferenceInput,
    DescribeReferenceTool,
    DiscoverReferenceAssembliesInput,
    DiscoverReferenceAssembliesTool,
    InfraspecificNames,
    RequirementGrounding,
    ResolveTaxonInput,
    ResolveTaxonTool,
    SampleAttestationInput,
    SelectedAssemblyEvidence,
    SourceObservation,
    TaxonomyAdapterObservation,
    TaxonomySourceRecord,
    VerifySampleAttestationTool,
)
from bio_neb.reference_discovery.reference_journey import (
    ReferenceAssembliesRecorded,
    ReferenceDescriptionRecorded,
    SampleAttestationRecorded,
    TaxonResolutionRecorded,
    project_reference_identity,
)

JOURNEY_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "reference-journey"))
PLAN_ID = "plan_" + "a" * 64
_REFERENCE_NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
_COMPANION_NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def reference_source(authority: str, label: str) -> SourceObservation:
    return SourceObservation(
        authority=authority,
        adapter_version="recorded/v1",
        source_version="recorded-source-v1",
        request_identity=f"https://api.ncbi.nlm.nih.gov/{label}",
        response_sha256="e" * 64,
        observed_at=_REFERENCE_NOW,
        citation_url=f"https://api.ncbi.nlm.nih.gov/{label}",
        elapsed_time_ms=4,
        response_bytes=100,
    )


def companion_source(label: str = "companions") -> SourceObservation:
    return SourceObservation(
        authority="ncbi_datasets",
        adapter_version="recorded/v1",
        source_version="recorded-source/v1",
        request_identity=f"https://api.ncbi.nlm.nih.gov/{label}",
        response_sha256="e" * 64,
        observed_at=_COMPANION_NOW,
        citation_url=f"https://api.ncbi.nlm.nih.gov/{label}",
        elapsed_time_ms=4,
        response_bytes=100,
    )


# One small gzip reference the discovered genomic FASTA names; acquisition reads
# these bytes, and their md5 is what the publisher's listing states.
SOURCE_PLAIN = b">chr-offline\nACGTACGTACGTACGT\n"
SOURCE_GZIP = gzip.compress(SOURCE_PLAIN, mtime=0)
SOURCE_MD5 = hashlib.md5(SOURCE_GZIP).hexdigest()
_FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all/offline"


def source_acquisition():
    """What acquiring ``SOURCE_GZIP`` measures; computed from the literal bytes."""
    from bio_neb.journey_path import ReferenceAcquisition

    return ReferenceAcquisition(
        source_md5=SOURCE_MD5,
        source_byte_count=len(SOURCE_GZIP),
        source_sha256=hashlib.sha256(SOURCE_GZIP).hexdigest(),
        plain_fasta_byte_count=len(SOURCE_PLAIN),
        plain_fasta_sha256=hashlib.sha256(SOURCE_PLAIN).hexdigest(),
    )


def companion_record(
    asset_id: str = "ncbi-datasets:GCF_000001405.40:GENOMIC_NUCLEOTIDE_FASTA",
    *,
    assembly_accession: str = "GCF_000001405.40",
    role: str = "genomic_fasta",
    authority: bool = True,
    observed_source: bool = False,
):
    return CompanionSourceRecord(
        asset_id=asset_id,
        assembly_accession=assembly_accession,
        role=role,
        publisher="NCBI Datasets",
        release_version=assembly_accession if authority else None,
        representation="GENOMIC_NUCLEOTIDE_FASTA",
        authority_locator=(
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            f"{assembly_accession}/download_summary"
            if authority
            else None
        ),
        # The genomic FASTA's publisher file, md5 and HEAD size, as the NCBI
        # adapter observes them.
        **(
            {
                "source_locator": f"{_FTP}/{assembly_accession}_genomic.fna.gz",
                "checksum_algorithm": "md5",
                "checksum": SOURCE_MD5,
                "byte_size": len(SOURCE_GZIP),
                "checksum_authority_locator": f"{_FTP}/md5checksums.txt",
            }
            if observed_source and authority and role == "genomic_fasta"
            else {}
        ),
    )


# NCBI's assembly report of the planning fakes' reference: one 16-base sequence named
# chr-offline in NCBI's FASTA, as `SOURCE_PLAIN` holds it.
OFFLINE_ASSEMBLY_REPORT = (
    "# Sequence-Name\tSequence-Role\tAssigned-Molecule\tAssigned-Molecule-Location/Type"
    "\tGenBank-Accn\tRelationship\tRefSeq-Accn\tAssembly-Unit\tSequence-Length"
    "\tUCSC-style-name\n"
    "offline\tassembled-molecule\t1\tChromosome\tCM-offline.1\t=\tchr-offline"
    "\tPrimary Assembly\t16\tchrOffline\n"
)


class AssemblyReportFake:
    """NCBI's assembly report beside a verified genomic FASTA (#464): ``report``, by
    default `OFFLINE_ASSEMBLY_REPORT`; ``reports`` lists each one read. Like the fakes'
    other reads it makes no GET, unless ``answers``: then it answers as NCBI's FTP
    does, so planning retains the report (#458) and an Investigation carries it."""

    report: str = OFFLINE_ASSEMBLY_REPORT
    answers: bool = False

    def fetch_assembly_sequences(self, locator: str):
        from bio_neb.agentic_journey.source_answers import answer_received
        from bio_neb.providers.ncbi_datasets import parse_assembly_report

        self.__dict__.setdefault("reports", []).append(locator)
        body = self.report.encode()
        if self.answers:
            answer_received(
                locator.removesuffix("_genomic.fna.gz") + "_assembly_report.txt",
                200,
                body,
            )
        return parse_assembly_report(body)

    def fetch_accession_sequences(self, accession: str):
        """The report of ``accession`` read before any companion is described (#459
        D1): ``reports_by_accession`` answers another assembly's, else ``report``."""
        from bio_neb.agentic_journey.source_answers import answer_received
        from bio_neb.providers.ncbi_datasets import parse_assembly_report

        self.__dict__.setdefault("reports", []).append(f"accession:{accession}")
        report = getattr(self, "reports_by_accession", {}).get(accession, self.report)
        body = report.encode()
        if self.answers:
            answer_received(f"{_FTP}/{accession}_assembly_report.txt", 200, body)
        return parse_assembly_report(body)


class CompanionAdapter(AssemblyReportFake):
    """NCBI's companions of the fakes' reference; ``contact_count`` counts the
    descriptions, ``reports`` the assembly reports read (#464)."""

    contact_count = 0

    def __init__(
        self,
        assembly_accession: str,
        asset_ids=("source-owned-opaque-id",),
        report: str = OFFLINE_ASSEMBLY_REPORT,
    ):
        self.assembly_accession = assembly_accession
        self.asset_ids = asset_ids
        self.report = report
        self.reports: list[str] = []

    def describe_companions(self, assembly_accession: str, role: str):
        self.contact_count += 1
        return CompanionAdapterObservation(
            records=tuple(
                companion_record(
                    asset_id,
                    assembly_accession=assembly_accession,
                    role=role,
                    observed_source=True,
                )
                for asset_id in self.asset_ids
            ),
            source=companion_source("agent-companion"),
        )


def companion_arguments(tool):
    # A family's call reasoning (Nemotron's ``reasoning``) has no const.
    return {
        name: tool.parameters_json_schema["properties"][name].get("const", "Scripted.")
        for name in tool.parameters_json_schema["required"]
    }


def _organism_names(value):
    """Every organism name a record's copied result writes."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"organism_name", "scientific_name"} and isinstance(item, str):
                yield item
            else:
                yield from _organism_names(item)
    elif isinstance(value, list):
        for item in value:
            yield from _organism_names(item)


def recipe_selection_arguments(journey, tool, *, judgment="diploid", quote=None):
    """The model's recipe selection: the offered recipe when several are, and its
    ploidy judgment citing the Run's organism record whose result holds ``quote``,
    by default the first offered record's own organism name (#470: the record the
    reference stage kept, #469 Q2)."""
    properties = tool.parameters_json_schema["properties"]
    by_id = {event["event_id"]: event for event in journey.read()}
    offered = properties["organism_record_event_id"]["enum"]
    if quote is None:
        quote = next(_organism_names(by_id[offered[0]]["payload"]["result"]))
    record = next(
        event_id
        for event_id in offered
        if quote in json.dumps(by_id[event_id]["payload"]["result"])
    )
    return {
        **(
            {"candidate_id": properties["candidate_id"]["enum"][0]}
            if "candidate_id" in properties
            else {}
        ),
        "organism_ploidy": judgment,
        "organism_record_event_id": record,
        "organism_quote": quote,
    }


class ReferenceAdapter:
    # The fake agrees with the real port's one-taxon contract (ADR 0020).
    taxon_term_refusal = NcbiDatasetsReferenceAdapter.taxon_term_refusal

    def __init__(
        self,
        *,
        ambiguous: bool = False,
        scientific_name: str = "Homo sapiens",
        ncbi_tax_id: int = 9606,
        accession: str = "GCF_000001405.40",
        newer: str | None = None,
    ) -> None:
        """NCBI's taxon ``scientific_name`` (``ncbi_tax_id``), whose discovery lists
        ``accession`` (by default human's GRCh38.p14); with ``newer``, ``accession``
        is ``replaced`` and NCBI names ``newer`` its current version, listed after it
        (#459: an older reference)."""
        self.ambiguous = ambiguous
        self.scientific_name = scientific_name
        self.ncbi_tax_id = ncbi_tax_id
        self.accession = accession
        self.newer = newer
        self.contact_count = 0

    def resolve_taxon(self, query: str) -> TaxonomyAdapterObservation:
        self.contact_count += 1
        return TaxonomyAdapterObservation(
            records=(
                TaxonomySourceRecord(
                    ncbi_tax_id=self.ncbi_tax_id,
                    scientific_name=self.scientific_name,
                    rank="species",
                ),
            ),
            source=reference_source("ncbi_taxonomy", f"taxonomy/{query}"),
        )

    def discover_assemblies(
        self, ncbi_tax_id: int, query: AssemblyQuery
    ) -> AssemblyAdapterObservation:
        self.contact_count += 1
        # NCBI's search text narrows the listing to the isolate it names.
        searched = query.search_text[0] if query.search_text else None
        if self.ambiguous and searched is None:
            records = (
                self._assembly(self.accession, ncbi_tax_id, "isolate-A"),
                self._assembly("GCF_000009999.1", ncbi_tax_id, "isolate-B"),
            )
        else:
            records = (
                self._assembly(
                    "GCF_000009999.1" if searched else self.accession,
                    ncbi_tax_id,
                    searched,
                ),
            )
            if self.newer is not None and not searched:
                records = (*records, self._assembly(self.newer, ncbi_tax_id, None))
        return AssemblyAdapterObservation(
            records=records,
            source=reference_source("ncbi_datasets", f"genome/taxon/{ncbi_tax_id}"),
            total_count=len(records),
        )

    def describe_assembly(self, assembly_accession: str) -> AssemblyAdapterObservation:
        self.contact_count += 1
        return AssemblyAdapterObservation(
            records=(self._assembly(assembly_accession, self.ncbi_tax_id, None),),
            source=reference_source(
                "ncbi_datasets", f"genome/accession/{assembly_accession}"
            ),
            total_count=1,
        )

    def _assembly(
        self, accession: str, ncbi_tax_id: int, strain: str | None
    ) -> AssemblySourceRecord:
        replaced = self.newer is not None and accession == self.accession
        return AssemblySourceRecord(
            assembly_accession=accession,
            current_accession=self.newer if replaced else None,
            assembly_name=f"assembly-{accession}",
            organism_name=self.scientific_name,
            ncbi_tax_id=ncbi_tax_id,
            infraspecific_names=InfraspecificNames(isolate=strain),
            assembly_status="replaced" if replaced else "current",
            refseq_category="reference genome",
            assembly_level="chromosome",
        )


def reference_choice_arguments(journey, reference, accession=None):
    """The model's choice of a row of the latest discovery page: the row the
    researcher selected, else ``accession``, else the page's first row, quoting the
    researcher's own words for the organism (#471 Q2 a)."""
    from types import SimpleNamespace

    from bio_neb.reference_discovery.reference_action_policy import (
        applicable_reference_receipt,
        reference_assertion,
        reference_discovery,
    )

    page = reference_discovery(reference)
    receipt = applicable_reference_receipt(reference)
    selected = receipt.assertion.selected_candidate_id if receipt else None
    accession = (
        selected.removeprefix("assembly:")
        if selected
        else accession or page.rows[0].assembly_accession
    )
    assertion = reference_assertion(
        SimpleNamespace(journey=journey), reference, "taxon_identity"
    )
    return {
        "discovery_event_id": reference.discovery_event_id,
        "assembly_accession": accession,
        "quote": assertion.asserted_value,
        "reason": "The listed assembly matches the stated sample organism and analysis needs.",
    }


def sample_arguments(journey):
    """The sample name a model types from the researcher's latest statement about the
    sample, quoting it (#472 Q2 a): the plan's typed value, else the last name the
    answer writes in the sample format (the model's reading, not the product's)."""
    from types import SimpleNamespace

    from bio_neb.application.planning_read import read_planning
    from bio_neb.reference_discovery.reference_action_policy import reference_assertion
    from bio_neb.reference_discovery.reference_identity_tool import (
        SAMPLE_IDENTIFIER_RULE,
    )

    assertion = reference_assertion(
        SimpleNamespace(journey=journey),
        read_planning(journey).reference,
        "sample_identity",
    )
    sample = assertion.value or SAMPLE_IDENTIFIER_RULE.stated(assertion.asserted_value)[
        -1
    ].rstrip(".")
    return {"sample_identifier": sample, "quote": sample}


def supplied_reference_arguments(journey, properties):
    """The accession a model types from the latest researcher statement the Tool lets
    it cite that writes one, with that statement's event id; None when none does."""
    from bio_neb.application.planning_read import planning_evidence
    from bio_neb.reference_discovery.reference_identity_tool import (
        ASSEMBLY_ACCESSION_RULE,
    )
    from bio_neb.reference_discovery.reference_journey import researcher_text

    citable = set(properties["assertion_event_id"].get("enum", ()))
    for event in reversed(planning_evidence(journey)):
        text = researcher_text(event)
        if event["event_id"] not in citable or text is None:
            continue
        accessions = ASSEMBLY_ACCESSION_RULE.stated(text)
        if accessions:
            return {
                "assembly_accession": accessions[-1],
                "assertion_event_id": event["event_id"],
                "quote": accessions[-1],
            }
    return None


# The order a well-informed model takes the reference and companion stages in.
REFERENCE_CALL_ORDER = (
    "record_supplied_reference",
    "attest_sample",
    "resolve_taxon",
    "discover_assemblies",
    "record_reference_choice",
    "inspect_reference",
    "describe_companion",
    "search_prebuilt_indexes",
    "read_prebuilt_index",
    "suggest_assembly_with_index",
    "record_prebuilt_index_choice",
)


def untried_taxon_terms(journey) -> tuple[str, ...]:
    """The taxon terms a model would try next: the stated TaxId, then the quote."""
    from types import SimpleNamespace

    from bio_neb.application.planning_read import read_planning
    from bio_neb.reference_discovery.reference_action_policy import reference_assertion

    view = read_planning(journey).reference
    assertion = reference_assertion(
        SimpleNamespace(journey=journey), view, "taxon_identity"
    )
    if assertion is None:
        return ()
    tried = {
        f.result.get("query")
        for f in view.facts
        if f.event_type == "TaxonResolutionRecorded"
    }
    terms = (
        None if assertion.value is None else str(assertion.value),
        assertion.asserted_value,
    )
    return tuple(
        term for term in dict.fromkeys(terms) if term is not None and term not in tried
    )


def reference_call(journey, tools, *, index_first=False):
    """The next reference or companion call a well-informed model makes, choosing its
    own terms from the request and the recorded results; None when none applies.
    With ``index_first`` the model checks Galaxy's indexes before it describes any
    companion (#459 D1: the index is the verified assembly's, not its FASTA's).
    """
    from types import SimpleNamespace

    from bio_neb.application.planning_read import read_planning
    from bio_neb.reference_discovery.reference_action_policy import selected_reference
    from bio_neb.reference_discovery.reference_identity_tool import selected_accession
    from bio_neb.reference_discovery.reference_tools import reference_facts_projection

    view = read_planning(journey)
    order = REFERENCE_CALL_ORDER
    if index_first:
        order = (
            *(n for n in order if n != "describe_companion"),
            "describe_companion",
        )
    for name in order:
        if name not in tools:
            continue
        properties = tools[name].parameters_json_schema["properties"]
        bound = {k: v["const"] for k, v in properties.items() if "const" in v}
        if name == "record_supplied_reference":
            supplied = supplied_reference_arguments(journey, properties)
            if supplied is None:
                continue
            return name, {
                **bound,
                **supplied,
                "reason": "Inspect the exact assembly supplied in the researcher request.",
            }
        if name == "attest_sample":
            return name, sample_arguments(journey)
        if name == "resolve_taxon":
            terms = untried_taxon_terms(journey)
            if view.reference.resolved_taxon is None and terms:
                return name, {"term": terms[0]}
        elif name == "discover_assemblies":
            if (
                arguments := _discovery_call(journey, view.reference, index_first)
            ) is not None:
                return name, arguments
        elif name == "record_reference_choice":
            return name, reference_choice_arguments(journey, view.reference)
        elif name == "inspect_reference":
            accession = selected_accession(selected_reference(view.reference))
            # A reference revalidation clears the current description, so the
            # chosen accession is inspected again.
            if view.reference.description_event_id is not None and any(
                f.event_type == "ReferenceDescriptionRecorded"
                and f.result.get("assembly_accession") == accession
                for f in view.reference.facts
            ):
                continue
            return name, {"accession": accession}
        elif name == "describe_companion":
            companions = reference_facts_projection(SimpleNamespace(journey=journey))
            accession = view.reference.verified_reference.assembly_accession
            for work in companions.work_items:
                if work.work_item_id not in companions.pending_work_item_ids:
                    continue
                selections = [
                    d
                    for d in companions.asset_dispositions
                    if d.work_item_id == work.work_item_id
                ]
                selection_id = selections[-1].response_event_id if selections else None
                if any(
                    f.work_item_id == work.work_item_id
                    and f.tool_input["assembly_accession"] == accession
                    and f.tool_input["selection_event_id"] == selection_id
                    for f in companions.facts
                ):
                    continue
                return name, {"accession": accession, "asset": work.role}
        elif name in PREBUILT_INDEX_TOOLS:
            call = _prebuilt_index_call(journey, name)
            if call is not None:
                return name, call

    return None


PREBUILT_INDEX_TOOLS = (
    "search_prebuilt_indexes",
    "read_prebuilt_index",
    "record_prebuilt_index_choice",
    "suggest_assembly_with_index",
)


def _discovery_call(journey, reference, index_first):
    """A discovery a well-informed model makes: the first, or, for a model that checks
    Galaxy first (``index_first``), during the index work when Galaxy lists no index of
    the verified assembly (#459 live a1 F1)."""
    taxid = reference.resolved_taxon.ncbi_tax_id
    if reference.verified_reference is not None:
        if index_first and _another_assembly_wanted(journey):
            return {"taxid": taxid, "assembly_version": "all_assemblies"}
        return None
    if reference.discovery_event_id is None:
        return {"taxid": taxid}
    return None


def _another_assembly_wanted(journey) -> bool:
    """Whether a well-informed model lists the organism's assemblies during the index
    work: Galaxy listed no index of the verified assembly, and no page names its newer
    version yet."""
    from types import SimpleNamespace

    from bio_neb.reference_discovery.reference_facts_journey import (
        validate_prebuilt_index_fact,
    )
    from bio_neb.reference_discovery.reference_tools import reference_facts_projection

    work = reference_facts_projection(SimpleNamespace(journey=journey)).prebuilt_index
    if work is None or work.choice is not None or not work.listing_event_ids:
        return False
    by_id = {e["event_id"]: e for e in journey.read()}
    listings = [validate_prebuilt_index_fact(by_id[i]) for i in work.listing_event_ids]
    verified = work.grounding.assembly_accession
    if any(
        getattr(listing.result, "candidates", ())
        for listing in listings
        if listing.tool_input.search_text == verified
    ):
        return False
    if _newer_of(journey, verified) is not None:
        return False
    # Once listed, the organism's assemblies are read, not listed again.
    position = {e["event_id"]: i for i, e in enumerate(journey.read())}
    first_listing = min(position[i] for i in work.listing_event_ids)
    return not any(
        e["event_type"] == "ReferenceAssembliesRecorded"
        and position[e["event_id"]] > first_listing
        for e in journey.read()
    )


def _newer_of(journey, accession):
    """The current version NCBI's discovery rows name for ``accession``, if any: a
    copied ``current_accession`` the model reads (#459)."""
    from bio_neb.application.planning_read import read_planning

    reference = read_planning(journey).reference
    for fact in reference.facts:
        if fact.event_type != "ReferenceAssembliesRecorded":
            continue
        for row in fact.result.get("rows", ()):
            if row["assembly_accession"] == accession and row.get("current_accession"):
                return row["current_accession"]
    return None


def _prebuilt_index_call(journey, name):
    """A well-informed model's index work (#464): search BRC for the verified
    accession, read the first listed candidate, choose a read that holds every
    requested sequence, else none; after a revise reopens it, choose again. When
    Galaxy lists no index of the verified accession and NCBI names a newer version
    of it, search and read that version's index against its own report (#459)."""
    from types import SimpleNamespace

    from bio_neb.reference_discovery.reference_facts_journey import (
        validate_prebuilt_index_fact,
    )
    from bio_neb.reference_discovery.reference_tools import reference_facts_projection

    work = reference_facts_projection(SimpleNamespace(journey=journey)).prebuilt_index
    if work is None or work.choice is not None:
        return None
    verified = work.grounding.assembly_accession
    by_id = {e["event_id"]: e for e in journey.read()}
    listings = [validate_prebuilt_index_fact(by_id[i]) for i in work.listing_event_ids]
    reads = {i: validate_prebuilt_index_fact(by_id[i]) for i in work.read_event_ids}
    holding = [
        i
        for i, read in reads.items()
        if read.bound_candidate() is not None
        and read.tool_input.assembly_accession is None
    ]
    searched = [listing.tool_input.search_text for listing in listings]
    newer = _newer_of(journey, verified)
    listed_verified = any(
        getattr(listing.result, "candidates", ())
        for listing in listings
        if listing.tool_input.search_text == verified
    )
    if name == "search_prebuilt_indexes":
        if not listings:
            return {"repository": "brc", "search_text": verified}
        if not listed_verified and newer is not None and newer not in searched:
            return {"repository": "brc", "search_text": newer}
        return None
    other = [
        i
        for i, read in reads.items()
        if read.bound_candidate() is not None
        and read.tool_input.assembly_accession is not None
    ]
    if name == "suggest_assembly_with_index":
        # Galaxy holds no index of the verified assembly but its newer version's.
        if holding or not other or work.suggestion_event_ids:
            return None
        return {
            "read_event_id": other[-1],
            "source": "ncbi_dbsnp",
            "reason": "Galaxy has no index of the requested assembly; its current "
            "version's index holds that version's main assembly.",
        }
    if name == "read_prebuilt_index":
        if holding or other:
            return None
        read_values = {read.tool_input.value for read in reads.values()}
        for event_id, listing in zip(work.listing_event_ids, listings, strict=True):
            other = listing.tool_input.search_text != verified
            for candidate in getattr(listing.result, "candidates", ()):
                if candidate.value not in read_values:
                    return {
                        "listing_event_id": event_id,
                        "value": candidate.value,
                        **({"assembly_accession": newer} if other else {}),
                    }
        return None
    return {
        # No index cites no read: the model leaves read_event_id out.
        "index": "chosen" if holding else "none",
        **({"read_event_id": holding[-1]} if holding else {}),
        "reason": "Galaxy's index of the verified assembly holds every requested "
        "sequence."
        if holding
        else "No listed Galaxy index holds every requested sequence.",
    }


def reference_const_arguments(tool, **choices):
    arguments = {}
    for name in tool.parameters_json_schema.get("required", []):
        schema = tool.parameters_json_schema["properties"][name]
        if "const" in schema:
            arguments[name] = schema["const"]
        elif name in choices:
            arguments[name] = choices[name]
        elif name == "max_candidates":
            arguments[name] = 3
        else:
            arguments[name] = None
    return arguments


INPUT_SET_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "reference-input-set"))
SAMPLE_REQUIREMENT_ID = "requirement_" + "b" * 32
TAXON_REQUIREMENT_ID = "requirement_" + "c" * 32
REFERENCE_REQUIREMENT_ID = "requirement_" + "d" * 32


def identity(label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reference-journey:{label}"))


def payload(fact) -> dict[str, object]:
    value = fact.model_dump(mode="json")
    value.pop("event_type")
    return value


def append(
    timeline: list[dict[str, object]],
    label: str,
    event_type: str,
    event_payload: dict[str, object],
) -> dict[str, object]:
    event = {
        "journey_id": JOURNEY_ID,
        "event_id": identity(label),
        "event_type": event_type,
        "parent_ids": [timeline[-1]["event_id"]] if timeline else [],
        "payload": event_payload,
    }
    timeline.append(event)
    return event


class FakeAdapter:
    taxon_term_refusal = NcbiDatasetsReferenceAdapter.taxon_term_refusal

    def __init__(self, taxonomy, discovery, exact) -> None:
        self.taxonomy = taxonomy
        self.discovery = discovery
        self.exact = exact
        self.contact_count = 0

    def resolve_taxon(self, query: str):
        del query
        self.contact_count += 1
        return self.taxonomy

    def discover_assemblies(self, ncbi_tax_id: int, query):
        del ncbi_tax_id, query
        self.contact_count += 1
        return self.discovery

    def describe_assembly(self, assembly_accession: str):
        del assembly_accession
        self.contact_count += 1
        return self.exact


def assembly(accession: str) -> AssemblySourceRecord:
    return AssemblySourceRecord(
        assembly_accession=accession,
        assembly_name=f"assembly-{accession}",
        organism_name="Homo sapiens",
        ncbi_tax_id=9606,
        assembly_status="current",
        refseq_category="reference genome",
        assembly_level="chromosome",
    )


def plan_fact(
    sample_assertion_event_id: str,
    taxon_assertion_event_id: str,
    context_event_id: str,
) -> ProposalRequirementsIdentified:
    validation = ToolEvidenceValidation(requirement="Verify with exact Tool evidence.")
    plan = ProposalInformationPlan(
        plan_id=PLAN_ID,
        request_event_id=context_event_id,
        registration_event_id=identity("registration"),
        inspection_event_id=identity("inspection"),
        input_set_id=INPUT_SET_ID,
        requirements=(
            ProposalInformationRequirement(
                requirement_id=SAMPLE_REQUIREMENT_ID,
                kind="sample_identity",
                subject_key="sample_identity",
                domain="scientific",
                meaning="Exact researcher-attested sample identity.",
                knowledge_state="asserted",
                value="sample-001",
                evidence=(
                    RequestAssertionEvidence(
                        source_event_id=sample_assertion_event_id,
                        start_offset=0,
                        end_offset=10,
                        exact_text="sample-001",
                    ),
                ),
                acquisition_route="use_supplied_assertion",
                validation=validation,
                identifier_rule="sample_identifier",
                verification=SlotVerification(source=None),
            ),
            ProposalInformationRequirement(
                requirement_id=TAXON_REQUIREMENT_ID,
                kind="taxon_identity",
                subject_key="taxon_identity",
                domain="scientific",
                meaning="Exact NCBI taxon identity.",
                knowledge_state="to_find",
                value=None,
                evidence=(
                    RequestAssertionEvidence(
                        source_event_id=taxon_assertion_event_id,
                        start_offset=11,
                        end_offset=23,
                        exact_text="Homo sapiens",
                    ),
                ),
                acquisition_route="validate_with_tool",
                validation=validation,
                verification=SlotVerification(source="NCBI Taxonomy"),
            ),
            ProposalInformationRequirement(
                requirement_id=REFERENCE_REQUIREMENT_ID,
                kind="reference_assembly_identity",
                subject_key="reference_assembly",
                domain="scientific",
                meaning="Exact current reference assembly identity.",
                knowledge_state="to_find",
                value=None,
                evidence=(ContextEvidence(source_event_id=context_event_id),),
                acquisition_route="validate_with_tool",
                validation=validation,
                verification=SlotVerification(source="NCBI Datasets"),
            ),
        ),
    )
    return ProposalRequirementsIdentified(
        attempt_id=identity("planning-attempt"),
        input_set_id=INPUT_SET_ID,
        plan=plan,
    )


def attempt_start(timeline: list[dict[str, object]], label: str) -> str:
    attempt_id = identity(label)
    append(
        timeline,
        f"{label}-start",
        "AttemptStarted",
        {
            "attempt_id": attempt_id,
            "agent_role": "planning_agent",
            "input_set_id": INPUT_SET_ID,
        },
    )
    return attempt_id


def grounding(
    requirement_id: str,
    kind: str,
    subject_key: str,
    *sources: str,
) -> RequirementGrounding:
    return RequirementGrounding(
        journey_id=JOURNEY_ID,
        plan_id=PLAN_ID,
        requirement_id=requirement_id,
        kind=kind,
        subject_key=subject_key,
        source_event_ids=tuple(sources),
    )


def build_timeline() -> tuple[list[dict[str, object]], dict[str, int]]:
    timeline: list[dict[str, object]] = []
    context = append(timeline, "context", "ResearchRequestRegistered", {})
    # The researcher's words the sample name is typed from.
    sample_assertion = append(
        timeline,
        "sample-assertion",
        "ResearchRequestRegistered",
        {"prompt": "The sample is sample-001."},
    )
    taxon_assertion = append(timeline, "taxon-assertion", "UserAssertion", {})
    plan = plan_fact(
        sample_assertion["event_id"], taxon_assertion["event_id"], context["event_id"]
    )
    plan_event = append(timeline, "plan", plan.event_type, payload(plan))
    stages = {"plan": len(timeline)}

    sample_attempt = attempt_start(timeline, "sample-attempt")
    sample_grounding = grounding(
        SAMPLE_REQUIREMENT_ID,
        "sample_identity",
        "sample_identity",
        plan_event["event_id"],
        sample_assertion["event_id"],
    )
    sample_input = SampleAttestationInput(
        grounding=sample_grounding,
        input_set_id=INPUT_SET_ID,
        sample_identifier="sample-001",
        assertion_event_id=sample_assertion["event_id"],
        quote="sample-001",
    )
    sample_result = VerifySampleAttestationTool().verify(
        sample_input,
        expected_grounding=sample_grounding,
        expected_input_set_id=INPUT_SET_ID,
        expected_sample_identifier="sample-001",
    )
    sample_event = append(
        timeline,
        "sample-result",
        "SampleAttestationRecorded",
        payload(
            SampleAttestationRecorded(
                attempt_id=sample_attempt,
                tool_input=sample_input,
                result=sample_result,
            )
        ),
    )
    stages["sample"] = len(timeline)

    taxonomy = TaxonomyAdapterObservation(
        records=(
            TaxonomySourceRecord(
                ncbi_tax_id=9606,
                scientific_name="Homo sapiens",
                rank="species",
            ),
        ),
        source=reference_source("ncbi_taxonomy", "taxonomy/9606"),
    )
    assemblies = AssemblyAdapterObservation(
        records=(assembly("GCF_000001405.40"), assembly("GCF_000009999.1")),
        source=reference_source("ncbi_datasets", "genome/taxon/9606"),
        total_count=2,
    )
    exact = AssemblyAdapterObservation(
        records=(assembly("GCF_000009999.1"),),
        source=reference_source("ncbi_datasets", "genome/accession/GCF_000009999.1"),
        total_count=1,
    )
    adapter = FakeAdapter(taxonomy, assemblies, exact)

    taxon_attempt = attempt_start(timeline, "taxon-attempt")
    taxon_grounding = grounding(
        TAXON_REQUIREMENT_ID,
        "taxon_identity",
        "taxon_identity",
        plan_event["event_id"],
        taxon_assertion["event_id"],
        sample_event["event_id"],
    )
    taxon_input = ResolveTaxonInput(
        grounding=taxon_grounding,
        query="Homo sapiens",
        assertion_event_id=taxon_assertion["event_id"],
    )
    taxon_result = ResolveTaxonTool().resolve(
        taxon_input,
        expected_grounding=taxon_grounding,
        expected_query="Homo sapiens",
        adapter=adapter,
    )
    taxon_event = append(
        timeline,
        "taxon-result",
        "TaxonResolutionRecorded",
        payload(
            TaxonResolutionRecorded(
                attempt_id=taxon_attempt,
                tool_input=taxon_input,
                result=taxon_result,
                source_answers=(),
            )
        ),
    )
    stages["taxon"] = len(timeline)
    taxon_candidate = taxon_result.candidates[0]
    resolved_taxon = project_reference_identity(timeline).resolved_taxon

    discovery_attempt = attempt_start(timeline, "discovery-attempt")
    reference_grounding = grounding(
        REFERENCE_REQUIREMENT_ID,
        "reference_assembly_identity",
        "reference_assembly",
        plan_event["event_id"],
        context["event_id"],
        taxon_event["event_id"],
    )
    discovery_input = DiscoverReferenceAssembliesInput(
        grounding=reference_grounding,
        taxon=resolved_taxon,
    )
    discovery_result = DiscoverReferenceAssembliesTool().discover(
        discovery_input,
        expected_grounding=reference_grounding,
        expected_taxon=resolved_taxon,
        adapter=adapter,
    )
    discovery_event = append(
        timeline,
        "discovery-result",
        "ReferenceAssembliesRecorded",
        payload(
            ReferenceAssembliesRecorded(
                attempt_id=discovery_attempt,
                tool_input=discovery_input,
                result=discovery_result,
                source_answers=(),
            )
        ),
    )
    stages["discovery"] = len(timeline)

    for index in range(5):
        hitl_attempt = attempt_start(timeline, f"clarify-attempt-{index}")
        request = PlanningHitlRequested(
            attempt_id=hitl_attempt,
            plan_event_id=plan_event["event_id"],
            plan_id=PLAN_ID,
            requirement_id=REFERENCE_REQUIREMENT_ID,
            subject_key="reference_assembly",
            issue="Reference selection requires researcher review.",
            question="Please clarify the intended reference.",
            evidence_event_ids=(discovery_event["event_id"],),
        )
        requested = append(
            timeline,
            f"clarify-request-{index}",
            request.event_type,
            payload(request),
        )
        response = make_planning_hitl_responded(
            requested["event_id"],
            ClarifyPlanningHitlResponse(
                action="clarify", instruction=f"Review iteration {index}."
            ),
        )
        append(
            timeline,
            f"clarify-response-{index}",
            response.event_type,
            payload(response),
        )

    selection_attempt = attempt_start(timeline, "selection-attempt")
    candidates = tuple(
        PlanningHitlCandidate(
            candidate_id=f"assembly:{item.assembly_accession}",
            asserted_value=item.assembly_accession,
            display_label=item.assembly_name,
            evidence_event_ids=(discovery_event["event_id"],),
        )
        for item in discovery_result.rows
    )
    selection_request = PlanningHitlRequested(
        attempt_id=selection_attempt,
        plan_event_id=plan_event["event_id"],
        plan_id=PLAN_ID,
        requirement_id=REFERENCE_REQUIREMENT_ID,
        subject_key="reference_assembly",
        issue="Two materially tied reference assemblies remain.",
        question="Which exact assembly should be validated?",
        evidence_event_ids=(discovery_event["event_id"],),
        candidates=candidates,
    )
    requested = append(
        timeline,
        "selection-request",
        selection_request.event_type,
        payload(selection_request),
    )
    stages["pending"] = len(timeline)
    response = make_planning_hitl_responded(
        requested["event_id"],
        SelectPlanningHitlResponse(
            action="select", candidate_id=candidates[1].candidate_id
        ),
    )
    response_event = append(
        timeline, "selection-response", response.event_type, payload(response)
    )
    stages["responded"] = len(timeline)

    describe_attempt = attempt_start(timeline, "describe-attempt")
    selected_candidate = discovery_result.rows[1]
    description_grounding = grounding(
        REFERENCE_REQUIREMENT_ID,
        "reference_assembly_identity",
        "reference_assembly",
        plan_event["event_id"],
        context["event_id"],
        taxon_event["event_id"],
        discovery_event["event_id"],
        response_event["event_id"],
    )
    selected = SelectedAssemblyEvidence(
        row=selected_candidate,
        discovery_evidence_id=discovery_result.evidence_id,
        discovery_event_id=discovery_event["event_id"],
        selection_event_id=response_event["event_id"],
    )
    description_input = DescribeReferenceInput(
        grounding=description_grounding,
        selected=selected,
        assembly_accession=selected_candidate.assembly_accession,
    )
    description_result = DescribeReferenceTool().describe(
        description_input,
        expected_grounding=description_grounding,
        expected_selection=selected,
        adapter=adapter,
    )
    append(
        timeline,
        "description-result",
        "ReferenceDescriptionRecorded",
        payload(
            ReferenceDescriptionRecorded(
                attempt_id=describe_attempt,
                tool_input=description_input,
                result=description_result,
                source_answers=(),
            )
        ),
    )
    stages["final"] = len(timeline)
    self_check = taxon_candidate.ncbi_tax_id
    assert self_check == 9606 and adapter.contact_count == 3
    return timeline, stages


class RecordedCompanionAdapter(AssemblyReportFake):
    def __init__(self, records=(), *, error=None):
        self.records = records
        self.error = error
        self.contact_count = 0

    def describe_companions(self, assembly_accession: str, role: str):
        del assembly_accession, role
        self.contact_count += 1
        if self.error is not None:
            raise self.error
        return CompanionAdapterObservation(
            records=self.records,
            source=companion_source(),
        )


COMPANION_REQUIREMENT_ID = "requirement_" + "e" * 32


def companion_timeline(*, respond: bool = True, role_set: str = "genomic_fasta"):
    timeline, _ = build_timeline()
    plan_event = next(
        event
        for event in timeline
        if event["event_type"] == "ProposalRequirementsIdentified"
    )
    context_event_id = timeline[0]["event_id"]
    requirement = ProposalInformationRequirement(
        requirement_id=COMPANION_REQUIREMENT_ID,
        kind="companion_asset",
        subject_key="reference_companions",
        domain="scientific",
        meaning="Determine required assembly-bound companion assets.",
        knowledge_state="to_find",
        value=None,
        evidence=(ContextEvidence(source_event_id=context_event_id),),
        acquisition_route="ask_planning_hitl",
        validation=ToolEvidenceValidation(
            requirement="Verify each selected assembly/role with exact Tool evidence."
        ),
    )
    plan_event["payload"]["plan"]["requirements"].append(
        requirement.model_dump(mode="json")
    )
    reference = project_reference_identity(timeline)
    role_attempt = attempt_start(timeline, "companion-role-attempt")
    candidate = PlanningHitlCandidate(
        candidate_id=f"companion-role-set:{role_set}",
        asserted_value=role_set,
        display_label=f"Assembly roles: {role_set}",
        evidence_event_ids=(reference.description_event_id,),
    )
    request = PlanningHitlRequested(
        attempt_id=role_attempt,
        plan_event_id=plan_event["event_id"],
        plan_id=PLAN_ID,
        requirement_id=COMPANION_REQUIREMENT_ID,
        subject_key="reference_companions",
        issue="The plan does not assert which assembly-owned companions are required.",
        question="Which source-supported role is required for this proposal?",
        evidence_event_ids=(reference.description_event_id,),
        candidates=(candidate,),
    )
    requested = append(
        timeline, "companion-role-request", request.event_type, payload(request)
    )
    if not respond:
        return timeline, None, None
    responded = make_planning_hitl_responded(
        requested["event_id"],
        SelectPlanningHitlResponse(
            action="select", candidate_id=candidate.candidate_id
        ),
    )
    response_event = append(
        timeline, "companion-role-response", responded.event_type, payload(responded)
    )
    grounding = CompanionRequirementGrounding(
        journey_id=JOURNEY_ID,
        plan_id=PLAN_ID,
        requirement_id=COMPANION_REQUIREMENT_ID,
        subject_key="reference_companions",
        source_event_ids=(
            plan_event["event_id"],
            context_event_id,
            reference.description_event_id,
            response_event["event_id"],
        ),
    )
    binding = VerifiedReferenceBinding(
        reference_event_id=reference.description_event_id,
        assembly_accession=reference.verified_reference.assembly_accession,
        reference_evidence_id=reference.verified_reference.evidence_id,
    )
    work = derive_companion_work_items(
        grounding=grounding,
        reference=binding,
        selected_roles=tuple(sorted(role_set.split("+"))),
        role_selection_event_ids=(response_event["event_id"],),
    )
    return timeline, work[0] if len(work) == 1 else work, response_event


def append_observation(
    timeline,
    work,
    *,
    label: str,
    records,
    assembly_accession: str | None = None,
    selected_asset_id: str | None = None,
    selection_event_id: str | None = None,
    index_choice: bool = True,
):
    attempt_id = attempt_start(timeline, f"{label}-attempt")
    tool_input = DescribeCompanionInput(
        work_item=work,
        assembly_accession=assembly_accession or work.reference.assembly_accession,
        selected_asset_id=selected_asset_id,
        selection_event_id=selection_event_id,
    )
    result = DescribeCompanionTool().describe(
        tool_input,
        expected_work_item=work,
        adapter=RecordedCompanionAdapter(records),
    )
    observed = append(
        timeline,
        label,
        "CompanionObservationRecorded",
        payload(
            CompanionObservationRecorded(
                attempt_id=attempt_id,
                tool_input=tool_input,
                result=result,
                source_answers=(),
            )
        ),
    )
    if index_choice and getattr(result, "outcome", None) == "verified":
        append_index_choice(timeline, observed, label=f"{label}-index")
    return observed


def append_index_rechoice(timeline, *, label: str):
    """The model's index choice again after an Execution Review revise reopened it
    (owner #464 I2): none, for the latest verified genomic FASTA's observation."""
    observation = next(
        event
        for event in reversed(timeline)
        if event["event_type"] == "CompanionObservationRecorded"
        and event["payload"]["result"].get("outcome") == "verified"
        and event["payload"]["result"]["work_item"]["role"] == "genomic_fasta"
    )
    return append_index_choice(timeline, observation, label=label)


def append_index_choice(timeline, observation, *, label: str):
    """The model's choice of no Galaxy index (the CPU build) for a verified genomic
    FASTA's observation, which completes its companion work (#464)."""
    from bio_neb.reference_discovery.prebuilt_index_tool import PrebuiltIndexGrounding
    from bio_neb.reference_discovery.reference_facts_journey import PrebuiltIndexChosen

    result = observation["payload"]["result"]
    (candidate,) = result["candidates"]
    work = result["work_item"]
    if work["role"] != "genomic_fasta" or candidate.get("source_locator") is None:
        return None
    return append(
        timeline,
        label,
        "PrebuiltIndexChosen",
        payload(
            PrebuiltIndexChosen(
                attempt_id=observation["payload"]["attempt_id"],
                grounding=PrebuiltIndexGrounding(
                    reference_event_id=work["reference"]["reference_event_id"],
                    assembly_accession=work["reference"]["assembly_accession"],
                ),
                read_event_id=None,
                candidate=None,
                reason="No listed Galaxy index holds every requested sequence.",
            )
        ),
    )


def compatible_record(
    work, asset_id: str = "opaque-current", *, authority: bool = True
):
    return companion_record(
        asset_id,
        assembly_accession=work.reference.assembly_accession,
        authority=authority,
        observed_source=True,
    )


def sample_requirement(timeline):
    plan = next(
        event
        for event in timeline
        if event["event_type"] == "ProposalRequirementsIdentified"
    )
    return plan, next(
        item
        for item in plan["payload"]["plan"]["requirements"]
        if item["kind"] == "sample_identity"
    )


def unresolved_sample_review():
    """A plan whose sample is unknown, with its open sample review."""

    timeline, stages = build_timeline()
    timeline = copy.deepcopy(timeline[: stages["plan"]])
    plan, sample = sample_requirement(timeline)
    sample.update(
        knowledge_state="to_find", value=None, acquisition_route="ask_planning_hitl"
    )
    del sample["identifier_rule"]
    sample["evidence"] = [
        {"source_kind": "planning_context", "source_event_id": timeline[0]["event_id"]}
    ]
    attempt_id = identity("unresolved-sample-attempt")
    append(
        timeline,
        "unresolved-sample-attempt-start",
        "AttemptStarted",
        {
            "attempt_id": attempt_id,
            "agent_role": "planning_agent",
            "input_set_id": INPUT_SET_ID,
        },
    )
    request = PlanningHitlRequested(
        attempt_id=attempt_id,
        plan_event_id=plan["event_id"],
        plan_id=PLAN_ID,
        requirement_id=SAMPLE_REQUIREMENT_ID,
        subject_key="sample_identity",
        issue="A consequence-bearing sample label is unresolved.",
        question="What sample label should be bound to the registered FASTQs?",
        evidence_event_ids=(plan["event_id"],),
    )
    event = append(
        timeline, "unresolved-sample-review", request.event_type, payload(request)
    )
    return timeline, event["event_id"]
