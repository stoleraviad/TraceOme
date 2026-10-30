"""One assembly request answered in one step: a pick, a choice card or a cache hit.

`ask` first looks in the local store: a request already chosen answers from the cache with
no contact. Otherwise it describes the requested assembly and lists its species through the
product's NCBI Datasets adapter, lists the Galaxy candidates, and picks without asking when
an exact-accession prebuilt matches every contig by M5 and passes every index check (the
owner's rule: only the file source changes, never the science).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from bio_neb.content import canonical_json, canonical_json_digest
from bio_neb.prebuilt_index.catalogue import CatalogueRow
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyAdapterObservation,
    AssemblySourceRecord,
    SourceObservation,
)
from tools.reference_index.fit import FitReport

from .cache import ReferenceIndexCache
from .contracts import (
    AskAnswer,
    Candidate,
    Choice,
    ChoiceCard,
    ChoiceOption,
    Cost,
    OptionFit,
    SnvClaim,
    SpeciesListing,
)
from .datasets import Datasets
from .fetch import Fetcher
from .galaxy_rule import (
    assembly_series,
    candidate_id,
    datacache_url,
    galaxy_candidates,
)
from .operations import (
    BWA_CATALOGUES,
    FASTA_CATALOGUE,
    Refused,
    candidate_fasta,
    catalogue_rows,
    fai_rows,
    fasta_rows,
    fasta_sizes,
    fetch_bundle,
    fetch_ncbi,
    find_candidates,
    fit_candidate,
    index_rows,
    listed_candidate,
)
from .store import load_bundle, store_bundle


def _sent(fetcher: Fetcher, ncbi: Datasets) -> int:
    """The GETs this run's downloads and NCBI reads have made so far."""
    return fetcher.transport.contact_count + ncbi.contact_count


def ask(
    cache: ReferenceIndexCache,
    fetcher: Fetcher,
    ncbi: Datasets,
    accession: str,
) -> AskAnswer:
    sent = _sent(fetcher, ncbi)
    chosen = choice_path(cache, accession)
    if chosen.exists():
        choice = Choice.model_validate_json(chosen.read_bytes())
        return AskAnswer(
            requested_accession=accession,
            outcome="cache_hit",
            choice=choice,
            bundle=load_bundle(cache, choice.bundle_digest),
            contacts=_sent(fetcher, ncbi) - sent,
        )
    described = ncbi.describe(accession)
    requested = _requested(described, accession)
    species = ncbi.species(requested.ncbi_tax_id)
    listed = find_candidates(cache, accession)
    cache.save_candidates(listed)
    refused, unfetchable = [], list(_not_candidates(cache, accession))
    for candidate in listed.candidates:
        if candidate.relation != "exact":
            continue
        report = _fitted(cache, fetcher, accession, requested.assembly_name, candidate)
        if report.verdict == "refused":
            refused.append(_prebuilt_option(candidate, report, requested))
            continue
        bundle = store_bundle(cache, report, candidate_fasta(cache, candidate))
        choice = Choice(
            requested=requested,
            decision="auto_pick",
            option=_prebuilt_option(candidate, report, requested),
            card_sha256=None,
            bundle_digest=bundle.bundle_digest,
            fit_report_sha256=canonical_json_digest(report.model_dump(mode="json")),
        )
        _save_choice(cache, choice)
        return AskAnswer(
            requested_accession=accession,
            outcome="picked",
            choice=choice,
            bundle=bundle,
            contacts=_sent(fetcher, ncbi) - sent,
        )
    listing, newer, reference_sources = _reference(cache, ncbi, requested, species)
    newer_options, newer_unfetchable = (
        _newer_options(cache, fetcher, requested, newer) if newer else ((), ())
    )
    offered = (_build_option(requested, requested), *refused, *newer_options)
    unfetchable.extend(newer_unfetchable)
    older, not_offered, alias_sources = _older_options(
        cache,
        ncbi,
        requested,
        {option.candidate_id for option in offered}
        | {identity for identity, _ in unfetchable},
    )
    card = ChoiceCard(
        requested=requested,
        species=listing,
        options=(*offered, *older),
        default_option_id=f"build:{accession}",
        not_offered=(
            *_missing_catalogues(cache),
            *(reason for _, reason in unfetchable),
            *not_offered,
        ),
        sources=(
            described.source,
            species.source,
            *reference_sources,
            *alias_sources,
        ),
    )
    _save_card(cache, card)
    return AskAnswer(
        requested_accession=accession,
        outcome="card",
        card=card,
        contacts=_sent(fetcher, ncbi) - sent,
    )


def _missing_catalogues(cache: ReferenceIndexCache) -> tuple[str, ...]:
    """Galaxy's prebuilt indexes come only from the saved catalogues: a card made without
    them names their fetch, not only the build (#394 gap 9)."""
    missing = [
        name
        for name in (*BWA_CATALOGUES, FASTA_CATALOGUE[0])
        if not (cache.root / "catalogues" / name).exists()
    ]
    if not missing:
        return ()
    return (
        f"Galaxy's catalogues are not in the cache ({', '.join(missing)}), so no "
        "Galaxy prebuilt is offered: run python -m tools.reference_index catalogues, "
        "then ask again",
    )


def take(
    cache: ReferenceIndexCache,
    fetcher: Fetcher,
    ncbi: Datasets,
    accession: str,
    option_id: str,
    snvs: tuple[SnvClaim, ...] = (),
) -> AskAnswer:
    """Take a prebuilt option of the request's card: fetch it, fit it against its
    assembly's NCBI FASTA, store it and record the choice with the SNVs to move."""
    sent = _sent(fetcher, ncbi)
    answer = ask(cache, fetcher, ncbi, accession)
    if answer.card is None:
        raise Refused("already_chosen", f"{accession}: {answer.outcome}")
    card = answer.card
    options = [option for option in card.options if option.option_id == option_id]
    if not options:
        raise Refused("not_an_option", option_id)
    option = options[0]
    if option.candidate_id is None or option.cost.work != "fetch_prebuilt":
        raise Refused("not_a_fetchable_prebuilt", option_id)
    if option.fit.sequence == "refused":
        raise Refused("fit_refused", "; ".join(option.fit.reasons))
    target = option.assembly_accession
    cache.save_candidates(find_candidates(cache, target))
    candidate = listed_candidate(cache, target, option.candidate_id)
    report = _fitted(cache, fetcher, target, option.assembly_name, candidate)
    if report.verdict == "refused":
        raise Refused("fit_refused", "; ".join(report.reasons))
    bundle = store_bundle(cache, report, candidate_fasta(cache, candidate))
    choice = Choice(
        requested=card.requested,
        decision="taken",
        option=option,
        card_sha256=canonical_json_digest(card.model_dump(mode="json")),
        bundle_digest=bundle.bundle_digest,
        fit_report_sha256=canonical_json_digest(report.model_dump(mode="json")),
        snvs_to_move=snvs,
    )
    _save_choice(cache, choice)
    return AskAnswer(
        requested_accession=accession,
        outcome="taken",
        choice=choice,
        bundle=bundle,
        contacts=_sent(fetcher, ncbi) - sent,
    )


def choice_path(cache: ReferenceIndexCache, accession: str) -> Path:
    return cache.root / "choices" / f"{accession}.json"


def _save_choice(cache: ReferenceIndexCache, choice: Choice) -> None:
    path = choice_path(cache, choice.requested.assembly_accession)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        target.write(canonical_json(choice.model_dump(mode="json")))


def _save_card(cache: ReferenceIndexCache, card: ChoiceCard) -> None:
    """Each distinct card is kept once, under its digest."""
    content = canonical_json(card.model_dump(mode="json"))
    digest = canonical_json_digest(card.model_dump(mode="json"))
    path = (
        cache.root / "cards" / card.requested.assembly_accession / f"{digest[:12]}.json"
    )
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        target.write(content)


def _requested(
    described: AssemblyAdapterObservation, accession: str
) -> AssemblySourceRecord:
    records = [r for r in described.records if r.assembly_accession == accession]
    if len(records) != 1:
        raise Refused("not_described_by_ncbi", accession)
    return records[0]


MAX_REFERENCE_DESCRIPTIONS = 10
"""At most this many of the species' Galaxy rows are described to find its reference
genome when the listing is truncated without one (#330 run plan, choice 2)."""


def _reference(
    cache: ReferenceIndexCache,
    ncbi: Datasets,
    requested: AssemblySourceRecord,
    species: AssemblyAdapterObservation,
) -> tuple[SpeciesListing, AssemblySourceRecord | None, tuple[SourceObservation, ...]]:
    """The species' reference genome as NCBI marks it, and that assembly as the newer
    option when it is not the request and the request is no longer current or older."""
    # NCBI marks both twins (GenBank and RefSeq) as the reference genome; the request's
    # own prefix is preferred, as Galaxy keys its rows by one of them.
    prefix = requested.assembly_accession[:4]
    listed = sorted(
        (record for record in species.records if _is_reference(record)),
        key=lambda record: not record.assembly_accession.startswith(prefix),
    )
    found, basis = (listed[0], "listing") if listed else (None, "not_found")
    sources = []
    if found is None and species.next_page_token is not None:
        for accession in _species_rows(cache, requested)[:MAX_REFERENCE_DESCRIPTIONS]:
            described = ncbi.describe(accession)
            sources.append(described.source)
            hits = [
                record
                for record in described.records
                if record.assembly_accession == accession and _is_reference(record)
            ]
            if hits:
                found, basis = hits[0], "described_galaxy_row"
                break
    listing = SpeciesListing(
        ncbi_tax_id=requested.ncbi_tax_id,
        assemblies=species.records,
        total_count=species.total_count,
        truncated=species.next_page_token is not None,
        reference_genome=found.assembly_accession if found else None,
        reference_basis=basis,
    )
    newer = (
        found
        if found is not None
        and found.assembly_accession != requested.assembly_accession
        and (
            requested.assembly_status != "current"
            or (found.release_date or "") > (requested.release_date or "")
        )
        else None
    )
    return listing, newer, tuple(sources)


def _is_reference(record: AssemblySourceRecord) -> bool:
    return (record.refseq_category or "").casefold() == "reference genome"


def _species_rows(
    cache: ReferenceIndexCache, requested: AssemblySourceRecord
) -> list[str]:
    """The accessions of Galaxy rows that name the requested species, in catalogue
    order, the request's own series left out (tool-only until #450)."""
    accessions = []
    for row in catalogue_rows(cache):
        accession = next(
            (k for k in (row.value, row.dbkey) if assembly_series(k)), None
        )
        if (
            accession
            and accession not in accessions
            and assembly_series(accession)
            != assembly_series(requested.assembly_accession)
            and (requested.organism_name or "").casefold() in row.name.casefold()
        ):
            accessions.append(accession)
    return accessions


def _newer_options(
    cache: ReferenceIndexCache,
    fetcher: Fetcher,
    requested: AssemblySourceRecord,
    newer: AssemblySourceRecord,
) -> tuple[tuple[ChoiceOption, ...], tuple[tuple[str, str], ...]]:
    """Each fetchable Galaxy prebuilt of the newer assembly, then its CPU build; and
    each other Galaxy prebuilt of it with the reason it is not offered."""
    listed = find_candidates(cache, newer.assembly_accession)
    cache.save_candidates(listed)
    prebuilt = []
    for candidate in listed.candidates:
        if candidate.relation != "exact":
            continue
        fetch_bundle(
            cache,
            fetcher,
            newer.assembly_accession,
            candidate.candidate_id,
            fai_only=True,
        )
        prebuilt.append(
            ChoiceOption(
                option_id=f"prebuilt:{candidate.candidate_id}",
                action="use_newer_prebuilt",
                assembly_accession=newer.assembly_accession,
                assembly_name=newer.assembly_name,
                candidate_id=candidate.candidate_id,
                cost=_fetch_cost(Path(f"{candidate_fasta(cache, candidate)}.fai")),
                fit=_moved(
                    newer, "newer_assembly", "checked against its NCBI FASTA when taken"
                ),
            )
        )
    return (*prebuilt, _build_option(newer, requested)), _not_candidates(
        cache, newer.assembly_accession
    )


class UcscAlias(NamedTuple):
    """The NCBI assembly a UCSC build stands for, and the row's proof: where it is shown
    and what it shows (owner, #393 Q5, G4)."""

    accession: str
    name: str
    proof_source: str
    proof_window: str


UCSC_ALIASES = {
    "susScr3": UcscAlias(
        "GCF_000003025.5",
        "Sscrofa10.2",
        "docs/development/reviews/issue330-reference-choice/proof.md (pig card)",
        "NCBI's dataset report names GCF_000003025.5 Sscrofa10.2 (prebuilt:galaxy:susScr3)",
    ),
    "galGal4": UcscAlias(
        "GCF_000002315.3",
        "Gallus_gallus-4.0",
        "docs/development/reviews/issue330-reference-choice/proof.md (chicken card)",
        "NCBI's dataset report names GCF_000002315.3 Gallus_gallus-4.0 "
        "(prebuilt:galaxy:galGal4)",
    ),
}
"""Galaxy's older indexes keyed by UCSC build, with the NCBI accession and assembly name
each stands for (owner-approved run plan, #330 choice 1), one proven row per case. NCBI
stays the identity source: the card offers one only when NCBI gives that accession that
name. A species row keyed by a UCSC build with no row here says so. The rows stay in code
until the internalize links them as a table (owner, #393 G4)."""


def _older_options(
    cache: ReferenceIndexCache,
    ncbi: Datasets,
    requested: AssemblySourceRecord,
    offered: set[str | None],
) -> tuple[tuple[ChoiceOption, ...], tuple[str, ...], tuple[SourceObservation, ...]]:
    """Galaxy rows of an older version of the requested assembly (by accession, or by
    a UCSC alias), and the species' other Galaxy rows, listed as not offered. Tool-only
    until #450: these rows include ones the product's rule does not list."""
    options, not_offered, sources = [], [], []
    seen = set(offered)
    for row in catalogue_rows(cache):
        identity = candidate_id(row.catalogue, row.value)
        if identity in seen:
            continue
        alias = UCSC_ALIASES.get(row.dbkey)
        accession, name = (
            (alias.accession, alias.name)
            if alias
            else (
                next((k for k in (row.value, row.dbkey) if assembly_series(k)), None),
                None,
            )
        )
        if accession and _older(accession, requested.assembly_accession):
            seen.add(identity)
            described = ncbi.describe(accession)
            sources.append(described.source)
            records = [
                r for r in described.records if r.assembly_accession == accession
            ]
            if len(records) != 1 or name not in (None, records[0].assembly_name):
                not_offered.append(f"{identity}: NCBI does not name {accession} {name}")
                continue
            options.append(
                ChoiceOption(
                    option_id=f"prebuilt:{identity}",
                    action="use_older_prebuilt",
                    assembly_accession=accession,
                    assembly_name=records[0].assembly_name,
                    candidate_id=identity,
                    cost=_unmeasured_cost(cache, row),
                    fit=_moved(records[0], "older_assembly", "not recommended"),
                )
            )
        elif requested.organism_name.casefold() in row.name.casefold():
            seen.add(identity)
            unaliased = (
                f"; no UCSC_ALIASES row for {row.dbkey}" if accession is None else ""
            )
            not_offered.append(
                f"{identity}: another assembly of {requested.organism_name}{unaliased}"
            )
    return tuple(options), tuple(not_offered), tuple(sources)


def _older(accession: str, requested: str) -> bool:
    """An earlier version of the same accession (GenBank and RefSeq versions are
    independent, so only the same prefix compares)."""
    prefix, _, version = accession.partition(".")
    wanted, _, wanted_version = requested.partition(".")
    return prefix == wanted and int(version) < int(wanted_version)


def _unmeasured_cost(cache: ReferenceIndexCache, row: CatalogueRow) -> Cost:
    if reason := _why_not_a_candidate(cache, row):
        return Cost(
            work="not_fetchable",
            download_bytes=None,
            basis="unknown",
            detail=f"not fetchable here: {reason}",
        )
    return Cost(
        work="fetch_prebuilt",
        download_bytes=None,
        basis="unknown",
        detail="from Galaxy's datacache; its size is not measured",
    )


def _moved(assembly: AssemblySourceRecord, relation, detail: str) -> OptionFit:
    accession = assembly.assembly_accession
    return OptionFit(
        sequence=relation,
        detail=f"{relation.replace('_', ' ')} {accession} ({assembly.assembly_name}); {detail}",
        snv_chain=(
            f"move: the request's SNVs onto {accession} by a database placement, "
            "else flank re-alignment to its index (#332)",
            "look up: each SNV through the variant source tool, dbSNP for a human "
            "assembly and EVA otherwise (#331)",
        ),
    )


def _not_candidates(
    cache: ReferenceIndexCache, accession: str
) -> tuple[tuple[str, str], ...]:
    """Tool-only until #450: each Galaxy row of exactly `accession` the product's rule
    does not list, with why, so the card can say it is not offered."""
    reasons: dict[str, str] = {}
    for row in catalogue_rows(cache):
        identity = candidate_id(row.catalogue, row.value)
        if row.value != accession or identity in reasons:
            continue
        if reason := _why_not_a_candidate(cache, row):
            reasons[identity] = (
                f"{identity}: not fetchable here ({reason}), so its fit cannot be "
                "checked"
            )
    return tuple(reasons.items())


def _why_not_a_candidate(cache: ReferenceIndexCache, row: CatalogueRow) -> str | None:
    """Why the product's rule (`galaxy_candidates`) does not list a row, or None."""
    indexes, fastas = index_rows(cache), fasta_rows(cache)
    identity = candidate_id(row.catalogue, row.value)
    if any(
        found.candidate_id == identity
        for found in galaxy_candidates(row.value, indexes, fastas)
    ):
        return None
    if datacache_url(row.path) is None:
        return "Galaxy gives no datacache path for its index and FASTA"
    if row not in indexes:
        return "its catalogue is not a datacache index catalogue"
    count = sum(1 for f in fastas if f.value == row.value and datacache_url(f.path))
    return f"{FASTA_CATALOGUE[0]} names {count or 'no'} FASTA{'s' if count > 1 else ''} for it"


BWA_BYTES_PER_BASE = 1.75
"""bwa 0.7's index of a genome of L bases: `.bwt` about L, `.sa` about L/2 (interval
32), `.pac` about L/4; `.amb` and `.ann` are small."""


def _fetch_cost(fai: Path) -> Cost:
    """The FASTA's exact size from its `.fai`, plus the BWA files by
    `BWA_BYTES_PER_BASE`."""
    rows = fai_rows(fai)
    bases = sum(length for _, length, *_ in rows)
    fasta = fasta_sizes(fai)[0]
    return Cost(
        work="fetch_prebuilt",
        download_bytes=fasta + round(BWA_BYTES_PER_BASE * bases),
        basis="estimated_from_fai",
        detail=(
            f"{len(rows)} contigs, {bases} bases: the FASTA ({fasta} bytes) and the "
            "five BWA files from Galaxy's datacache; the CPU build is skipped"
        ),
    )


def _build_option(
    assembly: AssemblySourceRecord, requested: AssemblySourceRecord
) -> ChoiceOption:
    same = assembly.assembly_accession == requested.assembly_accession
    return ChoiceOption(
        option_id=f"build:{assembly.assembly_accession}",
        action="build_requested" if same else "build_newer",
        assembly_accession=assembly.assembly_accession,
        assembly_name=assembly.assembly_name,
        candidate_id=None,
        cost=Cost(
            work="cpu_build",
            download_bytes=None,
            basis="unknown",
            detail=(
                "the NCBI FASTA, then the CPU job's bwa index, samtools faidx and "
                "samtools dict: hours for a genome of about 1 Gb, once, then cached"
            ),
        ),
        fit=OptionFit(
            sequence="requested_assembly",
            detail="the requested assembly itself; its SNV positions apply as they are",
        )
        if same
        else _moved(assembly, "newer_assembly", "built by the CPU job"),
    )


def _fitted(
    cache: ReferenceIndexCache,
    fetcher: Fetcher,
    accession: str,
    assembly_name: str,
    candidate: Candidate,
) -> FitReport:
    """The candidate's fit against the assembly's NCBI FASTA: the retained report when
    the candidate carries one, else fetched and checked now."""
    if candidate.fit is not None:
        return cache.load_fit_by_digest(
            accession, candidate.candidate_id, candidate.fit.report_sha256
        )
    fetch_bundle(cache, fetcher, accession, candidate.candidate_id, fai_only=False)
    if not list((cache.root / "ncbi" / accession).glob("*_genomic.fna")):
        fetch_ncbi(cache, fetcher, accession, _ncbi_name(assembly_name))
    return fit_candidate(cache, accession, candidate.candidate_id)


def _ncbi_name(assembly_name: str) -> str:
    """NCBI's genomes folder spells the name with each character outside
    [A-Za-z0-9._-] as "_" (as the product adapter's `_genomes_directory` does)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", assembly_name)


def _prebuilt_option(
    candidate: Candidate, report: FitReport, assembly: AssemblySourceRecord
) -> ChoiceOption:
    exact = report.verdict != "refused"
    return ChoiceOption(
        option_id=f"prebuilt:{candidate.candidate_id}",
        action="use_prebuilt",
        assembly_accession=assembly.assembly_accession,
        assembly_name=assembly.assembly_name,
        candidate_id=candidate.candidate_id,
        cost=Cost(
            work="fetch_prebuilt",
            download_bytes=sum(file.byte_count for file in report.candidate_files),
            basis="fetched",
            detail="fetched and checked; the CPU build is skipped",
        ),
        fit=OptionFit(
            sequence="exact_sequence" if exact else "refused",
            detail=(
                f"every contig matched by M5 ({report.names} names) and every "
                "index check passed"
                if exact
                else "the fit check refused this prebuilt"
            ),
            reasons=report.reasons,
            fit_report_sha256=canonical_json_digest(report.model_dump(mode="json")),
        ),
    )
