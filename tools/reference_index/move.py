"""Move SNVs stated on assembly A to assembly B whose BWA index is in the local store.

Flank re-alignment (`selection-pattern.md` at `77cacf1f`, step 7): about 200 bases each side
of the SNV are read from A's FASTA and aligned with `bwa mem` to B's index; the primary
alignment's position, strand and MAPQ, and B's base there, give the result.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bio_neb.content import canonical_json
from bio_neb.failure import ProviderError
from tools.variant_sources import VariantSources
from tools.variant_sources.contracts import ClaimAnswer, Route, VariantClaim

from .cache import ReferenceIndexCache
from .choice import choice_path
from .contracts import Choice, FastaRead, MovedSnv, MoveRecord, Place, SnvToMove
from .local_tools import LocalTools

FLANK_BASES = 200
MIN_MAPQ = 30
MIN_SCORE_FRACTION = 0.9
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_STRAND = {"+": "forward", "-": "reverse"}
_CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")


class MoveRefused(Exception):
    """The move cannot run as asked, and no record is kept: an SNV does not hold on A as
    stated (refused before any lookup or alignment), the cache cannot serve it, or a
    lookup of its rs ID was refused by the variant source's run plan or brought no
    answer outside that source's failure module."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Moved:
    """The moves, with the digest of the query FASTA aligned (`None` when every SNV had a
    database placement) and `bwa mem`'s SAM."""

    moves: tuple[MovedSnv, ...]
    query_sha256: str | None
    alignment: bytes


@dataclass(frozen=True)
class _Flanks:
    """The SNV's base with the flanks read from A; `offset` is the base's index."""

    sequence: str
    offset: int


def move_snvs(
    snvs: Sequence[SnvToMove],
    *,
    source_fasta: Path,
    target_assembly: str,
    target_fasta: Path,
    tools: LocalTools,
    queries: Path,
    database: Callable[[VariantClaim], ClaimAnswer] | None = None,
) -> Moved:
    """Each SNV's result on B. An SNV with an rs ID asks `database` (#331's
    `VariantSources.answer`) for its placement on B first; the others, and those it does
    not place on B, are re-aligned, all in one `bwa mem` run. The query FASTA is kept as
    `queries/<sha256>.fa` and aligned from there, so the SAM's header names it and the
    same query gives the same SAM."""
    lengths = _lengths(source_fasta, tools)
    flanks = [_flanks(snv, source_fasta, lengths, tools) for snv in snvs]
    placed: dict[int, MovedSnv] = {}
    answers: dict[int, tuple[ClaimAnswer, str]] = {}
    for index, snv in enumerate(snvs):
        if snv.rsid is None or database is None:
            continue
        answer = database(
            VariantClaim(
                label=snv.label, assembly_accession=target_assembly, rsid=snv.rsid
            )
        )
        moved = _database_placed(snv, answer, target_assembly, target_fasta, tools)
        if isinstance(moved, str):
            answers[index] = (answer, moved)
        else:
            placed[index] = moved
    unplaced = [index for index in range(len(snvs)) if index not in placed]
    query = "".join(
        f">{index}\n{flanks[index].sequence}\n" for index in unplaced
    ).encode()
    alignment, query_sha256 = b"", None
    if unplaced:
        query_sha256 = hashlib.sha256(query).hexdigest()
        path = queries / f"{query_sha256}.fa"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(query)
        alignment = tools.mem(target_fasta, path)
    primaries = _primary_alignments(alignment)
    moves = tuple(
        placed.get(index)
        or _realigned(
            snv,
            flanks[index],
            primaries[str(index)],
            target_assembly,
            target_fasta,
            tools,
            *answers.get(index, (None, None)),
        )
        for index, snv in enumerate(snvs)
    )
    return Moved(moves, query_sha256, alignment)


def _database_placed(
    snv: SnvToMove,
    answer: ClaimAnswer,
    target_assembly: str,
    target_fasta: Path,
    tools: LocalTools,
) -> MovedSnv | str:
    """The SNV's result from the database's placements of its rs ID on B, or why they do
    not place it there. A placement counts only when its alleles, on B's forward strand,
    are the SNV's (B in A's orientation) or their complement (B reversed), and then only
    when B's FASTA has the REF they imply: an rs ID alone does not make the SNV."""
    placements = _database_alleles(answer)
    if placements is None:
        return _outcome(answer)
    if {snv.ref, snv.alt} in ({"A", "T"}, {"C", "G"}):
        return (
            f"{_outcome(answer)}, but {snv.ref}>{snv.alt} is its own complement: the "
            "database's alleles cannot show its strand on B"
        )
    forward = (snv.ref, snv.alt)
    reverse = (_COMPLEMENT[snv.ref], _COMPLEMENT[snv.alt])
    loci = sorted(
        (contig, position, "+" if forward in alleles else "-")
        for (contig, position), alleles in placements.items()
        if forward in alleles or reverse in alleles
    )
    if not loci:
        stated = sorted({pair for alleles in placements.values() for pair in alleles})
        return (
            f"{_outcome(answer)}, but its alleles on B "
            f"({', '.join(f'{ref}>{alt}' for ref, alt in stated) or 'none'}) do not "
            f"support {snv.ref}>{snv.alt}"
        )
    places = tuple(
        Place(
            contig=contig,
            position=position,
            strand=strand,
            ref=tools.bases(target_fasta, contig, position, position),
        )
        for contig, position, strand in loci
    )
    moved = {
        "snv": snv,
        "target_assembly": target_assembly,
        "method": "database_placement",
        "places": places,
        "database": answer,
    }
    source = answer.route
    if len(places) > 1:
        return MovedSnv(
            **moved,
            result="several_places",
            place=None,
            reason=f"{source} places {snv.rsid} as {snv.ref}>{snv.alt} at "
            f"{len(places)} loci on B",
        )
    [place] = places
    ref, alt = forward if place.strand == "+" else reverse
    if place.ref != ref:
        return (
            f"{_outcome(answer)}: {snv.ref}>{snv.alt} at {place.contig}:{place.position}, "
            f"but B's FASTA has {place.ref} there, not {ref}"
        )
    if place.strand == "+":
        return MovedSnv(
            **moved,
            result="unique",
            place=place,
            target_ref=ref,
            target_alt=alt,
            reason=f"{source} places {snv.rsid} once on B as {ref}>{alt}",
        )
    return MovedSnv(
        **moved,
        result="strand_flipped",
        place=place,
        target_ref=ref,
        target_alt=alt,
        reason=f"{source} places {snv.rsid} once on B as {ref}>{alt}, the SNV "
        "complemented: B is reversed there",
    )


def _database_alleles(
    answer: ClaimAnswer,
) -> dict[tuple[str, int], set[tuple[str, str]]] | None:
    """Each locus the database placed the rs ID at on B, with its SNV alleles there (REF,
    ALT on B's forward strand); `None` when it placed it nowhere on B."""
    loci: dict[tuple[str, int], set[tuple[str, str]]] = {}
    # Several loci on B are the model's to name (`placement_unnamed`); this tool has
    # no model, and its own rule reads them all.
    placed = {"resolved", "placement_unnamed"}
    if answer.eva is not None and answer.eva.status in placed:
        # The product's EVA result: candidates on B's own contig names; only an SNV
        # allele counts here.
        for candidate in answer.eva.candidates:
            alleles = loci.setdefault(
                (candidate.contig, candidate.one_based_position), set()
            )
            if len(candidate.ref) == len(candidate.alt) == 1:
                alleles.add((candidate.ref, candidate.alt))
        return loci
    if answer.dbsnp is not None and answer.dbsnp.status in placed:
        for candidate in answer.dbsnp.candidates:
            loci.setdefault(
                (candidate.contig, candidate.one_based_position), set()
            ).add((candidate.ref, candidate.alt))
        return loci
    return None


def _lengths(fasta: Path, tools: LocalTools) -> dict[str, int]:
    index = Path(f"{fasta}.fai")
    if not index.exists():
        tools.faidx(fasta)
    return {
        fields[0]: int(fields[1])
        for fields in (line.split("\t") for line in index.read_text().splitlines())
    }


def _flanks(
    snv: SnvToMove, fasta: Path, lengths: dict[str, int], tools: LocalTools
) -> _Flanks:
    if snv.contig not in lengths:
        raise MoveRefused(
            "contig_not_on_source", f"{snv.label}: {fasta} has no sequence {snv.contig}"
        )
    if snv.position > lengths[snv.contig]:
        raise MoveRefused(
            "position_not_on_source",
            f"{snv.label}: {snv.contig} has {lengths[snv.contig]} bases",
        )
    start = max(1, snv.position - FLANK_BASES)
    end = min(lengths[snv.contig], snv.position + FLANK_BASES)
    flanks = _Flanks(tools.bases(fasta, snv.contig, start, end), snv.position - start)
    base = flanks.sequence[flanks.offset]
    if base != snv.ref:
        raise MoveRefused(
            "ref_not_on_source",
            f"{snv.label}: {snv.contig}:{snv.position} is {base} on {fasta}, "
            f"not the stated REF {snv.ref}",
        )
    return flanks


def _primary_alignments(sam: bytes) -> dict[str, list[str]]:
    return {
        fields[0]: fields
        for fields in (line.split("\t") for line in sam.decode().splitlines())
        if not fields[0].startswith("@") and not int(fields[1]) & 0x900
    }


def _realigned(
    snv: SnvToMove,
    flank: _Flanks,
    fields: list[str],
    target_assembly: str,
    target_fasta: Path,
    tools: LocalTools,
    database: ClaimAnswer | None,
    unplaced: str | None,
) -> MovedSnv:
    """The SNV's result from its flanks' primary alignment (a SAM record), after the
    database's answer, when there was one, did not place it on B (`unplaced` says why)."""
    found = _alignment_result(snv, flank, fields, target_fasta, tools)
    if database is not None:
        found["reason"] = f"{unplaced}; {found['reason']}"
    return MovedSnv(
        snv=snv,
        target_assembly=target_assembly,
        method="flank_realignment",
        database=database,
        **found,
    )


def _alignment_result(
    snv: SnvToMove,
    flank: _Flanks,
    fields: list[str],
    target_fasta: Path,
    tools: LocalTools,
) -> dict:
    if int(fields[1]) & 0x4:
        return {
            "result": "not_found",
            "place": None,
            "reason": "bwa mem did not align the flanks to B",
        }
    tags = dict((tag[:2], tag[5:]) for tag in fields[11:])
    primary = _place(
        flank,
        fields[2],
        int(fields[3]),
        "-" if int(fields[1]) & 0x10 else "+",
        fields[5],
        target_fasta,
        tools,
        mapq=int(fields[4]),
        score=int(tags["AS"]),
    )
    if primary is None:
        return {
            "result": "not_found",
            "place": None,
            "reason": f"bwa mem aligned the flanks to {fields[2]}:{fields[3]} "
            f"({fields[5]}), but the SNV's base is clipped or inserted there: B lacks it",
        }
    alternatives = [
        _place(flank, contig, abs(int(start)), start[0], cigar, target_fasta, tools)
        for contig, start, cigar, _ in (
            hit.split(",") for hit in tags.get("XA", "").split(";") if hit
        )
    ]
    places = (primary, *filter(None, alternatives))
    if primary.mapq < MIN_MAPQ:
        return {
            "result": "several_places",
            "place": None,
            "places": places,
            "reason": f"bwa mem gave MAPQ {primary.mapq} (under {MIN_MAPQ}): the flanks "
            f"align at {len(places)} places on B",
        }
    if primary.score < MIN_SCORE_FRACTION * len(flank.sequence):
        return {
            "result": "not_found",
            "place": None,
            "places": places,
            "reason": f"bwa mem's best alignment scores {primary.score}, under "
            f"{MIN_SCORE_FRACTION} of the {len(flank.sequence)} flank bases",
        }
    strand = primary.strand
    expected = snv.ref if strand == "+" else _COMPLEMENT[snv.ref]
    if primary.ref != expected:
        return {
            "result": "ref_changed",
            "place": primary,
            "places": places,
            "reason": f"bwa mem placed the flanks once, on B's {_STRAND[strand]} strand, "
            f"where B has {primary.ref}, not {expected}",
        }
    if strand == "+":
        return {
            "result": "unique",
            "place": primary,
            "places": places,
            "target_ref": snv.ref,
            "target_alt": snv.alt,
            "reason": "bwa mem placed the flanks once, on B's forward strand",
        }
    ref, alt = _COMPLEMENT[snv.ref], _COMPLEMENT[snv.alt]
    return {
        "result": "strand_flipped",
        "place": primary,
        "places": places,
        "target_ref": ref,
        "target_alt": alt,
        "reason": "bwa mem placed the flanks once, on B's reverse strand: the SNV is "
        f"{ref}>{alt} on B",
    }


def _outcome(answer: ClaimAnswer) -> str:
    """What the variant source answered for the rs ID on B, in a few words."""
    if answer.failure is not None:
        return (
            f"{answer.route}: {answer.failure.failure_class} ({answer.failure.route})"
        )
    if answer.not_checked is not None:
        return f"{answer.route}: not_checked ({answer.not_checked})"
    found = answer.eva or answer.dbsnp
    return f"{answer.route}: {found.status}"


def _place(
    flank: _Flanks,
    contig: str,
    start: int,
    strand: str,
    cigar: str,
    target_fasta: Path,
    tools: LocalTools,
    *,
    mapq: int | None = None,
    score: int | None = None,
) -> Place | None:
    """Where one alignment of the flanks puts the SNV's base, and B's base there; `None`
    when the alignment clips the base or has it as an insertion."""
    offset = flank.offset if strand == "+" else len(flank.sequence) - 1 - flank.offset
    position = _reference_position(start, cigar, offset)
    if position is None:
        return None
    return Place(
        contig=contig,
        position=position,
        strand=strand,
        ref=tools.bases(target_fasta, contig, position, position),
        mapq=mapq,
        cigar=cigar,
        score=score,
    )


def _reference_position(start: int, cigar: str, offset: int) -> int | None:
    """The 1-based reference position aligned to the query base at `offset` (0-based, in
    the alignment's orientation), or `None` when that base is clipped or inserted."""
    query, reference = 0, start
    for count, operation in _CIGAR.findall(cigar):
        count = int(count)
        if operation in "M=X":
            if query <= offset < query + count:
                return reference + offset - query
            query += count
            reference += count
        elif operation in "IS":
            if query <= offset < query + count:
                return None
            query += count
        elif operation in "DN":
            reference += count
    return None


def move(
    cache: ReferenceIndexCache,
    tools: LocalTools,
    target_assembly: str,
    *,
    choice: str | None = None,
    source_assembly: str | None = None,
    source_fasta: Path | None = None,
    snvs: Sequence[SnvToMove] = (),
    rsids: Mapping[str, str],
    lookups: VariantSources,
    source: Route | None = None,
) -> MoveRecord:
    """Move SNVs onto `target_assembly`'s stored index and keep the record: a choice's
    SNVs to move (`choice`, the requested assembly, read from its NCBI FASTA in the cache),
    or SNVs stated on a given FASTA. `rsids` gives SNVs, by label, the rs ID their database
    placement is asked for, through #331's variant source (`lookups`), from the ``source``
    the caller names (#469: no code routes by assembly). The query, the SAM and the record
    are written to `moves/`."""
    if choice is not None:
        chosen = Choice.model_validate_json(choice_path(cache, choice).read_bytes())
        if chosen.option.assembly_accession != target_assembly:
            raise MoveRefused(
                "target_not_chosen",
                f"{choice}'s choice is {chosen.option.assembly_accession}",
            )
        digest = chosen.bundle_digest
        source_assembly, source_fasta = choice, _ncbi_fasta(cache, choice)
        snvs = tuple(SnvToMove.of_snv(snv) for snv in chosen.snvs_to_move)
    else:
        digest = _stored_digest(cache, target_assembly)
    snvs = _with_rsids(snvs, rsids)
    if rsids and source is None:
        raise MoveRefused(
            "source_missing", "an rs ID's placement is asked of the source you name"
        )
    made = lookups.contact_count
    target_fasta = cache.root / "store" / "sha256" / digest / "reference.fa"
    try:
        moved = move_snvs(
            snvs,
            source_fasta=source_fasta,
            target_assembly=target_assembly,
            target_fasta=target_fasta,
            tools=tools,
            queries=cache.root / "moves" / "queries",
            database=None
            if source is None
            else (lambda claim: lookups.answer(claim, source)),
        )
    except ProviderError as error:
        # A request the product's transport refuses before contact; a contact the
        # failure module gave up on is the claim's answer instead.
        raise MoveRefused("lookup_refused", error.detail) from error
    record = MoveRecord(
        made_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_assembly=source_assembly,
        source_fasta=_read(source_fasta),
        target_assembly=target_assembly,
        target_bundle_digest=digest,
        target_fasta=_read(target_fasta),
        flank_bases=FLANK_BASES,
        min_mapq=MIN_MAPQ,
        min_score_fraction=MIN_SCORE_FRACTION,
        bwa=tools.identity("bwa"),
        samtools=tools.identity("samtools"),
        query_sha256=moved.query_sha256,
        alignment_sha256=_kept_alignment(cache, moved.alignment),
        moves=moved.moves,
        contact_count=lookups.contact_count - made,
    )
    content = canonical_json(record.model_dump(mode="json"))
    path = (
        cache.root / "moves" / target_assembly / f"{record.made_at.replace(':', '')}-"
        f"{hashlib.sha256(content.encode()).hexdigest()[:12]}.json"
    )
    if path.exists() and path.read_text() == content:
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        target.write(content)
    return record


def _with_rsids(
    snvs: Sequence[SnvToMove], rsids: Mapping[str, str]
) -> tuple[SnvToMove, ...]:
    unknown = set(rsids) - {snv.label for snv in snvs}
    if unknown:
        raise MoveRefused("rsid_label_unknown", ", ".join(sorted(unknown)))
    return tuple(
        SnvToMove(**snv.model_dump() | {"rsid": rsids[snv.label]})
        if snv.label in rsids
        else snv
        for snv in snvs
    )


def _stored_digest(cache: ReferenceIndexCache, accession: str) -> str:
    """The stored bundle the choice records name for the assembly."""
    digests = {
        chosen.bundle_digest
        for chosen in (
            Choice.model_validate_json(path.read_bytes())
            for path in sorted((cache.root / "choices").glob("*.json"))
        )
        if chosen.option.assembly_accession == accession
    }
    if not digests:
        # Name the commands that store it.
        raise MoveRefused(
            "no_stored_index",
            f"0 stored bundles for {accession} in the choice records; store one with "
            "python -m tools.reference_index catalogues, then python -m "
            f"tools.reference_index ask {accession}",
        )
    if len(digests) > 1:
        raise MoveRefused(
            "several_stored_indexes",
            f"{len(digests)} stored bundles for {accession} in the choice records",
        )
    return digests.pop()


def _ncbi_fasta(cache: ReferenceIndexCache, accession: str) -> Path:
    """The assembly's NCBI FASTA as `fetch-ncbi` unpacks it."""
    found = sorted((cache.root / "ncbi" / accession).glob(f"{accession}_*_genomic.fna"))
    if len(found) != 1:
        raise MoveRefused(
            "source_fasta_missing",
            f"{len(found)} NCBI FASTAs for {accession} in {cache.root / 'ncbi'}",
        )
    return found[0]


def _read(fasta: Path) -> FastaRead:
    return FastaRead(
        path=str(fasta),
        byte_count=fasta.stat().st_size,
        fai_sha256=hashlib.sha256(Path(f"{fasta}.fai").read_bytes()).hexdigest(),
    )


def _kept_alignment(cache: ReferenceIndexCache, sam: bytes) -> str | None:
    """The SAM kept once under its sha256; `None` when nothing was aligned."""
    if not sam:
        return None
    digest = hashlib.sha256(sam).hexdigest()
    path = cache.root / "moves" / "alignments" / f"{digest}.sam"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sam)
    return digest
