"""A gene plus protein or coding change as genomic candidates (#376, G15).

A claim such as s3's ``PBP2 (Y127X)`` names no position; the model types its change
(#473 Q2 a), and nothing here reads change text. On each transcript of the gene, the
stated codon (or coding base) is walked to the Run's assembly through the transcript's CDS
segments, its reference bases are read and translated here, and every genomic change that
gives the stated residue becomes a candidate: one claim, a collection of candidates. Nothing
here contacts a source; the bases come through ``read_bases`` (``annotation.py`` reads them
with the product's ``ncbi_reference_span``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Literal

ReadBases = Callable[[str, int, int], str]
"""Plus-strand bases of ``contig`` from a 1-based ``start``, ``length`` long."""

_BASES = "TCAG"
GENETIC_CODES = {
    1: "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    2: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG",
    3: "FFLLSSSSYY**CCWWTTTTPPPPHHQQRRRRIIMMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    4: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    5: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSSSVVVVAAAADDEEGGGG",
    6: "FFLLSSSSYYQQCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    9: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    10: "FFLLSSSSYY**CCCWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    11: "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    12: "FFLLSSSSYY**CC*WLLLSPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    13: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSGGVVVVAAAADDEEGGGG",
    14: "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    15: "FFLLSSSSYY*QCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    16: "FFLLSSSSYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    21: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    22: "FFLLSS*SYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    23: "FF*LSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    24: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
    25: "FFLLSSSSYY**CCGWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    26: "FFLLSSSSYY**CC*WLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    27: "FFLLSSSSYYQQCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    28: "FFLLSSSSYYQQCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    29: "FFLLSSSSYYYYCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    30: "FFLLSSSSYYEECC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    31: "FFLLSSSSYYEECCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    32: "FFLLSSSSYY*WCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    33: "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
}
"""NCBI's translation tables by id, each its ``ncbieaa`` residues over the codons in
``_BASES`` order (NCBI ``gc.prt``, read 2026-10-27, sha256 2aecbdd0…; #469 spike). A
transcript's id is copied from its annotation (`Transcript.genetic_code`)."""


def translation(genetic_code: int) -> dict[str, str] | None:
    """The codon-to-residue map of NCBI's table ``genetic_code``, or None when NCBI's
    tables define no such id."""
    residues = GENETIC_CODES.get(genetic_code)
    if residues is None:
        return None
    return {
        "".join(codon): residue
        for codon, residue in zip(product(_BASES, repeat=3), residues, strict=True)
    }


_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


@dataclass(frozen=True)
class ProteinChange:
    """A residue change, one-letter; a stop is ``*``, as the model typed it (#473)."""

    ref: str
    number: int
    alt: str


@dataclass(frozen=True)
class CodingChange:
    """A coding-strand substitution at a CDS position (``c.N``)."""

    number: int
    ref: str
    alt: str


Change = ProteinChange | CodingChange | tuple[CodingChange, ProteinChange]


GeneticCodeOrigin = Literal["copied", "default"]
"""Where a transcript's genetic code came from (#469, owner Q3 a): ``copied`` from its
CDS's ``transl_table``, or ``default``, the INSDC rule that a CDS with no
``transl_table`` qualifier uses table 1."""


@dataclass(frozen=True)
class Transcript:
    """A protein-coding transcript on the Run's assembly: its CDS as genomic segments
    (1-based, inclusive, ``start <= end``) in the transcript's own 5' to 3' order, the
    genetic code its CDS is translated with and its annotation's own tags, copied
    (#469, ADR 0049)."""

    accession: str
    protein: str | None
    gene: str
    contig: str
    strand: Literal["+", "-"]
    cds: tuple[tuple[int, int], ...]
    genetic_code: int
    genetic_code_origin: GeneticCodeOrigin
    tags: tuple[str, ...] = ()
    # The first CDS row's phase in transcript order, copied: not 0 for a 5'-partial
    # CDS that starts mid-codon.
    cds_phase: int = 0


@dataclass(frozen=True)
class Candidate:
    """One genomic change that gives the stated change, in VCF form on the plus strand."""

    contig: str
    position: int
    ref: str
    alt: str
    bases_changed: int
    reference_codon: str
    alt_codon: str


@dataclass(frozen=True)
class TranscriptReading:
    """What one transcript says at the stated codon: its reference codon and residue,
    whether they are the stated ones, and the candidates when they are."""

    transcript: Transcript
    codon_number: int
    reference_codon: str | None
    reference_residue: str | None
    matches: bool
    reason: str | None
    candidates: tuple[Candidate, ...] = ()
    unplaced: tuple[str, ...] = ()
    """ALT codons whose changed bases an intron separates: no one VCF record holds them."""


def coding_position(transcript: Transcript, number: int) -> int | None:
    """The genomic position of CDS base ``number``, or None past the CDS."""
    offset = number - 1
    for start, end in transcript.cds:
        length = end - start + 1
        if offset < length:
            return start + offset if transcript.strand == "+" else end - offset
        offset -= length
    return None


def read_transcript(
    transcript: Transcript, change: Change, read_bases: ReadBases
) -> TranscriptReading:
    """The stated change on one transcript: the reference codon, read and translated,
    and every genomic change that gives the stated residue (or base)."""
    coding, protein = _parts(change)
    codon_number = (coding.number + 2) // 3 if coding else protein.number
    positions = tuple(
        coding_position(transcript, 3 * codon_number - offset) for offset in (2, 1, 0)
    )
    if None in positions:
        return _reading(
            transcript,
            codon_number,
            reason=f"{transcript.accession}'s CDS ends before codon {codon_number}",
        )
    if transcript.cds_phase != 0:
        return _reading(
            transcript,
            codon_number,
            reason=f"{transcript.accession}'s CDS starts mid-codon (phase "
            f"{transcript.cds_phase}): its codons are not counted from it",
        )
    code = translation(transcript.genetic_code)
    if code is None:
        return _reading(
            transcript,
            codon_number,
            reason=f"{transcript.accession}'s annotation names translation table "
            f"{transcript.genetic_code}, which NCBI's tables do not define",
        )
    plus = _plus_bases(transcript, positions, read_bases)
    codon = _coding_strand(transcript, positions, plus)
    residue = code.get(codon)
    if residue is None:
        return _reading(
            transcript,
            codon_number,
            codon,
            reason=f"{transcript.accession}'s codon {codon_number} reads {codon}",
        )
    if coding is not None:
        stated = codon[(coding.number - 1) % 3]
        if stated != coding.ref:
            return _reading(
                transcript,
                codon_number,
                codon,
                residue,
                reason=f"{transcript.accession} holds {stated} at c.{coding.number}; "
                f"the claim states {coding.ref}",
            )
    if protein is not None and residue != protein.ref:
        return _reading(
            transcript,
            codon_number,
            codon,
            residue,
            reason=f"{transcript.accession} codes {residue} at codon {codon_number}; "
            f"the claim states {protein.ref}",
        )
    if coding is not None:
        index = (coding.number - 1) % 3
        alt_codons = (codon[:index] + coding.alt + codon[index + 1 :],)
        if protein is not None and code[alt_codons[0]] != protein.alt:
            return _reading(
                transcript,
                codon_number,
                codon,
                residue,
                reason=f"on {transcript.accession}, c.{coding.number}{coding.ref}>"
                f"{coding.alt} gives {code[alt_codons[0]]}, not the stated "
                f"{protein.alt}",
            )
    else:
        alt_codons = tuple(
            sorted(
                (alt for alt, coded in code.items() if coded == protein.alt),
                key=lambda alt: (_changed(codon, alt), alt),
            )
        )
    candidates, unplaced = [], []
    for alt_codon in alt_codons:
        candidate = _candidate(transcript, positions, plus, codon, alt_codon)
        if candidate is None:
            unplaced.append(alt_codon)
        else:
            candidates.append(candidate)
    return TranscriptReading(
        transcript=transcript,
        codon_number=codon_number,
        reference_codon=codon,
        reference_residue=residue,
        matches=True,
        reason=None,
        candidates=tuple(candidates),
        unplaced=tuple(unplaced),
    )


def _parts(change: Change) -> tuple[CodingChange | None, ProteinChange | None]:
    if isinstance(change, tuple):
        return change
    if isinstance(change, CodingChange):
        return change, None
    return None, change


def _reading(transcript, codon_number, codon=None, residue=None, *, reason):
    return TranscriptReading(
        transcript=transcript,
        codon_number=codon_number,
        reference_codon=codon,
        reference_residue=residue,
        matches=False,
        reason=reason,
    )


def _runs(positions: tuple[int, ...]) -> list[tuple[int, int]]:
    """Contiguous genomic runs of the codon's positions, as (start, length)."""
    ordered = sorted(positions)
    runs = [[ordered[0], 1]]
    for position in ordered[1:]:
        if position == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([position, 1])
    return [(start, length) for start, length in runs]


def _plus_bases(transcript, positions, read_bases) -> dict[int, str]:
    bases = {}
    for start, length in _runs(positions):
        read = read_bases(transcript.contig, start, length).upper()
        bases.update({start + offset: base for offset, base in enumerate(read)})
    return bases


def _coding_strand(transcript, positions, plus: dict[int, str]) -> str:
    codon = "".join(plus[position] for position in positions)
    return codon if transcript.strand == "+" else codon.translate(_COMPLEMENT)


def _changed(codon: str, alt: str) -> int:
    return sum(1 for a, b in zip(codon, alt, strict=True) if a != b)


def _candidate(transcript, positions, plus, codon, alt_codon) -> Candidate | None:
    changed = [index for index in range(3) if codon[index] != alt_codon[index]]
    genomic = {positions[index]: alt_codon[index] for index in changed}
    low, high = min(genomic), max(genomic)
    if any(position not in positions for position in range(low, high + 1)):
        return None
    to_plus = (
        (lambda base: base)
        if transcript.strand == "+"
        else (lambda base: base.translate(_COMPLEMENT))
    )
    ref = "".join(plus[position] for position in range(low, high + 1))
    alt = "".join(
        to_plus(genomic[position]) if position in genomic else plus[position]
        for position in range(low, high + 1)
    )
    return Candidate(
        contig=transcript.contig,
        position=low,
        ref=ref,
        alt=alt,
        bases_changed=len(changed),
        reference_codon=codon,
        alt_codon=alt_codon,
    )


Outcome = Literal["candidates", "ambiguous", "not_found", "not_checked"]


@dataclass(frozen=True)
class Decision:
    """One claim's answer over its gene's transcripts. Each candidate names the
    transcripts that give it; ``ambiguous`` keeps every transcript's candidates."""

    outcome: Outcome
    reason: str | None
    readings: tuple[TranscriptReading, ...]
    candidates: tuple[tuple[Candidate, tuple[str, ...]], ...] = ()


def decide(
    change: Change,
    transcripts: tuple[Transcript, ...],
    read_bases: ReadBases,
    *,
    transcript: str | None = None,
) -> Decision:
    """Read the stated change on the transcripts the claim allows: the one the paper
    names, else every transcript of the gene, with its tags copied and no preference of
    code's (#469, owner Q5 a: the model's pick of a transcript is #473's). One codon on
    the genome gives ``candidates``, several give ``ambiguous``, none gives
    ``not_found``; a change no one VCF record can hold (its bases on both sides of an
    intron) is ``not_checked``."""
    chosen = _chosen(transcripts, transcript=transcript)
    if isinstance(chosen, str):
        return Decision("not_found", chosen, ())
    readings = tuple(read_transcript(each, change, read_bases) for each in chosen)
    matching = [reading for reading in readings if reading.matches]
    if not matching:
        reasons = "; ".join(reading.reason for reading in readings)
        return Decision("not_found", reasons, readings)
    if not any(reading.candidates for reading in matching):
        unplaced = "; ".join(
            f"{reading.transcript.accession}: {', '.join(reading.unplaced)}"
            for reading in matching
        )
        coding, protein = _parts(change)
        residue = protein.alt if protein else coding.alt
        reason = (
            f"every change giving {residue} at codon {matching[0].codon_number} changes "
            f"bases an intron separates ({unplaced}); no one VCF record holds it"
        )
        return Decision("not_checked", reason, readings)
    matching = [reading for reading in matching if reading.candidates]
    by_codon: dict[tuple, list[TranscriptReading]] = {}
    for reading in matching:
        by_codon.setdefault(_codon_place(reading), []).append(reading)
    candidates: dict[Candidate, list[str]] = {}
    for reading in matching:
        for candidate in reading.candidates:
            candidates.setdefault(candidate, []).append(reading.transcript.accession)
    listed = tuple((candidate, tuple(names)) for candidate, names in candidates.items())
    if len(by_codon) == 1:
        return Decision("candidates", None, readings, listed)
    first = matching[0]
    places = ", ".join(
        f"{group[0].transcript.accession} at {_span(group[0])}"
        for group in by_codon.values()
    )
    stated = _stated(change)
    reason = (
        f"{first.transcript.gene}'s transcripts place {stated} at different codons: "
        f"{places}"
    )
    return Decision("ambiguous", reason, readings, listed)


def _chosen(transcripts, *, transcript) -> tuple[Transcript, ...] | str:
    if transcript is not None:
        # A version the claim names is that version: another version of a transcript can
        # have other exons. A name without one takes the annotation's.
        base = transcript.split(".")[0]
        same = tuple(t for t in transcripts if t.accession.split(".")[0] == base)
        named = (
            tuple(t for t in same if t.accession == transcript)
            if "." in transcript
            else same
        )
        if named:
            return named
        gene = transcripts[0].gene if transcripts else "the gene"
        if same:
            held = ", ".join(t.accession for t in same)
            return (
                f"the claim names {transcript}; {gene}'s annotation on the Run's "
                f"assembly holds {held}, another version of it"
            )
        held = ", ".join(t.accession for t in transcripts) or "none"
        return (
            f"the claim names {transcript}; {gene} has no such transcript on the "
            f"Run's assembly ({held})"
        )
    return transcripts


def _codon_place(reading: TranscriptReading) -> tuple:
    transcript = reading.transcript
    return (
        transcript.contig,
        tuple(
            coding_position(transcript, 3 * reading.codon_number - offset)
            for offset in (2, 1, 0)
        ),
    )


def _span(reading: TranscriptReading) -> str:
    contig, positions = _codon_place(reading)
    return f"{contig}:{min(positions)}-{max(positions)}"


def _stated(change: Change) -> str:
    coding, protein = _parts(change)
    if protein is not None:
        return f"{protein.ref}{protein.number}"
    return f"c.{coding.number}"
