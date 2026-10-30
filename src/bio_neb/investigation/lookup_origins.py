"""Every lookup ending a variant lookup records, by its typed origin (#478, OQ1).

One place defines each ending on #474's pattern (``qualification_origins``): its
code, the typed fields it copies from what the source answered, its audit sentence
and its plain words. The audit sentence is the lookup fact's own ``conflicts`` and
``rationale``; the readable report, the item Review and the CLI read it. Both state
only what was observed, in the source's own words where it wrote any. Neither
tells the model or the researcher what to do: the next step is the Tool's model
view's (``investigation_agent.LOOKUP_VIEW``). Whether the answer fits the request is
the model's judgement; the Tool ran either way.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, StrictInt, StrictStr

from bio_neb.contracts import StrictFrozenContract

# The source's own words an audit sentence quotes, at most; the rest stays in the
# Tool's source answers and in the retained answer.
SOURCE_WORDS = 300


def source_words(body: bytes) -> str:
    """A source's answer body as an audit sentence quotes it: its text as sent,
    cut at ``SOURCE_WORDS`` characters with the cut named."""
    text = body.decode("utf-8", errors="replace").strip()
    if len(text) <= SOURCE_WORDS:
        return text
    return f"{text[:SOURCE_WORDS]}… ({len(text) - SOURCE_WORDS} more characters)"


class _Origin(StrictFrozenContract):
    def audit(self) -> str:
        """What the lookup observed, as its fact records it."""
        raise NotImplementedError

    def words(self) -> str:
        """The same observation in plain words, naming no code or field."""
        raise NotImplementedError


class SourceAnswered(_Origin):
    """A GET the source answered with a status other than 2xx (404 and 410 among
    them): its answer, in its own words (owner, #468 R1 b; #478 OQ1)."""

    code: Literal["source_answered"] = "source_answered"
    source: StrictStr
    status_code: StrictInt = Field(ge=100, le=599)
    key: StrictStr | None = None
    words_sent: StrictStr
    # The model's ``listed_none`` quote, when its words are not in this answer.
    unquoted: StrictStr | None = None

    def audit(self) -> str:
        about = f" for {self.key}" if self.key else ""
        said = f": {self.words_sent}" if self.words_sent else " with no body"
        sentence = f"{self.source} answered HTTP {self.status_code}{about}{said}"
        if self.unquoted is not None:
            sentence = f"The quote {self.unquoted!r} is not in its answer. {sentence}"
        return _sentence(sentence)

    def words(self) -> str:
        said = f": {self.words_sent}" if self.words_sent else " with nothing more"
        return f"{self.source} answered with status {self.status_code}{said}"


class RecordState(_Origin):
    """A dbSNP record that carries no placement snapshot (a withdrawn record, for
    example), by dbSNP's own key for the snapshot it carries."""

    code: Literal["record_state"] = "record_state"
    rsid: StrictStr
    state: StrictStr | None = None

    def audit(self) -> str:
        return _sentence(
            f"NCBI dbSNP's record of {self.rsid} carries "
            f"{self.state or 'no snapshot'}, not a placement snapshot"
        )

    def words(self) -> str:
        return (
            f"dbSNP keeps {self.rsid} as {self.state or 'no snapshot'}, with no "
            "placement on a genome"
        )


class Merged(_Origin):
    """A dbSNP record merged into another, by the current rsIDs it names."""

    code: Literal["merged"] = "merged"
    rsid: StrictStr
    into: tuple[StrictStr, ...] = Field(min_length=1)

    def audit(self) -> str:
        return _sentence(f"NCBI dbSNP merged {self.rsid} into {', '.join(self.into)}")

    def words(self) -> str:
        return f"dbSNP merged {self.rsid} into {', '.join(self.into)}"


class SequenceUnnamed(_Origin):
    """A written contig no sequence is named for yet on the assemblies the lookup
    reached, or a named record the assembly's report does not hold."""

    code: Literal["sequence_unnamed"] = "sequence_unnamed"
    contig: StrictStr
    assemblies: tuple[StrictStr, ...] = Field(min_length=1)
    missing_record: StrictStr | None = None
    # The model's quote naming the sequence, when membership refused it: not in the
    # researcher's words, or without the written contig.
    quote: StrictStr | None = None
    quote_fault: Literal["not_in_words", "without_contig"] | None = None

    def audit(self) -> str:
        listed = ", ".join(self.assemblies)
        if self.quote_fault == "not_in_words":
            return _sentence(
                f"The quote {self.quote!r} naming {listed}'s sequence is not in the "
                "researcher's words"
            )
        if self.quote_fault == "without_contig":
            return _sentence(
                f"The quote {self.quote!r} does not hold the written contig "
                f"{self.contig!r}"
            )
        if self.missing_record is not None:
            return _sentence(
                f"The assembly report of {listed} has no sequence {self.missing_record}"
            )
        return _sentence(
            f"No sequence is named for the written contig {self.contig!r} on {listed}"
        )

    def words(self) -> str:
        if self.missing_record is not None:
            return (
                f"the genome's report has no sequence {self.missing_record} for "
                f"{self.contig!r}"
            )
        return f"which sequence {self.contig!r} means on the genome is not named yet"


class PlacementUnnamed(_Origin):
    """A record with more than one locus on the Run's assembly and no placement
    named, or a named placement that is not one of its placements there."""

    code: Literal["placement_unnamed"] = "placement_unnamed"
    source: StrictStr
    rsid: StrictStr
    sequences: tuple[StrictStr, ...]
    loci: StrictInt | None = None
    named: StrictStr | None = None

    def audit(self) -> str:
        on = ", ".join(self.sequences) or "none"
        if self.named is not None and self.named not in self.sequences:
            return _sentence(
                f"{self.named} is not one of {self.rsid}'s placements on the Run's "
                f"assembly ({on})"
            )
        if self.named is not None:
            return _sentence(
                f"{self.source} places {self.rsid} at more than one locus on "
                f"{self.named}"
            )
        return _sentence(
            f"{self.source} places {self.rsid} at {self.loci} loci on the Run's "
            f"assembly, on {on}"
        )

    def words(self) -> str:
        return f"{self.rsid} has more than one place on the genome, and none is named"


class VersionUnnamed(_Origin):
    """A stated genome version that is not literally one of the Run's assembly's
    names, with no assembly named for it yet."""

    code: Literal["version_unnamed"] = "version_unnamed"
    version: StrictStr
    assembly_name: StrictStr
    assembly_accession: StrictStr

    def audit(self) -> str:
        return _sentence(
            f"The stated version {self.version!r} is not literally one of the Run's "
            f"assembly's names ({self.assembly_name}, {self.assembly_accession}), and "
            "no assembly is named for it yet"
        )

    def words(self) -> str:
        return (
            f"the genome version {self.version!r} the request names is not yet tied "
            "to a genome"
        )


class PositionWithoutAlleles(_Origin):
    """A position stated without its alleles and with no rsID: neither source lists
    rsIDs at a position without an allele, so it is not looked up, and its item is
    deferred out of the Run (owner #480 Q1)."""

    code: Literal["position_without_alleles"] = "position_without_alleles"
    contig: StrictStr
    position: StrictInt

    def audit(self) -> str:
        return _sentence(
            f"The request states {self.contig}:{self.position} without its alleles "
            "and with no rsID; the Run's sources list rsIDs only at a stated allele, "
            "so the position is not looked up"
        )

    def words(self) -> str:
        # No source is asked about it.
        return (
            f"it states {self.contig}:{self.position} without its alleles or an rsID, "
            "and the Run's sources list a variant only at a stated allele, so no "
            "source was asked about the position"
        )


class Malformed(_Origin):
    """An answer whose bytes could not be read as the source's format."""

    code: Literal["malformed"] = "malformed"
    what: StrictStr
    detail: StrictStr | None = None

    def audit(self) -> str:
        detail = f": {self.detail}" if self.detail else ""
        return _sentence(f"{self.what} could not be read{detail}")

    def words(self) -> str:
        return f"{self.what} could not be read"


class ModelMatchedName(StrictFrozenContract):
    """A sequence the model named for a written contig that is not literally one of
    its names (#467, owner RC5 1a), shown wherever the request is shown (#475)."""

    written: StrictStr
    sequence_name: StrictStr
    record: StrictStr
    assembly_accession: StrictStr

    def audit(self) -> str:
        return (
            f"{self.written} -> {self.sequence_name} ({self.record}) on "
            f"{self.assembly_accession}"
        )

    def words(self) -> str:
        return (
            f"the model read {self.written!r} as the sequence {self.sequence_name} "
            f"({self.record}) on {self.assembly_accession}"
        )


LookupOrigin = Annotated[
    Union[
        SourceAnswered,
        RecordState,
        Merged,
        SequenceUnnamed,
        PlacementUnnamed,
        VersionUnnamed,
        PositionWithoutAlleles,
        Malformed,
    ],
    Field(discriminator="code"),
]


def ending(origin: str | _Origin | None) -> tuple[str | None, dict]:
    """A lookup ending's sentence and the fact fields that keep its origin: an
    origin's audit sentence, or a sentence written where no origin is typed."""
    if origin is None or isinstance(origin, str):
        return origin, {}
    return origin.audit(), {"origin": origin}


def _sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


__all__ = [
    "LookupOrigin",
    "Malformed",
    "Merged",
    "ModelMatchedName",
    "PlacementUnnamed",
    "PositionWithoutAlleles",
    "RecordState",
    "SequenceUnnamed",
    "SourceAnswered",
    "VersionUnnamed",
    "ending",
    "source_words",
]
