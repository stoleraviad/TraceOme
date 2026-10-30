"""The records placement keeps (#367): a target's answer on the Run's assembly and the
sources it read, a source contact's failure, G15's gene-and-change answer and the flank
move the placement port answers.

Every model is frozen and rejects extra fields. Nothing here is a Journey event or grants
authority: a placement record is kept beside the request file it writes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.failure import FailureClass, FailureRoute, ResolvedFailurePolicy
from bio_neb.investigation.investigation_request import StatedChange, VersionForm
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
)
from bio_neb.target_request.gene_change import GeneticCodeOrigin

ALLELE_PATTERN = r"^[ACGTN]+$"
PLACED_SCHEMA = "bio-neb.paper-targets-placed/6"


class SourceFailure(FrozenContract):
    """The source contact the failure module gave up on, as the claim's outcome.

    `failure_class` is the routed class (`exhausted` once a transient fault used its
    attempts), `original_failure_class` the adapter's, and `route` what the resolved policy
    does with it (`ask`: the claim waits for a person, the other claims run on)."""

    tool: str
    failure_class: FailureClass
    original_failure_class: FailureClass
    contact_attempts: int = Field(ge=1)
    route: FailureRoute
    detail: str
    policy: ResolvedFailurePolicy


class LookupStop(FrozenContract):
    """The claim a run stopped on (a request the run plan refuses, an assembly report that
    could not be read, or our bug), why, and the claims it did not try. A source contact's
    failure is the claim's or the target's answer, not a stop."""

    claim_label: str
    error: str
    unattempted: tuple[str, ...]


GeneChangeOutcome = Literal["candidates", "ambiguous", "not_found", "not_checked"]


class StatedVersion(FrozenContract):
    """The genome version a paper states and the line that states it, as written."""

    name: str = Field(min_length=1, max_length=200)
    cited_line: str = Field(min_length=1)


class GeneChangeClaim(FrozenContract):
    """A finding stated as a gene and a protein or coding change (G15, #376), with no
    position: s3's ``PBP2 (Y127X)``, s17's ``RAPGEF5 (c.2624C>A p.Ser875*)``.

    `change` is the gene and change as the source writes them, with the parts the model
    typed (#473 Q2 a) and the transcript only when the words name one; `request_text`
    the line that states the finding. A change with no typed part is not read."""

    label: str = Field(min_length=1, max_length=200)
    change: StatedChange
    stated_assembly: StatedVersion | None = None
    request_text: str = Field(min_length=1)


class VersionResolution(FrozenContract):
    """The stated version against the Run's assembly (NCBI's dataset report): which of the
    Run's names it is, or `matched` None when it names another assembly or none NCBI
    knows (sacCer1 has no accession; #369's development GET g3)."""

    stated: str | None
    run_accession: str = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)
    run_name: str
    run_synonym: str | None
    run_paired_accession: str | None
    matched: VersionForm | None


class TranscriptRead(FrozenContract):
    """What one transcript gave at the stated codon, with its genetic code (and where
    that came from) and its annotation's tags, copied (#469)."""

    accession: str
    protein: str | None
    contig: str
    strand: Literal["+", "-"]
    genetic_code: int = Field(gt=0)
    genetic_code_origin: GeneticCodeOrigin
    tags: tuple[str, ...]
    cds_phase: int = Field(ge=0, le=2)
    codon_number: int = Field(gt=0)
    reference_codon: str | None
    reference_residue: str | None
    matches: bool
    reason: str | None
    unplaced: tuple[str, ...] = ()


class GeneCandidate(FrozenContract):
    """One genomic change on the Run's assembly that gives the stated change, in VCF form,
    and the transcripts that give it."""

    contig: str
    position: int = Field(gt=0)
    ref: str = Field(pattern=ALLELE_PATTERN)
    alt: str = Field(pattern=ALLELE_PATTERN)
    bases_changed: int = Field(ge=1, le=3)
    reference_codon: str
    alt_codon: str
    transcripts: tuple[str, ...] = Field(min_length=1)


class HgvsAnswer(FrozenContract):
    """NCBI's HGVS service's own answer for the name built from the claim's typed
    transcript and change (#473 Q3 a), copied: its HTTP status and body as sent (a 400
    "The submitted sequence is not human." included), and, for a coding name, every
    contextual SPDI it names, their equivalents on the Run's sequences and which of
    them are a candidate's. No verdict: a disagreement is the model's to read."""

    hgvs: str = Field(min_length=1)
    status_code: int = Field(ge=100, le=599)
    body: str
    spdis: tuple[str, ...] = ()
    on_run: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()


class GeneChangeAnswer(FrozenContract):
    """One claim's typed answer: `candidates`, `ambiguous` (each transcript's candidates
    kept, or none when the symbol names several genes), `not_found` or `not_checked`, with
    the reason; or the failure its source contact or source reading ended in. No claim is dropped and no candidate is made without its codon read."""

    claim: GeneChangeClaim
    outcome: GeneChangeOutcome | None
    reason: str | None
    version: VersionResolution | None
    transcripts: tuple[TranscriptRead, ...] = ()
    candidates: tuple[GeneCandidate, ...] = ()
    hgvs: HgvsAnswer | None = None
    failure: SourceFailure | None = None

    @model_validator(mode="after")
    def _one_answer(self) -> GeneChangeAnswer:
        if (self.outcome is None) == (self.failure is None):
            raise ValueError("an answer has an outcome or a failure, not both")
        if self.candidates and self.outcome not in ("candidates", "ambiguous"):
            raise ValueError("candidates come with `candidates` or `ambiguous` only")
        if self.outcome == "candidates" and not self.candidates:
            raise ValueError("a `candidates` answer has candidates")
        return self


Base = Literal["A", "C", "G", "T"]
FlankResult = Literal[
    "unique", "several_places", "not_found", "ref_changed", "strand_flipped"
]


class FlankMoveRequest(FrozenContract):
    """One SNV as stated on its version, to be moved onto the Run's stored index: the
    version's name, its sequence's accession, the 1-based position and one-base REF and
    ALT."""

    label: str = Field(min_length=1)
    assembly: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    position: int = Field(ge=1)
    ref: Base
    alt: Base

    @model_validator(mode="after")
    def _a_change(self) -> FlankMoveRequest:
        if self.ref == self.alt:
            raise ValueError("REF and ALT differ")
        return self


class FlankPlace(FrozenContract):
    """Where the moved base lands on the Run's assembly."""

    contig: str
    position: int = Field(ge=1)


class FlankMove(FrozenContract):
    """The placement port's answer for one SNV: the move's result, its accepted place on the
    Run's assembly, the alleles there (forward strand) and the move's reason."""

    result: FlankResult
    place: FlankPlace | None
    target_ref: Base | None = None
    target_alt: Base | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _the_result_follows_the_place(self) -> FlankMove:
        placed = self.result in {"unique", "ref_changed", "strand_flipped"}
        if placed != (self.place is not None):
            raise ValueError(f"a {self.result} result has a place only when placed")
        alleles = self.result in {"unique", "strand_flipped"}
        if alleles != (self.target_ref is not None and self.target_alt is not None):
            raise ValueError(
                "alleles on the Run are given for unique and strand_flipped only"
            )
        return self


class SequenceProof(FrozenContract):
    """Why a row's sequence is the stated version's contig: where the evidence is, and the
    matching window it shows (owner, #393 Q5: each row carries its proof)."""

    source: str = Field(min_length=1)
    window: str = Field(min_length=1)


class StatedSequenceRow(FrozenContract):
    """One stated version's contig, as a paper writes it, and the sequence it is."""

    version: str = Field(min_length=1)
    contig: str = Field(min_length=1)
    accession: str = Field(pattern=r"^[A-Z]{1,2}_?[0-9]{5,9}\.[0-9]+$")
    proof: SequenceProof


class StatedSequences(FrozenContract):
    """A Run's stated-sequence table (`bio-neb.stated-sequences/1`): the sequences of stated
    versions NCBI does not name, one proven row per case (owner, #393 Q5, Q7). A version
    matches case-folded and a contig exactly as written (#467: no code equates contig
    spellings; #459/#414 replace the table), so one row answers each pair."""

    schema_version: Literal["bio-neb.stated-sequences/1"]
    name: str = Field(min_length=1)
    rows: tuple[StatedSequenceRow, ...]

    @model_validator(mode="after")
    def _one_row_per_contig(self) -> StatedSequences:
        keys = [_row_key(row.version, row.contig) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("a table has one row per version and contig")
        return self

    def row(self, version: str, contig: str) -> StatedSequenceRow | None:
        key = _row_key(version, contig)
        return next(
            (row for row in self.rows if _row_key(row.version, row.contig) == key),
            None,
        )


def _row_key(version: str, contig: str) -> tuple[str, str]:
    return version.casefold(), contig


class TableUsed(FrozenContract):
    """The stated-sequence table a run read: its name and the sha256 of its bytes."""

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


PlacedAnswer = Literal[
    "kept", "moved", "placed", "passed", "ambiguous", "not_found", "not_checked"
]
ResolverStep = Literal[
    "same_version",
    "table",
    "database_equivalence",
    "flank_alignment",
    "g15",
    "rsid_only",
    "none",
]
"""The step that answered a target (owner, #393 Q8). A positioned target tries, in order:
the Run's own version (`same_version`); the Run's table row and a flank move from its
sequence (`table`); NCBI's Variation service on the sequence NCBI names
(`database_equivalence`); a flank move from that sequence (`flank_alignment`). A gene and
change is G15's (`g15`), an rsID alone `rsid_only`. `none`: no step answered, and the reason
names each step's miss."""


class StatedSequence(FrozenContract):
    """The stated contig's sequence on the stated version, and who named it: the Run's
    table, or NCBI (`assembly_accession`, the stated version's NCBI assembly)."""

    accession: str
    named_by: Literal["table", "ncbi"]
    assembly_accession: str | None = None

    @model_validator(mode="after")
    def _ncbi_names_its_assembly(self) -> StatedSequence:
        if (self.named_by == "ncbi") != (self.assembly_accession is not None):
            raise ValueError("a sequence NCBI names comes with NCBI's assembly")
        return self


class FlankMoveUnavailable(FrozenContract):
    """A flank move the target needs and the Run holds no mover for (owner, #449 Q1 a): the
    move is development-only (ADR 0020), so a Run passes none. Its class and route are the
    failure package's (`rejected`, routed `ask`): the target waits for its Review, retry or
    accept, and the other targets are answered. No contact was made."""

    kind: Literal["flank_move_unavailable"] = "flank_move_unavailable"
    tool: Literal["flank_move"] = "flank_move"
    step: Literal["table", "flank_alignment"]
    stated_sequence: StatedSequence
    failure_class: FailureClass
    route: FailureRoute
    detail: str = Field(min_length=1)


class Equivalence(FrozenContract):
    """Database equivalence (step 3): the stated allele as an SPDI on the stated sequence,
    and the equivalents NCBI's Variation service gives on the Run's sequences."""

    spdi: str
    on_run: tuple[str, ...]


class RunPosition(FrozenContract):
    """Where a kept or moved target is on the Run's assembly."""

    contig: str
    position: int = Field(ge=1)
    ref: str | None = None
    alt: str | None = None


class TargetPlacement(FrozenContract):
    """One lane P target's answer on the Run's assembly (#385, #393), and the indexes of
    the ``/2`` items written for it.

    `kept`: its position is stated on the Run's version; `moved`: a step placed it on the
    Run's assembly (`position`); `placed`: G15 placed its gene and change (`gene_change`);
    `passed`: its rsID only, which main looks up; `ambiguous`, `not_found`, `not_checked`: why
    it has no one place, said in `reason`. `step` names the step that answered. A source
    contact the failure module gave up on is `failure`, with no answer, and so is a flank move
    a Run holds no mover for (`FlankMoveUnavailable`); once the owner accepts it
    (`accepted`), it is written as unplaced with the failure named."""

    label: str = Field(min_length=1)
    quote: str
    answer: PlacedAnswer | None
    step: ResolverStep
    reason: str = Field(min_length=1)
    version: VersionResolution | None = None
    matched_name: str | None = None
    stated_sequence: StatedSequence | None = None
    equivalence: Equivalence | None = None
    move: FlankMove | None = None
    gene_change: GeneChangeAnswer | None = None
    position: RunPosition | None = None
    failure: SourceFailure | FlankMoveUnavailable | None = None
    accepted: bool = False
    requests: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _one_answer(self) -> TargetPlacement:
        if (self.answer is None) == (self.failure is None):
            raise ValueError("a placement has an answer or a failure, not both")
        if self.accepted and self.failure is None:
            raise ValueError("only a failure is accepted")
        if (self.position is not None) != (self.answer in ("kept", "moved")):
            raise ValueError("a kept or moved target, and only one, has its position")
        return self


class PaperTargetsPlaced(FrozenContract):
    """One `place` run (#393): every lane P target's answer, the table it read, its failure
    and contact counts, the targets waiting for you, where it stopped if it did, and the
    ``/2`` file the writer wrote, if any.

    A failure routed to `ask` waits (`waiting`) until the owner retries or accepts it; no
    ``/2`` is written while one waits. A plan-level problem stops the run (`stopped`): a
    request the product's transport refuses, an unreadable record or our bug."""

    schema_version: Literal["bio-neb.paper-targets-placed/6"] = PLACED_SCHEMA
    made_at: str
    assembly_accession: str
    paper_targets_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    asserted_by: str | None
    table: TableUsed | None
    resumed_from: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    answers: tuple[TargetPlacement, ...]
    failure_count: int = Field(ge=0)
    waiting: tuple[str, ...]
    stopped: LookupStop | None
    request_file: str | None = None
    request_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    contact_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _counts_follow_the_answers(self) -> PaperTargetsPlaced:
        failed = [answer for answer in self.answers if answer.failure is not None]
        if self.failure_count != len(failed):
            raise ValueError("failure_count counts the answers' failures")
        asking = tuple(
            answer.label
            for answer in failed
            if answer.failure.route == "ask" and not answer.accepted
        )
        if self.waiting != asking:
            raise ValueError("waiting names each failure that asks and is not accepted")
        if self.request_file is not None and (self.stopped or self.waiting):
            raise ValueError("no request file is written while a run stops or waits")
        if self.paper_targets_sha256 is None and self.stopped is None:
            raise ValueError("only a run that stopped has no targets digest")
        return self
