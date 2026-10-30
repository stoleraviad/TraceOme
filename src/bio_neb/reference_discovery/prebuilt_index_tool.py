"""The planning model's Galaxy prebuilt-index Tools: their inputs, copied results and
source reads (#464; ADR 0049: code copies, the model decides, code checks membership).

`search_prebuilt_indexes` reads one of Galaxy's four datacache repositories (its BWA-MEM
index catalogue and its FASTA catalogue, each one GET) and copies every row as published;
its result is one page of the rows matching the model's search text, with the counts and
the next offset. `read_prebuilt_index` reads one listed row's candidate: its `.ann` (one
GET) names its contigs and their lengths, a HEAD gives each of its seven files' size, and
its contigs are paired with NCBI's copied rows of the Run's assembly (`contigs.pair_contigs`).
Nothing is ranked or judged: which index the Run binds is the model's choice
(`record_prebuilt_index_choice`), checked only by membership.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, Union

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.contracts import FrozenContract
from bio_neb.failure import ProviderError
from bio_neb.parabricks.bwa_index import FormatInvalid, read_ann
from bio_neb.prebuilt_index.catalogue import (
    PAGE_ROWS,
    REPOSITORIES,
    CatalogueRow,
    CatalogueUnreadable,
    GalaxyCandidate,
    matches,
    parse_loc,
    row_candidate,
)
from bio_neb.prebuilt_index.choice import PrebuiltFile
from bio_neb.prebuilt_index.contigs import IndexContigs, pair_contigs
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
    ReferenceAdapterError,
    ReferenceAdapterFailure,
)

GALAXY_OPERATION = "galaxy_prebuilt_index"
"""The failure surface of the Tools' Galaxy contacts (a catalogue is about 1.1 MB)."""
NCBI_OPERATION = "describe_companion"
"""The surface of NCBI's assembly report, when no retained answer of it is reachable."""

PrebuiltRepository = Literal["indexes", "managed", "brc", "vgp"]


class PrebuiltIndexGrounding(FrozenContract):
    """The verified assembly an index search answers: the description that verified it
    and its accession. The index is the assembly's, searched before any of its
    companions is described (owner, #459 D1)."""

    reference_event_id: StrictStr
    assembly_accession: StrictStr = Field(pattern=ASSEMBLY_ACCESSION_PATTERN)


class SearchPrebuiltIndexesInput(FrozenContract):
    grounding: PrebuiltIndexGrounding
    repository: PrebuiltRepository
    search_text: StrictStr | None = Field(default=None, min_length=1, max_length=200)
    offset: StrictInt = Field(default=0, ge=0)


class ListedIndexRow(FrozenContract):
    """One BWA-MEM catalogue row as Galaxy publishes it, and the candidate it lists
    when its index and FASTA are both on the datacache."""

    catalogue: StrictStr
    value: StrictStr
    dbkey: StrictStr
    name: StrictStr
    path: StrictStr
    candidate_id: StrictStr | None


class PrebuiltIndexesObserved(FrozenContract):
    """One page of a repository's BWA-MEM rows matching the search text, in Galaxy's
    order, with the catalogue's whole count, the matches' count and the next offset."""

    status: Literal["observed"] = "observed"
    repository: PrebuiltRepository
    search_text: StrictStr | None
    offset: StrictInt = Field(ge=0)
    rows: tuple[ListedIndexRow, ...] = Field(max_length=PAGE_ROWS)
    candidates: tuple[GalaxyCandidate, ...]
    """The candidate of each page row that lists one, in row order."""
    total_rows: StrictInt = Field(ge=0)
    total_matches: StrictInt = Field(ge=0)
    next_offset: StrictInt | None = Field(default=None, gt=0)
    summary: StrictStr

    @model_validator(mode="after")
    def candidates_are_the_rows(self) -> PrebuiltIndexesObserved:
        if [c.candidate_id for c in self.candidates] != [
            row.candidate_id for row in self.rows if row.candidate_id is not None
        ]:
            raise ValueError("a listing's candidates are its rows' candidates")
        return self

    def candidate(self, value: str) -> GalaxyCandidate | None:
        """The page's candidate listed under ``value``: a read's membership check."""
        return next(
            (candidate for candidate in self.candidates if candidate.value == value),
            None,
        )


class PrebuiltIndexToolFailed(FrozenContract):
    status: Literal["failed"] = "failed"
    failure: ReferenceAdapterFailure
    summary: StrictStr


SearchPrebuiltIndexesResult = Annotated[
    Union[PrebuiltIndexesObserved, PrebuiltIndexToolFailed],
    Field(discriminator="status"),
]


class ReadPrebuiltIndexInput(FrozenContract):
    grounding: PrebuiltIndexGrounding
    listing_event_id: StrictStr
    value: StrictStr = Field(min_length=1, max_length=300)
    candidate: GalaxyCandidate
    assembly_accession: StrictStr | None = Field(
        default=None,
        pattern=ASSEMBLY_ACCESSION_PATTERN,
        exclude_if=lambda value: value is None,
    )
    """Another assembly of the Run's organism the contigs are paired against, when the
    verified one has no fitting index (owner, #459 D1: its newer version, against its
    own report); None pairs against the verified assembly."""

    @model_validator(mode="after")
    def the_listed_row(self) -> ReadPrebuiltIndexInput:
        if self.candidate.value != self.value:
            raise ValueError("a read names the listed row it reads")
        return self

    @property
    def pairing_accession(self) -> str:
        """The assembly whose NCBI report the contigs are paired against."""
        return self.assembly_accession or self.grounding.assembly_accession

    @property
    def other_assembly(self) -> str | None:
        """The paired assembly when it is not the verified one."""
        paired = self.pairing_accession
        return None if paired == self.grounding.assembly_accession else paired


class PrebuiltIndexReadObserved(FrozenContract):
    """A candidate's files and sizes, and its contigs beside the Run's sequences."""

    status: Literal["observed"] = "observed"
    candidate: GalaxyCandidate
    files: tuple[PrebuiltFile, ...]
    contigs: IndexContigs
    assembly_report_event_id: StrictStr | None
    """The retained GET of NCBI's assembly report the pairing read; None when this
    call made its own GET of it (then it is among the call's source answers)."""
    summary: StrictStr


ReadPrebuiltIndexResult = Annotated[
    Union[PrebuiltIndexReadObserved, PrebuiltIndexToolFailed],
    Field(discriminator="status"),
]


def _failure(error: Exception) -> ReferenceAdapterFailure:
    """A Galaxy or NCBI answer the Tool could not copy, as the loop routes it: the
    source's own answer (a client status) reaches the model (#458)."""
    if isinstance(error, ReferenceAdapterError):
        return error.failure
    if isinstance(error, ProviderError):
        answered = error.evidence.get("source_code") == "source_answered"
        return ReferenceAdapterFailure(
            failure_class=error.failure_class,
            original_failure_class=error.evidence.get(
                "original_failure_class", error.failure_class
            ),
            contact_attempt=error.evidence.get("contact_attempt", 1),
            code="source_answered" if answered else "http_error",
            public_message=(error.detail or "Galaxy's datacache failed.")[:500],
            retryable=error.failure_class == "transient",
        )
    return ReferenceAdapterFailure(
        code="invalid_response",
        public_message=f"Galaxy's answer is not the shape it is read as: {error}"[:500],
        retryable=False,
    )


def _failed(error: Exception) -> PrebuiltIndexToolFailed:
    failure = _failure(error)
    return PrebuiltIndexToolFailed(failure=failure, summary=failure.public_message)


Contact = Callable[[str, Callable[[], Any]], Any]
"""``contact(surface, call)``: one source contact on its failure surface
(`source_contacts.call_source`)."""


def search_catalogues(
    tool_input: SearchPrebuiltIndexesInput, *, galaxy: Any, contact: Contact
) -> SearchPrebuiltIndexesResult:
    """The repository's BWA-MEM and FASTA catalogues, each read whole, and the page of
    BWA-MEM rows the model's search text matches from its offset."""
    repository = REPOSITORIES[tool_input.repository]
    try:
        index_rows = _catalogue(
            galaxy, contact, repository.index_catalogue, tool_input.repository, "bwa"
        )
        fasta_rows = _catalogue(
            galaxy, contact, repository.fasta_catalogue, tool_input.repository, "fasta"
        )
    except (ProviderError, CatalogueUnreadable, UnicodeDecodeError) as error:
        return _failed(error)
    matched = [row for row in index_rows if matches(row, tool_input.search_text)]
    page = matched[tool_input.offset : tool_input.offset + PAGE_ROWS]
    found = [row_candidate(row, fasta_rows) for row in page]
    end = tool_input.offset + len(page)
    next_offset = end if end < len(matched) else None
    searched = (
        f" matching {tool_input.search_text!r}"
        if tool_input.search_text is not None
        else ""
    )
    return PrebuiltIndexesObserved(
        repository=tool_input.repository,
        search_text=tool_input.search_text,
        offset=tool_input.offset,
        rows=tuple(
            ListedIndexRow(
                **row.model_dump(),
                candidate_id=None if candidate is None else candidate.candidate_id,
            )
            for row, candidate in zip(page, found, strict=True)
        ),
        candidates=tuple(candidate for candidate in found if candidate is not None),
        total_rows=len(index_rows),
        total_matches=len(matched),
        next_offset=next_offset,
        summary=(
            f"Galaxy's {tool_input.repository} BWA-MEM catalogue lists "
            f"{len(index_rows)} rows, {len(matched)}{searched}; rows "
            f"{tool_input.offset + 1 if page else tool_input.offset}-{end} are shown"
            + (f", offset {next_offset} reads the next page." if next_offset else ".")
        ),
    )


def _catalogue(galaxy, contact, url, repository, kind) -> tuple[CatalogueRow, ...]:
    name = f"{repository}-{'bwa_mem_index' if kind == 'bwa' else 'fasta_indexes'}.loc"
    body = contact(GALAXY_OPERATION, lambda: galaxy.get(url))
    return parse_loc(body.decode("utf-8"), catalogue=name)


def read_candidate(
    tool_input: ReadPrebuiltIndexInput,
    *,
    galaxy: Any,
    contact: Contact,
    sequences: Callable[[], tuple[Sequence[Any], str | None]],
) -> ReadPrebuiltIndexResult:
    """The candidate's `.ann` contigs paired with NCBI's rows of the Run's assembly,
    and each file's size from a HEAD; nothing large is downloaded.

    ``sequences()`` gives NCBI's copied rows and the retained report GET they were
    read from (None when it made its own GET)."""
    candidate = tool_input.candidate
    files = candidate.files()
    try:
        ann = read_ann(contact(GALAXY_OPERATION, lambda: galaxy.get(files["bwa_ann"])))
        rows, report_event_id = sequences()
        sized = tuple(
            PrebuiltFile(
                role=role,
                url=url,
                byte_count=contact(GALAXY_OPERATION, lambda url=url: galaxy.size(url)),
            )
            for role, url in files.items()
        )
    except (ProviderError, FormatInvalid) as error:
        return _failed(error)
    contigs = pair_contigs(
        tool_input.pairing_accession,
        rows,
        [(name, layout[1]) for name, layout in zip(ann.names, ann.layout, strict=True)],
    )
    paired = sum(1 for pair in contigs.requested if pair.status == "paired")
    against = (
        ""
        if tool_input.other_assembly is None
        else f" against {tool_input.other_assembly}'s own NCBI report"
    )
    return PrebuiltIndexReadObserved(
        candidate=candidate,
        files=sized,
        contigs=contigs,
        assembly_report_event_id=report_event_id,
        summary=(
            f"{candidate.candidate_id} has {len(ann.names)} contigs{against}: "
            f"{paired} of {len(contigs.requested)} reference sequences paired at "
            f"their lengths ({contigs.renamed} under another name), "
            f"{len(contigs.extras)} extra, {len(contigs.ambiguous)} ambiguous; it "
            + (
                "holds the main assembly."
                if contigs.holds_every_requested_sequence
                else "does not hold the main assembly."
            )
        ),
    )


__all__ = [
    "GALAXY_OPERATION",
    "NCBI_OPERATION",
    "ListedIndexRow",
    "PrebuiltIndexGrounding",
    "PrebuiltIndexReadObserved",
    "PrebuiltIndexToolFailed",
    "PrebuiltIndexesObserved",
    "PrebuiltRepository",
    "ReadPrebuiltIndexInput",
    "ReadPrebuiltIndexResult",
    "SearchPrebuiltIndexesInput",
    "SearchPrebuiltIndexesResult",
    "read_candidate",
    "search_catalogues",
]
