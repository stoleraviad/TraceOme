"""Lane P's answers for tests of its targets (#350, #395) and of the chain where lane P's
writer meets lane Rf's reader (#386): #349's retained staged-spike answers and synthetic
answers that code keeps from model readings (``kept_answer``). Moved here from
``tests/test_paper_targets.py`` so both test modules share one source (executable tests may
not import executable tests).
"""

from pathlib import Path

from bio_neb.investigation.investigation_request import StatedAssembly
from bio_neb.paper_reading.contracts import (
    KeptFinding,
    PaperAnswer,
    PaperDocument,
)
from bio_neb.paper_reading.reader import Paper
from bio_neb.target_request.paper_targets import (
    PaperIdentity,
    ReferenceCandidate,
    SourceLine,
    StatedReference,
    VariantCandidate,
    candidate_id,
    reference_key,
)
from tests.paper_reading_support import kept_answer, reading_of

RUNS = Path(__file__).parent / "fixtures/paper_retrieval/answers"

KCNH2_ROW = (
    "KCNH2\tchr7:150951552/NM_000238.4\tc.1841C>T p. (Ala614Val)\tMissense variant"
    "\tD/PD\trs199472944\tPathogenic"
)
AKAP9_ROW = (
    "AKAP9\tchr7:92042078/NM_005751.5\tc.4960_4961del p. (Arg1654GlyfsTer23)"
    "\tFrameshift Indels\tNA/NA\trs779447911\tNot Reported (US)"
)
TTN_ROW = (
    "TTN\tchr2:178532658/NM_173,648.4\tc.103957C>T p. (Arg34653Cys)\tMissense variant"
    "\tNA/PD\trs773002407\tUS"
)
H1_PAPER = "paper:doi:10.3389/fgene.2024.1395012"
# h1's reference candidate's first line (2.5 Genomic data analysis), which states hg38.
S17_EQUCAB3 = StatedAssembly(
    name="EquCab3.0",
    cited_line=(
        "The only associated \u2018HIGH\u2019 effect nonsense variant was on chr4 (EquCab3.0 "
        "chr4:54108297) in exon 26 of RAPGEF5 (c.2624C>A p.Ser875*) and was exclusive to TBs."
    ),
)
S3_SACCER1 = StatedAssembly(
    name="sacCer1",
    cited_line=(
        "Reads were aligned to the UCSC sacCer1 reference sequence using Maq [23] with "
        "default parameters for single-end reads (12,274,183 evolved strain and 10,441,548 "
        "parental strain reads), to a coverage of \u226599.8%."
    ),
)


def recorded_answer(run: str) -> PaperAnswer:
    """A #349 retained answer, read as the reader's answer. Derived: its ``documents``
    listing predates the stored-text pointer (#383 Q2: a document by its text's digest)
    and no target reads it, so it is dropped; every other value is as retained."""
    import json

    retained = json.loads((RUNS / run / "answer.json").read_text(encoding="utf-8"))
    retained.pop("documents", None)
    return PaperAnswer.model_validate(retained)


def located(line: SourceLine, contig: str, position: int, ref: str, alt: str):
    """Synthetic: one located candidate stated on ``line``."""

    return VariantCandidate(
        candidate_id=candidate_id("v", f"{contig}:{position}:{ref}>{alt}"),
        contig=contig,
        position=position,
        ref=ref,
        alt=alt,
        sources=(line,),
    )


def synthetic_answer(*findings: KeptFinding) -> PaperAnswer:
    return PaperAnswer(
        question="Which variants does the paper report?",
        paper=PaperIdentity(doi="10.1000/synthetic", resolved_by="user_supplied"),
        findings=findings,
    )


def synthetic_reference(*texts: str, name: str = "GRCh38") -> StatedReference:
    """Synthetic: a reference candidate named on each of ``texts``."""

    lines = []
    start = 0
    for text in texts:
        lines.append(
            SourceLine(
                document_id="main",
                section="Methods",
                start=start,
                end=start + len(text),
                text=text,
            )
        )
        start += len(text) + 1
    return StatedReference(
        candidate=ReferenceCandidate(
            candidate_id=candidate_id("r", reference_key(name)),
            name=name,
            spellings=(name,),
            sources=tuple(lines),
        ),
        reason="The Methods state it.",
    )


def paper_of(*paragraphs: str) -> tuple[str, Paper]:
    """Synthetic: a main text of these paragraphs under one Results heading."""

    text = "## Results\n" + "\n".join(paragraphs) + "\n"
    return text, Paper(
        PaperIdentity(doi="10.1000/synthetic", resolved_by="user_supplied"),
        [
            PaperDocument(
                document_id="main",
                kind="main_text",
                url="https://example.org/paper",
                text=text,
            )
        ],
    )


def span_at(paper: Paper, offset: int) -> str:
    """The span id the reading Tools show for the main text's line holding ``offset``."""

    return paper.span_of("main", paper.lines["main"].at(offset))


def stated(slot: str, written: str, span: str, **more) -> dict:
    """One value as the model states it: its slot, its written text and its span."""

    return {"slot": slot, "written": written, "span": span, **more}


def read_as(
    text: str,
    paper: Paper,
    *findings: list[dict],
    reference: str | None = None,
    aliases: tuple[str, ...] = (),
) -> PaperAnswer:
    """The answer code keeps from these model readings (``kept_answer``): each finding its
    stated values, and the reference by the last line that writes it (the Methods), with
    the other names that line writes for it. Every finding is accepted."""

    answer = kept_answer(
        "Which variants does the paper report?",
        reading_of(
            *({"values": values, "reason": "R."} for values in findings),
            reference=None
            if reference is None
            else {
                "name": reference,
                "aliases": list(aliases),
                "span": span_at(paper, text.rindex(reference)),
                "reason": "The Methods name it.",
            },
            not_answered=None if reference else "No version is stated.",
        ),
        paper,
    )
    assert not (answer.refused or answer.unresolved), (
        answer.refused,
        answer.unresolved,
    )
    return answer


S3_SENTENCE = (
    "We detected four single-point differences: the previously-described point "
    "mutations at PBP2 (Y127X), SGF73 (E294X), and UPF3 (G6W) [4], and an additional "
    "point mutation intergenic to RRN3 and YPK1 (chr11: 207,469:A>C)."
)
S3_METHODS = (
    "Reads were aligned to the UCSC sacCer1 reference sequence using Maq with default "
    "parameters."
)
# The located finding's quote: code's, from its first to its last kept value.
S3_LOCATED = "YPK1 (chr11: 207,469:A>C)"
# s3's four findings as the model states them, all on the sentence's span (main:p2).
S3_READING = (
    [stated("gene", "PBP2", "main:p2"), stated("change", "Y127X", "main:p2")],
    [stated("gene", "SGF73", "main:p2"), stated("change", "E294X", "main:p2")],
    [stated("gene", "UPF3", "main:p2"), stated("change", "G6W", "main:p2")],
    [
        stated("gene", "YPK1", "main:p2"),
        stated("contig", "chr11", "main:p2"),
        stated("position", "207,469", "main:p2", number=207469),
        stated("alleles", "A>C", "main:p2"),
    ],
)
S17_SENTENCE = (
    "The only associated 'HIGH' effect nonsense variant was on chr4 (EquCab3.0 "
    "chr4:54108297) in exon 26 of RAPGEF5 (c.2624C>A p.Ser875*) and was exclusive to TBs."
)
S17_ROW = "4\t54108297\tG\tT"


def s3_answer() -> PaperAnswer:
    text, paper = paper_of(S3_SENTENCE, S3_METHODS)
    return read_as(text, paper, *S3_READING, reference="sacCer1")


def s17_answer() -> PaperAnswer:
    text, paper = paper_of(
        S17_SENTENCE, "CHROM\tPOS\tREF\tALT", S17_ROW, "Reads were mapped to EquCab3.0."
    )
    # The sentence is main:p2; the table's header row t1:r1 and its row t1:r2.
    return read_as(
        text,
        paper,
        [
            stated("gene", "RAPGEF5", "main:p2"),
            stated("change", "c.2624C>A", "main:p2"),
            stated("change", "p.Ser875*", "main:p2"),
            stated("contig", "4", "t1:r2", column="CHROM"),
            stated("position", "54108297", "t1:r2", column="POS", number=54108297),
            stated("ref", "G", "t1:r2", column="REF"),
            stated("alt", "T", "t1:r2", column="ALT"),
        ],
        reference="EquCab3.0",
    )
