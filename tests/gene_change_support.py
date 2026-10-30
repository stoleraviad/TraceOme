"""Lane P's ``bio-neb.paper-targets/1`` record, hand-built for lane Rf's tests (#376, #385).

The shape is the reader's ``PaperTargets`` (``target_request/paper_targets.py``), written
here as data, as lane Rf reads it without lane P's record models; the chain test checks that
lane P's model accepts it (``tests/test_paper_targets_chain.py``).
Its reference is s3's retained #349 answer (sacCer1's Methods line). Below it, G15's
scripted NCBI answers by URL, shaped as #376's development GETs.
"""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path

from bio_neb.failure import FailureOverrides, FailureTuning
from tests.sequence_naming_support import names_answers

LINE = (
    "Reads were aligned to the UCSC sacCer1 reference sequence using Maq [23] with "
    "default parameters"
)


def _line(text, document="main", section="Results"):
    return {
        "document_id": document,
        "section": section,
        "start": 0,
        "end": len(text),
        "text": text,
    }


def paper_target(text, change=None, position=None, stated=None, rsid=None):
    """One lane P target: main's request item for the paper's line ``text``, as lane P's
    ``PaperTarget`` validates it (its ``not_checked`` follows from the request)."""
    unchecked = (
        None
        if rsid is not None
        else "not_located"
        if position is None
        else "no_alleles"
        if position.get("ref") is None
        else None
    )
    return {
        "request": {
            "request_text": text,
            "rsid": rsid,
            "position": position,
            "stated_assembly": stated,
            "normalized_question_action": "observe_variant",
        },
        "quote": _line(text),
        "kind": "not_stated" if position is None and rsid is None else "snv",
        "not_checked": unchecked,
        "change": change,
        "findings": [
            {
                "variants": [],
                "passages": [_line(text)],
                "reason": "r",
                "gene": None,
                "unlocated": None,
            }
        ],
    }


REFERENCE = {
    "candidate": {
        "candidate_id": "r-b8a9216d7b93",
        "name": "sacCer1",
        "spellings": ["sacCer1"],
        "sources": [_line(LINE, section="Methods")],
    },
    "reason": "the Methods state it",
}


def paper_targets(targets, reference=REFERENCE):
    """s3's record (sacCer1 stated in its Methods) holding ``targets``."""
    return {
        "schema_version": "bio-neb.paper-targets/1",
        "question": "Which point mutations does the paper report?",
        "paper": {"doi": "10.1186/1471-2164-11-88", "resolved_by": "user_supplied"},
        "asserted_by": "paper:doi:10.1186/1471-2164-11-88",
        "reference": reference,
        "targets": targets,
        "not_answered": None,
    }


# Scripted NCBI answers for G15, by URL.

API = "https://api.ncbi.nlm.nih.gov"
YEAST = "GCF_000146045.2"
HUMAN = "GCF_000001405.40"
NO_BACKOFF = FailureTuning(operation=FailureOverrides(backoff_seconds=0))


class Answer:
    def __init__(self, status, body=b""):
        self.status = status
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers = Message()

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Opener:
    def __init__(self, answers):
        self.answers = {
            url: list(answer) if isinstance(answer, list) else answer
            for url, answer in answers.items()
        }
        self.sent = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.sent.append(url)
        answer = self.answers[url]
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if answer.status >= 300:
            raise urllib.error.HTTPError(
                url, answer.status, "scripted", answer.headers, io.BytesIO(answer.body)
            )
        return answer


def dataset_report(accession, name, synonym, paired, taxid, status="current"):
    """NCBI's dataset report of one assembly, in the development GETs' shape. The
    product's reader requires its status: R64 and GRCh38.p14 are current, and EquCab3.0's
    RefSeq report is `suppressed` (#369's development GET g4)."""
    return {
        "reports": [
            {
                "accession": accession,
                "paired_accession": paired,
                "assembly_info": {
                    "assembly_name": name,
                    "synonym": synonym,
                    "assembly_status": status,
                },
                "organism": {"tax_id": taxid},
            }
        ]
    }


def span_url(contig, start, length):
    return f"{API}/variation/v0/spdi/{contig}:{start - 1}:{length}:{'N' * length}/contextual"


def span(contig, start, bases):
    return {
        "data": {
            "seq_id": contig,
            "position": start - 1,
            "deleted_sequence": bases,
            "inserted_sequence": "N" * len(bases),
        }
    }


# A yeast-like gene on R64: one mRNA, CDS 1001-1030 on the plus strand.
YEAST_GFF_TEXT = "\n".join(
    [
        "##gff-version 3",
        "NC_001134.8\tRefSeq\tgene\t1001\t1030\t.\t+\t.\tID=gene-YB1;Name=PBP2;gene=PBP2",
        "NC_001134.8\tRefSeq\tmRNA\t1001\t1030\t.\t+\t.\tID=rna-NM_1.3;Parent=gene-YB1;"
        "gene=PBP2;transcript_id=NM_1.3",
        "NC_001134.8\tRefSeq\tCDS\t1001\t1030\t.\t+\t0\tID=cds-NP_1.3;Parent=rna-NM_1.3;"
        "gene=PBP2;protein_id=NP_1.3",
        "",
    ]
)


def gff_zip(gff):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"ncbi_dataset/data/{YEAST}/genomic.gff", gff)
    return buffer.getvalue()


YEAST_REPORT = f"{API}/datasets/v2/genome/accession/{YEAST}/dataset_report"
YEAST_GFF = (
    f"{API}/datasets/v2/genome/accession/{YEAST}/download"
    "?include_annotation_type=GENOME_GFF"
)


def yeast_answers(**more):
    return {
        YEAST_REPORT: Answer(
            200, dataset_report(YEAST, "R64", "sacCer3", "GCA_000146045.2", 559292)
        ),
        YEAST_GFF: Answer(200, gff_zip(YEAST_GFF_TEXT)),
        # Codon 2 is 1004-1006: TAC (Tyr).
        span_url("NC_001134.8", 1004, 3): Answer(200, span("NC_001134.8", 1004, "TAC")),
        **more,
    }


# The resolver's scripted NCBI answers, shared by lane Rf's placement tests and the chain
# test where lane P's writer meets lane Rf's reader.

HUMAN_REPORT = f"{API}/datasets/v2/genome/accession/{HUMAN}/dataset_report"
HG38_LINE = "alignment to the reference genome (hg38)"
FIXTURES = Path(__file__).parent / "fixtures" / "paper_targets"
NCBI = FIXTURES / "ncbi"
SACCER1_TABLE = FIXTURES / "stated-sequences-saccer1.json"


def listing_url(taxid, stated, page_token=None):
    """NCBI's search for a stated version as placement sends it, with no model: the
    stated words, every version, one report per pair (#471 R4 a)."""
    from bio_neb.investigation.investigation_sources import literal_version_query
    from bio_neb.providers.ncbi_datasets import genome_report_url

    return genome_report_url(literal_version_query(stated, taxid, page_token))


def names_urls(accession):
    """The two GETs of an assembly's sequence names (#467): its links, its report."""
    return tuple(names_answers(accession, b'{"reports":[]}'))


def names(accession, sequence_report):
    """An assembly's sequence-names answers, derived from a retained sequence report
    (`tests.sequence_naming_support.names_answers`)."""
    return {
        url: Answer(200, body)
        for url, body in names_answers(accession, sequence_report).items()
    }


def equivalents_url(spdi):
    return f"{API}/variation/v0/spdi/{spdi}/all_equivalent_contextual"


NO_EQUIVALENT = Answer(
    200, (NCBI / "spdi-NC_001143.9-207822-A-C-equivalents.json").read_bytes()
)


def yeast_listing(stated):
    """NCBI's search for ``stated`` on yeast (synthetic: #393's recorded listing of
    every RefSeq version of taxon 559292 answers each search)."""
    return {
        listing_url(559292, stated): Answer(
            200, (NCBI / "taxon-559292-refseq-all-versions.json").read_bytes()
        )
    }


def human_answers():
    return {
        HUMAN_REPORT: Answer(
            200, dataset_report(HUMAN, "GRCh38.p14", "hg38", "GCA_000001405.29", 9606)
        )
    }


HORSE = "GCF_002863925.1"
HORSE_REPORT = f"{API}/datasets/v2/genome/accession/{HORSE}/dataset_report"


def horse_answers():
    return {
        HORSE_REPORT: Answer(
            200,
            dataset_report(
                HORSE, "EquCab3.0", "equCab3", "GCA_002863925.1", 9796, "suppressed"
            ),
        )
    }
