"""The one request writer (#367): G15's candidates as items of a ``/3`` request file.

Each candidate is a placed position on the Run's assembly that the Run's own reader accepts
(``read_investigation_request``). The paper's gene and change stay its stated target and the
paper's version and line its stated version; neither is rewritten as the Run's
(CONTEXT **Placed position**). Migrated from the retired ``/2`` G15 writer's tests (#376).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bio_neb.investigation.investigation_request import (
    REQUEST_SCHEMA,
    StatedChange,
    read_investigation_request,
)
from bio_neb.target_request.paper_targets import PaperTarget, PaperTargets
from bio_neb.target_request.records import (
    GeneCandidate,
    GeneChangeAnswer,
    GeneChangeClaim,
    StatedVersion,
    TargetPlacement,
    VersionResolution,
)
from bio_neb.target_request.request_document import request_items
from tests.gene_change_support import LINE, paper_target, paper_targets

QUOTE = "the previously-described point mutations at PBP2 (Y127X)"


def _version(stated="sacCer1", matched=None):
    return VersionResolution(
        stated=stated,
        run_accession="GCF_000146045.2",
        run_name="R64",
        run_synonym="sacCer3",
        run_paired_accession="GCA_000146045.2",
        matched=matched,
    )


def _claim(stated=True, cited_line=LINE):
    return GeneChangeClaim(
        label="t1-PBP2",
        change=StatedChange(
            gene="PBP2", change="Y127X", protein={"ref": "Y", "codon": 127, "alt": "*"}
        ),
        stated_assembly=StatedVersion(name="sacCer1", cited_line=cited_line)
        if stated
        else None,
        request_text=QUOTE,
    )


def _candidate(alt="A", codon="TAA"):
    return GeneCandidate(
        contig="NC_001134.8",
        position=683808,
        ref="C",
        alt=alt,
        bases_changed=1,
        reference_codon="TAC",
        alt_codon=codon,
        transcripts=("NM_001178581.3",),
    )


def _placement(outcome="candidates", reason=None, claim=None, version=None):
    candidates = (
        (_candidate(), _candidate("G", codon="TAG"))
        if outcome in ("candidates", "ambiguous")
        else ()
    )
    found = GeneChangeAnswer(
        claim=claim or _claim(),
        outcome=outcome,
        reason=reason,
        version=version or _version(),
        candidates=candidates,
    )
    answer = {"candidates": "placed"}.get(outcome, outcome)
    return TargetPlacement(
        label="t1",
        quote=QUOTE,
        answer=answer,
        step="g15",
        reason=reason or "G15 placed PBP2 (Y127X) at one codon",
        version=found.version,
        gene_change=found,
    )


def _target():
    return PaperTarget.model_validate(
        paper_target(QUOTE, {"gene": "PBP2", "text": "Y127X", "transcript": None})
    )


class RequestItemsTests(unittest.TestCase):
    def _read(self, items):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = root / "request.json"
        document = {
            "schema_version": REQUEST_SCHEMA,
            "asserted_by": "paper:doi:10.1186/x",
            "requests": items,
        }
        path.write_text(json.dumps(document))
        return read_investigation_request(str(path))

    def _paper(self):
        return PaperTargets.model_validate(paper_targets([]))

    def test_every_candidate_is_a_placed_position_the_run_reads(self):
        request = self._read(request_items(_placement(), _target(), self._paper()))

        self.assertEqual(request.asserted_by, "paper:doi:10.1186/x")
        self.assertEqual(
            [
                (i.placed.contig, i.placed.position, i.placed.ref, i.placed.alt)
                for i in request.items
            ],
            [("NC_001134.8", 683808, "C", "A"), ("NC_001134.8", 683808, "C", "G")],
        )
        # The paper's version and line stay stated as written, never the Run's accession.
        self.assertEqual(
            {
                (
                    i.position,
                    i.gene_change.gene,
                    i.gene_change.change,
                    i.stated_assembly.name,
                    i.stated_assembly.cited_line,
                    i.placement.step,
                )
                for i in request.items
            },
            {(None, "PBP2", "Y127X", "sacCer1", LINE, "gene_change")},
        )

    def test_the_request_text_is_the_papers_line_and_the_candidates_placement(self):
        [first, _] = request_items(_placement(), _target(), self._paper())

        self.assertEqual(
            first["request"],
            f"{QUOTE} (placement: placed: candidate 1 of 2: NC_001134.8:683808 C>A gives "
            "codon TAC>TAA on NM_001178581.3 (1 base changed), from that assembly's "
            "annotation)",
        )
        self.assertEqual(
            first["placement"]["reason"],
            "candidate 1 of 2: NC_001134.8:683808 C>A gives codon TAC>TAA on "
            "NM_001178581.3 (1 base changed), from that assembly's annotation",
        )

    def test_ambiguous_candidates_are_written_and_say_why(self):
        reason = "PBP2's transcripts place Y127 at different codons"

        items = request_items(_placement("ambiguous", reason), _target(), self._paper())

        self.assertEqual(len(items), 2)
        self.assertTrue(
            items[0]["placement"]["reason"].startswith(
                f"ambiguous ({reason}), candidate 1 of 2:"
            )
        )

    def test_an_answer_without_candidates_is_written_unplaced_with_its_reason(self):
        [item] = request_items(
            _placement("not_found", "no gene PBP2"), _target(), self._paper()
        )

        self.assertEqual(
            (item["target"], item["placement"]),
            (
                {"gene_change": {"gene": "PBP2", "change": "Y127X"}},
                {"step": "none", "reason": "not_found: no gene PBP2"},
            ),
        )

    def test_a_claim_whose_paper_states_no_version_states_none(self):
        placement = _placement(claim=_claim(stated=False), version=_version(None))

        request = self._read(request_items(placement, _target(), self._paper()))

        self.assertEqual({item.stated_assembly for item in request.items}, {None})

    def test_a_multi_line_cited_line_is_written_as_one_line(self):
        placement = _placement(claim=_claim(cited_line="a\n  b sacCer1 "))

        items = request_items(placement, _target(), self._paper())

        self.assertEqual(items[0]["stated_assembly"]["cited_line"], "a b sacCer1")
