"""NCBI's answers that move one position from GRCh37 to GRCh38 (#462 R2's #393
development GETs): the listing of every human RefSeq version, GRCh37.p13's sequence
report, the SPDI's equivalents, GRCh38's bases and dbSNP's rsID there. Shared by the
Investigation's in-Run move (`test_position_targets.InRunPlacementTests`) and the
planning check of a suggested assembly's targets (#459 D3)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.refsnp_retained_bodies import RS429358
from tests.sequence_naming_support import assembly_report_text
from tests.sequential_investigation_support import SourceResponsesOnly

GRCH37 = "GCF_000001405.25"
GRCH38 = "GCF_000001405.40"


def grch37_move_sources():
    """rs429358's allele chr19:45411941 T>C on GRCh37, NC_000019.10:44908684 on
    GRCh38: the sources that answer its move."""
    placement = Path(__file__).parent / "fixtures" / "paper_targets" / "ncbi"
    report = json.loads(SourceResponsesOnly().assembly_body)
    report["reports"][0]["organism"] = {"tax_id": 9606}

    class _Sources(SourceResponsesOnly):
        def fetch_sequence_names(self, assembly_accession):  # noqa: ANN001, ANN201
            if assembly_accession != "GCF_000001405.25":
                return super().fetch_sequence_names(assembly_accession)
            return self._answer(
                "ncbi_sequence_names",
                assembly_accession,
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                "GCF_000001405.25/assembly_report.txt",
                200,
                assembly_report_text(
                    (placement / "gcf-000001405.25-sequence-report.json").read_bytes()
                ),
            )

        def fetch_taxon_assemblies(self, taxon_query):  # noqa: ANN001, ANN201
            # NCBI's search for the model's words (synthetic: the recorded
            # listing of every RefSeq version, whatever the query asks).
            return self._answer(
                "ncbi_taxon_assemblies",
                taxon_query,
                f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/{taxon_query}",
                200,
                (placement / "taxon-9606-refseq-all-versions.json").read_bytes(),
            )

        def fetch_spdi_equivalents(self, spdi):  # noqa: ANN001, ANN201
            return self._answer(
                "ncbi_spdi_equivalents",
                spdi,
                f"https://api.ncbi.nlm.nih.gov/variation/v0/spdi/{spdi}/"
                "all_equivalent_contextual",
                200,
                (
                    placement / "spdi-NC_000019.9-45411940-T-C-equivalents.json"
                ).read_bytes(),
            )

    return _Sources(
        assembly_body=json.dumps(report).encode(),
        refsnp_bodies={"rs429358": RS429358},
        reference_bases={"NC_000019.10:44908683:1": "T"},
        listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
    )
