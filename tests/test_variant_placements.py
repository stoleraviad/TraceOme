"""Variant placements are copied, and the model names one by membership (#468, ADR 0049).

The reader (`providers/variant_placements.py`) is driven over #466's recorded answers
(`tests/recorded_sources.py`); the dbSNP lookup (`normalize_refsnp`) over the same record
and over a copy of it given a second placement on the Run's assembly, derived here and
named as such. Every expected value is read from the recorded body by this test, never
from the reader under test.
"""

from __future__ import annotations

import copy
import json
import unittest

from bio_neb.agentic_journey.tool_execution import tool_model_view
from bio_neb.investigation.investigation import RetrievedSource
from bio_neb.investigation.investigation_agent import LOOKUP_VIEW
from bio_neb.investigation.investigation_sources import normalize_refsnp
from bio_neb.providers.variant_placements import (
    PlacementsUnreadable,
    eva_clustered_placements,
    eva_submitted_variants,
    refsnp_placements,
)
from tests.recorded_sources import fixture

GRCH38 = "GCF_000001405.40"
GRCH37 = "GCF_000001405.25"


def _recorded(fixture_id: str, index: int = 0):
    return json.loads(fixture(fixture_id).answers[index].body)


def _refsnp(document: dict, assembly: str, **call):
    body = json.dumps(document).encode()
    source = RetrievedSource.create(
        operation="fetch_refsnp",
        request_identity="https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
        + document["refsnp_id"],
        status_code=200,
        observed_at="2026-10-27T12:00:00Z",
        elapsed_time_ms=1,
        raw_response=body,
    )
    return normalize_refsnp(
        rsid=f"rs{document['refsnp_id']}",
        assembly_accession=assembly,
        source=source,
        raw_response=body,
        **call,
    )


class RefsnpCopyTests(unittest.TestCase):
    def test_every_placement_and_field_dbsnp_writes_is_copied_in_order(self) -> None:
        document = _recorded("variant-dbsnp-non-ptlp")
        recorded = document["primary_snapshot_data"]["placements_with_allele"]

        copied = refsnp_placements(document)

        self.assertEqual(
            [p.sequence for p in copied], [item["seq_id"] for item in recorded]
        )
        for placement, item in zip(copied, recorded, strict=True):
            with self.subTest(sequence=placement.sequence):
                annotation = item["placement_annot"]
                self.assertEqual(
                    (
                        placement.dbsnp.is_ptlp,
                        placement.dbsnp.seq_type,
                        placement.dbsnp.mol_type,
                        placement.dbsnp.is_mismatch,
                    ),
                    (
                        item["is_ptlp"],
                        annotation["seq_type"],
                        annotation["mol_type"],
                        annotation["is_mismatch"],
                    ),
                )
                self.assertEqual(
                    [
                        (a.assembly_accession, a.is_top_level, a.is_alt, a.is_patch)
                        for a in placement.dbsnp.assemblies
                    ],
                    [
                        (
                            t["assembly_accession"],
                            t["is_top_level"],
                            t["is_alt"],
                            t["is_patch"],
                        )
                        for t in annotation["seq_id_traits_by_assembly"]
                    ],
                )
                self.assertEqual(
                    [
                        (
                            a.seq_id,
                            a.position,
                            a.deleted_sequence,
                            a.inserted_sequence,
                            a.hgvs,
                        )
                        for a in placement.dbsnp.alleles
                    ],
                    [
                        (
                            a["allele"]["spdi"]["seq_id"],
                            a["allele"]["spdi"]["position"],
                            a["allele"]["spdi"]["deleted_sequence"],
                            a["allele"]["spdi"]["inserted_sequence"],
                            a["hgvs"],
                        )
                        for a in item["alleles"]
                    ],
                )
                self.assertFalse(placement.on_run_assembly)

    def test_an_answer_without_its_documented_shape_is_unreadable(self) -> None:
        document = _recorded("variant-dbsnp-non-ptlp")
        cases = {
            "no seq_id": lambda placement: placement.pop("seq_id"),
            "alleles not a list": lambda placement: placement.update(alleles={}),
            "an SPDI without its position": lambda placement: placement["alleles"][0][
                "allele"
            ]["spdi"].pop("position"),
        }
        for name, change in cases.items():
            with self.subTest(name):
                broken = copy.deepcopy(document)
                change(broken["primary_snapshot_data"]["placements_with_allele"][0])

                with self.assertRaises(PlacementsUnreadable):
                    refsnp_placements(broken)


class EvaCopyTests(unittest.TestCase):
    def test_every_clustered_record_and_field_eva_writes_is_copied(self) -> None:
        recorded = _recorded("variant-many-placements")

        copied = eva_clustered_placements(recorded)

        self.assertEqual(
            [
                (
                    p.sequence,
                    p.eva.accession,
                    p.eva.assembly_accession,
                    p.eva.taxonomy_accession,
                    p.eva.start,
                    p.eva.type,
                    p.eva.created_date,
                )
                for p in copied
            ],
            [
                (
                    item["data"]["contig"],
                    item["accession"],
                    item["data"]["assemblyAccession"],
                    item["data"]["taxonomyAccession"],
                    item["data"]["start"],
                    item["data"]["type"],
                    item["data"]["createdDate"],
                )
                for item in recorded
            ],
        )

    def test_a_submitted_variant_of_another_cluster_is_unreadable(self) -> None:
        submitted = _recorded("variant-many-placements", 1)

        with self.assertRaisesRegex(PlacementsUnreadable, "names another cluster"):
            eva_submitted_variants(submitted, 1)
        self.assertEqual(
            len(eva_submitted_variants(submitted, 109500000)), len(submitted)
        )


def _two_loci_on_grch38() -> dict:
    """Derived from the recorded rs429358: its GRCh38 placement copied onto a synthetic
    alt-locus sequence (NT_187636.1) of the same assembly, as dbSNP writes an alt locus
    (is_ptlp false, is_alt true)."""
    document = _recorded("variant-dbsnp-non-ptlp")
    placements = document["primary_snapshot_data"]["placements_with_allele"]
    alt = copy.deepcopy(placements[0])
    alt["seq_id"], alt["is_ptlp"] = "NT_187636.1", False
    alt["placement_annot"]["seq_id_traits_by_assembly"][0].update(
        is_alt=True, is_chromosome=False
    )
    for allele in alt["alleles"]:
        allele["allele"]["spdi"].update(seq_id="NT_187636.1", position=100)
    placements.append(alt)
    return document


class PlacementChoiceTests(unittest.TestCase):
    """More than one locus on the Run's assembly is the model's to choose (owner #468
    Q1 a): the lookup asks, and code checks only that the named placement is one of the
    record's on the Run's assembly."""

    def test_the_run_placement_is_marked_with_no_top_level_filter(self) -> None:
        document = _recorded("variant-dbsnp-non-ptlp")

        result = _refsnp(document, GRCH37)

        self.assertEqual(
            [p.sequence for p in result.placements if p.on_run_assembly],
            ["NC_000019.9"],
        )
        self.assertEqual(result.record_state, "primary_snapshot_data")
        self.assertEqual(result.status, "resolved")

    def test_two_loci_on_the_run_ask_for_a_placement(self) -> None:
        result = _refsnp(_two_loci_on_grch38(), GRCH38)

        self.assertEqual(result.status, "placement_unnamed")
        self.assertEqual(
            sorted({c.contig for c in result.candidates}),
            ["NC_000019.10", "NT_187636.1"],
        )
        # The fact states only what dbSNP placed; the asking is the Tool's model
        # view's next step, which names the `placement` argument.
        self.assertEqual(
            result.conflicts,
            (
                "NCBI RefSNP places rs429358 at 2 loci on the Run's assembly, on "
                "NC_000019.10, NT_187636.1.",
            ),
        )
        self.assertEqual(result.origin.code, "placement_unnamed")
        self.assertEqual(result.origin.loci, 2)
        view = tool_model_view(
            LOOKUP_VIEW,
            {
                "event_id": "lookup-1",
                "payload": {"result": result.model_dump(mode="json")},
            },
        )
        self.assertEqual(view["statement"], result.conflicts[0])
        self.assertIn("`name_placement`", view["next"])

    def test_a_named_placement_on_the_run_resolves_to_it(self) -> None:
        for named in ("NC_000019.10", "NT_187636.1"):
            with self.subTest(named=named):
                result = _refsnp(_two_loci_on_grch38(), GRCH38, placement=named)

                self.assertEqual(result.status, "resolved")
                self.assertEqual({c.contig for c in result.candidates}, {named})
                self.assertEqual(result.named_placement, named)

    def test_a_named_placement_off_the_run_is_refused(self) -> None:
        result = _refsnp(_two_loci_on_grch38(), GRCH38, placement="NC_000019.9")

        self.assertEqual(result.status, "placement_unnamed")
        self.assertIn(
            "NC_000019.9 is not one of rs429358's placements on the Run's assembly",
            result.conflicts[0],
        )
        self.assertEqual(result.named_placement, "NC_000019.9")

    def test_a_named_placement_is_checked_even_when_the_exact_target_resolves(self):
        """Spec r1 SP3: a position lookup binds an exact target, and a named placement
        off the Run's assembly is still refused, never recorded beside a resolution."""
        from bio_neb.investigation.investigation_request import RequestedVariantIdentity

        exact = RequestedVariantIdentity("NC_000019.10", 44908684, "T", "C")

        refused = _refsnp(
            _two_loci_on_grch38(),
            GRCH38,
            requested_variant=exact,
            placement="NC_000019.9",
        )
        named = _refsnp(
            _two_loci_on_grch38(),
            GRCH38,
            requested_variant=exact,
            placement="NC_000019.10",
        )

        self.assertEqual(refused.status, "placement_unnamed")
        self.assertIn("not one of", refused.conflicts[0])
        self.assertEqual(
            (named.status, [c.contig for c in named.candidates]),
            ("resolved", ["NC_000019.10"]),
        )


if __name__ == "__main__":
    unittest.main()
