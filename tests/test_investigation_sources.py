"""RefSNP normalization (merges, one candidate per listed ALT of any class) and the
target shapes it feeds."""

from __future__ import annotations

import copy
import hashlib
import unittest

from pydantic import ValidationError

from bio_neb.content import canonical_json
from bio_neb.investigation.investigation import (
    ResolvedVariantCandidate,
    RetrievedSource,
    VariantTarget,
    WorkItem,
)
from bio_neb.investigation.investigation_observation import observation_target
from bio_neb.investigation.investigation_sources import normalize_refsnp
from bio_neb.investigation.lookup_origins import Merged
from tests.refsnp_retained_bodies import (
    RS2228570,
    RS4149056,
    RS4149056_DOCUMENT,
    RS60037639,
    RS754204844,
    RS756427959,
    RS878871521,
    refsnp_body,
)

GRCH38 = "GCF_000001405.40"


def _normalized(query: str, body: bytes, *, requested_rsid: str | None = None):
    source = RetrievedSource.create(
        operation="fetch_refsnp",
        request_identity="https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
        + query.removeprefix("rs"),
        status_code=200,
        observed_at="2026-10-15T10:50:00Z",
        elapsed_time_ms=1,
        raw_response=body,
    )
    return normalize_refsnp(
        rsid=query,
        requested_rsid=requested_rsid,
        assembly_accession=GRCH38,
        source=source,
        raw_response=body,
    )


class RefSnpMergeTests(unittest.TestCase):
    def test_a_merged_rsid_names_its_current_rsid_without_a_target(self) -> None:
        result = _normalized("rs60037639", RS60037639)

        self.assertEqual(result.status, "merged")
        self.assertEqual(result.merged_into, ("rs4149056",))
        self.assertEqual(result.verified_rsid, "rs60037639")
        self.assertEqual(result.candidates, ())
        self.assertEqual(
            result.conflicts,
            ("NCBI dbSNP merged rs60037639 into rs4149056.",),
        )
        # The ending is the typed origin's audit sentence: what dbSNP observed,
        # no instruction; the next step is the Tool's model view's.
        self.assertEqual(result.origin, Merged(rsid="rs60037639", into=("rs4149056",)))
        self.assertEqual(result.assessment.verdict, "merged")
        self.assertEqual(result.assessment.rationale, result.conflicts[0])

    def test_a_current_record_lists_the_rsids_dbsnp_merged_into_it(self) -> None:
        result = _normalized("rs4149056", RS4149056, requested_rsid="rs60037639")

        self.assertEqual(result.requested_rsid, "rs60037639")
        self.assertEqual(result.verified_rsid, "rs4149056")
        self.assertEqual(result.merged_rsids, ("rs52816141", "rs60037639"))
        # dbSNP flattens merges: rs2228570 lists rs10735810 and the two rsIDs
        # that had been merged into rs10735810 before it.
        self.assertEqual(
            _normalized("rs2228570", RS2228570).merged_rsids,
            (
                "rs8179174",
                "rs10735810",
                "rs17881966",
                "rs52811041",
                "rs56641119",
                "rs57067622",
                "rs117559231",
            ),
        )
        self.assertEqual(_normalized("rs60037639", RS60037639).merged_rsids, ())


class RefSnpAlleleTests(unittest.TestCase):
    def test_each_source_backed_snv_alt_is_its_own_candidate(self) -> None:
        cases = {
            # The retained bodies list two and three SNV ALTs on GRCh38.
            "rs4149056": (RS4149056, "NC_000012.12", 21178615, "T", ("A", "C")),
            "rs2228570": (RS2228570, "NC_000012.12", 47879112, "A", ("C", "G", "T")),
        }
        for rsid, (body, contig, position, ref, alts) in cases.items():
            with self.subTest(rsid=rsid):
                result = _normalized(rsid, body)

                self.assertEqual(result.status, "resolved")
                self.assertEqual(result.conflicts, ())
                self.assertEqual(
                    tuple(
                        (
                            item.assembly_accession,
                            item.assembly_name,
                            item.contig,
                            item.one_based_position,
                            item.ref,
                            item.alt,
                            item.aliases,
                        )
                        for item in result.candidates
                    ),
                    tuple(
                        (GRCH38, "GRCh38.p14", contig, position, ref, alt, (rsid,))
                        for alt in alts
                    ),
                )
                self.assertEqual(result.assessment.verdict, "accepted")

    def test_indel_alleles_are_candidates_as_dbsnp_states_them(self) -> None:
        # The retained bodies. Each listed allele is a candidate in dbSNP's
        # reference-expanded SPDI form at its one locus; the Work item, not the
        # source, holds the VCF-comparable form.
        cases = {
            "rs756427959": (
                RS756427959,
                "NC_000001.11",
                14398,
                "TGTTG",
                ("TG",),
            ),
            "rs754204844": (RS754204844, "NC_000001.11", 175160810, "C", ("",)),
            "rs878871521": (
                RS878871521,
                "NC_012920.1",
                303,
                "C" * 7,
                tuple("C" * count for count in (8, 9, 10, 18, 21)),
            ),
        }
        for rsid, (body, contig, position, ref, alts) in cases.items():
            with self.subTest(rsid=rsid):
                result = _normalized(rsid, body)

                self.assertEqual(result.status, "resolved")
                self.assertEqual(result.conflicts, ())
                self.assertEqual(
                    tuple(
                        (item.contig, item.one_based_position, item.ref, item.alt)
                        for item in result.candidates
                    ),
                    tuple((contig, position, ref, alt) for alt in alts),
                )
                self.assertNotIn("unsupported_alleles", result.model_dump())

    def test_snv_indel_and_mnv_alleles_of_one_locus_are_all_candidates(self) -> None:
        # Derived from rs4149056 (synthetic): its T>A allele becomes the
        # insertion T>TT, then the MNV TC>GA beside a TC>TG SNV-in-padding.
        mixed = copy.deepcopy(RS4149056_DOCUMENT)
        placement = mixed["primary_snapshot_data"]["placements_with_allele"][0]
        placement["alleles"][1]["allele"]["spdi"]["inserted_sequence"] = "TT"
        mnv = copy.deepcopy(RS4149056_DOCUMENT)
        placement = mnv["primary_snapshot_data"]["placements_with_allele"][0]
        for allele, inserted in zip(
            placement["alleles"], ("TC", "GA", "TG"), strict=False
        ):
            allele["allele"]["spdi"]["deleted_sequence"] = "TC"
            allele["allele"]["spdi"]["inserted_sequence"] = inserted

        for document, expected in (
            (mixed, (("T", "TT"), ("T", "C"))),
            (mnv, (("TC", "GA"), ("TC", "TG"))),
        ):
            with self.subTest(expected=expected):
                result = _normalized("rs4149056", refsnp_body(document))

                self.assertEqual(result.status, "resolved")
                self.assertEqual(
                    tuple(sorted((item.ref, item.alt) for item in result.candidates)),
                    tuple(sorted(expected)),
                )


class VariantTargetShapeTests(unittest.TestCase):
    def test_targets_hold_no_single_base_or_rsid_only_cap(self) -> None:
        # Rs756427959 is the reference-expanded SPDI TGTTG>TG.
        locus = {
            "assembly_accession": GRCH38,
            "assembly_name": "GRCh38.p14",
            "contig": "NC_000001.11",
            "one_based_position": 14399,
            "ref": "TGTTG",
            "alt": "TG",
        }
        candidate = ResolvedVariantCandidate(**locus, aliases=())
        material = {
            **locus,
            "assessment_ids": ("evidence-assessment:a", "evidence-assessment:b"),
        }
        target = VariantTarget(target_id=_target_id(material), **material)

        self.assertEqual(candidate.aliases, ())
        self.assertNotIn("aliases", candidate.model_dump(mode="json"))
        self.assertEqual((target.ref, target.alt, target.aliases), ("TGTTG", "TG", ()))
        for ref, alt in (("T", "T"), ("", ""), ("t", "C")):
            with self.subTest(ref=ref, alt=alt), self.assertRaises(ValidationError):
                ResolvedVariantCandidate(**{**locus, "ref": ref, "alt": alt})

    def test_a_target_without_an_rsid_alias_reaches_the_vcf_lookup_input(
        self,
    ) -> None:
        material = {
            "assembly_accession": GRCH38,
            "assembly_name": "GRCh38.p14",
            "contig": "NC_000012.12",
            "one_based_position": 21178615,
            "ref": "T",
            "alt": "C",
            "assessment_ids": ("evidence-assessment:a", "evidence-assessment:b"),
        }
        work_item = WorkItem.create(
            sample_id="HG001",
            target=VariantTarget(target_id=_target_id(material), **material),
            source_ids=("retrieved-source:a", "retrieved-source:b"),
        )

        target = observation_target(work_item)

        self.assertEqual(
            (target.contig, target.one_based_position, target.ref, target.alt),
            ("NC_000012.12", 21178615, "T", "C"),
        )
        self.assertIsNone(target.display_label)


def _target_id(material: dict) -> str:
    """The target digest, computed independently of VariantTarget."""
    digest = hashlib.sha256(canonical_json(material).encode()).hexdigest()
    return f"variant-target:{digest}"


if __name__ == "__main__":
    unittest.main()
