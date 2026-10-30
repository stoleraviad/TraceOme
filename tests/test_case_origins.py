"""Every cross-request case of a collection view by its typed origin (#475 D): its
code, the typed fields it copies, its audit sentence and its plain words. The audit
sentence is what the CLI shows; no reader parses it.
"""

import unittest

from pydantic import TypeAdapter, ValidationError

from bio_neb.investigation import case_origins as cases
from bio_neb.investigation.qualification_origins import (
    RequestIssueRejected,
    RequestNotChecked,
    RequestRejected,
    RequestUnresolved,
)

# (origin, audit sentence, plain words), one per code and one per typed branch its
# audit or words read.
PINS = (
    (
        cases.MergedIdentity(requested="rs1", verified="rs2"),
        "dbSNP merged rs1 into rs2.",
        "dbSNP merged rs1 into rs2, so the request reads as rs2",
    ),
    (
        cases.SharedIdentity(requests=("rs1", "rs3"), rsid="rs2"),
        "The requests for rs1, rs3 all resolve to rs2.",
        "rs1, rs3 all name the same variant, rs2",
    ),
    (
        cases.ContradictorySources(rsid="rs2"),
        "The retained records of rs2 disagree on its locus or alleles.",
        "the sources' records of rs2 disagree on its position or alleles",
    ),
    (
        cases.ContradictorySources(contig="NC_1", position=10),
        "The records at NC_1:10 state different reference alleles.",
        "the records at NC_1:10 disagree on the reference allele",
    ),
    (
        cases.SharedLocus(
            identities=("NC_1:10 T>G", "rs1"), contig="NC_1", position=10
        ),
        "NC_1:10 T>G, rs1 share the locus NC_1:10.",
        "NC_1:10 T>G, rs1 are at the same position, NC_1:10",
    ),
    (
        cases.MultiAlt(request="rs1", alts=("C", "G")),
        "rs1 lists 2 alternate alleles (C, G), each its own Work item.",
        "rs1 lists 2 alternate alleles (C, G), each proposed as its own position",
    ),
    (
        cases.NoWorkItem(
            request="rs1",
            reason=RequestRejected(by="researcher:ana", reason="Not this one"),
        ),
        "rs1 has no Work item: rejected by researcher:ana: Not this one.",
        "no position was searched for rs1: rejected by researcher:ana, who wrote: "
        '"Not this one"',
    ),
    (
        cases.NoWorkItem(
            request="rs1",
            reason=RequestIssueRejected(
                by="researcher:ana", source="refsnp", verdict="not_found"
            ),
        ),
        "rs1 has no Work item: rejected by researcher:ana: refsnp: not_found finding.",
        "no position was searched for rs1: the variant lookup could not place it on "
        "the Run's reference, and researcher:ana rejected it",
    ),
    (
        cases.NoWorkItem(
            request="NC_1:10",
            reason=RequestNotChecked(
                reason_code="position_without_alleles", reason="No alleles stated."
            ),
        ),
        "NC_1:10 has no Work item: deferred: No alleles stated.",
        "no position was searched for NC_1:10: deferred out of the Run: it was "
        "stated without its alleles or an rsID, so no source was asked about it and "
        "the VCF was not searched; this does not say whether the sample carries a "
        "variant there",
    ),
    (
        cases.NoWorkItem(
            request="chrX:5",
            reason=RequestNotChecked(reason_code="not_placed", reason="No contig."),
        ),
        "chrX:5 has no Work item: not checked: No contig.",
        "no position was searched for chrX:5: no step could place it on the Run's "
        "reference as stated, so it was not checked",
    ),
    (
        cases.NoWorkItem(
            request="rs1",
            reason=RequestUnresolved(
                source="refsnp", verdict="source_answered", decision="approved"
            ),
        ),
        "rs1 has no Work item: refsnp: source_answered (approved, checked not "
        "verified).",
        "no position was searched for rs1: the variant lookup could not place it on "
        "the Run's reference; you accepted that finding as checked but not verified",
    ),
    (
        cases.OverlappingRecord(request="rs1"),
        "A record that overlaps rs1's locus sits beside its target.",
        "a variant call that overlaps rs1's position sits beside it",
    ),
    (
        cases.MixedAltOutcomes(request="rs1"),
        "rs1's alternate alleles have different lookup outcomes.",
        "rs1's alternate alleles had different results in the VCF",
    ),
    (
        cases.TrimmedMatch(request="rs1"),
        "rs1 matched a multi-base record only once its alleles were trimmed.",
        "rs1 matched a longer variant call only once its alleles were trimmed",
    ),
)


class CaseOriginPinTests(unittest.TestCase):
    def test_every_case_has_its_audit_sentence_and_plain_words(self):
        for origin, audit, words in PINS:
            with self.subTest(origin=origin):
                self.assertEqual((origin.audit(), origin.words()), (audit, words))

    def test_the_pin_covers_every_case_code(self):
        from typing import get_args

        from bio_neb.investigation.collection_view import CollectionCaseCode

        self.assertEqual(
            sorted({origin.code for origin, *_ in PINS}),
            sorted(get_args(CollectionCaseCode)),
        )

    def test_an_origin_round_trips_by_its_code(self):
        adapter = TypeAdapter(cases.CaseOrigin)
        for origin, *_ in PINS:
            with self.subTest(origin=origin):
                self.assertEqual(
                    adapter.validate_json(origin.model_dump_json()), origin
                )

    def test_contradictory_sources_name_one_record_or_one_locus(self):
        for fields in (
            {},
            {"rsid": "rs1", "contig": "NC_1", "position": 1},
            {"contig": "NC_1"},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(
                    ValidationError, "a contradiction names one rsID or one locus"
                ),
            ):
                cases.ContradictorySources(**fields)


if __name__ == "__main__":
    unittest.main()
