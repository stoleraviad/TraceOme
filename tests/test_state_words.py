"""Every recorded state's words from one owner (#475 C): the readable report's results
cell, the CLI's words and the explainer's claim word read one definition per state,
in place of the readable report's, the CLI's and the explainer's own tables.
"""

import unittest
from typing import get_args

from bio_neb.investigation import state_words
from bio_neb.reporting.explanation_members import ClaimWord

# (key, results cell, searched, carries the target, claim word, CLI words)
ITEM_PINS = (
    (
        "exact_allele_observed",
        "Found: the sample carries the target allele",
        True,
        True,
        "carries_target_allele",
        None,
    ),
    (
        "target_allele_not_carried",
        "Called here: the target allele is not carried",
        True,
        False,
        "target_allele_not_carried",
        None,
    ),
    (
        "target_genotype_unresolved",
        "Called here: the genotype is unresolved",
        True,
        False,
        "genotype_unresolved",
        None,
    ),
    (
        "locus_present_identity_mismatch",
        "A different variant was called here",
        True,
        False,
        "different_variant_at_locus",
        None,
    ),
    (
        "locus_overlapped",
        "A nearby variant call overlaps this position",
        True,
        False,
        "locus_overlapped",
        None,
    ),
    (
        "no_record",
        "Searched the VCF: no variant call here; genotype not established",
        True,
        False,
        "searched_no_record",
        None,
    ),
    (
        "ambiguous_records",
        "Several variant calls here; not decided",
        True,
        False,
        "ambiguous_records",
        None,
    ),
    ("refused", "The VCF search was refused", True, False, "lookup_refused", None),
    (
        "region_observed",
        "Searched the VCF across this region",
        True,
        False,
        "state_not_recorded",
        None,
    ),
    ("failed", "The VCF search failed", False, False, "lookup_failed", "failed"),
    (
        "approved_not_observed",
        "Approved; the Run ended before the VCF was searched",
        False,
        False,
        "approved_not_looked_up",
        "approved, not observed",
    ),
    (
        "not_approved",
        "Not approved for a VCF search",
        False,
        False,
        "not_approved",
        "proposed, not approved",
    ),
    (
        "unselected_candidate",
        "Another allele was chosen instead",
        False,
        False,
        "unselected_candidate",
        "a candidate the approval did not select",
    ),
    ("removed", "Removed by the researcher", False, False, "removed", "removed"),
    ("deferred", "Deferred by the researcher", False, False, "deferred", "deferred"),
)
# (state, reason code, results cell, claim word, CLI words)
REQUEST_PINS = (
    (
        "unresolved",
        None,
        "not placed on this reference; the VCF was not searched",
        "unresolved",
        "unresolved",
    ),
    (
        "rejected",
        None,
        "rejected; the VCF was not searched",
        "rejected",
        "rejected",
    ),
    (
        "not_checked",
        "not_placed",
        "could not be checked as stated; the VCF was not searched",
        "not_checked",
        "not checked",
    ),
    (
        "not_checked",
        "position_without_alleles",
        "deferred out of the Run: it was stated without its alleles or an rsID, so "
        "no source was asked about it and the VCF was not searched; this does not "
        "say whether the sample carries a variant there",
        "not_checked",
        "deferred out of the Run",
    ),
    (
        "removed",
        None,
        "removed by the researcher; the VCF was not searched",
        "removed",
        "removed",
    ),
    (
        "deferred",
        None,
        "deferred by the researcher; the VCF was not searched",
        "deferred",
        "deferred",
    ),
    (
        "not_proposed",
        None,
        "checked, but the Run ended before a search was proposed",
        "not_proposed",
        "checked, not proposed",
    ),
    (
        "never_run",
        None,
        "its check did not finish before the Run ended",
        "never_run",
        "never run",
    ),
    (
        "candidates",
        None,
        "no candidate was searched",
        "candidates",
        "candidates",
    ),
)


class StateWordsPinTests(unittest.TestCase):
    def test_every_item_key_has_its_cell_claim_and_cli_words(self):
        for key, cell, searched, carries, claim, cli in ITEM_PINS:
            with self.subTest(key=key):
                words = state_words.item_words(key)
                self.assertEqual(
                    (words.cell, words.searched, words.carries, words.claim, words.cli),
                    (cell, searched, carries, claim, cli),
                )

    def test_every_request_state_has_its_cell_claim_and_cli_words(self):
        for state, reason_code, cell, claim, cli in REQUEST_PINS:
            with self.subTest(state=state, reason_code=reason_code):
                words = state_words.request_words(state, reason_code)
                self.assertEqual(
                    (words.cell, words.claim, words.cli), (cell, claim, cli)
                )

    def test_a_request_s_row_words_are_its_cell_unless_it_names_positions(self):
        for state, reason, cell, *_ in REQUEST_PINS:
            with self.subTest(state=state, reason=reason):
                self.assertEqual(
                    state_words.request_words(state, reason).row,
                    "each position proposed for it is its own target"
                    if state == "candidates"
                    else cell,
                )

    def test_the_pins_cover_every_defined_state(self):
        self.assertEqual({key for key, *_ in ITEM_PINS}, set(state_words.ITEM_WORDS))
        self.assertEqual(
            {state for state, *_ in REQUEST_PINS},
            set(get_args(state_words.RequestState)),
        )

    def test_an_unknown_state_reads_as_not_recorded(self):
        for words in (
            state_words.item_words("no_such_state"),
            state_words.request_words(None),
        ):
            with self.subTest(words=words):
                self.assertEqual(words.claim, "state_not_recorded")
        self.assertEqual(state_words.item_words("x").cell, "State not recorded")
        self.assertEqual(state_words.request_words(None).cell, "state not recorded")

    def test_every_claim_word_is_one_the_explanation_records(self):
        claims = {words.claim for words in state_words.ITEM_WORDS.values()} | {
            state_words.request_words(state, reason).claim
            for state, reason, *_ in REQUEST_PINS
        }
        self.assertEqual(claims | {"state_not_recorded"}, set(get_args(ClaimWord)))


if __name__ == "__main__":
    unittest.main()
