"""The quote-membership check for a site the model typed (#472 Q1 a, ADR 0049).

The model types a site from the researcher's words; code checks only that the quote holds
the contig, the position and the bases it typed, and never reads a site out of the quote.
These cases are the spellings the retired position reader (`paper_text.written_position`)
was tested on, now held as membership.
"""

import unittest

from bio_neb.identifier_rule import site_in_quote


class SiteInQuoteTests(unittest.TestCase):
    def test_each_spelling_holds_the_site_typed_from_it(self) -> None:
        for quote, site in (
            ("V:2129317 C>T", ("V", 2129317, "C", "T")),
            ("chr7:117,559,590 C>T on hg19", ("chr7", 117559590, "C", "T")),
            ("NC_003283.11:g.2129317C>T", ("NC_003283.11", 2129317, "C", "T")),
            # The contig as written: the model names its sequence.
            (
                "Also check chromosome V position 2,129,317.",
                ("chromosome V", 2129317, None, None),
            ),
            ("Also check chromosome V position 2,129,317.", ("V", 2129317, None, None)),
            (
                "Does it carry chrX:765242 C>A (1-based, on GCF_000002985.6)?",
                ("chrX", 765242, "C", "A"),
            ),
            ("rs429358 at II:1234 A>G on WBcel235 or ce10", ("II", 1234, "A", "G")),
            (
                "NC_049231.1:g.69888722_69888723del",
                ("NC_049231.1", 69888722, None, None),
            ),
            # Two sites in one quote: which one is meant is the model's choice.
            ("II:1234 A>G and X:55 C>T", ("X", 55, "C", "T")),
        ):
            with self.subTest(quote=quote, site=site):
                self.assertTrue(site_in_quote(quote, *site))

    def test_a_site_the_quote_does_not_write_is_not_held(self) -> None:
        """ADR 0049: a model-named value is admitted only by membership in the words it
        quotes, so a part the quote does not write refuses the site."""
        for quote, site in (
            # A contig inside a longer name is not written on whole words.
            ("chr19:44908684 T>C", ("chr1", 44908684, "T", "C")),
            # A position the quote does not write, and one only part of a number.
            ("V:2129317 C>T", ("V", 2129318, "C", "T")),
            ("V:2129317 C>T", ("V", 2129, "C", "T")),
            # A base inside another word (the G of hg38) is no written base.
            ("chr19:45411941 T>C on hg38", ("chr19", 45411941, "T", "G")),
            ("V:2129317 C>T", ("V", 2129317, "CT", "T")),
        ):
            with self.subTest(quote=quote, site=site):
                self.assertFalse(site_in_quote(quote, *site))


if __name__ == "__main__":
    unittest.main()
