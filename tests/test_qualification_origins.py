"""Every report qualification's typed origin, its audit sentence and its plain words
(#474 acceptance: one pin maps every origin to its words; owner A1, 2026-10-28).

The audit sentences are the report basis's own text: those the Investigation's
evidence already stores keep their bytes (#474 Q1 a). The plain words are what the
readable report's body and the explainer read, so none carries audit wording: no
id, no state code, no "Work item", no "Finalization" (#460 LSP6, option c).
"""

import unittest

from bio_neb.investigation import qualification_origins as origins
from tests.qualification_origin_pins import AUDIT_WORDING, PINS


class QualificationOriginPinTests(unittest.TestCase):
    def test_every_origin_has_its_kind_audit_sentence_and_plain_words(self):
        for origin, subject, kind, audit, words in PINS:
            with self.subTest(origin=origin):
                self.assertEqual(
                    (origin.kind, origin.audit(subject), origin.words()),
                    (kind, audit, words),
                )

    def test_the_pin_covers_every_origin_code(self):
        self.assertEqual(
            sorted({origin.code for origin, *_ in PINS}),
            list(origins.origin_codes()),
        )

    def test_no_plain_words_carry_audit_wording(self):
        for origin, *_ in PINS:
            with self.subTest(origin=origin):
                self.assertIsNone(AUDIT_WORDING.search(origin.words()))

    def test_plain_words_lead_with_their_targets_or_stand_for_the_run(self):
        self.assertEqual(
            origins.plain_words(origins.CoverageNotMeasured(), ("rs1", "rs2")),
            "rs1, rs2: coverage was not measured here.",
        )
        self.assertEqual(
            origins.plain_words(
                origins.DiploidCallingJudgedOtherwise(judgment="unknown")
            ),
            "The variants were called as diploid, though the planning model judged "
            "this organism of unknown ploidy; you chose diploid calling.",
        )

    def test_an_origin_round_trips_by_its_code(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(origins.QualificationOrigin)
        for origin, *_ in PINS:
            with self.subTest(origin=origin):
                self.assertEqual(
                    adapter.validate_python(origin.model_dump(mode="json")), origin
                )


if __name__ == "__main__":
    unittest.main()
