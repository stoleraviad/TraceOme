"""What a source Tool shows of each answer its source sent (#458).

The answers are the source's own bytes: whole when they fit the return's share, cut
otherwise with every cut said, so the model reads what the source said and how much
it did not see.
"""

from __future__ import annotations

import hashlib
import json
import unittest

from bio_neb.agentic_journey.source_answers import (
    SOURCE_ANSWER_BYTES,
    AnsweredGet,
    shown_answers,
)
from bio_neb.content import canonical_json

EVA_410 = (
    b'[{"accession":160719159,"version":1,"data":{"assemblyAccession":'
    b'"GCA_000298735.2","taxonomyAccession":9940,"contig":"AMGL02039708.1",'
    b'"start":1933,"type":"SNV","validated":false,"mapWeight":null,'
    b'"createdDate":"2011-07-22T16:49:00"}}]'
)
"""EVA's answer for rs160719159 (#457 C. elegans a1, `eva-410-rs160719159.body`)."""


def _bytes(body) -> int:
    """A shown JSON body's size as the bound counts it: its canonical UTF-8 bytes."""
    return len(canonical_json(body).encode())


def _get(body: bytes, *, status: int = 200, operation: str = "eva_clustered_variant"):
    return AnsweredGet(
        operation=operation,
        request_identity="https://www.ebi.ac.uk/eva/webservices/identifiers/v1/x",
        status_code=status,
        body=body,
    )


class WholeAnswerTests(unittest.TestCase):
    def test_a_small_answer_is_shown_whole_with_its_status_and_identity(self) -> None:
        (answer,) = shown_answers((_get(EVA_410, status=410),))

        self.assertEqual(answer.status_code, 410)
        self.assertEqual(answer.operation, "eva_clustered_variant")
        self.assertEqual(answer.shown, "whole")
        self.assertEqual(answer.media, "json")
        self.assertEqual(answer.body, json.loads(EVA_410))
        self.assertEqual(answer.byte_count, 232)
        self.assertEqual(answer.sha256, hashlib.sha256(EVA_410).hexdigest())

    def test_an_answer_that_is_not_json_is_shown_as_its_text(self) -> None:
        (answer,) = shown_answers((_get(b"Not Found\n", status=404),))

        self.assertEqual((answer.media, answer.body), ("text", "Not Found\n"))
        self.assertEqual(answer.shown, "whole")

    def test_an_empty_answer_is_shown_as_empty_text(self) -> None:
        (answer,) = shown_answers((_get(b"", status=410),))

        self.assertEqual(
            (answer.media, answer.body, answer.shown), ("text", "", "whole")
        )

    def test_json_a_journey_cannot_hold_is_shown_as_text(self) -> None:
        # Spec r1 SP2: an overflowing number parses as infinity without a constant.
        for body in (b'{"frequency": NaN}', b'{"message":"invalid key","score":1e999}'):
            with self.subTest(body=body):
                (answer,) = shown_answers((_get(body, status=400),))

                self.assertEqual((answer.media, answer.body), ("text", body.decode()))

    def test_answers_keep_their_call_order(self) -> None:
        shown = shown_answers(
            (
                _get(b"[]", operation="eva_clustered_variant"),
                _get(b"{}", operation="eva_contig_alias"),
            )
        )

        self.assertEqual(
            [answer.operation for answer in shown],
            ["eva_clustered_variant", "eva_contig_alias"],
        )


class ExcerptTests(unittest.TestCase):
    def test_a_long_list_keeps_its_first_items_and_says_how_many_were_cut(self) -> None:
        body = json.dumps(
            {"placements": [{"seq_id": f"NC_{i:06d}.1"} for i in range(2000)]}
        )

        (answer,) = shown_answers((_get(body.encode()),))

        self.assertEqual(answer.shown, "excerpt")
        placements = answer.body["placements"]
        self.assertEqual(placements[0], {"seq_id": "NC_000000.1"})
        self.assertRegex(placements[-1], r"^… [0-9]+ more items$")
        kept = len(placements) - 1
        self.assertEqual(placements[-1], f"… {2000 - kept} more items")
        self.assertLessEqual(len(canonical_json(answer.body)), SOURCE_ANSWER_BYTES)

    def test_a_long_string_keeps_its_head_and_says_how_much_was_cut(self) -> None:
        body = json.dumps({"refsnp_id": "429358", "note": "A" * 40000})

        (answer,) = shown_answers((_get(body.encode()),))

        self.assertEqual(answer.body["refsnp_id"], "429358")
        note = answer.body["note"]
        head, _, tail = note.partition("… ")
        self.assertTrue(set(head) == {"A"})
        self.assertEqual(tail, f"{40000 - len(head)} more characters")

    def test_every_key_the_source_sent_stays_readable_in_an_excerpt(self) -> None:
        body = json.dumps(
            {
                "refsnp_id": "160719159",
                "merged_snapshot_data": {"merged_into": ["7412"]},
                "present_obs_movements": [{"allele": "x" * 50}] * 3000,
            }
        )

        (answer,) = shown_answers((_get(body.encode()),))

        self.assertEqual(
            set(answer.body),
            {"refsnp_id", "merged_snapshot_data", "present_obs_movements"},
        )
        self.assertEqual(answer.body["merged_snapshot_data"], {"merged_into": ["7412"]})

    def test_long_text_keeps_its_head_and_says_how_much_was_cut(self) -> None:
        body = "<html>" + "b" * 40000

        (answer,) = shown_answers((_get(body.encode(), status=403),))

        self.assertEqual((answer.media, answer.shown), ("text", "excerpt"))
        self.assertTrue(answer.body.startswith("<html>bbb"))
        self.assertRegex(answer.body, r"… [0-9]+ more characters$")
        self.assertLessEqual(len(answer.body.encode()), SOURCE_ANSWER_BYTES)

    def test_small_answers_stay_whole_beside_a_large_one(self) -> None:
        large = json.dumps([{"n": i, "pad": "p" * 40} for i in range(5000)]).encode()

        shown = shown_answers((_get(large), _get(EVA_410, status=410), _get(b"[]")))

        self.assertEqual([a.shown for a in shown], ["excerpt", "whole", "whole"])
        total = sum(_bytes(a.body) for a in shown)
        self.assertLessEqual(total, SOURCE_ANSWER_BYTES)

    def test_the_bound_counts_bytes_for_multibyte_text(self) -> None:
        # Spec r1 SP1: 10,000 three-byte characters are 30,014 bytes of JSON.
        json_body = json.dumps({"message": "漢" * 10000}, ensure_ascii=False).encode()
        text_body = ("漢" * 10000).encode()

        json_answer, text_answer = shown_answers(
            (_get(json_body, status=400), _get(text_body, status=400))
        )

        self.assertEqual((json_answer.shown, text_answer.shown), ("excerpt",) * 2)
        self.assertEqual(json_answer.byte_count, len(json_body))
        self.assertLessEqual(
            _bytes(json_answer.body) + len(text_answer.body.encode()),
            SOURCE_ANSWER_BYTES,
        )
        head, _, tail = text_answer.body.partition("… ")
        self.assertEqual(set(head), {"漢"})
        self.assertEqual(tail, f"{10000 - len(head)} more characters")

    def test_the_same_bytes_are_always_shown_the_same_way(self) -> None:
        large = json.dumps({"a": list(range(20000)), "b": "c" * 30000}).encode()

        self.assertEqual(shown_answers((_get(large),)), shown_answers((_get(large),)))


if __name__ == "__main__":
    unittest.main()
