"""The explanation draft the model writes and its one check (#448 D1, #460).

A member writes its request's short ``why``, its ``why_references`` and
``limitations``; the combination writes the ``conclusion``, its references,
``would_change``, ``limitations`` and ``not_done``. Code fills every per-row
fact, so neither writes a claim. One check reads the raw arguments and returns
the admitted draft or its faults covering shape and claims together, which the
failure package words as one correction (ADR 0043).

#387's 23 refused explainer outputs are the fixture
(``tests/fixtures/issue387/explainer-refused-drafts.json``, copied with each
Attempt's recorded binding): every one was refused for its shape only.
"""

import json
import unittest
from pathlib import Path

from bio_neb.reporting.explanation_rows import ExplanationRow, ReportQualification

_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures/issue387/explainer-refused-drafts.json"
)


def _drafts():
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return [(run["run"], draft) for run in document["runs"] for draft in run["drafts"]]


def _target(row):
    """A recorded row as the target the model reads now (#474 Q3 a, a derived
    input): its number, the item kind named a position, and no state note."""
    return ExplanationRow(
        **{
            key: value
            for key, value in row.items()
            if key not in {"row", "kind", "state_note"}
        },
        number=row["row"],
        kind="position" if row["kind"] == "item" else row["kind"],
        words="recorded words",
    )


def _origin(text):
    """The typed origin a recorded qualification's sentence was written from (#474,
    a derived input): each audit sentence is its origin's own."""
    from bio_neb.investigation import qualification_origins as origins

    if text.startswith("Coverage was not measured for Work item "):
        return origins.CoverageNotMeasured()
    if text.startswith("Request "):
        return origins.RequestUnresolved(
            source="refsnp",
            verdict=text.split("refsnp: ")[1].split()[0],
            decision="approved",
        )
    known = (
        origins.ObservationNotInferred(),
        origins.ObservationProvenance(run_provenance="unestablished"),
        origins.TargetAltNotCarried(),
        *(
            origins.ObservationScope(outcome=outcome)
            for outcome in (
                "exact_allele_observed",
                "no_record",
                "target_allele_not_carried",
            )
        ),
    )
    return {origin.audit(): origin for origin in known}[text]


def _binding(draft):
    from bio_neb.investigation.qualification_origins import plain_words
    from bio_neb.reporting.explanation_draft import DraftBinding

    recorded = draft["binding"]
    return DraftBinding(
        stage=draft["stage"],
        targets=tuple(_target(row) for row in recorded["rows"]),
        qualifications=tuple(
            ReportQualification(
                qualification=item["qualification"],
                kind=item["kind"],
                origin=_origin(item["text"]).code,
                words=plain_words(_origin(item["text"])),
                targets=tuple(item.get("rows", ())),
            )
            for item in recorded["qualifications"]
        ),
        evidence_event_ids=tuple(recorded["evidence_event_ids"]),
    )


def _flat(stage, output):
    """The draft as the /3 flat shape holds it, derived from #387's recorded one:
    the answer's text becomes a member's ``why`` or the combination's
    ``conclusion`` (cut at the field's length, which #387 did not have), its
    references (sent as JSON text by some calls) when it sent any, the
    limitations, and a combination's accounting as ``not_done`` beside a stated
    ``would_change`` and a stated ``title`` (#491), and no listed genotype (#473:
    #387 listed none); the claims and request list code fills are dropped."""
    answer = output["answer"]
    references = (
        answer["references"] if isinstance(answer, dict) else output.get("references")
    )
    if isinstance(references, str):
        references = json.loads(references)
    text = answer["text"] if isinstance(answer, dict) else answer
    name = "why" if stage == "member" else "conclusion"
    flat = {name: text[: 500 if stage == "member" else 1500]}
    if references is not None:
        flat[f"{name}_references"] = references
    if stage == "combination":
        flat["would_change"] = "A call at these positions would change the answer."
    flat["limitations"] = output["limitations"]
    if "accounting" in output:
        flat["not_done"] = output["accounting"]
    flat["genotypes"] = []
    if stage == "combination":
        flat["title"] = "Whether the sample carries its targets"
    return flat


class RefusedDraftReplayTests(unittest.TestCase):
    def test_the_fixture_is_387s_23_shape_refusals(self):
        drafts = _drafts()
        self.assertEqual(
            {run: sum(1 for name, _ in drafts if name == run) for run, _ in drafts},
            {"human-a2": 20, "celegans-a2": 3},
        )
        self.assertEqual(
            sorted(
                {tuple(draft["recorded_errors"]) for _run, draft in drafts}, key=len
            ),
            [
                ("answer:model_type",),
                ("references:extra_forbidden", "answer:model_type"),
                (
                    "claims.0.references:extra_forbidden",
                    "claims.1.references:extra_forbidden",
                    "claims:too_short",
                ),
            ],
        )

    def test_each_shape_only_draft_is_admitted_in_its_flat_form(self):
        from bio_neb.reporting.explanation_draft import (
            CombinationDraft,
            MemberDraft,
            Refused,
            check,
        )

        def text_only(result):
            # Recorded prose naming ids and claim words, which the check now
            # refuses; no shape fault remains in the flat form.
            return tuple(
                code
                for code in result.codes
                if not code.startswith("explanation_internal_term:")
            )

        admitted = 0
        for run, draft in _drafts():
            flat = _flat(draft["stage"], draft["output"])
            with self.subTest(run=run, attempt=draft["attempt_id"]):
                result = check(_binding(draft), flat)
                if "why_references" not in flat and draft["stage"] == "member":
                    # Two members cited nothing for their answer: that fault is
                    # not the shape's, and the flat form names only it.
                    self.assertIsInstance(result, Refused)
                    self.assertEqual(text_only(result), ("why_references:missing",))
                    continue
                if isinstance(result, Refused):
                    self.assertEqual(text_only(result), ())
                    continue
                admitted += 1
                self.assertIsInstance(
                    result,
                    MemberDraft if draft["stage"] == "member" else CombinationDraft,
                )
                self.assertEqual(result.model_dump(mode="json"), flat)
        # Some recorded drafts named no id or state word; they still admit.
        self.assertGreater(admitted, 0)

    def test_each_draft_as_sent_gets_one_correction_naming_its_shape(self):
        from bio_neb.reporting.explanation_draft import Refused, check

        for run, draft in _drafts():
            with self.subTest(run=run, attempt=draft["attempt_id"]):
                refused = check(_binding(draft), draft["output"])
                self.assertIsInstance(refused, Refused)
                # Code fills the claims and the request list; the draft may
                # not carry them.
                written = "claims" if draft["stage"] == "member" else "requests"
                self.assertIn(f"{written}:extra_forbidden", refused.codes)
                sentences = " ".join(refused.faults.sentences)
                self.assertIn(f"{written} is no field of this output", sentences)
                # The correction names the output's fields by reading them.
                name = "why" if draft["stage"] == "member" else "conclusion"
                self.assertIn(f"{name}_references is missing", sentences)
                self.assertEqual(
                    refused.faults.fields[:2], (name, f"{name}_references")
                )


def _row(row, kind, member="member-b", **fields):
    return ExplanationRow(
        number=row,
        kind=kind,
        request_member_id=member,
        subject=f"subject-{row}",
        **{"words": "recorded words", **fields},
    )


class MemberCheckTests(unittest.TestCase):
    ROWS = (
        _row(3, "request", state="candidates"),
        _row(
            4,
            "position",
            state="observed",
            work_item_id="work-item:bb",
            outcome="exact_allele_observed",
            genotype="0/1",
        ),
    )
    QUALIFICATIONS = (
        ReportQualification(
            qualification=7,
            kind="gap",
            origin="coverage_not_measured",
            words="Coverage was not measured here.",
            targets=(4,),
        ),
    )

    def _binding(self):
        from bio_neb.reporting.explanation_draft import DraftBinding

        return DraftBinding(
            stage="member",
            targets=self.ROWS,
            qualifications=self.QUALIFICATIONS,
            evidence_event_ids=("e1", "e2"),
        )

    def _draft(self, **changes):
        draft = {
            "why": "Row 4 carries the target allele, genotype 0/1.",
            "why_references": ["e1"],
            "limitations": [
                {
                    "qualifications": [7],
                    "text": "Qualification 7 limits how far row 4's call is read.",
                    "references": ["e2"],
                }
            ],
            "genotypes": [],
        }
        draft.update(changes)
        return draft

    def test_a_flat_draft_over_its_binding_is_admitted(self):
        from bio_neb.reporting.explanation_draft import MemberDraft, check

        admitted = check(self._binding(), self._draft())
        self.assertIsInstance(admitted, MemberDraft)
        self.assertEqual(admitted.why_references, ("e1",))

    def test_a_why_is_short(self):
        # A why is one table cell, at most 500 characters.
        from bio_neb.reporting.explanation_draft import MemberDraft, check

        self.assertIsInstance(
            check(self._binding(), self._draft(why="y" * 500)), MemberDraft
        )
        self.assertEqual(
            check(self._binding(), self._draft(why="y" * 501)).codes,
            ("why:string_too_long",),
        )

    def test_a_shape_fault_and_a_claim_fault_get_one_correction(self):
        from bio_neb.reporting.explanation_draft import Refused, check

        draft = self._draft(limitations=[], references=["e1"])
        del draft["why_references"]
        refused = check(self._binding(), draft)
        self.assertIsInstance(refused, Refused)
        self.assertEqual(
            refused.codes,
            (
                "references:extra_forbidden",
                "why_references:missing",
                "explanation_qualifications_missing:7",
            ),
        )
        # One refusal: each fault's sentence, in that order.
        sentences = refused.faults.sentences
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].startswith("references is no field"))
        self.assertTrue(sentences[1].startswith("why_references is missing"))
        self.assertIn("Qualification 7", sentences[2])

    def test_a_wrong_json_type_in_a_limitation_is_refused_not_raised(self):
        # Standards ST1 / Spec SP1 r1: the text checks read what the arguments
        # carry, whatever JSON a field holds.
        from bio_neb.reporting.explanation_draft import Refused, check

        for value in (7, True, 1.5, "7", {"7": 7}):
            with self.subTest(value=value):
                draft = self._draft()
                draft["limitations"][0]["qualifications"] = value
                refused = check(self._binding(), draft)
                self.assertIsInstance(refused, Refused)
                self.assertEqual(
                    refused.codes[-1], "explanation_qualifications_missing:7"
                )
                self.assertTrue(
                    refused.codes[0].startswith("limitations.0.qualifications:")
                )
        draft = self._draft()
        draft["limitations"][0]["qualifications"] = [True, 7]
        self.assertEqual(
            check(self._binding(), draft).codes[-1],
            "limitations.0.qualifications.0:int_type",
        )

    def test_the_text_keeps_its_structural_checks(self):
        """Its references stay in scope; a listed genotype is its target's call
        (ADR 0049 membership, #473 owner Q5 a). Coverage words are the model's
        own: ADR 0049 retires the word ban (#474 owner Q2 a)."""
        from bio_neb.reporting.explanation_draft import Refused, check

        refused = check(
            self._binding(),
            self._draft(
                why="Row 4 reads 1/1 with good coverage.",
                why_references=["e9"],
                genotypes=[{"target": 4, "genotype": "1/1", "quote": "reads 1/1"}],
            ),
        )
        self.assertIsInstance(refused, Refused)
        self.assertEqual(
            [code.partition(":")[0] for code in refused.codes],
            [
                "explanation_reference_outside_scope",
                "explanation_genotype_unsupported",
            ],
        )

    def test_coverage_and_read_depth_words_are_the_model_s_own(self):
        """ADR 0049 and #474 owner Q2 a: no word list judges the text; the report
        states coverage from its typed origins."""
        from bio_neb.reporting.explanation_draft import MemberDraft, check

        admitted = check(
            self._binding(),
            self._draft(
                why="The VCF called the target allele, genotype 0/1; read depth and "
                "coverage were not measured here, as the report states."
            ),
        )
        self.assertIsInstance(admitted, MemberDraft)


class GenotypeListTests(unittest.TestCase):
    """The model lists each genotype its text writes with its target and quote;
    code checks membership only: the target is bound, the genotype is its call,
    the quote stands in the text and writes it (ADR 0049, #473 owner Q5 a)."""

    ROWS = MemberCheckTests.ROWS
    QUALIFICATIONS = MemberCheckTests.QUALIFICATIONS
    _binding = MemberCheckTests._binding
    _draft = MemberCheckTests._draft
    LISTED = {"target": 4, "genotype": "0/1", "quote": "genotype 0/1"}

    def _codes(self, **changes):
        from bio_neb.reporting.explanation_draft import Refused, check

        refused = check(self._binding(), self._draft(**changes))
        self.assertIsInstance(refused, Refused)
        return refused.codes

    def test_a_listed_genotype_is_kept_on_the_admitted_draft(self):
        from bio_neb.reporting.explanation_draft import MemberDraft, check

        admitted = check(self._binding(), self._draft(genotypes=[self.LISTED]))
        self.assertIsInstance(admitted, MemberDraft)
        self.assertEqual(
            [item.model_dump() for item in admitted.genotypes],
            [{"target": 4, "genotype": "0/1", "quote": "genotype 0/1"}],
        )

    def test_a_genotype_quoted_from_a_limitation_is_checked_too(self):
        from bio_neb.reporting.explanation_draft import MemberDraft, check

        draft = self._draft(
            why="Row 4 carries the target allele.",
            genotypes=[{"target": 4, "genotype": "0/1", "quote": "call 0/1"}],
        )
        draft["limitations"][0]["text"] = "Qualification 7 limits the call 0/1."
        self.assertIsInstance(check(self._binding(), draft), MemberDraft)

    def test_a_target_the_binding_does_not_list_is_refused(self):
        self.assertEqual(
            self._codes(
                genotypes=[{"target": 9, "genotype": "0/1", "quote": "genotype 0/1"}]
            ),
            ("explanation_genotype_target_unknown:genotypes.1",),
        )

    def test_a_genotype_that_is_not_its_targets_call_is_refused(self):
        # Target 3 is the request: it carries no call.
        self.assertEqual(
            self._codes(
                genotypes=[
                    {"target": 4, "genotype": "1/1", "quote": "genotype 0/1"},
                    {"target": 3, "genotype": "0/1", "quote": "genotype 0/1"},
                ]
            ),
            (
                "explanation_genotype_unsupported:genotypes.1",
                "explanation_genotype_not_in_quote:genotypes.1",
                "explanation_genotype_unsupported:genotypes.2",
            ),
        )

    def test_a_quote_that_is_not_the_texts_words_is_refused(self):
        self.assertEqual(
            self._codes(
                genotypes=[
                    {"target": 4, "genotype": "0/1", "quote": "genotype 0/1 call"}
                ]
            ),
            ("explanation_genotype_quote_outside_text:genotypes.1",),
        )

    def test_a_quote_that_does_not_write_the_genotype_is_refused(self):
        # "0/10" is no writing of 0/1: the genotype stands on whole words.
        self.assertEqual(
            self._codes(
                why="Row 4 carries the target allele, 0/10 reads.",
                genotypes=[{"target": 4, "genotype": "0/1", "quote": "0/10 reads"}],
            ),
            ("explanation_genotype_not_in_quote:genotypes.1",),
        )

    def test_unlisted_prose_is_not_read_for_a_genotype(self):
        """Code reads no genotype out of prose (ADR 0049; #473 owner Q5 a retires
        the GENOTYPE pattern): a ratio the model did not list is its own words."""
        from bio_neb.reporting.explanation_draft import MemberDraft, check

        admitted = check(
            self._binding(),
            self._draft(why="Row 4 carries the target allele; 1/2 of the reads do."),
        )
        self.assertIsInstance(admitted, MemberDraft)

    def test_the_list_is_part_of_the_draft_s_shape(self):
        draft = self._draft(genotypes=[self.LISTED])
        del draft["genotypes"]
        self.assertEqual(self._codes_of(draft), ("genotypes:missing",))

    def _codes_of(self, draft):
        from bio_neb.reporting.explanation_draft import Refused, check

        refused = check(self._binding(), draft)
        self.assertIsInstance(refused, Refused)
        return refused.codes


class InternalTermTests(MemberCheckTests):
    """#460 live a1 F1: the written text names targets and sources in plain words;
    an exact identifier or claim word of the binding belongs to its references
    and rows, never the prose the page shows."""

    def test_an_identifier_or_claim_word_in_the_text_is_refused(self):
        from bio_neb.reporting.explanation_draft import Refused, check

        for place, changes, token in (
            (
                "why",
                {"why": "Row 4 reads exact_allele_observed."},
                "exact_allele_observed",
            ),
            ("why", {"why": "It was called; see e2."}, "e2"),
            (
                "limitations.1",
                {
                    "limitations": [
                        {
                            "qualifications": [7],
                            "text": "It limits work-item:bb only.",
                            "references": ["e2"],
                        }
                    ]
                },
                "work-item:bb",
            ),
            ("why", {"why": "member-b carries it."}, "member-b"),
            # A qualification's origin code is the binding's own token.
            (
                "why",
                {"why": "It reads coverage_not_measured."},
                "coverage_not_measured",
            ),
            # Spec live r2 LSP4: label punctuation after the token still names it.
            (
                "why",
                {"why": "exact_allele_observed: the sample carries the allele."},
                "exact_allele_observed",
            ),
            ("why", {"why": "The evidence is e2: the sample carries it."}, "e2"),
        ):
            with self.subTest(token=token):
                refused = check(self._binding(), self._draft(**changes))
                self.assertIsInstance(refused, Refused)
                self.assertEqual(refused.codes, (f"explanation_internal_term:{place}",))
                self.assertIn(token, " ".join(refused.faults.sentences))

    def test_plain_words_and_english_state_words_are_admitted(self):
        # Live r1 LSP1: a recorded state that is an English word stays usable;
        # only opaque ids and snake_case labels are refused.
        from bio_neb.reporting.explanation_draft import MemberDraft, check

        for why in (
            "The VCF was searched at this position and called the target allele, "
            "genotype 0/1; the request's candidates were checked.",
            "The researcher removed this target before its VCF search; another was "
            "rejected and one stays unresolved.",
        ):
            with self.subTest(why=why):
                admitted = check(self._binding(), self._draft(why=why))
                self.assertIsInstance(admitted, MemberDraft)


class CombinationCheckTests(unittest.TestCase):
    def _binding(self):
        from bio_neb.reporting.explanation_draft import DraftBinding

        return DraftBinding(
            stage="combination",
            targets=MemberCheckTests.ROWS,
            qualifications=(),
            evidence_event_ids=("e1",),
        )

    def _combined(self, **changes):
        draft = {
            "conclusion": "The one request carries its target allele.",
            "conclusion_references": ["e1"],
            "would_change": "A call at another position would not change it.",
            "limitations": [],
            "not_done": "Nothing was excluded.",
            "genotypes": [],
            "title": "The sample carries its one target allele",
        }
        draft.update(changes)
        return draft

    def test_the_combination_writes_a_short_title_checked_like_its_texts(self):
        # The explainer titles the readable report; the title is at most 120
        # characters and keeps the exact-term check of the texts.
        from bio_neb.reporting.explanation_draft import CombinationDraft, check

        self.assertEqual(
            check(self._binding(), self._combined(title="x" * 120)).title, "x" * 120
        )
        self.assertIsInstance(
            check(self._binding(), self._combined()), CombinationDraft
        )
        for title, code in (
            ("x" * 121, "title:string_too_long"),
            ("None is not_explained", "explanation_internal_term:title"),
        ):
            with self.subTest(code=code):
                self.assertEqual(
                    check(self._binding(), self._combined(title=title)).codes,
                    (code,),
                )
        untitled = self._combined()
        del untitled["title"]
        self.assertEqual(check(self._binding(), untitled).codes, ("title:missing",))

    def test_a_flat_conclusion_is_admitted(self):
        from bio_neb.reporting.explanation_draft import CombinationDraft, check

        admitted = check(self._binding(), self._combined())
        self.assertIsInstance(admitted, CombinationDraft)
        self.assertEqual(
            admitted.would_change, "A call at another position would not change it."
        )

    def test_the_conclusion_is_short_but_longer_than_a_why(self):
        # The conclusion tops the report, at most 1,500 characters.
        from bio_neb.reporting.explanation_draft import CombinationDraft, check

        self.assertIsInstance(
            check(self._binding(), self._combined(conclusion="x" * 1500)),
            CombinationDraft,
        )
        self.assertEqual(
            check(self._binding(), self._combined(conclusion="x" * 1501)).codes,
            ("conclusion:string_too_long",),
        )

    def test_the_conclusion_s_texts_keep_the_structural_checks(self):
        """A genotype listed from what would change the answer is checked like the
        conclusion's: its target's call, by membership (ADR 0049, #473 owner Q5
        a). Coverage words are the model's own: ADR 0049 retires the word ban
        (#474 owner Q2 a)."""
        from bio_neb.reporting.explanation_draft import Refused, check

        refused = check(
            self._binding(),
            self._combined(
                would_change="A 1/1 call would change it.",
                not_done="Coverage was not read.",
                genotypes=[{"target": 4, "genotype": "1/1", "quote": "A 1/1 call"}],
            ),
        )
        self.assertIsInstance(refused, Refused)
        self.assertEqual(
            [code.partition(":")[0] for code in refused.codes],
            ["explanation_genotype_unsupported"],
        )

    def test_a_settled_status_word_in_what_was_not_done_is_refused(self):
        # Live a1 (Standards live r1): "none is not_explained" is the machine word.
        from bio_neb.reporting.explanation_draft import Refused, check

        refused = check(
            self._binding(),
            self._combined(
                not_done="Both requests were explained; none is not_explained."
            ),
        )
        self.assertIsInstance(refused, Refused)
        self.assertEqual(refused.codes, ("explanation_internal_term:not_done",))

    def test_a_request_list_is_no_longer_the_models_to_write(self):
        from bio_neb.reporting.explanation_draft import Refused, check

        refused = check(
            self._binding(),
            self._combined(requests=[{"row": 3, "claims": ["candidates"]}]),
        )
        self.assertIsInstance(refused, Refused)
        self.assertEqual(refused.codes, ("requests:extra_forbidden",))


if __name__ == "__main__":
    unittest.main()
