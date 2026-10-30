"""Each explanation member's rows, claims and checks (#408, owner parts 1-3).

A member explains one request of a published report from that request's own
rows. Each row's claim word is computed from the state the approved basis
recorded for it (the owner's table, part 3); a member's draft states one claim
per row, and admission compares those claim fields with the rows, not the prose.
The combination lists every request with the claim words its member stated.

The rows here are written by hand from the basis vocabulary
(``collection_view.ItemState``/``RequestState`` and ``ObservationOutcome``).
"""

import unittest

from bio_neb.reporting.explanation_rows import ExplanationRow


def _item(row, member, outcome=None, state="observed", genotype=None, item="w1"):
    return ExplanationRow(
        number=row,
        kind="position",
        request_member_id=member,
        subject=f"target-{row}",
        state=state,
        work_item_id=f"work-item:{item}",
        outcome=outcome,
        genotype=genotype,
        words="recorded words",
    )


def _request(row, member, subject, state="candidates"):
    return ExplanationRow(
        number=row,
        kind="request",
        request_member_id=member,
        subject=subject,
        state=state,
        words="recorded words",
    )


class ClaimWordTests(unittest.TestCase):
    def test_each_recorded_state_reads_as_the_owners_claim_word(self):
        from bio_neb.reporting.explanation_members import row_claim

        observed = {
            "exact_allele_observed": "carries_target_allele",
            "target_allele_not_carried": "target_allele_not_carried",
            "target_genotype_unresolved": "genotype_unresolved",
            "locus_present_identity_mismatch": "different_variant_at_locus",
            "locus_overlapped": "locus_overlapped",
            "no_record": "searched_no_record",
            "ambiguous_records": "ambiguous_records",
            "refused": "lookup_refused",
        }
        for outcome, word in observed.items():
            with self.subTest(outcome=outcome):
                self.assertEqual(row_claim(_item(2, "m", outcome)), word)
        not_observed = {
            "failed": "lookup_failed",
            "approved_not_observed": "approved_not_looked_up",
            "not_approved": "not_approved",
            "unselected_candidate": "unselected_candidate",
            "removed": "removed",
            "deferred": "deferred",
        }
        for state, word in not_observed.items():
            with self.subTest(state=state):
                self.assertEqual(row_claim(_item(2, "m", state=state)), word)
        for state in (
            "candidates",
            "unresolved",
            "rejected",
            "removed",
            "deferred",
            "not_proposed",
            "never_run",
            # A request not checked by what it states, or deferred out of the Run
            # after its lookup (it read as state_not_recorded).
            "not_checked",
        ):
            with self.subTest(request=state):
                self.assertEqual(row_claim(_request(1, "m", "rs1", state)), state)
        # A row the basis recorded no state for, and an observed item with no
        # outcome, claim only that.
        self.assertEqual(row_claim(_item(2, "m", state=None)), "state_not_recorded")
        self.assertEqual(row_claim(_request(1, "m", "rs1", None)), "state_not_recorded")
        self.assertEqual(row_claim(_item(2, "m", None)), "state_not_recorded")


class MemberBindingTests(unittest.TestCase):
    """A member binds its request's rows, the qualifications that name the
    request, and the evidence events that name it (design note, binding rules)."""

    ROWS = (
        _request(1, "member-a", "rs12"),
        _item(2, "member-a", "no_record", item="aa"),
        _request(3, "member-b", "rs123"),
        _item(4, "member-b", "exact_allele_observed", genotype="0/1", item="bb"),
        _item(5, "member-b", "target_allele_not_carried", item="aa"),
        _request(6, "member-c", "rs999", state="unresolved"),
    )

    def test_each_request_binds_its_own_rows_in_report_order(self):
        from bio_neb.reporting.explanation_members import member_rows

        self.assertEqual(
            {
                member: [row.number for row in rows]
                for member, rows in member_rows(self.ROWS)
            },
            {"member-a": [1, 2], "member-b": [3, 4, 5], "member-c": [6]},
        )
        self.assertEqual(
            [member for member, _rows in member_rows(self.ROWS)],
            ["member-a", "member-b", "member-c"],
        )

    def test_a_qualification_goes_to_the_requests_whose_rows_it_concerns(self):
        # Ids, never the text, place a qualification. One over every looked-up
        # row (2, 4 and 5 here) is the Run's and goes to the combination, as
        # does one with no ids; one over some rows goes to each member whose
        # rows it concerns, with only that member's rows.
        from types import SimpleNamespace

        from bio_neb.investigation import qualification_origins as origins
        from bio_neb.reporting.explanation_members import (
            combination_qualifications,
            member_qualifications,
        )
        from bio_neb.reporting.explanation_rows import numbered_qualifications
        from bio_neb.reporting.finalization import Qualification

        report = SimpleNamespace(
            limitations=(
                Qualification(
                    text="Coverage, callability and causality are not inferred.",
                    origin=origins.ObservationNotInferred(),
                    work_item_ids=("work-item:aa", "work-item:bb"),
                ),
                Qualification(
                    text="Run provenance is unestablished.",
                    origin=origins.ObservationProvenance(
                        run_provenance="unestablished"
                    ),
                ),
            ),
            conflicts=(),
            gaps=(
                Qualification(
                    text="Coverage was not measured for Work item work-item:aa.",
                    origin=origins.CoverageNotMeasured(),
                    work_item_ids=("work-item:aa",),
                ),
                Qualification(
                    text="Coverage was not measured for Work item work-item:bb.",
                    origin=origins.CoverageNotMeasured(),
                    work_item_ids=("work-item:bb",),
                ),
                Qualification(
                    text="Request rs999 has no Work item: refsnp: not_found.",
                    origin=origins.RequestUnresolved(
                        source="refsnp", verdict="not_found"
                    ),
                    request_member_ids=("member-c",),
                ),
                # Its words name rs12 (member-a); its id is member-b's item.
                Qualification(
                    text="rs12 was checked against its own record.",
                    origin=origins.ObservationScope(outcome="exact_allele_observed"),
                    work_item_ids=("work-item:bb",),
                ),
            ),
        )
        qualifications = numbered_qualifications(self.ROWS, report)
        self.assertEqual(
            [(item.qualification, item.kind, item.targets) for item in qualifications],
            [
                (1, "limitation", (2, 4, 5)),
                (2, "limitation", ()),
                (3, "gap", (2, 5)),
                (4, "gap", (4,)),
                (5, "gap", (6,)),
                (6, "gap", (4,)),
            ],
        )
        self.assertEqual(
            {
                member: [
                    (item.qualification, item.targets)
                    for item in member_qualifications(self.ROWS, member, qualifications)
                ]
                for member in ("member-a", "member-b", "member-c")
            },
            {
                "member-a": [(3, (2,))],
                "member-b": [(3, (5,)), (4, (4,)), (6, (4,))],
                "member-c": [(5, (6,))],
            },
        )
        self.assertEqual(
            [
                item.qualification
                for item in combination_qualifications(self.ROWS, qualifications)
            ],
            [1, 2],
        )

    def test_no_qualification_is_placed_by_reading_its_words(self):
        # The grep guard D2 asks for: placement reads ids only.
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/bio_neb/reporting/explanation_members.py"
        ).read_text(encoding="utf-8")
        import re

        self.assertIsNone(
            re.search(r"\.search\(\s*item\.text", source),
            "a qualification is placed by a pattern over its words",
        )

    def test_a_members_evidence_is_the_events_that_name_its_request(self):
        from bio_neb.reporting.explanation_members import member_scope

        events = [
            {"event_id": "e1", "payload": {"work_item_id": "work-item:aa"}},
            {"event_id": "e2", "payload": {"nested": {"rsid": "rs123"}}},
            {"event_id": "e3", "payload": {"text": "rs1234 is another SNP"}},
            {"event_id": "e4", "payload": {"request_member_id": "member-c"}},
            {"event_id": "e5", "payload": {"work_item_id": "work-item:aa"}},
            {"event_id": "pub", "payload": {}},
        ]
        scope = ("e1", "e2", "e3", "e4", "pub")
        rows = dict(MemberBindingTests._grouped(self.ROWS))
        self.assertEqual(
            {
                member: member_scope(events, scope, own, "pub")
                for member, own in rows.items()
            },
            {
                # e5 names work item aa but is outside the report's scope.
                "member-a": ("e1", "pub"),
                "member-b": ("e1", "e2", "pub"),
                "member-c": ("e4", "pub"),
            },
        )

    @staticmethod
    def _grouped(rows):
        from bio_neb.reporting.explanation_members import member_rows

        return member_rows(rows)


class RowClaimTests(unittest.TestCase):
    """Code fills each row's claim from its recorded state (#448 D1, owner Q4
    a): the word the owner's table gives and the genotype the row carries."""

    def test_every_row_is_claimed_once_by_its_recorded_state(self):
        from bio_neb.reporting.explanation_members import row_claims

        rows = (
            _request(3, "member-b", "rs123"),
            _item(4, "member-b", "exact_allele_observed", genotype="0/1", item="bb"),
            _item(5, "member-b", "no_record", item="cc"),
            _item(6, "member-b", state="not_approved", item="dd"),
        )
        self.assertEqual(
            [claim.model_dump(mode="json") for claim in row_claims(rows)],
            [
                {"target": 3, "claim": "candidates"},
                {"target": 4, "claim": "carries_target_allele", "genotype": "0/1"},
                {"target": 5, "claim": "searched_no_record"},
                {"target": 6, "claim": "not_approved"},
            ],
        )


class RowWordsTests(unittest.TestCase):
    """Each row carries its typed owner's plain words (#475 A): the explainer reads
    what happened instead of guessing it from a claim code (#457 H5, F-a3-6; #480
    live a1 L2)."""

    @staticmethod
    def _rows(*snps):
        from bio_neb.investigation.collection_view import CollectionView
        from bio_neb.reporting.explanation_rows import explanation_rows

        return explanation_rows(CollectionView(snps=snps, cases=()))

    @staticmethod
    def _snp(member, text, **fields):
        from bio_neb.investigation.collection_view import RequestedSnp

        return RequestedSnp(
            request_member_id=member,
            request_text=text,
            requested_rsid=None,
            verified_rsid=None,
            source_status=None,
            locus=None,
            **{"items": (), **fields},
        )

    @staticmethod
    def _lookup(origin, authority="ncbi", status_code=200):
        from bio_neb.investigation.collection_view import RequestLookup

        return RequestLookup(
            authority=authority,
            operation="lookup",
            asked="key",
            status_code=status_code,
            outcome="not_found",
            verdict="not_found",
            rationale=origin.audit(),
            source_id="source:1",
            origin=origin,
        )

    def test_a_position_stated_without_alleles_reads_that_no_source_was_asked(self):
        # As the ended view builds it: a deferred member keeps no lookup, so its
        # state origin's words say it.
        from bio_neb.investigation.qualification_origins import RequestNotChecked

        (row,) = self._rows(
            self._snp(
                "m1",
                "19:44908822",
                state="not_checked",
                state_origin=RequestNotChecked(
                    reason_code="position_without_alleles",
                    reason="it states 19:44908822 without its alleles or an rsID",
                ),
            )
        )
        self.assertEqual(
            row.words,
            "deferred out of the Run: it was stated without its alleles or an rsID, "
            "so no source was asked about it and the VCF was not searched; this does "
            "not say whether the sample carries a variant there",
        )

    def test_an_answer_without_a_body_reads_as_that_answer(self):
        from bio_neb.investigation import lookup_origins
        from bio_neb.investigation.qualification_origins import RequestUnresolved

        origin = lookup_origins.SourceAnswered(
            source="EVA's variant listing", status_code=404, words_sent=""
        )
        (row,) = self._rows(
            self._snp(
                "m1",
                "rs1",
                state="unresolved",
                state_origin=RequestUnresolved(source="refsnp", verdict="not_found"),
                lookup=self._lookup(origin, authority="eva", status_code=404),
            )
        )
        self.assertEqual(
            row.words,
            "EVA's variant listing answered with status 404 with nothing more; the "
            "variant lookup could not place it on the Run's reference",
        )

    def test_a_request_with_candidates_and_its_positions_read_their_states(self):
        from bio_neb.investigation.collection_view import RequestedSnpItem
        from bio_neb.investigation.qualification_origins import ItemNotApproved

        rows = self._rows(
            self._snp(
                "m1",
                "rs1",
                state="candidates",
                items=(
                    RequestedSnpItem(
                        work_item_id="w1",
                        target="1:10 A>G",
                        outcome="no_record",
                        state="observed",
                    ),
                    RequestedSnpItem(
                        work_item_id="w2",
                        target="1:10 A>T",
                        state="not_approved",
                        state_origin=ItemNotApproved(),
                    ),
                ),
            )
        )
        self.assertEqual(
            [row.words for row in rows],
            [
                "each position proposed for it is its own target",
                "Searched the VCF: no variant call here; genotype not established",
                "proposed, but not approved for a VCF search",
            ],
        )


class AsserterWordsTests(unittest.TestCase):
    """Only a paper asserter (``paper:…``) is named "the paper …"; the researcher's
    prompt-named targets read as the researcher's (M smoke 028a063a7 F1: they read
    "Asserted by: the paper researcher")."""

    PAPER = "paper:doi:10.1000/x"

    def _snp(self, member, asserted_by):
        from bio_neb.investigation.collection_view import RequestedSnp

        return RequestedSnp(
            request_member_id=member,
            request_text=member,
            requested_rsid=member,
            verified_rsid=None,
            source_status=None,
            locus=None,
            items=(),
            asserted_by=asserted_by,
        )

    def test_the_explanation_rows_name_only_a_paper_asserter(self):
        from bio_neb.investigation.collection_view import CollectionView
        from bio_neb.investigation.investigation_request import RESEARCHER
        from bio_neb.reporting.explanation_rows import explanation_rows

        rows = explanation_rows(
            CollectionView(
                snps=(
                    self._snp("rs1", None),
                    self._snp("rs2", RESEARCHER),
                    self._snp("rs3", self.PAPER),
                ),
                cases=(),
            )
        )
        self.assertEqual(
            [row.asserted_by for row in rows],
            [None, None, "the paper doi:10.1000/x"],
        )

    def test_the_readable_report_names_the_researchers_targets_you(self):
        from types import SimpleNamespace

        from bio_neb.investigation.investigation_request import RESEARCHER
        from bio_neb.reporting.readable_report import ReadableRow, _Page

        def row(asserted_by):
            return ReadableRow(
                number=1,
                request_member_id="m1",
                target="rs1",
                label="rs1",
                read_as="rs1",
                state="searched",
                items=(),
                lookup=None,
                asserted_by=asserted_by,
            )

        page = SimpleNamespace(papers={})
        read = SimpleNamespace(papers={self.PAPER: "P1"})
        self.assertEqual(_Page.from_whom(page, row(None)), "you")
        self.assertEqual(_Page.from_whom(page, row(RESEARCHER)), "you")
        self.assertEqual(
            _Page.from_whom(page, row(self.PAPER)), "the paper doi:10.1000/x"
        )
        self.assertEqual(_Page.from_whom(read, row(self.PAPER)), "P1")


if __name__ == "__main__":
    unittest.main()
