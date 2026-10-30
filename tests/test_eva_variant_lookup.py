"""An EVA rsID lookup reaches the shared lookup result's typed outcomes, on the Run's
contig names (#366), with every placement EVA states copied (#468, ADR 0049).

The lookup is driven through its real entry, `look_up_eva_rsid`, with recorded EVA
bodies (`tests/eva_retained_bodies.py`), NCBI's recorded report of the Run's assembly
(`tests/location_retained_bodies.py`) or synthetic ones named as such; the fake only
answers the GETs the lookup asks for, in order.
"""

from __future__ import annotations

import json
import unittest

from bio_neb.investigation.investigation import RetrievedSource
from bio_neb.investigation.investigation_request import RequestedVariantIdentity
from bio_neb.investigation.investigation_sources import look_up_eva_rsid
from bio_neb.investigation.lookup_origins import Malformed
from tests.eva_retained_bodies import (
    EVA_CONTIG_ALIAS,
    EVA_IDENTIFIERS,
    PIG_CM000823_5_ALIAS,
    PIG_RS5334475172,
    PIG_RS5334475172_SUBMITTED,
)
from tests.location_retained_bodies import SSCROFA11_REPORT

SSCROFA11 = "GCF_000003025.6"
NEWER_PIG = "GCF_054392235.1"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession"


def _report(accession: str, paired: str, name: str) -> bytes:
    """A synthetic NCBI Datasets report of an assembly, in its recorded shape."""
    return json.dumps(
        {
            "reports": [
                {
                    "accession": accession,
                    "paired_accession": paired,
                    "assembly_info": {
                        "assembly_name": name,
                        "assembly_status": "current",
                    },
                }
            ],
            "total_count": 1,
        }
    ).encode()


RUN_REPORTS = {
    ("fetch_assembly_report", SSCROFA11): (200, SSCROFA11_REPORT),
    ("fetch_assembly_report", NEWER_PIG): (
        200,
        _report(NEWER_PIG, "GCA_054392235.1", "Sscrofa_newer"),
    ),
    ("fetch_assembly_report", "GCA_000003025.6"): (
        200,
        _report("GCA_000003025.6", SSCROFA11, "Sscrofa11.1"),
    ),
}


class _Eva:
    """Answers each asked GET from a table and keeps the order it was asked in."""

    def __init__(self, answers: dict[tuple[str, str], tuple[int, bytes]]) -> None:
        self.answers = answers
        self.asked: list[tuple[str, str]] = []

    def __call__(self, operation: str, key: str) -> tuple[RetrievedSource, bytes]:
        self.asked.append((operation, key))
        status, body = self.answers[(operation, key)]
        url = {
            "eva_clustered_variant": f"{EVA_IDENTIFIERS}/clustered-variants/{key[2:]}",
            "eva_submitted_variants": f"{EVA_IDENTIFIERS}/clustered-variants/{key[2:]}/submitted",
            "eva_contig_alias": f"{EVA_CONTIG_ALIAS}/chromosomes/genbank/{key}",
            "fetch_assembly_report": f"{DATASETS}/{key}/dataset_report",
        }[operation]
        return (
            RetrievedSource.create(
                operation=operation,
                request_identity=url,
                status_code=status,
                observed_at="2026-10-17T12:00:00Z",
                elapsed_time_ms=1,
                raw_response=body,
            ),
            body,
        )


def _pig(**extra) -> _Eva:
    return _Eva(
        {
            **RUN_REPORTS,
            ("eva_clustered_variant", "rs5334475172"): (200, PIG_RS5334475172),
            ("eva_contig_alias", "CM000823.5"): (200, PIG_CM000823_5_ALIAS),
            ("eva_submitted_variants", "rs5334475172"): (
                200,
                PIG_RS5334475172_SUBMITTED,
            ),
            **extra,
        }
    )


class RecordedPigTests(unittest.TestCase):
    def test_the_pig_deletion_resolves_on_the_runs_refseq_contig(self) -> None:
        eva = _pig()

        result = look_up_eva_rsid(
            rsid="rs5334475172", assembly_accession=SSCROFA11, fetch=eva
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.lookup_source, "eva")
        self.assertEqual(result.verified_rsid, "rs5334475172")
        self.assertEqual(
            [
                (
                    c.assembly_accession,
                    c.assembly_name,
                    c.contig,
                    c.one_based_position,
                    c.ref,
                    c.alt,
                    c.aliases,
                )
                for c in result.candidates
            ],
            [
                (
                    SSCROFA11,
                    "Sscrofa11.1",
                    "NC_010454.4",
                    3556402,
                    "GTGTCCCCCTCGC",
                    "",
                    ("rs5334475172",),
                )
            ],
        )
        self.assertEqual(result.source.operation, "eva_clustered_variant")
        self.assertEqual(result.source.authority, "eva")
        self.assertEqual(
            [s.operation for s in result.eva.supporting_sources],
            ["fetch_assembly_report", "eva_contig_alias", "eva_submitted_variants"],
        )
        self.assertEqual(result.eva.run_insdc_assembly_accession, "GCA_000003025.6")
        [placement] = result.placements
        self.assertEqual(
            (
                placement.eva.assembly_accession,
                placement.sequence,
                placement.eva.start,
                placement.eva.type,
                placement.eva.taxonomy_accession,
                placement.on_run_assembly,
                placement.run_contig,
                [(a.ss, a.reference, a.alternate) for a in placement.eva.submitted],
            ),
            (
                "GCA_000003025.6",
                "CM000823.5",
                3556402,
                "DEL",
                9823,
                True,
                "NC_010454.4",
                [("ss7881321083", "GTGTCCCCCTCGC", "")],
            ),
        )
        self.assertEqual(
            eva.asked,
            [
                ("eva_clustered_variant", "rs5334475172"),
                ("fetch_assembly_report", SSCROFA11),
                ("eva_contig_alias", "CM000823.5"),
                ("eva_submitted_variants", "rs5334475172"),
            ],
        )

    def test_the_pig_rsid_is_incompatible_on_a_newer_pig_assembly(self) -> None:
        eva = _pig()

        result = look_up_eva_rsid(
            rsid="rs5334475172", assembly_accession=NEWER_PIG, fetch=eva
        )

        self.assertEqual(
            (result.status, result.candidates, result.conflicts),
            ("incompatible", (), ("EVA had no placement on the exact assembly.",)),
        )
        [placement] = result.placements
        self.assertEqual(
            (placement.on_run_assembly, placement.run_contig), (False, None)
        )
        # The newer assembly's INSDC twin, as NCBI pairs it; EVA places the rsID on
        # another, so no alias and no submitted variant is asked.
        self.assertEqual(result.eva.run_insdc_assembly_accession, "GCA_054392235.1")
        self.assertEqual(
            eva.asked,
            [
                ("eva_clustered_variant", "rs5334475172"),
                ("fetch_assembly_report", NEWER_PIG),
            ],
        )

    def test_a_named_placement_off_the_record_is_refused_beside_an_exact_target(
        self,
    ) -> None:
        """A named placement is checked by membership even when the exact target
        resolves (owner #468 Q1 a; Spec r1 SP3)."""
        result = look_up_eva_rsid(
            rsid="rs5334475172",
            assembly_accession=SSCROFA11,
            fetch=_pig(),
            requested_variant=RequestedVariantIdentity(
                "NC_010454.4", 3556402, "GTGTCCCCCTCGC", ""
            ),
            placement="NC_999999.1",
        )

        self.assertEqual(result.status, "placement_unnamed")
        self.assertIn("NC_999999.1 is not one of", result.conflicts[0])

    def test_a_request_naming_another_allele_conflicts(self) -> None:
        result = look_up_eva_rsid(
            rsid="rs5334475172",
            assembly_accession=SSCROFA11,
            fetch=_pig(),
            requested_variant=RequestedVariantIdentity(
                "NC_010454.4", 3556402, "G", "A"
            ),
        )

        self.assertEqual(result.status, "conflicted")
        self.assertEqual(
            result.conflicts, ("EVA did not contain the exact requested target.",)
        )
        self.assertEqual(len(result.candidates), 1)


def _clustered(accession: int, contig: str, start: int, kind: str = "SNV") -> bytes:
    """A synthetic clustered-variant answer in EVA's recorded shape."""
    return json.dumps(
        [
            {
                "accession": accession,
                "version": 1,
                "data": {
                    "assemblyAccession": "GCA_000003025.6",
                    "taxonomyAccession": 9823,
                    "contig": contig,
                    "start": start,
                    "type": kind,
                },
            }
        ]
    ).encode()


def _submitted(cluster: int, *alleles: tuple[int, str, str], start: int) -> bytes:
    """Synthetic submitted variants in EVA's recorded shape."""
    return json.dumps(
        [
            {
                "accession": ss,
                "version": 1,
                "data": {
                    "referenceSequenceAccession": "GCA_000003025.6",
                    "projectAccession": "PRJEB00001",
                    "contig": "CM000823.5",
                    "start": start,
                    "referenceAllele": reference,
                    "alternateAllele": alternate,
                    "clusteredVariantAccession": cluster,
                },
            }
            for ss, reference, alternate in alleles
        ]
    ).encode()


def _snv(*alleles, cluster: int = 81000001, overrides=None) -> _Eva:
    rsid = f"rs{cluster}"
    return _Eva(
        {
            **RUN_REPORTS,
            ("eva_clustered_variant", rsid): (
                200,
                _clustered(cluster, "CM000823.5", 1000),
            ),
            ("eva_contig_alias", "CM000823.5"): (200, PIG_CM000823_5_ALIAS),
            ("eva_submitted_variants", rsid): (
                200,
                _submitted(81000001, *alleles, start=1000),
            ),
            **(overrides or {}),
        }
    )


class SyntheticAnswerTests(unittest.TestCase):
    def test_each_submitted_snv_alternate_is_its_own_candidate(self) -> None:
        result = look_up_eva_rsid(
            rsid="rs81000001",
            assembly_accession=SSCROFA11,
            fetch=_snv((1, "A", "G"), (2, "A", "T"), (3, "A", "G")),
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual(
            [(c.contig, c.one_based_position, c.ref, c.alt) for c in result.candidates],
            [("NC_010454.4", 1000, "A", "G"), ("NC_010454.4", 1000, "A", "T")],
        )

    def test_a_request_naming_one_alternate_keeps_that_candidate(self) -> None:
        result = look_up_eva_rsid(
            rsid="rs81000001",
            assembly_accession=SSCROFA11,
            fetch=_snv((1, "A", "G"), (2, "A", "T")),
            requested_variant=RequestedVariantIdentity("NC_010454.4", 1000, "A", "T"),
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual([c.alt for c in result.candidates], ["T"])

    def test_a_gca_run_keeps_evas_insdc_contig_name(self) -> None:
        eva = _snv((1, "A", "G"))

        result = look_up_eva_rsid(
            rsid="rs81000001", assembly_accession="GCA_000003025.6", fetch=eva
        )

        self.assertEqual(
            [(c.assembly_accession, c.contig) for c in result.candidates],
            [("GCA_000003025.6", "CM000823.5")],
        )
        # EVA already names the sequence as a GCA_ Run does: no alias is asked.
        self.assertNotIn(("eva_contig_alias", "CM000823.5"), eva.asked)

    def test_an_empty_clustered_answer_is_not_found(self) -> None:
        eva = _Eva({("eva_clustered_variant", "rs9"): (200, b"[]")})

        result = look_up_eva_rsid(rsid="rs9", assembly_accession=SSCROFA11, fetch=eva)

        self.assertEqual(
            (result.status, result.conflicts, result.verified_rsid),
            ("not_found", ("EVA has no clustered variant rs9.",), None),
        )
        self.assertEqual(eva.asked, [("eva_clustered_variant", "rs9")])

    def test_a_non_2xx_answer_of_any_lookup_get_is_the_sources_to_read(self) -> None:
        """No status has a fixed meaning: the lookup ends on the answer, for the model to
        read in the Tool's source answers (ADR 0049; owner #468 R1 b, R4 a)."""
        cases = {
            "clustered 404": (("eva_clustered_variant", "rs81000001"), 404, 1),
            "clustered 410": (("eva_clustered_variant", "rs81000001"), 410, 1),
            "the Run's report 400": (("fetch_assembly_report", SSCROFA11), 400, 2),
            "contig alias 404": (("eva_contig_alias", "CM000823.5"), 404, 3),
            "submitted 410": (("eva_submitted_variants", "rs81000001"), 410, 4),
        }
        for name, (get, status, asked) in cases.items():
            with self.subTest(name):
                eva = _snv((1, "A", "G"), overrides={get: (status, b'{"m": "x"}')})

                result = look_up_eva_rsid(
                    rsid="rs81000001", assembly_accession=SSCROFA11, fetch=eva
                )

                self.assertEqual(
                    (result.status, result.candidates), ("source_answered", ())
                )
                self.assertIn(
                    f"answered HTTP {status} for {get[1]}", result.conflicts[0]
                )
                self.assertEqual(eva.asked[-1], get)
                self.assertEqual(len(eva.asked), asked)

    def test_unreadable_or_foreign_answers_never_resolve(self) -> None:
        cases = {
            "not a list": (
                {("eva_clustered_variant", "rs81000001"): (200, b'{"a": 1}')},
                "malformed",
            ),
            "another cluster's record": (
                {
                    ("eva_clustered_variant", "rs81000001"): (
                        200,
                        _clustered(81000002, "CM000823.5", 1000),
                    )
                },
                "conflicted",
            ),
            "a submitted variant of another cluster": (
                {
                    ("eva_submitted_variants", "rs81000001"): (
                        200,
                        _submitted(81000002, (1, "A", "G"), start=1000),
                    )
                },
                "malformed",
            ),
            # A copied name is never judged by its shape (ADR 0049); one that is
            # not one line of text is no name.
            "an alias naming the Run contig on two lines": (
                {
                    ("eva_contig_alias", "CM000823.5"): (
                        200,
                        PIG_CM000823_5_ALIAS.replace(
                            b"NC_010454.4", b"NC_010454.4\\nx"
                        ),
                    )
                },
                "malformed",
            ),
            "an alias naming the Run contig as empty": (
                {
                    ("eva_contig_alias", "CM000823.5"): (
                        200,
                        PIG_CM000823_5_ALIAS.replace(b'"NC_010454.4"', b'""'),
                    )
                },
                "malformed",
            ),
            # The Run's assembly name comes from NCBI's report of it, no longer
            # from EVA's alias; a padded one names nothing.
            "NCBI's report naming the Run's assembly with padding": (
                {
                    ("fetch_assembly_report", SSCROFA11): (
                        200,
                        SSCROFA11_REPORT.replace(b'"Sscrofa11.1"', b'" Sscrofa11.1"'),
                    )
                },
                "malformed",
            ),
            "an unreadable contig alias": (
                {("eva_contig_alias", "CM000823.5"): (200, b"[1]")},
                "malformed",
            ),
        }
        for name, (answers, status) in cases.items():
            with self.subTest(name):
                result = look_up_eva_rsid(
                    rsid="rs81000001",
                    assembly_accession=SSCROFA11,
                    fetch=_snv((1, "A", "G"), overrides=answers),
                )

                self.assertEqual(result.status, status)
                self.assertEqual(result.candidates, ())

    def test_an_alias_assembly_identifier_of_another_shape_is_malformed(self) -> None:
        for field, value in (
            ("refseq", ["GCF_000003025.6"]),
            ("refseq", {"accession": "GCF_000003025.6"}),
            ("insdcAccession", 6),
        ):
            with self.subTest(field=field, value=value):
                alias = json.loads(PIG_CM000823_5_ALIAS)
                alias["_embedded"]["chromosomeEntities"][0]["assembly"][field] = value
                eva = _snv(
                    (1, "A", "G"),
                    overrides={
                        ("eva_contig_alias", "CM000823.5"): (
                            200,
                            json.dumps(alias).encode(),
                        )
                    },
                )

                result = look_up_eva_rsid(
                    rsid="rs81000001", assembly_accession=SSCROFA11, fetch=eva
                )

                # The ending is the typed origin's audit sentence.
                self.assertEqual(
                    (result.status, result.candidates, result.conflicts),
                    (
                        "malformed",
                        (),
                        (
                            "EVA's placements could not be read: EVA's contig-alias "
                            "answer was malformed.",
                        ),
                    ),
                )
                self.assertEqual(
                    result.origin,
                    Malformed(
                        what="EVA's placements",
                        detail="EVA's contig-alias answer was malformed.",
                    ),
                )
                self.assertNotIn(("eva_submitted_variants", "rs81000001"), eva.asked)

    def test_an_alias_with_no_refseq_twin_is_off_a_gcf_run(self) -> None:
        alias = json.loads(PIG_CM000823_5_ALIAS)
        alias["_embedded"]["chromosomeEntities"][0]["assembly"]["refseq"] = None
        result = look_up_eva_rsid(
            rsid="rs81000001",
            assembly_accession=SSCROFA11,
            fetch=_snv(
                (1, "A", "G"),
                overrides={
                    ("eva_contig_alias", "CM000823.5"): (
                        200,
                        json.dumps(alias).encode(),
                    )
                },
            ),
        )

        self.assertEqual(result.status, "incompatible")

    def test_a_clustered_contig_that_is_no_one_line_name_is_malformed_unasked(
        self,
    ) -> None:
        """EVA's contig is copied, never judged by an INSDC shape (#467, ADR 0049); a
        value that is not one line of text names nothing to ask about."""
        eva = _snv(
            (1, "A", "G"),
            overrides={
                ("eva_clustered_variant", "rs81000001"): (
                    200,
                    _clustered(81000001, "CM000823.5\nx", 1000),
                )
            },
        )

        result = look_up_eva_rsid(
            rsid="rs81000001", assembly_accession=SSCROFA11, fetch=eva
        )

        self.assertEqual(
            (result.status, result.candidates, result.conflicts),
            (
                "malformed",
                (),
                (
                    "EVA's placements could not be read: EVA's clustered-variant "
                    "records were malformed.",
                ),
            ),
        )
        self.assertEqual(result.origin.code, "malformed")
        self.assertEqual(result.assessment.rationale, result.conflicts[0])
        self.assertEqual(eva.asked, [("eva_clustered_variant", "rs81000001")])

    def test_a_contig_alias_404_is_evas_answer_not_an_off_run_verdict(self) -> None:
        """#375 B6 migrated: a 404 has no fixed meaning (ADR 0049; owner #468 R1 b). The
        placement stays copied on the Run's assembly, and the lookup ends on EVA's own
        answer for the model to read."""
        result = look_up_eva_rsid(
            rsid="rs81000001",
            assembly_accession=SSCROFA11,
            fetch=_snv(
                (1, "A", "G"),
                overrides={("eva_contig_alias", "CM000823.5"): (404, b"{}")},
            ),
        )

        self.assertEqual(result.status, "source_answered")
        [placement] = result.placements
        self.assertTrue(placement.on_run_assembly)

    def test_evas_contig_naming_is_passed_as_is_and_its_names_copied(self) -> None:
        """EVA's own contigNamingConvention option reaches EVA unchanged (owner #468
        R3); under REFSEQ EVA already names the sequence as a GCF_ Run does, so no alias
        is asked. The spike's real REFSEQ answer renamed C. elegans' BX284603.4 to
        NC_003281.10; this body is synthetic in its shape."""
        refseq = _clustered(81000001, "NC_010454.4", 1000)
        submitted = _submitted(81000001, (1, "A", "G"), start=1000).replace(
            b"CM000823.5", b"NC_010454.4"
        )
        named = "rs81000001?contigNamingConvention=REFSEQ"
        eva = _Eva(
            {
                **RUN_REPORTS,
                ("eva_clustered_variant", named): (200, refseq),
                ("eva_submitted_variants", named): (200, submitted),
            }
        )

        result = look_up_eva_rsid(
            rsid="rs81000001",
            assembly_accession=SSCROFA11,
            fetch=eva,
            contig_naming="REFSEQ",
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.eva.contig_naming, "REFSEQ")
        self.assertEqual(
            [(c.contig, c.alt) for c in result.candidates], [("NC_010454.4", "G")]
        )
        self.assertEqual(
            eva.asked,
            [
                ("eva_clustered_variant", named),
                ("fetch_assembly_report", SSCROFA11),
                ("eva_submitted_variants", named),
            ],
        )

    def test_a_naming_eva_does_not_document_is_refused_before_contact(self) -> None:
        eva = _snv((1, "A", "G"))

        with self.assertRaisesRegex(ValueError, "one EVA documents"):
            look_up_eva_rsid(
                rsid="rs81000001",
                assembly_accession=SSCROFA11,
                fetch=eva,
                contig_naming="refseq",
            )
        self.assertEqual(eva.asked, [])

    def test_evas_alias_pages_are_read_until_the_runs_pair(self) -> None:
        """EVA pages its alias answer (10 by default; #468 spike): the lookup reads
        its next page rather than cut it silently (ADR 0049), within the general Tool
        budget (owner R2 a: no cap of the lookup's own)."""
        first = json.loads(PIG_CM000823_5_ALIAS)
        other = first["_embedded"]["chromosomeEntities"][0]
        first["_embedded"]["chromosomeEntities"] = [
            {
                **other,
                "assembly": {
                    **other["assembly"],
                    "insdcAccession": "GCA_1.1",
                    "refseq": "GCF_1.1",
                },
            }
        ]
        first["page"] = {"size": 1, "totalElements": 2, "totalPages": 2, "number": 0}
        second = json.loads(PIG_CM000823_5_ALIAS)
        second["page"] = {"size": 1, "totalElements": 2, "totalPages": 2, "number": 1}
        eva = _snv(
            (1, "A", "G"),
            overrides={
                ("eva_contig_alias", "CM000823.5"): (200, json.dumps(first).encode()),
                ("eva_contig_alias", "CM000823.5?page=1"): (
                    200,
                    json.dumps(second).encode(),
                ),
            },
        )

        result = look_up_eva_rsid(
            rsid="rs81000001", assembly_accession=SSCROFA11, fetch=eva
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual([c.contig for c in result.candidates], ["NC_010454.4"])
        self.assertIn(("eva_contig_alias", "CM000823.5?page=1"), eva.asked)

    def test_a_dbsnp_result_states_that_eva_does_not_apply(self) -> None:
        from bio_neb.investigation.investigation_sources import normalize_refsnp
        from tests.refsnp_retained_bodies import RS2228570

        source = RetrievedSource.create(
            operation="fetch_refsnp",
            request_identity="https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/2228570",
            status_code=200,
            observed_at="2026-10-17T12:00:00Z",
            elapsed_time_ms=1,
            raw_response=RS2228570,
        )
        result = normalize_refsnp(
            rsid="rs2228570",
            assembly_accession="GCF_000001405.40",
            source=source,
            raw_response=RS2228570,
        )

        dumped = result.model_dump(mode="json")
        self.assertEqual(
            (dumped["lookup_source"], dumped["eva"]), ("ncbi_dbsnp", "not_applicable")
        )


if __name__ == "__main__":
    unittest.main()
