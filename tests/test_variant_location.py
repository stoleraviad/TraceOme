"""A position target is placed on the Run's assembly before any record is read (#369).

The lookup is driven through its real entry, `look_up_stated_position`, with the
development GETs' bodies (`tests/location_retained_bodies.py`) and the recorded dbSNP and
EVA bodies, or synthetic ones named as such. The fake answers only the GETs the lookup
asks for, in order, so an unasked GET is never answered and an extra one fails.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from bio_neb.agentic_journey.tool_execution import tool_model_view
from bio_neb.investigation.investigation import (
    BodylessAnswer,
    RetrievedSource,
    VersionNaming,
)
from bio_neb.investigation.investigation_agent import LOOKUP_VIEW
from bio_neb.investigation.investigation_request import (
    PlacedPosition,
    StatedAssembly,
    StatedPosition,
    StatedRegion,
)
from bio_neb.investigation.investigation_sources import (
    look_up_region as _look_up_region,
)
from bio_neb.investigation.investigation_sources import (
    look_up_stated_position as _look_up_stated_position,
)
from bio_neb.investigation.lookup_origins import SequenceUnnamed, VersionUnnamed
from bio_neb.providers.eva_investigation import EvaInvestigationSourceAdapter
from bio_neb.providers.ncbi_datasets import MetadataResponse
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceResponse,
    NcbiInvestigationSourceAdapter,
)
from tests.eva_retained_bodies import (
    PIG_CM000823_5_ALIAS,
    PIG_RS5334475172,
    PIG_RS5334475172_SUBMITTED,
)
from tests.location_retained_bodies import (
    EVA_AT_PIG_DEL,
    GRCH37_NAMES,
    GRCH38_NAMES,
    GRCH38_REPORT,
    SPAN_NC_000019_10_T,
    SPDI_RS429358,
    SPDI_UNLISTED,
    SSCROFA11_NAMES,
    SSCROFA11_REPORT,
)
from tests.refsnp_retained_bodies import RS429358, RS4149056, RS60037639
from tests.sequence_naming_support import assembly_report_text, literal_choices

GRCH38 = "GCF_000001405.40"
SSCROFA11 = "GCF_000003025.6"
# Synthetic, in the retained span body's shape: no GET read these five bases.
SPAN_NC_000019_10_TGCGG = (
    b'{"data":{"seq_id":"NC_000019.10","position":44908683,'
    b'"deleted_sequence":"TGCGG","inserted_sequence":"NNNNN"}}'
)
# The retained pig span (g18): the fourteen bases at NC_010454.4:3556401.
SPAN_NC_010454_4 = (
    b'{"data":{"seq_id":"NC_010454.4","position":3556400,'
    b'"deleted_sequence":"GGTGTCCCCCTCGC","inserted_sequence":"NNNNNNNNNNNNNN"}}'
)


def _lookup_view(result):
    """The lookup Tool's model view of ``result`` recorded as its fact (#478)."""
    return tool_model_view(
        LOOKUP_VIEW,
        {"event_id": "lookup-1", "payload": {"result": result.model_dump(mode="json")}},
    )


def _scripted_naming(kwargs: dict, contig: str) -> dict:
    """The scripted model names, on each assembly the fake answers names of, the record
    literally named the written contig (`tests.sequence_naming_support`, #467); a test
    of another naming passes its own ``choices``."""
    fetch = kwargs["fetch"]
    answers = getattr(fetch, "answers", None) or getattr(
        getattr(fetch, "inner", None), "answers", {}
    )
    kwargs.setdefault("choices", literal_choices(answers, contig))
    kwargs.setdefault("words", (contig,))
    return kwargs


def _scripted_listing(kwargs: dict) -> dict:
    """The scripted model's reading of a listing's 404 (#468, owner R5 b): when the fake
    answers NCBI's "does not match any RSID", the call quotes it as ``listed_none``; a
    test of another reading passes its own (`ListingAnswerTests`)."""
    fetch = kwargs["fetch"]
    answers = getattr(fetch, "answers", None) or getattr(
        getattr(fetch, "inner", None), "answers", {}
    )
    if (404, SPDI_UNLISTED) in answers.values():
        kwargs.setdefault("listed_none", "does not match any RSID")
    return kwargs


def _scripted_source(kwargs: dict) -> dict:
    """The scripted model's choice of variant source (#469: the model chooses from the
    Tools' metadata, no code routes): dbSNP on a human Run, else EVA; a test of another
    choice passes its own."""
    human = kwargs["assembly_accession"][4:13] in ("000001405", "009914755")
    kwargs.setdefault("source", "ncbi_dbsnp" if human else "eva")
    return kwargs


def look_up_stated_position(**kwargs):
    position = kwargs.get("placed") or kwargs["position"]
    return _look_up_stated_position(
        **_scripted_listing(_scripted_naming(_scripted_source(kwargs), position.contig))
    )


def look_up_region(**kwargs):
    return _look_up_region(**_scripted_naming(kwargs, kwargs["region"].contig))


class _Sources:
    """Answers each asked GET from a table and keeps the order it was asked in."""

    def __init__(self, answers: dict[tuple[str, str], tuple[int, bytes]]) -> None:
        self.answers = answers
        self.asked: list[tuple[str, str]] = []

    def __call__(self, operation: str, key: str) -> tuple[RetrievedSource, bytes]:
        self.asked.append((operation, key))
        status, body = self.answers[(operation, key)]
        return (
            RetrievedSource.create(
                operation=operation,
                request_identity=f"https://source.example/{operation}/{key}",
                status_code=status,
                observed_at="2026-10-17T12:00:00Z",
                elapsed_time_ms=1,
                raw_response=body,
            ),
            body,
        )


def _human(extra=None) -> _Sources:
    return _Sources(
        {
            ("fetch_assembly_report", GRCH38): (200, GRCH38_REPORT),
            ("ncbi_sequence_names", GRCH38): (200, GRCH38_NAMES),
            **(extra or {}),
        }
    )


def _request(contig, position, ref=None, alt=None, *, rsid=None, assembly=None):
    return {
        "position": StatedPosition(contig=contig, position=position, ref=ref, alt=alt),
        "stated_assembly": assembly,
        "rsid": rsid,
        "requested_rsid": rsid,
    }


def _candidates(result):
    return [
        (c.assembly_name, c.contig, c.one_based_position, c.ref, c.alt, c.aliases)
        for c in result.candidates
    ]


class HumanPositionTests(unittest.TestCase):
    def test_a_chromosome_name_is_placed_and_its_listed_rsid_read(self) -> None:
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (200, SPDI_RS429358),
                ("fetch_refsnp", "rs429358"): (200, RS429358),
                # No version is stated: the allele is listed on GRCh37 too.
                ("ncbi_sequence_names", "GCF_000001405.25"): (200, GRCH37_NAMES),
                ("ncbi_spdi_rsids", "NC_000019.9:44908683:T:C"): (404, SPDI_UNLISTED),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 44908684, "T", "C"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(
            sources.asked,
            [
                ("fetch_assembly_report", GRCH38),
                ("ncbi_sequence_names", GRCH38),
                ("ncbi_reference_span", "NC_000019.10:44908683:1"),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"),
                ("fetch_refsnp", "rs429358"),
                ("ncbi_sequence_names", "GCF_000001405.25"),
                ("ncbi_spdi_rsids", "NC_000019.9:44908683:T:C"),
            ],
        )
        self.assertEqual(
            (result.status, result.requested_rsid, result.verified_rsid),
            ("resolved", None, "rs429358"),
        )
        (listing,) = result.location.version_listings
        self.assertEqual(
            (listing.assembly_name, listing.contig, listing.listed_rsids),
            ("GRCh37.p13", "NC_000019.9", ()),
        )
        self.assertEqual(
            _candidates(result),
            [("GRCh38.p14", "NC_000019.10", 44908684, "T", "C", ("rs429358",))],
        )
        location = result.location
        self.assertEqual(
            (
                location.run_contig,
                location.reference_sequence,
                location.reference_bases,
                location.listed_rsids,
                location.run_assembly_name,
                location.run_assembly_synonym,
            ),
            ("NC_000019.10", "NC_000019.10", "T", ("rs429358",), "GRCh38.p14", "hg38"),
        )
        self.assertEqual(result.source.operation, "fetch_refsnp")
        self.assertEqual(
            [s.operation for s in location.supporting_sources],
            [
                "fetch_assembly_report",
                "ncbi_sequence_names",
                "ncbi_reference_span",
                "ncbi_spdi_rsids",
            ],
        )

    def test_a_position_no_source_lists_is_the_stated_allele_unlisted(self) -> None:
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:5"): (
                    200,
                    SPAN_NC_000019_10_TGCGG,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:TGCGG:A"): (
                    404,
                    SPDI_UNLISTED,
                ),
            }
        )

        result = look_up_stated_position(
            **_request(
                "19",
                44908684,
                "TGCGG",
                "A",
                assembly=StatedAssembly(name="GRCh38.p14", cited_line="Methods"),
            ),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "unlisted")
        self.assertIsNone(result.verified_rsid)
        # The one source asked lists none, not every source: EVA was never
        # asked.
        self.assertEqual(
            result.assessment.rationale,
            "dbSNP lists no variant at the stated position; the Run's reference holds "
            "the stated REF.",
        )
        # The stated allele on the Run's name, trimmed of its shared bases.
        self.assertEqual(
            _candidates(result),
            [("GRCh38.p14", "NC_000019.10", 44908684, "TGCGG", "A", ())],
        )
        self.assertEqual(result.location.reference_bases, "TGCGG")
        self.assertEqual(result.location.listed_rsids, ())
        self.assertEqual(result.source.operation, "ncbi_spdi_rsids")
        self.assertNotIn("fetch_refsnp", [operation for operation, _ in sources.asked])

    def test_a_placed_target_is_looked_up_at_its_placed_position(self) -> None:
        # Placement moved hg19's 19:45411941 to GRCh38's NC_000019.10:44908684. The
        # lookup reads the placed position and records both, the stated one as stated
        # on hg19; without its placement the hg19 position is incompatible.
        stated = StatedPosition(contig="19", position=45411941, ref="TGCGG", alt="A")
        placed = PlacedPosition(
            contig="NC_000019.10", position=44908684, ref="TGCGG", alt="A"
        )
        hg19 = StatedAssembly(name="hg19", cited_line="aligned to hg19")
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:5"): (
                    200,
                    SPAN_NC_000019_10_TGCGG,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:TGCGG:A"): (
                    404,
                    SPDI_UNLISTED,
                ),
            }
        )

        result = look_up_stated_position(
            position=stated,
            placed=placed,
            stated_assembly=hg19,
            rsid=None,
            requested_rsid=None,
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "unlisted")
        self.assertEqual(
            _candidates(result),
            [("GRCh38.p14", "NC_000019.10", 44908684, "TGCGG", "A", ())],
        )
        self.assertEqual(
            (
                result.location.stated,
                result.location.placed,
                result.location.stated_assembly,
            ),
            (stated, placed, hg19),
        )

    def test_a_region_is_named_on_the_runs_assembly_with_its_reference_bases(self):
        # A recorded Run's NC_012920.1:300-312 (its 302 and 310 records). The span
        # body is synthetic, in the retained span's shape.
        bases = "AAACCCCCCCTCC"
        span = (
            '{"data":{"seq_id":"NC_012920.1","position":299,'
            f'"deleted_sequence":"{bases}","inserted_sequence":"{"N" * 13}"}}}}'
        ).encode()
        sources = _human({("ncbi_reference_span", "NC_012920.1:299:13"): (200, span)})
        region = StatedRegion(contig="MT", start=300, end=312)

        result = look_up_region(
            region=region, assembly_accession=GRCH38, fetch=sources, max_bases=10_000
        )

        self.assertEqual(
            (result.status, result.candidates, result.verified_rsid),
            ("region_located", (), None),
        )
        # A region lists no rsID: it names no variant source and reads NCBI alone.
        self.assertEqual(result.lookup_source, "ncbi")
        self.assertEqual(
            (
                result.location.region,
                result.location.run_contig,
                result.location.reference_bases,
                result.location.stated,
            ),
            (region, "NC_012920.1", bases, None),
        )
        self.assertEqual(
            [operation for operation, _ in sources.asked],
            ["fetch_assembly_report", "ncbi_sequence_names", "ncbi_reference_span"],
        )
        for region, status, reason in (
            # No record holds chr99 and the model named none.
            (
                StatedRegion(contig="chr99", start=1, end=2),
                "sequence_unnamed",
                f"No sequence is named for the written contig 'chr99' on {GRCH38}.",
            ),
            (
                StatedRegion(contig="MT", start=16000, end=16600),
                "incompatible",
                "ends at 16569",
            ),
            (
                StatedRegion(contig="MT", start=1, end=10_001),
                "incompatible",
                "more than 10000 bases",
            ),
        ):
            with self.subTest(reason=reason):
                refused = look_up_region(
                    region=region,
                    assembly_accession=GRCH38,
                    fetch=_human(),
                    max_bases=10_000,
                )
                self.assertEqual(refused.status, status)
                self.assertIn(reason, refused.conflicts[0])

    def test_a_contig_no_record_holds_asks_the_model_to_name_its_sequence(self):
        """No code verdict on the written name (ADR 0049; owner #467 D1 a): with no
        named sequence the lookup asks the model to name one, after reading the names.
        The fact states only the observation (#478); the asking is the Tool's model
        view's next step."""
        sources = _human()

        result = look_up_stated_position(
            **_request("chr99", 10, "A", "C"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "sequence_unnamed")
        self.assertEqual(
            result.conflicts,
            (f"No sequence is named for the written contig 'chr99' on {GRCH38}.",),
        )
        self.assertEqual(
            result.origin, SequenceUnnamed(contig="chr99", assemblies=(GRCH38,))
        )
        view = _lookup_view(result)
        self.assertEqual(view["statement"], result.conflicts[0])
        self.assertIn("search_sequence_names", view["next"])
        self.assertIsNone(result.location.run_contig)
        self.assertEqual(len(sources.asked), 2)

    def test_a_position_past_its_sequence_end_is_incompatible(self):
        result = look_up_stated_position(
            **_request("MT", 16570, "A", "C"),
            assembly_accession=GRCH38,
            fetch=_human(),
        )

        self.assertEqual(result.status, "incompatible")
        self.assertEqual(
            result.conflicts,
            ("NC_012920.1 ends at 16569; the request states 16570.",),
        )

    def test_the_runs_own_version_by_any_of_its_names_is_placed(self) -> None:
        """The Run's version by one of its own names, literally, its paired GenBank
        accession among them, is the Run's with no model turn. Any other spelling
        (no patch suffix, another case) is the model's to name: the lookup asks
        (``version_unnamed``), and the model's naming of the Run's assembly places it,
        recorded ``model_matched`` (#471 Q3 a; #417's strip and folding superseded)."""

        def look_up(name, version=None):
            sources = _human(
                {
                    ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                        200,
                        SPAN_NC_000019_10_T,
                    ),
                    ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (
                        200,
                        SPDI_RS429358,
                    ),
                    ("fetch_refsnp", "rs429358"): (200, RS429358),
                }
            )
            return look_up_stated_position(
                **_request(
                    "chr19",
                    44908684,
                    "T",
                    "C",
                    assembly=StatedAssembly(name=name, cited_line="Methods"),
                ),
                assembly_accession=GRCH38,
                fetch=sources,
                version=version,
            )

        for name in ("hg38", "GRCh38.p14", GRCH38, "GCA_000001405.29"):
            with self.subTest(literal=name):
                result = look_up(name)
                self.assertEqual(result.status, "resolved")
                self.assertEqual(result.location.version_naming.match, "literal")
        for name in ("GRCh38", "grch38", GRCH38.lower()):
            with self.subTest(named=name):
                self.assertEqual(look_up(name).status, "version_unnamed")
                named = VersionNaming(
                    written=name,
                    assembly_accession=GRCH38,
                    quote=name,
                    match="model_matched",
                )
                result = look_up(name, named)
                self.assertEqual(result.status, "resolved")
                self.assertEqual(result.location.version_naming, named)

    def test_a_position_placed_by_the_runs_pair_passes_the_intent_check(self) -> None:
        # A finding's intent check reads the version the lookup named: the Run's own
        # (its paired accession, literally, names it), never names matched again; a
        # naming of another version is a difference.
        from bio_neb.investigation.investigation import _intent_difference

        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (200, SPDI_RS429358),
                ("fetch_refsnp", "rs429358"): (200, RS429358),
            }
        )
        stated = StatedAssembly(name="GCA_000001405.29", cited_line="Methods")
        result = look_up_stated_position(
            **_request("chr19", 44908684, "T", "C", assembly=stated),
            assembly_accession=GRCH38,
            fetch=sources,
        )
        self.assertEqual(result.status, "resolved")
        self.assertEqual(
            result.location.run_assembly_paired_accession, "GCA_000001405.29"
        )
        self.assertIsNone(_intent_difference(result, ()))
        self.assertEqual(
            (
                result.location.version_naming.assembly_accession,
                result.location.version_naming.match,
            ),
            (GRCH38, "literal"),
        )
        other = result.model_copy(
            update={
                "location": result.location.model_copy(
                    update={
                        "version_naming": result.location.version_naming.model_copy(
                            update={"assembly_accession": "GCA_000001405.28"}
                        )
                    }
                )
            }
        )
        self.assertIsNotNone(_intent_difference(other, ()))

    def test_more_than_one_listed_rsid_is_ambiguous_and_none_is_read(self) -> None:
        # Synthetic: the service's answer shape with two rsIDs.
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (
                    200,
                    b'{"data":{"rsids":[429358,7412]}}',
                ),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 44908684, "T", "C"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.location.listed_rsids, ("rs429358", "rs7412"))
        self.assertNotIn("fetch_refsnp", [operation for operation, _ in sources.asked])


# GRCh38's retained report with its organism, in the report's shape (synthetic: the
# retained body keeps only the fields the lookup read).
GRCH38_REPORT_TAXON = GRCH38_REPORT.replace(
    b'"assembly_info"', b'"organism":{"tax_id":9606},"assembly_info"'
)
PLACEMENT_NCBI = Path(__file__).parent / "fixtures" / "paper_targets" / "ncbi"
GRCH37 = "GCF_000001405.25"
APOE_HG19_SPDI = "NC_000019.9:45411940:T:C"
HG19 = StatedAssembly(name="hg19", cited_line="Also check chr19:45411941 T>C on hg19.")
# The model's checked naming of hg19 as GRCh37.p13: NCBI's search does not cover a
# UCSC name, so the match is the model's.
HG19_NAMED = VersionNaming(
    written="hg19", assembly_accession=GRCH37, quote="hg19", match="model_matched"
)


def _hg19(extra=None) -> _Sources:
    """The GETs placement makes for an hg19 position on GRCh38.p14 once the model named
    hg19 as GRCh37.p13 (#471): #393's development GETs (GRCh37's sequence report, the
    SPDI's equivalents); code lists no taxon's assemblies."""
    return _Sources(
        {
            ("fetch_assembly_report", GRCH38): (200, GRCH38_REPORT_TAXON),
            ("ncbi_sequence_names", GRCH38): (200, GRCH38_NAMES),
            ("ncbi_sequence_names", GRCH37): (
                200,
                assembly_report_text(
                    (
                        PLACEMENT_NCBI / "gcf-000001405.25-sequence-report.json"
                    ).read_bytes()
                ),
            ),
            ("ncbi_spdi_equivalents", APOE_HG19_SPDI): (
                200,
                (
                    PLACEMENT_NCBI / "spdi-NC_000019.9-45411940-T-C-equivalents.json"
                ).read_bytes(),
            ),
            **(extra or {}),
        }
    )


class StatedVersionPlacementTests(unittest.TestCase):
    """A position stated on another version is placed on the Run's assembly inside the
    lookup (#462 R2, owner 2026-10-26): NCBI names the stated sequence, its Variation
    service gives the stated allele one equivalent on the Run's sequences, and the moved
    position goes back through the same lookup, as a researcher's amendment does. A dead
    end ends with its reason and asks the item's issue Review (R1 a); a position with
    neither alleles nor an rsID is not looked up at all (owner #480 Q1)."""

    def test_an_hg19_position_is_moved_and_looked_up_on_the_runs_assembly(self):
        sources = _hg19(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (200, SPDI_RS429358),
                ("fetch_refsnp", "rs429358"): (200, RS429358),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 45411941, "T", "C", assembly=HG19),
            assembly_accession=GRCH38,
            fetch=sources,
            version=HG19_NAMED,
        )

        self.assertEqual(
            sources.asked,
            [
                ("fetch_assembly_report", GRCH38),
                ("ncbi_sequence_names", GRCH37),
                ("ncbi_spdi_equivalents", APOE_HG19_SPDI),
                ("ncbi_sequence_names", GRCH38),
                ("ncbi_reference_span", "NC_000019.10:44908683:1"),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"),
                ("fetch_refsnp", "rs429358"),
            ],
        )
        self.assertEqual(
            (result.status, result.verified_rsid), ("resolved", "rs429358")
        )
        self.assertEqual(
            (
                result.location.stated,
                result.location.placed,
                result.location.stated_assembly,
            ),
            (
                StatedPosition(contig="chr19", position=45411941, ref="T", alt="C"),
                PlacedPosition(
                    contig="NC_000019.10", position=44908684, ref="T", alt="C"
                ),
                HG19,
            ),
        )
        self.assertEqual(
            _candidates(result),
            [("GRCh38.p14", "NC_000019.10", 44908684, "T", "C", ("rs429358",))],
        )

    def test_a_version_no_assembly_is_named_for_asks_the_model_to_name_it(self):
        """A stated version that is not literally one of the Run's names and that the
        model has not named ends the lookup ``version_unnamed``, asking for the
        assembly it means; code lists no taxon's assemblies and refuses no listing
        (#471 Q3 a; #393 G1 A's RefSeq-only listing superseded, #375 B11). The fact
        states only the observation (#478); the asking is the Tool's model view's."""
        sources = _hg19()
        result = look_up_stated_position(
            **_request(
                "chr19",
                45411941,
                "T",
                "C",
                assembly=StatedAssembly(name="WBcel235", cited_line="on WBcel235"),
            ),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual((result.status, result.candidates), ("version_unnamed", ()))
        self.assertEqual(sources.asked, [("fetch_assembly_report", GRCH38)])
        self.assertEqual(
            result.conflicts,
            (
                "The stated version 'WBcel235' is not literally one of the Run's "
                "assembly's names (GRCh38.p14, GCF_000001405.40), and no assembly is "
                "named for it yet.",
            ),
        )
        self.assertEqual(
            result.origin,
            VersionUnnamed(
                version="WBcel235",
                assembly_name="GRCh38.p14",
                assembly_accession=GRCH38,
            ),
        )
        view = _lookup_view(result)
        self.assertEqual(view["statement"], result.conflicts[0])
        self.assertIn("search_assemblies", view["next"])
        self.assertNotIn("search_assemblies", result.conflicts[0])

    def test_a_position_without_alleles_or_rsid_is_not_looked_up(self):
        """Owner #480 Q1 (2026-10-29, "if we cant find it its out"; supersedes the
        dead end that asked the item's issue Review): no source lists rsIDs at a
        position without an allele, so one with neither alleles nor an rsID ends after
        the Run's assembly report, on its own version or another, and is deferred."""
        for name, request, sources, named in (
            (
                "another version",
                _request("chr19", 45411941, assembly=HG19),
                _hg19(),
                {"version": HG19_NAMED},
            ),
            (
                "the Run's version",
                _request(
                    "19",
                    44908822,
                    assembly=StatedAssembly(name="GRCh38", cited_line="on GRCh38."),
                ),
                _human(),
                {},
            ),
        ):
            with self.subTest(name):
                result = look_up_stated_position(
                    **request, assembly_accession=GRCH38, fetch=sources, **named
                )

                self.assertEqual(
                    (result.status, result.origin.code, result.candidates),
                    ("not_found", "position_without_alleles", ()),
                )
                self.assertEqual(sources.asked, [("fetch_assembly_report", GRCH38)])
                position = request["position"]
                self.assertEqual(
                    result.conflicts,
                    (
                        f"The request states {position.contig}:{position.position} "
                        "without its alleles and with no rsID; the Run's sources list "
                        "rsIDs only at a stated allele, so the position is not looked "
                        "up.",
                    ),
                )

    def test_a_source_answer_that_is_no_listing_is_the_reason_not_a_stop(self):
        """#458: a non-2xx answer with a body is the source's answer, never a stop; the
        lookup ends on it for the model to read, with no verdict (ADR 0049; owner #468
        R4 a)."""
        sources = _hg19(
            {("ncbi_spdi_equivalents", APOE_HG19_SPDI): (400, b'{"error":"bad"}')}
        )

        result = look_up_stated_position(
            **_request("chr19", 45411941, "T", "C", assembly=HG19),
            assembly_accession=GRCH38,
            fetch=sources,
            version=HG19_NAMED,
        )

        self.assertEqual((result.status, result.candidates), ("source_answered", ()))
        self.assertEqual(
            result.conflicts,
            (
                "NCBI's equivalents of the stated allele for NC_000019.9:45411940:T:C "
                'answered HTTP 400: {"error":"bad"}.',
            ),
        )
        self.assertEqual(
            (result.origin.code, result.origin.status_code), ("source_answered", 400)
        )
        self.assertEqual(result.assessment.rationale, result.conflicts[0])


class PlacementAnswerBoundaryTests(unittest.TestCase):
    """A placement answer its readers cannot use ends the lookup with its reason, so
    the member's issue Review asks (Standards r1 ST1, Spec r1 SP3); it never escapes
    as an exception. Each answer is synthetic and named as such."""

    def _placed(self, answer):
        return look_up_stated_position(
            **_request("chr19", 45411941, "T", "C", assembly=HG19),
            assembly_accession=GRCH38,
            fetch=_hg19(answer),
            version=HG19_NAMED,
        )

    def test_a_names_report_that_does_not_read_ends_the_lookup_malformed(self):
        """A report that is not ten columns a row is the reader's typing (#458)."""
        result = self._placed(
            {("ncbi_sequence_names", GRCH37): (200, b"19\tassembled-molecule\n")}
        )

        self.assertEqual((result.status, result.candidates), ("malformed", ()))
        self.assertIn("not ten columns", result.conflicts[0])

    def test_a_refseq_name_of_any_shape_is_copied_and_is_the_next_querys_key(self):
        """No shape judges a copied sequence name (ADR 0049; #467 retires #462's
        Standards r2/r3 ST1 rule, `SEQUENCE_ACCESSION`): the model-named record's RefSeq
        name, ``NZ_``-style or not, is the next GET's key, which the provider
        percent-encodes."""
        names = (
            b"# Sequence-Name\n19\tassembled-molecule\t19\tChromosome\tna\t=\t"
            b"NZ_X1.1\tPrimary Assembly\t59128983\tchr19\n"
        )
        key = "NZ_X1.1:45411940:T:C"
        sources = _hg19(
            {
                ("ncbi_sequence_names", GRCH37): (200, names),
                ("ncbi_spdi_equivalents", key): (400, b'{"error":"none"}'),
            }
        )
        result = look_up_stated_position(
            **_request("chr19", 45411941, "T", "C", assembly=HG19),
            assembly_accession=GRCH38,
            fetch=sources,
            version=HG19_NAMED,
        )

        self.assertIn(("ncbi_spdi_equivalents", key), sources.asked)
        self.assertEqual(result.status, "source_answered")
        self.assertIn(f"for {key} answered HTTP 400", result.conflicts[0])

    def test_an_equivalent_that_changes_nothing_is_no_substitution(self):
        body = (
            b'{"data":{"spdis":[{"seq_id":"NC_000019.10","position":44908683,'
            b'"deleted_sequence":"T","inserted_sequence":"T"}]}}'
        )
        result = self._placed({("ncbi_spdi_equivalents", APOE_HG19_SPDI): (200, body)})

        self.assertEqual((result.status, result.candidates), ("incompatible", ()))
        self.assertIn(
            "gives NC_000019.9:45411940:T:C no substitution on the Run's sequences "
            "(NC_000019.10:44908683:T:T)",
            result.conflicts[0],
        )


class ReferenceAndRsidPositionTests(unittest.TestCase):
    def test_a_stated_ref_the_runs_reference_does_not_hold_conflicts(self) -> None:
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 44908684, "G", "C"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "conflicted")
        self.assertEqual(
            result.conflicts,
            (
                "The Run's reference holds T at NC_000019.10:44908684; the request "
                "states G.",
            ),
        )
        # No allele is listed for a REF the reference does not hold.
        self.assertNotIn(
            "ncbi_spdi_rsids", [operation for operation, _ in sources.asked]
        )

    def test_an_rsid_stated_at_a_position_without_alleles_reads_that_rsid(self):
        sources = _human({("fetch_refsnp", "rs429358"): (200, RS429358)})

        result = look_up_stated_position(
            **_request("chr19", 44908000, rsid="rs429358"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(sources.asked[2:], [("fetch_refsnp", "rs429358")])
        self.assertEqual(
            (result.status, result.requested_rsid, result.verified_rsid),
            ("resolved", "rs429358", "rs429358"),
        )
        # The record's own place; the stated one is the intent check's to compare.
        self.assertEqual(
            _candidates(result),
            [("GRCh38.p14", "NC_000019.10", 44908684, "T", "C", ("rs429358",))],
        )
        self.assertEqual(result.location.run_contig, "NC_000019.10")


class ReviewCorrectionTests(unittest.TestCase):
    """Standards r1 (S2-S4): every REF is read, a merge is followed in the same
    lookup, and a malformed report is typed where it is read."""

    def test_a_genbank_only_sequence_reads_its_ref_on_its_genbank_name(self):
        # Synthetic, in the retained report's shape: a GCA Run whose sequence has
        # no RefSeq twin.
        molecules = (
            b'{"reports":[{"assembly_accession":"GCA_000003025.6","chr_name":"12",'
            b'"ucsc_style_name":"chr12","genbank_accession":"CM000823.5",'
            b'"role":"assembled-molecule","sequence_name":"Chr12",'
            b'"length":61602749}],"total_count":1}'
        )
        report = (
            b'{"reports":[{"accession":"GCA_000003025.6","assembly_info":'
            b'{"assembly_name":"Sscrofa11.1","assembly_status":"current"}}],'
            b'"total_count":1}'
        )
        span = (
            b'{"data":{"seq_id":"CM000823.5","position":3556400,'
            b'"deleted_sequence":"G","inserted_sequence":"N"}}'
        )
        sources = _Sources(
            {
                ("fetch_assembly_report", "GCA_000003025.6"): (200, report),
                ("ncbi_sequence_names", "GCA_000003025.6"): (
                    200,
                    assembly_report_text(molecules),
                ),
                ("ncbi_reference_span", "CM000823.5:3556400:1"): (200, span),
            }
        )

        result = look_up_stated_position(
            **_request("chr12", 3556401, "A", "C"),
            assembly_accession="GCA_000003025.6",
            fetch=sources,
        )

        self.assertEqual(result.status, "conflicted")
        self.assertEqual(result.location.reference_bases, "G")
        self.assertEqual(
            result.conflicts,
            (
                "The Run's reference holds G at CM000823.5:3556401; the request "
                "states A.",
            ),
        )

    def test_a_merged_stated_rsid_is_followed_in_the_same_lookup(self) -> None:
        sources = _human(
            {
                ("fetch_refsnp", "rs60037639"): (200, RS60037639),
                ("fetch_refsnp", "rs4149056"): (200, RS4149056),
            }
        )

        result = look_up_stated_position(
            **_request("chr12", 21178615, rsid="rs60037639"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(
            sources.asked[2:],
            [("fetch_refsnp", "rs60037639"), ("fetch_refsnp", "rs4149056")],
        )
        self.assertEqual(
            (result.status, result.requested_rsid, result.verified_rsid),
            ("resolved", "rs60037639", "rs4149056"),
        )
        self.assertIn("rs60037639", result.merged_rsids)
        self.assertEqual(
            [s.operation for s in result.location.supporting_sources][-1],
            "fetch_refsnp",
        )

    def test_a_malformed_report_shape_is_typed_and_ends_the_lookup(self) -> None:
        for operation, body in (
            ("fetch_assembly_report", b'{"reports":1}'),
            # A row that is not ten columns, and bytes that are not text.
            ("ncbi_sequence_names", b"19\tassembled-molecule\t19\n"),
            ("ncbi_sequence_names", b"\xff\xfe19\n"),
        ):
            with self.subTest(operation=operation, body=body):
                sources = _human({(operation, GRCH38): (200, body)})

                result = look_up_stated_position(
                    **_request("chr19", 44908684, "T", "C"),
                    assembly_accession=GRCH38,
                    fetch=sources,
                )

                self.assertEqual(result.status, "malformed")
                self.assertEqual(sources.asked[-1], (operation, GRCH38))


class ListingAnswerTests(unittest.TestCase):
    """A listing service's 404 is its own answer, with no fixed meaning (ADR 0049): the
    lookup ends on it for the model to read, and goes on to an empty listing only when
    the model quotes that answer saying so, checked by membership in it (owner #468
    R5 b)."""

    def _unlisted(self, **call):
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:5"): (
                    200,
                    SPAN_NC_000019_10_TGCGG,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:TGCGG:A"): (
                    404,
                    SPDI_UNLISTED,
                ),
            }
        )
        return _look_up_stated_position(
            **_scripted_naming(
                {
                    **_request(
                        "19",
                        44908684,
                        "TGCGG",
                        "A",
                        assembly=StatedAssembly(
                            name="GRCh38.p14", cited_line="Methods"
                        ),
                    ),
                    "assembly_accession": GRCH38,
                    "fetch": sources,
                    "source": "ncbi_dbsnp",
                    **call,
                },
                "19",
            )
        )

    def test_a_listing_404_with_no_quote_ends_on_the_sources_answer(self) -> None:
        result = self._unlisted()

        self.assertEqual((result.status, result.candidates), ("source_answered", ()))
        self.assertEqual(result.source.operation, "ncbi_spdi_rsids")
        self.assertEqual(result.source.status_code, 404)
        # The fact states NCBI's own answer; the model view's next step, not the
        # fact, names the `listed_none` call.
        self.assertEqual(
            result.conflicts,
            (
                "NCBI's rsIDs at the stated allele answered HTTP 404: "
                + SPDI_UNLISTED.decode()
                + ".",
            ),
        )
        self.assertIn("does not match any RSID", result.conflicts[0])
        self.assertNotIn("listed_none", result.conflicts[0])
        view = _lookup_view(result)
        self.assertEqual(view["statement"], result.conflicts[0])
        self.assertIn("`quote_listed_none`", view["next"])
        self.assertIsNone(result.location.listed_none)

    def test_a_quote_the_answer_holds_reads_the_listing_as_empty(self) -> None:
        # The model may quote the answer's text or one of its JSON strings.
        for quote in (
            "does not match any RSID",
            "SPDI 'NC_000019.10:44908683:TGCGG:A'",
        ):
            with self.subTest(quote=quote):
                result = self._unlisted(listed_none=quote)

                self.assertEqual(result.status, "unlisted")
                self.assertEqual(result.location.listed_none, quote)
                self.assertEqual(result.location.listed_rsids, ())

    def test_an_empty_option_the_call_sent_reopens_as_the_lookup_recorded_it(self):
        # The lookup records an empty listed_none or variant_type as not sent; its
        # receipt must reopen from the same call, which admission records as called,
        # the blank absent.
        from bio_neb.agentic_journey.action_policy import effective_arguments
        from bio_neb.investigation.investigation import _named_as_called

        for option in ("listed_none", "variant_type"):
            with self.subTest(option=option):
                call = {
                    "source": "ncbi_dbsnp",
                    "sequences": [
                        {
                            "assembly_accession": GRCH38,
                            "record": "NC_000019.10",
                            "quote": "19",
                        }
                    ],
                    option: "",
                }
                result = self._unlisted(**{option: ""})

                self.assertIsNone(getattr(result.location, option))
                self.assertTrue(_named_as_called(effective_arguments(call, {}), result))

    def test_an_empty_placement_the_call_sent_binds_the_lookups_recorded_result(self):
        # The lookup records an empty placement as none named, as admission records
        # the call; a different placement still does not bind.
        from bio_neb.agentic_journey.action_policy import effective_arguments
        from bio_neb.agentic_journey.source_answers import AnsweredGet, shown_answers
        from bio_neb.investigation.investigation import (
            RefSnpToolResultRecorded,
            _options_as_called,
        )

        result = self._unlisted(placement="")
        fact = RefSnpToolResultRecorded(
            attempt_id="attempt",
            source_event_id="source",
            collection_member_id="member",
            tool_call_id="call",
            query="19:44908684",
            result=result,
            source_answers=shown_answers(
                (
                    AnsweredGet(
                        operation=result.source.operation,
                        request_identity=result.source.request_identity,
                        status_code=result.source.status_code,
                        body=b"",
                    ),
                )
            ),
        )

        self.assertIsNone(result.named_placement)
        self.assertTrue(
            _options_as_called(effective_arguments({"placement": ""}, {}), fact)
        )
        self.assertFalse(_options_as_called({"placement": "NC_000019.10"}, fact))

    def _bodyless(self, **call):
        """EVA's listing answered HTTP 404 with no body, as at C. elegans a4's
        X:15696111 and III:4821044 (Journey `51399e20`, 0-byte answers); the pig
        position stands in for the worm's, whose report bodies are not retained here."""
        sources = _Sources(
            {
                ("fetch_assembly_report", SSCROFA11): (200, SSCROFA11_REPORT),
                ("ncbi_sequence_names", SSCROFA11): (200, SSCROFA11_NAMES),
                ("ncbi_reference_span", "NC_010454.4:3556400:14"): (
                    200,
                    SPAN_NC_010454_4,
                ),
                (
                    "eva_clustered_variants_at",
                    "GCA_000003025.6:CM000823.5:3556402:DEL",
                ): (404, b""),
            }
        )
        return look_up_stated_position(
            **_request(
                "chr12",
                3556401,
                "GGTGTCCCCCTCGC",
                "G",
                assembly=StatedAssembly(name="Sscrofa11.1", cited_line="Methods"),
            ),
            assembly_accession=SSCROFA11,
            fetch=sources,
            **call,
        )

    def test_a_bodyless_404_with_no_decision_ends_on_its_status(self) -> None:
        result = self._bodyless()

        self.assertEqual((result.status, result.candidates), ("source_answered", ()))
        self.assertEqual(result.origin.status_code, 404)
        self.assertEqual(result.origin.words_sent, "")

    def test_a_decision_citing_a_bodyless_404_reads_the_listing_as_empty(self):
        # The decision cites the answer's status with no body; the result is
        # unlisted, and its audit says what that answer does not establish.
        result = self._bodyless(listed_none=BodylessAnswer(status_code=404))

        self.assertEqual(result.status, "unlisted")
        self.assertEqual(result.location.listed_none, BodylessAnswer(status_code=404))
        self.assertEqual(result.location.listed_rsids, ())
        self.assertEqual(
            _candidates(result),
            [("Sscrofa11.1", "NC_010454.4", 3556401, "GGTGTCCCCCTCGC", "G", ())],
        )
        self.assertEqual(
            result.assessment.rationale,
            "EVA's variants at the stated position answered HTTP 404 with no body, "
            "read as no rsID listed there; a body-less 404 does not establish that "
            "no variant exists at the position. The Run's reference holds the "
            "stated REF.",
        )

    def test_a_bodyless_404_at_an_insertion_keeps_its_status_and_limit(self):
        # Spec r1 R1-SP2: an insertion's empty REF reads no reference base, and the
        # audit still names the body-less answer and what it does not establish.
        sources = _human({("ncbi_spdi_rsids", "NC_000019.10:44908683::A"): (404, b"")})
        result = _look_up_stated_position(
            **_scripted_naming(
                {
                    **_request(
                        "19",
                        44908684,
                        "",
                        "A",
                        assembly=StatedAssembly(
                            name="GRCh38.p14", cited_line="Methods"
                        ),
                    ),
                    "assembly_accession": GRCH38,
                    "fetch": sources,
                    "source": "ncbi_dbsnp",
                    "listed_none": BodylessAnswer(status_code=404),
                },
                "19",
            )
        )

        self.assertEqual(result.status, "unlisted")
        self.assertEqual(
            result.assessment.rationale,
            "NCBI's rsIDs at the stated allele answered HTTP 404 with no body, read "
            "as no rsID listed there; a body-less 404 does not establish that no "
            "variant exists at the position. An insertion's empty REF has no "
            "reference base to read.",
        )

    def test_the_readable_report_says_what_a_bodyless_404_does_not_establish(self):
        # The explainer's row reads the unlisted result's typed words, and the
        # page shows the source's own answer, an empty body, never "could not be
        # placed".
        from bio_neb.investigation.collection_view import _lookup
        from bio_neb.reporting.readable_report import source_answer_cell

        result = self._bodyless(listed_none=BodylessAnswer(status_code=404))
        line = source_answer_cell(_lookup(result), b"")
        self.assertEqual(
            line, "EVA `eva_clustered_variants_at` HTTP 404 (not found), empty body"
        )
        self.assertNotIn("could not be placed", line)

    def test_a_status_decision_another_answer_does_not_match_is_not_read(self):
        # The decision holds only for the answer it cites: another status, or a
        # body, leaves the lookup on the source's answer.
        result = self._bodyless(listed_none=BodylessAnswer(status_code=410))
        self.assertEqual(result.status, "source_answered")

        result = self._unlisted(listed_none=BodylessAnswer(status_code=404))
        self.assertEqual(result.status, "source_answered")

    def test_a_whitespace_body_is_not_a_bodyless_answer(self) -> None:
        """Owner #480 Q3 a limits a status-only decision to an answer with zero body
        bytes (review r1 ST1): whitespace is a body, so the lookup stays on the
        source's answer, admission refuses the status, and the one predicate the
        decision's append and reopen check share says no."""
        from types import SimpleNamespace

        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.investigation.investigation import answered_without_body
        from bio_neb.investigation.investigation_agent import _decision_errors

        def answered(body, **call):
            sources = _human(
                {
                    ("ncbi_reference_span", "NC_000019.10:44908683:5"): (
                        200,
                        SPAN_NC_000019_10_TGCGG,
                    ),
                    ("ncbi_spdi_rsids", "NC_000019.10:44908683:TGCGG:A"): (404, body),
                }
            )
            return _look_up_stated_position(
                **_scripted_naming(
                    {
                        **_request(
                            "19",
                            44908684,
                            "TGCGG",
                            "A",
                            assembly=StatedAssembly(
                                name="GRCh38.p14", cited_line="Methods"
                            ),
                        ),
                        "assembly_accession": GRCH38,
                        "fetch": sources,
                        "source": "ncbi_dbsnp",
                        **call,
                    },
                    "19",
                )
            )

        empty, spaces = answered(b""), answered(b" \n\t")
        self.assertEqual(spaces.origin.words_sent, "")
        self.assertTrue(answered_without_body(empty, 404))
        self.assertFalse(answered_without_body(spaces, 404))
        self.assertEqual(
            answered(b" \n\t", listed_none=BodylessAnswer(status_code=404)).status,
            "source_answered",
        )
        self.assertEqual(
            answered(b"", listed_none=BodylessAnswer(status_code=404)).status,
            "unlisted",
        )
        lookup = {
            "event_id": "lookup-1",
            "event_type": "RefSnpToolResultRecorded",
            "payload": {"result": spaces.model_dump(mode="json")},
        }
        self.assertEqual(
            _decision_errors(
                SimpleNamespace(source_events=[lookup], replaced_result_ids=()),
                ProposedAgentAction(
                    "quote_listed_none",
                    "call-1",
                    {"refsnp_event_id": "lookup-1", "status_code": 404},
                ),
            ),
            ("listed_none_status_not_bodyless",),
        )

    def test_a_bodyless_other_version_listing_after_the_record_is_decided(self):
        """Spec r2 R2-SP1 (owner #480 Q3 a): with no version stated, another version's
        listing follows the record read; its body-less 404 ends the lookup on that
        following GET, which the status decision cites, not the record."""
        from types import SimpleNamespace

        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.investigation.investigation import answered_without_body
        from bio_neb.investigation.investigation_agent import _decision_errors

        def looked_up(**call):
            sources = _human(
                {
                    ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                        200,
                        SPAN_NC_000019_10_T,
                    ),
                    ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (
                        200,
                        SPDI_RS429358,
                    ),
                    ("fetch_refsnp", "rs429358"): (200, RS429358),
                    ("ncbi_sequence_names", "GCF_000001405.25"): (200, GRCH37_NAMES),
                    ("ncbi_spdi_rsids", "NC_000019.9:44908683:T:C"): (404, b""),
                }
            )
            return look_up_stated_position(
                **_request("chr19", 44908684, "T", "C"),
                assembly_accession=GRCH38,
                fetch=sources,
                **call,
            )

        ended = looked_up()
        self.assertEqual(ended.status, "source_answered")
        self.assertEqual(
            (ended.source.operation, ended.source.status_code), ("fetch_refsnp", 200)
        )
        self.assertTrue(answered_without_body(ended, 404))
        lookup = {
            "event_id": "lookup-1",
            "event_type": "RefSnpToolResultRecorded",
            "payload": {"result": ended.model_dump(mode="json")},
        }
        self.assertEqual(
            _decision_errors(
                SimpleNamespace(source_events=[lookup], replaced_result_ids=()),
                ProposedAgentAction(
                    "quote_listed_none",
                    "call-1",
                    {"refsnp_event_id": "lookup-1", "status_code": 404},
                ),
            ),
            (),
        )
        self.assertEqual(
            looked_up(listed_none=BodylessAnswer(status_code=404)).status, "resolved"
        )

    def test_a_listed_none_decision_is_admitted_only_on_its_basis(self) -> None:
        # A status is cited only for an answer with no body, and an answer with
        # words is quoted; one basis, never both or neither.
        from types import SimpleNamespace

        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.investigation.investigation_agent import _decision_errors

        def errors(result, **arguments):
            lookup = {
                "event_id": "lookup-1",
                "event_type": "RefSnpToolResultRecorded",
                "payload": {"result": result.model_dump(mode="json")},
            }
            deps = SimpleNamespace(source_events=[lookup], replaced_result_ids=())
            return _decision_errors(
                deps,
                ProposedAgentAction(
                    "quote_listed_none",
                    "call-1",
                    {"refsnp_event_id": "lookup-1", **arguments},
                ),
            )

        bodyless, worded = self._bodyless(), self._unlisted()
        self.assertEqual(errors(bodyless, status_code=404), ())
        self.assertEqual(
            errors(bodyless, status_code=410), ("listed_none_status_not_bodyless",)
        )
        self.assertEqual(
            errors(worded, status_code=404), ("listed_none_status_not_bodyless",)
        )
        self.assertEqual(errors(worded, quote="does not match any RSID"), ())
        self.assertEqual(errors(bodyless), ("listed_none_basis_missing",))
        self.assertEqual(
            errors(worded, quote="does not match any RSID", status_code=404),
            ("listed_none_basis_missing",),
        )

    def test_a_quote_the_answer_does_not_hold_is_refused(self) -> None:
        result = self._unlisted(listed_none="no variant is known here")

        self.assertEqual((result.status, result.candidates), ("source_answered", ()))
        self.assertIn(
            "The quote 'no variant is known here' is not in", result.conflicts[0]
        )


class RegionAnswerTests(unittest.TestCase):
    def test_a_regions_non_2xx_reference_answer_is_ncbis_to_read(self) -> None:
        """Every source GET of locate_variant, a region's among them, ends on a non-2xx
        answer for the model to read, never malformed (owner #468 R4 a; Spec r1 SP2)."""
        for status in (301, 400, 404, 410):
            with self.subTest(status=status):
                sources = _human(
                    {
                        ("ncbi_reference_span", "NC_000019.10:44908683:3"): (
                            status,
                            b'{"error":"the source explains its refusal"}',
                        ),
                    }
                )

                result = look_up_region(
                    region=StatedRegion(contig="chr19", start=44908684, end=44908686),
                    stated_assembly=None,
                    assembly_accession=GRCH38,
                    fetch=sources,
                    max_bases=1000,
                )

                self.assertEqual(
                    (result.status, result.candidates), ("source_answered", ())
                )
                self.assertEqual(result.source.status_code, status)
                self.assertIn(f"HTTP {status}", result.conflicts[0])


class UnstatedVersionTests(unittest.TestCase):
    """Spec r1 S2, the owner's rule for a target with no stated version: it is
    checked on the Run's version, and a position no source confirms there asks."""

    def test_an_unlisted_position_with_no_stated_version_asks(self) -> None:
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:5"): (
                    200,
                    SPAN_NC_000019_10_TGCGG,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:TGCGG:A"): (
                    404,
                    SPDI_UNLISTED,
                ),
            }
        )

        result = look_up_stated_position(
            **_request("19", 44908684, "TGCGG", "A"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.candidates, ())
        self.assertEqual(
            result.conflicts,
            (
                "The request states no genome version, and dbSNP lists no variant "
                "at NC_000019.10:44908684 on the Run's GRCh38.p14; the Run's reference "
                "holds its REF there.",
            ),
        )


class ReviewRoundTwoTests(unittest.TestCase):
    """Standards r2 (S6) and Spec r2 (S2): a merge cycle ends typed, and a position
    no version is stated for names no other variant on another version."""

    def test_a_merge_cycle_is_malformed_with_no_repeated_get(self) -> None:
        def merged(number, into):
            return (
                b'{"refsnp_id":"' + number + b'","merged_snapshot_data":'
                b'{"merged_into":["' + into + b'"]}}'
            )

        class _Once(_Sources):
            def __call__(self, operation, key):
                if (operation, key) in self.asked:
                    raise AssertionError(f"repeated GET {operation} {key}")
                return super().__call__(operation, key)

        # Synthetic, in the retained merged record's shape: two records naming
        # each other as current.
        sources = _Once(
            {
                ("fetch_assembly_report", GRCH38): (200, GRCH38_REPORT),
                ("ncbi_sequence_names", GRCH38): (200, GRCH38_NAMES),
                ("fetch_refsnp", "rs11"): (200, merged(b"11", b"22")),
                ("fetch_refsnp", "rs22"): (200, merged(b"22", b"11")),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 44908000, rsid="rs11"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "malformed")
        self.assertEqual(
            result.conflicts,
            (
                "NCBI dbSNP's merge records could not be read: they form a cycle: "
                "rs11 -> rs22 -> rs11.",
            ),
        )
        # A typed ending.
        self.assertEqual(result.origin.code, "malformed")

    def test_an_unread_other_version_is_no_negative_evidence(self) -> None:
        # Spec r3 S6: only a 404 or a well-formed list is an answer. Two records
        # literally named chr19: which one is the model's to name, so the scripted
        # model names none and the lookup asks.
        two_19 = (
            b"19\tassembled-molecule\t19\tChromosome\tna\t=\tNC_000019.9\t"
            b"Primary Assembly\t59128983\tchr19\n"
            b"19b\tunplaced-scaffold\tna\tna\tna\t=\tNW_000019.1\t"
            b"Primary Assembly\t1000\tchr19\n"
        )
        cases = (
            ("listing {}", GRCH37_NAMES, (200, b"{}"), "malformed", "GRCh37.p13"),
            (
                "listing bad rsids",
                GRCH37_NAMES,
                (200, b'{"data":{"rsids":"bad"}}'),
                "malformed",
                "GRCh37.p13",
            ),
            (
                "report shape",
                b"19\tassembled-molecule\n",
                None,
                "malformed",
                "not ten columns",
            ),
            ("named twice", two_19, None, "sequence_unnamed", "GCF_000001405.25"),
        )
        for name, report, listing, status, cited in cases:
            with self.subTest(case=name):
                answers = {
                    ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                        200,
                        SPAN_NC_000019_10_T,
                    ),
                    ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (
                        200,
                        SPDI_RS429358,
                    ),
                    ("fetch_refsnp", "rs429358"): (200, RS429358),
                    ("ncbi_sequence_names", "GCF_000001405.25"): (200, report),
                }
                if listing is not None:
                    answers[("ncbi_spdi_rsids", "NC_000019.9:44908683:T:C")] = listing

                result = look_up_stated_position(
                    **_request("chr19", 44908684, "T", "C"),
                    assembly_accession=GRCH38,
                    fetch=_human(answers),
                )

                self.assertEqual(result.status, status)
                self.assertIn(cited, result.conflicts[0])

    def test_a_merge_record_of_another_rsid_authorizes_no_redirect(self) -> None:
        # Standards r3 S8: a GET for rs11 answered by rs99's merge record.
        class _NoRedirect(_Sources):
            def __call__(self, operation, key):
                if (operation, key) == ("fetch_refsnp", "rs22"):
                    raise AssertionError("a wrong-identity merge was followed")
                return super().__call__(operation, key)

        sources = _NoRedirect(
            {
                ("fetch_assembly_report", GRCH38): (200, GRCH38_REPORT),
                ("ncbi_sequence_names", GRCH38): (200, GRCH38_NAMES),
                ("fetch_refsnp", "rs11"): (
                    200,
                    b'{"refsnp_id":"99","merged_snapshot_data":{"merged_into":["22"]}}',
                ),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 44908000, rsid="rs11"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "conflicted")
        self.assertEqual(
            result.conflicts,
            ("NCBI RefSNP identity did not match the requested rsID.",),
        )

    def test_a_position_another_version_lists_another_variant_at_asks(self):
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
                ("ncbi_spdi_rsids", "NC_000019.10:44908683:T:C"): (200, SPDI_RS429358),
                ("fetch_refsnp", "rs429358"): (200, RS429358),
                ("ncbi_sequence_names", "GCF_000001405.25"): (200, GRCH37_NAMES),
                # Synthetic, in the service's shape: GRCh37 lists another rsID at
                # the same numbers.
                ("ncbi_spdi_rsids", "NC_000019.9:44908683:T:C"): (
                    200,
                    b'{"data":{"rsids":[12345]}}',
                ),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 44908684, "T", "C"),
            assembly_accession=GRCH38,
            fetch=sources,
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(
            result.conflicts,
            (
                "The request states no genome version, and its position names "
                "another variant on another version: GRCh37.p13 lists rs12345 at "
                "NC_000019.9:44908684; the Run's GRCh38.p14 lists rs429358.",
            ),
        )


class PositionPropertyTests(unittest.TestCase):
    """#346 D5's property check at this seam: the same flow over another locus,
    sequence name or allele shape gives the same kind of outcome."""

    def test_every_locus_name_and_allele_shape_is_placed_the_same_way(self):
        # (stated name, its RefSeq name in the retained report, position, REF, ALT)
        cases = (
            ("chr1", "NC_000001.11", 14397, "CTGT", "C"),  # deletion
            ("X", "NC_000023.11", 2781480, "G", "GA"),  # insertion
            ("chrM", "NC_012920.1", 310, "T", "C"),  # SNV, UCSC name
            ("MT", "NC_012920.1", 302, "AC", "GT"),  # MNV
            ("CM000681.2", "NC_000019.10", 44908684, "T", "C"),  # GenBank name
        )
        for name, refseq, position, ref, alt in cases:
            with self.subTest(name=name, ref=ref, alt=alt):
                span = f"{refseq}:{position - 1}:{len(ref)}"
                bases = (
                    b'{"data":{"seq_id":"'
                    + refseq.encode()
                    + b'","position":'
                    + str(position - 1).encode()
                    + b',"deleted_sequence":"'
                    + ref.encode()
                    + b'","inserted_sequence":"'
                    + b"N" * len(ref)
                    + b'"}}'
                )
                spdi = f"{refseq}:{position - 1}:{ref}:{alt}"
                sources = _human(
                    {
                        ("ncbi_reference_span", span): (200, bases),
                        ("ncbi_spdi_rsids", spdi): (404, SPDI_UNLISTED),
                    }
                )

                result = look_up_stated_position(
                    **_request(
                        name,
                        position,
                        ref,
                        alt,
                        assembly=StatedAssembly(name="hg38", cited_line="Methods"),
                    ),
                    assembly_accession=GRCH38,
                    fetch=sources,
                )

                self.assertEqual(result.status, "unlisted")
                self.assertEqual(result.location.run_contig, refseq)
                (candidate,) = result.candidates
                self.assertEqual(
                    candidate.target_identity,
                    StatedPosition(
                        contig=refseq, position=position, ref=ref, alt=alt
                    ).identity,
                )

    def test_a_ref_off_the_reference_conflicts_at_any_locus(self):
        for name, refseq, position in (
            ("chr1", "NC_000001.11", 14397),
            ("MT", "NC_012920.1", 310),
        ):
            with self.subTest(name=name):
                span = f"{refseq}:{position - 1}:1"
                bases = (
                    b'{"data":{"seq_id":"'
                    + refseq.encode()
                    + b'","position":'
                    + str(position - 1).encode()
                    + b',"deleted_sequence":"A","inserted_sequence":"N"}}'
                )
                result = look_up_stated_position(
                    **_request(name, position, "G", "T"),
                    assembly_accession=GRCH38,
                    fetch=_human({("ncbi_reference_span", span): (200, bases)}),
                )
                self.assertEqual(result.status, "conflicted")
                self.assertEqual(
                    result.conflicts,
                    (
                        f"The Run's reference holds A at {refseq}:{position}; the "
                        "request states G.",
                    ),
                )


class SourceChoiceTests(unittest.TestCase):
    """The variant source a position's rsIDs are listed by is the model's choice (#469,
    owner RC2 1 and Q1 a, ADR 0049): no code routes by assembly."""

    def test_a_human_position_asked_of_eva_is_listed_by_evas_own_query(self) -> None:
        # EVA lists no human variant at a human position (a live probe): that 404
        # is EVA's answer, for the model to read and ask dbSNP instead.
        listing = (
            "eva_clustered_variants_at",
            "GCA_000001405.29:CM000681.2:44908684:SNV",
        )
        sources = _human(
            {
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
                listing: (404, b""),
            }
        )

        result = look_up_stated_position(
            **_request("chr19", 44908684, "T", "C"),
            assembly_accession=GRCH38,
            fetch=sources,
            source="eva",
        )

        self.assertEqual(
            sources.asked,
            [
                ("fetch_assembly_report", GRCH38),
                ("ncbi_sequence_names", GRCH38),
                ("ncbi_reference_span", "NC_000019.10:44908683:1"),
                listing,
            ],
        )
        self.assertEqual(
            (result.status, result.lookup_source), ("source_answered", "eva")
        )
        # Reopen binds a position's call to the source it named, exactly (ADR 0038):
        # the same result under another source is a changed call (review r1 ST1).
        from bio_neb.investigation.investigation import _named_as_called

        named = {
            "sequences": [
                {
                    "assembly_accession": naming.assembly_accession,
                    "record": naming.record,
                    "quote": naming.quote,
                }
                for naming in result.location.sequence_namings
                if naming.quote is not None
            ]
        }
        self.assertTrue(_named_as_called({**named, "source": "eva"}, result))
        self.assertFalse(_named_as_called({**named, "source": "ncbi_dbsnp"}, result))
        self.assertFalse(_named_as_called(named, result))

    def test_the_models_variant_type_is_sent_to_eva_as_is_and_recorded(self) -> None:
        # EVA answers a type it does not file the variant under with an empty 404
        # (a live probe): the model's INDEL for a deletion EVA files as DEL.
        listing = (
            "eva_clustered_variants_at",
            "GCA_000003025.6:CM000823.5:3556402:INDEL",
        )
        sources = _Sources(
            {
                ("fetch_assembly_report", SSCROFA11): (200, SSCROFA11_REPORT),
                ("ncbi_sequence_names", SSCROFA11): (200, SSCROFA11_NAMES),
                ("ncbi_reference_span", "NC_010454.4:3556400:14"): (
                    200,
                    SPAN_NC_010454_4,
                ),
                listing: (404, b""),
            }
        )

        result = look_up_stated_position(
            **_request("chr12", 3556401, "GGTGTCCCCCTCGC", "G"),
            assembly_accession=SSCROFA11,
            fetch=sources,
            variant_type="INDEL",
        )

        self.assertEqual(sources.asked[-1], listing)
        self.assertEqual(result.status, "source_answered")
        self.assertEqual(result.location.variant_type, "INDEL")

    def test_a_copied_name_in_the_chosen_convention_reaches_eva_whole(self) -> None:
        """EVA's convention is the model's (#469, owner SP4 a): the listing sends the
        sequence's copied name in it as EVA's referenceName, whatever text NCBI's
        report writes (review r2 SP6: a name with a colon)."""
        result, urls = self._listing("GENBANK_SEQUENCE_NAME", name="chromosome:19")

        self.assertEqual(len(urls), 1)
        self.assertEqual(
            parse_qs(urlsplit(urls[0]).query)["referenceName"], ["chromosome:19"]
        )
        self.assertEqual(result.location.contig_naming, "GENBANK_SEQUENCE_NAME")

    def test_a_refseq_listing_needs_only_the_refseq_name(self) -> None:
        """Under REFSEQ the listing sends the RefSeq name; a missing GenBank name is
        not its reason to stop (#469 SP4 a, review r2 SP7)."""
        result, urls = self._listing("REFSEQ", remove_genbank=True)

        self.assertEqual(len(urls), 1, result.conflicts)
        self.assertEqual(
            parse_qs(urlsplit(urls[0]).query)["referenceName"], ["NC_000019.10"]
        )

    def _listing(self, naming, *, name=None, remove_genbank=False):
        # The model names NC_000019.10's record; its copied Sequence-Name or GenBank
        # column is changed here (synthetic rows the real report reader admits), and
        # EVA's listing goes through the real provider URL builder over a fake HTTP
        # transport that answers an empty list.
        rows = []
        for line in GRCH38_NAMES.decode().splitlines():
            columns = line.split("\t")
            if len(columns) == 10 and columns[6] == "NC_000019.10":
                if name is not None:
                    columns[0] = name
                if remove_genbank:
                    columns[4] = "na"
                line = "\t".join(columns)
            rows.append(line)
        sources = _human(
            {
                ("ncbi_sequence_names", GRCH38): (
                    200,
                    ("\n".join(rows) + "\n").encode(),
                ),
                ("ncbi_reference_span", "NC_000019.10:44908683:1"): (
                    200,
                    SPAN_NC_000019_10_T,
                ),
            }
        )
        urls = []

        class _Transport:
            contact_count = 0

            def get(self, url):
                urls.append(url)
                return InvestigationSourceResponse(
                    url, 200, b"[]", "2026-10-27T17:30:00Z", 1
                )

        adapter = EvaInvestigationSourceAdapter(_Transport())

        def fetch(operation, key):
            if operation != "eva_clustered_variants_at":
                return sources(operation, key)
            response = adapter.fetch_eva_clustered_variants_at(key)
            return RetrievedSource.create(
                operation=operation,
                request_identity=response.request_identity,
                status_code=response.status_code,
                observed_at=response.observed_at,
                elapsed_time_ms=response.elapsed_time_ms,
                raw_response=response.body,
            ), response.body

        fetch.answers = sources.answers
        result = look_up_stated_position(
            **_request(
                "NC_000019.10",
                44908684,
                "T",
                "C",
                assembly=StatedAssembly(name="GRCh38.p14", cited_line="Methods"),
            ),
            assembly_accession=GRCH38,
            source="eva",
            fetch=fetch,
            contig_naming=naming,
        )
        return result, urls


class EvaPositionTests(unittest.TestCase):
    def test_an_eva_position_with_no_stated_version_asks(self) -> None:
        # Spec r3 S2: EVA names no other version to check the position on, so with
        # none stated the Run's coordinates alone do not settle it.
        sources = _Sources(
            {
                ("fetch_assembly_report", SSCROFA11): (200, SSCROFA11_REPORT),
                ("ncbi_sequence_names", SSCROFA11): (200, SSCROFA11_NAMES),
                ("ncbi_reference_span", "NC_010454.4:3556400:14"): (
                    200,
                    SPAN_NC_010454_4,
                ),
                (
                    "eva_clustered_variants_at",
                    "GCA_000003025.6:CM000823.5:3556402:DEL",
                ): (200, EVA_AT_PIG_DEL),
                ("eva_clustered_variant", "rs5334475172"): (200, PIG_RS5334475172),
                ("eva_contig_alias", "CM000823.5"): (200, PIG_CM000823_5_ALIAS),
                ("eva_submitted_variants", "rs5334475172"): (
                    200,
                    PIG_RS5334475172_SUBMITTED,
                ),
            }
        )

        result = look_up_stated_position(
            **_request("chr12", 3556401, "GGTGTCCCCCTCGC", "G"),
            assembly_accession=SSCROFA11,
            fetch=sources,
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(
            result.conflicts,
            (
                "The request states no genome version, and EVA's lookup reads the "
                "Run's Sscrofa11.1 only; the position is not checked on any other "
                "version.",
            ),
        )

    def test_a_pig_deletion_is_found_at_its_eva_form_and_its_cluster_read(self):
        for name in ("Sscrofa11.1", SSCROFA11):
            with self.subTest(name=name):
                self._pig_deletion(name)

    def _pig_deletion(self, name):
        sources = _Sources(
            {
                ("fetch_assembly_report", SSCROFA11): (200, SSCROFA11_REPORT),
                ("ncbi_sequence_names", SSCROFA11): (200, SSCROFA11_NAMES),
                ("ncbi_reference_span", "NC_010454.4:3556400:14"): (
                    200,
                    SPAN_NC_010454_4,
                ),
                (
                    "eva_clustered_variants_at",
                    "GCA_000003025.6:CM000823.5:3556402:DEL",
                ): (200, EVA_AT_PIG_DEL),
                ("eva_clustered_variant", "rs5334475172"): (200, PIG_RS5334475172),
                ("eva_contig_alias", "CM000823.5"): (200, PIG_CM000823_5_ALIAS),
                ("eva_submitted_variants", "rs5334475172"): (
                    200,
                    PIG_RS5334475172_SUBMITTED,
                ),
            }
        )

        result = look_up_stated_position(
            **_request(
                "chr12",
                3556401,
                "GGTGTCCCCCTCGC",
                "G",
                assembly=StatedAssembly(name=name, cited_line="Methods"),
            ),
            assembly_accession=SSCROFA11,
            fetch=sources,
        )

        self.assertEqual(
            (result.status, result.lookup_source, result.verified_rsid),
            ("resolved", "eva", "rs5334475172"),
        )
        self.assertEqual(
            _candidates(result),
            [
                (
                    "Sscrofa11.1",
                    "NC_010454.4",
                    3556402,
                    "GTGTCCCCCTCGC",
                    "",
                    ("rs5334475172",),
                )
            ],
        )
        self.assertEqual(
            [s.operation for s in result.location.supporting_sources],
            [
                "fetch_assembly_report",
                "ncbi_sequence_names",
                "ncbi_reference_span",
                "eva_clustered_variants_at",
            ],
        )
        self.assertEqual(result.location.reference_bases, "GGTGTCCCCCTCGC")


class PositionGetTests(unittest.TestCase):
    """Each position GET is one documented path; a malformed key never leaves."""

    def test_each_position_get_reads_its_documented_path(self) -> None:
        ncbi_sent, eva_sent = [], []

        class _Ncbi:
            contact_count = 0

            def get(self, url, *, max_bytes):
                ncbi_sent.append(url)
                return MetadataResponse(
                    request_identity=url,
                    citation_url=url,
                    status_code=200,
                    source_version=None,
                    body=b"{}",
                    observed_at=datetime(2026, 10, 17, tzinfo=UTC),
                    elapsed_time_ms=1,
                )

        class _Eva:
            contact_count = 0

            def get(self, url):
                eva_sent.append(url)
                return InvestigationSourceResponse(url, 200, b"[]", "2026-10-17Z", 1)

        ncbi = NcbiInvestigationSourceAdapter(_Ncbi())
        # An answer that names no FTP directory is the operation's answer.
        ncbi.fetch_sequence_names(GRCH38)
        ncbi.fetch_reference_span("NC_000019.10:44908683:2")
        ncbi.fetch_spdi_rsids("NC_000019.10:44908683:T:C")
        EvaInvestigationSourceAdapter(_Eva()).fetch_eva_clustered_variants_at(
            "GCA_000003025.6:CM000823.5:3556402:DEL"
        )

        self.assertEqual(
            ncbi_sent,
            [
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                "GCF_000001405.40/links",
                "https://api.ncbi.nlm.nih.gov/variation/v0/spdi/"
                "NC_000019.10:44908683:2:NN/contextual",
                "https://api.ncbi.nlm.nih.gov/variation/v0/spdi/"
                "NC_000019.10:44908683:T:C/rsids",
            ],
        )
        self.assertEqual(
            eva_sent,
            [
                "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants"
                "?assemblyId=GCA_000003025.6&referenceName=CM000823.5&start=3556402"
                "&variantType=DEL"
            ],
        )

    def test_a_malformed_key_is_refused_before_contact(self) -> None:
        """A key's structure is the adapter's own safety rule; a sequence name is never
        judged by its shape (ADR 0049), only percent-encoded."""
        ncbi = NcbiInvestigationSourceAdapter(object())
        eva = EvaInvestigationSourceAdapter(object())
        for call in (
            lambda: ncbi.fetch_sequence_names("GRCh38"),
            lambda: ncbi.fetch_reference_span("NC_000019.10:10:0"),
            lambda: ncbi.fetch_spdi_rsids("NC_000019.10:10:T:C/../x"),
            lambda: eva.fetch_eva_clustered_variants_at("GCF_000003025.6:CM1.1:1:SNV"),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_a_sequence_name_of_any_shape_is_a_percent_encoded_key(self) -> None:
        sent = []

        class _Ncbi:
            contact_count = 0

            def get(self, url, *, max_bytes):
                sent.append(url)
                return MetadataResponse(
                    request_identity=url,
                    citation_url=url,
                    status_code=200,
                    source_version=None,
                    body=b"{}",
                    observed_at=datetime(2026, 10, 17, tzinfo=UTC),
                    elapsed_time_ms=1,
                )

        NcbiInvestigationSourceAdapter(_Ncbi()).fetch_reference_span("chr Un/1:10:1")

        self.assertEqual(
            sent,
            [
                "https://api.ncbi.nlm.nih.gov/variation/v0/spdi/"
                "chr%20Un%2F1:10:1:N/contextual"
            ],
        )


class LookupOriginCoverageTests(unittest.TestCase):
    """Every malformed variant lookup ending carries its typed origin, through the
    real position lookup (review r1 SP1)."""

    def _malformed(self, answers, what):
        result = look_up_stated_position(
            **_request("chr19", 44908684, "T", "C"),
            assembly_accession=GRCH38,
            fetch=_human(answers),
        )
        self.assertEqual(
            (result.status, result.origin.code), ("malformed", "malformed")
        )
        self.assertEqual(result.origin.what, what)
        self.assertEqual(result.conflicts, (result.origin.audit(),))
        self.assertNotIn("source_answers", result.conflicts[0])

    def test_a_malformed_run_assembly_report_is_typed(self):
        self._malformed(
            {("fetch_assembly_report", GRCH38): (200, b"{}")},
            "NCBI's report of the Run's assembly",
        )

    def test_a_malformed_reference_span_is_typed(self):
        self._malformed(
            {("ncbi_reference_span", "NC_000019.10:44908683:1"): (200, b"{}")},
            "NCBI's reference bases at the stated position",
        )

    def test_a_typed_ending_without_its_origin_is_refused(self):
        """#478 OQ1 (typed origins on #474's pattern): a typed lookup ending is
        recorded only with its origin, so no producer writes free text for it."""
        import json

        from pydantic import ValidationError

        from bio_neb.investigation.investigation import RefSnpToolResult

        result = look_up_stated_position(
            **_request("chr19", 44908684, "T", "C"),
            assembly_accession=GRCH38,
            fetch=_human(
                {("ncbi_reference_span", "NC_000019.10:44908683:1"): (200, b"{}")}
            ),
        )
        self.assertEqual(result.status, "malformed")
        stripped = result.model_dump(mode="json")
        del stripped["origin"]
        with self.assertRaisesRegex(
            ValidationError, "RefSNP ending has no typed origin"
        ):
            RefSnpToolResult.model_validate_json(json.dumps(stripped))

    def test_the_typed_endings_inventory(self):
        from bio_neb.investigation.investigation import TYPED_ENDINGS

        self.assertEqual(
            TYPED_ENDINGS,
            {
                "source_answered",
                "merged",
                "sequence_unnamed",
                "placement_unnamed",
                "version_unnamed",
                "malformed",
            },
        )


if __name__ == "__main__":
    unittest.main()
