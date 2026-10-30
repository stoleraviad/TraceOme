"""Every lookup ending's typed origin, its audit sentence and its plain words (#478,
owner OQ1 on #474's pattern).

The audit sentence is the lookup fact's own conflict and rationale, which the readable
report, the item Review and the CLI read. It states what the source answered, in its
own words where it wrote any, and holds no instruction: the next step is the Tool's
model view's alone.
"""

import re
import unittest

from pydantic import TypeAdapter

from bio_neb.agentic_journey.tool_execution import function_tool_names
from bio_neb.investigation import lookup_origins as origins

PINS = (
    (
        origins.SourceAnswered(
            source="EVA's clustered variant",
            status_code=410,
            key="rs1",
            words_sent='{"message":"rs1 was merged into rs2 on GRCh37"}',
        ),
        "EVA's clustered variant answered HTTP 410 for rs1: "
        '{"message":"rs1 was merged into rs2 on GRCh37"}.',
        "EVA's clustered variant answered with status 410: "
        '{"message":"rs1 was merged into rs2 on GRCh37"}',
    ),
    (
        origins.SourceAnswered(
            source="NCBI RefSNP", status_code=404, key="rs9", words_sent=""
        ),
        "NCBI RefSNP answered HTTP 404 for rs9 with no body.",
        "NCBI RefSNP answered with status 404 with nothing more",
    ),
    (
        origins.SourceAnswered(
            source="NCBI's rsIDs at the allele",
            status_code=404,
            words_sent="does not match any RSID",
            unquoted="no rsID here",
        ),
        "The quote 'no rsID here' is not in its answer. NCBI's rsIDs at the allele "
        "answered HTTP 404: does not match any RSID.",
        "NCBI's rsIDs at the allele answered with status 404: does not match any RSID",
    ),
    (
        origins.RecordState(rsid="rs3", state="withdrawn_snapshot_data"),
        "NCBI dbSNP's record of rs3 carries withdrawn_snapshot_data, not a placement "
        "snapshot.",
        "dbSNP keeps rs3 as withdrawn_snapshot_data, with no placement on a genome",
    ),
    (
        origins.Merged(rsid="rs60037639", into=("rs4149056",)),
        "NCBI dbSNP merged rs60037639 into rs4149056.",
        "dbSNP merged rs60037639 into rs4149056",
    ),
    (
        origins.SequenceUnnamed(contig="III", assemblies=("GCF_000002985.6",)),
        "No sequence is named for the written contig 'III' on GCF_000002985.6.",
        "which sequence 'III' means on the genome is not named yet",
    ),
    (
        origins.SequenceUnnamed(
            contig="III", assemblies=("GCF_000002985.6",), missing_record="NC_1.1"
        ),
        "The assembly report of GCF_000002985.6 has no sequence NC_1.1.",
        "the genome's report has no sequence NC_1.1 for 'III'",
    ),
    (
        origins.SequenceUnnamed(
            contig="III",
            assemblies=("GCF_000002985.6",),
            quote="chr 3",
            quote_fault="not_in_words",
        ),
        "The quote 'chr 3' naming GCF_000002985.6's sequence is not in the "
        "researcher's words.",
        "which sequence 'III' means on the genome is not named yet",
    ),
    (
        origins.PlacementUnnamed(
            source="EVA", rsid="rs5", sequences=("NC_1.1", "NC_2.1"), loci=2
        ),
        "EVA places rs5 at 2 loci on the Run's assembly, on NC_1.1, NC_2.1.",
        "rs5 has more than one place on the genome, and none is named",
    ),
    (
        origins.PlacementUnnamed(
            source="NCBI RefSNP", rsid="rs5", sequences=("NC_1.1",), named="NC_9.1"
        ),
        "NC_9.1 is not one of rs5's placements on the Run's assembly (NC_1.1).",
        "rs5 has more than one place on the genome, and none is named",
    ),
    (
        origins.VersionUnnamed(
            version="hg19", assembly_name="GRCh38.p14", assembly_accession="GCF_1.40"
        ),
        "The stated version 'hg19' is not literally one of the Run's assembly's names "
        "(GRCh38.p14, GCF_1.40), and no assembly is named for it yet.",
        "the genome version 'hg19' the request names is not yet tied to a genome",
    ),
    (
        origins.PositionWithoutAlleles(contig="19", position=44908822),
        "The request states 19:44908822 without its alleles and with no rsID; the "
        "Run's sources list rsIDs only at a stated allele, so the position is not "
        "looked up.",
        # No source is asked about it.
        "it states 19:44908822 without its alleles or an rsID, and the Run's sources "
        "list a variant only at a stated allele, so no source was asked about the "
        "position",
    ),
    (
        origins.Malformed(
            what="NCBI RefSNP's answer", detail="it is not one JSON object"
        ),
        "NCBI RefSNP's answer could not be read: it is not one JSON object.",
        "NCBI RefSNP's answer could not be read",
    ),
)

# What an instruction to the model or the researcher reads like: a Tool's name, an
# argument to send, or a verb that tells the reader what to do next.
_INSTRUCTION = re.compile(
    r"\b(call|look (it |that )?up|read its|quote (them|its)|name one|send|ask|search)\b"
    r"|source_answers|=<",
    re.IGNORECASE,
)


class LookupOriginTests(unittest.TestCase):
    def test_every_origin_reads_its_audit_sentence_and_plain_words(self):
        for origin, audit, words in PINS:
            with self.subTest(origin=origin):
                self.assertEqual((origin.audit(), origin.words()), (audit, words))

    def test_the_pin_covers_every_origin(self):
        self.assertEqual(
            {type(origin) for origin, *_ in PINS},
            {
                origins.SourceAnswered,
                origins.RecordState,
                origins.Merged,
                origins.SequenceUnnamed,
                origins.PlacementUnnamed,
                origins.VersionUnnamed,
                origins.PositionWithoutAlleles,
                origins.Malformed,
            },
        )

    def test_no_ending_tells_anyone_what_to_do(self):
        tools = function_tool_names("investigation_planner")
        for origin, *_ in PINS:
            for text in (origin.audit(), origin.words()):
                with self.subTest(origin=origin, text=text):
                    self.assertIsNone(_INSTRUCTION.search(text))
                    self.assertFalse(any(tool in text for tool in tools))

    def test_a_model_matched_sequence_name_reads_its_audit_form_and_plain_words(self):
        # The match is typed; one owner renders it.
        name = origins.ModelMatchedName(
            written="chrMT",
            sequence_name="MT",
            record="NC_012920.1",
            assembly_accession="GCF_000001405.40",
        )
        self.assertEqual(
            (name.audit(), name.words()),
            (
                "chrMT -> MT (NC_012920.1) on GCF_000001405.40",
                "the model read 'chrMT' as the sequence MT (NC_012920.1) on "
                "GCF_000001405.40",
            ),
        )
        self.assertEqual(
            TypeAdapter(origins.ModelMatchedName).validate_json(name.model_dump_json()),
            name,
        )

    def test_a_long_answer_is_quoted_with_its_cut_named(self):
        body = ("x" * 350).encode()
        self.assertEqual(
            origins.source_words(body), "x" * 300 + "… (50 more characters)"
        )
        self.assertEqual(origins.source_words(b"  short \n"), "short")

    def test_an_origin_round_trips_by_its_code(self):
        adapter = TypeAdapter(origins.LookupOrigin)
        for origin, *_ in PINS:
            with self.subTest(origin=origin):
                self.assertEqual(
                    adapter.validate_json(origin.model_dump_json()), origin
                )


class ReaderTests(unittest.TestCase):
    def test_the_readable_report_shows_a_410_ending_in_the_source_s_own_words(self):
        # The readers show the observed fact and the source's words, and nothing
        # in them tells anyone what to do. The page copies the source's own answer
        # from its raw artifact.
        from bio_neb.investigation.collection_view import _lookup
        from bio_neb.investigation.investigation import RetrievedSource
        from bio_neb.investigation.investigation_sources import normalize_refsnp
        from bio_neb.reporting.readable_report import source_answer_cell

        body = b'{"error":{"message":"rs1 was withdrawn; see rs2"}}'
        source = RetrievedSource.create(
            operation="fetch_refsnp",
            request_identity="https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/1",
            status_code=410,
            observed_at="2026-10-29T00:00:00Z",
            elapsed_time_ms=1,
            raw_response=body,
        )
        result = normalize_refsnp(
            rsid="rs1",
            requested_rsid="rs1",
            assembly_accession="GCF_000001405.40",
            source=source,
            raw_response=body,
        )
        sentence = (
            'NCBI RefSNP answered HTTP 410 for rs1: {"error":{"message":"rs1 was '
            'withdrawn; see rs2"}}.'
        )
        self.assertEqual(
            (result.status, result.conflicts, result.origin.code),
            ("source_answered", (sentence,), "source_answered"),
        )
        line = source_answer_cell(_lookup(result), body)
        self.assertEqual(
            line,
            "NCBI `fetch_refsnp` HTTP 410 (gone): "
            '`{"error":{"message":"rs1 was withdrawn; see rs2"}}`',
        )
        self.assertIsNone(_INSTRUCTION.search(line))

    def test_the_readable_report_states_what_a_deferred_position_establishes(self):
        # A position deferred for want of alleles says what it does not establish,
        # and does not read as "could not be placed".
        from bio_neb.reporting.readable_report import ReadableRow

        deferred = ReadableRow(
            number=2,
            request_member_id="deferred",
            target="19:44,908,822",
            label="19:44908822",
            read_as="19:44908822",
            state="not_checked",
            items=(),
            lookup=None,
            reason_code="position_without_alleles",
        )
        self.assertEqual(
            deferred.not_searched,
            "Deferred out of the Run: it was stated without its alleles or an rsID, "
            "so no source was asked about it and the VCF was not searched; this does "
            "not say whether the sample carries a variant there",
        )
        self.assertNotIn("could not be placed", deferred.not_searched)
        self.assertIsNone(_INSTRUCTION.search(deferred.not_searched))


if __name__ == "__main__":
    unittest.main()
