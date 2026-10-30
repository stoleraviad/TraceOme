"""Each brittle rule meets a real recorded answer and fails it (#466, ADR 0049).

Every test drives the product's own adapter and reader over `tests/recorded_sources.py`'s
recorded answers (or, for a researcher's or a paper's words, the text itself) and asserts
what ADR 0049 asks: code copies what the source returned, the model decides, code checks
membership. A rule that still fails one is `expected_red`, naming the rule and the ticket
whose fix removes the mark; since #464 and #486 none is. The fixtures' own loading is
proved, green, in `tests/test_recorded_sources.py`.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from bio_neb.agentic_journey.source_contacts import call_source
from bio_neb.failure import FailureOverrides, FailureTuning, run_failure_tuning
from bio_neb.investigation.investigation import (
    InvestigationInput,
    InvestigationVcfIdentity,
    RetrievedSource,
    UserAssertion,
)
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedAssembly,
    StatedChange,
    StatedPosition,
    researcher_rsids,
)
from bio_neb.investigation.investigation_sources import (
    literal_version,
    look_up_eva_rsid,
    look_up_stated_position,
    normalize_refsnp,
    stated_version_sequence,
)
from bio_neb.investigation.vcf_observation import (
    DeterministicVcfObservationTool,
    ObserveVariantInput,
)
from bio_neb.paper_reading.answer import judge, kept_finding
from bio_neb.paper_reading.contracts import FindingPointer
from bio_neb.paper_reading.reader import Paper
from bio_neb.parabricks.execution_planning_journey import (
    ORGANISM_RECORD_EVENTS,
    quoted_in_record,
)
from bio_neb.providers.ncbi_assembly_report import SequenceRecord
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.prebuilt_index_tool import (
    PrebuiltIndexGrounding,
    ReadPrebuiltIndexInput,
    SearchPrebuiltIndexesInput,
    read_candidate,
    search_catalogues,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyQuery,
    AssemblySourceRecord,
    DiscoverReferenceAssembliesInput,
    DiscoverReferenceAssembliesTool,
    ReferenceAdapterError,
    RequirementGrounding,
    ResolvedTaxonEvidence,
)
from bio_neb.reporting.explanation_draft import DraftBinding, MemberDraft, check
from bio_neb.reporting.explanation_rows import ExplanationRow
from bio_neb.target_request.ncbi_sources import NcbiSources, answer_claim
from bio_neb.target_request.records import GeneChangeClaim
from bio_neb.target_request.source_failure import SourceFailed
from tests.paper_reading_support import document, finding, value
from tests.recorded_sources import fixture, recorded_ports
from tests.sequential_investigation_support import SequentialAdaptiveModel, run_attempt

NO_BACKOFF = FailureTuning(operation=FailureOverrides(backoff_seconds=0))


def _named(contig: str):
    """The sequence choice a stated-version test does not test: any assembly's record
    named ``contig`` (#467 makes the sequence the model's or a literal name's choice;
    these tests are about the version)."""

    def named(accession: str, name: str) -> SequenceRecord:
        return SequenceRecord(
            sequence_name=contig,
            sequence_role="assembled-molecule",
            assigned_molecule=None,
            location_type=None,
            genbank=None,
            relationship="=",
            refseq=contig,
            assembly_unit="Primary Assembly",
            length=1,
            ucsc=None,
        )

    return named


def _json(fixture_id: str, index: int):
    return json.loads(fixture(fixture_id).answers[index].body)


class _Recorded(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(run_failure_tuning(NO_BACKOFF))

    def ncbi_sources(self, *fixture_ids: str) -> NcbiSources:
        cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        return NcbiSources(recorded_ports(*fixture_ids).ncbi, cache)

    def copied_sequences(self, fixture_id: str, accession: str):
        try:
            return self.ncbi_sources(fixture_id).sequences(accession)
        except SourceFailed as refused:
            self.fail(f"the sequence report went unread: {refused}")

    def stated(self, fixture_id: str, accession: str, contig: str, position: int):
        return look_up_stated_position(
            position=StatedPosition(contig=contig, position=position),
            stated_assembly=None,
            rsid=None,
            requested_rsid=None,
            assembly_accession=accession,
            fetch=recorded_ports(fixture_id).fetch(),
            # A position with no allele and no rsID asks no variant source.
            source="ncbi_dbsnp",
        )

    def assert_no_code_verdict_on_the_name(self, result, contig: str) -> None:
        self.assertNotIn(
            f"names no sequence {contig}.",
            " ".join(result.conflicts),
            "code judged the written name; the model names the sequence by membership",
        )


class NamingTests(_Recorded):
    def test_an_nz_refseq_name_is_copied_with_the_report(self) -> None:
        sequences = self.copied_sequences("naming-nz-sequence", "GCF_000743255.1")

        self.assertIn(
            {"NZ_CP009072.1", "CP009072.1"},
            [
                {"NZ_CP009072.1", "CP009072.1"} & sequence.names
                for sequence in sequences
            ],
        )

    def test_an_unplaced_wgs_scaffold_is_copied_with_its_names(self) -> None:
        sequences = self.copied_sequences("naming-wgs-scaffold", "GCF_000002285.3")

        held = {name for sequence in sequences for name in sequence.names}
        self.assertTrue(
            {"AAEX03018827.1", "NW_003726146.1", "chrUn_AAEX03018827"} <= held
        )

    def test_a_mitochondrion_written_chrmt_is_not_refused_by_code(self) -> None:
        result = self.stated("naming-mitochondrion", "GCF_000001405.40", "chrMT", 3243)

        self.assert_no_code_verdict_on_the_name(result, "chrMT")

    def test_an_arabic_chromosome_on_a_roman_named_assembly_is_not_refused_by_code(
        self,
    ) -> None:
        result = self.stated(
            "naming-roman-chromosome", "GCF_000146045.2", "chr4", 207822
        )

        self.assert_no_code_verdict_on_the_name(result, "chr4")

    def test_a_tair_chromosome_name_is_not_refused_by_code(self) -> None:
        result = self.stated("naming-plant-chromosome", "GCF_000001735.4", "Chr1", 1000)

        self.assert_no_code_verdict_on_the_name(result, "Chr1")

    def test_a_scaffold_of_a_large_assembly_is_reachable_from_its_copied_names(
        self,
    ) -> None:
        # The first scaffold NCBI's own first page lists.
        page = _json("naming-large-report", 1)
        scaffold = next(
            row for row in page["reports"] if row["role"] != "assembled-molecule"
        )
        sequences = self.copied_sequences("naming-large-report", "GCF_018294505.1")

        self.assertIn(
            scaffold["refseq_accession"],
            {sequence.refseq for sequence in sequences},
        )

    def test_a_large_assembly_report_is_read_whole(self) -> None:
        ports = recorded_ports("naming-large-report")
        locator = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/018/294/505/"
            "GCF_018294505.1_IWGSC_CS_RefSeq_v2.1/"
            "GCF_018294505.1_IWGSC_CS_RefSeq_v2.1_genomic.fna.gz"
        )
        report = fixture("naming-large-report").answers[3].body
        rows = [
            line for line in report.decode().splitlines() if not line.startswith("#")
        ]

        try:
            sequences = call_source(
                "describe_companion",
                lambda: ports.datasets.fetch_assembly_sequences(locator),
            )
        except ReferenceAdapterError as refused:
            self.fail(f"the report went unread: {refused}")
        self.assertEqual(len(sequences), len(rows))

    def test_a_mouse_reference_requests_its_chromosomes(self) -> None:
        read = _galaxy_read("naming-galaxy-unit", MOUSE, MOUSE_FASTA)

        # 21 chromosomes and the mitochondrion: the unit is named C57BL/6J, not
        # "Primary Assembly".
        self.assertGreaterEqual(len(read.contigs.requested), 22)
        self.assertTrue(read.contigs.holds_every_requested_sequence, read.summary)

    def test_a_brc_search_for_homo_sapiens_lists_grch38_with_its_candidate(self):
        ports = recorded_ports("naming-galaxy-unit")
        listed = search_catalogues(
            SearchPrebuiltIndexesInput(
                grounding=_grounding(HUMAN, HUMAN_FASTA),
                repository="brc",
                search_text="Homo sapiens",
            ),
            galaxy=ports.galaxy,
            contact=call_source,
        )

        self.assertEqual(
            [(row.value, row.candidate_id) for row in listed.rows],
            [(HUMAN, f"galaxy:brc:{HUMAN}")],
        )
        self.assertEqual((listed.total_rows, listed.total_matches), (5568, 1))

    def test_a_search_pages_every_row_galaxy_publishes(self) -> None:
        ports = recorded_ports("naming-galaxy-unit")

        def page(offset):
            return search_catalogues(
                SearchPrebuiltIndexesInput(
                    grounding=_grounding(HUMAN, HUMAN_FASTA),
                    repository="brc",
                    offset=offset,
                ),
                galaxy=ports.galaxy,
                contact=call_source,
            )

        first, last = page(0), page(5550)
        self.assertEqual((len(first.rows), first.next_offset), (25, 25))
        self.assertEqual((len(last.rows), last.next_offset), (18, None))
        # Galaxy's order, no row cut: the last page ends on the catalogue's last row.
        loc = fixture("naming-galaxy-unit").answers[0].body.decode().splitlines()
        self.assertEqual(last.rows[-1].value, loc[-1].split("\t")[0])

    def test_the_human_index_pairs_every_requested_sequence_and_lists_alt_loci(self):
        # Galaxy BRC's GRCh38 index holds the alt loci; they are extras with NCBI's
        # role, disclosed in the Review, not a refusal.
        read = _galaxy_read(
            "naming-galaxy-unit naming-galaxy-human naming-mitochondrion",
            HUMAN,
            HUMAN_FASTA,
        )

        self.assertTrue(read.contigs.holds_every_requested_sequence, read.summary)
        self.assertEqual(len(read.contigs.requested), 191)
        roles = [extra.ncbi_role for extra in read.contigs.extras]
        self.assertEqual(roles.count("alt-scaffold"), 261)
        self.assertIn("chr1_KI270762v1_alt", {e.name for e in read.contigs.extras})
        self.assertEqual(len(read.files), 7)


HUMAN = "GCF_000001405.40"
HUMAN_FASTA = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    "GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
)
MOUSE = "GCF_000001635.27"
MOUSE_FASTA = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
    "GCF_000001635.27_GRCm39/GCF_000001635.27_GRCm39_genomic.fna.gz"
)


def _grounding(accession: str, locator: str) -> PrebuiltIndexGrounding:
    del locator
    return PrebuiltIndexGrounding(
        reference_event_id=str(uuid.uuid4()), assembly_accession=accession
    )


def _galaxy_read(fixture_ids: str, accession: str, locator: str):
    """BRC's index of ``accession`` read as `read_prebuilt_index` reads it: listed by
    its accession, its `.ann` paired with NCBI's recorded report of the assembly."""
    ports = recorded_ports(*fixture_ids.split())
    grounding = _grounding(accession, locator)
    listed = search_catalogues(
        SearchPrebuiltIndexesInput(
            grounding=grounding, repository="brc", search_text=accession
        ),
        galaxy=ports.galaxy,
        contact=call_source,
    )
    return read_candidate(
        ReadPrebuiltIndexInput(
            grounding=grounding,
            listing_event_id=str(uuid.uuid4()),
            value=accession,
            candidate=listed.candidate(accession),
        ),
        galaxy=ports.galaxy,
        contact=call_source,
        sequences=lambda: (
            call_source(
                "describe_companion",
                lambda: ports.datasets.fetch_assembly_sequences(locator),
            ),
            None,
        ),
    )


def _discover(fixture_id: str, taxid: int, query: AssemblyQuery | None = None):
    """Discovery's copied page for ``taxid`` over the recorded answers."""
    current = RequirementGrounding(
        journey_id=str(uuid.uuid4()),
        plan_id="plan_" + "a" * 64,
        requirement_id="requirement_" + "b" * 32,
        kind="reference_assembly_identity",
        subject_key="reference_assembly",
        source_event_ids=(str(uuid.uuid4()),),
    )
    taxon = ResolvedTaxonEvidence(
        candidate_id=f"taxon:ncbitaxon:{taxid}",
        ncbi_tax_id=taxid,
        candidate_evidence_id="taxon-candidate:" + "d" * 64,
        result_evidence_id="taxon-result:" + "e" * 64,
        source_event_id=str(uuid.uuid4()),
    )
    return DiscoverReferenceAssembliesTool().discover(
        DiscoverReferenceAssembliesInput(
            grounding=current, taxon=taxon, query=query or AssemblyQuery()
        ),
        expected_grounding=current,
        expected_taxon=taxon,
        adapter=recorded_ports(fixture_id).datasets,
    )


def _accessions(page) -> list[str]:
    return [row["accession"] for row in page["reports"]]


class AssemblyTests(_Recorded):
    """Discovery copies NCBI's own page and a stated version is named by NCBI's own
    search, never by a code filter, rank, cut, patch strip or latest pick (#471, ADR
    0049), on #466's recorded answers."""

    def discover(self, fixture_id: str, taxid: int, **options):
        return _discover(fixture_id, taxid, AssemblyQuery(**options))

    def literal(self, fixture_id: str, stated: str, taxid: int):
        return literal_version(
            stated, taxid, self.ncbi_sources(fixture_id).genome_reports
        )

    def test_every_row_of_the_species_page_reaches_the_model(self) -> None:
        """E. coli's listing, over half a million assemblies: one page of 100 rows,
        each as NCBI lists it, its descendant taxa's rows among them, with NCBI's
        total_count; the next page token reads the next 100."""
        first_page, second_page = (
            _json("assembly-species-pages", 4),
            _json("assembly-species-pages", 7),
        )

        first = self.discover("assembly-species-pages", 562)
        second = self.discover(
            "assembly-species-pages", 562, page_token=first.next_page_token
        )

        self.assertEqual(
            [row.assembly_accession for row in first.rows], _accessions(first_page)
        )
        self.assertEqual(first.total_count, first_page["total_count"])
        self.assertGreater(first.total_count, 500_000)
        self.assertTrue(any(row.ncbi_tax_id != 562 for row in first.rows))
        self.assertEqual(
            [row.assembly_accession for row in second.rows], _accessions(second_page)
        )
        self.assertFalse(
            {row.assembly_accession for row in first.rows}
            & {row.assembly_accession for row in second.rows}
        )

    def test_ncbis_reference_only_option_lists_the_species_reference_genomes(
        self,
    ) -> None:
        """The model's ``reference_only`` reaches NCBI as NCBI's own filter, which
        answers E. coli's two reference genomes; code adds none."""
        result = self.discover("assembly-species-pages", 562, reference_only=True)

        self.assertEqual(
            [row.assembly_accession for row in result.rows],
            _accessions(_json("assembly-species-pages", 5)),
        )
        self.assertEqual(result.total_count, 2)

    def test_a_stated_version_on_a_paged_species_listing_is_looked_up(self) -> None:
        """ASM584v2 on E. coli: NCBI's own search for the stated words finds it
        literally, so no listing of half a million assemblies is refused past one
        page (#471 R4 a)."""
        found = self.literal("assembly-species-pages", "ASM584v2", 562)

        self.assertEqual(found.assembly_accession, "GCF_000005845.2")
        self.assertEqual(
            stated_version_sequence(
                found.assembly_accession, found.assembly_name, _named("NC_000913.3")
            ),
            ("NC_000913.3", "GCF_000005845.2"),
        )

    def test_every_row_of_the_strain_page_reaches_the_model(self) -> None:
        """MG1655's own TaxId lists 101 assemblies: both pages reach the model, every
        row with its infraspecific names, and no row is matched by a strain substring
        or checked by its lineage."""
        first = self.discover("assembly-strain", 511145)
        second = self.discover(
            "assembly-strain", 511145, page_token=first.next_page_token
        )

        self.assertEqual(
            [row.assembly_accession for row in (*first.rows, *second.rows)],
            _accessions(_json("assembly-strain", 1))
            + _accessions(_json("assembly-strain", 2)),
        )
        self.assertEqual(first.total_count, len(first.rows) + len(second.rows))
        self.assertIsNone(second.next_page_token)

    def test_a_genbank_only_assembly_is_looked_up(self) -> None:
        """Loblolly pine has no RefSeq assembly: NCBI's search over every source names
        its GenBank Ptaeda2.0, with no RefSeq-only listing (#393 G1 A, superseded)."""
        found = self.literal("assembly-genbank-only", "Ptaeda2.0", 3352)

        self.assertEqual(
            (found.assembly_accession, found.assembly_name),
            ("GCA_000404065.3", "Ptaeda2.0"),
        )

    def test_a_suppressed_assembly_is_copied_with_its_status_not_rejected(self) -> None:
        """Horse EquCab3.0's suppressed RefSeq row is copied with NCBI's status and
        every infraspecific field, beside its current GenBank twin."""
        result = self.discover("assembly-suppressed", 9796)

        suppressed = result.row("GCF_002863925.1")
        self.assertEqual(suppressed.assembly_status, "suppressed")
        self.assertIsNotNone(suppressed.suppression_reason)
        self.assertEqual(
            suppressed.infraspecific_names.model_dump(exclude_none=True),
            {"breed": "thoroughbred", "isolate": "Twilight", "sex": "female"},
        )
        self.assertEqual(result.row("GCA_002863925.1").assembly_status, "current")

    def test_grch37_is_not_resolved_to_a_patch_release_code_chose(self) -> None:
        """NCBI lists GRCh37 and its patch releases; with no model, the stated words
        name GRCh37 itself, literally, never GRCh37.p13 by a patch strip and a latest
        pick (#417, superseded)."""
        found = self.literal("assembly-stated-grch37", "GRCh37", 9606)

        self.assertEqual(
            (found.assembly_accession, found.assembly_name),
            ("GCF_000001405.13", "GRCh37"),
        )

    def test_an_assembly_type_is_copied_and_described_as_no_ploidy(self) -> None:
        """NCBI writes ``haploid`` for diploid Arabidopsis TAIR10: the field is copied
        as written, and its description says it is how the assembly represents the
        genome, not the organism's ploidy (#470 handoff)."""
        described = recorded_ports(
            "naming-plant-chromosome"
        ).datasets.describe_assembly("GCF_000001735.4")

        self.assertEqual(described.records[0].assembly_type, "haploid")
        self.assertIn(
            "not the organism's ploidy",
            AssemblySourceRecord.model_json_schema()["properties"]["assembly_type"][
                "description"
            ],
        )


class _OfferedTools(SequentialAdaptiveModel):
    """The scripted model, keeping the Tools each request offers it."""

    def __init__(self, **choices) -> None:
        super().__init__(**choices)
        self.offered: list[set[str]] = []

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        self.offered.append({tool.name for tool in info.function_tools})
        return await super().respond(messages, info)


class VariantSourceTests(_Recorded):
    def refsnp(self, fixture_id: str, rsid: str, assembly: str):
        source, body = recorded_ports(fixture_id).fetch()("fetch_refsnp", rsid)
        return normalize_refsnp(
            rsid=rsid, assembly_accession=assembly, source=source, raw_response=body
        )

    def eva(self, fixture_id: str, rsid: str, assembly: str):
        return look_up_eva_rsid(
            rsid=rsid,
            assembly_accession=assembly,
            fetch=recorded_ports(fixture_id).fetch(),
        )

    def test_evas_410_is_the_sources_answer_not_a_verdict(self) -> None:
        result = self.eva("variant-eva-retired", "rs160719159", "GCF_000002985.6")

        self.assertEqual(
            (result.source.status_code, result.status, result.candidates),
            (410, "source_answered", ()),
        )
        self.assertIn("HTTP 410", result.conflicts[0])

    def test_a_non_human_run_is_offered_both_lookup_tools(self) -> None:
        """Both lookup Tools on every Run, the model choosing (#469, owner RC2 1, ADR
        0049): a C. elegans Run's model asks dbSNP first, reads its "no record" and finds
        the rsID through EVA, on #466's recorded answers through a real Attempt."""
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/CE01.vcf.gz",
                byte_count=1234,
                sha256="d" * 64,
            ),
            sample_id="CE01",
            assembly_accession="GCF_000002985.6",
        )
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs7089734242 in sample CE01.",
            rsid="rs7089734242",
            predecessor_ids=(investigation_input.input_id,),
        )
        model = _OfferedTools(lookups=("fetch_refsnp", "fetch_eva_variant"))
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory),
                model,
                recorded_ports("variant-non-human-snp").investigation,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            lookups = [
                (
                    event["payload"]["result"]["lookup_source"],
                    event["payload"]["result"]["status"],
                )
                for event in read_evidence(journey)
                if event["event_type"] == "RefSnpToolResultRecorded"
            ]

        self.assertTrue(model.offered)
        for offered in model.offered:
            self.assertLessEqual({"fetch_refsnp", "fetch_eva_variant"}, offered)
        # dbSNP's 404 is its own answer, with no verdict.
        self.assertEqual(
            lookups, [("ncbi_dbsnp", "source_answered"), ("eva", "resolved")]
        )
        self.assertEqual(projection.status, "review_pending")

    def test_a_position_is_listed_by_evas_own_variant_type_the_model_names(
        self,
    ) -> None:
        """The model names EVA as the source and EVA's own variantType for its listing
        (#469, owner Q1 a and R1 a, ADR 0049): on the recorded answers EVA files the
        C. elegans deletion under DEL and answers INDEL with an empty 404, its own
        answer for the model to read."""
        stated = {
            "position": StatedPosition(
                contig="III", position=931672, ref="TAG", alt="T"
            ),
            "stated_assembly": StatedAssembly(
                name="WBcel235", cited_line="aligned to WBcel235"
            ),
            "rsid": None,
            "requested_rsid": None,
            "assembly_accession": "GCF_000002985.6",
            "fetch": recorded_ports(
                "variant-eva-by-position", "variant-non-human-snp"
            ).fetch(),
            # No model names the sequence here: III is literally one of its names.
            "literal": True,
            "source": "eva",
        }

        indel = look_up_stated_position(**stated, variant_type="INDEL")
        deletion = look_up_stated_position(**stated, variant_type="DEL")

        self.assertEqual(
            (indel.status, indel.source.status_code, indel.location.variant_type),
            ("source_answered", 404, "INDEL"),
        )
        self.assertEqual(
            (deletion.status, deletion.lookup_source, deletion.verified_rsid),
            ("resolved", "eva", "rs7089734242"),
        )
        self.assertEqual(deletion.location.listed_rsids, ("rs7089734242",))

    def test_evas_naming_convention_the_model_names_reads_its_own_name_for_it(
        self,
    ) -> None:
        """EVA's contigNamingConvention is the model's to name (#469, owner SP4 a, ADR
        0049): EVA's by-position query reads the sequence's name in that convention,
        taken from the Run's own report; its answer, named as asked, is checked by
        membership in that report's names; a convention the report has no name in
        ends the lookup saying so, before any GET.

        This test is about the listing: the record GETs it leads to under REFSEQ were
        not recorded (R2's ceiling), so the fake answers them 404, synthetic."""
        recorded = recorded_ports(
            "variant-eva-by-position", "variant-non-human-snp"
        ).fetch()

        def fetch(operation: str, key: str):
            if operation != "eva_clustered_variant" or "REFSEQ" not in key:
                return recorded(operation, key)
            source = RetrievedSource.create(
                operation=operation,
                request_identity=f"synthetic:{key}",
                status_code=404,
                observed_at="2026-10-27T17:16:33Z",
                elapsed_time_ms=1,
                raw_response=b"",
            )
            return source, b""

        stated = {
            "position": StatedPosition(
                contig="III", position=931672, ref="TAG", alt="T"
            ),
            "stated_assembly": StatedAssembly(
                name="WBcel235", cited_line="aligned to WBcel235"
            ),
            "rsid": None,
            "requested_rsid": None,
            "assembly_accession": "GCF_000002985.6",
            "fetch": fetch,
            "literal": True,
            "source": "eva",
        }

        refseq = look_up_stated_position(**stated, contig_naming="REFSEQ")
        ucsc = look_up_stated_position(**stated, contig_naming="UCSC")

        self.assertEqual(
            refseq.location.supporting_sources[-1].request_identity,
            "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants"
            "?assemblyId=GCA_000002985.3&referenceName=NC_003281.10&start=931673"
            "&variantType=DEL&contigNamingConvention=REFSEQ",
        )
        self.assertEqual(refseq.location.listed_rsids, ("rs7089734242",))
        self.assertEqual(refseq.location.contig_naming, "REFSEQ")
        # WBcel235's report gives chromosome III no UCSC-style name: no name to send,
        # so no GET; the fact states only that observation, and the model reads why
        # and chooses another convention.
        self.assertEqual(
            (ucsc.status, ucsc.location.contig_naming), ("incompatible", "UCSC")
        )
        self.assertEqual(
            ucsc.conflicts,
            (
                "NCBI's report of the Run's WBcel235 gives NC_003281.10 no name under "
                "EVA's UCSC.",
            ),
        )
        self.assertFalse(
            [
                source
                for source in ucsc.location.supporting_sources
                if source.operation.startswith("eva_")
            ]
        )

    def test_dbsnp_answers_a_non_human_rsid_with_404_and_eva_holds_it(self) -> None:
        # Green: the two answers a Run offered both Tools reads. dbSNP's 404 is its
        # own answer for the model to read, with no verdict.
        dbsnp = self.refsnp("variant-non-human-snp", "rs7089734242", "GCF_000002985.6")
        self.assertEqual(
            (dbsnp.status, dbsnp.source.status_code), ("source_answered", 404)
        )

    def test_every_eva_placement_is_copied_without_a_cap(self) -> None:
        clustered = _json("variant-many-placements", 0)

        result = self.eva("variant-many-placements", "rs109500000", "GCF_002263795.3")

        self.assertNotEqual(result.status, "ambiguous", result.conflicts)
        self.assertEqual(
            [(p.eva.assembly_accession, p.sequence) for p in result.placements],
            [
                (item["data"]["assemblyAccession"], item["data"]["contig"])
                for item in clustered
            ],
        )
        # Only the placement on the Run's INSDC twin, as NCBI's recorded report pairs
        # ARS-UCD2.0 (GCA_002263795.4), is on the Run's assembly.
        self.assertEqual(
            [p.eva.assembly_accession for p in result.placements if p.on_run_assembly],
            ["GCA_002263795.4"],
        )

    def test_a_grch37_placement_that_is_not_ptlp_is_copied(self) -> None:
        result = self.refsnp("variant-dbsnp-non-ptlp", "rs429358", "GCF_000001405.25")

        self.assertIn(
            ("NC_000019.9", 45411941, "T", "C"),
            [(c.contig, c.one_based_position, c.ref, c.alt) for c in result.candidates],
        )

    def test_a_merged_dbsnp_record_names_its_current_rsid(self) -> None:
        # Green control: dbSNP's merge field is copied today.
        result = self.refsnp("variant-merged", "rs60037639", "GCF_000001405.40")

        self.assertEqual(
            (result.status, result.merged_into), ("merged", ("rs4149056",))
        )

    def test_a_withdrawn_record_is_copied_as_withdrawn_not_absent(self) -> None:
        result = self.refsnp("variant-dbsnp-withdrawn", "rs94", "GCF_000001405.40")

        self.assertNotIn(result.status, {"not_found", "malformed"}, result.conflicts)
        self.assertEqual(result.record_state, "withdrawn_snapshot_data")


HUMAN_RUN = "GCF_000001405.40"


class IdentifierTests(unittest.TestCase):
    def test_an_rsid_the_researcher_rejects_is_not_admissible(self) -> None:
        # The model types the rsID an answer means, quoting it, and the reading
        # replaces the member's rsIDs; no code reads the answer's words.
        from bio_neb.investigation.investigation import InvestigationTargetRead
        from bio_neb.investigation.investigation_request import (
            StatedTarget,
            typed_answers,
        )

        request = InvestigationRequestItem(request_text="rs429358", rsid="rs429358")
        words = "Not rs429358, I meant rs7412."
        answer = {
            "action": "clarify",
            "instruction": words,
            "response_event_id": "answer",
        }
        read = InvestigationTargetRead(
            attempt_id="attempt",
            collection_member_id="member",
            revision_number=1,
            source="answer",
            source_event_id="answer",
            words=words,
            quote="I meant rs7412",
            target=StatedTarget(rsid="rs7412"),
        ).model_dump(mode="json")

        self.assertEqual(
            researcher_rsids(request, typed_answers(request, [answer], [read])),
            ("rs7412",),
        )
        # Unread, the answer's words state nothing code admits.
        self.assertEqual(researcher_rsids(request, [answer]), ("rs429358",))

    def test_an_hgvs_g_site_in_free_text_is_a_site(self) -> None:
        # The model types the site from an answer's HGVS g. words, and the quote
        # check admits it; no position grammar reads it (ADR 0049). Its change (the
        # deletion) stays in the quote.
        from bio_neb.investigation.investigation_request import StatedPosition
        from bio_neb.parabricks.proposal_information import (
            AnsweredTargetsDraft,
            ground_answered_targets,
        )

        quote = "NC_049231.1:g.69888722_69888723del"
        draft = AnsweredTargetsDraft.model_validate(
            {
                "targets": [
                    {
                        "state": "asserted",
                        "quote": quote,
                        "site": {"contig": "NC_049231.1", "position": 69888722},
                    }
                ]
            }
        )

        (target,) = ground_answered_targets(draft, f"I meant {quote}.")

        self.assertEqual(
            target.site, StatedPosition(contig="NC_049231.1", position=69888722)
        )

    def test_ncbis_hgvs_answers_are_copied_whatever_the_organism(self) -> None:
        """NCBI's HGVS service answers human sequences only; each answer is copied with
        its status and body, a 400 included, never skipped or judged by code (#473
        owner Q3 a, owner 2026-10-28: not human-only; ADR 0049). The coding name's
        allele is followed to its equivalents on the Run's sequences."""
        # The Run's GRCh38 sequence names, as recorded for naming-mitochondrion.
        sources = BiologyTests.ncbi_sources(
            self,
            "identifiers-hgvs-human",
            "identifiers-hgvs-text",
            "naming-mitochondrion",
        )
        human = sources.hgvs("NM_004333.6:c.1799T>A", HUMAN_RUN, coding=True)
        protein = sources.hgvs("NP_004324.2:p.V600E", HUMAN_RUN, coding=False)
        dog = sources.hgvs(
            "NC_049231.1:g.69888722_69888723del", "GCF_011100685.1", coding=False
        )

        self.assertEqual(
            (human.status_code, human.spdis, human.on_run),
            (200, ("NM_004333.6:2024:T:A",), ("NC_000007.14:140753335:A:T",)),
        )
        self.assertEqual(
            json.loads(protein.body)["data"]["spdis"][0]["seq_id"], "NP_004324.2"
        )
        self.assertEqual((protein.status_code, protein.on_run), (200, ()))
        self.assertEqual(
            (dog.status_code, dog.spdis, json.loads(dog.body)["error"]["message"]),
            (
                400,
                (),
                "Error while processing 'NC_049231.1:g.69888722_69888723del': "
                "The submitted sequence is not human.",
            ),
        )

    def test_a_question_for_the_samples_result_is_not_read_by_words(self) -> None:
        """#472 Q4 a (ADR 0049): a question's kind is typed by its asker (asks_about),
        and the sample's own result is no kind a researcher answers, so the schema
        refuses it; no word list reads the question."""
        from typing import get_args

        from pydantic import ValidationError

        from bio_neb.investigation.investigation import (
            InvestigationQuestion,
            QuestionKind,
        )

        self.assertNotIn("sample_result", get_args(QuestionKind))
        with self.assertRaises(ValidationError) as refused:
            InvestigationQuestion(
                assertion_id="assertion",
                collection_member_id="member",
                revision_number=1,
                request=InvestigationRequestItem(
                    request_text="rs429358", rsid="rs429358"
                ),
                question="What does HG001 show at rs429358?",
                missing_information="The sample's call there.",
                asks_about="sample_result",
            )
        self.assertEqual(
            [(error["loc"], error["type"]) for error in refused.exception.errors()],
            [(("asks_about",), "literal_error")],
        )


def _vcf_rows(name: str) -> list[tuple[str, int, str, str, str]]:
    """The tiny VCF's records as (contig, position, REF, ALT, GT)."""
    rows = []
    for line in fixture("biology-ploidy").inputs[name].decode().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        rows.append((fields[0], int(fields[1]), fields[3], fields[4], fields[9]))
    return rows


def _assessed(row: tuple[str, int, str, str, str]):
    """One record read and assessed as the VCF lookup reads its query output."""
    contig, position, ref, alt, genotype = row
    target = ObserveVariantInput(
        assembly="GCF_000000000.1",
        contig=contig,
        one_based_position=position,
        ref=ref,
        alt=alt.split(",")[0],
    )
    payload = (
        f"{contig}\t{position}\t{ref}\t{alt}\t.\t.\tS\t{genotype}\t.\t.\t.\n"
    ).encode()
    (record,) = DeterministicVcfObservationTool._parse_records(payload, target)
    return DeterministicVcfObservationTool._assess_genotype(record, 1)


class BiologyTests(_Recorded):
    def test_haploid_and_polyploid_calls_are_read_by_their_alleles(self) -> None:
        # Green guard: the reader takes any number of alleles under one separator.
        cases = {
            row[4]: _assessed(row).state
            for row in _vcf_rows("haploid.vcf") + _vcf_rows("polyploid.vcf")
            if not ("|" in row[4] and "/" in row[4])
        }
        self.assertEqual(
            cases,
            {
                "1": "target_alt_carried",
                "0": "target_alt_not_carried",
                "0/0/1/1": "target_alt_carried",
                "0/0/0/0": "target_alt_not_carried",
            },
        )

    def test_a_haploid_organisms_recorded_record_holds_the_words_a_judgment_cites(
        self,
    ) -> None:
        """#470 (owner Q5 b): the ploidy judgment cites the organism record the
        reference stage kept (#469 Q2), here a recorded E. coli discovery answer; code
        checks only that the quote stands in it, and no copied field is a ploidy."""
        record = {
            "event_type": "ReferenceAssembliesRecorded",
            "payload": {
                "result": _discover("assembly-species-pages", 562).model_dump(
                    mode="json"
                )
            },
        }

        self.assertIn(record["event_type"], ORGANISM_RECORD_EVENTS)
        self.assertTrue(quoted_in_record(record, "Escherichia coli"))
        self.assertFalse(quoted_in_record(record, "Bacillus subtilis"))
        self.assertFalse(quoted_in_record(record, "Escherichia col"))
        self.assertNotIn("ploidy", json.dumps(record))

    def test_a_haploid_call_states_no_phase(self) -> None:
        (carried, _) = _vcf_rows("haploid.vcf")

        self.assertIsNone(_assessed(carried).phased)

    def test_a_mixed_phase_polyploid_call_is_read(self) -> None:
        """VCF 4.3 lets each allele carry its own separator, so `0/1|1` names three
        alleles and no one phase for the call (owner #466 Q7 a, #470 Q4 a)."""
        mixed = next(row for row in _vcf_rows("polyploid.vcf") if "|" in row[4])

        assessed = _assessed(mixed)

        self.assertEqual(
            (assessed.state, assessed.allele_indexes, assessed.phased),
            ("target_alt_carried", (0, 1, 1), None),
        )

    def test_a_haploid_rows_explanation_is_not_judged_by_a_prose_pattern(self) -> None:
        """The model lists each genotype it writes with its target and quote, and
        code checks only membership: the target is bound, the genotype is its call
        and the quote stands in the text (ADR 0049, #473 owner Q5 a). A haploid
        call ``1`` is checked; "1/2 of the reads" is no genotype the model listed."""
        binding = DraftBinding(
            stage="member",
            targets=(
                ExplanationRow(
                    number=4,
                    kind="position",
                    request_member_id="member-b",
                    subject="subject-4",
                    state="observed",
                    work_item_id="work-item:bb",
                    outcome="exact_allele_observed",
                    genotype="1",
                    words="recorded words",
                ),
            ),
            qualifications=(),
            evidence_event_ids=("e1",),
        )

        admitted = check(
            binding,
            {
                "why": "The VCF calls genotype 1 at this haploid position; 1/2 of "
                "the reads cover it.",
                "why_references": ["e1"],
                "limitations": [],
                "genotypes": [
                    {"target": 4, "genotype": "1", "quote": "calls genotype 1"}
                ],
            },
        )

        self.assertIsInstance(admitted, MemberDraft)

    def test_a_mitochondrial_tga_codon_reads_as_tryptophan(self) -> None:
        """The genetic code is copied from the annotation (#469, owner RC2 2, ADR 0049):
        yeast ATP6's CDS writes transl_table=3, under which codon 68's TGA is Trp, so the
        stated W68X is placed; the standard table would read it as a stop."""
        claim = GeneChangeClaim(
            label="atp6-w68",
            change=StatedChange(
                gene="ATP6",
                change="W68X",
                protein={"ref": "W", "codon": 68, "alt": "*"},
            ),
            request_text="ATP6 W68X",
        )

        answer = answer_claim(
            self.ncbi_sources("biology-genetic-code"), "GCF_000146045.2", claim
        )

        self.assertEqual(answer.outcome, "candidates", answer)
        [read] = answer.transcripts
        self.assertEqual(
            (
                read.genetic_code,
                read.genetic_code_origin,
                read.reference_codon,
                read.reference_residue,
            ),
            (3, "copied", "TGA", "W"),
        )
        # Under table 3 a stop is TAA or TAG: each candidate's codon is one of them.
        self.assertEqual(
            {candidate.alt_codon for candidate in answer.candidates}, {"TAA", "TAG"}
        )


class PaperTests(unittest.TestCase):
    def test_the_model_names_a_rows_ref_and_alt_cells_without_ref_alt_label_words(
        self,
    ) -> None:
        # The model names which cells are REF and ALT; code copies them from the cited
        # cells, whatever their columns are called.
        table = fixture("paper-allele-labels").inputs["table.tsv"].decode()
        paper = Paper(None, [document("main", f"# Results\nTable 1\n{table}")])

        judged = judge(
            FindingPointer.model_validate(
                finding(
                    value("contig", "chr2", "t1:r2", column="Chromosome"),
                    value(
                        "position",
                        "1630127",
                        "t1:r2",
                        column="Position",
                        number=1630127,
                    ),
                    value("ref", "C", "t1:r2", column="Major allele"),
                    value("alt", "A", "t1:r2", column="Minor allele"),
                )
            ),
            paper,
        )

        self.assertEqual((judged.outcome, judged.reasons), ("accepted", ()))
        [variant] = kept_finding(judged).variants
        self.assertEqual((variant.ref, variant.alt), ("C", "A"))


if __name__ == "__main__":
    unittest.main()
