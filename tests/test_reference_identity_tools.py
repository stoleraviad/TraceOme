"""Focused contracts for grounded sample, taxon, and reference identity Tools."""

from __future__ import annotations

import json
import unittest
import uuid
from datetime import UTC, datetime

from pydantic import ValidationError

from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_RULE,
    NCBI_TAXON_ID_RULE,
    SAMPLE_IDENTIFIER_RULE,
    AssemblyAdapterObservation,
    AssemblyQuery,
    AssemblySourceRecord,
    DescribeReferenceInput,
    DescribeReferenceTool,
    DiscoverReferenceAssembliesInput,
    DiscoverReferenceAssembliesTool,
    InfraspecificNames,
    ReferenceAdapterError,
    ReferenceToolCallRejected,
    RequirementGrounding,
    ResolvedTaxonEvidence,
    ResolveTaxonInput,
    ResolveTaxonTool,
    SampleAttestationInput,
    SelectedAssemblyEvidence,
    SourceObservation,
    TaxonomyAdapterObservation,
    TaxonomySourceRecord,
    VerifySampleAttestationTool,
)
from tests.content_identity_vectors import assert_json_identity, expected_digest, fields

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def event_id() -> str:
    return str(uuid.uuid4())


def grounding(
    kind: str, *, sources: tuple[str, ...] | None = None
) -> RequirementGrounding:
    return RequirementGrounding(
        journey_id=str(uuid.uuid4()),
        plan_id="plan_" + "a" * 64,
        requirement_id="requirement_" + "b" * 32,
        kind=kind,
        subject_key={
            "sample_identity": "sample_identity",
            "taxon_identity": "taxon_identity",
            "reference_assembly_identity": "reference_assembly",
        }[kind],
        source_event_ids=sources or (event_id(),),
    )


def source(
    authority: str = "ncbi_datasets", *, request: str = "assembly"
) -> SourceObservation:
    return SourceObservation(
        authority=authority,
        adapter_version="recorded/v1",
        source_version="recorded-source-v1",
        request_identity=f"https://api.ncbi.nlm.nih.gov/{request}",
        response_sha256="c" * 64,
        observed_at=NOW,
        citation_url=f"https://api.ncbi.nlm.nih.gov/{request}",
        elapsed_time_ms=7,
        response_bytes=100,
    )


def assembly(
    accession: str,
    *,
    status: str = "current",
    tax_id: int = 9606,
    strain: str | None = None,
    category: str | None = "reference genome",
    level: str | None = "Chromosome",
    current_accession: str | None = None,
) -> AssemblySourceRecord:
    return AssemblySourceRecord(
        assembly_accession=accession,
        assembly_name=f"assembly-{accession}",
        organism_name="Homo sapiens" if tax_id == 9606 else "Escherichia coli K-12",
        ncbi_tax_id=tax_id,
        infraspecific_names=InfraspecificNames(strain=strain),
        assembly_status=status,
        refseq_category=category,
        assembly_level=level,
        current_accession=current_accession,
    )


class FakeAdapter:
    def __init__(
        self,
        *,
        taxonomy: object | None = None,
        discovery: object | None = None,
        exact: object | None = None,
        failure: ReferenceAdapterError | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.discovery = discovery
        self.exact = exact
        self.failure = failure
        self.contact_count = 0
        self.calls: list[tuple[str, object]] = []

    def _result(self, name: str, argument: object, value: object) -> object:
        self.contact_count += 1
        self.calls.append((name, argument))
        if self.failure is not None:
            raise self.failure
        return value

    def resolve_taxon(self, query: str) -> TaxonomyAdapterObservation:
        return self._result("resolve_taxon", query, self.taxonomy)  # type: ignore[return-value]

    def discover_assemblies(
        self, ncbi_tax_id: int, query: AssemblyQuery
    ) -> AssemblyAdapterObservation:
        return self._result("discover_assemblies", (ncbi_tax_id, query), self.discovery)  # type: ignore[return-value]

    def describe_assembly(self, assembly_accession: str) -> AssemblyAdapterObservation:
        return self._result("describe_assembly", assembly_accession, self.exact)  # type: ignore[return-value]


def resolved_taxon(tax_id: int = 9606) -> ResolvedTaxonEvidence:
    return ResolvedTaxonEvidence(
        candidate_id=f"taxon:ncbitaxon:{tax_id}",
        ncbi_tax_id=tax_id,
        candidate_evidence_id="taxon-candidate:" + "d" * 64,
        result_evidence_id="taxon-result:" + "e" * 64,
        source_event_id=event_id(),
    )


def discovery_input(
    current_grounding: RequirementGrounding,
    taxon: ResolvedTaxonEvidence,
    **updates,
) -> DiscoverReferenceAssembliesInput:
    return DiscoverReferenceAssembliesInput(
        grounding=current_grounding,
        taxon=taxon,
        **updates,
    )


class ReferenceIdentifierRuleTests(unittest.TestCase):
    """Each rule is a format matcher: a typed value matches a quote that states it."""

    def test_sample_identifier_matches_a_whole_safe_token_of_its_quote(self) -> None:
        for quote in ("HG001", "Sample HG001, human", "sample HG001."):
            with self.subTest(quote=quote):
                self.assertTrue(SAMPLE_IDENTIFIER_RULE.matches("HG001", quote))
        for value, quote in (
            ("HG00", "Sample HG001"),
            ("hg001", "Sample HG001"),
            ("HG001", "Sample HG/001"),
            ("HG001 ", "Sample HG001"),
        ):
            with self.subTest(value=value, quote=quote):
                self.assertFalse(SAMPLE_IDENTIFIER_RULE.matches(value, quote))
        self.assertIsNone(SAMPLE_IDENTIFIER_RULE.source)

    def test_sample_identifier_characters_are_part_of_the_identifier(self) -> None:
        # A legal trailing ``_``/``-``/``.`` belongs to the identifier; only a
        # sentence's closing full stop may follow a whole identifier.
        for value in ("HG001_", "HG001-", "HG001."):
            with self.subTest(value=value):
                self.assertTrue(SAMPLE_IDENTIFIER_RULE.matches(value, value))
        for quote in ("HG001_", "HG001-", "HG001_x"):
            with self.subTest(quote=quote):
                self.assertFalse(SAMPLE_IDENTIFIER_RULE.matches("HG001", quote))
        self.assertTrue(
            SAMPLE_IDENTIFIER_RULE.matches("HG001", "The sample is HG001. Thanks")
        )

    def test_taxon_id_matches_a_whole_number_its_quote_states(self) -> None:
        for quote in (
            "9606",
            "The organism is 9606.",
            "human (NCBI Taxonomy 9606)",
            "Homo sapiens (TaxId: 9606)",
            "Homo sapiens, NCBI Taxonomy ID 9606",
        ):
            with self.subTest(quote=quote):
                self.assertTrue(NCBI_TAXON_ID_RULE.matches("9606", quote))
        for value, quote in (
            ("9606", "human"),
            ("960", "NCBI Taxonomy 9606"),
            ("38", "GRCh38"),
            ("40", "GCF_000001405.40"),
            ("09606", "taxon 09606"),
            ("1", "1.2"),
            ("2", "1.2"),
            ("9606", "9606.5"),
        ):
            with self.subTest(value=value, quote=quote):
                self.assertFalse(NCBI_TAXON_ID_RULE.matches(value, quote))
        self.assertEqual(NCBI_TAXON_ID_RULE.source, "NCBI Taxonomy")

    def test_assembly_accession_matches_only_its_exact_versioned_spelling(self) -> None:
        self.assertTrue(
            ASSEMBLY_ACCESSION_RULE.matches(
                "GCF_000001405.40", "reference GCF_000001405.40 / GRCh38.p14"
            )
        )
        for value, quote in (
            ("GCF_000001405.40", "GRCh38"),
            ("GCF_000001405", "GCF_000001405.40"),
            ("gcf_000001405.40", "GCF_000001405.40"),
            ("GCF_000001405.40", "gcf_000001405.40"),
            ("GCF_000001405.40", "GCF_000001405.40_GRCh38.p14_genomic.fna.gz"),
            ("GCF_000001405.39", "GCF_000001405.40"),
        ):
            with self.subTest(value=value, quote=quote):
                self.assertFalse(ASSEMBLY_ACCESSION_RULE.matches(value, quote))
        self.assertEqual(ASSEMBLY_ACCESSION_RULE.source, "NCBI Datasets")

    def test_a_value_the_model_chose_from_several_is_admitted_by_its_quote(
        self,
    ) -> None:
        """ADR 0049 ("code reading free text for a value" is a rule a surface may not
        set), #472 Q5 a: an answer stating several values leaves the choice to the
        model, and each is admitted by membership in the words it quotes; nothing
        picks one. Replaces B38 (`answer_value`, retired)."""
        answer = "Use GCF_000001405.40, or else GCA_000001405.29."
        for value in ("GCF_000001405.40", "GCA_000001405.29"):
            with self.subTest(value=value):
                self.assertTrue(ASSEMBLY_ACCESSION_RULE.matches(value, answer))
        self.assertFalse(ASSEMBLY_ACCESSION_RULE.matches("GCF_000001405.39", answer))
        self.assertTrue(
            NCBI_TAXON_ID_RULE.matches("10090", "NCBI taxon 9606 or NCBI taxon 10090")
        )
        self.assertTrue(SAMPLE_IDENTIFIER_RULE.matches("HG001", "The sample is HG001."))
        self.assertFalse(hasattr(ASSEMBLY_ACCESSION_RULE, "answer_value"))


class SampleAttestationTests(unittest.TestCase):
    def test_exact_attribution_is_verified_but_filename_identity_is_not_inferred(
        self,
    ) -> None:
        assertion = event_id()
        current = grounding("sample_identity", sources=(assertion,))
        input_set_id = event_id()
        tool_input = SampleAttestationInput(
            grounding=current,
            input_set_id=input_set_id,
            sample_identifier="sample-001",
            assertion_event_id=assertion,
            quote="sample-001",
        )

        result = VerifySampleAttestationTool().verify(
            tool_input,
            expected_grounding=current,
            expected_input_set_id=input_set_id,
            expected_sample_identifier="sample-001",
        )

        self.assertEqual((result.status, result.resolution), ("verified", "bound"))
        self.assertEqual(result.sample_identifier, "sample-001")
        self.assertNotIn("filename", result.model_dump_json())

    def test_invalid_sample_identifier_format_is_a_truthful_negative(self) -> None:
        assertion = event_id()
        current = grounding("sample_identity", sources=(assertion,))
        invalid = SampleAttestationInput(
            grounding=current,
            input_set_id=event_id(),
            sample_identifier="sample with spaces",
            assertion_event_id=assertion,
            quote="sample with spaces",
        )
        negative = VerifySampleAttestationTool().verify(
            invalid,
            expected_grounding=current,
            expected_input_set_id=invalid.input_set_id,
            expected_sample_identifier="sample with spaces",
        )
        self.assertEqual(
            (negative.status, negative.resolution), ("negative", "invalid_format")
        )

    def test_wrong_sample_input_set_is_rejected(self) -> None:
        assertion = event_id()
        current = grounding("sample_identity", sources=(assertion,))
        invalid = SampleAttestationInput(
            grounding=current,
            input_set_id=event_id(),
            sample_identifier="sample-001",
            assertion_event_id=assertion,
            quote="sample-001",
        )
        with self.assertRaisesRegex(
            ReferenceToolCallRejected, "sample_input_set_ungrounded"
        ):
            VerifySampleAttestationTool().verify(
                invalid,
                expected_grounding=current,
                expected_input_set_id=event_id(),
                expected_sample_identifier="sample-001",
            )

    def test_changed_asserted_identifier_is_rejected_with_exact_event_identity(
        self,
    ) -> None:
        assertion = event_id()
        current = grounding("sample_identity", sources=(assertion,))
        input_set_id = event_id()
        with self.assertRaisesRegex(
            ReferenceToolCallRejected, "sample_identifier_ungrounded"
        ):
            VerifySampleAttestationTool().verify(
                SampleAttestationInput(
                    grounding=current,
                    input_set_id=input_set_id,
                    sample_identifier="invented-sample",
                    assertion_event_id=assertion,
                    quote="invented-sample",
                ),
                expected_grounding=current,
                expected_input_set_id=input_set_id,
                expected_sample_identifier="asserted-sample",
            )


class TaxonToolTests(unittest.TestCase):
    def test_positive_negative_and_ambiguous_taxonomy_are_observations(self) -> None:
        """One taxon resolves whatever NCBI's provisional or deleted flag says: the
        flag is copied onto the candidate and is the model's to weigh (#471 Q5 a,
        #375 C5). Two taxa stay ``ambiguous`` under the owner's conflict rule: a real
        ambiguity goes to the Review and is never silently picked."""
        assertion = event_id()
        current = grounding("taxon_identity", sources=(assertion,))
        tool_input = ResolveTaxonInput(
            grounding=current,
            query="human",
            assertion_event_id=assertion,
        )
        cases = (
            (
                (
                    TaxonomySourceRecord(
                        ncbi_tax_id=9606, scientific_name="Homo sapiens", rank="species"
                    ),
                ),
                "resolved",
            ),
            (
                (
                    TaxonomySourceRecord(
                        ncbi_tax_id=9606,
                        scientific_name="Homo sapiens",
                        provisional=True,
                    ),
                ),
                "resolved",
            ),
            (
                (
                    TaxonomySourceRecord(
                        ncbi_tax_id=9606, scientific_name="Homo sapiens", deleted=True
                    ),
                ),
                "resolved",
            ),
            ((), "not_found"),
            (
                (
                    TaxonomySourceRecord(
                        ncbi_tax_id=10090, scientific_name="Mus musculus"
                    ),
                    TaxonomySourceRecord(
                        ncbi_tax_id=10091, scientific_name="Mus spretus"
                    ),
                ),
                "ambiguous",
            ),
        )
        for records, expected in cases:
            with self.subTest(expected=expected, records=records):
                adapter = FakeAdapter(
                    taxonomy=TaxonomyAdapterObservation(
                        records=records,
                        source=source("ncbi_taxonomy", request="taxonomy/human"),
                    )
                )
                result = ResolveTaxonTool().resolve(
                    tool_input,
                    expected_grounding=current,
                    expected_query="human",
                    adapter=adapter,
                )
                self.assertEqual(result.status, "observed")
                self.assertEqual(result.resolution, expected)
                self.assertEqual(
                    [c.identity_status for c in result.candidates],
                    [
                        "deleted"
                        if r.deleted
                        else "provisional"
                        if r.provisional
                        else "resolved"
                        for r in records
                    ],
                )
                assert_json_identity(
                    self,
                    result.evidence_id,
                    result,
                    "grounding query resolution candidates source",
                    "taxon-result:",
                )
                assert_json_identity(
                    self,
                    result.source.observation_id,
                    result.source,
                    "authority adapter_version source_version request_identity response_sha256 "
                    "observed_at citation_url elapsed_time_ms response_bytes",
                    "source-observation:",
                )
                by_taxon = {record.ncbi_tax_id: record for record in records}
                for candidate in result.candidates:
                    material = {
                        "record": fields(
                            by_taxon[candidate.ncbi_tax_id].model_dump(mode="json"),
                            "ncbi_tax_id scientific_name rank lineage lineage_tax_ids "
                            "merged_or_secondary_tax_ids provisional deleted",
                        ),
                        "query": "human",
                        "source": fields(
                            result.source.model_dump(mode="json"),
                            "authority adapter_version source_version request_identity response_sha256 "
                            "observed_at citation_url elapsed_time_ms response_bytes",
                        ),
                    }
                    self.assertEqual(
                        candidate.evidence_id,
                        "taxon-candidate:" + expected_digest(material),
                    )
                self.assertEqual(adapter.contact_count, 1)

    def test_cross_subject_call_is_rejected_before_contact(self) -> None:
        assertion = event_id()
        current = grounding("taxon_identity", sources=(assertion,))
        different = current.model_copy(
            update={"requirement_id": "requirement_" + "9" * 32}
        )
        adapter = FakeAdapter()
        with self.assertRaisesRegex(
            ReferenceToolCallRejected, "requirement_subject_ungrounded"
        ):
            ResolveTaxonTool().resolve(
                ResolveTaxonInput(
                    grounding=different,
                    query="9606",
                    assertion_event_id=assertion,
                ),
                expected_grounding=current,
                expected_query="9606",
                adapter=adapter,
            )
        self.assertEqual(adapter.contact_count, 0)

    def test_changed_query_is_rejected_before_contact_with_exact_event_identity(
        self,
    ) -> None:
        assertion = event_id()
        current = grounding("taxon_identity", sources=(assertion,))
        adapter = FakeAdapter()
        with self.assertRaisesRegex(
            ReferenceToolCallRejected, "taxon_query_ungrounded"
        ):
            ResolveTaxonTool().resolve(
                ResolveTaxonInput(
                    grounding=current,
                    query="invented organism",
                    assertion_event_id=assertion,
                ),
                expected_grounding=current,
                expected_query="Homo sapiens",
                adapter=adapter,
            )
        self.assertEqual(adapter.contact_count, 0)

    def test_operational_failure_is_not_scientific_negative(self) -> None:
        assertion = event_id()
        current = grounding("taxon_identity", sources=(assertion,))
        adapter = FakeAdapter(
            failure=ReferenceAdapterError(
                "transport_error", "source unavailable", retryable=True
            )
        )
        result = ResolveTaxonTool().resolve(
            ResolveTaxonInput(
                grounding=current, query="9606", assertion_event_id=assertion
            ),
            expected_grounding=current,
            expected_query="9606",
            adapter=adapter,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure.code, "transport_error")
        # The shared operation window: three attempts by default.
        self.assertEqual(adapter.contact_count, 3)


class AssemblyDiscoveryTests(unittest.TestCase):
    def test_every_row_is_copied_in_ncbis_order_with_its_status(self) -> None:
        """A page of NCBI's genome reports reaches the model whole: every row in
        NCBI's order, a replaced or suppressed one with its status, NCBI's count and
        next page token; code ranks, cuts and judges none (#471, ADR 0049)."""
        current = grounding("reference_assembly_identity")
        taxon = resolved_taxon()
        reports = (
            assembly("GCA_000001405.39", status="replaced", category=None),
            assembly("GCF_000009999.1", status="suppressed"),
            assembly("GCF_000001405.40"),
        )
        adapter = FakeAdapter(
            discovery=AssemblyAdapterObservation(
                records=reports,
                source=source(request="genome/taxon/9606"),
                total_count=2592,
                next_page_token="page-two",
            )
        )

        result = DiscoverReferenceAssembliesTool().discover(
            discovery_input(current, taxon),
            expected_grounding=current,
            expected_taxon=taxon,
            adapter=adapter,
        )

        self.assertEqual(result.status, "observed")
        self.assertEqual(result.rows, reports)
        self.assertEqual(
            [row.assembly_status for row in result.rows],
            ["replaced", "suppressed", "current"],
        )
        self.assertEqual(
            (result.total_count, result.next_page_token), (2592, "page-two")
        )
        self.assertNotIn("resolution", result.model_dump())

    def test_the_models_query_reaches_ncbi_as_it_chose_it(self) -> None:
        """The model's NCBI options go to the port unchanged, in NCBI's spelling:
        no strain substring, no lineage check, no attribution rank (#471 R1 a)."""
        current = grounding("reference_assembly_identity")
        taxon = resolved_taxon(562)
        query = AssemblyQuery(
            search_text=("MG1655",),
            assembly_source="refseq",
            assembly_level=("complete_genome",),
            page_size=20,
        )
        adapter = FakeAdapter(
            discovery=AssemblyAdapterObservation(
                records=(assembly("GCF_000005845.2", tax_id=511145, strain="K-12"),),
                source=source(request="genome/taxon/562"),
                total_count=1,
            )
        )

        result = DiscoverReferenceAssembliesTool().discover(
            discovery_input(current, taxon, query=query),
            expected_grounding=current,
            expected_taxon=taxon,
            adapter=adapter,
        )

        self.assertEqual(adapter.calls, [("discover_assemblies", (562, query))])
        self.assertEqual(
            query.parameters(),
            (
                ("filters.assembly_source", "refseq"),
                ("filters.assembly_level", "complete_genome"),
                ("filters.search_text", "MG1655"),
                ("page_size", "20"),
            ),
        )
        self.assertEqual(result.query, query)
        self.assertEqual(result.rows[0].infraspecific_names.strain, "K-12")

    def test_wrong_taxon_evidence_is_rejected_before_contact(self) -> None:
        current = grounding("reference_assembly_identity")
        expected = resolved_taxon(9606)
        supplied = resolved_taxon(10090)
        adapter = FakeAdapter()
        with self.assertRaisesRegex(
            ReferenceToolCallRejected, "resolved_taxon_ungrounded"
        ):
            DiscoverReferenceAssembliesTool().discover(
                discovery_input(current, supplied),
                expected_grounding=current,
                expected_taxon=expected,
                adapter=adapter,
            )
        self.assertEqual(adapter.contact_count, 0)

    def test_a_page_longer_than_the_model_asked_for_is_invalid_output(self) -> None:
        """A port that returns more rows than the page size asked is no NCBI page:
        the context-safety bound holds (#471 R2 c)."""
        current = grounding("reference_assembly_identity")
        taxon = resolved_taxon()
        adapter = FakeAdapter(
            discovery=AssemblyAdapterObservation(
                records=(assembly("GCF_000001405.40"), assembly("GCF_000001405.39")),
                source=source(request="genome/taxon/9606"),
                total_count=2,
            )
        )

        result = DiscoverReferenceAssembliesTool().discover(
            discovery_input(current, taxon, query=AssemblyQuery(page_size=1)),
            expected_grounding=current,
            expected_taxon=taxon,
            adapter=adapter,
        )

        self.assertEqual(
            (result.status, result.failure.code), ("failed", "adapter_output_invalid")
        )


class ExactReferenceTests(unittest.TestCase):
    def selected(
        self, current: RequirementGrounding
    ) -> tuple[SelectedAssemblyEvidence, FakeAdapter]:
        taxon = resolved_taxon()
        discovery_adapter = FakeAdapter(
            discovery=AssemblyAdapterObservation(
                records=(assembly("GCF_000001405.40"),),
                source=source(request="genome/taxon/9606"),
                total_count=1,
            )
        )
        discovery = DiscoverReferenceAssembliesTool().discover(
            discovery_input(current, taxon),
            expected_grounding=current,
            expected_taxon=taxon,
            adapter=discovery_adapter,
        )
        selected = SelectedAssemblyEvidence(
            row=discovery.rows[0],
            discovery_evidence_id=discovery.evidence_id,
            discovery_event_id=event_id(),
            selection_event_id=event_id(),
        )
        return selected, discovery_adapter

    def test_an_exact_rows_status_is_copied_not_judged(self) -> None:
        """A replaced or suppressed exact row resolves with NCBI's status copied
        onto it; code maps no status to a verdict (#471 Q5 a). Identity still
        decides: another TaxId than the selection's is ``source_disagreement``."""
        current = grounding("reference_assembly_identity")
        selected, _ = self.selected(current)
        for record, expected in (
            (assembly("GCF_000001405.40"), "resolved"),
            (
                assembly(
                    "GCF_000001405.40",
                    status="replaced",
                    current_accession="GCF_000001405.41",
                ),
                "resolved",
            ),
            (assembly("GCF_000001405.40", status="suppressed"), "resolved"),
            (assembly("GCF_000001405.40", tax_id=10090), "source_disagreement"),
        ):
            with self.subTest(expected=expected, status=record.assembly_status):
                adapter = FakeAdapter(
                    exact=AssemblyAdapterObservation(
                        records=(record,),
                        source=source(request="genome/accession/GCF_000001405.40"),
                        total_count=1,
                    )
                )
                result = DescribeReferenceTool().describe(
                    DescribeReferenceInput(
                        grounding=current,
                        selected=selected,
                        assembly_accession="GCF_000001405.40",
                    ),
                    expected_grounding=current,
                    expected_selection=selected,
                    adapter=adapter,
                )
                self.assertEqual(
                    (result.status, result.resolution), ("observed", expected)
                )
                self.assertEqual(result.references[0].record, record)

    def test_a_replaced_version_the_model_chose_is_described_by_ncbi(self) -> None:
        """NCBI answers a replaced version's plain report with exactly ``{}``; the
        step asks again for every version and describes it, so a GRCh37 row the
        model chose from a discovery page resolves (#471 live a2). Synthetic bodies
        in NCBI's shape; the plain ``{}`` is the answer live a2 recorded."""
        from bio_neb.providers.ncbi_datasets import (
            MetadataResponse,
            NcbiDatasetsReferenceAdapter,
        )

        report = {
            "accession": "GCF_000001405.25",
            "organism": {"organism_name": "Homo sapiens", "tax_id": 9606},
            "assembly_info": {
                "assembly_name": "GRCh37.p13",
                "assembly_status": "previous",
                "assembly_level": "Chromosome",
            },
        }
        bodies = [b"{}", json.dumps({"reports": [report], "total_count": 1}).encode()]
        urls: list[str] = []

        class Transport:
            def get(self, url, *, max_bytes):
                urls.append(url)
                return MetadataResponse(
                    request_identity=url,
                    citation_url=url,
                    status_code=200,
                    body=bodies.pop(0),
                    source_version="synthetic-ncbi",
                    observed_at=datetime(2026, 10, 28, tzinfo=UTC),
                    elapsed_time_ms=1,
                )

        current = grounding("reference_assembly_identity")
        taxon = resolved_taxon()
        discovery = DiscoverReferenceAssembliesTool().discover(
            discovery_input(current, taxon),
            expected_grounding=current,
            expected_taxon=taxon,
            adapter=FakeAdapter(
                discovery=AssemblyAdapterObservation(
                    records=(assembly("GCF_000001405.25", status="previous"),),
                    source=source(request="genome/taxon/9606"),
                    total_count=1,
                )
            ),
        )
        selected = SelectedAssemblyEvidence(
            row=discovery.rows[0],
            discovery_evidence_id=discovery.evidence_id,
            discovery_event_id=event_id(),
            selection_event_id=event_id(),
        )

        result = DescribeReferenceTool().describe(
            DescribeReferenceInput(
                grounding=current,
                selected=selected,
                assembly_accession="GCF_000001405.25",
            ),
            expected_grounding=current,
            expected_selection=selected,
            adapter=NcbiDatasetsReferenceAdapter(Transport()),
        )

        plain = (
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            "GCF_000001405.25/dataset_report"
        )
        self.assertEqual(
            urls, [plain, f"{plain}?filters.assembly_version=all_assemblies"]
        )
        self.assertEqual((result.status, result.resolution), ("observed", "resolved"))
        self.assertEqual(result.references[0].record.assembly_status, "previous")

    def test_stale_selection_is_rejected_before_contact(self) -> None:
        current = grounding("reference_assembly_identity")
        selected, _ = self.selected(current)
        stale = selected.model_copy(update={"discovery_event_id": event_id()})
        adapter = FakeAdapter()
        with self.assertRaisesRegex(
            ReferenceToolCallRejected, "selected_reference_ungrounded"
        ):
            DescribeReferenceTool().describe(
                DescribeReferenceInput(
                    grounding=current,
                    selected=stale,
                    assembly_accession=stale.row.assembly_accession,
                ),
                expected_grounding=current,
                expected_selection=selected,
                adapter=adapter,
            )
        self.assertEqual(adapter.contact_count, 0)


class ContractStrictnessTests(unittest.TestCase):
    def test_contracts_forbid_extras_unversioned_accessions_and_undocumented_options(
        self,
    ) -> None:
        """An NCBI option takes only NCBI's documented values, and a page at most
        the context-safety bound of 100 rows (#471 R1 a, R2 c)."""
        current = grounding("reference_assembly_identity")
        with self.assertRaises(ValidationError):
            RequirementGrounding.model_validate({**current.model_dump(), "extra": True})
        with self.assertRaises(ValidationError):
            AssemblySourceRecord(assembly_accession="GRCh38")
        for options in (
            {"assembly_source": "both"},
            {"assembly_level": ("Chromosome",)},
            {"assembly_version": "latest"},
            {"page_size": 101},
            {"page_size": 0},
            {"first_release_date": "2020"},
            {"strain_or_isolate": "K-12"},
        ):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                AssemblyQuery(**options)


if __name__ == "__main__":
    unittest.main()
