from __future__ import annotations

import unittest
import uuid

from pydantic import ValidationError

from bio_neb.reference_discovery.reference_companions_tool import (
    CompanionRequirementGrounding,
    CompanionSourceRecord,
    CompanionToolCallRejected,
    DescribeCompanionInput,
    DescribeCompanionTool,
    VerifiedReferenceBinding,
    companion_asset_disposition_id,
    derive_companion_work_items,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    ReferenceAdapterError,
)
from tests.content_identity_vectors import expected_digest, fields
from tests.reference_fakes import (
    OFFLINE_ASSEMBLY_REPORT,
)
from tests.reference_fakes import (
    RecordedCompanionAdapter as FakeAdapter,
)
from tests.reference_fakes import (
    companion_record as record,
)

JOURNEY_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "companion-tools"))
PLAN_EVENT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "companion-plan"))
REFERENCE_EVENT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "companion-reference"))
SELECTION_EVENT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "companion-selection"))


def work_item(role: str = "genomic_fasta"):
    grounding = CompanionRequirementGrounding(
        journey_id=JOURNEY_ID,
        plan_id="plan_" + "a" * 64,
        requirement_id="requirement_" + "b" * 32,
        subject_key="reference_companions",
        source_event_ids=(PLAN_EVENT_ID, REFERENCE_EVENT_ID, SELECTION_EVENT_ID),
    )
    reference = VerifiedReferenceBinding(
        reference_event_id=REFERENCE_EVENT_ID,
        assembly_accession="GCF_000001405.40",
        reference_evidence_id="reference:verified",
    )
    return derive_companion_work_items(
        grounding=grounding,
        reference=reference,
        selected_roles=(role,),
        role_selection_event_ids=(SELECTION_EVENT_ID,),
    )[0]


class ReferenceCompanionToolTests(unittest.TestCase):
    def test_work_collection_is_hitl_derived_and_stable(self):
        first = work_item()
        second = work_item()
        self.assertEqual(first, second)
        self.assertEqual(first.role, "genomic_fasta")
        self.assertEqual(first.grounding.plan_id, "plan_" + "a" * 64)
        self.assertEqual(first.grounding.requirement_id, "requirement_" + "b" * 32)
        self.assertEqual(first.grounding.subject_key, "reference_companions")
        material = {
            "plan_id": "plan_" + "a" * 64,
            "requirement_id": "requirement_" + "b" * 32,
            "subject_key": "reference_companions",
            "reference_event_id": REFERENCE_EVENT_ID,
            "assembly_accession": "GCF_000001405.40",
            "role": "genomic_fasta",
        }
        self.assertEqual(
            first.work_item_id, "companion-work:" + expected_digest(material)
        )
        disposition = {
            "work_item_id": first.work_item_id,
            "asset_id": "source-owned:café",
        }
        self.assertEqual(
            companion_asset_disposition_id(first.work_item_id, disposition["asset_id"]),
            "companion-asset-disposition:" + expected_digest(disposition),
        )
        with self.assertRaisesRegex(ValueError, "Planning HITL"):
            derive_companion_work_items(
                grounding=first.grounding,
                reference=first.reference,
                selected_roles=("genomic_fasta",),
                role_selection_event_ids=(),
            )

    def test_exact_source_identity_survives_and_one_compatible_asset_verifies(self):
        expected = work_item()
        opaque = "opaque/source-owned café id with punctuation:42"
        result = DescribeCompanionTool().describe(
            DescribeCompanionInput(
                work_item=expected, assembly_accession="GCF_000001405.40"
            ),
            expected_work_item=expected,
            adapter=FakeAdapter((record(opaque),)),
        )
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.outcome, "verified")
        self.assertEqual(result.candidates[0].asset_id, opaque)
        candidate = result.candidates[0]
        source_material = fields(
            result.source.model_dump(mode="json"),
            "authority adapter_version source_version request_identity response_sha256 "
            "observed_at citation_url elapsed_time_ms response_bytes",
        )
        candidate_material = {
            "asset": fields(
                record(opaque).model_dump(mode="json"),
                "asset_id assembly_accession role publisher release_version representation "
                "byte_size checksum_algorithm checksum authority_locator",
            ),
            "source": source_material,
            "work_item_id": expected.work_item_id,
            "compatibility": "compatible",
        }
        self.assertEqual(
            candidate.evidence_id,
            "companion-candidate:" + expected_digest(candidate_material),
        )
        result_material = {
            "assembly_accession": "GCF_000001405.40",
            "work_item": fields(
                expected.model_dump(mode="json"),
                "work_item_id grounding reference role",
            ),
            "selected_asset_id": None,
            "selection_event_id": None,
            "outcome": "verified",
            "candidates": [
                fields(
                    candidate.model_dump(mode="json"),
                    "asset_id assembly_accession role publisher release_version representation "
                    "byte_size checksum_algorithm checksum authority_locator compatibility source evidence_id",
                )
            ],
            "source": source_material,
            "source_truncated": False,
        }
        self.assertEqual(
            result.evidence_id, "companion-result:" + expected_digest(result_material)
        )
        self.assertIsNone(result.candidates[0].checksum)
        self.assertNotEqual(
            result.candidates[0].source.response_sha256, result.candidates[0].checksum
        )

    def test_publisher_file_identity_reaches_the_candidate_and_its_evidence(self):
        expected = work_item()
        directory = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
            "GCF_000001405.40_GRCh38.p14"
        )
        published = record("all_genomic_fasta").model_copy(
            update={
                "source_locator": directory
                + "/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
                "byte_size": 972_898_531,
                "checksum_algorithm": "md5",
                "checksum": "c" * 32,
                "checksum_authority_locator": directory + "/md5checksums.txt",
            }
        )
        result = DescribeCompanionTool().describe(
            DescribeCompanionInput(
                work_item=expected, assembly_accession="GCF_000001405.40"
            ),
            expected_work_item=expected,
            adapter=FakeAdapter((published,)),
        )
        self.assertEqual(result.outcome, "verified")
        (candidate,) = result.candidates
        self.assertEqual(
            (
                candidate.source_locator,
                candidate.byte_size,
                candidate.checksum_algorithm,
                candidate.checksum,
                candidate.checksum_authority_locator,
            ),
            (
                directory + "/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
                972_898_531,
                "md5",
                "c" * 32,
                directory + "/md5checksums.txt",
            ),
        )
        candidate_material = {
            "asset": {
                "asset_id": "all_genomic_fasta",
                "assembly_accession": "GCF_000001405.40",
                "role": "genomic_fasta",
                "publisher": "NCBI Datasets",
                "release_version": "GCF_000001405.40",
                "representation": "GENOMIC_NUCLEOTIDE_FASTA",
                "byte_size": 972_898_531,
                "checksum_algorithm": "md5",
                "checksum": "c" * 32,
                "authority_locator": "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/"
                "accession/GCF_000001405.40/download_summary",
                "source_locator": directory
                + "/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
                "checksum_authority_locator": directory + "/md5checksums.txt",
            },
            "source": result.source.model_dump(mode="json"),
            "work_item_id": expected.work_item_id,
            "compatibility": "compatible",
            # The Run's reference sequences, read once from NCBI's assembly report
            # beside the verified FASTA: the fake report's one sequence.
            "reference_sequences": [{"name": "chr-offline", "length": 16}],
        }
        self.assertEqual(
            [(c.name, c.length) for c in candidate.reference_sequences],
            [("chr-offline", 16)],
        )
        self.assertEqual(
            candidate.evidence_id,
            "companion-candidate:" + expected_digest(candidate_material),
        )

    def test_the_report_splits_into_the_runs_sequences_and_the_names_to_drop(self):
        """Owner, #464 ("primey for the ref we are running"): alt loci and patches are
        outside the Run's reference; their FASTA names are carried for the CPU build to
        drop, and they reach the candidate's evidence."""
        expected = work_item()
        adapter = FakeAdapter((record("all_genomic_fasta", observed_source=True),))
        adapter.report = OFFLINE_ASSEMBLY_REPORT + "".join(
            f"{name}\t{role}\t1\tChromosome\t{genbank}\t=\t{refseq}"
            f"\t{unit}\t{length}\tna\n"
            for name, role, genbank, refseq, unit, length in (
                ("un1", "unplaced-scaffold", "KI1.1", "NT_1.1", "Primary Assembly", 9),
                ("alt1", "alt-scaffold", "KI2.1", "NT_2.1", "ALT_REF_LOCI_1", 8),
                ("fix1", "fix-patch", "ML3.1", "NW_3.1", "PATCHES", 7),
                ("nov1", "novel-patch", "KN4.1", "na", "PATCHES", 6),
            )
        )

        result = DescribeCompanionTool().describe(
            DescribeCompanionInput(
                work_item=expected, assembly_accession="GCF_000001405.40"
            ),
            expected_work_item=expected,
            adapter=adapter,
        )

        (candidate,) = result.candidates
        self.assertEqual(
            [c.name for c in candidate.reference_sequences], ["chr-offline", "NT_1.1"]
        )
        # A GCF assembly's FASTA names RefSeq accessions; a row without one is in
        # no FASTA of it, so there is nothing to drop.
        self.assertEqual(candidate.excluded_sequences, ("NT_2.1", "NW_3.1"))
        material = {
            "asset": record("all_genomic_fasta", observed_source=True).model_dump(
                mode="json"
            ),
            "source": result.source.model_dump(mode="json"),
            "work_item_id": expected.work_item_id,
            "compatibility": "compatible",
            "reference_sequences": [
                {"name": "chr-offline", "length": 16},
                {"name": "NT_1.1", "length": 9},
            ],
            "excluded_sequences": ["NT_2.1", "NW_3.1"],
        }
        self.assertEqual(
            candidate.evidence_id,
            "companion-candidate:" + expected_digest(material),
        )

    def test_absent_partial_multiple_ambiguous_and_incompatible_remain_typed(self):
        expected = work_item()
        tool = DescribeCompanionTool()

        def outcome(records):
            return tool.describe(
                DescribeCompanionInput(
                    work_item=expected, assembly_accession="GCF_000001405.40"
                ),
                expected_work_item=expected,
                adapter=FakeAdapter(records),
            ).outcome

        self.assertEqual(outcome(()), "absent")
        self.assertEqual(outcome((record(authority=False),)), "partial")
        self.assertEqual(outcome((record("asset-1"), record("asset-2"))), "multiple")
        self.assertEqual(
            outcome(
                (
                    record("asset-1"),
                    record("wrong", assembly_accession="GCF_000001405.39"),
                )
            ),
            "ambiguous",
        )
        self.assertEqual(
            outcome((record("wrong", assembly_accession="GCF_000001405.39"),)),
            "incompatible",
        )

    def test_selection_is_disposition_but_still_tool_validated(self):
        expected = work_item()
        selection_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "asset-selection"))
        grounded = expected.model_copy(
            update={
                "grounding": expected.grounding.model_copy(
                    update={
                        "source_event_ids": (
                            *expected.grounding.source_event_ids,
                            selection_id,
                        )
                    }
                )
            }
        )
        selected = DescribeCompanionInput(
            work_item=grounded,
            assembly_accession="GCF_000001405.40",
            selected_asset_id="asset-wrong",
            selection_event_id=selection_id,
        )
        result = DescribeCompanionTool().describe(
            selected,
            expected_work_item=grounded,
            adapter=FakeAdapter(
                (record("asset-wrong", assembly_accession="GCF_000001405.39"),)
            ),
        )
        self.assertEqual(result.outcome, "incompatible")

    def test_strict_work_item_mismatch_rejects_before_contact(self):
        expected = work_item()
        annotation = work_item("annotation")
        adapter = FakeAdapter((record(),))
        with self.assertRaisesRegex(CompanionToolCallRejected, "ungrounded"):
            DescribeCompanionTool().describe(
                DescribeCompanionInput(
                    work_item=annotation, assembly_accession="GCF_000001405.40"
                ),
                expected_work_item=expected,
                adapter=adapter,
            )
        self.assertEqual(adapter.contact_count, 0)

    def test_transport_failure_is_operational_not_scientific_negative(self):
        expected = work_item()
        result = DescribeCompanionTool().describe(
            DescribeCompanionInput(
                work_item=expected, assembly_accession="GCF_000001405.40"
            ),
            expected_work_item=expected,
            adapter=FakeAdapter(
                error=ReferenceAdapterError(
                    "transport_error", "source unavailable", retryable=True
                )
            ),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure.code, "transport_error")
        # The shared operation window retries the transient fault and retains how
        # it ended; the evidence id covers that whole failure record.
        self.assertEqual(
            (
                result.failure.original_failure_class,
                result.failure.failure_class,
                result.failure.contact_attempt,
            ),
            ("transient", "exhausted", 3),
        )
        failure_material = {
            "work_item_id": expected.work_item_id,
            "failure": result.failure.model_dump(mode="json"),
        }
        self.assertEqual(
            result.evidence_id, "companion-failure:" + expected_digest(failure_material)
        )
        self.assertFalse(hasattr(result, "outcome"))

    def test_contracts_reject_unversioned_reference_and_invented_checksum(self):
        with self.assertRaises(ValidationError):
            CompanionSourceRecord(
                asset_id="opaque",
                assembly_accession="GRCh38",
                role="genomic_fasta",
                publisher="NCBI Datasets",
                representation="GENOMIC_NUCLEOTIDE_FASTA",
            )
        with self.assertRaises(ValidationError):
            CompanionSourceRecord(
                asset_id="opaque",
                assembly_accession="GCF_000001405.40",
                role="genomic_fasta",
                publisher="NCBI Datasets",
                representation="GENOMIC_NUCLEOTIDE_FASTA",
                checksum_algorithm="sha256",
                checksum="e" * 32,
            )
        for invented in (
            {
                "checksum_authority_locator": "https://ftp.ncbi.nlm.nih.gov/x/md5checksums.txt"
            },
            {"source_locator": "ftp://ftp.ncbi.nlm.nih.gov/x_genomic.fna.gz"},
        ):
            with self.subTest(invented=invented), self.assertRaises(ValidationError):
                record().model_validate({**record().model_dump(), **invented})


if __name__ == "__main__":
    unittest.main()


class CompanionPortTests(unittest.TestCase):
    """The companion port declares every read its Tool makes: a verified genomic
    FASTA's description reads the assembly's sequences (standards r1 ST3; ADR 0020)."""

    def test_an_adapter_without_the_sequence_read_is_not_the_port(self):
        from bio_neb.reference_discovery.reference_companions_tool import (
            ReferenceCompanionAdapter,
        )

        class Advertised:
            contact_count = 0

            def describe_companions(self, assembly_accession, role):
                raise AssertionError((assembly_accession, role))

        self.assertNotIsInstance(Advertised(), ReferenceCompanionAdapter)

    def test_each_supplied_adapter_conforms(self):
        from bio_neb.providers.ncbi_datasets import (
            NcbiDatasetsReferenceAdapter,
            UrllibNcbiTransport,
        )
        from bio_neb.reference_discovery.reference_companions_tool import (
            ReferenceCompanionAdapter,
        )
        from tests.reference_fakes import CompanionAdapter

        for adapter in (
            NcbiDatasetsReferenceAdapter(UrllibNcbiTransport()),
            CompanionAdapter("GCF_000001405.40"),
            FakeAdapter(),
        ):
            with self.subTest(adapter=type(adapter).__name__):
                self.assertIsInstance(adapter, ReferenceCompanionAdapter)
