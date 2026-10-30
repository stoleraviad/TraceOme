"""Current semantic scientific continuation publication and read-only reopen."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb.investigation import scientific_continuation as continuation
from bio_neb.journey import canonical_json
from bio_neb.parabricks import result as parabricks_result
from tests.run_prerequisites import prerequisite
from tests.terminal_journey_fakes import published_continuation


class ScientificContinuationTests(unittest.TestCase):
    def test_publication_uses_real_terminal_projection_and_reopens_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            command_file = str(prerequisite("executed").journey_file)
            journey_file = Path(command_file)
            output = root / "scientific-continuation.json"
            before = journey_file.read_bytes()

            # Execution path events are retained inside ExecutionPathRecorded
            # envelopes; their identity is the wrapped event.
            timeline = [
                event["payload"]
                if event["event_type"] == "ExecutionPathRecorded"
                else event
                for event in map(json.loads, journey_file.read_bytes().splitlines())
            ]
            receipt = continuation.publish_scientific_continuation(
                continuation.PublishScientificContinuationCommand(
                    execution_journey_file=command_file,
                    output=str(output),
                )
            )

            payload = output.read_bytes()
            reopened = continuation.ScientificContinuationPacket.model_validate_json(
                payload, strict=True
            )
            self._assert_public_evidence_meaning(reopened, timeline)
            self.assertEqual(
                payload,
                canonical_json(reopened.model_dump(mode="json")).encode("utf-8"),
            )
            self.assertFalse(payload.endswith(b"\n"))
            self.assertEqual(receipt.byte_count, len(payload))
            self.assertEqual(receipt.continuation_id, reopened.continuation_id)
            self.assertEqual(receipt.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(journey_file.read_bytes(), before)

            with (
                patch("socket.socket", side_effect=AssertionError("network contact")),
                patch.object(
                    subprocess,
                    "Popen",
                    side_effect=AssertionError("process contact"),
                ),
                patch.object(
                    continuation.tempfile,
                    "mkstemp",
                    side_effect=AssertionError("replay write"),
                ),
                self.assertRaises(
                    continuation.ScientificContinuationPublicationRefused
                ) as raised,
            ):
                continuation.publish_scientific_continuation(
                    continuation.PublishScientificContinuationCommand(
                        execution_journey_file=command_file,
                        output=str(output),
                    )
                )
            self.assertEqual(
                raised.exception.code, "scientific_continuation_output_exists"
            )
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(journey_file.read_bytes(), before)

    def test_publication_refuses_an_input_the_run_never_acquired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            command_file = str(prerequisite("executed").journey_file)
            output = root / "scientific-continuation.json"
            interpret = parabricks_result.parabricks_result

            # Deliberately malformed retained result: read_2 has no acquired
            # transfer proof, and preparation supplied no expected checksum.
            def without_read_2_proof(journey):
                result = interpret(journey)
                return result.model_copy(
                    update={
                        "input_proofs": tuple(
                            proof
                            for proof in result.input_proofs
                            if proof.role != "read_2"
                        )
                    }
                )

            with (
                patch.object(
                    parabricks_result,
                    "parabricks_result",
                    side_effect=without_read_2_proof,
                ),
                self.assertRaises(continuation.ScientificContinuationRefused) as raised,
            ):
                continuation.publish_scientific_continuation(
                    continuation.PublishScientificContinuationCommand(
                        execution_journey_file=command_file,
                        output=str(output),
                    )
                )
            self.assertEqual(raised.exception.code, "input_unacquired")
            self.assertEqual(raised.exception.details, ("read_2",))
            self.assertFalse(output.exists())

    def _assert_public_evidence_meaning(self, packet, timeline):
        by_type = {}
        for event in timeline:
            by_type.setdefault(event["event_type"], []).append(event)
        self.assertEqual(packet.journey_id, timeline[0]["journey_id"])
        self.assertEqual(packet.admitted_facts, ())
        self.assertFalse(packet.grants_live_access)
        self.assertFalse(packet.source_admission.scientifically_admitted)
        self.assertEqual(
            tuple(item.role for item in packet.verified_objects),
            ("aligned_bam", "VCF", "worker_receipt", "execution_manifest"),
        )
        self.assertEqual(packet.vcf, packet.verified_objects[1])
        self.assertEqual(packet.reference.assembly_accession, "GCF_000001405.40")
        self.assertEqual(packet.sample.sample_id, "sample-001")
        # The fixture supplies no expected FASTQ checksum, so each input's
        # identity must be the digest of the bytes the Run acquired.
        (registration,) = by_type["RunInputsRegistered"]
        self.assertEqual(
            {item.role: item.sha256 for item in packet.inputs},
            {
                item["role"]: hashlib.sha256(
                    Path(item["registered_path"]).read_bytes()
                ).hexdigest()
                for item in registration["payload"]["inputs"]
            },
        )
        for field, event_type in (
            ("proposal_locator", "ExecutionProposalSubmitted"),
            # The approving response is itself the execution authority.
            ("approval_locator", "ExecutionHitlResponded"),
            ("receipt_locator", "H100ArtifactsVerified"),
        ):
            self.assertEqual(
                getattr(packet.execution, field).event_id,
                by_type[event_type][-1]["event_id"],
            )
        expected_source_types = {
            "TaxonResolutionRecorded",
            "ReferenceAssembliesRecorded",
            "ReferenceDescriptionRecorded",
            "CompanionObservationRecorded",
        }
        self.assertEqual(
            {s.source_kind for s in packet.retrieved_sources},
            (expected_source_types - {"CompanionObservationRecorded"})
            | {"tool_observation"},
        )
        self.assertEqual(
            {s.locator.event_id for s in packet.retrieved_sources},
            {e["event_id"] for kind in expected_source_types for e in by_type[kind]},
        )
        for source in packet.retrieved_sources:
            self.assertEqual(source.role, "retrieved_source")
            self.assertFalse(source.scientifically_admitted)
        self.assertTrue(packet.user_assertions)
        self.assertTrue(packet.candidate_evidence)
        for assertion in packet.user_assertions:
            self.assertEqual(assertion.role, "user_assertion")
            self.assertFalse(assertion.scientifically_verified)
        for candidate in packet.candidate_evidence:
            self.assertEqual(candidate.role, "candidate_evidence")
            self.assertFalse(candidate.scientifically_admitted)
        self.assertEqual(
            {p.review_kind for p in packet.path_receipts}, {"planning", "execution"}
        )
        self.assertTrue(
            all(not p.scientifically_verified for p in packet.path_receipts)
        )
        by_id = {e["event_id"]: e for e in timeline}
        for locator in continuation.iter_causal_locators(packet):
            event = by_id[locator.event_id]
            self.assertEqual(locator.journey_id, timeline[0]["journey_id"])
            self.assertEqual(locator.event_type, event["event_type"])
            if "kind" in event:
                position = next(
                    i
                    for i, item in enumerate(timeline)
                    if item["event_id"] == event["event_id"]
                )
                expected_parents = tuple(
                    dict.fromkeys(
                        (*event["rests_on"], timeline[position - 1]["event_id"])
                    )
                )
            else:
                expected_parents = tuple(event["parent_ids"])
            self.assertEqual(locator.parent_event_ids, expected_parents)

    def test_publication_failure_after_link_removes_target_and_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            packet = continuation.ScientificContinuationPacket.model_validate_json(
                published_continuation(root / "source").read_bytes(), strict=True
            )
            target = root / "scientific-continuation.json"
            with (
                patch.object(
                    continuation,
                    "_strict_published_packet",
                    side_effect=continuation.ScientificContinuationPublicationRefused(
                        "scientific_continuation_reopen_mismatch", "output"
                    ),
                ),
                self.assertRaises(
                    continuation.ScientificContinuationPublicationRefused
                ) as raised,
            ):
                continuation._publish_packet(packet, target)
            self.assertEqual(
                raised.exception.code, "scientific_continuation_reopen_mismatch"
            )
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".bio-neb-scientific-continuation-*")), [])
