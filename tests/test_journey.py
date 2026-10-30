import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from bio_neb.journey import (
    ArtifactReference,
    Journey,
    JourneyEvent,
    JourneyProjection,
    JourneyValidationError,
    canonical_json,
)


class JourneyTests(unittest.TestCase):
    def test_journey_path_arguments_share_one_boundary_type_with_named_reasons(self):
        """Every command field naming a Journey refuses a bad path with its reason."""
        import importlib
        import pkgutil

        from pydantic import BaseModel

        packages = (
            "application",
            "input_inspection",
            "investigation",
            "parabricks",
            "reference_discovery",
            "reporting",
        )
        fields = {
            "journey_file",
            "parent_journey_file",
            "execution_journey_file",
            "source_journey_file",
        }
        commands = {}
        for package in packages:
            root = importlib.import_module(f"bio_neb.{package}")
            for module in pkgutil.walk_packages(root.__path__, f"bio_neb.{package}."):
                for name, value in vars(importlib.import_module(module.name)).items():
                    if (
                        isinstance(value, type)
                        and issubclass(value, BaseModel)
                        and value.__module__ == module.name
                        and (name.endswith("Command") or name == "ObserveRun")
                    ):
                        for field in fields & set(value.model_fields):
                            commands[f"{module.name}.{name}.{field}"] = (value, field)
        # The caller-less reference Attempt commands, the typed
        # AdmitReferenceSourceCommand and the investigation-coverage command are retired,
        # leaving at least 21.
        self.assertGreaterEqual(len(commands), 21)
        for value, reason in (
            ("", "journey_file_empty"),
            ("runs/journey.jsonl", "journey_file_not_absolute"),
        ):
            for identity, (command, field) in sorted(commands.items()):
                with self.subTest(command=identity, reason=reason):
                    with self.assertRaises(ValidationError) as refused:
                        command.model_validate({field: value})
                    self.assertIn(
                        ((field,), reason),
                        [
                            (item["loc"], item["type"])
                            for item in refused.exception.errors()
                        ],
                    )

    def test_preview_is_contact_free_and_stale_state_refuses_before_append(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            state = journey.project()
            frozen = journey.path.read_bytes()
            preview = journey.preview(
                state, "UserMessage", {"content": "candidate"}, [started["event_id"]]
            )
            self.assertEqual(preview.events[-1].payload, {"content": "candidate"})
            self.assertEqual(journey.path.read_bytes(), frozen)
            self.assertEqual(journey.project(), state)
            _, current = journey.record_against(
                state, "UserMessage", {"content": "accepted"}, [started["event_id"]]
            )
            self.assertEqual(current, journey.project())
            frozen = journey.path.read_bytes()
            with self.assertRaisesRegex(JourneyValidationError, "state"):
                journey.record_against(
                    state, "UserMessage", {"content": "stale"}, [started["event_id"]]
                )
            self.assertEqual(journey.path.read_bytes(), frozen)

    def test_processing_interval_shares_one_writer_and_rejects_competing_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            competitor = Journey.open(journey.path)
            with journey.processing():
                received = Journey.open(journey.path)
                self.assertIs(received, journey)
                received.record(
                    "UserMessage", {"content": "first"}, [started["event_id"]]
                )
                before = journey.path.read_bytes()
                with self.assertRaisesRegex(JourneyValidationError, "writer"):
                    competitor.record(
                        "UserMessage", {"content": "stale"}, [started["event_id"]]
                    )
                self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual(Journey.open(journey.path).read(), journey.read())

    def test_failed_append_poison_prevents_reusing_a_partial_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            before = journey.read()
            with patch("bio_neb.journey.os.fsync", side_effect=OSError("disk failed")):
                with self.assertRaises(OSError):
                    journey.record("UserMessage", {}, [started["event_id"]])
            self.assertEqual(journey.read(), before)
            retained = journey.path.read_bytes()
            with self.assertRaisesRegex(JourneyValidationError, "unusable"):
                journey.record("UserMessage", {}, [started["event_id"]])
            self.assertEqual(journey.path.read_bytes(), retained)

    def test_canonical_content_digest_uses_utf8_and_refuses_non_finite(self):
        from bio_neb.content import sha256_bytes
        from bio_neb.journey import canonical_json_digest

        self.assertEqual(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        # Independently hashed literal UTF-8: {"a":[1,true,null],"z":"café"}.
        expected = "fafdedd93d19c910cbdda0aa3f0f91ee2f287edbd78d712e4fa39d41db0763c6"
        self.assertEqual(
            canonical_json_digest({"z": "café", "a": [1, True, None]}), expected
        )
        self.assertEqual(
            canonical_json_digest({"a": [1, True, None], "z": "café"}), expected
        )
        self.assertNotEqual(
            canonical_json_digest({"a": [None, True, 1], "z": "café"}), expected
        )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_json_digest({"nested": [value]})
        with self.assertRaises(TypeError):
            canonical_json_digest(object())

    def test_append_and_reopen_preserve_canonical_typed_lineage_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record(
                "JourneyStarted",
                {"contract_version": "1.0"},
            )
            message = journey.record(
                "UserMessage",
                {"contract_version": "1.0", "content": "exact prompt"},
                parent_ids=[started["event_id"]],
            )

            reopened = Journey.open(journey.path)
            projection = reopened.project()

            self.assertIsInstance(projection, JourneyProjection)
            self.assertTrue(
                all(isinstance(event, JourneyEvent) for event in projection.events)
            )
            self.assertEqual(reopened.read(), journey.read())
            self.assertEqual(message["parent_ids"], [started["event_id"]])
            raw_lines = journey.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                raw_lines, [canonical_json(event) for event in journey.read()]
            )

    def test_projection_mutation_is_isolated_from_retained_journey_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record(
                "JourneyStarted",
                {"contract_version": "1.0"},
            )
            journey.record(
                "UserMessage",
                {"contract_version": "1.0", "content": "exact prompt"},
                parent_ids=[started["event_id"]],
            )
            before_projection = journey.read()
            retained_bytes = journey.path.read_bytes()

            local_projection = journey.project()
            local_projection.events[0].payload["contract_version"] = "mutated"
            local_projection.events[1].parent_ids.clear()
            local_projection.events[0].artifact_references.append(object())  # type: ignore[arg-type]
            self.assertEqual(journey.read(), before_projection)
            self.assertEqual(journey.path.read_bytes(), retained_bytes)

    def test_event_validation_rejects_invalid_value_before_append(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())

            with self.assertRaises(JourneyValidationError):
                journey.record(
                    "JourneyStarted",
                    {"contract_version": "1.0"},
                    occurred_at="2026-08-05T12:00:00",
                )
            with self.assertRaises(JourneyValidationError):
                journey.record("JourneyStarted", {"not_json": object()})
            with self.assertRaises(JourneyValidationError):
                journey.record("JourneyStarted", {"invalid_unicode": "\ud800"})
            invalid_reference = {
                "contract_version": "1.0",
                "artifact_id": "sha256:" + "A" * 64,
                "byte_count": 1,
                "journey_id": journey.journey_id,
                "media_type": "text/plain",
                "source_event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            }
            with self.assertRaises(JourneyValidationError):
                journey.record(
                    "SourceRetained",
                    {},
                    event_id=invalid_reference["source_event_id"],
                    artifact_references=[invalid_reference],
                )

            self.assertEqual(journey.path.read_bytes(), b"")

    def test_typed_contracts_forbid_unknown_or_noncanonical_fields(self):
        event = {
            "contract_version": "1.0",
            "artifact_references": [],
            "event_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
            "event_type": "JourneyStarted",
            "journey_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "occurred_at": "2026-08-05T12:00:00Z",
            "parent_ids": [],
            "payload": {},
        }
        with self.assertRaises(ValidationError):
            JourneyEvent.model_validate(event)

        reference = {
            "contract_version": "1.0",
            "artifact_id": "sha256:" + "A" * 64,
            "byte_count": 1,
            "journey_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "media_type": "text/plain",
            "source_event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        }
        with self.assertRaises(ValidationError):
            ArtifactReference.model_validate(reference)

    def test_open_rejects_noncanonical_and_invalid_retained_data(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            journey.record("JourneyStarted", {})
            retained = journey.read()[0]

            journey.path.write_text(json.dumps(retained) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(JourneyValidationError, "non-canonical JSON"):
                Journey.open(journey.path)

            retained["unexpected"] = True
            journey.path.write_bytes((canonical_json(retained) + "\n").encode("utf-8"))
            with self.assertRaisesRegex(
                JourneyValidationError, "invalid Journey event"
            ):
                Journey.open(journey.path)

    def test_open_rejects_noncanonical_framing_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            journey.record("JourneyStarted", {})
            canonical_line = canonical_json(journey.read()[0])

            for retained in (
                canonical_line.encode("utf-8"),
                (canonical_line + "\r\n").encode("utf-8"),
            ):
                journey.path.write_bytes(retained)
                with self.assertRaisesRegex(JourneyValidationError, "framing"):
                    Journey.open(journey.path)

            nonfinite = canonical_line.replace(
                '"payload":{}', '"payload":{"value":NaN}'
            )
            journey.path.write_bytes((nonfinite + "\n").encode("utf-8"))
            with self.assertRaisesRegex(JourneyValidationError, "invalid JSON value"):
                Journey.open(journey.path)

    def test_open_binds_retained_identity_to_the_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            journey.record("JourneyStarted", {})
            moved_directory = Path(directory) / "not-the-journey-id"
            journey.run_directory.rename(moved_directory)

            with self.assertRaisesRegex(JourneyValidationError, "directory"):
                Journey.open(moved_directory / "journey.jsonl")

    def test_artifact_content_address_deduplicates_with_exact_source_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            artifact_event = journey.record_with_artifact(
                "SourceRetained",
                {"name": "observation.txt"},
                b"same scientific bytes",
                "text/plain",
                parent_ids=[started["event_id"]],
            )
            reference = ArtifactReference.model_validate(
                artifact_event["artifact_references"][0]
            )

            self.assertEqual(reference.journey_id, journey.journey_id)
            self.assertEqual(reference.source_event_id, artifact_event["event_id"])
            self.assertEqual(journey.read_artifact(reference), b"same scientific bytes")
            self.assertEqual(
                journey.artifact_path(reference).parent, journey.artifacts_directory
            )

            repeated = journey.record_with_artifact(
                "SourceRetainedAgain",
                {},
                b"same scientific bytes",
                "text/plain",
                parent_ids=[artifact_event["event_id"]],
            )
            repeated_reference = ArtifactReference.model_validate(
                repeated["artifact_references"][0]
            )
            self.assertEqual(repeated_reference.artifact_id, reference.artifact_id)
            self.assertEqual(
                journey.artifact_path(repeated_reference),
                journey.artifact_path(reference),
            )
            self.assertNotEqual(
                repeated_reference.source_event_id, reference.source_event_id
            )

    def test_forged_artifact_lineage_is_rejected_without_partial_append(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            artifact_event = journey.record_with_artifact(
                "SourceRetained",
                {"name": "observation.txt"},
                b"same scientific bytes",
                "text/plain",
                parent_ids=[started["event_id"]],
            )
            repeated = journey.record_with_artifact(
                "SourceRetainedAgain",
                {},
                b"same scientific bytes",
                "text/plain",
                parent_ids=[artifact_event["event_id"]],
            )
            repeated_reference = ArtifactReference.model_validate(
                repeated["artifact_references"][0]
            )
            forged_reference = repeated_reference.model_dump(mode="json")
            forged_reference["source_event_id"] = started["event_id"]
            before_forged_append = journey.path.read_bytes()
            with self.assertRaisesRegex(
                JourneyValidationError, "did not retain this reference"
            ):
                journey.record(
                    "ForgedArtifactLineage",
                    {},
                    parent_ids=[repeated["event_id"]],
                    artifact_references=[forged_reference],
                )
            self.assertEqual(journey.path.read_bytes(), before_forged_append)

    def test_tampered_or_missing_artifact_fails_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            artifact_event = journey.record_with_artifact(
                "SourceRetained",
                {"name": "observation.txt"},
                b"same scientific bytes",
                "text/plain",
                parent_ids=[started["event_id"]],
            )
            reference = ArtifactReference.model_validate(
                artifact_event["artifact_references"][0]
            )
            repeated = journey.record_with_artifact(
                "SourceRetainedAgain",
                {},
                b"same scientific bytes",
                "text/plain",
                parent_ids=[artifact_event["event_id"]],
            )
            before = journey.path.read_bytes()
            journey.artifact_path(reference).write_bytes(b"tampered")
            with self.assertRaisesRegex(JourneyValidationError, "different bytes"):
                journey.record_with_artifact(
                    "SourceRetained",
                    {},
                    b"same scientific bytes",
                    "text/plain",
                    parent_ids=[repeated["event_id"]],
                )
            self.assertEqual(journey.path.read_bytes(), before)
            with self.assertRaisesRegex(
                JourneyValidationError, "digest does not match"
            ):
                Journey.open(journey.path)
            journey.artifact_path(reference).unlink()
            with self.assertRaisesRegex(JourneyValidationError, "missing"):
                Journey.open(journey.path)

    def test_artifact_and_event_retention_are_fsynced(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            before = journey.path.read_bytes()
            synced = []
            real_fsync = os.fsync

            def sync_descriptor(fd):
                path = Path(os.readlink(f"/proc/self/fd/{fd}"))
                content = path.read_bytes()
                if not synced:
                    self.assertEqual(journey.path.read_bytes(), before)
                synced.append((path, content))
                real_fsync(fd)

            with patch("bio_neb.journey.os.fsync", side_effect=sync_descriptor):
                event = journey.record_with_artifact(
                    "SourceRetained",
                    {},
                    b"fsynced bytes",
                    "application/octet-stream",
                )
            artifact = journey.artifact_path(event["artifact_references"][0])
            self.assertEqual([path for path, _ in synced], [artifact, journey.path])
            self.assertEqual(synced[0][1], b"fsynced bytes")
            self.assertEqual(json.loads(synced[1][1].splitlines()[-1]), event)
            self.assertEqual(Journey.open(journey.path).read()[-1], event)

    def test_owned_artifact_bytes_survive_reads_appends_and_caller_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            retained = journey.record_with_artifact(
                "SourceRetained",
                {"values": ["original"]},
                b"exact control bytes",
                "text/plain",
                parent_ids=[started["event_id"]],
            )
            reference = retained["artifact_references"][0]
            path = journey.artifact_path(reference)
            reads = []
            original_read = Path.read_bytes

            def read_bytes(selected):
                if selected == path:
                    reads.append(selected)
                return original_read(selected)

            with patch.object(Path, "read_bytes", read_bytes):
                owned = Journey.open(journey.path)
                self.assertEqual(len(reads), 1)
                owned.read()[1]["payload"]["values"].append("changed")
                owned.project().events[1].payload["values"].append("changed")
                self.assertEqual(owned.read()[1]["payload"], {"values": ["original"]})
                self.assertEqual(owned.read_artifact(reference), b"exact control bytes")
                self.assertEqual(len(reads), 1)
                owned.record(
                    "UnrelatedSemanticEvent",
                    {},
                    parent_ids=[retained["event_id"]],
                )
                self.assertEqual(owned.read_artifact(reference), b"exact control bytes")
                self.assertEqual(len(reads), 1)
                Journey.open(journey.path)
                self.assertEqual(len(reads), 2)
                path.write_bytes(b"hostile replacement")
                self.assertEqual(owned.read_artifact(reference), b"exact control bytes")
                self.assertEqual(len(reads), 2)
                with self.assertRaisesRegex(JourneyValidationError, "digest"):
                    Journey.open(journey.path)

    def test_retained_view_inherits_owner_verified_bytes_and_refolds_its_prefix(self):
        from bio_neb.recording import processing_pass

        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            retained = journey.record_with_artifact(
                "SourceRetained",
                {"values": ["original"]},
                b"exact control bytes",
                "text/plain",
                parent_ids=[started["event_id"]],
            )
            reference = retained["artifact_references"][0]
            path = journey.artifact_path(reference)
            with journey.processing(), processing_pass(journey) as interval:
                journey.record("Later", {}, [retained["event_id"]])
                interval.close("completed")
            owner = Journey.open(journey.path)
            # JourneyStarted, SourceRetained, PassStarted, Later: the Pass is open.
            prefix = b"".join(owner.retained_prefix.splitlines(keepends=True)[:4])
            reads = []
            original_read = Path.read_bytes

            def read_bytes(selected):
                if selected == path:
                    reads.append(selected)
                return original_read(selected)

            path.write_bytes(b"hostile replacement")
            with patch.object(Path, "read_bytes", read_bytes):
                view = owner.retained_view(prefix)
                self.assertEqual(view.read_artifact(reference), b"exact control bytes")
            # The owner proved these bytes at its open; the view reuses that proof.
            self.assertEqual(reads, [])
            self.assertEqual(view.read(), owner.read()[:4])
            self.assertEqual(view.retained_prefix, prefix)
            # Family meaning is re-derived for the prefix, not inherited.
            self.assertEqual(
                view.checked_view("processing_pass").active,
                (interval.pass_id, owner.read()[2]["event_id"]),
            )
            self.assertIsNone(owner.checked_view("processing_pass").active)
            with self.assertRaisesRegex(JourneyValidationError, "read-only"):
                view.record("Forged", {}, [retained["event_id"]])
            # Only an independent open is the boundary for current disk bytes.
            with self.assertRaisesRegex(JourneyValidationError, "digest"):
                Journey.open(journey.path)

    def test_reopen_projection_and_artifact_need_only_the_external_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            external_root = Path(directory).resolve() / "external-runs"
            source = Path(directory) / "temporary-source.txt"
            source.write_bytes(b"offline body")
            journey = Journey.create(external_root)
            started = journey.record("JourneyStarted", {})
            retained = journey.record_with_artifact(
                "SourceRetained",
                {"source_name": source.name},
                source.read_bytes(),
                "text/plain",
                parent_ids=[started["event_id"]],
            )
            source.unlink()
            os.environ.pop("NEBIUS_TOKEN", None)
            script = (
                "import builtins,json,socket,sys; "
                "original=builtins.__import__; "
                "builtins.__import__=lambda name,*a,**k: "
                "(_ for _ in ()).throw(RuntimeError('live dependency forbidden')) "
                "if name.startswith(('bio_neb.agentic_journey.adapters',"
                "'bio_neb.agentic_journey.runtime_config','pydantic_ai','langgraph')) "
                "else original(name,*a,**k); "
                "from bio_neb.journey import Journey,canonical_json; "
                "socket.socket=lambda *a,**k: "
                "(_ for _ in ()).throw(RuntimeError('network forbidden')); "
                "j=Journey.open(sys.argv[1]); p=j.project(); "
                "r=p.events[-1].artifact_references[0]; "
                "print(canonical_json({'artifact':j.read_artifact(r).decode(),"
                "'event_types':[e.event_type for e in p.events],"
                "'reference':r.model_dump(mode='json')}))"
            )
            environment = {
                key: value
                for key, value in os.environ.items()
                if "TOKEN" not in key and "KEY" not in key
            }
            completed = subprocess.run(
                [sys.executable, "-c", script, str(journey.path)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            receipt = json.loads(completed.stdout)

            self.assertEqual(
                receipt["event_types"], ["JourneyStarted", "SourceRetained"]
            )
            self.assertEqual(receipt["reference"], retained["artifact_references"][0])
            self.assertEqual(receipt["artifact"], "offline body")
            self.assertFalse(source.exists())

    def test_refresh_admits_only_append_extensions_and_checks_each_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            payload = {"values": ["original"]}
            retained = journey.record_with_artifact(
                "SourceRetained", payload, b"original bytes", "text/plain"
            )
            payload["values"].append("caller mutation")
            self.assertEqual(journey.read()[0]["payload"], {"values": ["original"]})
            supplied_events = journey.project().events
            owner = Journey(journey.path, journey.journey_id, supplied_events)
            supplied_events[0].payload["values"].clear()
            self.assertEqual(owner.read(), journey.read())
            journey.record("ApprovalChanged", {}, [retained["event_id"]])
            self.assertFalse(owner.is_current())
            path = journey.artifact_path(retained["artifact_references"][0])
            reads = []
            original_read = Path.read_bytes

            def read_bytes(selected):
                if selected == path:
                    reads.append(selected)
                return original_read(selected)

            with patch.object(Path, "read_bytes", read_bytes):
                owner.refresh()
                self.assertTrue(owner.is_current())
                self.assertEqual(owner.read(), journey.read())
                self.assertEqual(reads, [])
            hostile = journey.read()
            hostile[0]["payload"]["values"] = ["forged"]
            journey.path.write_text(
                "".join(canonical_json(event) + "\n" for event in hostile)
            )
            with self.assertRaisesRegex(JourneyValidationError, "prefix"):
                owner.refresh()
            self.assertEqual(owner.read(), journey.read())
            hostile = journey.read()
            del hostile[0]["contract_version"]
            journey.path.write_text(
                "".join(canonical_json(event) + "\n" for event in hostile)
            )
            with self.assertRaisesRegex(JourneyValidationError, "prefix"):
                owner.refresh()
            self.assertEqual(owner.read(), journey.read())
            hostile = journey.read()
            # Restore the admitted bytes before exercising a separate hostile reference.
            journey.path.write_text(
                "".join(canonical_json(event) + "\n" for event in hostile)
            )
            # The same digest never exempts another reference's byte count.
            second = journey.record_with_artifact(
                "SourceRetainedAgain", {}, b"original bytes", "text/plain"
            )
            hostile.append(second)
            hostile[-1]["artifact_references"][0]["byte_count"] += 1
            journey.path.write_text(
                "".join(canonical_json(event) + "\n" for event in hostile)
            )
            with self.assertRaisesRegex(JourneyValidationError, "byte count"):
                Journey.open(journey.path)

    def test_events_view_shares_one_read_only_prefix_while_read_stays_a_copy(self):
        from copy import deepcopy

        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {"labels": {"a": [1]}})
            journey.record("UserMessage", {"content": "x"}, [started["event_id"]])
            view = journey.events_view()
            self.assertEqual(list(view), journey.read())
            # Internal readers share the held objects instead of copying history.
            self.assertIs(journey.events_view()[0], view[0])
            refused = (
                lambda: view[0]["payload"].__setitem__("labels", {}),
                lambda: view[0]["payload"]["labels"]["a"].append(2),
                lambda: view[1]["parent_ids"].clear(),
                lambda: view[1].update(event_type="Forged"),
                lambda: view[1].pop("payload"),
            )
            for mutate in refused:
                with self.assertRaises(TypeError):
                    mutate()
            self.assertEqual(list(journey.events_view()), journey.read())

            local = deepcopy(view[0])
            local["payload"]["labels"]["a"].append(2)
            copied = journey.read()
            copied[0]["payload"]["labels"]["a"].append(3)
            self.assertEqual(
                journey.events_view()[0]["payload"], {"labels": {"a": [1]}}
            )

            appended = journey.record(
                "UserMessage", {"content": "y"}, [started["event_id"]]
            )
            extended = journey.events_view()
            self.assertEqual(extended[-1], appended)
            self.assertIs(extended[0], view[0])

    def test_only_a_family_whose_meaning_refuses_mutation_is_shared(self):
        """#278 Standards S1: Execution's held state carries mutable mappings."""
        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            journey.record("JourneyStarted", {})
            self.assertIs(
                journey.shared_checked_view("planning_review"),
                journey.shared_checked_view("planning_review"),
            )
            with self.assertRaisesRegex(ValueError, "execution.*not shared read-only"):
                journey.shared_checked_view("execution")
            isolated = journey.checked_view("execution")
            isolated.amendments["location"] = "caller-only"
            self.assertEqual(journey.checked_view("execution").amendments, {})

    def test_held_head_revision_and_membership_follow_appends_and_equal_reopen(self):
        from bio_neb.recording import (
            ContactScope,
            evidence_revision,
            processing_pass,
        )

        with tempfile.TemporaryDirectory() as directory:
            journey = Journey.create(Path(directory).resolve())
            self.assertEqual(
                (journey.head_event_id, journey.revision, journey.latest_event()),
                (None, 0, None),
            )
            started = journey.record("JourneyStarted", {})
            self.assertEqual(journey.head_event_id, started["event_id"])
            self.assertEqual(journey.revision, 1)
            self.assertTrue(journey.has_event_type("JourneyStarted"))
            self.assertFalse(journey.has_event_type("PassStarted"))
            with journey.processing(), processing_pass(journey) as interval:
                request = journey.record(
                    "Request", {"attempt_id": "a"}, [started["event_id"]]
                )
                pass_started = journey.latest_event({"PassStarted", "PassClosed"})
                self.assertEqual(pass_started["event_type"], "PassStarted")
                self.assertEqual(journey.latest_event(), request)
                quiet = evidence_revision(journey)
                self.assertEqual(evidence_revision(journey), quiet)
                interval.journal.start(
                    ContactScope(attempt_id="a", request_id=request["event_id"]),
                    "inspect",
                )
                # Active journal growth changes the key without a semantic append.
                self.assertNotEqual(evidence_revision(journey), quiet)
                interval.close("completed")
            self.assertEqual(
                journey.latest_event(["PassStarted", "PassClosed"])["event_type"],
                "PassClosed",
            )
            artifact_reads = []
            original_read = Path.read_bytes

            def read_bytes(selected):
                if selected.parent == journey.artifacts_directory:
                    artifact_reads.append(selected)
                return original_read(selected)

            with patch.object(Path, "read_bytes", read_bytes):
                retained = journey.record_with_artifact(
                    "SourceRetained",
                    {"values": ["original"]},
                    b"exact control bytes",
                    "text/plain",
                    parent_ids=[request["event_id"]],
                )
                self.assertEqual(
                    journey.read_artifact(retained["artifact_references"][0]),
                    b"exact control bytes",
                )
            # Writing proved the new bytes once; the writer never reads them back.
            self.assertEqual(artifact_reads, [])
            self.assertEqual(journey.head_event_id, retained["event_id"])
            self.assertEqual(journey.revision, len(journey.read()))

            reopened = Journey.open(journey.path)
            self.assertEqual(reopened.project(), journey.project())
            self.assertEqual(
                reopened.checked_view("processing_pass"),
                journey.checked_view("processing_pass"),
            )
            self.assertEqual(
                (reopened.head_event_id, reopened.revision),
                (journey.head_event_id, journey.revision),
            )
            self.assertEqual(evidence_revision(reopened), evidence_revision(journey))

    def test_first_family_write_in_an_interval_folds_over_the_lazy_pass_start(self):
        from pydantic import BaseModel

        from bio_neb.journey_contracts import SemanticFamily, semantic_families
        from bio_neb.recording import processing_pass

        class Counted(BaseModel):
            pass

        def families(event_types, original=semantic_families):
            counted = SemanticFamily(
                "counted",
                {"Counted": Counted},
                lambda events: [event["event_type"] for event in events],
            )
            return (
                *original(event_types),
                *((counted,) if "Counted" in event_types else ()),
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("bio_neb.journey.semantic_families", families),
        ):
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            with journey.processing(), processing_pass(journey):
                journey.record("Counted", {}, [started["event_id"]])
                self.assertEqual(
                    journey.checked_view("counted"),
                    ["JourneyStarted", "PassStarted", "Counted"],
                )
            reopened = Journey.open(journey.path)
            self.assertEqual(
                reopened.checked_view("counted"), journey.checked_view("counted")
            )

    def test_stepped_family_folds_only_the_appended_fact_on_write(self):
        from pydantic import BaseModel

        from bio_neb.journey_contracts import SemanticFamily, semantic_families

        class Counted(BaseModel):
            pass

        folded, stepped = [], []

        def fold(events):
            folded.append(len(events))
            return tuple(e["event_id"] for e in events if e["event_type"] == "Counted")

        def step(state, event, prefix):
            stepped.append(len(prefix))
            return (*state, event["event_id"])

        def families(event_types, original=semantic_families):
            counted = SemanticFamily("counted", {"Counted": Counted}, fold, step=step)
            return (*original(event_types), counted)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("bio_neb.journey.semantic_families", families),
        ):
            journey = Journey.create(Path(directory).resolve())
            started = journey.record("JourneyStarted", {})
            first = journey.record("Counted", {}, [started["event_id"]])
            journey.record("Other", {}, [first["event_id"]])
            second = journey.record("Counted", {}, [first["event_id"]])
            self.assertEqual(folded, [0])
            self.assertEqual(stepped, [1, 3])
            self.assertEqual(
                journey.checked_view("counted"), (first["event_id"], second["event_id"])
            )
            # Independent open folds the whole history once and agrees.
            reopened = Journey.open(journey.path)
            self.assertEqual(folded, [0, 4])
            self.assertEqual(
                reopened.checked_view("counted"), journey.checked_view("counted")
            )


if __name__ == "__main__":
    unittest.main()
