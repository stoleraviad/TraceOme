import tempfile
import unittest
from pathlib import Path

from bio_neb.journey import Journey
from tools.journey_replay.oracle import (
    ReplayError,
    _replicate_bundle,
    compare_captures,
    main,
    normalize_meaning,
)


class JourneyReplayOracleTests(unittest.TestCase):
    def _capture(self, sha, meaning):
        return {
            "schema_version": "bio-neb.journey-replay-capture/1",
            "source": {"checkout": f"/checkout/{sha[0]}", "sha": sha},
            "input": {
                "journey_file": "/retained/journey.jsonl",
                "journey": {"byte_count": 1, "sha256": "0" * 64},
                "bundle_file_count": 1,
            },
            "contact": {
                "network": "blocked",
                "subprocess": "blocked",
                "filesystem_writes": "blocked",
            },
            "normalization": {
                "schema_version": "bio-neb.journey-normalization/1",
                "journey_identities": (
                    "Values in journey_id, *_journey_id and *_journey_ids fields "
                    "are mapped by first schema-aware occurrence."
                ),
                "event_identities": (
                    "Values in event_id, parent_ids, *_event_id and *_event_ids "
                    "fields are mapped by event order; references outside the retained "
                    "stream use a separate mapping by first occurrence."
                ),
                "attempt_identities": (
                    "Values in attempt_id, *_attempt_id and *_attempt_ids fields "
                    "are mapped by first schema-aware occurrence."
                ),
                "occurrence_times": (
                    "Only occurred_at on dictionaries with the Journey event envelope "
                    "is replaced."
                ),
                "preserved": (
                    "List order, parent relationships, event types, payload values, "
                    "content-addressed identities, artifact digests, byte counts and "
                    "all other timestamps remain exact."
                ),
            },
            "meaning": meaning,
            "limits": [
                "Equality covers only the retained Journey prefix supplied to both revisions."
            ],
        }

    def test_replay_refuses_output_inside_inspected_bundle_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory).resolve() / "00000000-0000-4000-8000-000000000001"
            bundle.mkdir()
            journey_file = bundle / "journey.jsonl"
            journey_file.write_text("{}\n", encoding="utf-8")
            forbidden = bundle / "replay-output"

            status = main(
                [
                    "replay",
                    "--baseline-checkout",
                    str(Path(directory).resolve()),
                    "--baseline-sha",
                    "a" * 40,
                    "--candidate-checkout",
                    str(Path(directory).resolve()),
                    "--candidate-sha",
                    "b" * 40,
                    "--journey-file",
                    str(journey_file),
                    "--output-directory",
                    str(forbidden),
                ]
            )

            self.assertEqual(status, 2)
            self.assertFalse(forbidden.exists())

    def test_replay_replica_is_independent_of_retained_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            journey = Journey.create(root / "retained")
            journey.record("RetainedEvent", {"meaning": "original"})
            retained_events = journey.read()

            replica_file, retained_inventory = _replicate_bundle(
                journey.path, root / "replica-workspace"
            )
            reopened = Journey.open(replica_file)
            reopened.record("ReplicaEvent", {"meaning": "changed"})

            self.assertEqual(Journey.open(journey.path).read(), retained_events)
            self.assertEqual(len(reopened.read()), 2)
            self.assertIn("journey.jsonl", retained_inventory)

    def test_journey_event_attempt_identities_and_times_normalize_to_same_meaning(
        self,
    ):
        def events(namespace, day, hour, finished_second):
            def identity(suffix):
                return f"{namespace}0000000-0000-4000-8000-00000000000{suffix}"

            return [
                {
                    "journey_id": identity(1),
                    "event_id": identity(2),
                    "parent_ids": [],
                    "event_type": "AttemptStarted",
                    "occurred_at": f"2026-09-{day}T{hour}:00:00Z",
                    "payload": {
                        "attempt_id": identity(3),
                        "scientific_value": ("00000000-0000-4000-8000-000000000002"),
                    },
                    "artifact_references": [],
                },
                {
                    "journey_id": identity(1),
                    "event_id": identity(4),
                    "parent_ids": [identity(2)],
                    "event_type": "AttemptSucceeded",
                    "occurred_at": (f"2026-09-{day}T{hour}:00:{finished_second:02d}Z"),
                    "payload": {
                        "attempt_id": identity(3),
                        "started_event_id": identity(2),
                        "provenance_origin_event_id": (
                            "90000000-0000-4000-8000-000000000009"
                        ),
                        "artifact_digest": "sha256:unchanged",
                    },
                    "artifact_references": [],
                },
            ]

        baseline_events = events("0", "10", "10", 1)
        candidate_events = events("1", "11", "11", 2)

        baseline = normalize_meaning(
            baseline_events,
            {
                "journey_id": baseline_events[0]["journey_id"],
                "head_event_id": baseline_events[-1]["event_id"],
                "attempt_id": baseline_events[0]["payload"]["attempt_id"],
            },
            {"lineage": baseline_events},
        )
        candidate = normalize_meaning(
            candidate_events,
            {
                "journey_id": candidate_events[0]["journey_id"],
                "head_event_id": candidate_events[-1]["event_id"],
                "attempt_id": candidate_events[0]["payload"]["attempt_id"],
            },
            {"lineage": candidate_events},
        )

        self.assertEqual(candidate, baseline)
        self.assertEqual(baseline["raw_events"][1]["parent_ids"], ["event:000000"])
        self.assertEqual(
            baseline["raw_events"][1]["payload"]["started_event_id"],
            "event:000000",
        )
        self.assertEqual(
            baseline["raw_events"][0]["payload"]["scientific_value"],
            "00000000-0000-4000-8000-000000000002",
        )
        extended = normalize_meaning(
            [
                *baseline_events,
                {
                    "journey_id": baseline_events[0]["journey_id"],
                    "event_id": "00000000-0000-4000-8000-000000000005",
                    "parent_ids": [baseline_events[-1]["event_id"]],
                    "event_type": "ReplayOraclePerturbation",
                    "occurred_at": "2026-09-10T10:00:02Z",
                    "payload": {"marker": "detector-proof"},
                    "artifact_references": [],
                },
            ],
            {},
            {},
        )
        self.assertEqual(extended["raw_events"][:2], baseline["raw_events"])

    def test_differences_are_exact_and_require_investigation(self):
        baseline = self._capture(
            "a" * 40,
            {
                "raw_events": [{"payload": {"kept": 1, "removed": True}}],
                "observe": {"status": "ready", "shape": [1]},
                "review_journey": {},
            },
        )
        candidate = self._capture(
            "b" * 40,
            {
                "raw_events": [
                    {"payload": {"added": True, "kept": 2}},
                    {"event_type": "AddedEvent"},
                ],
                "observe": {"status": "ready", "shape": {"value": 1}},
                "review_journey": {},
            },
        )

        result = compare_captures(baseline, candidate)

        self.assertEqual(result["outcome"], "difference_requires_investigation")
        self.assertFalse(result["equivalent"])
        self.assertEqual(
            {(item["classification"], item["path"]) for item in result["differences"]},
            {
                ("added_field", "/raw_events/0/payload/added"),
                ("changed_value", "/raw_events/0/payload/kept"),
                ("removed_field", "/raw_events/0/payload/removed"),
                ("changed_structure", "/raw_events/1"),
                ("changed_structure", "/observe/shape"),
            },
        )
        self.assertEqual(result["investigation"]["status"], "required")
        self.assertEqual(
            set(result["investigation"]["allowed_dispositions"]),
            {"regression", "ticket_authorized_change", "oracle_limitation"},
        )

        malformed = dict(baseline)
        malformed.pop("contact")
        with self.assertRaises(ReplayError):
            compare_captures(malformed, candidate)

    def test_normalizer_does_not_hide_parent_payload_or_artifact_changes(self):
        events = [
            {
                "journey_id": "00000000-0000-4000-8000-000000000001",
                "event_id": "00000000-0000-4000-8000-000000000002",
                "parent_ids": [],
                "event_type": "EvidenceRecorded",
                "occurred_at": "2026-09-10T10:00:00Z",
                "payload": {
                    "claim": "A",
                    "digest_event_id": "sha256:one",
                    "scientific_value": "00000000-0000-4000-8000-000000000003",
                },
                "artifact_references": [
                    {
                        "journey_id": "00000000-0000-4000-8000-000000000001",
                        "source_event_id": "00000000-0000-4000-8000-000000000002",
                        "artifact_id": "sha256:one",
                        "byte_count": 1,
                        "media_type": "text/plain",
                        "role": "evidence",
                    }
                ],
            },
            {
                "journey_id": "00000000-0000-4000-8000-000000000001",
                "event_id": "00000000-0000-4000-8000-000000000003",
                "parent_ids": ["00000000-0000-4000-8000-000000000002"],
                "event_type": "EvidenceAccepted",
                "occurred_at": "2026-09-10T10:00:01Z",
                "payload": {},
                "artifact_references": [],
            },
        ]
        changed = [dict(item) for item in events]
        changed[0] = {
            **changed[0],
            "payload": {
                "claim": "B",
                "digest_event_id": "sha256:two",
                "scientific_value": "00000000-0000-4000-8000-000000000002",
            },
            "artifact_references": [
                {**changed[0]["artifact_references"][0], "artifact_id": "sha256:two"}
            ],
        }
        changed[1] = {
            **changed[1],
            "parent_ids": ["00000000-0000-4000-8000-000000000003"],
        }

        baseline = self._capture("a" * 40, normalize_meaning(events, {}, {}))
        candidate = self._capture("b" * 40, normalize_meaning(changed, {}, {}))

        paths = {
            item["path"]
            for item in compare_captures(baseline, candidate)["differences"]
        }

        self.assertIn("/raw_events/0/payload/claim", paths)
        self.assertIn("/raw_events/0/payload/digest_event_id", paths)
        self.assertIn("/raw_events/0/payload/scientific_value", paths)
        self.assertIn("/raw_events/0/artifact_references/0/artifact_id", paths)
        self.assertIn("/raw_events/1/parent_ids/0", paths)


if __name__ == "__main__":
    unittest.main()
