import builtins
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb.input_inspection.local_files import LocalFilesAdapter
from bio_neb.input_inspection.tool import (
    InputMetadataAdapter,
    InputMetadataObservation,
    RegisteredInputPath,
)


class LocalFilesAdapterTests(unittest.TestCase):
    def test_structurally_satisfies_tool_owned_adapter_contract(self):
        adapter: InputMetadataAdapter = LocalFilesAdapter()

        self.assertIsInstance(adapter, InputMetadataAdapter)

    def test_observe_is_stat_only_ordered_and_collection_shaped(self):
        for cardinality in (1, 2, 4):
            with (
                self.subTest(cardinality=cardinality),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                paths = []
                for index in range(cardinality):
                    path = root / f"sample-{index}.fastq.gz"
                    path.write_bytes(f"reads-{index}".encode())
                    paths.append(path)
                requests = tuple(
                    RegisteredInputPath(
                        role=f"READ-{index + 1}", registered_path=str(path)
                    )
                    for index, path in enumerate(paths)
                )
                stat_calls = []

                def stat_only(path, *, follow_symlinks, stat_calls=stat_calls):
                    stat_calls.append((path, follow_symlinks))
                    return os.stat(path, follow_symlinks=follow_symlinks)

                adapter = LocalFilesAdapter(
                    stat_provider=stat_only,
                    clock=iter(
                        ("2026-08-06T10:00:00Z", "2026-08-06T10:00:01Z")
                    ).__next__,
                )
                with patch.object(
                    builtins,
                    "open",
                    side_effect=AssertionError("FASTQ bytes must not be opened"),
                ):
                    observation = adapter.observe(requests)

                self.assertIsInstance(observation, InputMetadataObservation)
                self.assertEqual(
                    stat_calls,
                    [(str(path), True) for path in paths],
                )
                self.assertEqual(
                    [item.role for item in observation.inputs],
                    [item.role for item in requests],
                )
                self.assertEqual(
                    [item.registered_path for item in observation.inputs],
                    [item.registered_path for item in requests],
                )
                self.assertTrue(all(item.exists for item in observation.inputs))
                self.assertTrue(all(item.is_regular for item in observation.inputs))
                self.assertEqual(observation.started_at, "2026-08-06T10:00:00Z")
                self.assertEqual(observation.completed_at, "2026-08-06T10:00:01Z")

    def test_expected_missing_path_exceptions_map_to_the_same_stable_facts(self):
        requests = tuple(
            RegisteredInputPath(
                role=role, registered_path=str(Path("/registered") / name)
            )
            for role, name in (
                ("READ-1", "missing.fastq"),
                ("READ-2", "parent-file/child.fastq"),
            )
        )

        def missing(path, *, follow_symlinks):
            self.assertTrue(follow_symlinks)
            if path.endswith("missing.fastq"):
                raise FileNotFoundError(path)
            raise NotADirectoryError(path)

        result = LocalFilesAdapter(
            stat_provider=missing,
            clock=iter(("2026-08-06T10:00:00Z", "2026-08-06T10:00:01Z")).__next__,
        ).observe(requests)

        expected = {
            "exists": False,
            "is_regular": False,
            "size_bytes": None,
            "filesystem_identity": None,
            "modified_at_ns": None,
        }
        self.assertEqual(len(result.inputs), 2)
        for item, request in zip(result.inputs, requests, strict=False):
            self.assertEqual(item.role, request.role)
            self.assertEqual(item.registered_path, request.registered_path)
            self.assertEqual(
                item.model_dump(exclude={"role", "registered_path"}), expected
            )

    def test_unexpected_filesystem_exceptions_are_not_reclassified_as_missing(self):
        request = RegisteredInputPath(
            role="READ-1", registered_path="/registered/denied.fastq"
        )
        denied = PermissionError("permission denied")

        def reject(path, *, follow_symlinks):
            raise denied

        with self.assertRaises(PermissionError) as raised:
            LocalFilesAdapter(stat_provider=reject).observe((request,))
        self.assertIs(raised.exception, denied)

    def test_empty_request_collection_is_rejected_before_observation(self):
        with self.assertRaisesRegex(
            ValueError, "at least one metadata request is required"
        ):
            LocalFilesAdapter().observe(())


if __name__ == "__main__":
    unittest.main()
