import copy
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any, Sequence

from bio_neb.input_inspection.tool import (
    InputMetadataAdapter,
    InputMetadataFailure,
    InputMetadataInspectionFailed,
    InputMetadataInspectionTool,
    InputMetadataObservation,
    InspectInputMetadataInput,
    RegisteredInputPath,
)
from tests.input_metadata_fakes import (
    ReturningAdapter,
    metadata,
    observation,
)


class MalformedAdapter:
    def __init__(self, value: Any):
        self.value = value

    def observe(self, requests: Sequence[RegisteredInputPath]) -> Any:
        return copy.deepcopy(self.value)


class RaisingAdapter:
    def __init__(self, error: Exception):
        self.error = error

    def observe(
        self, requests: Sequence[RegisteredInputPath]
    ) -> InputMetadataObservation:
        raise self.error


def tool_input(paths):
    return InspectInputMetadataInput(
        input_set_id=str(uuid.uuid4()),
        inputs=tuple(
            RegisteredInputPath(role=role, registered_path=str(path))
            for role, path in paths
        ),
    )


def assert_adapter_failure(test, result, input_set_id):
    test.assertEqual(result.status, "failed")
    test.assertEqual(result.input_set_id, input_set_id)
    test.assertEqual(
        [failure.model_dump() for failure in result.failures],
        [
            {
                "code": "adapter_output_invalid",
                "role": None,
                "registered_path": None,
            }
        ],
    )
    test.assertIsNone(result.observation)
    test.assertEqual(result.summary, "Supplied input metadata inspection failed.")


class InputMetadataInspectionToolTests(unittest.TestCase):
    def test_ordered_collection_success_for_one_two_and_more_inputs(self):
        for cardinality in (1, 2, 4):
            with (
                self.subTest(cardinality=cardinality),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                paths = [
                    (f"READ-{index + 1}", root / f"sample-{index}.fastq.gz")
                    for index in range(cardinality)
                ]
                expected = tool_input(paths)
                raw = observation(
                    [
                        metadata(role, path, size=index + 1, inode=str(index + 10))
                        for index, (role, path) in enumerate(paths)
                    ]
                )
                adapter: InputMetadataAdapter = ReturningAdapter(raw)

                result = InputMetadataInspectionTool().inspect(expected, adapter)

                self.assertEqual(result.status, "succeeded")
                self.assertEqual(result.input_set_id, expected.input_set_id)
                self.assertEqual(
                    [(item.role, item.registered_path) for item in adapter.requests],
                    [(item.role, item.registered_path) for item in expected.inputs],
                )
                self.assertEqual(
                    [(item.role, item.registered_path) for item in result.inputs],
                    [(item.role, item.registered_path) for item in expected.inputs],
                )
                self.assertEqual(
                    [item.size_bytes for item in result.inputs],
                    list(range(1, cardinality + 1)),
                )
                self.assertNotIn("metadata_stable", result.inputs[0].model_dump())
                self.assertEqual(len(result.observation.inputs), cardinality)

    def test_stable_failures_are_collected_in_contract_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            duplicate = root / "duplicate.fastq"
            missing = root / "missing.fastq"
            directory_path = root / "directory"
            empty = root / "empty.fastq"
            changed = root / "changed.fastq"
            paths = [
                ("DUPLICATE-A", duplicate),
                ("MISSING", missing),
                ("DIRECTORY", directory_path),
                ("EMPTY", empty),
                ("CHANGED", changed),
                ("DUPLICATE-B", duplicate),
            ]
            expected = tool_input(paths)
            first = [
                metadata("DUPLICATE-A", duplicate, inode="1"),
                {
                    "role": "MISSING",
                    "registered_path": str(missing),
                    "exists": False,
                    "is_regular": False,
                    "size_bytes": None,
                    "filesystem_identity": None,
                    "modified_at_ns": None,
                },
                {
                    **metadata("DIRECTORY", directory_path, inode="2"),
                    "is_regular": False,
                },
                metadata("EMPTY", empty, size=0, inode="3"),
                metadata("CHANGED", changed, inode="4", modified=400),
                metadata("DUPLICATE-B", duplicate, inode="1"),
            ]

            result = InputMetadataInspectionTool().inspect(
                expected, ReturningAdapter(observation(first))
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(
                [
                    (failure.code, failure.role, failure.registered_path)
                    for failure in result.failures
                ],
                [
                    ("input_paths_duplicate", None, str(duplicate)),
                    ("input_missing", "MISSING", str(missing)),
                    ("input_not_regular", "DIRECTORY", str(directory_path)),
                    ("input_empty", "EMPTY", str(empty)),
                ],
            )

    def test_exact_observation_is_retained_for_a_domain_failure(self):
        path = Path("/registered/empty.fastq.gz")
        expected = tool_input((("READ-1", path),))
        raw = observation([metadata("READ-1", path, size=0)])
        result = InputMetadataInspectionTool().inspect(expected, ReturningAdapter(raw))
        self.assertEqual(result.status, "failed")
        self.assertEqual([f.code for f in result.failures], ["input_empty"])
        self.assertEqual(result.observation.model_dump(mode="json"), raw)

    def test_unexpected_adapter_errors_preserve_their_actual_cause(self):
        selected = tool_input((("READ-1", Path("/registered/reads.fastq")),))
        for error in (
            PermissionError("denied"),
            OSError("filesystem unavailable"),
            RuntimeError("adapter defect"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                self.assertRaises(type(error)) as caught,
            ):
                InputMetadataInspectionTool().inspect(selected, RaisingAdapter(error))
            self.assertIs(caught.exception, error)

    def test_malformed_adapter_failure_cannot_be_mixed_with_fabricated_input_facts(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "sole failure"):
            InputMetadataInspectionFailed(
                input_set_id="12c48764-edb7-43da-812e-ff42d8978d73",
                failures=(
                    InputMetadataFailure(code="adapter_output_invalid"),
                    InputMetadataFailure(
                        code="input_missing",
                        role="READ-1",
                        registered_path="/registered/missing.fastq",
                    ),
                ),
                observation=None,
            )

    def test_malformed_adapter_outputs_fail_closed(self):
        root = Path(tempfile.gettempdir()).resolve() / "registered-input-malformed"
        first_path = root / "first.fastq.gz"
        second_path = root / "second.fastq.gz"
        expected = tool_input((("READ-1", first_path), ("READ-2", second_path)))
        valid_items = [
            metadata("READ-1", first_path, inode="1"),
            metadata("READ-2", second_path, inode="2"),
        ]
        valid = observation(valid_items)
        malformed = {}

        malformed["not-an-object"] = []
        value = copy.deepcopy(valid)
        value["unexpected"] = "forbidden"
        malformed["extra-field"] = value
        value = copy.deepcopy(valid)
        del value["completed_at"]
        malformed["missing-field"] = value
        value = copy.deepcopy(valid)
        value["inputs"].reverse()
        malformed["pass-order-substitution"] = value
        value = copy.deepcopy(valid)
        value["inputs"].pop()
        malformed["pass-drop"] = value
        value = copy.deepcopy(valid)
        value["inputs"][1] = copy.deepcopy(value["inputs"][0])
        malformed["pass-duplicate"] = value
        value = copy.deepcopy(valid)
        value["inputs"][0]["role"] = "SUBSTITUTED"
        malformed["role-identity-substitution"] = value
        value = copy.deepcopy(valid)
        value["inputs"][0]["registered_path"] = str(root / "substituted.fastq.gz")
        malformed["path-identity-substitution"] = value
        for field, bad_value in (
            ("exists", 1),
            ("is_regular", 1),
            ("size_bytes", True),
            ("modified_at_ns", False),
        ):
            value = copy.deepcopy(valid)
            value["inputs"][0][field] = bad_value
            malformed[f"strict-{field}"] = value
        value = copy.deepcopy(valid)
        value["inputs"][0]["size_bytes"] = -1
        malformed["negative-size"] = value
        value = copy.deepcopy(valid)
        value["inputs"][0].update({"exists": False, "is_regular": False})
        malformed["missing-with-fabricated-metadata"] = value
        value = copy.deepcopy(valid)
        value["inputs"][0]["filesystem_identity"] = None
        malformed["existing-with-missing-metadata"] = value
        malformed["invalid-timestamp"] = observation(
            valid_items, started="not-a-timestamp"
        )
        malformed["naive-timestamp"] = observation(
            valid_items, started="2026-08-06T10:00:00"
        )
        malformed["reversed-window"] = observation(
            valid_items,
            started="2026-08-06T10:00:02Z",
            completed="2026-08-06T10:00:01Z",
        )

        for name, value in malformed.items():
            with self.subTest(case=name):
                result = InputMetadataInspectionTool().inspect(
                    expected,
                    MalformedAdapter(value),  # type: ignore[arg-type]
                )
                assert_adapter_failure(self, result, expected.input_set_id)


if __name__ == "__main__":
    unittest.main()
