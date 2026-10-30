"""Issue98 inert artifact/flow evidence, with no reusable execution authority."""

import json
from pathlib import Path

from bio_neb.content import sha256_bytes
from bio_neb.devtools.parabricks_fixture_export import (
    ExecutionFixture,
    load_execution_fixture,
)


def load_issue98_execution_fixture() -> ExecutionFixture:
    repository = Path(__file__).resolve().parents[1]
    registration = json.loads(
        (repository / "tests/testing_ownership.json").read_bytes()
    )["additional_execution_fixtures"]["issue98"]
    content = (repository / registration["path"]).read_bytes()
    if (len(content), sha256_bytes(content)) != (
        registration["byte_count"],
        registration["sha256"],
    ):
        raise AssertionError("issue98 execution fixture identity changed")
    return load_execution_fixture(content)
