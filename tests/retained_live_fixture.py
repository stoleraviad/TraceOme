"""Registered retained-live evidence by registration name, with no reusable authority.

Each named registration in `additional_execution_fixtures` pins its document's bytes; the
document's own schema picks the execution, pre-Execution or reached-prefix loader.
"""

import json
from pathlib import Path

from bio_neb.content import sha256_bytes
from bio_neb.devtools.parabricks_fixture_export import (
    ExecutionFixture,
    PreExecutionFixture,
    load_execution_fixture,
    load_pre_execution_fixture,
)
from bio_neb.devtools.reached_fixture_export import ReachedFixture, load_reached_fixture

_LOADERS = {
    "bio-neb.execution-fixture/1": load_execution_fixture,
    "bio-neb.pre-execution-fixture/1": load_pre_execution_fixture,
    "bio-neb.reached-fixture/1": load_reached_fixture,
}


def load_retained_live_fixture(
    name: str, *, content: bytes | None = None
) -> ExecutionFixture | PreExecutionFixture | ReachedFixture:
    """Load one registration; ``content`` reopens candidate bytes in place of its file."""
    repository = Path(__file__).resolve().parents[1]
    registration = json.loads(
        (repository / "tests/testing_ownership.json").read_bytes()
    )["additional_execution_fixtures"][name]
    if content is None:
        content = (repository / registration["path"]).read_bytes()
    if (len(content), sha256_bytes(content)) != (
        registration["byte_count"],
        registration["sha256"],
    ):
        raise AssertionError(f"retained-live fixture identity changed: {name}")
    return _LOADERS[json.loads(content)["schema_version"]](content)
