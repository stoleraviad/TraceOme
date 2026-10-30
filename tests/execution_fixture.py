"""Registered current semantic execution evidence for artifact/flow tests only.

The source is the retained live #22 Run. This loader reads only the checked-in
inert export; it grants no live authority and never visits the original Run.
The ownership document is its identity catalog.
"""

import hashlib
import json
from pathlib import Path

from bio_neb.devtools.parabricks_fixture_export import (
    ExecutionFixture,
    load_execution_fixture,
)


def load_current_execution_fixture() -> ExecutionFixture:
    repository = Path(__file__).resolve().parents[1]
    registration = json.loads(
        (repository / "tests/testing_ownership.json").read_bytes()
    )["execution_fixture"]
    content = (repository / registration["path"]).read_bytes()
    if (len(content), hashlib.sha256(content).hexdigest()) != (
        registration["byte_count"],
        registration["sha256"],
    ):
        raise AssertionError("current execution fixture identity changed")
    return load_execution_fixture(content)
