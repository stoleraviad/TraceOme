"""The retained historical Parabricks fixture bundle used by its regression tests.

The inert bundle records the accepted Ticket-93 cache miss and Ticket-94 cache hit.
Its exact reference identity is fixture evidence only; product/sample selection and
live acceptance remain separate concerns owned by the Agent/HITL and live paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from bio_neb.devtools.parabricks_fixture_export import (
    FixtureScenario,
    ParabricksRunFixtureBundle,
    review_fixture,
)

ACTIVE_PARABRICKS_FIXTURE_BUNDLE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "parabricks-run-fixture.json"
)
ACTIVE_PARABRICKS_FIXTURE_BUNDLE_BYTE_COUNT = 15_129
ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SHA256 = (
    "939060b68698bd2593a069bb4d344a06c3d868dfd6668958a26adecb8dd4d5ad"
)
ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SOURCE_TICKETS = (93, 94)


def load_active_parabricks_fixture_bundle() -> ParabricksRunFixtureBundle:
    """Strictly validate the historical inert Parabricks fixture bundle."""

    content = review_fixture(
        ACTIVE_PARABRICKS_FIXTURE_BUNDLE_PATH,
        expected_sha256=ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SHA256,
    )
    if len(content) != ACTIVE_PARABRICKS_FIXTURE_BUNDLE_BYTE_COUNT:
        raise AssertionError("active Parabricks fixture bundle byte count changed")
    return ParabricksRunFixtureBundle.model_validate_json(content)


def get_active_parabricks_fixture_scenario(
    cache_outcome: Literal["miss", "hit"],
) -> FixtureScenario:
    """Select one inert test scenario explicitly; never select product/live input."""

    if cache_outcome not in {"miss", "hit"}:
        raise ValueError("cache_outcome must be 'miss' or 'hit'")
    return load_active_parabricks_fixture_bundle().scenario(cache_outcome)


__all__ = [
    "ACTIVE_PARABRICKS_FIXTURE_BUNDLE_BYTE_COUNT",
    "ACTIVE_PARABRICKS_FIXTURE_BUNDLE_PATH",
    "ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SHA256",
    "ACTIVE_PARABRICKS_FIXTURE_BUNDLE_SOURCE_TICKETS",
    "get_active_parabricks_fixture_scenario",
    "load_active_parabricks_fixture_bundle",
]
