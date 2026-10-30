"""Named defaults for the Investigation's observation tools (owner, #344 Q9).

They are defaults, not hard-wired policy: a caller may pass other values, and a
later tool or the Agent may choose its own. Coverage has no producer in phase 1;
its defaults are the parameters a coverage tool records when it fills a Work
item's coverage field. The VCF table's chunk size and window are selected by the
Run's runtime file (``investigation_table``), these values when it names none;
so is the coordinator's concurrency bound (``investigation_concurrency``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr

from bio_neb.contracts import FrozenContract


class VcfLookupDefaults(FrozenContract):
    """Limits of the deterministic VCF lookup's commands."""

    preparation_timeout_seconds: StrictInt = Field(default=3_600, gt=0)
    query_timeout_seconds: StrictInt = Field(default=60, gt=0)
    query_output_byte_limit: StrictInt = Field(default=1_048_576, gt=0)
    window_flank_max_bases: StrictInt = Field(default=1_000, gt=0)


class VcfTableDefaults(FrozenContract):
    """The VCF table's window and the table reading's chunk size (owner, #361 Q2).

    Every item's window reads ``window_flank_bases`` on each side of its target;
    one table reading Attempt reads ``chunk_size`` items. Each fact records the
    value it used, so a Run that selects others stays readable.
    """

    chunk_size: StrictInt = Field(default=10, gt=0)
    window_flank_bases: StrictInt = Field(default=100, ge=0)


class RegionDefaults(FrozenContract):
    """A region target's bounds (owner, #367 readiness Q3 a): at most
    ``record_limit`` VCF records are reported as rows, each further one counted; a
    region longer than ``max_bases`` is not checked (its reference bases are read in
    one GET)."""

    record_limit: StrictInt = Field(default=50, gt=0)
    max_bases: StrictInt = Field(default=10_000, gt=0)


class InvestigationConcurrency(FrozenContract):
    """How many independent Investigation Attempts the coordinator runs at once.

    Planner members and table-reader chunks each run as their own Attempt, at
    most this many at a time, and are collected (#105, owner Q5: 4 each). The
    bound changes no recorded fact's meaning, only how many contacts overlap.
    """

    planner_members: StrictInt = Field(default=4, gt=0)
    table_chunks: StrictInt = Field(default=4, gt=0)


def selected_concurrency(runtime_selections) -> InvestigationConcurrency:
    """The Run's recorded bound, or the named defaults when it records none."""
    return runtime_selections.investigation_concurrency or InvestigationConcurrency()


class CoverageDefaults(FrozenContract):
    """Read-depth parameters a coverage tool uses unless it chooses others."""

    minimum_base_quality: StrictInt = Field(default=20, ge=0)
    minimum_mapping_quality: StrictInt = Field(default=20, ge=0)
    excluded_flags: tuple[StrictStr, ...] = ("UNMAP", "SECONDARY", "QCFAIL", "DUP")
    overlapping_pairs: Literal["count_once", "count_each"] = "count_once"
    deletion_spanning_reads: Literal["excluded", "included"] = "excluded"
    minimum_depth: StrictInt = Field(default=10, ge=0)
    applies_baq: StrictBool = False


__all__ = [
    "CoverageDefaults",
    "InvestigationConcurrency",
    "VcfLookupDefaults",
    "VcfTableDefaults",
    "selected_concurrency",
]
