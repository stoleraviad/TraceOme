"""Contact-free export of a retained Journey prefix at a closed Pass as an inert fixture.

The Fork's cut rule admits the prefix (``read_fork_point``), and the application's
checked view of that prefix supplies what is retained, so an export freezes exactly
what a child Forked at the point would inherit. Export ≠ Fork: a Fork creates a Run
that can act; an export only freezes the reached prefix for offline tests. It
writes nothing to the Journey, contacts nothing and retains no authority document.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Sequence

from pydantic import Field, JsonValue, StringConstraints, model_validator

from bio_neb.contracts import StrictFrozenContract
from bio_neb.devtools.parabricks_fixture_export import (
    EvidenceFileIdentity,
    FixtureExportRefused,
    publish_exclusive,
)
from bio_neb.investigation.download_review import (
    InvestigationDownloadReviewAnswered,
    InvestigationDownloadReviewReceipt,
)
from bio_neb.investigation.investigation import (
    InvestigationAuthority,
    InvestigationReviewReceipt,
    InvestigationReviewResponded,
)
from bio_neb.journey import JourneyFileIdentity, JourneyValidationError, canonical_json
from bio_neb.parabricks.execution_proposal import (
    ExecutionAuthority,
    ExecutionHitlReceipt,
    ExecutionHitlResponded,
)
from bio_neb.parabricks.preparation_review import (
    PreparationInformationAnswered,
    PreparationReceipt,
)
from bio_neb.recording import PassDisposition
from bio_neb.reporting.finalization import (
    FinalizationReviewReceipt,
    FinalizationReviewRecorded,
)

_TEXT = Annotated[str, StringConstraints(min_length=1)]

# What an export may retain of the public observation at the point, by the
# observation's own names: the Run view's fields, or the whole Execution result
# the observation is while an Execution operation is the Run's boundary. The
# header (identity, status, frontier) is always retained; the rest is chosen per
# export, and a projection the observation does not hold is absent.
PROJECTIONS = (
    "execution_result",
    "pending_reviews",
    "progress",
    "execution_proposal",
    "scientific_investigation_handoff",
    "investigation_request",
    "investigation",
    "investigation_evidence",
    "investigation_members",
    "investigation_policy",
    "grounded_report",
    "finalization",
    "model_contact_postmortem",
    "failure_code",
    "failure_source_event_id",
)
DEFAULT_PROJECTIONS = (
    "execution_result",
    "pending_reviews",
    "model_contact_postmortem",
    "failure_code",
    "failure_source_event_id",
)
# Never retained from the Run view: the authority documents the Run acts on, the
# current Review again (it is one of ``pending_reviews``) and the raw contact
# rows (per-Pass accounting summarises them).
EXCLUDED_FIELDS = ("execution_authority", "review", "passes")
# Approval-bearing documents: the grants an approval created and the recorded
# answers of the approval Reviews (Execution, CPU preparation, Investigation,
# download, Finalization), both as projected receipts and as the raw facts a
# basis's accounting retains. An export omits them wherever a projection holds
# them, also inside recorded material such as a basis's accounting, and the
# loader refuses them. The identities that cite them stay as lineage.
APPROVAL_BEARING = (
    ExecutionAuthority,
    InvestigationAuthority,
    ExecutionHitlReceipt,
    ExecutionHitlResponded,
    PreparationReceipt,
    PreparationInformationAnswered,
    InvestigationReviewReceipt,
    InvestigationReviewResponded,
    InvestigationDownloadReviewReceipt,
    InvestigationDownloadReviewAnswered,
    FinalizationReviewReceipt,
    FinalizationReviewRecorded,
)

Projection = Literal[PROJECTIONS]
EligibilitySeam = Literal["not_computed", "execution_contact_review"]


class ReachedPassAccounting(StrictFrozenContract):
    """One Pass's contact journal as the checked view reads it; counts, not rows."""

    pass_id: _TEXT
    started_event_id: _TEXT
    closed_event_id: str | None
    # A Fork child's Passes before its origin are its parent's, never its own.
    inherited: bool
    disposition: Literal["active", "waiting", "completed", "interrupted", "held"]
    journal_complete: bool
    accounting_error: str | None
    started: dict[str, int]
    ended: dict[str, int]
    records: dict[str, int]


class ReachedEligibility(StrictFrozenContract):
    """The seam that judged eligibility; ``not_computed`` claims no verdict."""

    seam: EligibilitySeam
    contact_review_status: Literal["complete", "incomplete", "drift"] | None = None
    fixture_eligible: bool | None = None

    @model_validator(mode="after")
    def verdict_matches_its_seam(self):
        computed = self.seam != "not_computed"
        if computed != (self.contact_review_status is not None) or computed != (
            self.fixture_eligible is not None
        ):
            raise ValueError("an eligibility verdict needs the seam that computed it")
        return self


class ReachedFixtureScenario(StrictFrozenContract):
    """Inert evidence of one reached prefix; never an approval or a Run."""

    source_ticket: int = Field(gt=0)
    source_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    evidence_kind: Literal["synthetic", "retained_live"]
    fixture_grants_live_authority: Literal[False] = False
    journey: EvidenceFileIdentity
    prefix: JourneyFileIdentity
    point_disposition: PassDisposition
    view: Literal["run", "execution_result"]
    frontier: str | None
    status: _TEXT
    path_events: tuple[str, ...]
    passes: tuple[ReachedPassAccounting, ...] = Field(min_length=1)
    projections: dict[Projection, JsonValue]
    eligibility: ReachedEligibility

    @model_validator(mode="after")
    def no_approval_bearing_document(self):
        if _holds_approval(self.projections):
            raise ValueError("a reached fixture retains no approval-bearing document")
        return self

    @model_validator(mode="after")
    def closed_prefix(self):
        if (
            self.prefix.head_event_type != "PassClosed"
            or not self.path_events
            or self.path_events[-1] != "PassClosed"
            or self.passes[-1].closed_event_id != self.prefix.head_event_id
            or self.passes[-1].disposition != self.point_disposition
            or (self.view == "run") != (self.frontier is not None)
        ):
            raise ValueError("a reached fixture ends at the closed Pass it was cut at")
        return self


class ReachedFixture(StrictFrozenContract):
    schema_version: Literal["bio-neb.reached-fixture/1"] = "bio-neb.reached-fixture/1"
    fixture_grants_live_authority: Literal[False] = False
    scenarios: tuple[ReachedFixtureScenario, ...] = Field(min_length=1)


def _approval_document(value: dict) -> bool:
    """Whether a dumped object reads as one of the approval-bearing documents.

    Receipts of different Reviews share their keys, so the object must read
    under the document's own contract; computed fields are dumped, not read.
    """
    for kind in APPROVAL_BEARING:
        fields = kind.model_fields
        required = {name for name, info in fields.items() if info.is_required()}
        if (
            not required
            <= value.keys()
            <= fields.keys() | kind.model_computed_fields.keys()
        ):
            continue
        try:
            kind.model_validate_json(
                canonical_json({key: value[key] for key in value if key in fields})
            )
        except ValueError:
            continue
        return True
    return False


def _holds_approval(value) -> bool:
    """Whether dumped JSON holds an approval-bearing document anywhere."""
    if isinstance(value, dict):
        return _approval_document(value) or any(
            _holds_approval(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_holds_approval(item) for item in value)
    return False


def _without_approval(value):
    """Dumped JSON with every approval-bearing document removed where it sits,
    also inside recorded material such as a basis's accounting."""
    if isinstance(value, dict):
        return {
            key: _without_approval(item)
            for key, item in value.items()
            if not (isinstance(item, dict) and _approval_document(item))
        }
    if isinstance(value, list):
        return [
            _without_approval(item)
            for item in value
            if not (isinstance(item, dict) and _approval_document(item))
        ]
    return value


def _file_identity(path: Path) -> EvidenceFileIdentity:
    content = path.read_bytes()
    return EvidenceFileIdentity(
        path=str(path),
        byte_count=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _accounting(item, prefix) -> ReachedPassAccounting:
    first_own = prefix.revision - prefix.identity().event_count
    return ReachedPassAccounting(
        pass_id=item.pass_id,
        started_event_id=item.started_event_id,
        closed_event_id=item.closed_event_id,
        inherited=prefix.event_position(item.started_event_id) < first_own,
        disposition=item.disposition,
        journal_complete=item.journal_complete,
        accounting_error=item.accounting_error,
        started=dict(
            sorted(
                Counter(
                    row.operation for row in item.contacts if row.kind == "start"
                ).items()
            )
        ),
        ended=dict(
            sorted(
                Counter(
                    f"{row.operation}:{row.disposition}"
                    for row in item.contacts
                    if row.kind == "terminal"
                ).items()
            )
        ),
        records=dict(
            sorted(
                Counter(
                    _record_family(row) for row in item.contacts if row.kind == "record"
                ).items()
            )
        ),
    )


def _record_family(row) -> str:
    """A record row's family: its domain payload's schema, a domain scope
    marker (which carries no payload) or the retained evidence event type."""
    if row.domain is None:
        return row.evidence.event_type
    if row.domain.phase != "record":
        return f"domain:{row.domain.phase}"
    return row.domain.payload["schema_version"]


def _eligibility(seam: EligibilitySeam, prefix) -> ReachedEligibility:
    if seam == "not_computed":
        return ReachedEligibility(seam=seam)
    from bio_neb.journey_path import JourneyPath
    from bio_neb.parabricks.execution_contact_accounting import review_path_contacts

    try:
        # The Contact review (E12) reads only an execution path.
        review = review_path_contacts(JourneyPath.from_journey(prefix))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        raise FixtureExportRefused("eligibility_seam_inapplicable") from error
    return ReachedEligibility(
        seam=seam,
        contact_review_status=review.status,
        fixture_eligible=review.review.fixture_eligible,
    )


def compile_reached_fixture(
    journey_file: Path,
    *,
    point_event_id: str | None,
    source_commit: str,
    source_ticket: int,
    evidence_kind: str,
    projections: Sequence[str] = DEFAULT_PROJECTIONS,
    eligibility: EligibilitySeam = "not_computed",
) -> bytes:
    """Export the checked view of a retained prefix at one closed Pass.

    The point is a ``PassClosed`` of this file (omitted: its head), admitted by
    the Fork's own rule, whose refusal code the export keeps. The caller supplies
    the source revision and origin classification as provenance declarations.
    """
    from bio_neb.application.fork import ForkRefused, read_fork_point
    from bio_neb.application.queries import ObserveRun, observe
    from bio_neb.application.run_interaction import run_interaction
    from bio_neb.parabricks.result import (
        ParabricksExecutionOutcome,
        ParabricksNoncompletion,
    )
    from bio_neb.recording import read_passes

    if not projections or not set(projections) <= set(PROJECTIONS):
        raise FixtureExportRefused("projection_unknown")
    try:
        journey_file = journey_file.resolve(strict=True)
        source = _file_identity(journey_file)
        point, prefix = read_fork_point(str(journey_file), point_event_id)
    except ForkRefused as refusal:
        raise FixtureExportRefused(refusal.code) from refusal
    except OSError as error:
        raise FixtureExportRefused("parent_journey_unreadable") from error
    try:
        # The public observation as `observe` presents it at the point.
        observation = observe(
            ObserveRun(journey_file=str(journey_file)), journey=prefix
        )
        if isinstance(
            observation, (ParabricksExecutionOutcome, ParabricksNoncompletion)
        ):
            view, frontier = "execution_result", None
            held = {"execution_result": observation.model_dump(mode="json")}
        else:
            interaction = run_interaction(observation)
            view, frontier = "run", interaction.frontier
            held = interaction.model_dump(mode="json", exclude=set(EXCLUDED_FIELDS))
        passes = read_passes(prefix)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        # The cut reads Pass journals only; the checked view must read too.
        raise FixtureExportRefused("reached_view_unreadable") from error
    scenario_eligibility = _eligibility(eligibility, prefix)
    try:
        fixture = ReachedFixture(
            scenarios=(
                ReachedFixtureScenario(
                    source_ticket=source_ticket,
                    source_commit=source_commit,
                    evidence_kind=evidence_kind,
                    journey=source,
                    prefix=point.parent,
                    point_disposition=point.point_disposition,
                    view=view,
                    frontier=frontier,
                    status=observation.status,
                    path_events=tuple(
                        event["event_type"] for event in prefix.events_view()
                    ),
                    passes=tuple(_accounting(item, prefix) for item in passes),
                    # A projection the view does not hold at the point is absent.
                    projections=_without_approval(
                        {
                            name: held[name]
                            for name in sorted(set(projections))
                            if name in held
                        }
                    ),
                    eligibility=scenario_eligibility,
                ),
            )
        )
        if _file_identity(journey_file) != source:
            raise ValueError("Journey changed during export")
    except (OSError, ValueError, JourneyValidationError) as error:
        raise FixtureExportRefused("reached_evidence_invalid") from error
    return canonical_json(fixture.model_dump(mode="json")).encode()


def load_reached_fixture(content: bytes) -> ReachedFixture:
    """Strictly load inert exported evidence; never consult its original Run."""
    return ReachedFixture.model_validate_json(content)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    reached = subparsers.add_parser("export-reached")
    reached.add_argument("--journey-file", required=True)
    reached.add_argument("--point-event-id")
    reached.add_argument(
        "--projection", action="append", choices=PROJECTIONS, dest="projections"
    )
    reached.add_argument(
        "--eligibility",
        choices=("not_computed", "execution_contact_review"),
        default="not_computed",
    )
    reached.add_argument("--source-commit", required=True)
    reached.add_argument("--source-ticket", required=True, type=int)
    reached.add_argument(
        "--evidence-kind", required=True, choices=("synthetic", "retained_live")
    )
    reached.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    content = compile_reached_fixture(
        Path(args.journey_file),
        point_event_id=args.point_event_id,
        source_commit=args.source_commit,
        source_ticket=args.source_ticket,
        evidence_kind=args.evidence_kind,
        projections=args.projections or DEFAULT_PROJECTIONS,
        eligibility=args.eligibility,
    )
    publish_exclusive(Path(args.output), content)
    print(
        canonical_json(
            {
                "fixture": args.output,
                "byte_count": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PROJECTIONS",
    "APPROVAL_BEARING",
    "EXCLUDED_FIELDS",
    "PROJECTIONS",
    "ReachedEligibility",
    "ReachedFixture",
    "ReachedFixtureScenario",
    "ReachedPassAccounting",
    "compile_reached_fixture",
    "load_reached_fixture",
    "main",
]
