"""Offline engine behavior, using the end-of-dev-cleanup Run only as plan/authority input."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from bio_neb.failure import (
    Ask,
    FailureOverrides,
    FailurePolicy,
    FailureSurface,
    FailureTuning,
    Invariant,
    ProviderError,
    ResolvedFailurePolicy,
    ResolvedLimits,
    ResolvedStageLimits,
    ReviewAnswer,
    StageLimitOverrides,
    limits_for,
    policy_for,
    run_failure_tuning,
)
from bio_neb.journey_path import (
    ExecutionState,
    JourneyPath,
    PathFact,
    StageFailure,
    project_execution,
)
from bio_neb.parabricks.execution_contact_accounting import (
    ExecutionContactJournal,
    ExecutionContactJournalRow,
)
from bio_neb.parabricks.plan import (
    RetainedExecution,
    plan_execution,
)
from bio_neb.parabricks.runner import StageDef, run_stage
from bio_neb.providers.fake import FakeProvider
from bio_neb.providers.ports import Receipt
from tests.retained_execution_path import (
    approved_input,
    path_events,
    stored_objects,
)


def _stage_entry(**operation):
    """The failure section's entry for the engine's stage."""
    return {"scopes": {"stage": {"input_storage": {"operation": operation}}}}


def bucket(ctx, store):
    return store.fresh_bucket(ctx.plan.storage.bucket)


def upload(ctx, store):
    return store.put(
        ctx.plan.storage.bucket, ctx.plan.layout.source_object_key, b"offline"
    )


class FailurePolicyTests(unittest.TestCase):
    def test_an_invariant_holds_under_its_stored_name(self):
        # The stored word says what happens.
        policy = FailurePolicy()
        self.assertEqual(policy.model_dump()["invariant"], "hold")
        self.assertEqual(policy.route("invariant"), "hold")
        with self.assertRaises(ValidationError):
            FailurePolicy(invariant="terminal")

    def test_a_failure_asks_and_no_field_takes_the_retired_word_terminal(self):
        # The default asks the owning Review; "terminal" leaves every field, so
        # no route ends a Run.
        policy = FailurePolicy()
        for failure_class in ("exhausted", "rejected", "worker_failed", "semantic"):
            self.assertEqual(policy.route(failure_class), "ask", failure_class)
        self.assertEqual(policy.route("transient"), "retry")
        self.assertEqual(FailurePolicy(transient="ask").route("transient"), "ask")
        for field in (
            "transient",
            "exhausted",
            "rejected",
            "worker_failed",
            "semantic",
            "invariant",
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                FailurePolicy(**{field: "terminal"})

    def test_a_held_failure_names_an_id_exactly_when_it_is_an_invariant(self):
        from bio_neb.failure import HeldFailure

        held = HeldFailure(failure_class="invariant", invariant_id="bug", cause="x")
        self.assertEqual(held.evidence, {})
        with self.assertRaises(ValidationError):
            HeldFailure(failure_class="invariant", cause="x")
        with self.assertRaises(ValidationError):
            HeldFailure(failure_class="semantic", invariant_id="bug", cause="x")

    def test_resolution_preserves_default_routes_and_explicit_value_precedence(self):
        # With no entries every surface resolves today's defaults.
        defaults = {
            "retry_attempts": 3,
            "backoff_seconds": 1,
            "max_backoff_seconds": 30,
            "transient": "retry",
            "exhausted": "ask",
            "rejected": "ask",
            "worker_failed": "ask",
            "semantic": "ask",
            "invariant": "hold",
        }
        for surface in (
            FailureSurface(scope="stage", name="h100"),
            FailureSurface(scope="tool", name="fetch_refsnp"),
        ):
            resolved = policy_for(surface, FailureTuning())
            self.assertEqual(resolved.surface, surface)
            self.assertEqual(resolved.effective.model_dump(), defaults)
            self.assertEqual(resolved.global_overrides.model_dump(), {})
            self.assertEqual(resolved.surface_overrides.model_dump(), {})
        tuning = FailureTuning.model_validate(
            {
                "operation": {
                    "retry_attempts": 5,
                    "backoff_seconds": 2,
                    "max_backoff_seconds": 7,
                },
                "scopes": {
                    "stage": {
                        "h100": {"operation": {"transient": "ask", "retry_attempts": 3}}
                    }
                },
            }
        )
        resolved = policy_for(FailureSurface(scope="stage", name="h100"), tuning)
        self.assertEqual(resolved.effective.retry_attempts, 3)
        self.assertEqual(resolved.effective.backoff_seconds, 2)
        self.assertEqual(resolved.effective.max_backoff_seconds, 7)
        self.assertEqual(resolved.effective.transient, "ask")
        self.assertEqual(
            resolved.global_overrides.model_dump(),
            {
                "retry_attempts": 5,
                "backoff_seconds": 2,
                "max_backoff_seconds": 7,
            },
        )
        self.assertEqual(
            resolved.surface_overrides.model_dump(),
            {"transient": "ask", "retry_attempts": 3},
        )
        # The h100 entry tunes only h100; a sibling stage keeps the global value.
        sibling = policy_for(
            FailureSurface(scope="stage", name="admit_candidate"), tuning
        )
        self.assertEqual(sibling.effective.retry_attempts, 5)
        self.assertEqual(sibling.effective.transient, "retry")
        self.assertEqual(sibling.surface_overrides.model_dump(), {})

    def test_investigation_stage_surfaces_resolve_before_any_caller(self):
        # Accepted before any caller uses them.
        tuning = FailureTuning.model_validate(
            {
                "operation": {"retry_attempts": 4},
                "scopes": {
                    "stage": {"vcf_observation": {"operation": {"transient": "ask"}}}
                },
            }
        )
        for name in (
            "vcf_materialization",
            "bam_materialization",
            "vcf_observation",
        ):
            with self.subTest(stage=name):
                resolved = policy_for(FailureSurface(scope="stage", name=name), tuning)
                self.assertEqual(resolved.effective.retry_attempts, 4)
                self.assertEqual(
                    resolved.effective.transient,
                    "ask" if name == "vcf_observation" else "retry",
                )

    def test_execution_stage_surfaces_are_the_execution_stage_definitions(self):
        from bio_neb.failure import EXECUTION_STAGES
        from bio_neb.parabricks.h100_stage import H100
        from bio_neb.parabricks.input_storage import INPUT_STORAGE
        from bio_neb.parabricks.reference_stages import ADMIT_CANDIDATE, CPU_PREPARATION

        self.assertEqual(
            EXECUTION_STAGES,
            tuple(
                stage.name
                for stage in (INPUT_STORAGE, ADMIT_CANDIDATE, CPU_PREPARATION, H100)
            ),
        )

    def test_model_contact_resolves_default_then_global_then_role(self):
        # Two sends, one wait within a minute, a 180 s stall limit per send.
        defaults = {
            "contacts": 2,
            "base_delay_seconds": 30,
            "max_delay_seconds": 60,
            "max_wait_seconds": 60,
            "stall_seconds": 180,
        }
        planning = FailureSurface(scope="role", name="planning_agent")
        investigation = FailureSurface(scope="role", name="investigation_planner")
        resolved = policy_for(planning, FailureTuning())
        self.assertEqual(resolved.effective.model_dump(), defaults)
        self.assertEqual(set(resolved.sources().values()), {"default"})
        tuning = FailureTuning.model_validate(
            {
                "model_contact": {"max_wait_seconds": 90},
                "scopes": {
                    "role": {"planning_agent": {"model_contact": {"contacts": 3}}}
                },
            }
        )
        resolved = policy_for(planning, tuning)
        self.assertEqual(
            resolved.effective.model_dump(),
            {**defaults, "contacts": 3, "max_wait_seconds": 90},
        )
        self.assertEqual(
            resolved.sources(),
            {
                "contacts": "role",
                "base_delay_seconds": "default",
                "max_delay_seconds": "default",
                "max_wait_seconds": "global",
                "stall_seconds": "default",
            },
        )
        other = policy_for(investigation, tuning)
        self.assertEqual(
            other.effective.model_dump(), {**defaults, "max_wait_seconds": 90}
        )

    def test_source_tool_limits_resolve_from_code_with_the_run_total(self):
        # A Tool's limits are code, the module's defaults that the Tool's own
        # declaration overrides; the runtime sets only the Run total.
        defaults = {"contacts": 30, "response_bytes": 4 * 1024 * 1024}
        refsnp = FailureSurface(scope="tool", name="fetch_refsnp")
        companion = FailureSurface(scope="tool", name="describe_companion")
        resolved = limits_for(refsnp, FailureTuning())
        self.assertEqual(resolved.effective.model_dump(), defaults)
        self.assertEqual(resolved.tool_overrides.model_dump(), {})
        self.assertIsNone(resolved.run_contacts)
        self.assertEqual(set(resolved.sources().values()), {"default"})
        tuning = FailureTuning.model_validate({"limits": {"run_contacts": 40}})
        resolved = limits_for(companion, tuning)
        # The companion's surface reads the whole assembly report (cap a).
        self.assertEqual(
            resolved.effective.model_dump(),
            {**defaults, "response_bytes": 256 * 1024 * 1024},
        )
        self.assertEqual(
            resolved.sources(), {"contacts": "default", "response_bytes": "tool"}
        )
        self.assertEqual(resolved.run_contacts, 40)
        # The record validates against its own recorded layers.
        recorded = resolved.model_dump()
        self.assertEqual(ResolvedLimits.model_validate(recorded), resolved)
        changed = {**recorded, "effective": {**recorded["effective"], "contacts": 3}}
        with self.assertRaisesRegex(ValueError, "disagree with the Tool"):
            ResolvedLimits.model_validate(changed)
        del recorded["effective"]["contacts"]
        with self.assertRaisesRegex(ValueError, "every setting"):
            ResolvedLimits.model_validate(recorded)
        # A role has no limits; a Run that names none registers nothing.
        with self.assertRaisesRegex(ValueError, "per source Tool or stage"):
            limits_for(
                FailureSurface(scope="role", name="planning_agent"), FailureTuning()
            )
        self.assertEqual(FailureTuning().model_dump(), {})
        self.assertEqual(
            FailureTuning.model_validate({"limits": {}}).model_dump(), {"limits": {}}
        )

    def test_stage_part_limits_resolve_from_code_without_a_runtime_entry(self):
        # 300 s per 32 MiB object part, a module default the stage may
        # override in its own code declaration.
        bam = FailureSurface(scope="stage", name="bam_materialization")
        resolved = limits_for(bam, FailureTuning())
        self.assertIsInstance(resolved, ResolvedStageLimits)
        self.assertEqual(
            resolved.effective.model_dump(),
            {"part_seconds": 300, "process_seconds": 600},
        )
        self.assertEqual(
            resolved.sources(),
            {"part_seconds": "default", "process_seconds": "default"},
        )
        with patch.dict(
            "bio_neb.failure._faults.STAGE_LIMITS",
            {"bam_materialization": StageLimitOverrides(part_seconds=120)},
        ):
            declared = limits_for(bam, FailureTuning())
            other = limits_for(
                FailureSurface(scope="stage", name="vcf_materialization"),
                FailureTuning(),
            )
        self.assertEqual(declared.effective.part_seconds, 120)
        self.assertEqual(
            declared.sources(),
            {"part_seconds": "stage", "process_seconds": "default"},
        )
        self.assertEqual(other.effective.part_seconds, 300)
        # The record validates against its own recorded layers, not current code.
        recorded = declared.model_dump(mode="json")
        self.assertEqual(
            recorded["stage_overrides"], {"part_seconds": 120}, "sparse declaration"
        )
        self.assertEqual(ResolvedStageLimits.model_validate(recorded), declared)
        with self.assertRaisesRegex(ValueError, "disagree with the stage"):
            ResolvedStageLimits.model_validate(
                {
                    **recorded,
                    "effective": {"part_seconds": 300, "process_seconds": 600},
                }
            )
        with self.assertRaisesRegex(ValueError, "every setting"):
            ResolvedStageLimits.model_validate({**recorded, "effective": {}})
        # The Run's measurement declares its own process limit.
        health = limits_for(FailureSurface(scope="stage", name="run_health"))
        self.assertEqual(
            (health.effective.process_seconds, health.sources()["process_seconds"]),
            (3600, "stage"),
        )
        self.assertEqual(health.effective.part_seconds, 300)
        # No runtime entry tunes it: the failure section refuses a stage limit.
        with self.assertRaises(ValidationError):
            FailureTuning.model_validate(
                {"scopes": {"stage": {"bam_materialization": {"limits": {}}}}}
            )
        with self.assertRaises(ValidationError):
            FailureTuning.model_validate({"limits": {"part_seconds": 60}})

    def test_runtime_limits_hold_only_the_run_total(self):
        for section, path in (
            ({"limits": {"run_contacts": None}}, ("limits", "run_contacts")),
            ({"limits": {"run_contacts": 0}}, ("limits", "run_contacts")),
            ({"limits": {"contacts": 5}}, ("limits", "contacts")),
            ({"limits": {"response_bytes": 5}}, ("limits", "response_bytes")),
            (
                {"scopes": {"tool": {"fetch_refsnp": {"limits": {"contacts": 5}}}}},
                ("scopes", "tool", "fetch_refsnp", "limits"),
            ),
        ):
            with self.subTest(section=section):
                with self.assertRaises(ValidationError) as refused:
                    FailureTuning.model_validate(section)
                self.assertEqual(
                    [error["loc"] for error in refused.exception.errors()], [path]
                )

    def test_invariant_cannot_be_made_retryable(self):
        with self.assertRaisesRegex(ValueError, "invariant"):
            FailureOverrides(invariant="retry")
        with self.assertRaisesRegex(ValueError, "bugs use Invariant"):
            ProviderError("invariant", "bug")

    def test_incomplete_or_inconsistent_recorded_policy_is_rejected(self):
        tuning = FailureTuning.model_validate({"operation": {"retry_attempts": 5}})
        surface = FailureSurface(scope="stage", name="h100")
        recorded = policy_for(surface, tuning).model_dump()
        del recorded["effective"]["semantic"]
        with self.assertRaisesRegex(ValueError, "every setting"):
            ResolvedFailurePolicy.model_validate(recorded)
        recorded = policy_for(surface, tuning).model_dump()
        recorded["effective"]["retry_attempts"] = 3
        with self.assertRaisesRegex(ValueError, "disagrees with its overrides"):
            ResolvedFailurePolicy.model_validate(recorded)


class EngineTests(unittest.TestCase):
    def test_reading_retained_policy_ignores_changed_defaults_and_keeps_old_absence(
        self,
    ):
        state = self.run_engine(
            FakeProvider(
                {
                    ("input_storage", "bucket"): [
                        ProviderError("semantic", "invalid_marker")
                    ],
                }
            )
        )
        self.assertEqual(state.status, "execution_review_pending")
        old_path = self.journey.file.read_bytes()
        old_journal = self.journal.path.read_bytes()
        with patch(
            "bio_neb.failure._faults.DEFAULT_FAILURE_POLICY",
            FailurePolicy(
                retry_attempts=9,
                transient="ask",
                backoff_seconds=10,
            ),
        ):
            reopened = project_execution(JourneyPath.open(self.journey.file).read())
            failure = reopened.failures[0]
            recorded = ResolvedFailurePolicy.model_validate(
                failure.context["failure_policy"]
            )
            self.assertEqual(recorded.effective.retry_attempts, 3)
            self.assertEqual(recorded.effective.backoff_seconds, 1)
            self.assertEqual(recorded.effective.semantic, "ask")
            self.assertEqual(failure.context["route"], "ask")
            self.assertEqual(failure.context["original_failure_class"], "semantic")
            retained_row = ExecutionContactJournalRow.model_validate(self.rows()[0])
            self.assertEqual(retained_row.failure_policy.effective.semantic, "ask")
            # Historical records have no policy. Absence is never filled from defaults.
            historical_row = self.rows()[0]
            del historical_row["failure_policy"]
            self.assertIsNone(
                ExecutionContactJournalRow.model_validate(historical_row).failure_policy
            )
            historical_failure = failure.model_dump()
            del historical_failure["context"]
            self.assertEqual(
                StageFailure.model_validate(historical_failure).context, {}
            )
        self.assertEqual(self.journey.file.read_bytes(), old_path)
        self.assertEqual(self.journal.path.read_bytes(), old_journal)

    def test_policy_serialization_preserves_omission_and_explicit_defaults(self):
        # Entries and the failure section both cross Pydantic JSON boundaries.
        override = FailureOverrides(semantic="ask", retry_attempts=3)
        self.assertEqual(
            json.loads(override.model_dump_json()),
            {"semantic": "ask", "retry_attempts": 3},
        )
        restored = type(override).model_validate_json(override.model_dump_json())
        self.assertEqual(restored.model_fields_set, {"semantic", "retry_attempts"})
        # An explicit default stays explicit; an omission stays omitted.
        section = {"scopes": {"stage": {"h100": {"operation": {"retry_attempts": 3}}}}}
        tuning = FailureTuning.model_validate(section)
        self.assertEqual(json.loads(tuning.model_dump_json()), section)
        self.assertEqual(json.loads(FailureTuning().model_dump_json()), {})

    def test_no_capacity_selects_only_the_next_approved_runtime_and_replays_it(self):
        self.check_runtime_fallback()

    def test_gpu_amendment_keeps_approved_fallback_available_after_human_answer(self):
        self.check_runtime_fallback(amend=True)

    def test_no_capacity_after_submission_never_substitutes_an_existing_job(self):
        self.check_runtime_fallback(known_job=True)

    def check_runtime_fallback(self, *, amend=False, known_job=False):
        from bio_neb.journey_path import ExecutionBound
        from tests.reference_junction_fakes import approved_with_runtimes

        approved = approved_with_runtimes()
        journey = JourneyPath.create(Path(self.root.name) / "fallback")
        journey.path(
            journey.state,
            PathFact(
                event_type="ExecutionBound",
                kind="authority",
                grounding="human",
                **{"pass": 1},
                payload=ExecutionBound(
                    execution_id=approved.execution_id,
                    authority_id=approved.authority.authority_id,
                    proposal_id=approved.proposal.proposal_id,
                    run_id=approved.proposal.journey_id,
                    approved_input=approved.model_dump(mode="json"),
                ),
            ),
        )
        self.journey = journey
        self.state = project_execution(journey.read())
        self.plan = plan_execution(
            approved,
            RetainedExecution(self.state, approved.authority, approved.proposal),
        )
        self.journal = ExecutionContactJournal(
            Path(self.root.name) / "fallback-contacts.jsonl", approved.execution_id
        )
        platforms = []

        def submit(ctx, jobs):
            platforms.append(ctx.plan.runtimes.h100.platform)
            return jobs.submit(ctx.plan.layout.h100_job_name, {})

        def sample(ctx, jobs):
            return jobs.observe(ctx.receipts[-1].provider_ids[0], window=1)

        stage = replace(
            self.stage,
            name="h100",
            steps=(submit, sample) if known_job else (submit,),
            ports={"submit": "jobs", **({"sample": "jobs"} if known_job else {})},
            runtime="h100",
        )
        if amend:
            stopped = self.run_engine(
                FakeProvider(
                    {
                        ("h100", "submit"): [
                            ProviderError("rejected", "permission_denied")
                        ]
                    }
                ),
                stage,
            )
            answer = PathFact(
                event_type="FailureReviewAnswered",
                kind="authority",
                grounding="human",
                **{"pass": 2},
                rests_on=(stopped.latest_event_ids["FailureReviewRequested"],),
                payload=ReviewAnswer(
                    request_id=stopped.failure_review.request_id,
                    action="amend",
                    field="gpu_class",
                    value="alternate",
                    by="owner",
                ),
            )
            journey.path(journey.state, answer)
            self.state = project_execution(journey.read())
            self.journal.finalize()
            self.journal = ExecutionContactJournal(
                Path(self.root.name) / "amended-contacts.jsonl",
                approved.execution_id,
                execution_pass=2,
            )
            platforms.clear()
        provider = FakeProvider(
            {
                ("h100", "sample" if known_job else "submit"): [
                    ProviderError(
                        "rejected", "no_capacity", {"resource_absent": not known_job}
                    )
                ]
            }
        )
        state = self.run_engine(provider, stage)
        if known_job:
            self.assertIsInstance(state.failure_review, Ask)
            self.assertEqual(state.failure_review.retry_consequence, "reuse_intent")
            self.assertNotIn("RuntimeSubstituted", state.latest_event_ids)
            self.assertEqual(len(provider.jobs_by_name), 1)
            return
        self.assertEqual(platforms, ["gpu-alternative", "gpu-fallback"])
        self.assertEqual(state.inputs.state, "verified")
        event = next(
            event
            for event in journey.read()
            if event.event_type == "RuntimeSubstituted"
        )
        self.assertEqual((event.kind, event.grounding), ("decision", "human"))
        self.assertIn(state.latest_event_ids["ExecutionBound"], event.rests_on)
        self.assertEqual(event.payload.to_profile, "fallback")
        if amend:
            self.assertIn(
                self.state.latest_event_ids["FailureReviewAnswered"], event.rests_on
            )
        reopened = project_execution(JourneyPath.open(journey.file).read())
        plan = plan_execution(
            approved, RetainedExecution(reopened, approved.authority, approved.proposal)
        )
        self.assertEqual(plan.runtimes.h100.platform, "gpu-fallback")
        self.assertEqual(plan.runtimes.fallbacks, ())
        if amend:
            self.assertNotEqual(
                plan.layout.h100_job_name, self.plan.layout.h100_job_name
            )
        else:
            self.assertEqual(plan.layout.h100_job_name, self.plan.layout.h100_job_name)

    def test_fallback_exhaustion_and_unrelated_rejections_require_a_declared_review(
        self,
    ):
        # Reuse the public fallback scenario's setup, then start fresh for each route.
        from bio_neb.journey_path import ExecutionBound
        from tests.reference_junction_fakes import approved_with_runtimes

        for detail, expected_substitutions in (
            ("no_capacity", 1),
            ("permission_denied", 0),
        ):
            with self.subTest(detail=detail):
                self.start_run()
                approved = approved_with_runtimes()
                self.journey = JourneyPath.create(Path(self.root.name) / "fallback")
                self.journey.path(
                    self.journey.state,
                    PathFact(
                        event_type="ExecutionBound",
                        kind="authority",
                        grounding="human",
                        **{"pass": 1},
                        payload=ExecutionBound(
                            execution_id=approved.execution_id,
                            authority_id=approved.authority.authority_id,
                            proposal_id=approved.proposal.proposal_id,
                            run_id=approved.proposal.journey_id,
                            approved_input=approved.model_dump(mode="json"),
                        ),
                    ),
                )
                self.state = project_execution(self.journey.read())
                self.plan = plan_execution(
                    approved,
                    RetainedExecution(
                        self.state, approved.authority, approved.proposal
                    ),
                )
                self.journal = ExecutionContactJournal(
                    Path(self.root.name) / "fallback-contacts.jsonl",
                    approved.execution_id,
                )

                def submit(ctx, jobs):
                    return jobs.submit(ctx.plan.layout.h100_job_name, {})

                stage = replace(
                    self.stage,
                    name="h100",
                    steps=(submit,),
                    ports={"submit": "jobs"},
                    runtime="h100",
                )
                state = self.run_engine(
                    FakeProvider(
                        {
                            ("h100", "submit"): [
                                ProviderError(
                                    "rejected", detail, {"resource_absent": True}
                                )
                            ]
                            * 2
                        }
                    ),
                    stage,
                )
                self.assertEqual(
                    sum(
                        event.event_type == "RuntimeSubstituted"
                        for event in self.journey.read()
                    ),
                    expected_substitutions,
                )
                self.assertIsInstance(state.failure_review, Ask)
                self.assertEqual(state.failure_review.amendable, ("gpu_class",))
                self.assertIn("amend", state.failure_review.legal_actions)
                self.assertIsNone(state.review)

    def setUp(self):
        self.start_run()

    def start_run(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.journey = JourneyPath.create(Path(self.root.name))
        approved = approved_input()
        first = path_events()[0]
        self.journey.path(
            self.journey.state,
            PathFact.model_validate(
                {k: v for k, v in first.items() if k != "event_id"}
            ),
        )
        self.state = project_execution(self.journey.read())
        self.plan = plan_execution(
            approved,
            RetainedExecution(ExecutionState(), approved.authority, approved.proposal),
        )
        self.journal = ExecutionContactJournal(
            Path(self.root.name) / "contacts.jsonl", self.plan.identity.execution_id
        )
        self.live = []
        self.stage = StageDef(
            name="input_storage",
            steps=(bucket, upload),
            ports={"bucket": "store", "upload": "store"},
            path_fact=self.input_fact,
        )

    def input_fact(self, receipts):
        row = {k: v for k, v in path_events()[1].items() if k != "event_id"}
        row["payload"]["pointers"] = stored_objects(row["payload"]["pointers"])
        # As the product's `inputs_verified`: the bound input and any answer enabling
        # this stage again.
        row["rests_on"] = [
            self.state.latest_event_ids["ExecutionBound"],
            *self.state.stage_answer_ids.get("input_storage", ()),
        ]
        row["pass"] = self.state.pass_
        row["payload"]["receipt_id"] = receipts[-1].proof_ref
        return PathFact.model_validate(row)

    def run_engine(self, provider, stage=None, live=None, failure=None):
        """Run the stage under ``failure``, the Run's failure section (#223)."""
        with run_failure_tuning(FailureTuning.model_validate(failure or {})):
            return run_stage(
                self.state,
                self.plan,
                stage or self.stage,
                provider,
                journey=self.journey,
                journal=self.journal,
                live=self.live.append if live is None else live,
            )

    def rows(self):
        return [json.loads(line) for line in self.journal.path.read_text().splitlines()]

    def test_invalid_outcome_refuses_before_contact_journal_finalization(self):
        from bio_neb.journey_path import Outcome
        from bio_neb.parabricks.runner import record_outcome

        self.state = self.run_engine(FakeProvider())
        path_before = self.journey.file.read_bytes()
        journal_before = self.journal.path.read_bytes()
        invalid = PathFact(
            event_type="ExecutionOutcome",
            kind="outcome",
            grounding="verified",
            **{"pass": 1},
            rests_on=(self.state.latest_event_ids["InputsVerified"],),
            payload=Outcome(
                disposition="succeeded", contact_journal=str(self.journal.path)
            ),
        )
        with self.assertRaisesRegex(ValueError, "success requires verified H100"):
            record_outcome(self.state, self.journey, invalid, journal=self.journal)
        self.assertEqual(self.journey.file.read_bytes(), path_before)
        self.assertEqual(self.journal.path.read_bytes(), journal_before)
        valid = invalid.model_copy(
            update={
                "payload": Outcome(
                    disposition="incomplete",
                    milestone="input_storage",
                    contact_journal=str(self.journal.path),
                )
            }
        )
        state = record_outcome(self.state, self.journey, valid, journal=self.journal)
        self.assertTrue(self.journal.closed)
        self.assertIsNotNone(state.outcome.journal_sha256)
        self.assertEqual(project_execution(self.journey.read()), state)

    @patch("bio_neb.parabricks.runner.time.sleep")
    def test_transient_retry_then_two_steps_write_one_path_fact(self, sleep):
        provider = FakeProvider(
            {("input_storage", "bucket"): [ProviderError("transient", "try_again")]}
        )
        state = self.run_engine(provider)
        self.assertEqual(state.inputs.state, "verified")
        self.assertEqual(len(self.journey.read()), 2)
        self.assertEqual(project_execution(self.journey.read()), state)
        self.assertEqual(len(provider.buckets_by_name), 1)
        self.assertEqual(len(provider.objects), 1)
        self.assertEqual(sleep.call_count, 1)
        attempts = [
            r["attempt"]
            for r in self.rows()
            if r["step"] == "bucket" and r["phase"] == "claimed"
        ]
        self.assertEqual(attempts, [1, 2])
        self.assertTrue(self.live)

    @patch("bio_neb.parabricks.runner.time.sleep")
    def test_exhaustion_opens_generic_ask_and_retains_failure(self, sleep):
        provider = FakeProvider(
            {("input_storage", "bucket"): [ProviderError("transient", "try_again")] * 3}
        )
        state = self.run_engine(provider)
        self.assertIsInstance(state.failure_review, Ask)
        self.assertEqual(state.failures[0].failure_class, "exhausted")
        self.assertEqual(state.disposition, "incomplete")
        self.assertEqual(state.frontier, "failure_review_pending")
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(
            [e.event_type for e in self.journey.read()][1:],
            ["StageFailed", "FailureReviewRequested", "ExecutionOutcome"],
        )
        self.assertEqual(project_execution(self.journey.read()), state)
        self.assertTrue(self.journey.read_artifact(state.failures[0].evidence[0]))

    def test_adjustable_semantic_policy_and_provider_defaults(self):
        for failure_class in ("rejected", "worker_failed", "semantic"):
            with self.subTest(failure_class=failure_class):
                provider = FakeProvider(
                    {
                        ("input_storage", "bucket"): [
                            ProviderError(failure_class, "stopped")
                        ]
                    }
                )
                state = self.run_engine(provider)
                self.assertEqual(state.status, "execution_review_pending")
                self.start_run()

    def test_invariant_is_retained_then_reraised_and_bug_is_not_reclassified(self):
        def broken(ctx, identity):
            self.assertFalse(hasattr(ctx, "journey"))
            identity.whoami()
            raise Invariant("closure_bug", "invalid closure")

        stage = replace(self.stage, steps=(broken,), ports={"broken": "identity"})
        with self.assertRaisesRegex(Invariant, "invalid closure"):
            self.run_engine(FakeProvider(), stage)
        state = project_execution(self.journey.read())
        # Our bug holds the Execution for a fix; no outcome ends it.
        self.assertIsNone(state.outcome)
        self.assertEqual(state.status, "run_held")
        self.assertEqual(state.failures[0].invariant_id, "closure_bug")

    def test_programming_fault_preserves_immediate_submission_receipt(self):
        def broken(ctx, jobs):
            jobs.submit(ctx.plan.layout.cpu_job_name, {})
            raise TypeError("programming bug")

        stage = replace(self.stage, steps=(broken,), ports={"broken": "jobs"})
        with self.assertRaisesRegex(TypeError, "programming bug"):
            self.run_engine(FakeProvider(), stage)
        self.assertEqual(len(self.journey.read()), 1)
        self.assertTrue(
            any(
                r.get("receipt", {}).get("kind") == "job_submission"
                for r in self.rows()
                if r.get("receipt")
            )
        )

    def test_arbitrary_step_name_reuses_one_job_across_two_logical_attempts(self):
        from bio_neb.journey_path import Outcome
        from bio_neb.parabricks.execution_contact_accounting import (
            compile_execution_contact_evidence,
        )
        from bio_neb.parabricks.runner import record_outcome

        def launch_named_work(ctx, jobs):
            jobs.submit(ctx.plan.layout.cpu_job_name, {})
            return jobs.submit(ctx.plan.layout.cpu_job_name, {})

        stage = replace(
            self.stage, steps=(launch_named_work,), ports={"launch_named_work": "jobs"}
        )
        state = self.run_engine(FakeProvider(), stage)
        state = record_outcome(
            state,
            self.journey,
            PathFact(
                event_type="ExecutionOutcome",
                kind="outcome",
                grounding="verified",
                **{"pass": 1},
                rests_on=(state.latest_event_ids["InputsVerified"],),
                payload=Outcome(
                    disposition="incomplete",
                    milestone="input_storage",
                    contact_journal=str(self.journal.path),
                ),
            ),
            journal=self.journal,
        )
        review = compile_execution_contact_evidence(
            journal_bytes=self.journal.path.read_bytes(),
            binding=self.journal.binding(),
            expected_execution_id=self.plan.identity.execution_id,
            journey_events=[
                e.model_dump(mode="json", by_alias=True) for e in self.journey.read()
            ],
            journey_binding="offline-path",
        )
        self.assertTrue(review.complete, review.discrepancies)
        self.assertEqual(len(review.logical_attempts), 2)
        self.assertEqual(review.ledger.cpu_preparation_create.count, 1)
        self.assertEqual(review.ledger.h100_create.count, 0)
        self.assertEqual(review.ledger.provider_contact.status, "unmeasurable")

    def test_throwing_display_cannot_change_execution(self):
        def throwing(event):
            raise RuntimeError("display failed")

        state = self.run_engine(FakeProvider(), live=throwing)
        self.assertEqual(state.inputs.state, "verified")

    def test_named_mutations_require_exactly_one_resource(self):
        def submit(ctx, jobs):
            return jobs.submit(ctx.plan.layout.cpu_job_name, {})

        for step, port in [(bucket, "store"), (submit, "jobs")]:
            for count in [None, 0, 2]:
                with self.subTest(step=step.__name__, count=count):
                    self.start_run()
                    stage = replace(
                        self.stage, steps=(step,), ports={step.__name__: port}
                    )
                    provider = FakeProvider(
                        {
                            ("input_storage", step.__name__): [
                                Receipt(resource_count=count)
                            ]
                        }
                    )
                    with self.assertRaises(Invariant):
                        self.run_engine(provider, stage)
                    state = project_execution(self.journey.read())
                    self.assertEqual(state.status, "run_held")
                    self.assertEqual(
                        state.failures[0].invariant_id, "named_resource_ceiling"
                    )
                    returned = [
                        row for row in self.rows() if row["phase"] == "verified"
                    ]
                    self.assertEqual(returned[0]["ceiling_count"], count)

    def test_stage_observation_uses_shared_closure_without_poll_path_events(self):
        def watch(ctx, jobs):
            return ctx.observe(
                lambda: jobs.observe("job:offline", window=10), max_wait=0
            )

        stage = replace(
            self.stage,
            steps=(watch,),
            ports={"watch": "jobs"},
            closure=lambda samples: (
                samples[-1] if samples[-1].values["state"] == "COMPLETED" else None
            ),
        )
        state = self.run_engine(FakeProvider(), stage)
        self.assertEqual(state.inputs.state, "verified")
        self.assertEqual(len(self.journey.read()), 2)
        self.assertEqual(self.live[-1]["state"], "COMPLETED")

    def test_human_retry_reuses_bucket_and_preserves_prior_failure(self):
        provider = FakeProvider(
            {("input_storage", "upload"): [ProviderError("exhausted", "interrupted")]}
        )
        state = self.run_engine(provider)
        with self.assertRaisesRegex(ValueError, "designed stop"):
            run_stage(
                state,
                self.plan,
                self.stage,
                provider,
                journey=self.journey,
                journal=self.journal,
                live=None,
            )
        request = state.failure_review
        answer = PathFact(
            event_type="FailureReviewAnswered",
            kind="authority",
            grounding="human",
            **{"pass": 2},
            rests_on=(state.latest_event_ids["FailureReviewRequested"],),
            payload=ReviewAnswer(
                request_id=request.request_id, action="retry", by="owner"
            ),
        )
        _, self.state = self.journey.path(state, answer)
        self.journal = ExecutionContactJournal(
            Path(self.root.name) / "pass2.jsonl",
            self.plan.identity.execution_id,
            execution_pass=2,
        )
        state = self.run_engine(provider)
        self.assertEqual(state.inputs.state, "verified")
        self.assertEqual(state.failures[0].failure_class, "exhausted")
        self.assertFalse(state.unresolved_failure_ids)
        self.assertEqual(len(provider.buckets_by_name), 1)
        self.assertEqual(project_execution(self.journey.read()), state)

    @patch("bio_neb.parabricks.runner.random.uniform", return_value=0.25)
    @patch("bio_neb.parabricks.runner.time.sleep")
    def test_unrelated_stage_override_preserves_five_plan_attempts(self, sleep, jitter):
        # The global value gives five attempts; the stage entry tunes another class.
        state = self.run_engine(
            FakeProvider(
                {("input_storage", "bucket"): [ProviderError("transient", "again")] * 4}
            ),
            failure={
                "operation": {"retry_attempts": 5},
                **_stage_entry(semantic="ask"),
            },
        )
        self.assertEqual(state.inputs.state, "verified")
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(
            [call.args for call in jitter.call_args_list],
            [(0, 1), (0, 2), (0, 4), (0, 8)],
        )
        failed = [
            row
            for row in self.rows()
            if row["step"] == "bucket" and row["phase"] == "failed"
        ]
        self.assertTrue(failed)
        for row in failed:
            self.assertEqual(row["failure_class"], "transient")
            self.assertIn("failure_policy", row)
            self.assertEqual(row["failure_policy"]["effective"]["retry_attempts"], 5)
            self.assertEqual(
                row["failure_policy"]["global_overrides"], {"retry_attempts": 5}
            )
            self.assertEqual(
                row["failure_policy"]["surface_overrides"], {"semantic": "ask"}
            )
        self.assertEqual(state.failures, ())

    @patch("bio_neb.parabricks.runner.random.uniform", return_value=0.25)
    @patch("bio_neb.parabricks.runner.time.sleep")
    def test_explicit_stage_allowance_and_capped_backoff_retain_exhausted_policy(
        self, sleep, jitter
    ):
        state = self.run_engine(
            FakeProvider(
                {("input_storage", "bucket"): [ProviderError("transient", "again")] * 4}
            ),
            failure={
                "operation": {"retry_attempts": 5},
                **_stage_entry(
                    retry_attempts=3, backoff_seconds=4, max_backoff_seconds=5
                ),
            },
        )
        self.assertIsInstance(state.failure_review, Ask)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(
            [call.args for call in jitter.call_args_list], [(0, 4), (0, 5)]
        )
        failure = state.failures[0]
        self.assertEqual(failure.failure_class, "exhausted")
        self.assertIn("failure_policy", failure.context)
        self.assertEqual(failure.context["original_failure_class"], "transient")
        self.assertEqual(failure.context["route"], "ask")
        self.assertEqual(
            failure.context["failure_policy"]["surface_overrides"],
            {
                "retry_attempts": 3,
                "backoff_seconds": 4,
                "max_backoff_seconds": 5,
            },
        )
        proof = failure.context["proof_ref"]
        row = next(row for row in self.rows() if proof == f"cj:1:{row['sequence']}")
        self.assertEqual(row["failure_class"], "transient")
        self.assertEqual(row["failure_policy"], failure.context["failure_policy"])
        self.assertEqual(
            state.failure_review.failure_event_id, state.latest_event_ids["StageFailed"]
        )

    @patch("bio_neb.parabricks.runner.random.uniform", return_value=0.25)
    @patch("bio_neb.parabricks.runner.time.sleep")
    def test_plan_retry_budget_and_capped_jitter_are_honored(self, sleep, jitter):
        provider = FakeProvider(
            {("input_storage", "bucket"): [ProviderError("transient", "again")] * 3}
        )
        state = self.run_engine(provider, failure={"operation": {"retry_attempts": 2}})
        self.assertIsInstance(state.failure_review, Ask)
        self.assertEqual(sleep.call_args.args, (0.25,))
        jitter.assert_called_once_with(0, 1)

    def test_a_step_runs_its_contacts_under_its_stages_resolved_policy(self):
        # A part transfer inside a step retries each part by the stage's
        # policy; outside the step no stage is bound.
        from bio_neb.failure import UnboundOperationPolicy, bound_operation_policy

        seen = []

        def observed_bucket(ctx, store):
            seen.append(bound_operation_policy())
            return bucket(ctx, store)

        stage = StageDef(
            name="input_storage",
            steps=(observed_bucket, upload),
            ports={"observed_bucket": "store", "upload": "store"},
            path_fact=self.input_fact,
        )
        self.run_engine(FakeProvider(), stage, failure=_stage_entry(retry_attempts=2))
        (policy,) = seen
        self.assertEqual(policy.surface.name, "input_storage")
        self.assertEqual(policy.effective.retry_attempts, 2)
        self.assertIsInstance(bound_operation_policy(), UnboundOperationPolicy)

    def test_a_runtime_entry_naming_the_retired_word_terminal_is_refused(self):
        # "terminal" leaves every field, so a runtime failure entry cannot state
        # it.
        from bio_neb.failure import FailureTuning

        with self.assertRaises(ValidationError):
            FailureTuning.model_validate(
                {
                    "scopes": {
                        "stage": {
                            "input_storage": {"operation": {"rejected": "terminal"}}
                        }
                    }
                }
            )

    def test_h100_request_compiler_rejection_is_retained_as_our_invariant(self):
        from bio_neb.parabricks.h100_stage import H100

        provider = FakeProvider()
        state = self.run_engine(provider)
        # The Run's reference hit, with an intentionally malformed current plan:
        # the real compiler must reject an absent required BAM output.
        reference = path_events()[2]
        reference["rests_on"] = [state.latest_event_ids["InputsVerified"]]
        hit = reference["payload"]["reference"]
        hit["pointers"] = stored_objects(hit["pointers"])
        _, state = self.journey.path(
            state,
            PathFact.model_validate(
                {k: v for k, v in reference.items() if k != "event_id"}
            ),
        )
        plan = replace(
            self.plan,
            layout=self.plan.layout.model_copy(update={"h100_output_keys": ()}),
        )
        with self.assertRaisesRegex(Invariant, "h100_request_invalid"):
            run_stage(
                state,
                plan,
                H100,
                provider,
                journey=self.journey,
                journal=self.journal,
                live=None,
            )
        state = JourneyPath.open(self.journey.file).state
        self.assertIsNone(state.outcome)
        self.assertEqual(state.status, "run_held")
        self.assertEqual(state.failures[-1].invariant_id, "h100_request_invalid")
        self.assertEqual(state.failures[-1].step, "preflight")
        self.assertFalse(provider.jobs_by_name)
        self.assertIsNone(state.failure_review)

    def test_cpu_request_compiler_rejection_is_retained_as_our_invariant(self):
        from bio_neb.parabricks.reference_stages import CPU_PREPARATION

        provider = FakeProvider()
        state = self.run_engine(provider)
        plan = replace(
            self.plan,
            layout=self.plan.layout.model_copy(update={"prepared_object_keys": ()}),
        )
        with self.assertRaisesRegex(Invariant, "cpu_request_invalid"):
            run_stage(
                state,
                plan,
                CPU_PREPARATION,
                provider,
                journey=self.journey,
                journal=self.journal,
                live=None,
            )
        state = JourneyPath.open(self.journey.file).state
        self.assertIsNone(state.outcome)
        self.assertEqual(state.status, "run_held")
        self.assertEqual(state.failures[-1].invariant_id, "cpu_request_invalid")
        self.assertEqual(state.failures[-1].step, "verify_source")
        self.assertFalse(provider.jobs_by_name)
        self.assertIsNone(state.failure_review)

    def test_rejected_produced_fact_is_a_retained_invariant(self):
        def invalid_fact(receipts):
            valid = self.input_fact(receipts).model_dump(mode="json", by_alias=True)
            valid["rests_on"] = ["unknown-prerequisite"]
            return PathFact.model_validate(valid)

        with self.assertRaisesRegex(Invariant, "stage_path_fact_invalid"):
            self.run_engine(FakeProvider(), replace(self.stage, path_fact=invalid_fact))
        state = JourneyPath.open(self.journey.file).state
        self.assertIsNone(state.outcome)
        self.assertEqual(state.status, "run_held")
        self.assertNotIn("InputsVerified", state.latest_event_ids)
        failure = state.failures[-1]
        self.assertEqual(failure.invariant_id, "stage_path_fact_invalid")
        self.assertEqual(failure.step, "invalid_fact")
        self.assertEqual(failure.context["original_failure_class"], "invariant")
        self.assertEqual(failure.context["route"], "hold")
        self.assertEqual(failure.context["proof_ref"], failure.evidence[0].proof_ref)

    def test_path_fact_invariant_is_recorded_and_reraised(self):
        def bad_fact(receipts):
            raise Invariant("bad_fact", "closure cannot be verified")

        with self.assertRaises(Invariant):
            self.run_engine(FakeProvider(), replace(self.stage, path_fact=bad_fact))
        failure = project_execution(self.journey.read()).failures[0]
        self.assertEqual(failure.invariant_id, "bad_fact")
        self.assertEqual(failure.module, bad_fact.__module__)
        self.assertEqual(failure.step, "bad_fact")
        failed_rows = [row for row in self.rows() if row["phase"] == "failed"]
        self.assertEqual([row["step"] for row in failed_rows], ["path_fact:bad_fact"])
        self.assertEqual(failed_rows[0]["attempt"], 1)
