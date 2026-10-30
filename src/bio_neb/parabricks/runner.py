"""Parabricks internal stage runner using shared policy and observation.

Parabricks owns stage attempts, runtime alternatives and scoped routing. A port wrapper journals contacts before
calling effects and retains their receipts immediately, even when later code in
the same step fails. Steps own recipes; path_fact owns their verified conclusion.
Only ProviderError is routed to its Review. An Invariant is retained, then
escapes naming its StageFailed so the failure module holds the Run; arbitrary
programming faults propagate with the contact journal prefix left intact.
"""

from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any, Literal

from bio_neb.content import canonical_json_digest
from bio_neb.failure import (
    Ask,
    FailurePolicy,
    FailureSurface,
    Invariant,
    ProviderError,
    ResolvedFailurePolicy,
    operation_failure_policy,
    operation_retry_decision,
    policy_for,
    retry_consequence,
)
from bio_neb.journey import canonical_json
from bio_neb.journey_path import (
    Artifact,
    ExecutionState,
    Grounding,
    JourneyPath,
    Outcome,
    PathFact,
    PathKind,
    Pending,
    Producer,
    StageFailure,
)
from bio_neb.observation import LiveEmitter, ObservationPolicy, observe
from bio_neb.parabricks.execution_contact_accounting import ExecutionContactJournal
from bio_neb.parabricks.plan import ExecutionPlan
from bio_neb.providers.ports import PortName, Provider, Receipt, TransferSource


@dataclass(frozen=True)
class StepContext:
    plan: ExecutionPlan
    state: ExecutionState
    journal_row_writer: Callable[..., str]
    live: LiveEmitter
    failure_policy: ResolvedFailurePolicy
    receipts: tuple[Receipt, ...]
    closure: Callable[[tuple[Receipt, ...]], Receipt | None] | None
    step: Callable
    stage_name: str

    @property
    def policy(self) -> FailurePolicy:
        return self.failure_policy.effective

    def observe(
        self, source: Callable[[], Receipt], *, max_wait: float, window: int = 1
    ) -> Receipt:
        """Use this stage's closure with the plan's shared polling cadence."""
        if self.closure is None:
            raise Invariant("missing_closure", "observation stage has no closure")
        return observe(
            source,
            ObservationPolicy(
                interval=self.plan.policies.observation_interval_seconds,
                max_wait=max_wait,
                window=window,
                closure=self.closure,
                liveness=lambda sample: self.live(
                    {
                        "kind": "execution_observation",
                        "stage": self.stage_name,
                        "step": self.step.__name__,
                        "state": sample.values.get("state"),
                        "provider_ids": list(sample.provider_ids),
                    }
                ),
            ),
        )


@dataclass(frozen=True)
class StageDef:
    name: str
    steps: tuple[Callable[[StepContext, Any], Receipt], ...]
    path_fact: Callable[[tuple[Receipt, ...]], PathFact]
    ports: Mapping[str, PortName]
    closure: Callable[[tuple[Receipt, ...]], Receipt | None] | None = None
    artifacts: Callable[[tuple[Receipt, ...]], tuple[Artifact, ...]] | None = None
    runtime: Literal["h100"] | None = None

    def __post_init__(self):
        names = tuple(step.__name__ for step in self.steps)
        if (
            not self.name
            or self.runtime not in {None, "h100"}
            or not names
            or len(set(names)) != len(names)
            or set(self.ports) != set(names)
            or not set(self.ports.values()) <= {"store", "jobs", "identity"}
        ):
            raise ValueError(
                "stage requires uniquely named plain steps and exact port bindings"
            )


_METHODS = {
    "store": {"fresh_bucket", "put", "copy", "head", "read_small"},
    "jobs": {"submit", "observe"},
    "identity": {"whoami"},
}


class _ContactPort:
    """Expose just the selected port and retain each result at contact time.

    Arguments and returned bodies are deliberately absent from journal rows.
    Concrete providers must classify errors and return secret-free identifiers
    and evidence. The original receipt content remains available to the recipe.
    """

    def __init__(
        self,
        port: Any,
        name: PortName,
        ctx: StepContext,
        retain_receipt: Callable,
        retained_transfers: Callable,
    ):
        self._port, self._name, self._ctx = port, name, ctx
        self._retain_receipt = retain_receipt
        self._retained_transfers = retained_transfers

    def __getattr__(self, name: str):
        if name not in _METHODS[self._name]:
            raise AttributeError(name)
        effect = getattr(self._port, name)

        def contact(*args, **kwargs):
            writer = self._ctx.journal_row_writer
            transfer_intent = None
            content = args[2] if len(args) > 2 else kwargs.get("content")
            if (
                self._name == "store"
                and name == "put"
                and isinstance(content, TransferSource)
            ):
                transfer_intent = canonical_json_digest(
                    {
                        "execution_id": self._ctx.plan.identity.execution_id,
                        "authority_id": self._ctx.plan.identity.authority_id,
                        "bucket": args[0] if args else kwargs["bucket"],
                        "key": args[1] if len(args) > 1 else kwargs["key"],
                        "source": asdict(content),
                    }
                )
                for body, proof, retained in self._retained_transfers():
                    if body.get("transfer_intent") == transfer_intent:
                        return Receipt(
                            values=body["values"],
                            provider_ids=tuple(body["provider_ids"]),
                            proof_ref=proof,
                            contact=retained,
                        )
            intent = None
            if self._name == "jobs" and name == "submit":
                intent = self._retain_receipt(
                    kind="job_submission",
                    body={
                        "phase": "requested",
                        "name": args[0],
                        "project_id": self._ctx.plan.scope.project_id,
                    },
                )
            resource_kind = None
            resource_intent = None
            if self._name == "store" and name == "fresh_bucket":
                resource_kind, resource_intent = "bucket", args[0]
            elif self._name == "jobs" and name == "submit":
                resource_intent = args[0]
                resource_kind = (
                    "cpu_job"
                    if resource_intent == self._ctx.plan.layout.cpu_job_name
                    else "h100_job"
                    if resource_intent == self._ctx.plan.layout.h100_job_name
                    else None
                )
            identity = dict(
                logical_operation=f"{self._name}.{name}",
                resource_kind=resource_kind,
                resource_intent=resource_intent,
            )
            invoked = writer(phase="invoked", receipt=intent, **identity)
            started = time.monotonic()
            try:
                receipt = effect(*args, **kwargs)
            except ProviderError as error:
                job_id = error.evidence.get("provider_job_id")
                writer(
                    phase="failed",
                    invocation_ref=invoked,
                    **identity,
                    provider_ids=(job_id,) if isinstance(job_id, str) else (),
                    failure_class=error.failure_class,
                    failure_detail=error.detail,
                    duration=time.monotonic() - started,
                    receipt=intent,
                )
                raise
            if not isinstance(receipt, Receipt):
                raise Invariant("invalid_receipt", "provider did not return a Receipt")
            contact_receipt = receipt.contact
            if self._name == "store" and name in {"put", "copy", "head", "read_small"}:
                contact_receipt = self._retain_receipt(
                    kind="transfer_proof",
                    body={
                        "operation": name,
                        "transfer_intent": transfer_intent,
                        "values": receipt.values,
                        "provider_ids": receipt.provider_ids,
                        "provider_receipt": None
                        if receipt.contact is None
                        else receipt.contact.model_dump(mode="json"),
                    },
                )
            if self._name == "jobs" and name == "submit":
                contact_receipt = self._retain_receipt(
                    kind="job_submission",
                    body={
                        "values": receipt.values,
                        "provider_ids": receipt.provider_ids,
                    },
                )
            proof = writer(
                phase="verified",
                invocation_ref=invoked,
                **identity,
                receipt=contact_receipt,
                provider_ids=receipt.provider_ids,
                ceiling_count=receipt.resource_count,
                duration=time.monotonic() - started,
            )
            if (name in {"fresh_bucket", "submit"} and receipt.resource_count != 1) or (
                receipt.resource_count is not None and receipt.resource_count > 1
            ):
                raise Invariant(
                    "named_resource_ceiling",
                    "named mutation must attest exactly one resource",
                )
            return receipt.model_copy(
                update={"proof_ref": proof, "contact": contact_receipt}
            )

        return contact


def record_outcome(
    state: ExecutionState,
    journey: JourneyPath,
    fact: PathFact,
    *,
    journal: ExecutionContactJournal | None = None,
) -> ExecutionState:
    """Validate first, then close the pass's existing writer and bind final bytes."""
    journey.preview(state, fact)
    if journal is not None:
        if (
            not isinstance(fact.payload, Outcome)
            or fact.payload.contact_journal != str(journal.path)
            or journal.execution_id != state.bound.execution_id
            or journal.execution_pass != state.pass_
        ):
            raise ValueError("outcome requires its own pass journal")
        closed = journal.finalize()
        if closed is not None:
            fact = fact.model_copy(
                update={
                    "payload": fact.payload.model_copy(
                        update={
                            "contact_journal": closed.locator,
                            "journal_sha256": closed.sha256,
                            "journal_bytes": closed.byte_count,
                        }
                    )
                }
            )
    return journey.path(state, fact)[1]


def _route_failure(
    state: ExecutionState,
    stage: StageDef,
    ctx: StepContext,
    error: ProviderError | Invariant,
    failure_class: str,
    proof: str,
    journey: JourneyPath,
    journal: ExecutionContactJournal,
) -> ExecutionState:
    """Retain failure evidence, then take the policy's route.

    Ask returns the state stopped at its Review. Hold (our bug) raises an
    ``Invariant`` naming its StageFailed; the failure module holds the Run. No
    route ends the Execution.
    """
    evidence = error.evidence if isinstance(error, ProviderError) else {}
    resource = {}
    if stage.name in {"cpu_preparation", "h100"}:
        resource["job_name"] = (
            ctx.plan.layout.cpu_job_name
            if stage.name == "cpu_preparation"
            else ctx.plan.layout.h100_job_name
        )
        submission = next(
            (r for r in ctx.receipts if "create" in r.values and r.provider_ids), None
        )
        if submission is not None:
            resource["provider_job_id"] = submission.provider_ids[0]
        elif isinstance(evidence.get("provider_job_id"), str):
            resource["provider_job_id"] = evidence["provider_job_id"]
        if evidence.get("state") in {"FAILED", "CANCELLED", "ERROR"}:
            resource["terminal_state"] = evidence["state"]
    pointer = journey.attach(
        Artifact(
            content=canonical_json(
                {"detail": str(error), "evidence": evidence}
            ).encode(),
            role="stage_failure",
            producer=Producer(stage=stage.name),
            grounding=Grounding.AMBIGUOUS,
            proof_ref=proof,
        )
    )
    failure = StageFailure(
        module=ctx.step.__module__,
        stage=stage.name,
        step=ctx.step.__name__,
        failure_class=failure_class,
        detail=str(error),
        invariant_id=error.invariant_id if isinstance(error, Invariant) else None,
        context={
            **resource,
            "failure_policy": ctx.failure_policy.model_dump(mode="json"),
            "original_failure_class": error.failure_class
            if isinstance(error, ProviderError)
            else "invariant",
            "route": ctx.policy.route(failure_class),
            "proof_ref": proof,
        },
        evidence=(pointer,),
    )
    _, state = journey.path(
        state,
        PathFact(
            event_type="StageFailed",
            kind=PathKind.FAILURE,
            grounding=Grounding.AMBIGUOUS,
            **{"pass": state.pass_},
            payload=failure,
        ),
    )
    dependency = state.latest_event_ids["StageFailed"]
    if ctx.policy.route(failure.failure_class) == "hold":
        from bio_neb.agentic_journey.failure_cause import cause_identity

        raise Invariant(
            error.invariant_id,
            failure.detail,
            cause=error.cause or cause_identity(error),
            evidence={
                **error.evidence,
                "stage_failed_event_id": dependency,
                "stage": stage.name,
                "step": failure.step,
            },
        ) from error
    # A fault read from a job that exists and has not failed (a poll, a state
    # check, a log read) offers retry or abort only: an amendment's new job would
    # pay twice for a job that may still complete.
    read_fault = "provider_job_id" in resource and "terminal_state" not in resource
    amendable = tuple(
        field
        for field in (
            () if read_fault else ("gpu_class",) if stage.runtime == "h100" else ()
        )
        if field in ctx.plan.amendable
    )
    request = Ask(
        stage=stage.name,
        failure_event_id=dependency,
        request_id=str(uuid.uuid4()),
        reason=failure.detail,
        retry_consequence=retry_consequence(failure.context),
        amendment_consequences={field: "new_job_intent" for field in amendable},
        amendable=amendable,
        legal_actions=("retry", "abort", "amend") if amendable else ("retry", "abort"),
    )
    from bio_neb.parabricks.review import request_review

    return request_review(
        state,
        ctx.plan,
        request,
        journey=journey,
        contact_journal=str(journal.path),
        journal=journal,
    )


def _runtime_substitution(state, plan, stage, error, proof):
    if (
        error.failure_class != "rejected"
        or error.detail != "no_capacity"
        or error.evidence.get("resource_absent") is not True
        or not plan.runtimes.fallbacks
    ):
        return None
    from bio_neb.journey_path import RuntimeSubstituted

    selection = state.bound.approved_input["proposal"].get("runtime_selection")
    if selection is None:
        return None
    from bio_neb.parabricks.capability_policy import ExecutionRuntimeSelection

    selected = ExecutionRuntimeSelection.model_validate(selection).select(
        amended_profile_id=state.amendments.get("gpu_class"),
        substituted_profile_id=state.runtime_profile_id,
    )
    if selected.next_profile_id is None:
        return None
    rests_on = [state.latest_event_ids["ExecutionBound"]]
    if "gpu_class" in state.amendment_event_ids:
        rests_on.append(state.amendment_event_ids["gpu_class"])
    if "RuntimeSubstituted" in state.latest_event_ids:
        rests_on.append(state.latest_event_ids["RuntimeSubstituted"])
    return PathFact(
        event_type="RuntimeSubstituted",
        kind="decision",
        grounding="human",
        **{"pass": state.pass_},
        rests_on=tuple(rests_on),
        payload=RuntimeSubstituted(
            from_profile=selected.profile_id,
            to_profile=selected.next_profile_id,
            authority_id=plan.identity.authority_id,
            stage=stage.name,
            proof_ref=proof,
        ),
    )


def run_stage(
    state: ExecutionState,
    plan: ExecutionPlan,
    stage: StageDef,
    provider: Provider,
    *,
    journey: JourneyPath,
    journal: ExecutionContactJournal,
    live: Callable | None,
) -> ExecutionState:
    """Run one stage without re-reading or projecting retained history.

    Retry attempts reuse exactly the same plan and deterministic names. Providers
    own exists-prechecks; attempt counts never increase resource ceilings. A
    successful stage appends only its path_fact. A failure appends StageFailed,
    then asks its Review (a designed stop) or holds the Run for a fix.
    """
    if (
        state.bound is None
        or state.bound.execution_id != plan.identity.execution_id
        or state.bound.authority_id != plan.identity.authority_id
        or state.bound.proposal_id != plan.identity.proposal_id
        or journal.execution_id != plan.identity.execution_id
        or journal.execution_pass != state.pass_
    ):
        raise ValueError("stage requires the matching bound plan and journal pass")
    if (
        state.outcome is not None
        or state.unresolved_failure_ids
        or isinstance(state.review, Pending)
        or isinstance(state.failure_review, Ask)
    ):
        raise ValueError("stage cannot execute beyond a failure or designed stop")
    if (
        state.runtime_profile_id is not None
        or state.amendments
        or state.job_intent_answer_ids
    ):
        from bio_neb.parabricks.plan import rederive_plan

        plan = rederive_plan(plan, state)
    # The Run's recorded failure section tunes this stage.
    resolved_policy = policy_for(FailureSurface(scope="stage", name=stage.name))
    policy = resolved_policy.effective
    emit = LiveEmitter(live)
    receipts: list[Receipt] = []
    for step in stage.steps:
        attempt = 0
        transient_attempt = 0
        while True:
            attempt += 1
            writer = partial(
                journal.step,
                stage=stage.name,
                step=step.__name__,
                attempt=attempt,
                failure_policy=resolved_policy,
            )
            ctx = StepContext(
                plan,
                state,
                writer,
                emit,
                resolved_policy,
                tuple(receipts),
                stage.closure,
                step,
                stage.name,
            )
            writer(phase="claimed")
            started = time.monotonic()
            emit({"stage": stage.name, "step": step.__name__, "attempt": attempt})
            try:
                ports = provider.for_step(stage.name, step.__name__)
                port_name = stage.ports[step.__name__]
                # A part transfer inside the step retries by this policy.
                with operation_failure_policy(resolved_policy):
                    receipt = step(
                        ctx,
                        _ContactPort(
                            getattr(ports, port_name),
                            port_name,
                            ctx,
                            journal.retain_receipt,
                            partial(journal.retained_transfers, journey),
                        ),
                    )
                if not isinstance(receipt, Receipt):
                    raise Invariant(
                        "invalid_step_receipt", "step did not return a Receipt"
                    )
            except ProviderError as error:
                proof = writer(
                    phase="failed",
                    duration=time.monotonic() - started,
                    failure_class=error.failure_class,
                )
                failure_class = error.failure_class
                if (
                    stage.runtime == "h100"
                    and stage.ports[step.__name__] == "jobs"
                    and not any(
                        r.resource_count == 1 and r.provider_ids for r in receipts
                    )
                ):
                    substitution = _runtime_substitution(
                        state, plan, stage, error, proof
                    )
                    if substitution is not None:
                        from bio_neb.parabricks.plan import rederive_plan

                        _, state = journey.path(state, substitution)
                        plan = rederive_plan(plan, state)
                        transient_attempt = 0
                        continue
                if policy.route(failure_class) == "retry":
                    transient_attempt += 1
                    failure_class, ceiling = operation_retry_decision(
                        failure_class, transient_attempt, resolved_policy
                    )
                    if ceiling is not None:
                        time.sleep(random.uniform(0, ceiling))
                        continue
                return _route_failure(
                    state, stage, ctx, error, failure_class, proof, journey, journal
                )
            except Invariant as error:
                proof = writer(
                    phase="failed",
                    duration=time.monotonic() - started,
                    failure_class="invariant",
                )
                # An invariant holds: _route_failure raises, naming its StageFailed.
                return _route_failure(
                    state, stage, ctx, error, "invariant", proof, journey, journal
                )
            proof = writer(phase="verified", duration=time.monotonic() - started)
            receipts.append(
                receipt.model_copy(update={"proof_ref": receipt.proof_ref or proof})
            )
            break
    # The final fact is a distinct callable, not the last successful step.
    # Bind its own attribution and timer before evaluating the conclusion.
    writer = partial(
        journal.step,
        stage=stage.name,
        step=f"path_fact:{stage.path_fact.__name__}",
        attempt=1,
        failure_policy=resolved_policy,
    )
    ctx = replace(ctx, journal_row_writer=writer, step=stage.path_fact)
    started = time.monotonic()
    try:
        try:
            fact = stage.path_fact(tuple(receipts))
            journey.preview(state, fact)
        except ValueError as error:
            raise Invariant(
                "stage_path_fact_invalid", "stage_path_fact_invalid"
            ) from error
        if stage.artifacts is not None:
            for artifact in stage.artifacts(tuple(receipts)):
                journey.attach(artifact)
        return journey.path(state, fact)[1]
    except Invariant as error:
        proof = writer(
            phase="failed",
            duration=time.monotonic() - started,
            failure_class="invariant",
        )
        return _route_failure(
            state, stage, ctx, error, "invariant", proof, journey, journal
        )
