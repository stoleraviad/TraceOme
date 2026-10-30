"""Score one planning-layer eval run from its retained Journey, contact-free.

The scores assist the worker's review of the recorded evidence (ADR 0042); they are not the
run's result. Each check reads Journey events, and identifier format and turn faults come from
the product's own rules (`requirement_identifier_rule`, the fault ledger
`model_output_faults`), never a copied regex or count. It reads the Journey's raw JSON lines,
never the product's checked reader, so it scores a Journey of any shape (#355 `reopen`). Run

    python -m evals.score --case <case_id> --journey-file <journey.jsonl> --output <score.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from bio_neb.agentic_journey.action_policy import ProposedAgentAction, call_identity
from bio_neb.agentic_journey.continuation_policy import (
    OUTPUT_ALONE_KEY,
    TRUNCATION_KEY,
    UNOFFERED_KEY,
    model_output_faults,
)
from bio_neb.agentic_journey.model_turn import read_arguments
from bio_neb.investigation.investigation_request import RSID_RULE
from bio_neb.parabricks.execution_planning_journey import (
    ORGANISM_RECORD_EVENTS,
    quoted_in_record,
)
from bio_neb.parabricks.proposal_information import requirement_identifier_rule
from bio_neb.parabricks.run_method import GERMLINE_PLOIDY
from bio_neb.reference_discovery.reference_identity_tool import ASSEMBLY_ACCESSION_RULE
from evals.case import COMPILED, CompiledCase, Question, load_compiled

Event = Mapping[str, Any]
Finding = tuple[str, Any]  # (verdict, detail); verdict is pass, fail or not_reached

_FACT_SLOTS = ("sample_identity", "taxon_identity", "reference_assembly_identity")
# Events whose payload a Tool, a source or the researcher supplied: an identifier first seen
# in one of them is not the model's invention.
_SOURCE_EVENTS = frozenset(
    {
        "ResearchRequestRegistered",
        "RunInputsRegistered",
        "InputMetadataInspected",
        "TaxonResolutionRecorded",
        "ReferenceAssembliesRecorded",
        "ReferenceDescriptionRecorded",
        "CompanionObservationRecorded",
        "PrebuiltIndexesListed",
        "PrebuiltIndexRead",
        "AssemblyWithIndexSuggested",
        "RecipeObservationRecorded",
        "ResourceObservationRecorded",
        "PlanningHitlResponded",
    }
)
# Where an identifier can first appear in planning: typed by the model or returned by a source.
_MENTIONS = _SOURCE_EVENTS | {"ToolCallValidated", "ProposalRequirementsIdentified"}
_REVIEW_REQUESTS = frozenset(
    {
        "PlanningHitlRequested",
        "AgentWorkReviewRequested",
        "AgentEffortReviewRequested",
        "AgentFailureReviewRequested",
        "ModelAvailabilityReviewRequested",
        "InputInspectionReviewRequested",
        "FailureReviewRequested",
    }
)


# Turn faults that refuse part of a reply before anything of it runs (ADR 0044): a call the
# stage does not offer, an output sent beside calls, a cut-off reply. A copy or a resend is
# a diagnostic only.
_REFUSING_FAULTS = frozenset({UNOFFERED_KEY, OUTPUT_ALONE_KEY, TRUNCATION_KEY})


class ScorerError(ValueError):
    """The recorded evidence cannot be scored as the scorer reads it."""


def turn_faults(events: Sequence[Event]) -> list[dict]:
    """Every Attempt's model-output faults in Journey order, as the product's ledger counts
    them: refused calls, text endings, cut-off replies and, from main's routing on, each
    routed reply's faults (`AgentTurnRouted`)."""
    events = list(events)
    index = {e.get("event_id"): i for i, e in enumerate(events)}
    attempts = dict.fromkeys(
        e["payload"]["attempt_id"]
        for e in events
        if e["event_type"] == "AttemptStarted"
    )
    found: dict[tuple[str, str], dict] = {}
    for attempt in attempts:
        try:
            faults = model_output_faults(events, attempt_id=attempt)
        except (KeyError, ValueError) as error:
            raise ScorerError(f"fault ledger of Attempt {attempt}: {error}") from error
        # A continuation Attempt's ledger repeats its predecessors' faults: keep each once.
        for fault in faults:
            source = events[index[fault.event_id]]
            found.setdefault(
                (fault.event_id, fault.key),
                {
                    "attempt": source["payload"]["attempt_id"],
                    "key": fault.key,
                    "count": fault.count,
                    "event_id": fault.event_id,
                    "source": source["event_type"],
                },
            )
    return sorted(found.values(), key=lambda fault: index[fault["event_id"]])


def replies(events: Sequence[Event]) -> list[dict]:
    """One row per model reply (post-run examination row 6): output tokens against the
    ceiling its request actually sent, calls, distinct calls, and how routing reshaped it."""
    events = list(events)
    requests = {
        (e["payload"]["attempt_id"], e["payload"]["request_number"]): e
        for e in events
        if e["event_type"] == "ModelRequestRecorded"
    }
    # The ceiling each request's last send carried; the context budget lowers it.
    sent = {
        e["payload"].get("logical_request_event_id"): e["payload"].get(
            "final_max_completion_tokens"
        )
        for e in events
        if e["event_type"] == "ModelTransportRequestRecorded"
    }
    routed = {
        e["payload"]["response_event_id"]: e["payload"]
        for e in events
        if e["event_type"] == "AgentTurnRouted"
    }
    cut = {
        e["payload"]["response_event_id"]
        for e in events
        if e["event_type"] == "ModelCompletionTruncated"
    }
    rows = []
    for event in events:
        if event["event_type"] != "ModelResponseRecorded":
            continue
        payload = event["payload"]
        message = payload["compatible_adapter_evidence"]["response"]["message"]
        # The same call as the router reads it: Tool and canonical arguments,
        # whether sent as an object, JSON text or fenced JSON text.
        calls = [
            call_identity(
                ProposedAgentAction(
                    tool_name=part.get("tool_name"),
                    tool_call_id=part.get("tool_call_id") or "",
                    arguments=read_arguments(part.get("args")).value,
                )
            )
            for part in message.get("parts", ())
            if part.get("part_kind") == "tool-call"
        ]
        request = requests.get((payload["attempt_id"], payload["request_number"]))
        allowance = request and (
            sent.get(request.get("event_id"))
            or request["payload"]["compatible_adapter_evidence"]["request"][
                "model_settings"
            ].get("max_tokens")
        )
        route = routed.get(event.get("event_id"), {})
        rows.append(
            {
                "response_event_id": event.get("event_id"),
                "attempt": payload["attempt_id"],
                "request_number": payload["request_number"],
                "output_tokens": (message.get("usage") or {}).get("output_tokens"),
                "allowance": allowance,
                "finish_reason": message.get("finish_reason"),
                "cut_off": event.get("event_id") in cut,
                "calls": len(calls),
                "distinct_calls": len(set(calls)),
                "copies": len(route.get("copies", {})),
                "unoffered": len(route.get("unoffered_validation_event_ids", ())),
                "refused_outputs": len(route.get("refused_output_call_ids", ())),
                "dropped_call": route.get("dropped_call_id"),
                "faults": {f["key"]: f["count"] for f in route.get("faults", ())},
            }
        )
    return rows


class PlanningRun:
    """The planning-layer facts of one Journey, read once."""

    def __init__(self, events: Sequence[Event]):
        self.events = list(events)
        self.request = next(
            (e["payload"]["prompt"] for e in self._of("ResearchRequestRegistered")), ""
        )
        identified = self._of("ProposalRequirementsIdentified")
        self.plan_index = (
            self.events.index(identified[0]) if identified else len(self.events)
        )
        self.requirements: list[dict] = (
            identified[-1]["payload"]["plan"]["requirements"] if identified else []
        )
        self.turn_faults = turn_faults(self.events)

    def _of(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e["event_type"] == event_type]

    def results(self, event_type: str) -> list[dict]:
        return [e["payload"]["result"] for e in self._of(event_type)]

    def requirement(self, kind: str) -> dict | None:
        return next((r for r in self.requirements if r["kind"] == kind), None)

    def refused_turns(self) -> list[dict]:
        return [
            e["payload"]
            for e in self._of("AgentActionTurnDispositionRecorded")
            if e["payload"]["disposition"].endswith("_refused")
        ]

    def refused_before_plan(self) -> list[str]:
        return [
            (e["payload"].get("correction_key") or e["payload"]["disposition"])[:160]
            for e in self.events[: self.plan_index]
            if e["event_type"] == "AgentActionTurnDispositionRecorded"
            and e["payload"]["disposition"].endswith("_refused")
        ]

    def refusing_faults_before_plan(self) -> list[str]:
        """Routed or cut-off replies refused in part before the plan (owner, #355 `faults`)."""
        before = {e.get("event_id") for e in self.events[: self.plan_index]}
        return [
            fault["key"]
            for fault in self.turn_faults
            if fault["key"] in _REFUSING_FAULTS and fault["event_id"] in before
        ]

    def planning_reviews(self) -> list[str]:
        """Each planning Review by its requirement's kind; the subject key is the model's.

        A Review without its requirement, or naming one outside the accepted plan, is a
        scorer error (it fails the campaign, never the model).
        """
        kinds = {r["requirement_id"]: r["kind"] for r in self.requirements}
        reviews = []
        for event in self._of("PlanningHitlRequested"):
            requirement = event["payload"].get("requirement_id")
            if requirement not in kinds:
                raise ScorerError(
                    f"planning Review {event['event_id']} names requirement "
                    f"{requirement!r}, not one of the accepted plan's"
                )
            reviews.append(kinds[requirement])
        return reviews

    def terminal(self) -> str:
        if self._of("ExecutionHitlResponded"):
            return "execution_answered"
        if self._of("ExecutionProposalSubmitted"):
            return "execution_review"
        answered = {
            e["payload"].get("request_event_id")
            for e in self.events
            if e["event_type"] not in _REVIEW_REQUESTS
        }
        pending = [
            e
            for e in self.events
            if e["event_type"] in _REVIEW_REQUESTS and e["event_id"] not in answered
        ]
        closed = [
            i for i, e in enumerate(self.events) if e["event_type"] == "PassClosed"
        ]
        held = closed and self.events[closed[-1]]["payload"]
        # A Run the product held (an interrupt or a typed failure) reads by its class, unless
        # a Review was requested after the hold.
        if (
            held
            and held["disposition"] == "held"
            and not any(self.events.index(e) > closed[-1] for e in pending)
        ):
            return f"run_held:{(held.get('failure') or {}).get('failure_class')}"
        if pending:
            last = pending[-1]
            if last["event_type"] == "PlanningHitlRequested":
                return f"planning_review:{last['payload']['subject_key']}"
            return last["event_type"]
        failed = [
            e["event_type"]
            for e in self.events
            if e["event_type"] in {"AttemptFailed", "RunFailed"}
        ]
        return failed[-1] if failed else "unknown"


def _quote(requirement: dict) -> str | None:
    return next(
        (
            item["exact_text"]
            for item in requirement["evidence"]
            if item["source_kind"] == "research_request"
        ),
        None,
    )


def _plan_needed(check: Callable[[PlanningRun, dict], Finding]):
    def guarded(run: PlanningRun, spec: dict) -> Finding:
        if not run.requirements:
            return "not_reached", "no accepted plan"
        return check(run, spec)

    return guarded


@_plan_needed
def _kinds_present(run: PlanningRun, spec: dict) -> Finding:
    missing = sorted(set(spec["kinds"]) - {r["kind"] for r in run.requirements})
    return ("fail", {"missing": missing}) if missing else ("pass", None)


@_plan_needed
def _knowledge_states(run: PlanningRun, spec: dict) -> Finding:
    observed, wrong = {}, []
    for state, kinds in spec.items():
        for kind in kinds:
            requirement = run.requirement(kind)
            observed[kind] = requirement and requirement["knowledge_state"]
            if observed[kind] != state:
                wrong.append(kind)
    return ("fail" if wrong else "pass"), {"observed": observed, "wrong": wrong}


def _first_attempt(run: PlanningRun, spec: dict) -> Finding:
    # A refusal fails the first attempt even when no plan follows.
    refused = run.refused_before_plan()
    faults = run.refusing_faults_before_plan()
    if not run.requirements and not (refused or faults):
        return "not_reached", "no accepted plan"
    return ("fail" if refused or faults else "pass"), {
        "refused_before_plan": refused,
        "turn_faults_before_plan": faults,
    }


def _no_invented_identifier(run: PlanningRun, spec: dict) -> Finding:
    def found(text: str) -> set[str]:
        # The product's own readers: versioned accessions, and rsIDs in any spelling.
        return {*ASSEMBLY_ACCESSION_RULE.stated(text), *RSID_RULE.stated(text)}

    seen, invented = found(run.request), []
    for event in run.events:
        if event["event_type"] in _SOURCE_EVENTS:
            seen |= found(json.dumps(event["payload"]))
        elif event["event_type"] == "ToolCallValidated":
            arguments = json.dumps(event["payload"]["arguments"])
            invented += [
                {"tool": event["payload"]["tool_name"], "value": value}
                for value in sorted(found(arguments) - seen)
            ]
    return ("fail" if invented else "pass"), {"invented": invented}


def _terminal(run: PlanningRun, spec: dict) -> Finding:
    observed = run.terminal()
    return ("pass" if observed == spec["terminal"] else "fail"), {"observed": observed}


def _reviews_raised(run: PlanningRun, spec: dict) -> Finding:
    raised = Counter(run.planning_reviews())
    verdict = "pass" if set(raised) == set(spec["reviews"]) else "fail"
    return verdict, {"raised": dict(raised)}


@_plan_needed
def _plan_value(run: PlanningRun, spec: dict) -> Finding:
    requirement = run.requirement(spec["kind"])
    observed = requirement and {
        "state": requirement["knowledge_state"],
        "value": requirement["value"],
        "quote": _quote(requirement),
    }
    expected = observed and (observed["state"], str(observed["value"])) == (
        spec["state"],
        spec["value"],
    )
    return ("pass" if expected else "fail"), {"observed": observed}


def _taxon_resolved(run: PlanningRun, spec: dict) -> Finding:
    results = run.results("TaxonResolutionRecorded")
    if not results:
        return "not_reached", "no taxon resolution"
    resolved = [
        c["ncbi_tax_id"]
        for result in results
        if result.get("resolution") == "resolved"
        for c in result["candidates"]
    ]
    return ("pass" if spec["ncbi_tax_id"] in resolved else "fail"), {
        "resolved": resolved
    }


def _reference_verified(run: PlanningRun, spec: dict) -> Finding:
    results = run.results("ReferenceDescriptionRecorded")
    if not results:
        return "not_reached", "no reference description"
    # A resolved description verifies its record, whatever status NCBI copied onto
    # it.
    verified = [
        r["assembly_accession"]
        for result in results
        if result.get("resolution") == "resolved"
        for r in result["references"]
    ]
    return ("pass" if spec["accession"] in verified else "fail"), {"verified": verified}


def _chosen_accession(choice: dict) -> str | None:
    """The accession a choice names: a discovered row's, or a supplied one (#471)."""
    return choice["assembly_accession"]


def _reference_from_discovery(run: PlanningRun, spec: dict) -> Finding:
    accession = spec["accession"]
    # The first model-typed or source-returned mention: a model call or plan, or a result.
    first = next(
        (
            e["event_type"]
            for e in run.events
            if e["event_type"] in _MENTIONS
            and accession in ASSEMBLY_ACCESSION_RULE.stated(json.dumps(e["payload"]))
        ),
        None,
    )
    if first is None:
        return "not_reached", "accession never mentioned"
    chosen = [
        e["payload"]
        for e in run._of("ReferenceChoiceRecorded")
        if _chosen_accession(e["payload"]) == accession
    ]
    from_discovery = bool(chosen) and all(
        c["discovery_event_id"] and c["assertion_event_id"] is None for c in chosen
    )
    verdict = first == "ReferenceAssembliesRecorded" and from_discovery
    return ("pass" if verdict else "fail"), {
        "first_seen_in": first,
        "choices": len(chosen),
    }


@_plan_needed
def _investigation_items(run: PlanningRun, spec: dict) -> Finding:
    items = [
        {"value": r["value"], "quote": _quote(r), "purpose": r["investigation_purpose"]}
        for r in run.requirements
        if r["kind"] == "scientific_investigation"
    ]
    matching = [i for i in items if i["purpose"] == spec["purpose"]]
    if "values" in spec:
        values = [i["value"] for i in matching]
        verdict = sorted(values) == sorted(spec["values"])
    else:

        def sites(item):
            text = " ".join((item["quote"] or "").split())
            return [site for site in spec["quotes"] if site in text]

        per_item = [sites(i) for i in matching]
        verdict = (
            sorted(site for found in per_item for site in found)
            == sorted(spec["quotes"])
            and all(len(found) <= 1 for found in per_item)
            # No rsID typed for a positional site; a prose item's value is its quote.
            and not any(RSID_RULE.stated(i["value"] or "") for i in matching)
        )
    return ("pass" if verdict else "fail"), {"items": items}


def _ploidy_notice(run: PlanningRun, spec: dict) -> Finding:
    """The Execution Review states diploid calling beside the model's judgment of the
    organism, citing one of the Run's organism records by words that record holds, and
    asks exactly when the judgment is not diploid (#470; owner Q7 a). The scorer reads
    raw events, so it checks the notice's typed facts itself, with the product's own
    quote-membership rule; it never judges the organism."""
    proposals = run._of("ExecutionProposalSubmitted")
    if not proposals:
        return "not_reached", "no Execution Review"
    ploidy = proposals[-1]["payload"]["proposal"].get("ploidy")
    if ploidy is None:
        return "fail", {"ploidy": None}
    organism = ploidy["organism"]
    cited = next(
        (e for e in run.events if e["event_id"] == organism["record_event_id"]),
        None,
    )
    question = ploidy.get("question")
    observed = {
        "calling_ploidy": ploidy["calling_ploidy"],
        "states": bool(ploidy["statement"]),
        "judgment": organism["judgment"],
        "quote": organism["quote"],
        "cites": cited and cited["event_type"],
        "quoted": cited is not None
        and cited["event_type"] in ORGANISM_RECORD_EVENTS
        and quoted_in_record(cited, organism["quote"]),
        "asks": isinstance(question, str) and bool(question),
    }
    verdict = (
        observed["calling_ploidy"] == GERMLINE_PLOIDY
        and observed["states"]
        and observed["cites"] in ORGANISM_RECORD_EVENTS
        and observed["quoted"]
        and observed["asks"] == (observed["judgment"] != "diploid")
        and observed["asks"] == spec["asks"]
    )
    return ("pass" if verdict else "fail"), observed


CHECKS: dict[str, Callable[[PlanningRun, dict], Finding]] = {
    "kinds_present": _kinds_present,
    "knowledge_states": _knowledge_states,
    "first_attempt": _first_attempt,
    "no_invented_identifier": _no_invented_identifier,
    "terminal": _terminal,
    "reviews_raised": _reviews_raised,
    "plan_value": _plan_value,
    "taxon_resolved": _taxon_resolved,
    "reference_verified": _reference_verified,
    "reference_from_discovery": _reference_from_discovery,
    "investigation_items": _investigation_items,
    "ploidy_notice": _ploidy_notice,
}


def _usable(run: PlanningRun) -> Finding:
    """Each asserted fact-slot value is one identifier in its own format, and the planning
    layer's consumers verified the sample, taxon, reference and companion; a consumer
    whose stage never ran is unreached, not failed."""
    if not run.requirements:
        return "not_reached", "no accepted plan"
    atomic = {}
    for kind in _FACT_SLOTS:
        requirement = run.requirement(kind)
        if requirement and requirement["knowledge_state"] == "asserted":
            value = str(requirement["value"])
            atomic[kind] = requirement_identifier_rule(kind).stated(value) == (value,)
    consumers = {
        "sample": "SampleAttestationRecorded",
        "taxon": "TaxonResolutionRecorded",
        "reference": "ReferenceDescriptionRecorded",
        "companion": "CompanionObservationRecorded",
    }
    consumed = {
        "sample": any(
            r["status"] == "verified" for r in run.results("SampleAttestationRecorded")
        ),
        "taxon": any(
            result.get("resolution") == "resolved"
            for result in run.results("TaxonResolutionRecorded")
        ),
        "reference": any(
            result.get("resolution") == "resolved"
            for result in run.results("ReferenceDescriptionRecorded")
        ),
        # A failed lookup (status "failed") carries no outcome.
        "companion": any(
            r["status"] == "observed" and r["outcome"] == "verified"
            for r in run.results("CompanionObservationRecorded")
        ),
    }
    # A consumer whose stage never ran leaves its value unconsumed, not refused: the
    # Run held before it.
    unreached = [name for name, kind in consumers.items() if not run._of(kind)]
    detail = {"atomic": atomic, "consumed": consumed, "unreached": unreached}
    reached = [consumed[name] for name in consumers if name not in unreached]
    if not all(atomic.values()) or not all(reached):
        return "fail", detail
    return ("not_reached" if unreached else "pass"), detail


def _diagnostics(run: PlanningRun) -> dict[str, Any]:
    responses = run._of("ModelResponseRecorded")
    usage = [
        e["payload"]["compatible_adapter_evidence"]["response"]["message"]["usage"]
        for e in responses
    ]
    models = {
        e["payload"]["compatible_adapter_evidence"]["model_name"]
        for e in run._of("ModelRequestRecorded")
    }
    finish = Counter(
        e["payload"]["compatible_adapter_evidence"]["response"]["message"][
            "finish_reason"
        ]
        for e in responses
    )
    source_stages = Counter(
        e["payload"]["stage"]
        for e in run._of("activity_succeeded")
        if e["payload"]["adapter"] == "ncbi_datasets"
    )
    return {
        "models": sorted(models),
        "attempts": len(run._of("AttemptStarted")),
        "model_requests_logical": len(run._of("ModelRequestRecorded")),
        "model_requests_physical": len(run._of("ModelTransportRequestRecorded")),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "finish_reasons": dict(finish),
        "truncations": len(run._of("ModelCompletionTruncated")),
        "refused_turns": [
            (p.get("correction_key") or p["disposition"])[:160]
            for p in run.refused_turns()
        ],
        "turn_faults": run.turn_faults,
        "replies": replies(run.events),
        "planning_reviews": run.planning_reviews(),
        # Planning source activities; physical GETs are in the contact journals
        # (evals.watch.source_gets).
        "source_activities": dict(source_stages),
        "terminal": run.terminal(),
    }


def score_planning(case: CompiledCase, events: Iterable[Event]) -> dict[str, Any]:
    """The planning layer's product gates, typed-question findings and diagnostics."""
    run = PlanningRun(list(events))
    accepted = ("pass", None) if run.requirements else ("fail", "no accepted plan")
    gates = {"accepted": accepted, "usable": _usable(run)}
    questions = [_question(run, q) for q in case.planning.questions]
    return {
        # /2: turn faults, reply rows, `run_held:<class>` terminals, the case digest.
        "schema_version": "bio-neb.eval-score/2",
        "case_id": case.case_id,
        "layer": "planning",
        "gates": {name: {"verdict": v, "detail": d} for name, (v, d) in gates.items()},
        "questions": questions,
        "diagnostics": _diagnostics(run),
    }


def _question(run: PlanningRun, question: Question) -> dict[str, Any]:
    verdict, detail = CHECKS[question.check](run, question.spec)
    return {
        "id": question.id,
        "check": question.check,
        "gate": question.gate,
        "verdict": verdict,
        "detail": detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.score")
    parser.add_argument("--case", required=True)
    parser.add_argument("--journey-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    content = args.journey_file.read_bytes()
    events = [json.loads(line) for line in content.splitlines() if line.strip()]
    score = score_planning(load_compiled(args.case), events)
    # The compiled case scored against: a later gold refresh keeps old scores readable.
    compiled = (COMPILED / "planning" / f"{args.case}.json").read_bytes()
    score["case"] = {"sha256": hashlib.sha256(compiled).hexdigest()}
    score["journey"] = {
        "path": str(args.journey_file),
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "events": len(events),
        "head_event_id": events[-1]["event_id"] if events else None,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(score, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
