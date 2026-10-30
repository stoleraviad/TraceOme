"""Small synthetic Runs for the GUI tests: a Journey and the `observe` snapshots at its pass end.

Everything here is written for the tests, not copied from a recorded Run. Each value only follows
the shape the GUI reads: `journey.jsonl` event lines for `journey_feed`, and `observe`'s
`bio-neb.run-interaction/1` document (fields the backend leaves out when empty are left out here too).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOURNEY_ID = "00000000-0000-4000-8000-000000000427"
SAMPLE = "NA00001"
# The Journey events that move the GUI's stage forward (the feed's stage starts below), in order.
STAGE_EVENTS = {
    "Plan": "AttemptStarted",
    "Approval": "ExecutionProposalSubmitted",
    "Execution": "ExecutionPathRecorded",
    "Investigation": "InvestigationStarted",
    "Results": "GroundedReportPublished",
}
T0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)


def event(n: int, event_type: str, payload: dict | None = None) -> dict:
    return {
        "event_id": f"event-{n}",
        "event_type": event_type,
        "journey_id": JOURNEY_ID,
        "occurred_at": (T0 + timedelta(minutes=n)).isoformat().replace("+00:00", "Z"),
        "payload": payload or {},
    }


# The payload parts the GUI reads beyond the feed's line; other events carry an empty payload.
PAYLOADS = {
    "ResearchRequestRegistered": {
        "prompt": f"Sample {SAMPLE}, short-read germline WGS. Which variants from the attached list does it carry?"
    },
    "ExecutionPathRecorded": {"event_type": "ExecutionBound", "grounding": "human", "payload": {}},
    "InputMetadataInspected": {"result": {"status": "succeeded", "inputs": [{"role": "read_1", "size_bytes": 1_000_000_000}]}},
    "PlanningHitlRequested": {"question": "Which reference should the plan use?", "issue": None, "candidates": []},
    "PlanningHitlResponded": {"request_event_id": "event-0", "response": {"action": "clarify", "instruction": "GRCh38"}},
}


def run_of(*types: str) -> list[dict]:
    """A Journey of these event types, one a minute; the request's prompt names no rsIDs, as a request that lists
    its targets in an attached file."""
    return [event(n, t, PAYLOADS.get(t)) for n, t in enumerate(types, 1)]


def then(events: list[dict], *types: str) -> list[dict]:
    """The events a command appends after `events`, numbered on from them."""
    return [event(n, t, PAYLOADS.get(t)) for n, t in enumerate(types, len(events) + 1)]


def journey(stage: str, *more: str) -> list[dict]:
    """A Run that reached `stage`, then had the `more` events, and whose last line closes a pass (so `observe` ran
    there)."""
    stages = ["Question", *STAGE_EVENTS]
    reached = stages[1 : stages.index(stage) + 1]
    return run_of("JourneyStarted", "ResearchRequestRegistered", "PassStarted", *(STAGE_EVENTS[s] for s in reached), *more,
                  "PassClosed")  # fmt: skip


# The events that begin each stage of `journey-feed` (`bio-neb.journey-feed/1`), as the backend's feed tells them.
FEED_STAGE_STARTS = {
    "JourneyStarted": "request",
    "AttemptStarted": "planning",
    "ExecutionProposalSubmitted": "execution_review",
    "ExecutionPathRecorded": "execution",
    "InvestigationStarted": "investigation",
    "FinalizationBasisCompiled": "finalization",
    "GroundedReportPublished": "report",
}
FEED_LEVELS = {"PassStarted": "quiet", "PassClosed": "quiet", "ModelRequestRecorded": "noise"}
FEED_STATUS = {"PlanningHitlRequested": "question", "PlanningHitlResponded": "answer", "ExecutionProposalSubmitted": "question",
               "ExecutionHitlResponded": "answer", "InvestigationReviewRequested": "question", "AttemptFailed": "failed"}  # fmt: skip


def told(event_type: str) -> str:
    """The synthetic feed's line for an event: distinct from anything the GUI's own narrator says."""
    return f"feed says {event_type}"


def feed(events: list[dict]) -> dict:
    """The `journey-feed --all` document for `events`: one item per line, in Journey order."""
    items, stage, pass_number = [], "request", 0
    for n, e in enumerate(events, 1):
        pass_number += e["event_type"] == "PassStarted"
        stage = FEED_STAGE_STARTS.get(e["event_type"], stage)
        items.append({
            "sequence": n, "event_id": e["event_id"], "event_type": e["event_type"], "occurred_at": e["occurred_at"],
            "stage": stage, "level": FEED_LEVELS.get(e["event_type"], "key"), "status": FEED_STATUS.get(e["event_type"], "done"),
            "text": told(e["event_type"]), "pass_number": pass_number, "payload_bytes": len(str(e["payload"])),
        })  # fmt: skip
    return {"schema_version": "bio-neb.journey-feed/1", "journey_file": "<run>/journey.jsonl", "journey_id": JOURNEY_ID,
            "event_count": len(events), "head_event_id": events[-1]["event_id"] if events else None, "items": items}  # fmt: skip


def snapshot(status: str, frontier: str, *, reviews: tuple[dict, ...] = (), **fields) -> dict:
    """An `observe` document; the first review is the backend's presentation focus (`review`)."""
    snap = {
        "schema_version": "bio-neb.run-interaction/1",
        "view": "concise",
        "frontier": frontier,
        "journey_id": JOURNEY_ID,
        "journey_file": "<run>/journey.jsonl",
        "status": status,
        "progress": {"activities": [], "model_evidence": [], "requirements": [], "tool_evidence": [], "lifecycle_evidence": []},
        "pending_reviews": list(reviews),
        "passes": [],
        "investigation_members": [],
        "investigation_policy": None,
    }
    if reviews:
        snap["review"] = reviews[0]
    snap.update(fields)
    return snap


def action(name: str, **properties: dict) -> dict:
    schema = {"action": {"const": name, "type": "string"}, **properties}
    return {"action": name, "response_schema": {"type": "object", "properties": schema, "required": list(schema)}}


def review(kind: str, request: dict, *actions: dict, event_id: str = "request-1") -> dict:
    return {"command_frontier": "answer", "kind": kind, "legal_actions": list(actions), "request": request,
            "request_event_id": event_id}  # fmt: skip


def investigation_request(*rsids: str) -> dict:
    items = [
        {"normalized_question_action": "observe_variant", "request_text": f"Does {SAMPLE} carry {r}?", "rsid": r} for r in rsids
    ]
    return {"artifact": None, "request": {"asserted_by": "researcher:test", "items": items}}


def candidate(contig: str, position: int, ref: str, alt: str) -> dict:
    return {"contig": contig, "one_based_position": position, "ref": ref, "alt": alt, "assembly_accession": "GCF_000001405.40"}


def member_id(rsid: str) -> str:
    return "request-member:" + hashlib.sha256(rsid.encode()).hexdigest()


def member(rsid: str, verdict: str, *alts: tuple[str, dict]) -> dict:
    """One requested target; `alts` are (work_item_id, candidate) pairs, one work item per ALT."""
    return {
        "request": {"normalized_question_action": "observe_variant", "request_text": f"Does {SAMPLE} carry {rsid}?", "rsid": rsid},
        "request_member_id": member_id(rsid),
        "work_item_ids": [w for w, _ in alts],
        "refsnp": {"status": "resolved" if alts else "not_found", "assessment": {"verdict": verdict, "rationale": f"{rsid}: {verdict}"},
                   "candidates": [c for _, c in alts], "source": {"elapsed_time_ms": 5, "observed_at": "2026-01-05T09:10:00Z"}},
        "assembly_report": {"assessment": {"verdict": "accepted"}},
    }  # fmt: skip


def proposal(*members: dict) -> dict:
    # A work item appears once even when two members (a merged rsID and its target) share it.
    work_items = {
        w: {"work_item_id": w, "target": c, "intended_action": "observe_variant"}
        for m in members
        for w, c in zip(m["work_item_ids"], m["refsnp"]["candidates"], strict=True)
    }
    return {
        "proposal_id": f"investigation-proposal:{DIGEST}",
        "request_members": list(members),
        "work_items": list(work_items.values()),
    }


def execution_proposal() -> dict:
    """The parts of an execution proposal the Approval view reads."""
    return {
        "proposal_id": "execution-proposal:1",
        "recipes": [{"candidate": {"output_kind": "VCF", "assay": "gDNA", "read_layout": "paired", "recipe_id": "germline-test"}}],
        "resources": [{"candidate": {"accelerator_count": 1, "accelerator_memory_gib": 80, "profile_id": "gpu-test", "cpu_count": 16,
                                     "memory_gib": 200, "availability": "not_assessed"}}],
        "reference": {"assembly_accession": "GCF_000001405.40", "record": {
            "assembly_name": "GRCh38.p14", "assembly_accession": "GCF_000001405.40", "organism_name": "Homo sapiens",
            "release_date": "2022-02-03"}},
        "inspected_inputs": {"inputs": [{"role": "read_1", "size_bytes": 2_000_000_000}]},
        "unresolved_execution_constraints": [],
        "fact_slots": [],
    }  # fmt: skip


QUESTION_TEXT = "The check of this target ended with an issue. Approve, amend or reject it."


def investigation_question(rsid: str, *, issue: str | None = None) -> dict:
    """A per-target question before any proposal. An issue finding carries `issue`; a question about intent
    leaves it out, as `observe` does."""
    question = {
        "assertion_id": "user-assertion:1", "collection_member_id": member_id(rsid), "revision_number": 1,
        "request": {"normalized_question_action": "observe_variant", "request_text": f"Does {SAMPLE} carry {rsid}?", "rsid": rsid},
        "question": QUESTION_TEXT if issue else f"Which variant do you mean by {rsid}?",
        "missing_information": f"NCBI RefSNP did not find {rsid}.",
    }  # fmt: skip
    if issue:
        question["issue"] = {"key": issue, "verdict": issue.split(": ")[-1], "source": "refsnp", "conflicts": []}
    return question


# Lane M's real response schemas per kind and action (`tests/schemas.json`, generated from the backend's response
# classes with synthetic request values; its `_source` names the commit and the generator).
SCHEMAS = json.loads(Path(__file__).with_name("schemas.json").read_text(encoding="utf-8"))
DIGEST = "0123456789abcdef" * 4


def failure_ask(*amendable: str) -> dict:
    """A failure Review's ask (`failure.Ask`): its request id and the fields an amendment may name."""
    return {"state": "pending", "stage": "test", "failure_event_id": "event-7", "request_id": "failure-request:1",
            "reason": "The step failed.", "legal_actions": ["retry", "abort"], "amendable": list(amendable)}  # fmt: skip


# The basis members a finalization correction can take out of (or put back into) the report.
REPORT_MEMBERSHIP = [
    {"member_id": f"admitted-observation-evidence:{'d' * 64}", "member_role": "admission", "included": True, "admitted": True,
     "payload_retained": True},
    {"member_id": f"retrieved-source:{'e' * 64}", "member_role": "source", "included": False, "admitted": False,
     "payload_retained": True},
]  # fmt: skip
# Each kind's request: only the parts a response binds to (ids), the rest left out.
KIND_REQUESTS = {
    "input": lambda: {"request": failure_ask("input_correction")},
    "junction": lambda: {"request": failure_ask("instruction")},
    "investigation_download": lambda: {"request": failure_ask()},
    "run_health": lambda: {"request": failure_ask()},
    "paper": lambda: {"request": failure_ask("paper_reference", "paper_file")},
    "execution_path": lambda: {"review": {**failure_ask("reference"), "stage": "cpu_preparation"}},
    "preparation": lambda: {"code": "missing", "missing_fields": ["configuration"]},
    "planning": lambda: {"question": "Is the input whole-genome sequencing?", "issue": "The read headers do not name the assay.",
                         "candidates": [{"candidate_id": "wgs", "display_label": "Yes, WGS"},
                                        {"candidate_id": "wes", "display_label": "No, exome"}]},
    "model_availability": dict,
    "agent_effort": dict,
    "agent_work": dict,
    "execution": lambda: {"proposal": {**execution_proposal(), "proposal_id": "execution-proposal:" + "1" * 64}},
    "investigation": lambda: {"proposal": proposal(*four_targets())},
    "investigation_finding": lambda: {"proposal": None, "question": investigation_question("rs30", issue="refsnp: not_found")},
    "investigation_question": lambda: {"proposal": None, "question": investigation_question("rs20")},
    "finalization": lambda: {"basis": {"basis_digest": f"finalization-basis:{DIGEST}", "report_membership": REPORT_MEMBERSHIP},
                             "source_head_event_id": "event-1166", "request_event_id": "request-1",
                             "legal_actions": ["approve", "correct", "stop"]},
}  # fmt: skip


# Two investigation reviews before any proposal: an issue finding (its own approve and amend) and a question about
# intent (clarify or reject, with the investigation's schemas).
ASKED_AS = {"investigation_finding": "investigation", "investigation_question": "investigation"}


def schema_review(kind: str, *names: str) -> dict:
    """A pending review of `kind` whose legal actions carry lane M's real response schemas (all of the kind's, or
    `names`), its request from `KIND_REQUESTS`."""
    actions = SCHEMAS["investigation" if kind == "investigation_question" else kind]
    chosen = names or tuple(actions)
    return review(ASKED_AS.get(kind, kind), KIND_REQUESTS[kind](),
                  *({"action": a, "response_schema": actions[a]} for a in chosen))  # fmt: skip


def four_targets() -> tuple[dict, ...]:
    """rs10 has two ALTs (two work items), rs20 one, rs30 none (RefSNP did not find it), and rs11 was merged
    into rs20, so it shares rs20's work item."""
    return (
        member("rs10", "accepted", (f"work-item:{'a' * 64}", candidate("NC_000001.11", 1000, "G", "A")),
               (f"work-item:{'b' * 64}", candidate("NC_000001.11", 1000, "G", "T"))),
        member("rs20", "accepted", (f"work-item:{'c' * 64}", candidate("NC_000002.12", 2000, "T", "C"))),
        member("rs30", "not_found"),
        member("rs11", "merged", (f"work-item:{'c' * 64}", candidate("NC_000002.12", 2000, "T", "C"))),
    )  # fmt: skip


def report_item(
    work_item_id: str, target: str, outcome: str, genotype: str | None, *concerns: str, state: str = "observed"
) -> dict:
    """One work item of `collection_view` once looked up and the Run has ended: its VCF cells and reading."""
    item = {"work_item_id": work_item_id, "target": target, "outcome": outcome, "genotype": genotype,
            "observation_id": f"sample-observation:{work_item_id}", "coverage": "not_measured",
            "reading": "concern" if concerns else "no_concern", "reading_note": f"{target}: {outcome}", "state": state,
            "window": {"flank_bases": 100, "outcome": "window_observed",
                       "records": [f"{target} | QUAL 50.0 | GT {genotype} GQ 99 DP 30"] if genotype else []}}  # fmt: skip
    if genotype:
        item.update(genotype_quality=99, read_depth=30, allele_depths=[14, 16], qual="50.0")
    if concerns:
        item["concerns"] = list(concerns)
    return item


def admitted(work_item_id: str, detail: str, scope: str) -> dict:
    """An admission of a work item's observation: the claim's sentence and its scope."""
    return {"status": "admitted", "work_item_id": work_item_id, "claim_scope": scope,
            "admitted_claim": {"observation_detail": detail, "claim_scope": scope}}  # fmt: skip


def grounded_report(*, response: dict | None = None) -> dict:
    """`observe`'s `grounded_report` once published (`bio-neb.grounded-report/7`), with the parts the Results page
    reads: rs10 with two ALTs (one carried, one not, with a concern), rs20 with no record and not admitted, rs30
    unresolved with an issue the researcher approved, rs40 with an issue the researcher rejected (no work item)."""
    snps = [
        {"request_member_id": member_id("rs10"), "request_text": f"Does {SAMPLE} carry rs10?", "requested_rsid": "rs10",
         "verified_rsid": "rs10", "source_status": "resolved", "locus": "GRCh38 NC_000001.11:1000 G", "state": "candidates", "no_work_reason": None,
         "cases": ["multi_alt"], "items": [
             report_item("work-item:10A", "NC_000001.11:1000 G>A", "exact_allele_observed", "0/1"),
             report_item("work-item:10T", "NC_000001.11:1000 G>T", "target_allele_not_carried", "0/1", "nearby_variant")]},
        {"request_member_id": member_id("rs20"), "request_text": f"Does {SAMPLE} carry rs20?", "requested_rsid": "rs20",
         "verified_rsid": "rs20", "source_status": "resolved", "locus": "GRCh38 NC_000002.12:2000 T", "state": "candidates", "no_work_reason": None,
         "items": [report_item("work-item:20C", "NC_000002.12:2000 T>C", "no_record", None, "absence_not_established")]},
        {"request_member_id": member_id("rs30"), "request_text": f"Does {SAMPLE} carry rs30?", "requested_rsid": "rs30",
         "verified_rsid": None, "source_status": "not_found", "locus": None, "state": "unresolved",
         "state_note": "refsnp: not_found (approved, checked not verified)", "no_work_reason": "refsnp: not_found", "items": []},
        {"request_member_id": member_id("rs40"), "request_text": f"Does {SAMPLE} carry rs40?", "requested_rsid": "rs40",
         "verified_rsid": "rs40", "source_status": "resolved", "locus": None, "state": "rejected",
         "state_note": "the researcher rejected this target", "no_work_reason": None, "items": []},
    ]  # fmt: skip
    source = f"retrieved-source:{'f' * 64}"
    report = {
        "schema_version": "bio-neb.grounded-report/7",
        "report_digest": f"grounded-report:{DIGEST}",
        "report_basis_digest": f"report-basis:{DIGEST}",
        "finalization_basis_digest": f"finalization-basis:{DIGEST}",
        "outcome": "investigation_evidence_ready",
        "question": {"membership": {"included": True}, "value": {"request_text": f"Does {SAMPLE} carry rs10?", "asserted_by": "researcher:test"}},
        "collection_view": {"snps": snps, "cases": [
            {"origin": {"code": "multi_alt", "request": "rs10", "alts": ["A", "T"]}, "request_member_ids": [member_id("rs10")],
             "work_item_ids": ["work-item:10A", "work-item:10T"], "evidence_ids": [source]}]},
        "work_items": [{"membership": {"included": True}, "value": {"work_item_id": w}}
                       for w in ("work-item:10A", "work-item:10T", "work-item:20C")],
        "admissions": [
            {"membership": {"included": True}, "value": admitted("work-item:10A", f"Sample {SAMPLE} carries the target ALT G>A with genotype 0/1.",
                                                 "Exact allele and genotype observation in this VCF only.")},
            {"membership": {"included": True}, "value": admitted("work-item:10T", f"Sample {SAMPLE} does not carry the target ALT G>T.",
                                                 "This sample's genotype at the requested REF does not carry the target ALT.")},
            {"membership": {"included": True}, "value": {"status": "not_admitted", "work_item_id": "work-item:20C", "reason": "no record is no claim",
                                         "claim_scope": "No record was observed at this locus in this variant-only VCF."}},
        ],
        "issue_items": [
            {"collection_member_id": member_id("rs30"), "request": {"rsid": "rs30", "request_text": f"Does {SAMPLE} carry rs30?"},
             "issue": {"key": "refsnp: not_found", "verdict": "not_found"}, "request_event_id": "event-1",
             "decision": "approved", "decided_by": "researcher:test", "response_event_id": "event-2"},
            {"collection_member_id": member_id("rs40"), "request": {"rsid": "rs40", "request_text": f"Does {SAMPLE} carry rs40?"},
             "issue": {"key": "refsnp: incompatible", "verdict": "incompatible"}, "request_event_id": "event-3",
             "decision": "rejected", "decided_by": "researcher:test", "response_event_id": "event-4"},
        ],
        "review_authority": {"request_event_id": "request-1", "response_head_event_id": "request-1",
                             "basis_digest": f"finalization-basis:{DIGEST}", "source_head_event_id": "event-1",
                             "response": response or {"action": "approve"}},
        "retrieved_sources": [{"membership": {"included": True}, "value": {
            "source_id": source, "authority": "ncbi", "operation": "fetch_refsnp", "status_code": 200, "elapsed_time_ms": 12,
            "observed_at": "2026-01-05T09:20:00Z", "request_identity": "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/10"}}],
        "source_assessments": [{"membership": {"included": True}, "value": {"source_id": source, "verdict": "accepted",
                                                             "rationale": "RefSNP resolved one locus."}}],
        "conflicts": [qualification("target_allele_not_carried", "The sample does not carry the requested target ALT in this VCF.")],
        "limitations": [qualification("observation_not_inferred",
                                      "Coverage, callability, biological absence, function, causality, and diagnosis are not inferred.")],
        "gaps": [qualification("coverage_not_measured", "Coverage was not measured for Work item work-item:10A.")],
        "evidence_pointers": [{"pointer_id": "finalization-evidence-pointer:1"}],
        "accounting": [{"membership": {"included": True},
                        "value": {"accounting_id": "finalization-accounting:1", "event_type": "GroundedReportPublished"}}],
        "report_membership": [{"member_id": "finalization-accounting:1", "included": True}],
        "inherited_model_contact_count": 3,
        "inherited_source_contact_count": 1,
        "report_external_contact_count": 0,
        "renderer_version": "bio-neb.grounded-report-renderer/2",
    }  # fmt: skip
    return {"status": "published", "report": report, "publication_event_id": "event-report",
            "artifact": {"artifact_id": f"sha256:{'9' * 64}", "byte_count": 4096}, "report_external_contact_count": 0}  # fmt: skip


def qualification(code: str, text: str) -> dict:
    """A limitation, conflict or gap as `grounded-report/7` keeps it: its sentence and its typed origin (#474)."""
    return {"origin": {"code": code}, "request_member_ids": [], "text": text, "work_item_ids": []}


def verified(role: str, location: str, byte_count: int) -> dict:
    return {"role": role, "location": location, "byte_count": byte_count, "sha256": "7" * 64, "grounding": "verified",
            "producer_stage": "h100", "proof_ref": f"sha256:{'8' * 64}"}  # fmt: skip


def run_artifacts() -> dict:
    """`run-artifacts`' `bio-neb.run-artifacts/1` document: the Run's verified files and their proofs."""
    return {
        "schema_version": "bio-neb.run-artifacts/1", "journey_id": JOURNEY_ID, "journey_file": "<run>/journey.jsonl",
        "run_root": "<run>", "execution_id": "parabricks-execution:1", "authority_id": "execution-authority:1",
        "outcome_event_id": "event-5", "verified_event_id": "event-6", "bucket": "bio-neb-test", "object_scope": "execution/test",
        "outputs": [verified("VCF", "s3://bio-neb-test/execution/test/variants.vcf", 48_808_433)],
        "inputs": [verified("read_1", "s3://bio-neb-test/inputs/read_1.fastq.gz", 2_000_000_000)],
        "reference": [verified("reference_fasta", "s3://bio-neb-test/reference/genome.fa", 3_100_000_000)],
        "local": [{"role": "VCF", "path": "investigation/vcf/variants.vcf.gz", "byte_count": 12_000_000, "sha256": "6" * 64,
                   "receipt": "investigation/vcf/receipt.json"}],
    }  # fmt: skip
