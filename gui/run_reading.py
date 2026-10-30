"""Reading the Run: pure readers over its Journey items and `observe` snapshots, and one page's `Viewer` of it.

Nothing here builds an element. The page, the stage views, the decision element and the Journey panel share these.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from journey_feed import FeedItem

# Where each review kind of `observe` is asked, and what it asks of the researcher. Operational holds and a failed
# agent (`junction`, raised at that agent's own stage) are asked where the Run is; a kind not listed here gets the
# same plain fallback.
REVIEW_STAGE = {"input": "Question", "paper": "Question", "planning": "Plan", "execution": "Approval",
                "preparation": "Approval", "execution_path": "Execution",
                "investigation": "Investigation", "investigation_download": "Investigation", "run_health": "Investigation",
                "finalization": "Results"}  # fmt: skip
# A review shown on a display section other than its stage's: a failed health measurement is asked on Health.
REVIEW_SECTION = {"run_health": "health"}
REVIEW_ASKS = {"input": "correct the inputs", "planning": "answer a planning question",
               "model_availability": "decide on model availability", "agent_effort": "decide on the agent's effort",
               "agent_work": "decide on the agent's work", "execution": "approve the plan",
               "preparation": "supply what the reference preparation needs", "execution_path": "decide on the execution path",
               "junction": "decide on a failed step", "investigation": "review the investigation",
               "investigation_download": "decide on a failed download", "finalization": "approve the report basis",
               "paper": "decide on a paper that could not be read", "run_health": "decide on a failed health measurement"}  # fmt: skip
CONSEQUENCE = {
    ("planning", "select"): "Records your choice as a researcher assertion (not a verified fact) and planning continues.",
    ("planning", "clarify"): "Sends your note to the planning agent; it asks again or continues.",
    ("planning", "reject"): "Rejects this one question; its evidence is kept.",
    ("planning", "stop"): "You confirm the evidence cannot be supplied. Planning stops here.",
    ("execution", "approve"): "Starts Parabricks on the proposed GPU. Uses paid cloud compute.",
    ("execution", "revise"): "Sends your instruction back to planning; nothing runs.",
    ("execution", "reject"): "Nothing runs. The proposal is closed.",
    ("investigation", "approve"): "Authorises observing the selected targets in your VCF.",
    ("investigation", "revise"): "Changes the target list (keep, retry, remove, clarify) and re-plans.",
    ("investigation", "stop"): "Ends the investigation with what is known now.",
    ("finalization", "approve"): "Approves this exact report basis; the Grounded report is derived from it.",
    ("finalization", "correct"): "Sends your correction back; a new basis comes for review before any report.",
    ("finalization", "stop"): "Stops here; no Grounded report is produced.",
}


def event_payload(raw: list[str], item: FeedItem) -> dict:
    """An event's own payload. The feed tells one line per event, so the views read the few values they show beyond
    it (the prompt, the input inspection, the planning exchange, which execution fact is the outcome) from the event (#429)."""
    return json.loads(raw[item.n - 1])["payload"]


def prompt_lines(raw: list[str]) -> list[str]:
    """The researcher's prompt, line by line, from the request event (the Journey's opening lines)."""
    events = (json.loads(line) for line in raw)
    prompt = next((e["payload"]["prompt"] for e in events if e["event_type"] == "ResearchRequestRegistered"), "")
    return [line.strip() for line in prompt.splitlines() if line.strip()]


def run_title(raw: list[str]) -> str:
    sample = re.search(r"Sample (\S+?),", " ".join(prompt_lines(raw)))
    return sample.group(1) if sample else "Run"


def input_lines(items: list[FeedItem], raw: list[str]) -> list[str]:
    """The input inspection's result, one line per input: its size when the inspection succeeded, else each failure
    (the two result shapes are told apart by `status`; a failure names no input when it concerns the whole set)."""
    inspected = next((i for i in items if i.event_type == "InputMetadataInspected"), None)
    if inspected is None:
        return []
    result = event_payload(raw, inspected)["result"]
    if result["status"] == "failed":
        return [f"{f['role'] or 'inputs'}: inspection failed ({f['code']})" for f in result["failures"]]
    return [input_line(f) for f in result["inputs"]]


def input_line(observed: dict) -> str:
    """One inspected input: its size, and where it is when it is a staged storage object (`s3://`, #488), which is
    copied within the provider instead of uploaded. A staged object that was not found has no size."""
    path = observed.get("registered_path") or ""
    size = f"{observed['size_bytes'] / 1e9:.1f} GB" if observed.get("size_bytes") is not None else "not found"
    return f"{observed['role']}: {size}" + (f" · staged object {path}" if path.startswith("s3://") else "")


def health(items: list[FeedItem], raw: list[str]) -> list[tuple[FeedItem, dict]]:
    """The Run's health facts (#414): its one measurement (numbers, or why not measured), and each failed measurement
    and its Review and answer, in Journey order."""
    kinds = ("RunHealthMeasured", "RunHealthMeasurementFailed", "RunHealthReviewRequested", "RunHealthReviewAnswered")
    return [(i, event_payload(raw, i)) for i in items if i.event_type in kinds]


def readable_report(journey: Path, items: list[FeedItem], raw: list[str]) -> tuple[str | None, str]:
    """The Run's own `report.md` as written in its Run root (`ReadableReportWritten`, #476), and a line on it: None
    and why when it is not written yet, or missing; the page is shown as written, with its sha256 checked against
    the Journey's record."""
    written = next((i for i in reversed(items) if i.event_type == "ReadableReportWritten"), None)
    if written is None:
        return None, "report.md is not written yet: the Run writes it after its report explanation is published."
    record = event_payload(raw, written)
    page = journey.parent.parent / record["file_name"]
    if not page.is_file():
        return None, f"report.md was written at {written.time} but is not in the Run root ({page.parent})."
    data = page.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        line = f"report.md changed after the Run wrote it: its sha256 differs from the record at {written.time}."
    else:
        line = f"report.md as the Run wrote it at {written.time} (sha256 {record['sha256'][:12]}… checked)."
    return data.decode("utf-8"), line


def planning_exchange(items: list[FeedItem], raw: list[str]) -> list[tuple[FeedItem, str]]:
    """Each planning question and answer with what it said: the question; the option chosen, the note or the reason."""
    out, labels = [], {}
    for item in items:
        if item.event_type == "PlanningHitlRequested":
            asked = event_payload(raw, item)
            labels.update({c["candidate_id"]: c["display_label"] for c in asked["candidates"]})
            out.append((item, asked["question"]))
        elif item.event_type == "PlanningHitlResponded":
            out.append((item, _answered(event_payload(raw, item)["response"], labels)))
    return out


def _answered(response: dict, labels: dict[str, str]) -> str:
    """A planning answer has one shape per action: select names an offered option, clarify a note, stop and reject a
    reason, confirm nothing more."""
    action = response["action"]
    if action == "select":
        return labels[response["candidate_id"]]
    if action == "clarify":
        return response["instruction"]
    if action in ("stop", "reject"):
        return f"{action}: {response['reason']}"
    return action


def execution_outcome(items: list[FeedItem], raw: list[str]) -> FeedItem | None:
    """The execution fact that records the Parabricks run's outcome."""
    facts = (i for i in items if i.event_type == "ExecutionPathRecorded")
    return next((i for i in facts if event_payload(raw, i)["event_type"] == "ExecutionOutcome"), None)


def snapshot_at(snaps: dict[int, dict], cursor: int) -> tuple[int, dict] | tuple[None, None]:
    keys = [k for k in snaps if k <= cursor]
    return (max(keys), snaps[max(keys)]) if keys else (None, None)


def pass_open(items: list[FeedItem], cursor: int) -> bool:
    for item in reversed(items[:cursor]):
        if item.event_type == "PassClosed":
            return False
        if item.event_type == "PassStarted":
            return True
    return False


def waiting_review(items, snaps, cursor) -> dict | None:
    """The pending review the researcher must answer now, from the newest snapshot at a pass end."""
    if pass_open(items, cursor):
        return None
    line, snap = snapshot_at(snaps, cursor)
    if snap is None or snap["frontier"] != "review_required":
        return None
    last_close = max((i.n for i in items[:cursor] if i.event_type == "PassClosed"), default=0)
    # `review` is the backend's presentation focus among the pending reviews.
    return snap.get("review") if line == last_close else None


def review_stage(review: dict, current: str) -> str:
    return REVIEW_STAGE.get(review["kind"], current)


def review_asks(review: dict) -> str:
    return REVIEW_ASKS.get(review["kind"]) or f"answer a review ({review['kind'].replace('_', ' ')})"


def question_text(request: dict) -> tuple[str | None, str]:
    """The question a review asks and its context line; a finding question nests both in `question`."""
    question = request.get("question")  # review kinds differ; only some ask a question
    if isinstance(question, dict):
        return question["question"], question["missing_information"]
    return question, request.get("issue") or ""


def consequence(kind: str, action: dict) -> str:
    """What an action does: the backend's own schema description when it gives one, else the GUI's sentence."""
    return action["response_schema"].get("description") or CONSEQUENCE.get((kind, action["action"]), "")


def human(td: timedelta) -> str:
    s = int(td.total_seconds())
    return f"{s} s" if s < 90 else f"{s // 60} min" if s < 5400 else f"{s // 3600} h {s % 3600 // 60} min"


def fmt_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def short(identifier: str | None, n: int = 10) -> str:
    if not identifier:
        return "?"
    if len(identifier) <= n + 24:
        return identifier
    head, _, tail = identifier.rpartition(":")
    return f"{head}:{tail[:n]}…" if head else f"{identifier[:n]}…"


def requested_targets(snap: dict | None) -> list[str]:
    """The targets the researcher asked for, as `observe` captured the request (from Planning on)."""
    captured = (snap or {}).get("investigation_request")  # absent until the request is captured
    return [i["rsid"] or i["request_text"] for i in captured["request"]["items"]] if captured else []


def investigation_proposal(snap: dict) -> dict | None:
    """The proposal under review, else the newest one the investigation holds."""
    review = next((r for r in snap["pending_reviews"] if r["kind"] == "investigation" and r["request"]["proposal"]), None)
    investigation = snap.get("investigation")  # absent before the investigation starts
    proposals = investigation["proposals"] if investigation else []
    return review["request"]["proposal"] if review else (proposals[-1] if proposals else None)


def members(snap: dict) -> list[dict]:
    """Per-target facts for the investigation views: one entry per requested target, one work item per ALT."""
    proposal = investigation_proposal(snap)
    if proposal is None:
        return []
    investigation = snap.get("investigation")
    authority = investigation["authority"] if investigation else None  # null until approval
    authorised = {w["work_item_id"] for w in authority["authorized_work_items"]} if authority else set()
    targets = {w["work_item_id"]: w["target"] for w in proposal["work_items"]}
    out = []
    for m in proposal["request_members"]:
        ref, asm = m["refsnp"], m["assembly_report"]  # null when that lookup did not run
        out.append({
            "rsid": m["request"]["rsid"] or m["request"]["request_text"],
            "status": ref["status"] if ref else "not looked up",
            "verdict": ref["assessment"]["verdict"] if ref else "not looked up",
            "rationale": ref["assessment"]["rationale"] if ref else "",
            "assembly": asm["assessment"]["verdict"] if asm else None,
            "source": ref["source"] if ref else None,
            "work_items": [{"work_item_id": w, "target": targets[w], "authorised": w in authorised} for w in m["work_item_ids"]],
        })  # fmt: skip
    return out


def allele(target: dict) -> str:
    return f"{target['contig']}:{target['one_based_position']:,} {target['ref']}>{target['alt']}"


def target_summary(snap: dict | None) -> str:
    """The runs list and Research count: requested targets, plus work items once a proposal exists."""
    asked = requested_targets(snap)
    if not asked:
        return "targets not known yet"
    # A merged rsID shares its target's work item, so count distinct work items.
    work = len({w["work_item_id"] for m in members(snap) for w in m["work_items"]}) if snap else 0
    return f"{len(asked)} targets" + (f" · {work} work items" if work else "")


@dataclass
class Viewer:
    """One page's copy of the Run and where it looks: the events read so far (`cursor`), the stage shown when the
    researcher looks away from the Run's own (`view`) and the element variants picked by URL. Replay copies the
    read-only module data and runs on a simulated clock (`sim`); live mode starts empty and appends."""

    journey: Path
    live: bool
    items: list[FeedItem]
    raw: list[str]
    snaps: dict[int, dict]
    refused: dict[int, str]
    variants: dict[str, str]
    # The refusal code when the backend's `journey-feed` refused this Run and the narrator tells the items instead.
    feed_refused: str | None = None
    # What `run-artifacts` said about the Run once its report is published: (document, None) or (None, refusal);
    # None until it is read.
    artifacts: tuple[dict | None, dict | None] | None = None
    cursor: int = 0
    view: str | None = None
    sim: datetime | None = None

    def now(self) -> datetime:
        """Replay runs on a simulated clock; live uses the wall clock."""
        return datetime.now(timezone.utc) if self.live or self.sim is None else self.sim

    def url(self, **change) -> str:
        q = {**self.variants, "at": self.cursor, **({"view": self.view} if self.view else {}), **change}
        return "/?" + urlencode({k: v for k, v in q.items() if v is not None})

    def current_stage(self) -> str:
        return self.items[self.cursor - 1].phase if self.items else "Question"

    def snapshot(self) -> dict | None:
        return snapshot_at(self.snaps, self.cursor)[1]

    def review(self) -> dict | None:
        return waiting_review(self.items, self.snaps, self.cursor)
