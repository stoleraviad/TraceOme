"""Journey events -> the Journey panel's feed items.

The items come from the backend's `journey-feed --all` (`bio-neb.journey-feed/1`, #281) through `cli_bridge`: one
line per event, its stage mapped to a header chip and its status to an icon. Only when the feed refuses a Run does
the GUI's own `Narrator` tell them: it reads `journey.jsonl` lines as plain JSON (no BioNeb import) and gives each
event an icon, a one-line title, a few bounded detail lines and a level. Both use the feed's levels, which decide
which Journey-panel views show an item:

- key:   a milestone or a decision (always shown)
- step:  a concrete step the agent or a tool took (story view and timeline)
- quiet: bookkeeping worth seeing in the timeline (passes, checkpoints, agent attempts)
- noise: model payloads and activity ticks (only with "all events"; payloads never rendered)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cli_bridge

PHASES = ["Question", "Plan", "Approval", "Execution", "Investigation", "Results"]

DISPLAY_STAGES = {
    "request": ("Prompt & inputs", "description"),
    "parabricks": ("Parabricks run", "memory"),
    "health": ("Health", "monitor_heart"),
    "investigation": ("Investigation", "travel_explore"),
    "findings": ("Findings", "fact_check"),
    "report": ("Report", "summarize"),
}

PHASE_SECTION = {
    "Question": "request",
    "Plan": "parabricks",
    "Approval": "parabricks",
    "Execution": "parabricks",
    "Investigation": "investigation",
    "Results": "report",
}


@dataclass
class FeedItem:
    n: int  # 1-based line number in journey.jsonl
    ts: datetime
    phase: str
    level: str
    icon: str
    color: str
    title: str
    event_type: str
    event_id: str
    size: int
    pass_no: int
    details: list[str] = field(default_factory=list)
    needs_you: bool = False
    you: bool = False  # a researcher action

    @property
    def time(self) -> str:
        return self.ts.strftime("%H:%M:%S")


def display_stage(item: FeedItem, raw: str = "") -> str:
    """Presentation grouping only; the backend phase and available actions remain authoritative."""
    if item.event_type == "ExecutionPathRecorded" and raw:
        if json.loads(raw).get("payload", {}).get("event_type") == "H100ArtifactsVerified":
            return "health"
    if item.event_type.startswith("RunHealth"):
        return "health"
    if item.event_type in {
        "InvestigationEvidenceRecorded",
        "InvestigationItemWindowRecorded",
        "InvestigationTableChunkRead",
        "InvestigationItemEvidenceCollected",
        "InvestigationWorkItemStarted",
    }:
        return "findings"
    if item.event_type.startswith(("Finalization", "ReportExplanation", "GroundedReport")):
        return "report"
    return PHASE_SECTION.get(item.phase, "request")


def display_part(item: FeedItem) -> str:
    names = {
        "ResearchRequestRegistered": "Research question",
        "RunInputsRegistered": "Registered files",
        "InvestigationRequestRegistered": "Requested targets",
        "PlanningHitlRequested": "Planning questions",
        "PlanningHitlResponded": "Planning questions",
        "ExecutionHitlResponded": "Execution approval",
        "ExecutionProposalSubmitted": "Execution proposal",
        "ExecutionPathRecorded": "Execution checks",
        "InvestigationReviewRequested": "Researcher review",
        "InvestigationReviewResponded": "Researcher review",
        "InvestigationAuthorityGranted": "Researcher review",
        "RefSnpToolResultRecorded": "Sources & identity",
        "InvestigationSourceRetrieved": "Sources & identity",
        "InvestigationEvidenceRecorded": "Variant observations",
        "InvestigationItemWindowRecorded": "Nearby records",
        "InvestigationTableChunkRead": "Reading notes",
        "InvestigationItemEvidenceCollected": "Collected evidence",
        "FinalizationBasisCompiled": "Report preparation",
        "FinalizationReviewRecorded": "Report review",
        "GroundedReportPublished": "Published report",
    }
    if item.event_type.startswith("ReportExplanation"):
        return "Report explanation"
    if item.event_type.startswith("Model"):
        return "Model activity"
    return names.get(item.event_type, "Other recorded steps")


_TOOLS = {
    "record_proposal_requirements": "Listed what the plan needs",
    "attest_sample": "Recorded the sample you named",
    "resolve_taxon": "Looked up the organism",
    "record_supplied_reference": "Recorded the reference you supplied",
    "inspect_reference": "Checked the reference with NCBI Datasets",
    "describe_companion": "Located the reference FASTA",
    "list_compatible_analysis_recipes": "Listed compatible analysis recipes",
    "request_recipe_eligibility_assertion": "Prepared a question about the input assay",
    "select_recipe_candidate": "Chose the analysis recipe",
    "lookup_compatible_resource_profiles": "Looked up compute profiles",
    "select_resource_candidate": "Chose the compute profile",
    "fetch_refsnp": "Queried NCBI RefSNP",
    "fetch_assembly_report": "Queried the NCBI assembly report",
}

_EXEC_STEPS = {
    "ExecutionBound": ("link", "Parabricks run bound to your approval"),
    "InputsVerified": ("cloud_done", "Reads copied to storage and verified"),
    "ReferenceResolved": ("dataset", "Reference prepared and verified"),
    "H100ArtifactsVerified": ("memory", "H100 finished: BAM and VCF verified"),
    "ExecutionOutcome": ("task_alt", "Parabricks run finished"),
}


def _gb(n: int | None) -> str:
    if not n:
        return "?"
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def _first(d: dict, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


class Narrator:
    """Stateful: phase and pass number depend on what came before."""

    def __init__(self) -> None:
        self.phase = "Question"
        self.pass_no = 0
        self.attempt_roles: dict[str, str] = {}
        self.last_tool: dict[str, str] = {}
        self.labels: dict[str, str] = {}  # HITL candidate id -> display label
        self.contact = ("", "")

    def item(self, n: int, line: str) -> FeedItem:
        event = json.loads(line)
        et = event["event_type"]
        p = event.get("payload") or {}
        level, icon, color, title, details = "quiet", "circle", "grey-6", et, []
        needs_you = you = False

        # Phase changes are driven by the backend's own milestones.
        if et == "PassStarted":
            self.pass_no += 1
        if et in ("ProposalRequirementsIdentified", "AttemptStarted") and self.phase == "Question":
            self.phase = "Plan"
        if et == "ExecutionProposalSubmitted":
            self.phase = "Approval"
        if et == "ExecutionHitlResponded":
            self.phase = "Execution"
        if et == "InvestigationStarted":
            self.phase = "Investigation"
        if "Report" in et and "Publish" in et:
            self.phase = "Results"

        if et == "JourneyStarted":
            level, icon, color, title = "key", "play_circle", "primary", "Run started"
        elif et == "ResearchRequestRegistered":
            level, icon, color, title = "key", "help_outline", "primary", "Your question was registered"
            details = [line.strip() for line in p.get("prompt", "").splitlines() if line.strip()]
        elif et == "RunInputsRegistered":
            roles = [i.get("role") for i in p.get("inputs", [])]
            level, icon, title = "step", "upload_file", f"{len(roles)} read files registered"
            details = [", ".join(r for r in roles if r)]
        elif et == "ParabricksConfigurationRegistered":
            level, icon, title = "step", "settings", "Compute configuration loaded"
        elif et == "InputMetadataInspected":
            files = _first(p, "result", "inputs", default=[])
            level, icon, title = "step", "fact_check", "Read files inspected"
            details = [f"{i.get('role')}: {_gb(i.get('size_bytes'))}" for i in files]
        elif et == "PassStarted":
            level, icon, title = "quiet", "play_arrow", f"Agent pass {self.pass_no} started"
        elif et == "PassClosed":
            disp = p.get("disposition", "")
            level, icon, title = "quiet", "pause", f"Agent pass {self.pass_no} closed ({disp})"
        elif et == "AttemptStarted":
            role = p.get("agent_role", "agent")
            self.attempt_roles[p.get("attempt_id", "")] = role
            level, icon, title = "quiet", "smart_toy", f"{role.replace('_', ' ').capitalize()} started"
        elif et == "AttemptSucceeded":
            level, icon, color, title = "quiet", "check", "positive", "Agent finished this step"
        elif et == "ToolCallValidated":
            name = p.get("tool_name") or "tool"
            self.last_tool[p.get("attempt_id", "")] = name
            level, icon, title = "quiet", "build", _TOOLS.get(name, name.replace("_", " "))
            query = _first(p, "arguments", "rsid") or _first(p, "arguments", "query")
            if query:
                title += f": {query}"
        elif et == "ProposalRequirementsIdentified":
            reqs = _first(p, "plan", "requirements", default=[]) or []
            level, icon, title = "key", "checklist", f"Plan needs {len(reqs) or 'several'} items settled"
        elif et == "SampleAttestationRecorded":
            level, icon, title = "step", "badge", "Sample identity recorded as your assertion"
        elif et == "TaxonResolutionRecorded":
            cands = _first(p, "result", "candidates", default=[]) or [{}]
            c = cands[0]
            level, icon, title = "step", "pets", f"Organism resolved: {c.get('scientific_name')} ({c.get('ncbi_tax_id')})"
        elif et == "ReferenceChoiceRecorded":
            level, icon, title = "step", "bookmark", f"Reference chosen: {p.get('assembly_accession')}"
        elif et == "ReferenceDescriptionRecorded":
            acc = _first(p, "result", "assembly_accession")
            level, icon, color, title = "step", "verified", "positive", f"Reference {acc} verified with NCBI Datasets"
        elif et == "CompanionObservationRecorded":
            level, icon, title = "step", "description", "Reference FASTA located at NCBI"
        elif et == "PlanningHitlRequested":
            level, icon, color, title, needs_you = "key", "front_hand", "warning", "Question for you", True
            details = [p.get("question") or p.get("issue") or ""]
            self.labels.update({c.get("candidate_id"): c.get("display_label") for c in p.get("candidates", [])})
            labels = [c.get("display_label") for c in p.get("candidates", [])]
            if labels:
                details.append("Options: " + " / ".join(labels))
        elif et == "PlanningHitlResponded":
            r = p.get("response", {})
            value = p.get("value") or self.labels.get(r.get("candidate_id"))
            level, icon, color, title, you = "key", "person", "primary", f"You answered: {r.get('action')}", True
            details = [f"value: {value}"] if value else []
        elif et == "RecipeObservationRecorded":
            cands = _first(p, "result", "candidates", default=[])
            ok = [c for c in cands if c.get("compatibility") == "compatible"]
            level, icon, title = "step", "science", f"{len(cands)} recipe(s) found, {len(ok)} compatible"
        elif et == "PlanningCandidateSelected":
            level, icon, title = "step", "done", f"Agent selected a {p.get('kind', 'candidate')}"
        elif et == "ResourceObservationRecorded":
            cands = _first(p, "result", "candidates", default=[])
            level, icon, title = "step", "memory", f"{len(cands)} compute profile(s) found"
            details = [
                f"{c.get('profile_id')}: {c.get('accelerator_count')}x GPU {c.get('accelerator_memory_gib')} GiB, "
                f"{c.get('cpu_count')} CPU, {c.get('memory_gib')} GiB RAM"
                for c in cands
            ]
        elif et == "ReferenceSourceAdmitted":
            level, icon, title = "step", "verified_user", "Reference source admitted"
        elif et == "ExecutionPackageCompiled":
            level, icon, title = "step", "inventory_2", "Execution package compiled"
        elif et == "ExecutionProposalSubmitted":
            level, icon, color, title, needs_you = "key", "front_hand", "warning", "Plan ready for your approval", True
        elif et == "ExecutionHitlResponded":
            action = _first(p, "response", "action")
            level, icon, color, title, you = "key", "person", "primary", f"You {action}d the plan", True
        elif et == "ExecutionPathRecorded":
            inner = p.get("event_type", "")
            icon, title = _EXEC_STEPS.get(inner, ("bolt", inner))
            level, color = "key", "positive" if p.get("grounding") == "verified" else "primary"
            pointers = _first(p, "payload", "pointers", default=[]) or _first(p, "payload", "reference", "pointers", default=[])
            if pointers:
                details = [f"{x.get('role')}: {_gb(x.get('bytes'))}" for x in pointers]
            if inner == "ExecutionOutcome":
                details = [f"outcome: {_first(p, 'payload', 'disposition')}"]
        elif et == "InvestigationStarted":
            level, icon, color, title = "key", "travel_explore", "primary", "Variant investigation started"
        elif et == "InvestigationAttemptBound":
            level, icon, title = "quiet", "link", "Investigation agent bound"
        elif et == "InvestigationSourceContactStarted":
            self.contact = (p.get("operation", ""), p.get("request_key", ""))
            level, icon, title = "quiet", "public", f"Contacting NCBI ({p.get('operation')}: {p.get('request_key')})"
        elif et == "InvestigationSourceRetrieved":
            s = p.get("source", {})
            source = {"fetch_refsnp": "RefSNP", "fetch_assembly_report": "assembly report"}.get(self.contact[0], "NCBI")
            level, icon = "step", "download_done"
            title = f"NCBI {source} answered for {self.contact[1]} in {s.get('elapsed_time_ms')} ms"
            details = [f"{_gb(s.get('raw_response_byte_count'))}, HTTP {s.get('status_code')}"]
        elif et == "RefSnpToolResultRecorded":
            r = p.get("result", {})
            verdict = _first(r, "assessment", "verdict")
            n_c = len(r.get("candidates", []))
            ok = verdict == "accepted"
            level, icon = "key", ("check_circle" if ok else "error_outline")
            color = "positive" if ok else "warning"
            title = f"{p.get('query')}: {'one target found' if ok else f'{verdict} ({n_c} targets)'}"
            details = [_first(r, "assessment", "rationale", default="")]
            for c in r.get("candidates", [])[:3]:
                details.append(
                    f"{c.get('contig')}:{c.get('one_based_position')} {c.get('ref')}>{c.get('alt')} ({c.get('assembly_name')})"
                )
        elif et == "AssemblyReportToolResultRecorded":
            verdict = _first(p, "result", "assessment", "verdict")
            level, icon, color, title = "step", "verified", "positive", f"Assembly confirmed ({verdict})"
        elif et == "InvestigationProposalRecorded":
            level, icon, title = "key", "assignment", "Investigation proposal ready"
        elif et == "InvestigationReviewRequested":
            actions = ", ".join(p.get("legal_actions", []))
            level, icon, color, title, needs_you = "key", "front_hand", "warning", "Review needed: investigation", True
            details = [f"you can: {actions}"]
        elif et == "InvestigationReviewResponded":
            r = p.get("response", {})
            items = r.get("work_item_ids") or []
            level, icon, color, title, you = "key", "person", "primary", f"You {r.get('action')}d {len(items)} target(s)", True
        elif et == "InvestigationAuthorityGranted":
            items = _first(p, "authority", "authorized_work_items", default=[])
            names = [", ".join(_first(i, "target", "aliases", default=[])) for i in items]
            level, icon, color, title = "key", "verified_user", "positive", f"Observation authorised: {', '.join(names)}"
        elif et in ("StructuredOutputCallRecorded", "AgentOutputValidated", "StructuredOutputValidationLinked"):
            level, icon, title = (
                "quiet",
                "rule",
                {
                    "StructuredOutputCallRecorded": "Agent wrote a structured answer",
                    "AgentOutputValidated": "Structured answer validated",
                    "StructuredOutputValidationLinked": "Validation linked",
                }[et],
            )
        elif et == "SliceCheckpointReached":
            level, icon, title = "quiet", "flag", f"Checkpoint: {str(p.get('checkpoint', '')).replace('_', ' ')}"
        elif et == "AgentActionTurnDispositionRecorded":
            count = p.get("correction_count") or 0
            level, icon, title = "noise", "replay", "Agent turn recorded"
            if count:
                level, icon, color, title = "quiet", "build_circle", "orange-8", f"Agent corrected a tool call ({count}x)"
                details = [str(p.get("correction_key", ""))[:200]]
        elif et == "ModelRequestRecorded":
            model = _first(p, "compatible_adapter_evidence", "model_name", default="model")
            level, icon, title = "noise", "psychology", f"Model request to {model.split('/')[-1]}"
        elif et == "ModelResponseRecorded":
            finish = _first(p, "compatible_adapter_evidence", "response", "message", "finish_reason")
            level, icon, title = "noise", "psychology_alt", f"Model answered ({finish})"
        elif et == "ModelTransportRequestRecorded":
            level, icon, title = "noise", "swap_horiz", "Model call sent"
        elif et in ("activity_started", "activity_succeeded"):
            ms = p.get("elapsed_ms")
            stage = str(p.get("stage", "")).replace("_", " ")
            level, icon = "noise", "timer"
            title = f"{stage} {'done in ' + format(ms / 1000, '.1f') + ' s' if et == 'activity_succeeded' else 'started'}"

        return FeedItem(
            n=n,
            ts=datetime.fromisoformat(event["occurred_at"].replace("Z", "+00:00")),
            phase=self.phase,
            level=level,
            icon=icon,
            color=color,
            title=title,
            event_type=et,
            event_id=event.get("event_id", ""),
            size=len(line),
            pass_no=self.pass_no,
            details=[d for d in details if d],
            needs_you=needs_you,
            you=you,
        )


# The backend feed's stages on the header chips; Finalization sits under Results.
FEED_PHASES = {"request": "Question", "planning": "Plan", "execution_review": "Approval", "execution": "Execution",
               "investigation": "Investigation", "finalization": "Results", "report": "Results"}  # fmt: skip
# The backend feed's statuses as an icon and its colour.
FEED_ICONS = {"done": ("check", "grey-7"), "question": ("front_hand", "warning"), "answer": ("person", "primary"),
              "verified": ("verified", "positive"), "failed": ("error_outline", "negative")}  # fmt: skip


def from_feed(items: list[dict]) -> list[FeedItem]:
    """`journey-feed` items (`bio-neb.journey-feed/1`) as Journey-panel rows: the feed's one line each, no details.

    Something from later work renders plainly instead of failing (owner, #427 "unknown kinds"): a stage the chips do
    not know stays on the stage before it, and a status without an icon gets a plain one.
    """
    rows, phase = [], PHASES[0]
    for item in items:
        phase = FEED_PHASES.get(item["stage"], phase)
        icon, color = FEED_ICONS.get(item["status"], ("circle", "grey-6"))
        rows.append(
            FeedItem(
                n=item["sequence"],
                ts=datetime.fromisoformat(item["occurred_at"].replace("Z", "+00:00")),
                phase=phase,
                level=item["level"],
                icon=icon,
                color=color,
                title=item["text"],
                event_type=item["event_type"],
                event_id=item["event_id"],
                size=item["payload_bytes"],
                pass_no=item["pass_number"],
                needs_you=item["status"] == "question",
                you=item["status"] == "answer",
            )
        )
    return rows


def journey_lines(journey_file: Path) -> list[str]:
    """The Journey's lines (none before the file exists)."""
    if not journey_file.exists():
        return []
    return [line for line in journey_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_items(journey_file: Path) -> tuple[list[FeedItem], list[str], str | None]:
    """Whole file -> items, raw lines (kept only to show a bounded excerpt on demand) and the feed's refusal.

    The items are the backend's `journey-feed`; only when it refuses does the narrator tell them, and the refusal
    code says so.
    """
    lines = journey_lines(journey_file)
    feed, refusal = cli_bridge.journey_feed(journey_file)
    if feed is not None:
        return from_feed(feed["items"]), lines, None
    narrator = Narrator()
    return [narrator.item(i, line) for i, line in enumerate(lines, 1)], lines, refusal


class JourneyTail:
    """Live mode: keep a byte offset and return only complete new lines."""

    def __init__(self, path: Path) -> None:
        self.path, self.offset = path, 0

    def poll(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        return [raw for raw in chunk[: end + 1].decode("utf-8").splitlines() if raw.strip()]


class LiveFeed:
    """Live mode: the items for the lines tailed so far. The backend's `journey-feed` re-reads the whole file on each
    call; once it refuses, the narrator tells every line instead, and `refused` holds the code."""

    def __init__(self, path: Path) -> None:
        self.path, self.refused = path, None
        self.narrator, self.told = Narrator(), []

    def items(self, lines: list[str]) -> list[FeedItem]:
        """Blocking (run it through `run.io_bound`): one item per line in `lines`, in Journey order."""
        if self.refused is None:
            feed, self.refused = cli_bridge.journey_feed(self.path)
            if feed is not None:
                # The file may have grown since `lines` was read; those items come with the next call.
                return from_feed(feed["items"][: len(lines)])
        self.told += [self.narrator.item(n, line) for n, line in enumerate(lines[len(self.told) :], len(self.told) + 1)]
        return list(self.told)
