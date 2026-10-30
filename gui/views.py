"""The centre of the page: one view per stage, and the plain review card for a review no stage view lays out.

The views read the viewer's snapshot at its cursor and the Journey items before it; a pending review's decision
element comes from `decide` (the decision module).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from nicegui import ui

import report
from journey_feed import PHASES, FeedItem
from run_reading import (
    REVIEW_ASKS,
    Viewer,
    allele,
    execution_outcome,
    human,
    input_line,
    input_lines,
    members,
    pass_open,
    planning_exchange,
    prompt_lines,
    requested_targets,
    review_asks,
    review_stage,
    target_summary,
)

# The kinds whose stage view lays out its own decision; every other review gets the plain review card.
VIEW_KINDS = {"planning": "Plan", "execution": "Approval", "investigation": "Investigation"}
CONSTRAINT_WORDS = {
    "prepared_reference_not_verified": "The prepared reference is not verified yet",
    "software_installation_not_verified": "The Parabricks installation is not verified yet",
    "provider_environment_availability_not_assessed": "Cloud capacity was not checked in advance",
    "accelerator_model_and_driver_not_verified": "GPU model and driver are not verified yet",
    "single_gpu_supported_not_recommended": "One GPU is supported but not recommended",
}


class StageViews:
    """Renders the stage the page shows; `exec_wait` is the running-Parabricks line the page's clock updates."""

    def __init__(self, viewer: Viewer, decide: Callable[[dict], None]) -> None:
        self.viewer = viewer
        self.decide = decide
        self.exec_wait: ui.label | None = None

    def render(self, stage: str, review: dict | None) -> None:
        """The stage's view, then the review card when the review is asked here and the view does not lay it out."""
        {"Question": self.question_view, "Plan": lambda: self.plan_view(review), "Approval": lambda: self.approval_view(review),
         "Execution": self.execution_view, "Investigation": lambda: self.investigation_view(review),
         "Results": self.results_view}[stage]()  # fmt: skip
        if review and review_stage(review, self.viewer.current_stage()) == stage and VIEW_KINDS.get(review["kind"]) != stage:
            self.review_card(review)

    def offer_header(self, kind: str, title: str, since: FeedItem | None) -> None:
        with ui.row().classes("w-full items-center gap-2 bg-amber-1 rounded p-2"):
            ui.icon("front_hand", color="amber-9")
            ui.label(f"Waiting for you · {title}").classes("font-bold")
            if since:
                ui.label(f"since {since.time}").classes("text-xs text-grey-7")

    def facts_list(self, slots: list[dict]) -> None:
        for s in slots:
            ok = s.get("status") == "verified"
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon("verified" if ok else "person", color="positive" if ok else "grey-7").classes("text-base")
                ui.label(f"{s.get('kind', '').replace('_', ' ')}: {s.get('verified_value') or s.get('requested_value')}").classes(
                    "text-sm"
                )
                ui.label(s.get("summary", "")).classes("text-xs text-grey-7")

    def question_view(self) -> None:
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label("Your question").classes("text-xs text-grey-7")
            for line in prompt_lines(self.viewer.raw):
                ui.label(line).classes("text-lg")
            with ui.row().classes("gap-1"):
                for r in requested_targets(self.viewer.snapshot()):
                    ui.chip(r, icon="location_on", color="blue-1").props("dense")
        with ui.row().classes("w-full gap-3"):
            with ui.card().props("flat bordered").classes("grow"):
                ui.label("Inputs").classes("font-bold")
                for d in input_lines(self.viewer.items[: self.viewer.cursor], self.viewer.raw):
                    ui.label(d).classes("text-sm mono")
            with ui.card().props("flat bordered").classes("grow"):
                ui.label("How it started").classes("font-bold")
                ui.label(f"cli begin · {self.viewer.items[0].ts:%d %b %Y %H:%M} UTC").classes("text-sm")
                ui.label("Starting Runs from the app is not in the first build.").classes("text-xs text-grey-7")

    def plan_view(self, review: dict | None) -> None:
        if review and review["kind"] == "planning":
            self.decide(review)
        asked = planning_exchange(self.viewer.items[: self.viewer.cursor], self.viewer.raw)
        if asked:
            with ui.card().props("flat bordered").classes("w-full"):
                ui.label("Questions so far").classes("font-bold")
                for i, said in asked:
                    ui.label(f"{i.time}  {'You answered' if i.you else 'Question'}: {said}").classes("text-sm")
        snap = self.viewer.snapshot()
        if snap:
            reqs = snap["progress"]["requirements"]
            done = sum(r["state"] == "resolved" for r in reqs)
            with ui.expansion(f"What the plan needs: {done} of {len(reqs)} settled", icon="checklist",
                              value=not review).classes("w-full").props("dense"):  # fmt: skip
                for r in reqs:
                    ok = r["state"] == "resolved"
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("check_circle" if ok else "radio_button_unchecked", color="positive" if ok else "grey-6")
                        ui.label(r["subject_key"].replace("_", " ")).classes("text-sm w-64")
                        ui.label(r["state"].replace("_", " ")).classes("text-xs text-grey-7")

    def approval_view(self, review: dict | None) -> None:
        # Only an execution review's proposal is an execution proposal; otherwise read observe's own.
        source = (review["request"]["proposal"] if review and review["kind"] == "execution" else None) or next(
            (
                s.get("execution_proposal")
                for k, s in sorted(self.viewer.snaps.items())
                if k <= self.viewer.cursor and s.get("execution_proposal")
            ),
            None,
        )
        if not source:
            ui.label("No plan to approve yet.").classes("text-grey-7")
            return
        if review and review["kind"] == "execution":
            self.offer_header(
                "execution",
                "approve the plan",
                next((i for i in self.viewer.items if i.event_type == "ExecutionProposalSubmitted"), None),
            )
        else:
            done = next((i for i in self.viewer.items[: self.viewer.cursor] if i.event_type == "ExecutionHitlResponded"), None)
            if done:
                ui.label(f"{done.title} at {done.time}.").classes("text-positive font-bold")
        recipe = source["recipes"][0]["candidate"]
        res = source["resources"][0]["candidate"]
        files = source["inspected_inputs"]["inputs"]
        with ui.grid(columns=2).classes("w-full gap-3"):
            for title, lines in (
                ("Analysis", [f"Parabricks germline → {recipe['output_kind']}", f"{recipe['assay']}, {recipe['read_layout']} reads",
                              f"recipe {recipe['recipe_id']}"]),
                ("Compute", [f"{res['accelerator_count']}x GPU, {res['accelerator_memory_gib']} GiB (profile {res['profile_id']})",
                             f"{res['cpu_count']} CPU, {res['memory_gib']} GiB RAM", f"availability: {res['availability'].replace('_', ' ')}"]),
                ("Reference", reference_lines(source)),
                ("Inputs", [input_line(f) for f in files]),
            ):  # fmt: skip
                with ui.card().props("flat bordered"):
                    ui.label(title).classes("font-bold")
                    for line in lines:
                        ui.label(line).classes("text-sm")
        warnings = source.get("unresolved_execution_constraints", [])
        if warnings:
            with ui.card().props("flat bordered").classes("w-full bg-orange-1"):
                ui.label(f"Not verified before you approve ({len(warnings)})").classes("font-bold")
                for w in warnings:
                    ui.label(f"• {CONSTRAINT_WORDS.get(w, w)}").classes("text-sm")
        with ui.expansion("Facts behind the plan", icon="fact_check").classes("w-full").props("dense"):
            self.facts_list(source.get("fact_slots", []))
        if review and review["kind"] == "execution":
            self.decide(review)

    def execution_view(self) -> None:
        steps = [i for i in self.viewer.items[: self.viewer.cursor] if i.event_type == "ExecutionPathRecorded"]
        started = next((i for i in self.viewer.items if i.event_type == "ExecutionHitlResponded"), None)
        if not started or started.n > self.viewer.cursor:
            ui.label("Execution has not started.").classes("text-grey-7")
            return
        running = pass_open(self.viewer.items, self.viewer.cursor) and self.viewer.current_stage() == "Execution"
        with ui.row().classes("w-full items-center gap-3"):
            if running:
                ui.spinner("dots", size="2em", color="primary")
                since = self.viewer.now() - self.viewer.items[self.viewer.cursor - 1].ts
                exec_wait = ui.label(f"Parabricks is running · last Journey event {human(since)} ago").classes("font-bold")
                self.exec_wait = exec_wait
            else:
                ui.icon("task_alt", color="positive").classes("text-2xl")
                ui.label(f"Parabricks finished at {steps[-1].time}" if steps else "").classes("font-bold")
        ui.label("No progress bar: these are the backend's recorded milestones. The Journey gets no heartbeat "
                 "while the GPU works, so long silences are real.").classes("text-xs text-grey-7")  # fmt: skip
        with ui.timeline(side="right", layout="dense").classes("w-full"):
            prev = started
            for s in steps:
                ui.timeline_entry(title=s.title, subtitle=f"{s.time} · +{human(s.ts - prev.ts)}",
                                  icon=s.icon, color="positive")  # fmt: skip
                prev = s
            if running:
                ui.timeline_entry("", title="Next milestone not recorded yet", subtitle="waiting", icon="hourglass_empty",
                                  color="grey-5")  # fmt: skip

    def tracks(self, snap: dict | None, ms: list[dict]) -> None:
        exec_done = execution_outcome(self.viewer.items[: self.viewer.cursor], self.viewer.raw)
        ok = sum(m["verdict"] == "accepted" for m in ms)
        policy = (snap or {}).get("investigation_policy") or {}
        blockers = ", ".join(b.replace("_", " ") for b in policy.get("completion_blockers", [])) or "none"
        research = f"{ok} resolved, {len(ms) - ok} not resolved" if ms else "checked; no proposal yet" if snap else "not started"
        with ui.grid(columns=3).classes("w-full gap-3"):
            for title, state, line, color in (
                ("Execution", "succeeded" if exec_done else "running", f"VCF verified at {exec_done.time}" if exec_done else "", "green-1"),
                ("Research", target_summary(snap), research, "blue-1"),
                ("Join", "blocked" if policy.get("completion_blockers") else "waiting", f"blockers: {blockers}", "orange-1"),
            ):  # fmt: skip
                with ui.card().props("flat bordered").classes(f"bg-{color}"):
                    ui.label(title).classes("text-xs text-grey-8 tracking-widest uppercase")
                    ui.label(state).classes("text-lg font-bold")
                    ui.label(line).classes("text-xs text-grey-8")

    def target_cards(self, ms: list[dict]) -> None:
        with ui.grid(columns=3).classes("w-full gap-3"):
            for m in ms:
                ok = m["verdict"] == "accepted"
                with ui.card().props("flat bordered").classes("gap-1").mark("target"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(m["rsid"]).classes("text-lg font-bold mono")
                        ui.badge("target found" if ok else m["verdict"], color="positive" if ok else "warning")
                    for w in m["work_items"]:
                        with ui.row().classes("items-center gap-1 no-wrap"):
                            ui.label(allele(w["target"])).classes("text-sm mono")
                            ui.label("authorised" if w["authorised"] else "not authorised").classes(
                                "text-xs " + ("text-positive" if w["authorised"] else "text-orange-9")
                            )
                    if not m["work_items"]:
                        ui.label("no work item: needs a clearer target first").classes("text-xs text-orange-9")
                    ui.label(m["rationale"]).classes("text-xs text-grey-8")
                    if s := m["source"]:
                        ui.label(f"NCBI RefSNP · {s['elapsed_time_ms']} ms · {s['observed_at'][11:19]}").classes(
                            "text-xs text-grey-7"
                        )

    def target_table(self, ms: list[dict]) -> None:
        """One row per target and work item (one per ALT); a target without a work item keeps one row."""
        rows = [{
            "row": f"{m['rsid']} {w['work_item_id'] if w else ''}", "rsid": m["rsid"], "verdict": m["verdict"],
            "where": allele(w["target"]) if w else "–", "assembly": m["assembly"] or "–",
            "authorised": ("yes" if w["authorised"] else "no") if w else "no work item",
        } for m in ms for w in (m["work_items"] or [None])]  # fmt: skip
        cols = [{"name": k, "label": k.capitalize(), "field": k, "align": "left"}
                for k in ("rsid", "verdict", "where", "assembly", "authorised")]  # fmt: skip
        ui.table(columns=cols, rows=rows, row_key="row").props("flat bordered dense").classes("w-full")

    def requested_list(self, snap: dict) -> None:
        """Before a proposal: the requested targets, and the finding questions waiting for an answer."""
        questions = {
            q["request"]["rsid"] or q["request"]["request_text"]: q for r in snap["pending_reviews"]
            if r["kind"] == "investigation" and (q := r["request"]["question"])
        }  # fmt: skip
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label(f"Requested targets ({len(requested_targets(snap))})").classes("font-bold")
            with ui.row().classes("gap-1"):
                for r in requested_targets(snap):
                    ui.chip(r, icon="front_hand" if r in questions else "location_on",
                            color="amber-2" if r in questions else "blue-1").props("dense").mark("target")  # fmt: skip
            for rsid, q in questions.items():
                # An issue finding names its issue; a question about intent has none (`issue` left out).
                issue = q.get("issue")
                ui.label(f"{rsid}: {issue['key'] + ' · ' if issue else ''}{q['missing_information']}").classes("text-sm")

    def investigation_view(self, review: dict | None) -> None:
        if PHASES.index(self.viewer.current_stage()) < PHASES.index("Investigation"):
            ui.label("Investigation has not started.").classes("text-grey-7")
            return
        snap = self.viewer.snapshot()
        if review and review["kind"] == "investigation":
            since = next(
                (i for i in reversed(self.viewer.items[: self.viewer.cursor]) if i.event_type == "InvestigationReviewRequested"),
                None,
            )
            self.offer_header("investigation", "review the investigation", since)
        ms = members(snap) if snap else []
        self.tracks(snap, ms)
        if ms:
            (self.target_table if self.viewer.variants["targets"] == "table" else self.target_cards)(ms)
        elif requested_targets(snap):
            self.requested_list(snap)
        else:
            ui.label("The investigation agent is querying NCBI; targets appear when the pass ends.").classes("text-grey-7")
        if review and review["kind"] == "investigation":
            remaining = review["request"].get("remaining_work_item_ids")
            if remaining:
                ui.label(f"{len(remaining)} work items still wait for your answer.").classes("text-sm")
            self.decide(review)

    def results_view(self) -> None:
        snap = self.viewer.snapshot()
        grounded = snap.get("grounded_report") if snap else None  # absent until the report projection exists
        if grounded and grounded["status"] == "published":
            published = next(
                (i for i in self.viewer.items[: self.viewer.cursor] if i.event_type == "GroundedReportPublished"), None
            )
            report.show(
                grounded, self.viewer.artifacts, published, live=self.viewer.live, findings=self.viewer.variants["findings"]
            )
            return
        ui.label("No Grounded report yet: the report is published after you approve the report basis.").classes("text-grey-7")

    def review_card(self, review: dict) -> None:
        """The plain review card: any review the stage view does not lay out itself, known kind or not."""
        since = next((i for i in reversed(self.viewer.items[: self.viewer.cursor]) if i.needs_you), None)
        self.offer_header(review["kind"], review_asks(review), since)
        if review["kind"] in REVIEW_ASKS:
            self.decide(review)
            return
        # A kind from later work: the GUI cannot know what answering it means, so it shows the review read-only.
        ui.label(f"This GUI has no form for the review kind '{review['kind']}' yet; answer it with the CLI.").classes(
            "text-xs text-grey-7"
        )
        ui.label("Legal actions: " + ", ".join(a["action"] for a in review["legal_actions"])).classes("text-sm mono")
        ui.code(json.dumps(review["request"], indent=1), language="json").classes("w-full text-xs").mark("raw-request")


def reference_lines(proposal: dict) -> list[str]:
    """The proposal's reference: the assembly as NCBI Datasets recorded it, the planning agent's choice with the words
    of yours it rests on (#459), and where its sequence comes from."""
    record = proposal["reference"]["record"]
    lines = [
        f"{record['assembly_name']} ({record['assembly_accession']})",
        f"{record['organism_name']}, released {record['release_date']}",
    ]
    choice = (proposal.get("reference_basis") or {}).get("choice")
    if choice:
        lines.append(f"chosen from your words “{choice['quote']}”" if choice.get("quote") else "chosen by the planning agent")
        if choice.get("reason"):
            lines.append(choice["reason"])
    admission = proposal.get("reference_source_admission")
    if admission:
        size = f" · {admission['expected_size_bytes'] / 1e9:.1f} GB" if admission.get("expected_size_bytes") else ""
        lines.append(
            f"sequence: {admission['source_mode'].replace('_', ' ')}{size}, {admission['expected_checksum_algorithm']} checked"
        )
    return lines
