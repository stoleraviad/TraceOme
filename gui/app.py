"""BioNeb researcher workspace over the CLI's Journey and observe projections. See README.md.

    RUN_DIR=<run folder with journey.jsonl> uv run python app.py           # recorded replay (default)
    LIVE=1 RUN_DIR=<run root>/<journey id> uv run python app.py            # follow a Run live

Every screen takes query parameters, so each variant has its own URL (README lists them):
at=<event line>, view=<stage>, journey=story|timeline|passes, decision=offer|form,
targets=cards|table, findings=cards|table, nav=chips|tabs, left=runs|hidden.

Rules kept from the CLI-bridge pattern: no BioNeb import; the Journey panel reads `journey-feed` (the GUI's own
narration only when the feed refuses); the centre reads `observe` snapshots. Following a live Run, a decision button
starts `answer` detached (and Resume `resume`) and the page shows the next `observe` when it ends; in a recorded replay
the buttons only show the exact CLI command (dry run).

This file holds the page: header, drawers, the centre's frame and the replay footer. The stage views
(`views.py`), the decision element (`decision.py`) and the Journey panel (`journey_panel.py`) build their parts
over the page's `Viewer` (`run_reading.py`).
"""

from __future__ import annotations

import bisect
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from nicegui import app, background_tasks, run, ui

import cli_bridge
import runner
from commands import Command, CommandBar
from decision import Decision
from journey_feed import PHASES, JourneyTail, LiveFeed, journey_lines, read_items
from journey_panel import JourneyPanel
from run_reading import (
    Viewer,
    human,
    pass_open,
    review_asks,
    review_stage,
    run_title,
    snapshot_at,
    target_summary,
    waiting_review,
)
from views import StageViews

HERE = Path(__file__).parent
if "RUN_DIR" not in os.environ:
    sys.exit("Set RUN_DIR to a Run folder that holds journey.jsonl (Runs live outside the repository).")
RUN_DIR = Path(os.environ["RUN_DIR"])
SNAPSHOT_DIR = Path(os.environ.get("SNAPSHOTS", HERE / "data" / "snapshots"))
LIVE = os.environ.get("LIVE") == "1"
JOURNEY = RUN_DIR / "journey.jsonl"
# The files a finalization answer names (correct, stop) are written here, outside the repository.
ANSWER_FILES = Path(os.environ.get("ANSWER_FILES") or Path(tempfile.gettempdir()) / "bioneb-gui-answers")
# Each command's stdout and stderr, outside the repository.
COMMAND_LOGS = Path(os.environ.get("COMMAND_LOGS") or Path(tempfile.gettempdir()) / "bioneb-gui-commands")
if not LIVE and not JOURNEY.exists():
    sys.exit(f"No journey.jsonl in {RUN_DIR}")

# Read-only module data, shared by every page (never mutated in replay).
ITEMS, RAW, FEED_REFUSED = ([], journey_lines(JOURNEY), None) if LIVE else read_items(JOURNEY)
SNAPS: dict[int, dict] = {int(p.stem): json.loads(p.read_text()) for p in SNAPSHOT_DIR.glob("*.json") if p.stem.isdigit()}
REFUSED: dict[int, str] = {
    int(p.name.split(".")[0]): json.loads(p.read_text()).get("code", "refused")
    for p in SNAPSHOT_DIR.glob("*.refused.json")
    if p.name.split(".")[0].isdigit()
}
# What `run-artifacts` said about the finished Run (make_snapshots.py records it): a document, or its refusal.
ARTIFACTS = next(
    ((json.loads(p.read_text()), None) if p.name == "artifacts.json" else (None, json.loads(p.read_text()))
     for p in (SNAPSHOT_DIR / "artifacts.json", SNAPSHOT_DIR / "artifacts.refused.json") if p.exists()),
    None,
)  # fmt: skip

# The command this app runs for the followed Run, shared by every page (live mode only; replay never answers).
COMMAND = Command(COMMAND_LOGS) if LIVE else None

PRIMARY = "#204d3e"
STAGE_ICON = {"Question": "help_outline", "Plan": "edit_note", "Approval": "gavel", "Execution": "memory",
              "Investigation": "travel_explore", "Results": "summarize"}  # fmt: skip


@ui.page("/")
def index(
    at: int | None = None, view: str | None = None, journey: str = "story", decision: str = "offer",
    targets: str = "cards", findings: str = "cards", nav: str = "chips", left: str = "hidden", play: int = 0,
    section: str = "auto", record: int | None = None, density: str = "simple", focus: str | None = None,
) -> None:  # fmt: skip
    ui.colors(primary=PRIMARY)
    ui.add_css(".mono{font-family:ui-monospace,Menlo,Consolas,monospace} .q-drawer{background:#fbfbfc}")
    ui.add_css(runner.CSS)
    density = "detailed" if density == "detailed" else "simple"
    sample = run_title(RAW)
    viewer = Viewer(
        journey=JOURNEY, live=LIVE, items=[] if LIVE else list(ITEMS), raw=[] if LIVE else list(RAW),
        snaps={} if LIVE else dict(SNAPS), refused={} if LIVE else dict(REFUSED), feed_refused=None if LIVE else FEED_REFUSED,
        artifacts=None if LIVE else ARTIFACTS,
        variants={"journey": journey, "decision": decision, "targets": targets, "findings": findings, "nav": nav, "left": left},
        view=view,
    )  # fmt: skip
    items, snaps, variants = viewer.items, viewer.snaps, viewer.variants
    variants.update(section=section, density=density)
    if record is not None:
        variants["record"] = str(record)
    if focus is not None:
        variants["focus"] = focus
    viewer.cursor = max(1, min(at or len(items), len(items))) if items else 0
    viewer.sim = items[viewer.cursor - 1].ts if items else None
    # The replay footer's own state; `key` is what the centre last rendered, so it rebuilds only when that changes.
    replay = {"playing": bool(play), "speed": 60, "key": None}
    result_state: dict = {"search": "", "outcome": "all", "expanded": set()}
    # Who answers: one name per browser, taken by every attribution field a response names.
    app.storage.user.setdefault("answering_as", "")
    answering_as = lambda: app.storage.user["answering_as"].strip()  # noqa: E731
    views = StageViews(viewer, Decision(JOURNEY, variants["decision"], answering_as, ANSWER_FILES, COMMAND).show)

    def stage_states() -> dict[str, str]:
        now = viewer.current_stage()
        review = viewer.review()
        out = {}
        for phase in PHASES:
            idx, now_idx = PHASES.index(phase), PHASES.index(now)
            out[phase] = "done" if idx < now_idx else "future" if idx > now_idx else "current"
        if review:
            out[review_stage(review, now)] = "needs_you"
        return out

    def status_line() -> tuple[str, str]:
        review = viewer.review()
        if review:
            return f"Waiting for you: {review_asks(review)}", "amber-3"
        if pass_open(items, viewer.cursor):
            item = items[viewer.cursor - 1]
            return f"Working: pass {item.pass_no} ({item.phase.lower()})", "blue-2"
        snap = viewer.snapshot()
        return (snap["status"].replace("_", " ") if snap else "Starting"), "grey-3"

    # ---------- header: the shared status banner ----------
    with ui.header().classes("runner-header column gap-0"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.icon("biotech", color="primary").classes("text-2xl")
            ui.label("BioNeb").classes("text-lg font-bold text-primary")
            title = ui.label(sample).classes("text-base text-grey-9")
            ui.chip("Live" if LIVE else "Recorded replay", icon="sensors" if LIVE else "history",
                    color="green-2" if LIVE else "blue-grey-2").props("dense square")  # fmt: skip
            ui.space()
            ui.input("Answering as", placeholder="your name").props("dense outlined").classes("w-44").bind_value(
                app.storage.user, "answering_as"
            ).mark("answering-as")
            status_chip = ui.chip("", color="grey-3").props("dense")
            clock = ui.label("").classes("text-xs text-grey-7 runner-clock")
            ui.button("Run details", icon="info", on_click=lambda: run_details()).props("flat dense no-caps")
            ui.toggle({"simple": "Simple", "detailed": "Detailed"}, value=density,
                      on_change=lambda e: ui.navigate.to(viewer.url(density=e.value))).props("dense no-caps")  # fmt: skip
        stage_bar = ui.row().classes("w-full items-center gap-1 pt-1")

    def render_stage_bar() -> None:
        stage_bar.clear()
        if density == "simple":
            with stage_bar:
                runner.navigation(viewer, section)
                if section != "auto" or viewer.view:
                    ui.button(
                        "Follow the Run",
                        icon="my_location",
                        on_click=lambda: ui.navigate.to(viewer.url(section="auto", view=None, record=None, focus=None)),
                    ).props("flat dense no-caps size=sm").classes("runner-follow")
            return
        if variants["nav"] != "chips":
            return
        states = stage_states()
        with stage_bar:
            for n, phase in enumerate(PHASES):
                s = states[phase]
                shown = (viewer.view or viewer.current_stage()) == phase
                color, text = {"done": ("green-1", "green-10"), "current": ("blue-1", "blue-10"),
                               "needs_you": ("amber-4", "black"), "future": ("grey-2", "grey-7")}[s]  # fmt: skip
                icon = {"done": "check", "needs_you": "front_hand"}.get(s, STAGE_ICON[phase])
                chip = ui.chip(phase, icon=icon, color=color, text_color=text, on_click=lambda p=phase: show_stage(p))
                chip.props("clickable square").classes("font-bold ring-2 ring-primary" if shown else "").mark(s.replace("_", "-"))
                if n < len(PHASES) - 1:
                    ui.icon("chevron_right", color="grey-5")
            if viewer.view:
                ui.button("Follow the Run", icon="my_location", on_click=lambda: show_stage(None)).props(
                    "flat dense no-caps size=sm"
                )

    def show_stage(phase: str | None) -> None:
        viewer.view = None if phase == viewer.current_stage() else phase
        render_stage_bar()
        centre.refresh()

    # ---------- left drawer: runs ----------
    runs_status = runs_title = None
    if variants["left"] == "runs":
        with ui.left_drawer(fixed=True).props("width=250 bordered"):
            ui.label("Runs").classes("text-base font-bold")
            ui.label(f"under {RUN_DIR.parent.name}/").classes("text-xs text-grey-7 mono")
            for folder in sorted(p.parent for p in RUN_DIR.parent.glob("*/journey.jsonl")):
                mine = folder.resolve() == RUN_DIR.resolve()
                with ui.card().props("flat bordered").classes("w-full p-2 " + ("bg-blue-1" if mine else "")):
                    label = ui.label(sample if mine else folder.name).classes("text-sm font-bold")
                    if mine:
                        runs_title = label
                    if mine and items:
                        ui.label(f"started {items[0].ts:%d %b %H:%M} · {len(items)} events").classes("text-xs text-grey-7")
                        runs_status = ui.label("").classes("text-xs")
            ui.separator().classes("my-2")
            ui.button("New run", icon="add").props("outline dense no-caps disable").classes("w-full")
            ui.label("First build follows Runs started with the CLI (decision E).").classes("text-xs text-grey-7")

    # ---------- centre ----------
    @ui.refreshable
    def centre() -> None:
        if not items:
            ui.label("Waiting for the first recorded event").classes("runner-section-title")
            return
        if density == "simple" or section != "auto":
            runner.main(viewer, views, section=section, record=record, focus=focus, density=density, state=result_state)
            return
        stage = viewer.view or viewer.current_stage()
        review = viewer.review()
        if variants["nav"] == "tabs":
            with (
                ui.tabs(value=stage, on_change=lambda e: show_stage(e.value)).classes("w-full").props("dense no-caps align=left")
            ):
                states = stage_states()
                for phase in PHASES:
                    tab = ui.tab(phase, icon={"done": "check", "needs_you": "front_hand"}.get(states[phase], STAGE_ICON[phase]))
                    if states[phase] == "needs_you":
                        tab.props("alert=amber")
        with ui.row().classes("w-full items-baseline gap-3"):
            ui.label(stage).classes("text-2xl font-bold")
            if viewer.view:
                ui.label(f"viewing an earlier/later stage · the Run is at {viewer.current_stage()}").classes(
                    "text-xs text-grey-7"
                )
        views.render(stage, review)

    def run_details() -> None:
        snap = viewer.snapshot()
        with ui.dialog() as dialog, ui.card().classes("w-[760px] max-w-full"):
            ui.label("Run details").classes("text-lg font-bold")
            rows = [
                ("Journey id", json.loads(viewer.raw[0])["journey_id"]),
                ("Journey file", str(JOURNEY)),
                ("Events read", f"{viewer.cursor} of {len(items)}"),
                ("Status / frontier", f"{snap['status']} / {snap['frontier']}" if snap else "no snapshot yet"),
                ("Snapshots", ", ".join(map(str, sorted(snaps))) + "  (lines where observe answered)"),
                ("observe refused", ", ".join(f"line {k}: {v}" for k, v in sorted(viewer.refused.items())) or "none"),
            ]
            if snap:
                pm = snap.get("model_contact_postmortem", {})
                rows += [
                    ("Model requests", f"{pm.get('logical_request_count')} logical, {pm.get('physical_contact_count')} physical, "
                                       f"{pm.get('terminal_disposition', '').replace('_', ' ')}"),
                    ("Passes", " · ".join(f"{p.get('disposition')} ({p.get('contact_count', 0)} contacts)" for p in snap.get("passes", []))),
                    ("Pending request", ", ".join(r["request_event_id"] for r in snap["pending_reviews"]) or "none"),
                ]  # fmt: skip
            for k, v in rows:
                with ui.row().classes("w-full no-wrap gap-3"):
                    ui.label(k).classes("w-40 text-sm text-grey-7 shrink-0")
                    ui.label(str(v)).classes("text-sm mono break-all")
            ui.button("Close", on_click=dialog.close).props("flat no-caps")
        dialog.open()

    with ui.column().classes("runner-shell"):
        with ui.element("div").classes("runner-grid"):
            with ui.column().classes("runner-main"):
                bar = CommandBar(COMMAND, lambda: COMMAND.start(cli_bridge.resume_argv(JOURNEY), "resume")) if LIVE else None
                centre()
            with ui.column().classes("runner-rail"):
                panel = JourneyPanel(viewer, compact=journey == "story")

    # ---------- footer: replay control (Recorded) or follow state (Live) ----------
    stops = sorted(set(snaps) | set(viewer.refused))
    with ui.footer().classes("bg-white text-black border-t px-4 py-1"):
        with ui.row().classes("w-full items-center no-wrap gap-3"):
            if LIVE:
                ui.icon("sensors", color="positive")
                ui.label(f"following {JOURNEY}").classes("text-xs mono")
                slider = None
            else:
                play_btn = ui.button(icon="pause" if replay["playing"] else "play_arrow", on_click=lambda: toggle_play()).props(
                    "flat round dense"
                )
                ui.toggle({10: "10x", 60: "60x", 600: "600x"}, value=60, on_change=lambda e: replay.update(speed=e.value)).props(
                    "dense size=sm"
                )
                ui.button(icon="skip_previous", on_click=lambda: jump(-1)).props("flat round dense").tooltip("Previous pass end")
                slider = ui.slider(min=1, max=max(len(items), 1), value=viewer.cursor,
                                   on_change=lambda e: replay.get("syncing") or set_cursor(int(e.value), from_slider=True)).classes("grow")  # fmt: skip
                ui.button(icon="skip_next", on_click=lambda: jump(1)).props("flat round dense").tooltip("Next pass end")
            position = ui.label("").classes("text-xs text-grey-7 mono whitespace-nowrap text-right")

    def toggle_play() -> None:
        replay["playing"] = not replay["playing"]
        if replay["playing"] and viewer.cursor >= len(items):
            set_cursor(1)
        play_btn.props(f"icon={'pause' if replay['playing'] else 'play_arrow'}")

    def jump(direction: int) -> None:
        target = [s for s in stops if s > viewer.cursor] if direction > 0 else [s for s in stops if s < viewer.cursor]
        if target:
            set_cursor(target[0] if direction > 0 else target[-1])

    def refresh_chrome() -> None:
        text, color = status_line()
        status_chip.text = text
        status_chip.props(f"color={color}")
        item = items[viewer.cursor - 1]
        gap = viewer.now() - item.ts
        clock.text = f"last event {item.time} UTC · {human(gap)} ago"
        panel.show_follow(replay["playing"])
        position.text = f"event {viewer.cursor}/{len(items)} · {item.ts:%H:%M:%S} · pass {item.pass_no}"
        targets = target_summary(viewer.snapshot())
        title.text = f"{sample} · {targets}"
        if runs_status is not None:
            runs_status.text = text
            runs_title.text = f"{sample} · {targets}"
        if views.exec_wait is not None and pass_open(items, viewer.cursor):
            try:
                views.exec_wait.text = f"Parabricks is running · last Journey event {human(gap)} ago"
            except Exception:  # element was rebuilt
                pass

    def set_cursor(c: int, from_slider: bool = False) -> None:
        if not items:
            return
        c = max(1, min(c, len(items)))
        if not from_slider:
            viewer.sim = items[c - 1].ts
        elif c != viewer.cursor:
            viewer.sim = items[c - 1].ts
        old_key = replay["key"]
        viewer.cursor = c
        if slider is not None and int(slider.value) != c:
            replay["syncing"] = True
            slider.value = c
            replay["syncing"] = False
        panel.update()
        line, _ = snapshot_at(snaps, c)
        item = items[c - 1]
        execution_steps = sum(i.event_type == "ExecutionPathRecorded" for i in items[:c])
        # Record pages include links and chronological context from the current prefix. Review forms keep
        # the coarser key below so quiet activity does not discard an in-progress researcher response.
        record_prefix = c if section in ("evidence", "finding") else None
        replay["key"] = (line, item.phase, pass_open(items, c), execution_steps, bool(waiting_review(items, snaps, c)),
                         viewer.artifacts is not None, record_prefix)  # fmt: skip
        if replay["key"] != old_key:
            render_stage_bar()
            centre.refresh()
        if bar is not None:
            bar.update(viewer.snapshot())
        refresh_chrome()

    times = [i.ts for i in items]  # grows with items in live mode

    def tick() -> None:
        if LIVE or not replay["playing"] or not items:
            return
        viewer.sim += timedelta(seconds=0.25 * replay["speed"])
        c = bisect.bisect_right(times, viewer.sim)
        if c >= len(items):
            replay["playing"] = False
            play_btn.props("icon=play_arrow")
            c = len(items)
        if c != viewer.cursor:
            old_sim = viewer.sim
            set_cursor(c)
            viewer.sim = old_sim
        refresh_chrome()

    if LIVE:
        tail, feed = JourneyTail(JOURNEY), LiveFeed(JOURNEY)
        # `artifacts`: the page's one `run-artifacts` read is reserved before it is awaited, so a pass that closes
        # while it runs starts no second read. `ended`: the app's commands this page has seen end. `observing` and
        # `again`: one `observe` at a time, and whether another was asked for while it ran.
        reading = {"busy": False, "artifacts": False, "ended": COMMAND.ended, "observing": False, "again": False}

        async def observe_now() -> None:
            """The page's one observer (pass ends and command exits both ask it): reads run one after another, so an
            older answer never lands after a newer one, and a request while one runs is served by one more read."""
            if reading["observing"]:
                reading["again"] = True
                return
            reading["observing"], reading["again"] = True, True
            try:
                while reading["again"]:
                    reading["again"] = False
                    await observe_once()
            finally:
                reading["observing"] = False

        async def observe_once() -> None:
            snapshot, refusal = await cli_bridge.observe(JOURNEY)
            if snapshot is None:
                viewer.refused[len(items)] = refusal or "refused"
            else:
                snaps[len(items)] = snapshot
                grounded = snapshot.get("grounded_report")  # absent until the report projection exists
                if not reading["artifacts"] and grounded and grounded["status"] == "published":
                    reading["artifacts"] = True  # once: a finished Run's files stay put
                    background_tasks.create(read_artifacts())  # its own task: later observes need not wait for it
            set_cursor(len(items))

        async def read_artifacts() -> None:
            viewer.artifacts = await cli_bridge.run_artifacts(JOURNEY)
            set_cursor(len(items))

        async def read_feed() -> None:
            """Tell the tailed lines until every one has its item; append only the new rows (or rebuild once, when the
            feed refuses and the narrator takes over)."""
            reading["busy"] = True
            try:
                while len(items) < len(viewer.raw):
                    told = await run.io_bound(feed.items, list(viewer.raw))
                    if feed.refused != viewer.feed_refused:
                        viewer.feed_refused = feed.refused
                        items[:], times[:] = told, [i.ts for i in told]
                        panel.show_source()
                        panel.rebuild()
                        new = told
                    else:
                        new = told[len(items) :]
                        items.extend(new)
                        times.extend(i.ts for i in new)
                    if not new:
                        break  # the feed has not caught up with the tail; the next tailed line asks again
                    viewer.sim = items[-1].ts
                    set_cursor(len(items))
                    if any(i.event_type == "PassClosed" for i in new):
                        background_tasks.create(observe_now())
            finally:
                reading["busy"] = False

        def poll() -> None:
            lines = tail.poll()
            viewer.raw.extend(lines)
            if lines and not reading["busy"]:
                background_tasks.create(read_feed())
            if COMMAND.ended != reading["ended"]:
                # A command ended: the next observe says where the Run is now, whatever its exit code said.
                reading["ended"] = COMMAND.ended
                if (outcome := COMMAND.outcome) is not None:
                    ui.notify(f"{COMMAND.name}: {outcome['code']}", type="negative")
                background_tasks.create(observe_now())
            bar.update(viewer.snapshot())

        ui.timer(0.5, poll)
        ui.timer(1.0, lambda: items and refresh_chrome())
    else:
        ui.timer(0.25, tick)
        set_cursor(viewer.cursor)


# The storage secret only signs this browser's id for `app.storage.user` (the "Answering as" name) on a local app.
# Localhost only by default: in live mode the decision buttons run commands with the Run's credentials, and the app
# has no login. HOST=0.0.0.0 exposes it (a recorded replay, or a network you trust).
ui.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8770)), show=False, reload=False, title="BioNeb",
       storage_secret=os.environ.get("STORAGE_SECRET", "bioneb-gui-local"))  # fmt: skip
