"""The persistent Journey: nested stages or Timeline and Passes views of retained feed items.

Rows are appended in place as the cursor moves (no virtual DOM, so no full rebuild per event); moving the cursor
back, or switching "all events", rebuilds the feed once.
"""

from __future__ import annotations

import json
from datetime import timedelta

from nicegui import ui

from journey_feed import DISPLAY_STAGES, FeedItem, display_part, display_stage
from run_reading import Viewer, fmt_size, human


class JourneyPanel:
    """Built inside the page's Journey rail; `update()` appends through the viewer's cursor."""

    def __init__(self, viewer: Viewer, *, compact: bool = False) -> None:
        self.viewer = viewer
        self.compact = compact
        self.show_all = False
        self.rendered = 0
        self.pass_boxes: dict = {}
        self.pass_phases: dict = {}
        self.top_phase = None
        self.last_ts = None
        with ui.row().classes("w-full items-center no-wrap"):
            ui.label("Journey").classes("text-base font-bold")
            ui.space()
            self.live_dot = ui.label("").classes("text-xs")
        if compact:
            controls = ui.expansion("Recording details").classes("w-full text-xs").props("dense duration=0")
        else:
            controls = ui.row().classes("w-full items-center gap-2 no-wrap")
        with controls:
            if not compact:
                ui.toggle({"story": "Story", "timeline": "Timeline", "passes": "Passes"}, value=viewer.variants["journey"],
                          on_change=lambda e: ui.navigate.to(viewer.url(journey=e.value))).props("dense no-caps size=sm")  # fmt: skip
            ui.switch("all events", on_change=lambda e: (setattr(self, "show_all", e.value), self.rebuild())).props(
                "dense size=xs"
            ).classes("text-xs")
        self.counts = ui.label("").classes("text-xs text-grey-7")
        self.fallback = ui.label("").classes("text-xs text-orange-9")
        self.show_source()
        ui.separator()
        self.feed = ui.column().classes("w-full gap-0")
        self.stage_boxes: dict = {}
        self.part_boxes: dict = {}
        if compact:
            self.create_stages()

    def create_stages(self) -> None:
        self.stage_boxes, self.part_boxes = {}, {}
        self.stage_counts: dict[str, int] = {}
        self.stage_key_titles: dict[str, str] = {}
        with self.feed:
            for n, (key, (title, icon)) in enumerate(DISPLAY_STAGES.items(), 1):
                with (
                    ui.expansion()
                    .classes("runner-journey-stage w-full")
                    .props(f'duration=0 data-stage="{key}"')
                    .mark("journey-stage") as stage
                ):
                    with stage.add_slot("header"):
                        with ui.column().classes("gap-1 grow min-w-0"):
                            ui.label(f"{n:02d} · {title}").classes("font-bold")
                            caption = ui.label("No activity recorded yet").classes("runner-stage-caption")
                        ui.icon(icon).classes("text-lg")
                    body = ui.column().classes("w-full gap-1 runner-journey-parts")
                self.stage_boxes[key] = (body, caption)

    def add_compact(self, item: FeedItem) -> None:
        key = display_stage(item, self.viewer.raw[item.n - 1])
        self.stage_counts[key] = self.stage_counts.get(key, 0) + 1
        body, caption = self.stage_boxes[key]
        if item.level in ("key", "step"):
            self.stage_key_titles[key] = item.title
        caption.text = self.stage_key_titles.get(key, f"{self.stage_counts[key]} recorded events · awaiting a key update")
        if not self.visible(item):
            return
        name = display_part(item)
        part_key = (key, name)
        if part_key not in self.part_boxes:
            with body, ui.expansion(name).classes("runner-journey-part w-full").props("duration=0") as part:
                rows = ui.column().classes("w-full gap-1")
            self.part_boxes[part_key] = (part, rows, 0)
        part, rows, count = self.part_boxes[part_key]
        with rows:
            self.row(item, dense=True)
        part.text = f"{name} · {count + 1}"
        self.part_boxes[part_key] = (part, rows, count + 1)

    def show_follow(self, playing: bool) -> None:
        """The header dot: following a live Run, replaying or paused."""
        self.live_dot.text = "● live" if self.viewer.live else ("● replaying" if playing else "○ paused")
        self.live_dot.classes(replace="text-xs " + ("text-positive" if self.viewer.live or playing else "text-grey-6"))

    def show_source(self) -> None:
        """Say so when the rows are the GUI's own narration because the backend's feed refused the Run."""
        refused = self.viewer.feed_refused
        self.fallback.text = f"The backend's Journey feed refused this Run ({refused}); the GUI's own narration is shown."
        self.fallback.set_visibility(refused is not None)

    def visible(self, item: FeedItem) -> bool:
        if self.show_all:
            return True
        allowed = {"story": ("key", "step"), "timeline": ("key", "step", "quiet"), "passes": ("key", "step")}
        return item.level in allowed[self.viewer.variants["journey"]]

    def open_event(self, item: FeedItem) -> None:
        if self.compact:
            ui.navigate.to(self.viewer.url(section="evidence", record=item.n))
            return
        with ui.dialog() as dialog, ui.card().classes("w-[760px] max-w-full"):
            ui.label(item.title).classes("text-lg font-bold")
            ui.label(f"{item.event_type} · line {item.n} · {item.ts:%Y-%m-%d %H:%M:%S} UTC · pass {item.pass_no} · "
                     f"{fmt_size(item.size)}").classes("text-xs text-grey-7 mono")  # fmt: skip
            for d in item.details:
                ui.label(d).classes("text-sm")
            excerpt = ui.column().classes("w-full")

            def show_raw() -> None:
                excerpt.clear()
                with excerpt:
                    text = json.dumps(json.loads(self.viewer.raw[item.n - 1]), indent=1)
                    ui.code(text[:4000] + ("\n… (cut at 4 KB)" if len(text) > 4000 else ""), language="json").classes(
                        "w-full text-xs max-h-96 overflow-auto"
                    )

            with ui.row():
                hint = "payload hidden" if item.size > 20_000 else "raw event"
                ui.button(f"Show {hint} (first 4 KB)", icon="data_object", on_click=show_raw).props("flat no-caps dense")
                ui.button("Close", on_click=dialog.close).props("flat no-caps dense")
        dialog.open()

    def row(self, item: FeedItem, dense: bool = False) -> ui.element:
        bold = "font-bold" if item.level == "key" else ""
        with ui.row().classes("w-full no-wrap items-start gap-2 py-1 px-1 cursor-pointer hover:bg-grey-2 rounded") as r:
            ui.icon(item.icon, color=item.color).classes("text-lg")
            with ui.column().classes("gap-0 grow min-w-0"):
                ui.label(item.title).classes(f"text-sm {bold} {'text-amber-10' if item.needs_you else ''}").mark(
                    "journey-event-title"
                )
                for d in item.details[: (1 if dense else 3)]:
                    ui.label(d).classes("text-xs text-grey-7 break-words")
                if self.viewer.variants["journey"] == "timeline":
                    ui.label(f"{item.event_type}{' · ' + fmt_size(item.size) + ' hidden' if item.size > 20_000 else ''}").classes(
                        "text-[10px] text-grey-6 mono"
                    )
            ui.label(item.time).classes("text-xs text-grey-6 mono")
        r.on("click", lambda i=item: self.open_event(i))
        return r

    def gap_row(self, gap: timedelta) -> ui.element:
        with ui.row().classes("w-full items-center gap-2 py-1 px-1") as r:
            ui.icon("hourglass_empty", color="grey-5").classes("text-sm")
            ui.label(f"{human(gap)} with no new Journey event").classes("text-xs text-grey-6 italic")
        return r

    def pass_box(self, item: FeedItem) -> ui.column:
        key = item.pass_no
        if key not in self.pass_boxes:
            with self.feed:
                with (
                    ui.expansion(f"Pass {key}" if key else "Before the first pass", icon="layers", value=True)
                    .classes("w-full border rounded mb-1")
                    .props("dense header-class='text-sm font-bold'") as exp
                ):
                    col = ui.column().classes("w-full gap-0")
            exp.move(self.feed, target_index=0)
            for other, _, _ in self.pass_boxes.values():
                other.value = False
            self.pass_boxes[key] = (exp, col, item.ts)
        return self.pass_boxes[key]

    def add(self, item: FeedItem) -> None:
        if self.compact:
            self.add_compact(item)
            return
        gap = item.ts - self.last_ts if self.last_ts else timedelta(0)
        if item.level in ("key", "step", "quiet"):
            self.last_ts = item.ts
        if self.viewer.variants["journey"] == "passes":
            exp, col, started = self.pass_box(item)
            phases = self.pass_phases.setdefault(item.pass_no, [])
            if item.level in ("key", "step") and item.phase not in phases:
                phases.append(item.phase)
            where = " → ".join(phases) or item.phase
            if item.event_type == "PassClosed":
                exp.text = f"Pass {item.pass_no} · {where} · {started:%H:%M}–{item.ts:%H:%M} ({human(item.ts - started)})"
            elif item.pass_no and "–" not in exp.text:
                exp.text = f"Pass {item.pass_no} · {where} · running since {started:%H:%M}"
            if not self.visible(item):
                return
            with col:
                r = self.row(item, dense=True)
            r.move(col, target_index=0)
            return
        if not self.visible(item):
            return
        if gap > timedelta(minutes=2):
            with self.feed:
                g = self.gap_row(gap)
            g.move(
                self.feed, target_index=0 if self.viewer.variants["journey"] == "timeline" or self.top_phase != item.phase else 1
            )
        if self.viewer.variants["journey"] == "story" and self.top_phase != item.phase:
            with self.feed:
                header = ui.label(item.phase.upper()).classes("text-xs font-bold text-primary tracking-widest pt-3 pb-1")
            header.move(self.feed, target_index=0)
            self.top_phase = item.phase
        with self.feed:
            r = self.row(item)
        r.move(self.feed, target_index=1 if self.viewer.variants["journey"] == "story" else 0)

    def rebuild(self) -> None:
        self.feed.clear()
        self.pass_boxes, self.top_phase, self.last_ts, self.pass_phases = {}, None, None, {}
        self.rendered = 0
        if self.compact:
            self.create_stages()
        self.update()

    def update(self) -> None:
        if self.viewer.cursor < self.rendered:
            return self.rebuild()
        for item in self.viewer.items[self.rendered : self.viewer.cursor]:
            self.add(item)
        self.rendered = self.viewer.cursor
        shown = sum(self.visible(i) for i in self.viewer.items[: self.viewer.cursor])
        self.counts.text = f"{self.viewer.cursor} events · {shown} shown · model payloads hidden"
