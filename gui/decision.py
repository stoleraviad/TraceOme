"""The decision element for one pending review, in the Offer or the Form variant (the handbook's element picks).

Its actions and fields come from the review's `legal_actions[].response_schema` and ids from the request
(`responses.py`). Following a live Run, a button starts the `answer` command (detached, through the app's `Command`) once the
action's schema accepts the response, and every decision button is disabled while a command runs. A recorded replay
shows a past moment of a Run the GUI does not follow, so there a button only shows the exact command and the schema's
verdict (a dry run).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from nicegui import ui

import cli_bridge
from commands import Command
from responses import (
    FILE_SCHEMAS,
    Field,
    build_response,
    file_action,
    names_answerer,
    new_target,
    new_target_kinds,
    open_fields,
    operation,
    problems,
    report_members,
    target_rows,
    union_kinds,
)
from run_reading import allele, consequence, members, question_text

PRIMARY_ACTIONS = ("approve", "select")
SECONDARY_ACTIONS = {"revise": "Request changes", "clarify": "Clarify", "amend": "Amend", "correct": "Correct"}
LOW_ACTIONS = ("reject", "stop", "abort")


class Answer:
    """The researcher's part of one action's response: one input per field its schema leaves open, built in the
    current NiceGUI container. `skip` names fields another element of the decision already answers. Each input is kept
    as a reader of its value; `values(by)` reads them all, `by` filling attribution fields of nested answers."""

    def __init__(self, review: dict, action: dict, skip: tuple[str, ...] = ()) -> None:
        self.review = review
        self.action = action
        self.inputs = {f.name: self.field(f) for f in open_fields(review, action) if f.name not in skip}

    def field(self, f: Field) -> Callable[[str | None], object]:
        label = f"{f.prop.get('title', f.name)} ({'required' if f.required else 'optional'})"
        req = self.review["request"]
        if f.name in FILE_SCHEMAS:
            return self.file(f.name)
        if f.name == "operations" and "oneOf" in f.prop.get("items", {}):
            return self.operations()
        if "discriminator" in f.prop.get("items", {}):
            return self.changes(f.name)
        if f.name == "work_item_ids" and req.get("proposal"):
            # After a partial approval the request names the work items still waiting; only those can be approved.
            items = self.work_items()
            if (remaining := req.get("remaining_work_item_ids")) is not None:
                items = {w: n for w, n in items.items() if w in remaining}
            element = (
                ui.select(items, multiple=True, value=list(items), label=f"Targets to observe ({label})")
                .classes("w-96")
                .mark("field-work_item_ids")
            )
        elif "enum" in f.prop:
            names = {c["candidate_id"]: c["display_label"] for c in req.get("candidates", [])}
            element = ui.select({v: names.get(v, v) for v in f.prop["enum"]}, value=f.prop["enum"][0], label=label).classes(
                "w-80"
            )
        elif f.name == "field" and (amendable := (req.get("request") or req.get("review") or {}).get("amendable")):
            element = ui.select(amendable, value=None, label=f"Field to amend ({label})").classes("w-80")
        elif f.name == "member_id" and req.get("basis"):
            options = dict(report_members(self.review))
            element = ui.select(options, value=next(iter(options), None), label=f"Report member ({label})").classes("w-full")
        elif f.prop.get("type") == "integer":
            number = ui.number(label, precision=0, min=f.prop.get("exclusiveMinimum", f.prop.get("minimum"))).classes("w-48")
            return lambda by: None if number.value is None else int(number.value)
        elif f.prop.get("type") == "boolean":
            element = ui.checkbox(label)
        elif f.prop.get("type") == "string":
            element = ui.textarea(label).classes("w-full").props("outlined autogrow").mark(f"field-{f.name}")
        else:
            # Objects, lists and unions the form cannot lay out: their JSON, checked against the schema on preview.
            text = ui.textarea(f"{label}, as JSON").classes("w-full mono").props("outlined autogrow").mark(f"field-{f.name}")
            return lambda by: parsed(text.value)
        return lambda by: element.value

    def file(self, name: str) -> Callable[[str | None], dict]:
        """A file the command names (owner, #430 Q1 A): its content answered like a response, from the copied schema."""
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label("Written to a file outside the repository; the command names it.").classes("text-xs text-grey-7")
            inner = Answer(self.review, file_action(name))
        return lambda by: build_response(self.review, file_action(name), inner.values(by), by=by)

    def changes(self, name: str) -> Callable[[str | None], list]:
        """A list of union items (a correction's requested changes): one row per item, each answered by its kind's own
        schema; the researcher adds rows."""
        kinds = dict(union_kinds(self.action, name))
        rows: list[dict] = []
        ui.label(f"{name.replace('_', ' ').capitalize()}: one row each").classes("text-sm font-bold pt-2")
        column = ui.column().classes("w-full gap-1")

        def add() -> None:
            with column, ui.row().classes("w-full items-start no-wrap gap-2"):
                kind = ui.select(list(kinds), value=next(iter(kinds)), label="Change").classes("w-64").mark("change-kind")
                box = ui.column().classes("grow")
            entry = {"kind": kind}

            def lay_out() -> None:
                box.clear()
                with box:
                    entry["answer"] = Answer(self.review, kind_action(entry))

            kind.on_value_change(lambda _: lay_out())
            lay_out()
            rows.append(entry)

        def kind_action(entry: dict) -> dict:
            return {"action": entry["kind"].value, "response_schema": kinds[entry["kind"].value]}

        add()
        ui.button("Add a change", icon="add", on_click=add).props("flat dense no-caps")
        return lambda by: [build_response(self.review, kind_action(e), e["answer"].values(by), by=by) for e in rows]

    def work_items(self) -> dict[str, str]:
        """Each work item named by its target; a merged rsID shares its target's item, which keeps the target's name."""
        names: dict[str, str] = {}
        for m in members({"pending_reviews": [self.review]}):
            for w in m["work_items"]:
                names.setdefault(w["work_item_id"], f"{m['rsid']} {allele(w['target'])}")
        return names

    def operations(self) -> Callable[[str | None], list]:
        """The per-target operations editor: one row per work item (one per ALT), and one per requested target with
        no work item; each row offers the operations the schema allows for it."""
        ui.label("Operations: one per work item (one per ALT)").classes("text-sm font-bold pt-2")
        names = self.work_items()
        rows = []
        for row in target_rows(self.review, self.action):
            with ui.row().classes("items-center gap-2"):
                ui.label(names.get(row.id) or f"{row.rsid}: no work item").classes("w-72 mono")
                if row.key == "work_item_id":
                    choice = ui.select(row.choices, value=row.choices[0]).classes("w-44")
                else:
                    # A target without a work item stays as it is unless the researcher picks an operation for it.
                    options = {None: "leave unchanged", **{c: c for c in row.choices}}
                    choice = ui.select(options, value=None).classes("w-44").mark(f"operation-{row.rsid}")
                text = ui.input("Corrected request").classes("w-72").mark(f"request-{row.rsid}")
                text.bind_visibility_from(choice, "value", backward=lambda v, r=row: v in r.needs_request)
            rows.append((row, choice, text))
        # Kinds that name no existing target (`add`): a new target per row the researcher adds, with its request.
        added: list[tuple[str, ui.input]] = []
        column = ui.column().classes("w-full gap-1")
        for kind in new_target_kinds(self.action):

            def add(kind: str = kind) -> None:
                with column, ui.row().classes("items-center gap-2"):
                    ui.label(f"new target ({kind})").classes("w-72 mono")
                    added.append((kind, ui.input("Request, e.g. Does NA00001 carry rs99?").classes("w-96").mark("new-target")))

            ui.button("Add a target", icon="add", on_click=add).props("flat dense no-caps")
        return lambda by: (
            [op for row, choice, text in rows if (op := operation(row, choice.value, text.value))]
            + [op for kind, text in added if (op := new_target(kind, text.value or ""))]
        )

    def values(self, by: str | None = None) -> dict:
        return {name: read(by) for name, read in self.inputs.items()}


def parsed(text: str | None):
    """A JSON field's value; text that is not JSON stays text, so the schema check names the field."""
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


class Decision:
    """Shows a review's decision element in the page's `decision=` variant. `command` is the app's running command when
    the page follows a live Run, None in a recorded replay (dry run)."""

    def __init__(
        self, journey: Path, variant: str, answering_as: Callable[[], str], answer_files: Path, command: Command | None
    ) -> None:
        self.journey = journey
        self.variant = variant
        self.answering_as = answering_as
        self.answer_files = answer_files  # where the files a command names are written, outside the repository
        self.command = command
        self.send = "Preview command" if command is None else "Send answer"

    def show(self, review: dict) -> None:
        (self.form if self.variant == "form" else self.offer)(review)

    def button(self, *args, **kwargs) -> ui.button:
        """A decision button: disabled while the app runs a command."""
        button = ui.button(*args, **kwargs)
        if self.command is not None:
            button.bind_enabled_from(self.command, "idle")
        return button

    def submit(self, review: dict, action: dict, fields: dict) -> None:
        """Start `answer` with the response, once who answers is known and the action's schema accepts it; otherwise
        (or in replay) show the exact command and what the schema refuses."""
        name = self.answering_as()
        if names_answerer(action) and not name:
            ui.notify("Set “Answering as” in the header first: this answer records who gives it.", type="warning")
            return
        if self.command is not None and not self.command.idle:
            ui.notify(f"Wait for {self.command.name} to end.", type="warning")
            return
        fields, refused = self.write_files(fields)
        response = build_response(review, action, fields, by=name)
        argv = cli_bridge.answer_argv(self.journey, review["request_event_id"], response)
        refused += problems(action, response)
        if self.command is not None and not refused:
            self.command.start(argv, f"answer {action['action']}")
            return
        with ui.dialog() as dialog, ui.card().classes("w-[820px] max-w-full"):
            ui.label("Not started:" if self.command else "A recorded replay never answers; this button would start:").classes(
                "font-bold"
            )
            cli = argv.index(cli_bridge.CLI[-1]) + 1  # shown from the CLI's module name on
            shown = " ".join(argv[cli - 1 : cli + 1]) + "".join(
                f" \\\n  {a}" if a.startswith("--") else f" '{a}'" if a.startswith("{") else f" {a}" for a in argv[cli + 1 :]
            )
            ui.code(shown, language="bash").classes("w-full text-xs").mark("argv")
            if refused:
                with ui.column().classes("gap-0 text-negative").mark("schema-problems"):
                    ui.label("The action's response_schema refuses this response:").classes("font-bold")
                    for line in refused:
                        ui.label(line).classes("text-xs mono")
            else:
                ui.label("The action's response_schema accepts this response.").classes("text-xs text-positive")
            if self.command is None:
                ui.label(
                    "Following a live Run, it starts detached in the BioNeb checkout; the next observe shows the result."
                ).classes("text-xs text-grey-7")
            ui.button("Close", on_click=dialog.close).props("flat no-caps")
        dialog.open()

    def write_files(self, fields: dict) -> tuple[dict, list[str]]:
        """Write each file the response names (finalization correct/stop) and put its path in its place. A file its
        schema refuses is not written; its path is still shown, with the problems."""
        out, refused = dict(fields), []
        for name in [n for n in fields if n in FILE_SCHEMAS]:
            text = json.dumps(fields[name], indent=1, sort_keys=True) + "\n"
            path = self.answer_files / f"{name.removesuffix('_file')}-{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
            if not (lines := problems(file_action(name), fields[name], where=name)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            refused += lines
            out[name] = str(path.resolve())
        return out, refused

    def preview(self, review: dict, action: dict, box: ui.column, skip: tuple[str, ...] = (), given=dict) -> None:
        """Submit at once when the schema leaves the researcher nothing to enter; otherwise open its fields in `box`."""
        if not [f for f in open_fields(review, action) if f.name not in skip]:
            self.submit(review, action, given())
            return
        box.clear()
        with box, ui.card().props("flat").classes("w-full bg-grey-1"):
            ui.label(f"{action['action'].replace('_', ' ').capitalize()}").classes("text-sm font-bold")
            answer = Answer(review, action, skip)
            self.button(self.send, icon="terminal" if self.command is None else "send",
                        on_click=lambda: self.submit(review, action, {**given(), **answer.values(self.answering_as())})).props("no-caps")  # fmt: skip

    def offer(self, review: dict) -> None:
        """Old Focused-Workspace hierarchy: one primary, a revealing secondary, a quiet reject/stop."""
        kind = review["kind"]
        actions = {a["action"]: a for a in review["legal_actions"]}
        req = review["request"]
        choice = {"candidate": None}
        question, context = question_text(req)
        with ui.card().props("flat bordered").classes("w-full"):
            if question:
                ui.label(question).classes("text-xl font-bold")
                ui.label(context).classes("text-sm text-grey-8")
            if kind == "planning":
                labels = {c["candidate_id"]: c["display_label"] for c in req.get("candidates", [])}
                if labels:
                    choice["candidate"] = next(iter(labels))
                    ui.radio(labels, value=choice["candidate"],
                             on_change=lambda e: choice.update(candidate=e.value)).props("inline")  # fmt: skip
                if not req.get("recommendation"):
                    ui.label("No recommendation from the agent: the evidence does not settle it.").classes("text-xs text-grey-7")
            box = ui.column().classes("w-full")
            with ui.row().classes("w-full items-center gap-2 pt-2"):
                for name in PRIMARY_ACTIONS:
                    if name in actions:
                        label = {
                            "approve": "Approve" + (" and start" if kind == "execution" else ""),
                            "select": "Use this answer",
                        }[name]
                        # The radio answers `select`'s candidate.
                        skip, given = (
                            (("candidate_id",), lambda: {"candidate_id": choice["candidate"]}) if name == "select" else ((), dict)
                        )
                        self.button(label, icon="check",
                                  on_click=lambda a=actions[name], s=skip, g=given: self.preview(review, a, box, s, g)
                                  ).props("no-caps unelevated")  # fmt: skip
                for name, label in SECONDARY_ACTIONS.items():
                    if name in actions:
                        self.button(label, icon="edit",
                                  on_click=lambda a=actions[name]: self.preview(review, a, box)).props("outline no-caps")  # fmt: skip
                # An action this GUI has no place for still gets a plain button (fallback).
                for name in [n for n in actions if n not in {*PRIMARY_ACTIONS, *SECONDARY_ACTIONS, *LOW_ACTIONS}]:
                    self.button(name.replace("_", " ").capitalize(),
                              on_click=lambda a=actions[name]: self.preview(review, a, box)).props("outline no-caps")  # fmt: skip
                ui.space()
                for name in LOW_ACTIONS:
                    if name in actions:
                        self.button(name.capitalize(), on_click=lambda a=actions[name]: self.confirm(a, review)).props(
                            "flat no-caps color=grey-8"
                        )
            with ui.column().classes("gap-0 pt-1"):
                for name in actions:
                    if consequence(kind, actions[name]):
                        ui.label(f"{name.capitalize()}: {consequence(kind, actions[name])}").classes("text-xs text-grey-7")

    def confirm(self, action: dict, review: dict) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-[640px] max-w-full"):
            ui.label(f"{action['action'].capitalize()}?").classes("text-lg font-bold")
            ui.label(consequence(review["kind"], action)).classes("text-sm")
            answer = Answer(review, action)
            with ui.row():
                ui.button("Keep going", on_click=dialog.close).props("flat no-caps")
                self.button(
                    f"Confirm {action['action']}",
                    color="negative",
                    on_click=lambda: (dialog.close(), self.submit(review, action, answer.values(self.answering_as()))),
                ).props("no-caps")
        dialog.open()

    def form(self, review: dict) -> None:
        """Generic: every field comes from the action's JSON response_schema."""
        actions = {a["action"]: a for a in review["legal_actions"]}
        req = review["request"]
        state = {"action": next(iter(actions))}
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label(question_text(req)[0] or f"{review['kind'].capitalize()} review").classes("text-xl font-bold")
            ui.label("Generated from legal_actions[].response_schema; ids are filled from the request.").classes(
                "text-xs text-grey-7"
            )
            ui.toggle(list(actions), value=state["action"], on_change=lambda e: (state.update(action=e.value), fields.refresh())
                      ).props("no-caps").mark("form-action")  # fmt: skip

            @ui.refreshable
            def fields() -> None:
                action = actions[state["action"]]
                for name, value in build_response(review, action, {}).items():
                    if name != "action":
                        ui.label(f"{name} = {value} (from the {'schema' if name in fixed(action) else 'request'})").classes(
                            "text-xs text-grey-7 mono"
                        )
                answer = Answer(review, action)
                self.button(self.send, icon="terminal" if self.command is None else "send",
                            on_click=lambda: self.submit(review, action, answer.values(self.answering_as()))).props("no-caps")  # fmt: skip

            fields()


def fixed(action: dict) -> set[str]:
    return {n for n, p in action["response_schema"].get("properties", {}).items() if "const" in p}
