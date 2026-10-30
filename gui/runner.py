"""The default researcher workspace over existing Viewer and command/view seams."""

from __future__ import annotations

from nicegui import ui

import evidence
import report
from journey_feed import DISPLAY_STAGES, PHASE_SECTION, display_stage
from run_reading import REVIEW_SECTION, Viewer, event_payload, fmt_size, health, pass_open, readable_report, review_stage

CSS = """
:root{--runner-ink:#1b302a;--runner-muted:#617268;--runner-line:#dce2d7}
body{background:#f5f5ef;color:var(--runner-ink);font:16px/1.5 Arial,'Helvetica Neue',sans-serif}.nicegui-content{padding:0;gap:0}
.q-btn{letter-spacing:0;text-transform:none;border-radius:8px;font-weight:600}.runner-header{background:#fff;color:#1b302a;border-bottom:1px solid var(--runner-line);box-shadow:none;padding:14px 3vw!important}.runner-header .q-chip{font-size:12px}.runner-header .q-field{font-size:13px}.runner-header .q-btn{font-size:13px}.runner-header-title{font-weight:750;letter-spacing:-.03em;font-size:21px}
.runner-shell{width:100%;max-width:1780px;margin:auto;padding:18px 3vw 70px;gap:20px}.runner-grid{display:grid;grid-template-columns:minmax(0,1fr) 326px;gap:26px;width:100%;align-items:start}.runner-main{min-width:0;width:100%;gap:17px}.runner-rail{position:sticky;top:135px;max-height:calc(100vh - 170px);overflow-y:auto;scrollbar-width:thin;background:#e9efdf;border:1px solid #d6dfca;border-radius:16px;padding:20px;gap:8px;min-width:0}.runner-rail>.q-separator{margin:4px 0}
.runner-nav{background:#e9ede2;border:1px solid #dce2d2;border-radius:10px;padding:5px;display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:4px;width:100%}.runner-nav .q-btn{min-height:39px;font-size:13px;color:#5b7154}.runner-nav .q-btn.active{background:#fff;color:#204d3e;box-shadow:0 2px 5px #223b2710}.runner-nav .q-btn .q-icon{font-size:17px}.runner-nav .needs-review{background:#fae4ac;color:#755315}
.runner-heading{font-size:42px;line-height:1.12;font-weight:750;letter-spacing:-.045em}.runner-section-title{font-size:25px;font-weight:700;letter-spacing:-.03em}.runner-kicker{font-size:11px;font-weight:750;letter-spacing:.09em;color:#6b7d68}.runner-muted{font-size:13px;color:var(--runner-muted);line-height:1.5;overflow-wrap:anywhere}.runner-question{font-size:18px;color:#566e5d;line-height:1.5}.runner-back{font-size:14px;font-weight:600;color:#204d3e;text-decoration:none;margin:5px 0}.runner-summary{padding:27px 29px;background:white;border:1px solid #d9e1d1;border-left:4px solid #4b7550;border-radius:17px;gap:17px}.runner-counts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;width:100%}.runner-count{padding:15px 17px;border:1px solid var(--tone-line);background:var(--soft);color:var(--tone);border-radius:11px;min-height:88px}.runner-count .q-btn__content{flex-direction:column;align-items:flex-start;gap:5px;width:100%}.runner-number{font-size:31px;font-weight:750;line-height:1.1;letter-spacing:-.04em}.runner-count-label{font-size:12px;font-weight:650;text-align:left;line-height:1.3}.runner-limit{font-size:13px;color:#796635;line-height:1.5}
.tone-observed{--tone:#13734e;--soft:#edf8f2;--tone-line:#c6e5d4}.tone-not-carried{--tone:#b13e57;--soft:#fff1f3;--tone-line:#f0cdd4}.tone-uncertain{--tone:#94620c;--soft:#fff8e9;--tone-line:#eeddb5}
.runner-results{border:1px solid var(--runner-line);background:#fff;border-radius:12px;overflow:hidden}.runner-finding{width:100%;border-bottom:1px solid #e4e9df}.runner-finding:last-child{border:0}.runner-finding>.q-expansion-item__container>.q-item{min-height:65px;padding:12px 17px}.runner-finding>.q-expansion-item__container>.q-expansion-item__content{padding:18px 20px;background:#f7f8f3}.runner-finding-header{display:flex;align-items:center;gap:15px;width:100%;min-width:0}.runner-snp{font-size:19px;font-weight:700;letter-spacing:-.025em;color:var(--tone);min-width:150px}.runner-outcome{font-size:13px;font-weight:650;line-height:1.3;padding:5px 8px;background:var(--soft);color:var(--tone);border:1px solid var(--tone-line);border-radius:5px}.runner-allele{padding:12px 0;border-bottom:1px solid #dce3d6}.runner-measurements{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;width:100%;padding:12px;background:#edf1e7;border-radius:8px}.runner-filters .q-field{font-size:14px;width:215px}.runner-filters .q-field__control{background:white;border-radius:8px}
.runner-journey-stage{border-left:1px solid #b9cdae;margin-left:8px;width:calc(100% - 8px)!important;padding:0 0 15px}.runner-journey-stage:last-child{border-left-color:transparent}.runner-journey-stage>.q-expansion-item__container>.q-item{padding:0 0 0 14px;min-height:0;font-size:15px;gap:5px}.runner-journey-stage>.q-expansion-item__container>.q-item>.q-item__section--side{padding-left:0}.runner-journey-stage>.q-expansion-item__container>.q-expansion-item__content{padding:10px 0 5px 14px}.runner-stage-caption{font-size:12px;line-height:1.4;color:#62745b;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.runner-journey-part>.q-expansion-item__container>.q-item{font-size:13px;padding:8px 0;min-height:35px;border-top:1px solid #ccd8c2;font-weight:600}.runner-journey-part>.q-expansion-item__container>.q-item>.q-item__section--side{padding-left:5px}.runner-journey-part .q-icon{font-size:17px}.runner-journey-parts .q-expansion-item__content .text-sm{font-size:13px}.runner-rail .text-xs{font-size:11px}.runner-rail .text-base{font-size:22px}.runner-rail .q-row .q-icon.text-lg{font-size:16px}.runner-rail .runner-journey-part .q-row .q-icon.text-lg{font-size:15px}
.evidence-block{padding:23px;background:white;border:1px solid var(--runner-line);border-radius:13px;overflow-wrap:anywhere}.evidence-label{width:170px;flex-shrink:0;font-size:13px;color:#617268}.evidence-value{font-size:16px;min-width:0;overflow-wrap:anywhere}.evidence-output{padding:12px 0;border-top:1px solid #dce2d7;width:100%}.evidence-technical{margin-top:10px}.evidence-technical .q-code{max-width:100%}.runner-main .q-card{box-shadow:none;border-color:var(--runner-line);border-radius:12px}.runner-main .q-expansion-item{max-width:100%}
.runner-rail{top:var(--rail-top,135px);max-height:calc(100vh - var(--rail-top,135px) - 25px)}body:has(.runner-follow){--rail-top:170px}
.report-md table{display:block;overflow-x:auto;border-collapse:collapse}.report-md th,.report-md td{min-width:7em;vertical-align:top;padding:6px 8px;border:1px solid var(--runner-line);font-size:14px}
@media(max-width:1450px){.runner-header .runner-clock{display:none}}
@media(max-width:1200px){body{--rail-top:180px}body:has(.runner-follow){--rail-top:215px}}
@media(max-width:900px){body{--rail-top:225px}body:has(.runner-follow){--rail-top:260px}}
@media(max-width:1200px){.runner-shell{padding:16px 22px 65px}.runner-grid{grid-template-columns:minmax(0,1fr) 285px;gap:18px}.runner-heading{font-size:35px}.runner-count{padding:13px 10px}.runner-count-label{font-size:11px}.runner-snp{min-width:120px}.runner-nav .q-btn{font-size:11px;padding:6px}.runner-nav .q-btn .q-icon{font-size:14px}.runner-rail{padding:16px}}
@media(max-width:900px){.runner-grid{grid-template-columns:minmax(0,1fr) 250px;gap:12px}.runner-counts{grid-template-columns:repeat(2,minmax(0,1fr))}.runner-finding-header{flex-wrap:wrap;gap:7px}.runner-nav{grid-template-columns:repeat(3,minmax(0,1fr))}.runner-header .runner-clock{display:none}.runner-measurements{grid-template-columns:repeat(2,minmax(0,1fr))}.evidence-field{flex-wrap:wrap!important;gap:3px!important}.evidence-label{width:100%}}
@media(max-width:640px){.runner-grid{grid-template-columns:1fr}.runner-rail{position:relative;top:auto;grid-row:1;max-height:300px}.runner-shell{padding:12px 12px 70px}.runner-header{padding:10px!important}.runner-header .q-row{flex-wrap:wrap!important}.runner-heading{font-size:30px}}
"""


def review_section(review: dict, current: str) -> str:
    """The display section a review is asked on: its own (`REVIEW_SECTION`), else its stage's."""
    return REVIEW_SECTION.get(review["kind"]) or PHASE_SECTION.get(review_stage(review, current), "request")


def selected_section(viewer: Viewer, section: str) -> str:
    if section in DISPLAY_STAGES or section in ("evidence", "finding"):
        return section
    if not viewer.view and viewer.review() and viewer.review()["kind"] in REVIEW_SECTION:
        return REVIEW_SECTION[viewer.review()["kind"]]
    phase = viewer.view or (review_stage(viewer.review(), viewer.current_stage()) if viewer.review() else viewer.current_stage())
    grounded = (viewer.snapshot() or {}).get("grounded_report") or {}
    return "findings" if phase == "Results" and grounded.get("status") == "published" else PHASE_SECTION.get(phase, "request")


def navigation(viewer: Viewer, section: str) -> None:
    chosen = selected_section(viewer, section)
    actual = PHASE_SECTION.get(viewer.current_stage(), "request")
    review = viewer.review()
    waiting = review_section(review, viewer.current_stage()) if review else None
    seen = {display_stage(i, viewer.raw[i.n - 1]) for i in viewer.items[: viewer.cursor]}
    with ui.element("nav").classes("runner-nav"):
        for key, (title, icon) in DISPLAY_STAGES.items():
            control = (
                ui.button(
                    title,
                    color=None,
                    icon=icon,
                    on_click=lambda k=key: ui.navigate.to(viewer.url(section=k, view=None, record=None)),
                )
                .props("flat no-caps")
                .classes("active" if key == chosen else "")
            )
            control.props(f'data-backend-phase="{viewer.current_stage()}"')
            if key == waiting:
                control.mark("needs-you").classes("needs-review")
            elif key == actual:
                control.mark("current")
            elif key in seen and key != "health":
                control.mark("done")


HEALTH_NUMBERS = (("Reads", "reads", "{:,}"), ("Mapped", "mapped_percent", "{:.2f} %"), ("Mean depth", "mean_depth", "{:.1f}x"),
                  ("Ti/Tv", "ti_tv", "{:.2f}"), ("Covered contigs", "covered_contigs", "{:,}"),
                  ("Covered bases", "covered_bases", "{:,}"), ("Covered fraction", "covered_fraction", "{:.1%}"))  # fmt: skip


def run_health(viewer: Viewer, views) -> None:
    """The Run's health as the Journey records it (#414): its one measurement, or why it was not measured, each failed
    measurement and its Review; the pending Review's decision when `observe` asks it."""
    facts = health(viewer.items[: viewer.cursor], viewer.raw)
    review = viewer.review()
    if not facts and not (review and review["kind"] == "run_health"):
        ui.label("No Run health measurement is recorded in this prefix.").classes("runner-muted")
    for item, p in facts:
        with ui.column().classes("evidence-block w-full gap-2").mark("run-health"):
            if item.event_type == "RunHealthMeasured":
                numbers = p.get("measurement")
                if numbers:
                    ui.label("Run health measured").classes("text-lg font-bold")
                    for label, key, shape in HEALTH_NUMBERS:
                        value = numbers.get(key)
                        evidence.field(label, shape.format(value) if value is not None else None)
                    if numbers.get("ti_tv") is None:
                        evidence.field("Ti/Tv", f"not measured ({numbers['ti_tv_reason'].replace('_', ' ')})")
                else:
                    ui.label("Run health not measured").classes("text-lg font-bold")
                    missing = p["not_measured"]
                    evidence.field("Why", f"{missing['code'].replace('_', ' ')}: {missing['detail']}")
            elif item.event_type == "RunHealthMeasurementFailed":
                ui.label("A health measurement failed").classes("text-lg font-bold")
                evidence.field("Fault", f"{p['code'].replace('_', ' ')}: {p['detail']}")
            elif item.event_type == "RunHealthReviewRequested":
                ui.label("Review asked: retry the measurement or abort").classes("text-lg font-bold")
            else:
                ui.label(f"Review answered: {p['response'].get('action', '?')}").classes("text-lg font-bold")
            ui.link("Inspect the record", evidence.record_link(viewer, item))
    if review and review["kind"] == "run_health":
        views.review_card(review)


def show_health(viewer: Viewer, views) -> None:
    ui.label("Health").classes("runner-heading")
    run_health(viewer, views)
    ui.label("Recorded output checks").classes("runner-muted")
    found = False
    for item in viewer.items[: viewer.cursor]:
        if item.event_type != "ExecutionPathRecorded":
            continue
        payload = event_payload(viewer.raw, item)
        if payload.get("event_type") != "H100ArtifactsVerified":
            continue
        found = True
        with ui.column().classes("evidence-block w-full gap-3"):
            for pointer in payload.get("payload", {}).get("pointers", []):
                if pointer.get("role") in ("aligned_bam", "VCF"):
                    evidence.field("Output", pointer["role"])
                    if pointer.get("bytes") is not None:
                        evidence.field("Size", fmt_size(pointer["bytes"]))
                    evidence.field("Grounding", pointer.get("grounding"))
            ui.link("Inspect the output-check record", evidence.record_link(viewer, item))
    if not found:
        ui.label("No output checks are recorded in this prefix.")
    ui.label("Output verification does not establish an overall run-health assessment.").classes("runner-limit")


def show_finding(viewer: Viewer, focus: str | None) -> None:
    grounded = (viewer.snapshot() or {}).get("grounded_report") or {}
    data = grounded.get("report") if grounded.get("status") == "published" else None
    snp = next(
        (
            s
            for s in (data or {}).get("collection_view", {}).get("snps", [])
            if (s["requested_rsid"] or s["request_text"]) == focus
        ),
        None,
    )
    ui.link("← All results", viewer.url(section="findings", focus=None, record=None)).classes("runner-back")
    ui.label(focus or "Finding").classes("runner-heading")
    if snp is None:
        ui.label("No published finding for this target is available at this point.")
        return
    for row in dict(report.finding_groups(data))[focus]:
        with ui.column().classes("evidence-block w-full gap-2"):
            for name in ("where", "genotype", "finding", "reading", "limitation"):
                evidence.field(name.capitalize(), row[name])
    ui.label("Recorded path for this finding").classes("runner-section-title")
    ids = {i["work_item_id"] for i in snp["items"]}
    names = {focus, snp.get("verified_rsid")}
    matches = []
    for item in viewer.items[: viewer.cursor]:
        if item.event_type not in {
            "RefSnpToolResultRecorded",
            "InvestigationEvidenceRecorded",
            "InvestigationItemWindowRecorded",
            "InvestigationTableChunkRead",
            "InvestigationItemEvidenceCollected",
        }:
            continue
        p = event_payload(viewer.raw, item)
        work = p.get("work_item") or {}
        linked = {p.get("work_item_id"), work.get("work_item_id")}
        linked.update(i.get("work_item_id") for i in p.get("items", []))
        linked.update(i.get("work_item_id") for i in p.get("readings", []))
        if ids & linked or (p.get("query") is not None and p["query"] in names):
            matches.append(item)
            ui.link(f"{item.time} · {item.title}", evidence.record_link(viewer, item))
    if not matches:
        ui.label("No per-target activity is available in this recorded prefix.").classes("runner-muted")


def show_report(viewer: Viewer, views) -> None:
    """The Run's own `report.md`, as written (owner, 2026-10-30), then the Grounded report it is drawn from."""
    ui.label(DISPLAY_STAGES["report"][0]).classes("runner-heading")
    page, line = readable_report(viewer.journey, viewer.items[: viewer.cursor], viewer.raw)
    ui.label(line).classes("runner-muted").mark("report-md-status")
    if page is not None:
        with ui.column().classes("evidence-block report-md w-full").mark("report-md"):
            ui.markdown(page, extras=["tables", "fenced-code-blocks", "code-friendly"])  # keep NC_000019 underscores
    review = viewer.review()
    with ui.expansion("Grounded report and its review", icon="fact_check", value=page is None).classes("w-full"):
        views.render("Results", review)


def main(viewer: Viewer, views, *, section: str, record: int | None, focus: str | None, density: str, state: dict) -> None:
    chosen = selected_section(viewer, section)
    if chosen == "evidence":
        evidence.show(viewer, record)
        return
    if chosen == "finding":
        show_finding(viewer, focus)
        return
    if chosen == "health":
        show_health(viewer, views)
        return
    if chosen == "report":
        show_report(viewer, views)
        return
    grounded = (viewer.snapshot() or {}).get("grounded_report") or {}
    if (
        chosen == "findings"
        and density == "simple"
        and grounded.get("status") == "published"
        and viewer.variants["findings"] != "table"
    ):
        published = next((i for i in reversed(viewer.items[: viewer.cursor]) if i.event_type == "GroundedReportPublished"), None)
        report.compact(grounded, viewer.artifacts, published, live=viewer.live, viewer=viewer, state=state)
        return
    phase = {"request": "Question", "investigation": "Investigation", "findings": "Results", "report": "Results"}.get(chosen)
    if chosen == "parabricks":
        current = viewer.view or (
            review_stage(viewer.review(), viewer.current_stage()) if viewer.review() else viewer.current_stage()
        )
        phase = current if current in ("Plan", "Approval", "Execution") else "Execution"
        with ui.row().classes("w-full items-center gap-2"):
            for name in ("Plan", "Approval", "Execution"):
                ui.button(name, on_click=lambda p=name: ui.navigate.to(viewer.url(section="parabricks", view=p))).props(
                    "flat no-caps"
                )
    ui.label(DISPLAY_STAGES[chosen][0]).classes("runner-heading")
    if not viewer.snapshot() and phase != "Question":
        with ui.column().classes("evidence-block w-full gap-3"):
            working = pass_open(viewer.items, viewer.cursor) and phase == viewer.current_stage()
            title = "Planning in progress" if working and phase == "Plan" else "Waiting for a recorded stage summary"
            ui.label(title).classes("runner-section-title")
            ui.label(
                "The Journey continues to record the path while this step runs."
                if working
                else "No stage summary is available at this point in the recording."
            ).classes("runner-question")
            ui.label("Open a Journey stage to inspect the steps already recorded.").classes("runner-muted")
        return
    views.render(phase or viewer.current_stage(), viewer.review())
