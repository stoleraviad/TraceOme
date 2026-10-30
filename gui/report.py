"""The Results page: the published Grounded report (`observe`'s `grounded_report`) and the Run's verified files
(`run-artifacts`).

Every value shown is the backend's: the findings are `collection_view`'s rows (one per requested target's work item)
with their admission's own sentence and scope, and the decisions, ended states, limitations and files are listed as
the report and `run-artifacts` give them. The page counts and lays out; it never judges a finding.
"""

from __future__ import annotations

from collections import Counter

from nicegui import ui

from journey_feed import FeedItem

DASH = "–"


def words(code: str) -> str:
    return code.replace("_", " ")


def finding_rows(report: dict) -> list[dict]:
    """One row per requested target's work item, in the researcher's order; a target without a work item keeps one
    row, its finding being its ended state and note. A finding is the item's admission: the admitted claim's
    sentence and scope, or why it was not admitted."""
    return [row for _, rows in finding_groups(report) for row in rows]


def finding_groups(report: dict) -> list[tuple[str, list[dict]]]:
    """`finding_rows` per requested target, in the researcher's order: (the target, its rows)."""
    admissions = {a["value"]["work_item_id"]: a["value"] for a in report["admissions"]}
    # A member the researcher took out of the report at Finalization stays retained, marked not included.
    excluded = {
        m["value"]["work_item_id"] for m in (*report["work_items"], *report["admissions"]) if not m["membership"]["included"]
    }
    groups = []
    for snp in report["collection_view"]["snps"]:
        target = snp["requested_rsid"] or snp["request_text"]
        rows: list[dict] = []
        groups.append((target, rows))
        if not snp["items"]:
            # An ended Run gives every request its state; a note says why, when the backend has one.
            note = snp.get("state_note") or snp["no_work_reason"]
            finding = f"{words(snp['state'])}: {note}" if note else words(snp["state"])
            rows.append({"row": snp["request_member_id"], "target": target, "where": DASH, "genotype": DASH,
                         "finding": finding, "reading": DASH, "limitation": DASH})  # fmt: skip
            continue
        for item in snp["items"]:
            admission = admissions.get(item["work_item_id"])
            if admission is None:
                finding = words(item["outcome"] or item["state"])
            elif admission["status"] == "admitted":
                finding = admission["admitted_claim"]["observation_detail"]
            else:
                finding = f"not admitted: {admission['reason']}"
            if item["state"] != "observed":
                finding = f"{words(item['state'])}: {item.get('state_note') or finding}"
            if item["work_item_id"] in excluded:
                finding = f"excluded from the report: {finding}"
            quality = [f"{k} {item[f]}" for k, f in (("GQ", "genotype_quality"), ("DP", "read_depth")) if f in item]
            reading = item.get("reading")
            if reading:
                reading = words(reading) + (": " + ", ".join(map(words, item["concerns"])) if item.get("concerns") else "")
                reading += f" · {item['reading_note']}" if item.get("reading_note") else ""
            rows.append({
                "row": f"{snp['request_member_id']} {item['work_item_id']}", "target": target, "where": item["target"],
                "genotype": " · ".join([item["genotype"], *quality]) if item["genotype"] else DASH, "finding": finding,
                "reading": reading or DASH, "limitation": admission["claim_scope"] if admission else DASH,
            })  # fmt: skip
    return groups


def not_carried_through(report: dict) -> list[str]:
    """Each requested target and work item the Run did not carry to an observation, with the backend's note."""
    lines = []
    for snp in report["collection_view"]["snps"]:
        target = snp["requested_rsid"] or snp["request_text"]
        if snp["state"] != "candidates":
            lines.append(f"{target}: {words(snp['state'])}" + (f" · {snp['state_note']}" if snp.get("state_note") else ""))
        for item in snp["items"]:
            if item["state"] != "observed":
                note = f" · {item['state_note']}" if item.get("state_note") else ""
                lines.append(f"{target} {item['target']}: {words(item['state'])}{note}")
    return lines


def said(line: dict) -> str:
    """A limitation, conflict or gap: its sentence (`grounded-report/7` keeps it in `text` beside its typed origin)."""
    return line["text"]


def question_text(question: dict) -> str:
    """The researcher's question: their own words (`research_goals`) when the request was drawn from a source, such
    as a paper whose passage is the first target's request text; else the request text."""
    return " ".join(question.get("research_goals") or []) or question["request_text"]


def case_line(case: dict) -> str:
    """A case across targets (`grounded-report/7`, #475): its typed origin's code and the fields it copies. The
    backend's own sentence for it is not in the report, so the GUI lists the typed fields as they are."""
    origin = case["origin"]
    fields = [f"{words(k)} {', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in origin.items() if k != "code"]
    return f"{words(origin['code'])}: " + "; ".join(fields) if fields else words(origin["code"])


def decisions(report: dict) -> list[str]:
    """The human decisions the report records: each issue target's answer and the finalization review."""
    review = report["review_authority"]["response"]
    lines = [f"Report basis: {words(review['action'])} (finalization review)"]
    out = sum(not m["included"] for m in report["report_membership"])
    if out:
        lines.append(f"Report membership: {out} of {len(report['report_membership'])} retained members excluded")
    for item in report.get("issue_items", []):  # left out when the basis has no issue targets
        by = f" by {item['decided_by']}" if item["decided_by"] else ""  # null: the Run ended before an answer
        lines.append(
            f"{item['request']['rsid'] or item['request']['request_text']} ({item['issue']['key']}): {item['decision']}{by}"
        )
    return lines


def size(byte_count: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if byte_count >= scale:
            return f"{byte_count / scale:.1f} {unit}"
    return f"{byte_count} B"


def artifact_rows(document: dict) -> list[dict]:
    """`run-artifacts`' files: the verified outputs, inputs and reference, then the files the Run materialized."""
    rows = [
        {"row": f"{group} {a['role']} {a['location']}", "group": group, "role": a["role"], "location": a["location"],
         "size": size(a["byte_count"]), "sha256": a["sha256"][:12], "proof": f"{a['grounding']} · {a['producer_stage']}"}
        for group in ("outputs", "inputs", "reference") for a in document[group]
    ]  # fmt: skip
    rows += [
        {"row": f"local {a['role']} {a['path']}", "group": "local", "role": a["role"], "location": a["path"],
         "size": size(a["byte_count"]), "sha256": a["sha256"][:12], "proof": a["receipt"]}
        for a in document.get("local", [])
    ]  # fmt: skip
    return rows


def table(rows: list[dict], keys: tuple[str, ...], marker: str) -> None:
    cols = [{"name": k, "label": k.capitalize(), "field": k, "align": "left"} for k in keys]
    ui.table(columns=cols, rows=rows, row_key="row").props("flat bordered dense wrap-cells").classes("w-full").mark(marker)


def show(grounded: dict, artifacts: tuple[dict | None, dict | None] | None, published: FeedItem | None, *,
         live: bool, findings: str = "cards") -> None:  # fmt: skip
    """The published report, then the Run's files (or why they are not shown)."""
    report = grounded["report"]
    # Left out when no requested target reached Investigation (a Run that ended before it): no findings then.
    collection = report.get("collection_view")
    with ui.row().classes("w-full items-center gap-2 bg-green-1 rounded p-2"):
        ui.icon("verified", color="positive")
        ui.label("Grounded report published" + (f" at {published.time}" if published else "")).classes("font-bold")
        ui.label(f"outcome: {words(report['outcome'])}").classes("text-xs text-grey-8")
    with ui.card().props("flat bordered").classes("w-full"):
        if report["question"]:
            ui.label("Your question").classes("text-xs text-grey-7")
            ui.label(question_text(report["question"]["value"])).classes("text-lg")
        if collection:
            work_items = {item["work_item_id"] for snp in collection["snps"] for item in snp["items"]}
            ui.label(f"{len(collection['snps'])} requested targets · {len(work_items)} work items").classes("text-sm")
        else:
            ui.label("This report has no Investigation collection: no requested target reached Investigation.").classes("text-sm")
    if collection:
        ui.label("Findings").classes("text-lg font-bold")
        if findings == "table":  # Cards (B) by default, the table (A) as the variant
            table(finding_rows(report), ("target", "where", "genotype", "finding", "reading", "limitation"), "findings")
        else:
            finding_cards(report)
    context(report, artifacts, live)


def context(report: dict, artifacts, live: bool) -> None:
    """The report's decisions, limits, sources and files, shared by both densities."""
    collection = report.get("collection_view")
    if collection and collection["cases"]:
        with ui.card().props("flat bordered").classes("w-full"):
            ui.label(f"Across targets ({len(collection['cases'])})").classes("font-bold")
            for case in collection["cases"]:
                ui.label(case_line(case)).classes("text-sm")
    with ui.grid(columns=2 if collection else 1).classes("w-full gap-3"):
        with ui.card().props("flat bordered").mark("decisions"):
            ui.label("Decisions").classes("font-bold")
            for line in decisions(report):
                ui.label(line).classes("text-sm")
        if collection:
            with ui.card().props("flat bordered").mark("not-carried"):
                ui.label("Not carried through").classes("font-bold")
                for line in not_carried_through(report) or ["Every requested target was observed."]:
                    ui.label(line).classes("text-sm")
    with ui.card().props("flat bordered").classes("w-full bg-orange-1").mark("limitations"):
        ui.label("Limitations").classes("font-bold")
        for line in report["limitations"]:
            ui.label(f"• {said(line)}").classes("text-sm")
        for line in report["conflicts"]:
            ui.label(f"• Conflict: {said(line)}").classes("text-sm")
    with ui.expansion(f"Gaps ({len(report['gaps'])})", icon="report_gmailerrorred").classes("w-full").props("dense"):
        for line in report["gaps"]:
            ui.label(said(line)).classes("text-sm")
    evidence(report)
    accounting(report)
    files(artifacts, live)
    with ui.expansion("Technical details", icon="tag").classes("w-full").props("dense").mark("technical"):
        for key in ("report_digest", "report_basis_digest", "finalization_basis_digest", "renderer_version"):
            ui.label(f"{words(key)}: {report[key]}").classes("text-xs mono break-all")
        ui.label(f"{len(report['evidence_pointers'])} evidence pointers · "
                 f"{report['inherited_model_contact_count']} model and {report['inherited_source_contact_count']} source "
                 f"contacts inherited · {report['report_external_contact_count']} made by the report").classes("text-xs")  # fmt: skip


OUTCOME_LABELS = {
    "exact_allele_observed": ("observed", "observed"),
    "target_allele_not_carried": ("not carried", "not-carried"),
    "no_record": ("No VCF record", "uncertain"),
}


def compact(grounded: dict, artifacts, published: FeedItem | None, *, live: bool, viewer, state: dict) -> None:
    """Collection overview with per-target disclosure; all claims retain their own scope."""
    report = grounded["report"]
    collection = report.get("collection_view")
    if not collection:
        show(grounded, artifacts, published, live=live)
        return
    snps = collection["snps"]
    unique = {i["work_item_id"]: i for s in snps for i in s["items"]}
    included = {m["value"]["work_item_id"]: m["membership"]["included"] for m in report["work_items"]}
    counts = Counter(i.get("outcome") for wid, i in unique.items() if included.get(wid, True))
    n = counts["exact_allele_observed"]
    with ui.column().classes("runner-summary w-full"):
        with ui.row().classes("w-full items-center"):
            ui.label("RESEARCH RESULTS").classes("runner-kicker")
            ui.space()
            ui.badge("Grounded report published", color="green-1").classes("text-green-9")
        ui.label(f"{n} target {'allele' if n == 1 else 'alleles'} observed.").classes("runner-heading")
        goals = ((viewer.snapshot() or {}).get("investigation_request") or {}).get("request", {}).get("research_goals", [])
        ui.label(" · ".join(goals) if goals else f"Results for all {len(snps)} requested targets.").classes("runner-question")
        ui.label(f"{len(snps)} requested targets · {len(unique)} distinct allele results").classes("runner-muted")
        data = [
            (counts[k], label, k)
            for k, label in [
                ("exact_allele_observed", "Alleles observed"),
                ("target_allele_not_carried", "Target alleles not carried"),
                ("no_record", "No VCF record"),
            ]
        ]
        data.append((sum(s["state"] == "unresolved" for s in snps), "Unresolved requests", "unresolved"))
        with ui.element("div").classes("runner-counts"):
            for count, label, key in data:
                tone = OUTCOME_LABELS.get(key, ("", "uncertain"))[1]
                with (
                    ui.button(on_click=lambda k=key: selector.set_value(k), color=None)
                    .props("flat no-caps")
                    .classes(f"runner-count tone-{tone}")
                ):
                    ui.label(str(count)).classes("runner-number")
                    ui.label(label).classes("runner-count-label")
        if unique and all(i.get("coverage") == "not_measured" for i in unique.values()):
            ui.label("Coverage not measured · no VCF record does not establish absence.").classes("runner-limit")
        if any(not value for value in included.values()):
            ui.label("Summary counts exclude work items removed from the report. Those items remain visible below.").classes(
                "runner-limit"
            )
    ui.label("Findings").classes("runner-section-title")
    with ui.row().classes("runner-filters w-full items-center"):
        search = ui.input(placeholder="Find a SNP…", value=state.get("search", "")).props("outlined dense clearable")
        options = {
            "all": "All outcomes",
            "exact_allele_observed": "Alleles observed",
            "target_allele_not_carried": "Target not carried",
            "no_record": "No VCF record",
            "unresolved": "Unresolved requests",
            "rejected": "Rejected requests",
        }
        selector = ui.select(options, value=state.get("outcome", "all")).props("outlined dense")
        total = ui.label("").classes("runner-muted")
    groups = dict(finding_groups(report))
    body = ui.column().classes("runner-results w-full gap-0")

    def refresh() -> None:
        state.update(search=search.value or "", outcome=selector.value)
        rows = [
            s
            for s in snps
            if state["search"].lower() in (s["requested_rsid"] or s["request_text"]).lower()
            and (
                selector.value == "all"
                or s["state"] == selector.value
                or any(i.get("outcome") == selector.value for i in s["items"])
            )
        ]
        total.text = f"{len(rows)} of {len(snps)} targets"
        body.clear()
        with body:
            for snp in rows:
                target = snp["requested_rsid"] or snp["request_text"]
                tones = {OUTCOME_LABELS.get(i.get("outcome"), ("", "uncertain"))[1] for i in snp["items"]}
                tone = "observed" if "observed" in tones else "not-carried" if "not-carried" in tones else "uncertain"
                with (
                    ui.expansion(value=target in state.setdefault("expanded", set()))
                    .classes(f"runner-finding tone-{tone}")
                    .props(f'duration=0 data-target="{target}"')
                    .mark("compact-finding") as row
                ):
                    row.on_value_change(lambda e, t=target: state["expanded"].add(t) if e.value else state["expanded"].discard(t))
                    with row.add_slot("header"):
                        with ui.element("div").classes("runner-finding-header"):
                            ui.label(target).classes("runner-snp")
                            with ui.row().classes("gap-2 grow items-center"):
                                labels = Counter()
                                for i in snp["items"]:
                                    word, shade = OUTCOME_LABELS.get(i.get("outcome"), (words(i["state"]), "uncertain"))
                                    text = word if i.get("outcome") == "no_record" else f"{i['target'].split()[-1]} {word}"
                                    if not included.get(i["work_item_id"], True):
                                        text += " · excluded"
                                    labels[(text, shade)] += 1
                                for (text, shade), repeats in labels.items():
                                    ui.label(text + (f" · {repeats} alleles" if repeats > 1 else "")).classes(
                                        f"runner-outcome tone-{shade}"
                                    )
                                if not snp["items"]:
                                    ui.label(words(snp["state"]).capitalize()).classes("runner-outcome tone-uncertain")
                    if snp.get("verified_rsid") and snp["verified_rsid"] != snp.get("requested_rsid"):
                        ui.label(f"Current identity: {snp['verified_rsid']}").classes("runner-muted")
                    for item in snp["items"]:
                        with ui.column().classes("runner-allele w-full gap-2"):
                            ui.label(item["target"]).classes("font-bold")
                            with ui.element("div").classes("runner-measurements"):
                                for label, value in [
                                    ("Genotype", item.get("genotype")),
                                    ("Read depth (DP)", item.get("read_depth")),
                                    ("Genotype quality (GQ)", item.get("genotype_quality")),
                                    ("Coverage", words(item.get("coverage") or "not recorded")),
                                ]:
                                    with ui.column().classes("gap-1"):
                                        ui.label(label).classes("runner-muted")
                                        ui.label(str(value) if value is not None else "—").classes("font-bold")
                    for finding in groups[target]:
                        ui.label(finding["finding"]).classes("text-base")
                        ui.label(finding["reading"]).classes("runner-muted")
                        ui.label(finding["limitation"]).classes("runner-muted")
                    ui.link("Trace this finding", viewer.url(section="finding", focus=target)).classes("runner-back")
            if not rows:
                ui.label("No targets match this selection.").classes("p-5 runner-muted")

    search.on_value_change(refresh)
    selector.on_value_change(refresh)
    refresh()
    with ui.expansion("Report context, evidence & files", icon="article").classes("w-full"):
        context(report, artifacts, live)


def finding_cards(report: dict) -> None:
    """The findings as one card per requested target: each work item's place and genotype, finding, reading and
    limitation; a target without a work item shows its ended state."""
    with ui.grid(columns=3).classes("w-full gap-3"):
        for target, rows in finding_groups(report):
            with ui.card().props("flat bordered").classes("gap-1").mark("finding-card"):
                ui.label(target).classes("text-lg font-bold mono")
                for row in rows:
                    if row["where"] != DASH:
                        place = row["where"] + (f" · {row['genotype']}" if row["genotype"] != DASH else "")
                        ui.label(place).classes("text-sm mono")
                    ui.label(row["finding"]).classes("text-sm")
                    for key, style in (("reading", "text-xs text-grey-8"), ("limitation", "text-xs text-grey-7")):
                        if row[key] != DASH:
                            ui.label(row[key]).classes(style)


def evidence(report: dict) -> None:
    """The window of the VCF behind each finding, and the sources the targets were checked against."""
    with ui.expansion("Evidence", icon="travel_explore").classes("w-full").props("dense").mark("evidence"):
        for snp in report["collection_view"]["snps"] if "collection_view" in report else []:
            for item in snp["items"]:
                window = item.get("window")  # left out until the item's window is read
                ui.label(f"{snp['requested_rsid'] or snp['request_text']} {item['target']}"
                         + (f" · {window['flank_bases']} bp window" if window else " · no window read")).classes("text-sm font-bold")  # fmt: skip
                if window:
                    for record in window["records"] or [window.get("refusal") or "no VCF record in the window"]:
                        ui.label(record).classes("text-xs mono")
        assessments = {a["value"]["source_id"]: a for a in report["source_assessments"]}
        ui.label(f"Sources ({len(report['retrieved_sources'])})").classes("text-sm font-bold pt-2")
        rows = [{
            "row": s["source_id"], "source": f"{s['authority']} {words(s['operation'])}", "request": s["request_identity"],
            "status": s["status_code"], "observed": s["observed_at"][:19].replace("T", " "),
            "assessment": assessed(assessments.get(s["source_id"])),
            "included": "included" if member["membership"]["included"] else "excluded",
        } for member in report["retrieved_sources"] for s in (member["value"],)]  # fmt: skip
        table(rows, ("source", "request", "status", "observed", "assessment", "included"), "sources")


def assessed(member: dict | None) -> str:
    """A source assessment's verdict, marked when the researcher excluded the assessment from the report."""
    if member is None:
        return DASH
    verdict = f"{member['value']['verdict']}: {member['value']['rationale']}"
    return verdict if member["membership"]["included"] else f"excluded from the report: {verdict}"


def accounting(report: dict) -> None:
    """The Run's accounting the report retains (every event that explains an outcome), by event type."""
    counts = Counter(entry["value"]["event_type"] for entry in report["accounting"])
    with (
        ui.expansion(f"Run accounting ({len(report['accounting'])} entries)", icon="receipt_long")
        .classes("w-full")
        .props("dense")
    ):
        with ui.column().classes("gap-0").mark("accounting"):
            for event_type, count in sorted(counts.items()):
                ui.label(f"{event_type}: {count}").classes("text-xs mono")


def files(artifacts: tuple[dict | None, dict | None] | None, live: bool) -> None:
    """`run-artifacts`' list, or why it is not shown."""
    with ui.card().props("flat bordered").classes("w-full").mark("artifacts"):
        ui.label("The Run's files").classes("font-bold")
        if artifacts is None:
            ui.label("Reading the Run's files (run-artifacts)…" if live else
                     "Not read: run-artifacts was not recorded for this replay.").classes("text-sm text-grey-7")  # fmt: skip
            return
        document, refusal = artifacts
        if document is None:
            fields = ", ".join(refusal["fields"])
            ui.label(f"run-artifacts refused this Run: {refusal['code']}" + (f" ({fields})" if fields else "")
                     + ". The report above stands; only its file list is missing.").classes("text-sm text-orange-9")  # fmt: skip
            return
        ui.label(f"Bucket {document['bucket']} · {document['object_scope']}").classes("text-xs text-grey-7 mono")
        table(artifact_rows(document), ("group", "role", "location", "size", "sha256", "proof"), "artifact-list")
