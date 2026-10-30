"""Readable, event-specific evidence pages over the selected Journey prefix.

This module presents retained facts. It never admits a claim, interprets an approval
as scientific evidence, retrieves a source, or fills absent measurements from later events.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from nicegui import ui

from journey_feed import FeedItem
from run_reading import Viewer, fmt_size


def field(label: str, value) -> None:
    if value is None or isinstance(value, dict):
        return
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if not isinstance(v, (dict, list)))
    with ui.row().classes("evidence-field w-full no-wrap gap-4"):
        ui.label(label).classes("evidence-label")
        ui.label(str(value)).classes("evidence-value")


def block(title: str):
    card = ui.column().classes("evidence-block w-full gap-3")
    with card:
        ui.label(title).classes("text-lg font-bold")
    return card


def decision(payload: dict) -> None:
    receipt = payload.get("receipt") or {}
    response = payload.get("response") or receipt.get("response") or {}
    with block("Choice and consequence"):
        field("Action", str(response.get("action", "Not recorded")).replace("_", " ").capitalize())
        for key, label in [
            ("instruction", "Instruction"),
            ("reason", "Reason"),
            ("candidate_id", "Selected option"),
            ("asserted_by", "Answered by"),
        ]:
            field(label, response.get(key))
        field("Authority", str(payload.get("authority", "Researcher response")).replace("_", " "))
        if "authorizes_execution" in payload:
            field("Execution authorised", "Yes" if payload["authorizes_execution"] else "No")
        field("Scientific validation", "Not established by this decision")


def sample_observation(payload: dict) -> None:
    work = payload.get("work_item") or {}
    target = work.get("target") or {}
    observation = payload.get("sample_observation") or {}
    result = observation.get("result") or {}
    assessment = payload.get("assessment") or {}
    admission = payload.get("admission") or {}
    record = result.get("record") or {}
    with block("Sample observation"):
        field("Sample", work.get("sample_id") or observation.get("sample_id"))
        field("Target", target.get("display_label") or target.get("aliases"))
        if target.get("contig") is not None:
            field("Position", f"{target['contig']}:{target.get('one_based_position', '—')}")
        if target.get("ref") is not None and target.get("alt") is not None:
            field("Requested allele", f"{target['ref']}>{target['alt']}")
        field("Recorded outcome", str(assessment.get("outcome") or result.get("outcome") or "Not recorded").replace("_", " "))
        rows = [
            {
                "sample": g.get("sample_name", "—"),
                "genotype": g.get("genotype", "—"),
                "depth": g.get("dp", "—"),
                "quality": g.get("gq", "—"),
            }
            for g in record.get("genotypes", [])
        ]
        if rows:
            columns = [
                {"name": k, "field": k, "label": title, "align": "left"}
                for k, title in [
                    ("sample", "Sample"),
                    ("genotype", "Genotype"),
                    ("depth", "Read depth (DP)"),
                    ("quality", "Genotype quality (GQ)"),
                ]
            ]
            ui.table(columns=columns, rows=rows, row_key="sample").props("flat bordered wrap-cells").classes("w-full")
        else:
            ui.label("No genotype measurements are recorded in this event.").classes("text-sm")
        field("Coverage", "Not recorded in this event")
    with block("Claim and limits"):
        claim = admission.get("admitted_claim") or {}
        field("Admission", admission.get("status"))
        field("Retained finding", claim.get("observation_detail"))
        field("Claim scope", admission.get("claim_scope") or result.get("claim_scope"))
        ui.label("Read depth is a VCF-record measurement. A missing record does not establish absence.").classes("text-sm")


def source(payload: dict) -> None:
    result = payload.get("result") or {}
    retained = payload.get("source") or result.get("source") or {}
    with block("Source and identity"):
        for label, value in [
            ("Requested target", payload.get("query")),
            ("Status", result.get("status")),
            ("Verified identity", result.get("verified_rsid")),
            ("Merged into", result.get("merged_into")),
            ("Authority", retained.get("authority")),
            ("Operation", retained.get("operation")),
            ("HTTP status", retained.get("status_code")),
            ("Retrieved at", retained.get("observed_at")),
        ]:
            field(label, value)
        assessment = result.get("assessment") or {}
        field("Assessment", assessment.get("verdict"))
        field("Rationale", assessment.get("rationale"))
        address = retained.get("request_identity")
        if isinstance(address, str) and urlsplit(address).scheme in ("https", "http"):
            ui.link("Open the recorded source address", address, new_tab=True)
        for candidate in result.get("candidates", []):
            field(
                "Candidate",
                f"{candidate.get('contig', '—')}:{candidate.get('one_based_position', '—')} "
                f"{candidate.get('ref', '—')}>{candidate.get('alt', '—')} · {candidate.get('assembly_name', '—')}",
            )
        field("Conflicts", result.get("conflicts"))


def output_check(payload: dict) -> None:
    facts = payload.get("payload") or {}
    with block("Execution record"):
        field("Step", str(payload.get("event_type", "Recorded execution step")).replace("_", " "))
        field("Grounding", payload.get("grounding"))
        field("Outcome", facts.get("outcome") or facts.get("status"))
        for pointer in facts.get("pointers", []):
            with ui.column().classes("evidence-output gap-1"):
                field("File role", pointer.get("role"))
                if pointer.get("bytes") is not None:
                    field("Size", fmt_size(pointer["bytes"]))
                field("Verification", pointer.get("grounding"))
        if payload.get("event_type") == "H100ArtifactsVerified":
            ui.label("These are recorded output checks. They do not establish a full run-health assessment.").classes("text-sm")


def window_records(payload: dict) -> None:
    result = payload.get("result") or {}
    target = result.get("target") or {}
    with block("Nearby VCF records"):
        field("Target", target.get("display_label"))
        if target.get("contig") is not None:
            field("Window", f"{target['contig']}:{result.get('window_start', '—')}–{result.get('window_end', '—')}")
        rows = []
        for record in result.get("records", []):
            for genotype in record.get("genotypes") or [{}]:
                rows.append(
                    {
                        "id": len(rows) + 1,
                        "position": f"{record.get('contig', '—')}:{record.get('one_based_position', '—')}",
                        "alleles": f"{record.get('ref', '—')}>{','.join(record.get('alts', []))}",
                        "sample": genotype.get("sample_name", "—"),
                        "genotype": genotype.get("genotype", "—"),
                        "depth": genotype.get("dp", "—"),
                        "quality": genotype.get("gq", "—"),
                    }
                )
        if rows:
            columns = [
                {"name": key, "field": key, "label": label, "align": "left"}
                for key, label in [
                    ("position", "Position"),
                    ("alleles", "Alleles"),
                    ("sample", "Sample"),
                    ("genotype", "GT"),
                    ("depth", "DP"),
                    ("quality", "GQ"),
                ]
            ]
            ui.table(columns=columns, rows=rows, row_key="id").props("flat bordered wrap-cells").classes("w-full")
        else:
            ui.label("No VCF records were returned in this recorded window.")
        field("Claim scope", result.get("claim_scope"))
        ui.label("These are nearby VCF records; read depth does not establish coverage or absence.").classes("runner-limit")


def reading_notes(payload: dict, target_labels: dict[str, str]) -> None:
    with block("Agent reading notes"):
        ui.label("These readings do not independently validate a scientific claim.").classes("runner-limit")
        for n, reading in enumerate(payload.get("readings", []), 1):
            with ui.column().classes("evidence-output gap-2"):
                work_id = reading.get("work_item_id")
                ui.label(target_labels.get(work_id, f"Work item {n}")).classes("font-bold")
                field("Reading", str(reading.get("reading", "Not recorded")).replace("_", " "))
                field("Concerns", [str(c).replace("_", " ") for c in reading.get("concerns", [])])
                field("Note", reading.get("note"))


def event_kind(item: FeedItem) -> str:
    name = item.event_type
    if name.endswith(("Responded", "ReviewRecorded")):
        return "Researcher decision"
    if name == "InvestigationEvidenceRecorded":
        return "Variant evidence"
    if name in ("RefSnpToolResultRecorded", "InvestigationSourceRetrieved"):
        return "Source evidence"
    if name == "ExecutionPathRecorded":
        return "Execution evidence"
    if name.startswith("Model"):
        return "Model activity"
    return "Recorded activity"


def record_link(viewer: Viewer, item: FeedItem) -> str:
    return viewer.url(section="evidence", record=item.n)


def show(viewer: Viewer, number: int | None) -> None:
    """Open one record, with explicit links and chronological context inside the viewed prefix."""
    items = viewer.items[: viewer.cursor]
    item = next((i for i in items if i.n == number), None)
    ui.link("← Back to the run", viewer.url(section="auto", record=None)).classes("runner-back")
    if item is None:
        ui.label("This record is not available at this point in the run.").classes("text-lg")
        return
    event = json.loads(viewer.raw[item.n - 1])
    payload = event.get("payload") or {}
    kind = event_kind(item)
    record_bytes = len(viewer.raw[item.n - 1].encode("utf-8"))
    size = f"{record_bytes} bytes" if record_bytes < 1024 else fmt_size(record_bytes)
    ui.label(kind).classes("runner-kicker")
    ui.label(item.title).classes("runner-heading")
    ui.label(f"{item.phase} · {item.time} UTC · event {item.n} · {size} recorded").classes("runner-muted")
    if kind == "Researcher decision":
        decision(payload)
    elif kind == "Variant evidence":
        sample_observation(payload)
    elif kind == "Source evidence":
        source(payload)
    elif kind == "Execution evidence":
        output_check(payload)
    elif item.event_type == "InvestigationItemWindowRecorded":
        window_records(payload)
    elif item.event_type == "InvestigationTableChunkRead":
        labels = {}
        for earlier in items:
            if earlier.n >= item.n:
                break
            if earlier.event_type == "InvestigationEvidenceRecorded":
                work = json.loads(viewer.raw[earlier.n - 1])["payload"].get("work_item") or {}
                target = work.get("target") or {}
                label = target.get("display_label") or ", ".join(target.get("aliases", []))
                if work.get("work_item_id") and label:
                    labels[work["work_item_id"]] = f"{label} · {target.get('ref', '—')}>{target.get('alt', '—')}"
        reading_notes(payload, labels)
    elif kind == "Model activity":
        with block("Model payload retained"):
            ui.label("The model's full request or response is retained in the recording. Its content is hidden here.")
            field("Recorded size", fmt_size(item.size))
    else:
        with block("What was recorded"):
            for detail in item.details:
                ui.label(detail).classes("text-base")
            if item.event_type == "ResearchRequestRegistered":
                field("Research question", payload.get("prompt"))
            elif item.event_type == "RunInputsRegistered":
                for entry in payload.get("inputs", []):
                    field(str(entry.get("role", "Input")), entry.get("registered_path"))
            else:
                for key, label in [
                    ("tool_name", "Tool"),
                    ("status", "Status"),
                    ("disposition", "Disposition"),
                    ("agent_role", "Agent role"),
                    ("question", "Question"),
                ]:
                    field(label, payload.get(key))
    by_id = {i.event_id: i for i in items}
    ids = [*(event.get("parent_ids") or [])]
    for container in (payload, payload.get("receipt") or {}):
        ids += [
            container[k]
            for k in ("request_event_id", "source_event_id", "evidence_event_id")
            if isinstance(container.get(k), str)
        ]
    linked = [by_id[k] for k in dict.fromkeys(ids) if k in by_id and by_id[k].n != item.n]
    if linked:
        with block("Linked records"):
            for linked_item in linked:
                ui.link(f"Event {linked_item.n} · {linked_item.title}", record_link(viewer, linked_item))
    with block("Recorded sequence"):
        ui.label("Nearby key steps in this stage; chronology does not by itself establish causality.").classes("text-sm")
        nearby = [i for i in items if i.phase == item.phase and i.level in ("key", "step")]
        before = [i for i in nearby if i.n < item.n][-3:]
        after = [i for i in nearby if i.n > item.n][:2]
        for point in [*before, item, *after]:
            if point.n == item.n:
                ui.label(f"{point.time} · {point.title}").classes("font-bold")
            else:
                ui.link(f"{point.time} · {point.title}", record_link(viewer, point))
    with ui.expansion("Technical record", icon="data_object").classes("w-full evidence-technical"):
        field("Event type", item.event_type)
        field("Event ID", item.event_id)
        field("Pass", item.pass_no)
        response = payload.get("response") or (payload.get("receipt") or {}).get("response") or {}
        field("Reviewed proposal", response.get("proposal_id"))
        raw = ui.column().classes("w-full")

        def reveal() -> None:
            raw.clear()
            text = json.dumps(event, indent=2, ensure_ascii=False)
            encoded = text.encode("utf-8")
            with raw:
                ui.code(encoded[:4000].decode("utf-8", errors="ignore"), language="json").classes("w-full max-h-96 overflow-auto")
                if len(encoded) > 4000:
                    ui.label("Excerpt limited to the first 4 KB.").classes("text-xs")

        ui.button("Show JSON excerpt", icon="code", on_click=reveal).props("flat no-caps")
