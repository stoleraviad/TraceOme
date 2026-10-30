"""A review's response, built from its action's `response_schema` and its request: no element is built here.

The schema names every field. Constants come from the schema, correlation ids from the request and the attribution
fields from the one "Answering as" name; the researcher is asked for the rest (`open_fields`). `problems` checks a
built response against the schema before a command is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

from jsonschema import Draft202012Validator

# Where each correlation id a schema may name sits in the review's request (the first path present wins). The
# schema decides whether a response carries the id; the request holds its value. A finalization answer's
# `head_event_id` is the review's own event (`finalization_commands` binds it to the pending request), not the
# basis's older `source_head_event_id`.
REQUEST_IDS = {
    "proposal_id": (("proposal", "proposal_id"),),
    "collection_member_id": (("question", "collection_member_id"),),
    "revision_number": (("question", "revision_number"),),
    "request_id": (("request", "request_id"), ("review", "request_id")),
    "basis_digest": (("basis", "basis_digest"),),
    "head_event_id": (("request_event_id",),),
}
# THE EXCEPTION to schema-driven answers. Finalization correct and stop name a file
# (`correction_file`, `reason_file`) whose content `observe` does not publish; the backend reads it as
# `bio-neb.finalization-correction/1` or `bio-neb.finalization-stop/1` (`reporting/finalization.py`
# `FinalizationCorrectionAssertion`, `FinalizationStopAssertion`). These are copies of those schemas, the correction's
# optional `investigation_revision` left out; `tests/test_responses.py` compares them with the backend's schemas as
# recorded in `tests/schemas.json` (regenerate it when lane M changes them). Remove them
# when `response_schema` publishes the file content (https://github.com/stoleraviad/BioNeb/discussions/1#discussioncomment-19123620).
FILE_SCHEMAS: dict[str, dict] = {
    "reason_file": {
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": "bio-neb.finalization-stop/1", "default": "bio-neb.finalization-stop/1",
                               "title": "Schema Version", "type": "string"},
            "asserted_by": {"minLength": 1, "title": "Asserted By", "type": "string"},
            "reason": {"minLength": 1, "title": "Reason", "type": "string"},
        },
        "required": ["asserted_by", "reason"],
        "title": "FinalizationStopAssertion",
        "type": "object",
    },
    "correction_file": {
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": "bio-neb.finalization-correction/1", "default": "bio-neb.finalization-correction/1",
                               "title": "Schema Version", "type": "string"},
            "asserted_by": {"minLength": 1, "title": "Asserted By", "type": "string"},
            "reason": {"minLength": 1, "title": "Reason", "type": "string"},
            "requested_changes": {
                "items": {
                    "discriminator": {"mapping": {
                        "add_scientific_fact": "#/$defs/InvestigationRequiredChange",
                        "change_assessment_or_admission": "#/$defs/InvestigationRequiredChange",
                        "change_observation": "#/$defs/InvestigationRequiredChange",
                        "change_source": "#/$defs/InvestigationRequiredChange",
                        "change_target_or_question": "#/$defs/InvestigationRequiredChange",
                        "new_research": "#/$defs/InvestigationRequiredChange",
                        "set_report_membership": "#/$defs/SetReportMembershipChange",
                    }, "propertyName": "action"},
                    "oneOf": [{"$ref": "#/$defs/SetReportMembershipChange"}, {"$ref": "#/$defs/InvestigationRequiredChange"}],
                },
                "minItems": 1,
                "title": "Requested Changes",
                "type": "array",
            },
        },
        "required": ["asserted_by", "reason", "requested_changes"],
        "title": "FinalizationCorrectionAssertion",
        "type": "object",
        "$defs": {
            "SetReportMembershipChange": {
                "additionalProperties": False,
                "properties": {
                    "action": {"const": "set_report_membership", "default": "set_report_membership", "title": "Action",
                               "type": "string"},
                    "member_id": {"title": "Member Id", "type": "string"},
                    "included": {"title": "Included", "type": "boolean"},
                },
                "required": ["member_id", "included"],
                "title": "SetReportMembershipChange",
                "type": "object",
            },
            "InvestigationRequiredChange": {
                "additionalProperties": False,
                "properties": {
                    "action": {"enum": ["add_scientific_fact", "change_source", "change_observation",
                                        "change_assessment_or_admission", "change_target_or_question", "new_research"],
                               "title": "Action", "type": "string"},
                    "description": {"minLength": 1, "title": "Description", "type": "string"},
                },
                "required": ["action", "description"],
                "title": "InvestigationRequiredChange",
                "type": "object",
            },
        },
    },
}  # fmt: skip


def file_action(name: str) -> dict:
    """A named file's content, answered like a response: its copied schema in place of a `response_schema`."""
    return {"action": name, "response_schema": FILE_SCHEMAS[name]}


# The fields that say who answers: every one the schema names takes the page's "Answering as" name.
ATTRIBUTION = ("asserted_by", "by")


@dataclass(frozen=True)
class Field:
    """One field the researcher answers: its schema (a `$ref` followed, an optional `null` dropped) and whether the
    schema requires it."""

    name: str
    prop: dict
    required: bool


def resolve(schema: dict, prop: dict) -> dict:
    """A property's own schema: a `$ref` followed into `$defs`, and `X | null` read as `X`."""
    if "$ref" in prop:
        prop = schema["$defs"][prop["$ref"].rsplit("/", 1)[-1]]
    options = [o for o in prop.get("anyOf", []) if o.get("type") != "null"]
    if len(options) == 1 and len(prop["anyOf"]) == 2:
        return resolve(schema, options[0])
    return prop


def _from_request(request: dict, paths: tuple[tuple[str, ...], ...]):
    for path in paths:
        value = request
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            return value
    return None


def given(review: dict, action: dict) -> dict:
    """The part of the response nobody types: the action, the schema's constants and the request's ids."""
    schema = action["response_schema"]
    props = schema.get("properties", {})
    out = {"action": action["action"]} if "action" in props else {}
    for name, prop in props.items():
        if name == "action":
            continue
        if "const" in prop:
            out[name] = prop["const"]
        elif name in REQUEST_IDS and (value := _from_request(review["request"], REQUEST_IDS[name])) is not None:
            out[name] = value
    return out


def open_fields(review: dict, action: dict) -> list[Field]:
    """The fields the researcher answers, in the schema's order (the attribution fields come from the name)."""
    schema = action["response_schema"]
    required = set(schema.get("required", []))
    fixed = {*given(review, action), *ATTRIBUTION}
    return [Field(n, resolve(schema, p), n in required) for n, p in schema.get("properties", {}).items() if n not in fixed]


def names_answerer(action: dict) -> bool:
    """Whether the response says who answers: its schema, or a file it names, has an attribution field."""
    props = action["response_schema"].get("properties", {})
    return any(name in props for name in ATTRIBUTION) or any(
        names_answerer(file_action(name)) for name in props if name in FILE_SCHEMAS
    )


def build_response(review: dict, action: dict, fields: dict, by: str | None = None) -> dict:
    """The response: what nobody types, `by` in every attribution field the schema names, and the researcher's values.
    A blank value leaves its field out; an empty selection stays, so the schema judges it (left out, `work_item_ids`
    would approve every target)."""
    response = given(review, action)
    props = action["response_schema"].get("properties", {})
    response.update({name: by for name in ATTRIBUTION if name in props and by})
    response.update({k: v for k, v in fields.items() if v not in (None, "")})
    return response


def problems(action: dict, response: dict, where: str | None = None) -> list[str]:
    """Where the response breaks its schema, one line per error; empty when the backend's own schema accepts it. A
    file's lines start with the field that names it (`where`)."""
    errors = Draft202012Validator(action["response_schema"]).iter_errors(response)
    return [f"{'.'.join(map(str, [*([where] if where else []), *e.absolute_path])) or 'response'}: {e.message}" for e in errors]


@dataclass(frozen=True)
class Row:
    """One target row of a per-target `operations` field: a work item (one per ALT), or a requested target with no
    work item, answered by its request member. `choices` are the operations the schema allows for that key (none for a
    rejected request member, which the backend does not rework), and `needs_request` those that also carry a corrected
    request. A request member's row may also stay unchanged: no operation (`operation(row, None, …)`)."""

    rsid: str
    key: str
    id: str
    choices: list[str]
    needs_request: set[str]


def target_rows(review: dict, action: dict) -> list[Row]:
    """The rows of an `operations` field: each work item once (a merged rsID shares its target's), then each requested
    target that has none."""
    kinds = union_kinds(action, "operations")

    def row(rsid: str, key: str, identifier: str) -> Row:
        allowed = [(name, own) for name, own in kinds if key in own.get("required", [])]
        return Row(
            rsid, key, identifier, [n for n, _ in allowed], {n for n, own in allowed if "request" in own.get("required", [])}
        )

    proposal = review["request"].get("proposal") or {}
    rows, seen = [], set()
    for m in proposal.get("request_members", []):
        rsid = m["request"]["rsid"] or m["request"]["request_text"]
        if not m["work_item_ids"]:
            member = row(rsid, "request_member_id", m["request_member_id"])
            # A rejected request keeps its accounting: `compile_investigation_collection_revision` reworks only
            # members with no work item and no rejection.
            rows.append(Row(member.rsid, member.key, member.id, [], set()) if m.get("rejection") else member)
        for w in m["work_item_ids"]:
            if w not in seen:
                seen.add(w)
                rows.append(row(rsid, "work_item_id", w))
    return sorted(rows, key=lambda r: r.key == "request_member_id")


def operation(row: Row, choice: str | None, request_text: str) -> dict | None:
    """One operation for a row, or none for a row left unchanged; a corrected request carries the researcher's text."""
    if choice is None:
        return None
    out = {"operation": choice, row.key: row.id}
    if choice in row.needs_request:
        out["request"] = {"request_text": request_text}
    return out


def union_kinds(action: dict, field: str) -> list[tuple[str, dict]]:
    """Each kind a list-of-union field allows (its discriminator's values, in the union's order), with that kind's own
    schema (carrying the parent's `$defs`), so a kind is answered like a response."""
    schema = action["response_schema"]
    items = resolve(schema, schema["properties"][field])["items"]
    key = items["discriminator"]["propertyName"]
    out = []
    for option in items["oneOf"]:
        own = {**resolve(schema, option), "$defs": schema.get("$defs", {})}
        names = own["properties"][key]
        out += [(name, own) for name in ([names["const"]] if "const" in names else names["enum"])]
    return out


def report_members(review: dict) -> list[tuple[str, str]]:
    """The finalization basis's report members a correction can include or leave out, each with a label."""
    return [
        (m["member_id"], f"{m['member_role']}: {m['member_id']} ({'in' if m['included'] else 'not in'} the report)")
        for m in review["request"]["basis"]["report_membership"]
    ]


def new_target_kinds(action: dict) -> list[str]:
    """The `operations` kinds that name no existing target (neither a work item nor a request member): each brings a
    new target with its own request."""
    keys = ("work_item_id", "request_member_id")
    return [name for name, own in union_kinds(action, "operations") if not set(keys) & set(own.get("required", []))]


def new_target(kind: str, request_text: str) -> dict | None:
    """One new-target operation with the researcher's request, or none while the request is blank."""
    return {"operation": kind, "request": {"request_text": request_text.strip()}} if request_text.strip() else None
