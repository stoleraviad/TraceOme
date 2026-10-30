"""Building a review's response from its `response_schema` and its request, and checking it against that schema.

The schemas are lane M's own (`tests/schemas.json`); `jsonschema` is the independent judge of each built response.
"""

from __future__ import annotations

import jsonschema
import pytest
from runs import DIGEST, SCHEMAS, member_id, schema_review

from responses import (
    FILE_SCHEMAS,
    build_response,
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

ADA = "researcher:ada"


def action_of(review: dict, name: str) -> dict:
    return next(a for a in review["legal_actions"] if a["action"] == name)


# A finalization answer binds the review's own event (the Journey head when it was asked), not the basis's older
# source head: `finalization_commands` refuses any other `head_event_id` (Spec r1 P1). (kind, action, what the
# researcher enters, the exact response): ids and constants come from the request and schema, and the attribution
# fields (`asserted_by`, `by`) from the one "Answering as" name.
FILLED = [
    ("execution", "approve", {}, {"action": "approve", "proposal_id": "execution-proposal:" + "1" * 64}),
    ("investigation", "approve", {}, {"action": "approve", "proposal_id": f"investigation-proposal:{DIGEST}"}),
    ("investigation_finding", "approve", {},
     {"action": "approve", "collection_member_id": member_id("rs30"), "revision_number": 1, "asserted_by": ADA}),
    ("junction", "retry", {}, {"state": "answered", "action": "retry", "request_id": "failure-request:1", "by": ADA}),
    ("execution_path", "abort", {"reason": "Wrong reference."},
     {"state": "answered", "action": "abort", "request_id": "failure-request:1", "by": ADA, "reason": "Wrong reference."}),
    ("finalization", "approve", {},
     {"action": "approve", "basis_digest": f"finalization-basis:{DIGEST}", "head_event_id": "request-1"}),
]  # fmt: skip


@pytest.mark.parametrize(("kind", "name", "entered", "expected"), FILLED, ids=[f"{k}-{a}" for k, a, *_ in FILLED])
def test_ids_and_constants_come_from_the_request_and_the_researcher_enters_only_the_rest(kind, name, entered, expected) -> None:
    review = schema_review(kind, name)
    action = action_of(review, name)
    asked = {f.name for f in open_fields(review, action)}
    assert asked.isdisjoint(set(expected) - set(entered)), "the researcher is asked for an id, a constant or the name"
    response = build_response(review, action, entered, by=ADA)
    assert response == expected
    jsonschema.validate(response, action["response_schema"])
    assert problems(action, response) == []


# What a researcher might enter for each field a schema requires; revise's operations have their own test.
ENTERED = {
    "reason": "The evidence does not support this.", "instruction": "Use GRCh38.",
    "additional_requests": 2, "candidate_id": "wes", "revalidation_scope": "reference",
    "operations": [{"operation": "keep", "work_item_id": f"work-item:{'a' * 64}"}],
}  # fmt: skip
# Each kind's actions as lane M asks them (investigation clarify and reject come with a question about intent).
EVERY = [(k, a) for k, actions in SCHEMAS.items() if k not in ("_source", "finalization_files") for a in actions
         if (k, a) not in {("investigation", "clarify"), ("investigation", "reject"), ("preparation", "supply")}
         and k != "finalization"] + [("investigation_question", "clarify"), ("investigation_question", "reject")]  # fmt: skip


@pytest.mark.parametrize(("kind", "name"), EVERY, ids=[f"{k}-{a}" for k, a in EVERY])
def test_every_action_of_every_kind_builds_a_response_its_schema_accepts(kind, name) -> None:
    review = schema_review(kind, name)
    action = action_of(review, name)
    entered = {f.name: ENTERED[f.name] for f in open_fields(review, action) if f.required}
    response = build_response(review, action, entered, by=ADA)
    jsonschema.validate(response, action["response_schema"])
    assert problems(action, response) == []


def test_a_response_missing_what_the_researcher_must_enter_names_the_field() -> None:
    review = schema_review("junction", "retry")
    action = action_of(review, "retry")
    assert problems(action, build_response(review, action, {}, by=ADA)) == []
    assert problems(action, build_response(review, action, {})) == ["response: 'by' is a required property"]


def test_revise_answers_each_work_item_once_and_a_target_without_one_by_its_request_member() -> None:
    """rs11 was merged into rs20 and shares its work item, so that item gets one row; rs30 has no work item."""
    review = schema_review("investigation", "revise")
    action = action_of(review, "revise")
    rows = target_rows(review, action)
    assert [(r.rsid, r.key, r.id) for r in rows] == [
        ("rs10", "work_item_id", f"work-item:{'a' * 64}"),
        ("rs10", "work_item_id", f"work-item:{'b' * 64}"),
        ("rs20", "work_item_id", f"work-item:{'c' * 64}"),
        ("rs30", "request_member_id", member_id("rs30")),
    ]
    assert rows[0].choices == ["keep", "retry", "amend", "remove", "defer", "clarify", "search_adjustment"]
    assert rows[3].choices == ["clarify_request", "amend_request"]
    assert rows[0].needs_request == {"amend", "clarify", "search_adjustment"}
    chosen = [("keep", ""), ("remove", ""), ("amend", "rs20 T>C"), ("clarify_request", "rs30 on chromosome 2")]
    ops = [operation(row, op, text) for row, (op, text) in zip(rows, chosen, strict=True)]
    assert ops == [
        {"operation": "keep", "work_item_id": f"work-item:{'a' * 64}"},
        {"operation": "remove", "work_item_id": f"work-item:{'b' * 64}"},
        {"operation": "amend", "work_item_id": f"work-item:{'c' * 64}", "request": {"request_text": "rs20 T>C"}},
        {
            "operation": "clarify_request",
            "request_member_id": member_id("rs30"),
            "request": {"request_text": "rs30 on chromosome 2"},
        },
    ]
    response = build_response(review, action, {"operations": ops}, by=ADA)
    jsonschema.validate(response, action["response_schema"])


def test_approving_with_no_target_selected_is_refused_not_read_as_approving_every_target() -> None:
    """`work_item_ids` left out approves every work item; an empty selection must not turn into that."""
    review = schema_review("investigation", "approve")
    action = action_of(review, "approve")
    response = build_response(review, action, {"work_item_ids": []})
    assert response["work_item_ids"] == []
    assert problems(action, response)


# The files finalization correct and stop name: the GUI writes them from typed fields.
BACKEND_FILES = SCHEMAS["finalization_files"]


def file_action(name: str) -> dict:
    return {"action": name, "response_schema": FILE_SCHEMAS[name]}


def test_the_guis_copy_of_the_finalization_file_shapes_matches_the_recorded_backend_schemas() -> None:
    """The copy is the marked exception to schema-driven answers. This compares it with the backend's schemas as
    recorded in `tests/schemas.json`, the correction's `investigation_revision` aside: a backend change shows here only
    after that fixture is regenerated (its `_source` names the generator; the GUI never imports `bio_neb`)."""
    assert set(FILE_SCHEMAS) == set(BACKEND_FILES)
    for name, copy in FILE_SCHEMAS.items():
        backend = BACKEND_FILES[name]
        assert copy["properties"] == {k: v for k, v in backend["properties"].items() if k != "investigation_revision"}
        assert copy["required"] == backend["required"]
        assert {k: backend["$defs"][k] for k in copy.get("$defs", {})} == copy.get("$defs", {})


def test_a_stop_writes_its_reason_file_from_the_typed_reason_and_the_answering_name() -> None:
    review = schema_review("finalization", "stop")
    assert [f.name for f in open_fields(review, file_action("reason_file"))] == ["reason"]
    content = build_response(review, file_action("reason_file"), {"reason": "The basis cannot support a report."}, by=ADA)
    assert content == {"schema_version": "bio-neb.finalization-stop/1", "asserted_by": ADA,
                       "reason": "The basis cannot support a report."}  # fmt: skip
    jsonschema.validate(content, BACKEND_FILES["reason_file"])


def test_a_correction_lists_report_membership_and_investigation_changes() -> None:
    review = schema_review("finalization", "correct")
    correction = file_action("correction_file")
    assert [f.name for f in open_fields(review, correction)] == ["reason", "requested_changes"]
    kinds = dict(union_kinds(correction, "requested_changes"))
    assert list(kinds) == ["set_report_membership", "add_scientific_fact", "change_source", "change_observation",
                           "change_assessment_or_admission", "change_target_or_question", "new_research"]  # fmt: skip
    assert [m for m, _ in report_members(review)] == [f"admitted-observation-evidence:{'d' * 64}", f"retrieved-source:{'e' * 64}"]
    membership = {"action": "set_report_membership", "response_schema": kinds["set_report_membership"]}
    research = {"action": "new_research", "response_schema": kinds["new_research"]}
    assert [f.name for f in open_fields(review, membership)] == ["member_id", "included"]
    assert [f.name for f in open_fields(review, research)] == ["description"]
    changes = [
        build_response(review, membership, {"member_id": f"admitted-observation-evidence:{'d' * 64}", "included": False}),
        build_response(review, research, {"description": "Look for a second source for rs20."}),
    ]
    assert changes == [
        {"action": "set_report_membership", "member_id": f"admitted-observation-evidence:{'d' * 64}", "included": False},
        {"action": "new_research", "description": "Look for a second source for rs20."},
    ]
    content = build_response(
        review, correction, {"reason": "One admission is not supported.", "requested_changes": changes}, by=ADA
    )
    jsonschema.validate(content, BACKEND_FILES["correction_file"])
    assert content["schema_version"] == "bio-neb.finalization-correction/1"


def test_finalization_correct_and_stop_name_who_answers_through_their_files() -> None:
    review = schema_review("finalization")
    assert [a["action"] for a in review["legal_actions"] if names_answerer(a)] == ["correct", "stop"]


def test_a_rejected_target_without_work_offers_no_operation_and_any_target_without_work_can_stay_unchanged() -> None:
    """Only a request member with no work item and no rejection can be reworked; the backend refuses any other
    `request_member_id` (Spec r1 P3). Leaving a row unchanged sends no operation."""
    review = schema_review("investigation", "revise")
    rs30 = next(m for m in review["request"]["proposal"]["request_members"] if not m["work_item_ids"])
    rs30.update(refsnp=None, assembly_report=None, rejection={"request_event_id": "question-1", "response_event_id": "answer-1",
                                                              "reason": "Outside this investigation.", "asserted_by": ADA})  # fmt: skip
    row = target_rows(review, action_of(review, "revise"))[-1]
    assert (row.rsid, row.key, row.choices) == ("rs30", "request_member_id", [])
    assert operation(row, None, "") is None


def test_revise_can_add_a_target_with_the_operation_that_names_no_existing_one() -> None:
    """The schema's `add` names neither a work item nor a request member: it brings a new target (closure check)."""
    review = schema_review("investigation", "revise")
    action = action_of(review, "revise")
    assert new_target_kinds(action) == ["add"]
    added = new_target("add", "Does NA00001 carry rs99?")
    assert added == {"operation": "add", "request": {"request_text": "Does NA00001 carry rs99?"}}
    assert new_target("add", "  ") is None
    response = build_response(review, action, {"operations": [added]}, by=ADA)
    jsonschema.validate(response, action["response_schema"])
