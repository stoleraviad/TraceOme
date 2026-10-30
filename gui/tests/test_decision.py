"""The decision element on the page: each action's fields come from its schema, and a button previews the exact
`answer` command (a dry run: nothing is started)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from nicegui import ui
from runs import DIGEST, journey, member_id, schema_review, snapshot

ADA = "researcher:ada"


def page(kind: str, *names: str) -> tuple[list[dict], dict]:
    review = schema_review(kind, *names)
    events = journey("Investigation")
    return events, {len(events): snapshot(f"{kind}_review_pending", "review_required", reviews=(review,))}


def previewed(user) -> dict:
    """The response in the newest previewed command's `--response-json` argument."""
    argv = max(user.find(marker="argv").elements, key=lambda e: e.id).content
    return json.loads(argv.split("--response-json '", 1)[1].rsplit("'", 1)[0])


async def test_an_action_asks_for_its_schemas_open_fields_and_previews_the_exact_command(replay) -> None:
    events, snaps = page("junction", "retry", "stop")
    async with replay(events, snaps) as user:
        await user.open("/?view=Investigation")
        user.find(kind=ui.button, content="Retry").click()
        user.find(marker="answering-as").type(ADA)
        user.find(kind=ui.button, content="Preview command").click()
        await user.should_see("bio_neb.cli answer")
        await user.should_see("A recorded replay never answers; this button would start:")
        assert previewed(user) == {"state": "answered", "action": "retry", "request_id": "failure-request:1", "by": ADA}
        await user.should_see("The action's response_schema accepts this response.")
        await user.should_not_see(marker="schema-problems")


async def test_a_preview_missing_a_required_field_says_what_the_schema_refuses(replay) -> None:
    events, snaps = page("planning")
    async with replay(events, snaps) as user:
        await user.open("/?view=Plan")
        user.find(kind=ui.button, content="Clarify").click()
        user.find(kind=ui.button, content="Preview command").click()
        await user.should_see("response: 'instruction' is a required property")


async def test_an_answer_that_names_who_answers_is_not_previewed_until_answering_as_is_set(replay) -> None:
    """Owner, #430 Q2 A: the attribution fields take the header's name, and the preview waits for it."""
    events, snaps = page("junction", "retry", "stop")
    async with replay(events, snaps) as user:
        await user.open("/?view=Investigation")
        user.find(kind=ui.button, content="Retry").click()
        await user.should_not_see(marker="field-by")
        user.find(kind=ui.button, content="Preview command").click()
        await user.should_see("Set “Answering as” in the header first")
        await user.should_not_see(marker="argv")


async def test_the_finalization_review_previews_approval_of_its_exact_basis_and_head(replay) -> None:
    events, snaps = page("finalization")
    async with replay(events, snaps) as user:
        await user.open("/?view=Results")
        user.find(kind=ui.button, content="Approve").click()
        assert previewed(user) == {
            "action": "approve",
            "basis_digest": f"finalization-basis:{DIGEST}",
            "head_event_id": "request-1",
        }


async def test_a_kind_from_later_work_shows_its_request_and_actions_read_only(replay) -> None:
    review = {"command_frontier": "answer", "kind": "a_kind_from_later_work", "request_event_id": "request-1",
              "request": {"asks": "something new"}, "legal_actions": [{"action": "go_on", "response_schema": {"type": "object"}}]}  # fmt: skip
    events = journey("Execution")
    snap = snapshot("a_kind_from_later_work_pending", "review_required", reviews=(review,))
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see("Waiting for you: answer a review (a kind from later work)")
        assert "something new" in user.find(marker="raw-request").elements.pop().content
        await user.should_see("go_on")
        await user.should_not_see(kind=ui.button, content="Go on")


async def test_request_changes_sends_one_operation_per_work_item_and_one_for_a_target_without_work_only_when_chosen(
    replay,
) -> None:
    events, snaps = page("investigation", "approve", "revise", "stop")
    async with replay(events, snaps) as user:
        await user.open("/?view=Investigation")
        user.find(kind=ui.button, content="Request changes").click()
        await user.should_see("rs30: no work item")
        user.find(marker="answering-as").type(ADA)
        user.find(kind=ui.button, content="Preview command").click()
        response = previewed(user)
        # A target without a work item starts unchanged: it sends no operation until one is chosen.
        assert response["operations"] == [
            {"operation": "keep", "work_item_id": f"work-item:{'a' * 64}"},
            {"operation": "keep", "work_item_id": f"work-item:{'b' * 64}"},
            {"operation": "keep", "work_item_id": f"work-item:{'c' * 64}"},
        ]
        assert response["proposal_id"] == f"investigation-proposal:{DIGEST}"
        user.find(marker="operation-rs30").elements.pop().value = "clarify_request"
        user.find(marker="request-rs30").type("rs30 on chromosome 2")
        user.find(kind=ui.button, content="Preview command").click()
        assert previewed(user)["operations"][-1] == {"operation": "clarify_request", "request_member_id": member_id("rs30"),
                                                     "request": {"request_text": "rs30 on chromosome 2"}}  # fmt: skip


async def test_the_form_variant_lays_out_the_chosen_actions_fields_and_shows_the_ids_it_fills(replay) -> None:
    events, snaps = page("investigation_finding")
    async with replay(events, snaps) as user:
        await user.open("/?view=Investigation&decision=form")
        user.find(marker="form-action").elements.pop().value = "approve"
        await user.should_not_see(marker="field-instruction")  # amend's instruction goes once approve's fields are laid out
        await user.should_see(f"collection_member_id = {member_id('rs30')} (from the request)")
        user.find(marker="answering-as").type(ADA)
        user.find(kind=ui.button, content="Preview command").click()
        assert previewed(user) == {"action": "approve", "collection_member_id": member_id("rs30"), "revision_number": 1,
                                   "asserted_by": ADA}  # fmt: skip


async def test_a_quiet_stop_asks_for_its_reason_in_the_confirm_and_previews_it(replay) -> None:
    events, snaps = page("planning")
    async with replay(events, snaps) as user:
        await user.open("/?view=Plan")
        user.find(kind=ui.button, content="Stop").click()
        user.find(kind=ui.textarea, marker="field-reason").type("The assay cannot be told from these reads.")
        user.find(kind=ui.button, content="Confirm stop").click()
        assert previewed(user) == {"action": "stop", "reason": "The assay cannot be told from these reads."}


async def test_use_this_answer_sends_the_option_picked_in_the_radio(replay) -> None:
    events, snaps = page("planning")
    async with replay(events, snaps) as user:
        await user.open("/?view=Plan")
        user.find(ui.radio).elements.pop().value = "wes"
        user.find(kind=ui.button, content="Use this answer").click()
        user.find(kind=ui.button, content="Preview command").click()  # its optional instruction opens first
        assert previewed(user) == {"action": "select", "candidate_id": "wes"}


async def test_a_work_item_two_rsids_share_is_labelled_with_its_own_target(replay) -> None:
    """rs11 was merged into rs20: their one work item is rs20's."""
    events, snaps = page("investigation", "revise")
    async with replay(events, snaps) as user:
        await user.open("/?view=Investigation&decision=form")
        await user.should_see("rs20 NC_000002.12:2,000 T>C")
        await user.should_not_see("rs11 NC_000002.12:2,000 T>C")


async def test_answering_as_is_remembered_for_this_browser(replay) -> None:
    events, snaps = page("junction", "retry", "stop")
    async with replay(events, snaps) as user:
        await user.open("/?view=Investigation")
        user.find(marker="answering-as").type(ADA)
        await user.open("/?view=Investigation")
        assert user.find(marker="answering-as").elements.pop().value == ADA


def written(user, field: str) -> tuple[Path, dict]:
    """The file a previewed finalization answer names, and its content."""
    path = Path(previewed(user)[field])
    return path, json.loads(path.read_text(encoding="utf-8"))


async def test_a_finalization_stop_writes_its_reason_file_and_previews_the_command_naming_it(replay) -> None:
    """Owner, #430 Q1 A: the GUI writes the file the stop names from the typed reason, outside the repository."""
    events, snaps = page("finalization")
    async with replay(events, snaps) as user:
        await user.open("/?view=Results")
        user.find(marker="answering-as").type(ADA)
        user.find(kind=ui.button, content="Stop").click()
        user.find(kind=ui.textarea, marker="field-reason").type("The basis cannot support a report.")
        user.find(kind=ui.button, content="Confirm stop").click()
        response = previewed(user)
        assert {k: v for k, v in response.items() if k != "reason_file"} == {
            "action": "stop", "basis_digest": f"finalization-basis:{DIGEST}", "head_event_id": "request-1"}  # fmt: skip
        path, content = written(user, "reason_file")
        assert path.is_absolute() and path.is_relative_to(Path(os.environ["ANSWER_FILES"]))
        assert content == {"schema_version": "bio-neb.finalization-stop/1", "asserted_by": ADA,
                           "reason": "The basis cannot support a report."}  # fmt: skip


async def test_a_finalization_correction_lists_each_requested_change_in_its_file(replay) -> None:
    events, snaps = page("finalization")
    async with replay(events, snaps) as user:
        await user.open("/?view=Results")
        user.find(marker="answering-as").type(ADA)
        user.find(kind=ui.button, content="Correct").click()
        user.find(kind=ui.textarea, marker="field-reason").type("One admission is not supported.")
        user.find(kind=ui.button, content="Add a change").click()
        second = sorted(user.find(marker="change-kind").elements, key=lambda e: e.id)[1]
        second.value = "new_research"
        await user.should_see(marker="field-description")
        user.find(kind=ui.textarea, marker="field-description").type("Look for a second source for rs20.")
        user.find(kind=ui.button, content="Preview command").click()
        await user.should_see("The action's response_schema accepts this response.")
        _, content = written(user, "correction_file")
        assert content == {
            "schema_version": "bio-neb.finalization-correction/1", "asserted_by": ADA, "reason": "One admission is not supported.",
            "requested_changes": [
                {"action": "set_report_membership", "member_id": f"admitted-observation-evidence:{'d' * 64}", "included": False},
                {"action": "new_research", "description": "Look for a second source for rs20."},
            ],
        }  # fmt: skip


async def test_a_correction_file_its_schema_refuses_is_not_written(replay) -> None:
    events, snaps = page("finalization")
    async with replay(events, snaps) as user:
        await user.open("/?view=Results")
        user.find(marker="answering-as").type(ADA)
        user.find(kind=ui.button, content="Correct").click()
        user.find(kind=ui.button, content="Preview command").click()
        await user.should_see("correction_file: 'reason' is a required property")
        assert not Path(previewed(user)["correction_file"]).exists()


async def test_a_follow_up_approval_offers_only_the_work_items_still_waiting(replay) -> None:
    """After a partial approval the review keeps the whole proposal and names what still waits; the backend refuses
    a selection outside it (Spec r1 P2)."""
    review = schema_review("investigation", "approve")
    review["request"]["remaining_work_item_ids"] = [f"work-item:{'c' * 64}"]
    events = journey("Investigation")
    snap = snapshot("investigation_review_pending", "review_required", reviews=(review,))
    async with replay(events, {len(events): snap}) as user:
        await user.open("/?view=Investigation")
        user.find(kind=ui.button, content="Approve").click()
        assert list(user.find(marker="field-work_item_ids").elements.pop().options) == [f"work-item:{'c' * 64}"]
        user.find(kind=ui.button, content="Preview command").click()
        assert previewed(user)["work_item_ids"] == [f"work-item:{'c' * 64}"]


async def test_request_changes_can_add_a_new_target(replay) -> None:
    events, snaps = page("investigation", "revise")
    async with replay(events, snaps) as user:
        await user.open("/?view=Investigation")
        user.find(marker="answering-as").type(ADA)
        user.find(kind=ui.button, content="Request changes").click()
        user.find(kind=ui.button, content="Add a target").click()
        user.find(marker="new-target").type("Does NA00001 carry rs99?")
        user.find(kind=ui.button, content="Preview command").click()
        assert previewed(user)["operations"][-1] == {"operation": "add", "request": {"request_text": "Does NA00001 carry rs99?"}}
        await user.should_see("The action's response_schema accepts this response.")
