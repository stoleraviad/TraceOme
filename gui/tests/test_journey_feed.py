"""The Journey drawer reads the backend's `journey-feed`: its lines, stages and statuses, and the narrator fallback."""

from __future__ import annotations

import pytest
from conftest import append, wait_to_see
from nicegui import ui
from nicegui.testing import User
from runs import SAMPLE, journey, snapshot, told
from runs import feed as feed_of


async def test_the_story_shows_the_feeds_lines(replay) -> None:
    events = journey("Plan")
    async with replay(events, {len(events): snapshot("planning", "running")}) as user:
        await user.open("/")
        await user.should_see(told("ResearchRequestRegistered"))
        await user.should_not_see("Your question was registered")


async def test_the_stage_chips_follow_the_feeds_stages_with_finalization_under_results(replay) -> None:
    events = journey("Investigation", "FinalizationBasisCompiled")
    async with replay(events, {len(events): snapshot("finalization_review_pending", "running")}) as user:
        await user.open("/")
        assert [chip.text for chip in user.find(marker="current").elements] == ["Report"]
        done = {chip.text for chip in user.find(marker="done").elements}
        assert done == {"Prompt & inputs", "Parabricks run", "Investigation"}


async def test_execution_begins_at_the_feeds_first_execution_fact_not_at_the_approval_answer(replay) -> None:
    events = journey("Approval", "ExecutionHitlResponded")
    async with replay(events, {len(events): snapshot("execution_approved", "running")}) as user:
        await user.open("/")
        assert [chip.text for chip in user.find(marker="current").elements] == ["Parabricks run"]
        assert user.find(marker="current").elements.pop().props["data-backend-phase"] == "Approval"


def row_icon(user: User, text: str) -> str:
    """The icon of the Journey row whose line is `text` (a row holds its icon, then its lines, then its time)."""
    (line,) = user.find(kind=ui.label, marker="journey-event-title", content=text).elements
    row = line.parent_slot.parent.parent_slot.parent
    return row.default_slot.children[0].props["name"]


async def test_each_feed_status_shows_its_icon(replay) -> None:
    events = journey("Plan", "PlanningHitlRequested", "PlanningHitlResponded", "AttemptFailed", "ExecutionPathRecorded")
    feed = feed_of(events)
    feed["items"][-2]["status"] = "verified"  # the execution fact
    async with replay(events, {len(events): snapshot("planning", "running")}, journey_feed=(feed, None)) as user:
        await user.open("/")
        assert row_icon(user, told("PlanningHitlRequested")) == "front_hand"
        assert row_icon(user, told("PlanningHitlResponded")) == "person"
        assert row_icon(user, told("AttemptFailed")) == "error_outline"
        assert row_icon(user, told("ExecutionPathRecorded")) == "verified"
        assert row_icon(user, told("ResearchRequestRegistered")) == "check"


async def test_the_question_and_plan_views_read_their_details_from_the_journey_lines(replay) -> None:
    """The feed has one line per item; the views read the prompt, the input sizes and the planning exchange from the
    Journey events themselves."""
    events = journey("Plan", "InputMetadataInspected", "PlanningHitlRequested", "PlanningHitlResponded")
    events[-4]["payload"] = {"result": {"status": "succeeded", "inputs": [{"role": "read_1", "size_bytes": 1_914_722_761}]}}
    events[-3]["payload"] = {"question": "Which assay and read layout?", "issue": "Not in the file metadata.",
                             "candidates": [{"candidate_id": "c1", "display_label": "gDNA / paired"}]}  # fmt: skip
    events[-2]["payload"] = {"request_event_id": events[-3]["event_id"], "response": {"action": "select", "candidate_id": "c1"}}
    async with replay(events, {len(events): snapshot("planning", "running")}) as user:
        await user.open("/?view=Question")
        await user.should_see(f"Sample {SAMPLE}, short-read germline WGS. Which variants from the attached list does it carry?")
        await user.should_see("read_1: 1.9 GB")
        await user.open("/?view=Plan")
        await user.should_see("Which assay and read layout?")
        await user.should_see("gDNA / paired")


async def test_a_refused_feed_falls_back_to_the_guis_own_narration_and_says_so(replay) -> None:
    events = journey("Plan")
    async with replay(
        events, {len(events): snapshot("planning", "running")}, journey_feed=(None, "journey_review_invalid")
    ) as user:
        await user.open("/")
        await user.should_see(
            "The backend's Journey feed refused this Run (journey_review_invalid); the GUI's own narration is shown."
        )
        await user.should_see("Your question was registered")
        await user.should_not_see(told("ResearchRequestRegistered"))


async def test_live_mode_appends_the_feeds_new_items_to_the_drawer_without_rebuilding_it(follow) -> None:
    events = journey("Plan", "PlanningHitlRequested")
    async with follow(events[:4]) as user:
        await user.open("/")
        await wait_to_see(user, told("AttemptStarted"))
        (row,) = user.find(kind=ui.label, marker="journey-event-title", content=told("AttemptStarted")).elements
        append(events[4:])
        await wait_to_see(user, told("PlanningHitlRequested"))
        assert user.find(kind=ui.label, marker="journey-event-title", content=told("AttemptStarted")).elements == {row}
        await user.should_not_see("refused this Run")


async def test_live_mode_falls_back_to_the_narration_when_the_feed_starts_refusing(follow) -> None:
    events = journey("Plan")
    answers = iter([(feed_of(events[:3]), None), (None, "journey_review_invalid")])
    async with follow(events[:3], journey_feed=lambda journey_file: next(answers)) as user:
        await user.open("/")
        await wait_to_see(user, told("ResearchRequestRegistered"))
        append(events[3:])
        await wait_to_see(
            user, "The backend's Journey feed refused this Run (journey_review_invalid); the GUI's own narration is shown."
        )
        await user.should_see("Your question was registered")
        await user.should_not_see(told("ResearchRequestRegistered"))


@pytest.mark.parametrize("refused", [None, "journey_review_invalid"], ids=["feed", "fallback"])
async def test_a_failed_input_inspection_shows_its_failures_not_sizes(replay, refused) -> None:
    """The inspection result is one of two shapes, told apart by `status`; a failed one has failures, no inputs."""
    events = journey("Question", "InputMetadataInspected")
    events[-2]["payload"] = {"result": {"status": "failed", "failures": [
        {"code": "input_missing", "role": "read_2", "registered_path": "/runs/read_2.fastq.gz"},
        {"code": "input_paths_duplicate", "role": None, "registered_path": "/runs/read_1.fastq.gz"}]}}  # fmt: skip
    async with replay(
        events, {len(events): snapshot("input_review_pending", "running")}, (None, refused) if refused else None
    ) as user:
        await user.open("/?view=Question")
        await user.should_see("read_2: inspection failed (input_missing)")
        await user.should_see("inputs: inspection failed (input_paths_duplicate)")
        if refused:
            await user.should_see(f"The backend's Journey feed refused this Run ({refused}); the GUI's own narration is shown.")


async def test_a_feed_stage_or_status_from_later_work_renders_plainly(replay) -> None:
    """A stage the chips do not know stays on the stage before it; a status without an icon gets a plain one."""
    events = journey("Plan", "AReviewFromLaterWork", "AStepFromLaterWork")
    feed = feed_of(events)
    feed["items"][-3]["stage"] = "a_stage_from_later_work"
    feed["items"][-3]["status"] = "a_status_from_later_work"
    feed["items"][-2]["stage"] = feed["items"][-1]["stage"] = "a_stage_from_later_work"
    async with replay(events, {len(events): snapshot("planning", "running")}, journey_feed=(feed, None)) as user:
        await user.open("/")
        assert [chip.text for chip in user.find(marker="current").elements] == ["Parabricks run"]
        assert row_icon(user, told("AReviewFromLaterWork")) == "circle"
        assert row_icon(user, told("AStepFromLaterWork")) == "check"
        await user.should_not_see(marker="needs-you")
