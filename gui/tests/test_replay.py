"""Replay of small synthetic Runs through the page: stage, banner and targets at a pass end."""

from __future__ import annotations

import pytest
from nicegui import ui
from runs import (
    QUESTION_TEXT,
    SAMPLE,
    action,
    candidate,
    execution_proposal,
    investigation_question,
    investigation_request,
    journey,
    member,
    proposal,
    review,
    snapshot,
)

from journey_feed import PHASES


def planning_review() -> dict:
    request = {"question": "Is the input whole-genome sequencing?", "issue": "The read headers do not name the assay.",
               "candidates": [{"candidate_id": "wgs", "display_label": "Yes, WGS"}, {"candidate_id": "wes", "display_label": "No, exome"}]}  # fmt: skip
    return review("planning", request, action("select", candidate_id={"enum": ["wgs", "wes"]}), action("stop"))


async def test_a_planning_review_shows_the_question_and_marks_plan_as_needing_you(replay) -> None:
    events = journey("Plan")
    snap = snapshot("planning_hitl_pending", "review_required", reviews=(planning_review(),))
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see("Waiting for you: answer a planning question")
        await user.should_see("Is the input whole-genome sequencing?")
        await user.should_see("Use this answer")
        assert [chip.text for chip in user.find(marker="needs-you").elements] == ["Parabricks run"]


# Every review kind the backend's `observe` can emit (`src/bio_neb/application/reviews.py`): the stage the Run
# is at when it is asked, where the GUI asks it, the banner, the kind's legal actions in the backend
# (`_legal_actions` and each request's own list) and the button each one gets.
KINDS = [
    ("input", "Question", "Question", "correct the inputs", {"retry": "Retry", "amend": "Amend", "abort": "Abort"}),
    ("paper", "Plan", "Question", "decide on a paper that could not be read",
     {"retry": "Retry", "amend": "Amend", "continue": "Continue", "abort": "Abort"}),
    ("planning", "Plan", "Plan", "answer a planning question",
     {"select": "Use this answer", "clarify": "Clarify", "reject": "Reject", "stop": "Stop"}),
    ("model_availability", "Plan", "Plan", "decide on model availability", {"retry": "Retry", "amend": "Amend", "stop": "Stop"}),
    ("agent_effort", "Investigation", "Investigation", "decide on the agent's effort", {"extend": "Extend", "stop": "Stop"}),
    ("agent_work", "Plan", "Plan", "decide on the agent's work", {"continue": "Continue", "reject": "Reject"}),
    ("execution", "Approval", "Approval", "approve the plan", {"approve": "Approve and start", "revise": "Request changes",
                                                              "reject": "Reject"}),
    ("preparation", "Execution", "Approval", "supply what the reference preparation needs", {"supply": "Supply", "stop": "Stop"}),
    ("execution_path", "Execution", "Execution", "decide on the execution path",
     {"approve_cpu_preparation": "Approve cpu preparation", "abort": "Abort"}),
    ("junction", "Plan", "Plan", "decide on a failed step", {"retry": "Retry", "amend": "Amend", "stop": "Stop"}),
    ("investigation", "Investigation", "Investigation", "review the investigation",
     {"approve": "Approve", "amend": "Amend", "reject": "Reject"}),
    ("investigation_download", "Investigation", "Investigation", "decide on a failed download", {"retry": "Retry", "abort": "Abort"}),
    ("run_health", "Investigation", "Investigation", "decide on a failed health measurement", {"retry": "Retry", "abort": "Abort"}),
    ("finalization", "Investigation", "Results", "approve the report basis", {"approve": "Approve", "correct": "Correct", "stop": "Stop"}),
    ("a_kind_from_later_work", "Execution", "Execution", "answer a review (a kind from later work)", {}),
]  # fmt: skip


def kind_request(kind: str) -> tuple[dict, str | None]:
    """The request parts the page reads for a kind, and the text it then shows. The GUI reads a planning
    question, an execution proposal and an investigation question; other kinds' request fields are left out."""
    if kind == "planning":
        return planning_review()["request"], "Is the input whole-genome sequencing?"
    if kind == "execution":
        return {"proposal": execution_proposal()}, "Parabricks germline → VCF"
    if kind == "investigation":
        return {"proposal": None, "question": investigation_question("rs30", issue="refsnp: not_found")}, QUESTION_TEXT
    return {}, None


@pytest.mark.parametrize(("kind", "run_at", "asked_at", "banner", "buttons"), KINDS, ids=[k[0] for k in KINDS])
async def test_each_review_kind_is_asked_at_its_stage_with_its_actions(replay, kind, run_at, asked_at, banner, buttons) -> None:
    events = journey(run_at)
    request, shows = kind_request(kind)
    pending = review(kind, request, *(action(name) for name in buttons))
    snap = snapshot(f"{kind}_review_pending", "review_required", reviews=(pending,))
    async with replay(events, {len(events): snap}) as user:
        await user.open(f"/?view={asked_at}")
        await user.should_see(f"Waiting for you: {banner}")
        chip = {"Question": "Prompt & inputs", "Plan": "Parabricks run", "Approval": "Parabricks run",
                "Execution": "Parabricks run", "Investigation": "Investigation", "Results": "Report"}[asked_at]  # fmt: skip
        # A failed health measurement is asked on Health, whichever stage shows it (`REVIEW_SECTION`).
        assert [chip.text for chip in user.find(marker="needs-you").elements] == ["Health" if kind == "run_health" else chip]
        if shows:
            await user.should_see(shows)
        for label in buttons.values():
            assert user.find(kind=ui.button, content=label).elements, f"no button {label!r}"


def three_targets() -> tuple[dict, ...]:
    """rs10 has two ALTs (two work items), rs20 one, rs30 none (RefSNP did not find it), and rs11 was merged
    into rs20, so it shares rs20's work item."""
    return (
        member("rs10", "accepted", ("work-item:10A", candidate("NC_000001.11", 1000, "G", "A")),
               ("work-item:10T", candidate("NC_000001.11", 1000, "G", "T"))),
        member("rs20", "accepted", ("work-item:20C", candidate("NC_000002.12", 2000, "T", "C"))),
        member("rs30", "not_found"),
        member("rs11", "merged", ("work-item:20C", candidate("NC_000002.12", 2000, "T", "C"))),
    )  # fmt: skip


async def test_an_investigation_review_shows_one_row_per_work_item_with_its_authorisation(replay) -> None:
    events = journey("Investigation")
    pending = review("investigation", {"proposal": proposal(*three_targets()), "remaining_work_item_ids": ["work-item:10T"]},
                     action("approve"), action("revise", instruction={"type": "string"}), action("stop"))  # fmt: skip
    authority = {"authorized_work_items": [{"work_item_id": "work-item:10A"}, {"work_item_id": "work-item:20C"}]}
    snap = snapshot("investigation_review_pending", "review_required", reviews=(pending,),
                    investigation_request=investigation_request("rs10", "rs20", "rs30", "rs11"),
                    investigation={"status": "review_pending", "proposals": [], "authority": authority})  # fmt: skip
    async with replay(events, {len(events): snap}) as user:
        await user.open("/?targets=table")
        await user.should_see("Waiting for you: review the investigation")
        await user.should_see("4 targets · 3 work items")
        rows = user.find(ui.table).elements.pop().rows
        assert [(r["rsid"], r["where"], r["authorised"]) for r in rows] == [
            ("rs10", "NC_000001.11:1,000 G>A", "yes"),
            ("rs10", "NC_000001.11:1,000 G>T", "no"),
            ("rs20", "NC_000002.12:2,000 T>C", "yes"),
            ("rs30", "–", "no work item"),
            ("rs11", "NC_000002.12:2,000 T>C", "yes"),
        ]
        assert len({r["row"] for r in rows}) == len(rows)
        await user.should_see("1 work items still wait for your answer.")
        await user.should_see("Request changes")


async def test_a_finding_question_before_any_proposal_shows_the_requested_targets(replay) -> None:
    """Lane M's per-target finding question (the #141 line-1024 shape): no proposal exists yet."""
    events = journey("Investigation")
    question = investigation_question("rs30", issue="refsnp: not_found")
    approve = action("approve")
    approve["response_schema"]["description"] = "Approve this finding: checked, not verified."
    pending = review("investigation", {"proposal": None, "question": question}, approve,
                     action("amend", instruction={"type": "string"}), action("reject", reason={"type": "string"}))  # fmt: skip
    snap = snapshot("investigation_review_pending", "review_required", reviews=(pending,),
                    investigation_request=investigation_request("rs10", "rs20", "rs30"),
                    investigation={"status": "review_pending", "proposals": [], "authority": None})  # fmt: skip
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see("Requested targets (3)")
        assert sorted(chip.text for chip in user.find(marker="target").elements) == ["rs10", "rs20", "rs30"]
        await user.should_see("3 targets")
        await user.should_see(QUESTION_TEXT)
        await user.should_see("rs30: refsnp: not_found · NCBI RefSNP did not find rs30.")
        await user.should_see("Amend")
        await user.should_see("Reject")
        await user.should_see("Approve: Approve this finding: checked, not verified.")
        await user.should_not_see("Authorises observing the selected targets")


async def test_the_runs_list_counts_the_targets_observe_reports_not_rsids_in_the_prompt(replay) -> None:
    events = journey("Plan")
    snap = snapshot("planning_hitl_pending", "review_required", reviews=(planning_review(),),
                    investigation_request=investigation_request("rs10", "rs20", "rs30"))  # fmt: skip
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see(f"{SAMPLE} · 3 targets")


@pytest.mark.parametrize("stage", PHASES)
async def test_a_held_run_shows_its_status_plainly_on_every_stage(replay, stage) -> None:
    """`run_held` (frontier `held`): no review, no report projection; every stage view still renders."""
    events = journey("Investigation")
    snap = snapshot("run_held", "held", investigation_request=investigation_request("rs10"))
    async with replay(events, {len(events): snap}) as user:
        await user.open(f"/?view={stage}")
        await user.should_see("run held")
        await user.should_not_see(marker="needs-you")


async def test_a_status_from_later_work_shows_as_plain_text(replay) -> None:
    events = journey("Execution")
    snap = snapshot("a_status_from_later_work", "a_frontier_from_later_work")
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see("a status from later work")


async def test_a_question_about_intent_without_an_issue_shows_with_its_actions(replay) -> None:
    """Lane M can ask about a target's intent before any proposal; that question has no `issue`."""
    events = journey("Investigation")
    pending = review("investigation", {"proposal": None, "question": investigation_question("rs20")},
                     action("clarify"), action("reject"))  # fmt: skip
    snap = snapshot("investigation_review_pending", "review_required", reviews=(pending,),
                    investigation_request=investigation_request("rs10", "rs20"),
                    investigation={"status": "review_pending", "proposals": [], "authority": None})  # fmt: skip
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see("Which variant do you mean by rs20?")
        await user.should_see("rs20: NCBI RefSNP did not find rs20.")
        await user.should_see("Clarify")
        await user.should_see("Reject")


async def test_the_approval_stage_shows_the_execution_plan_during_an_investigation_review(replay) -> None:
    events = journey("Investigation")
    pending = review("investigation", {"proposal": proposal(*three_targets())}, action("approve"), action("stop"))
    snap = snapshot("investigation_review_pending", "review_required", reviews=(pending,), execution_proposal=execution_proposal(),
                    investigation_request=investigation_request("rs10", "rs20", "rs30", "rs11"),
                    investigation={"status": "review_pending", "proposals": [], "authority": None})  # fmt: skip
    async with replay(events, {len(events): snap}) as user:
        await user.open("/?view=Approval")
        await user.should_see("Parabricks germline → VCF")
        assert [chip.text for chip in user.find(marker="needs-you").elements] == ["Investigation"]
