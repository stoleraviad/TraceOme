"""Decision buttons start the CLI's writing commands (#431), here the fake CLI (`fake_cli.py`): `answer` detached, the
decision buttons disabled while it runs, and on its exit the page shows what the next `observe` says."""

from __future__ import annotations

import asyncio
import json

from conftest import append, wait_to_see
from nicegui import ui
from runs import JOURNEY_ID, journey, schema_review, snapshot, then

import cli_bridge

ADA = "researcher:ada"


def waiting(kind: str, *names: str) -> tuple[list[dict], dict]:
    """A Run that reached Execution and waits for a `kind` review with these actions (all of the kind's when none)."""
    events = journey("Execution")
    return events, snapshot(f"{kind}_review_pending", "review_required", reviews=(schema_review(kind, *names),))


def junction() -> tuple[list[dict], dict]:
    """A failed agent's junction review, asked where the Run is: retry or stop."""
    return waiting("junction", "retry", "stop")


def answered(argv: list[str]) -> tuple[list[str], dict]:
    """An `answer` argv without its response, and the response."""
    assert argv[-2] == "--response-json"
    return argv[:-1], json.loads(argv[-1])


async def retry(user) -> None:
    await user.open("/")
    await wait_to_see(user, "Waiting for you")
    user.find(marker="answering-as").type(ADA)
    user.find(kind=ui.button, content="Retry").click()
    user.find(kind=ui.button, content="Send answer").click()


async def test_a_decision_starts_answer_and_the_page_then_shows_the_next_observe(answering, tmp_path) -> None:
    events, waiting = junction()
    step = {"command": "answer", "append": then(events, "PassStarted", "ExecutionPathRecorded", "PassClosed"),
            "snapshot": snapshot("execution_retried", "progress"), "stdout": {"frontier": "progress"}}  # fmt: skip
    async with answering(events, waiting, [step]) as (user, fake):
        await retry(user)
        await fake.started()
        journey_file = str(tmp_path / "runs" / JOURNEY_ID / "journey.jsonl")
        response = {"state": "answered", "action": "retry", "request_id": "failure-request:1", "by": ADA}
        (argv,) = fake.calls()
        assert answered(argv) == (
            ["answer", "--journey-file", journey_file, "--request-event-id", "request-1", "--response-json"], response)  # fmt: skip
        await wait_to_see(user, "execution retried")
        await user.should_not_see(kind=ui.button, content="Retry")


def decision_buttons(user) -> dict[str, bool]:
    """Each decision button on the page and whether it can be clicked."""
    names = ("Retry", "Stop", "Send answer")
    return {b.text: b.enabled for b in user.find(kind=ui.button).elements if b.text in names}


async def wait_for(condition, what: str) -> None:
    for _ in range(50):
        if condition():
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


async def test_decision_buttons_are_disabled_while_the_answer_runs_and_a_refusal_keeps_the_review(answering) -> None:
    events, waiting = junction()
    refused = {"status": "failed", "frontier": "refused", "code": "review_response_invalid", "fields": ["by"]}
    step = {"command": "answer", "until": "release", "exit": 2, "stderr": refused}
    async with answering(events, waiting, [step]) as (user, fake):
        await retry(user)
        await fake.started()
        await wait_for(lambda: decision_buttons(user) == {"Retry": False, "Stop": False, "Send answer": False}, "disabled")
        await wait_to_see(user, "Running: answer retry")
        fake.release()
        await wait_to_see(user, "answer retry: review_response_invalid (by)")
        # No Journey line came, so no pass end: the page observed because the command ended.
        await wait_for(lambda: [argv[0] for argv in fake.all_calls()][-2:] == ["answer", "observe"], "an observe after the exit")
        await wait_for(lambda: decision_buttons(user) == {"Retry": True, "Stop": True, "Send answer": True}, "enabled again")
        await user.should_see("Waiting for you")
        assert user.find(marker="answering-as").elements.pop().value == ADA


async def test_interrupt_signals_the_commands_process_group_and_the_held_run_offers_resume(answering, tmp_path) -> None:
    """The fake CLI's wrapper ignores SIGINT, as `uv run` stands between the GUI and the CLI: only a signal to the
    whole group reaches the command, which then holds the Run (#327)."""
    events, waiting = junction()
    interrupted = then(events, "PassStarted", "PassClosed")
    held = snapshot("run_held", "held", failure_code="interrupted")
    resumed = then(events + interrupted, "PassStarted", "ExecutionPathRecorded", "PassClosed")
    steps = [{"command": "answer", "until": "release", "append": interrupted, "snapshot": held},
             {"command": "resume", "append": resumed, "snapshot": snapshot("execution_resumed", "progress")}]  # fmt: skip
    async with answering(events, waiting, steps) as (user, fake):
        await retry(user)
        await fake.started()
        await wait_to_see(user, "Running: answer retry")
        user.find(kind=ui.button, content="Interrupt").click()
        await wait_to_see(user, "answer retry: interrupted")
        await wait_to_see(user, "The Run is held: interrupted.")
        user.find(kind=ui.button, content="Resume").click()
        await fake.started(2)
        journey_file = str(tmp_path / "runs" / JOURNEY_ID / "journey.jsonl")
        assert fake.calls()[1] == ["resume", "--journey-file", journey_file]
        await wait_to_see(user, "execution resumed")
        await user.should_not_see(kind=ui.button, content="Resume")


async def test_a_response_its_schema_refuses_is_not_started(answering) -> None:
    events, snap = waiting("planning")
    async with answering(events, snap, []) as (user, fake):
        await user.open("/?view=Plan")
        await wait_to_see(user, "Waiting for you")
        user.find(kind=ui.button, content="Clarify").click()
        user.find(kind=ui.button, content="Send answer").click()
        await user.should_see("Not started:")
        await user.should_see("response: 'instruction' is a required property")
        await asyncio.sleep(0.5)
        assert fake.calls() == []


async def test_an_observe_still_running_when_the_command_ends_cannot_hide_the_runs_next_state(answering, monkeypatch) -> None:
    """A pass-end observe that read the Run before the command ended and answers after it must not replace the state
    the command left (Spec r1 P1, Standards r1 S1)."""
    events, waiting = junction()
    held = snapshot("run_held", "held", failure_code="interrupted")
    interrupted = {"status": "failed", "code": "interrupted", "fields": ["answer"]}
    step = {"command": "answer", "until": "release", "exit": 130, "stderr": interrupted, "snapshot": held}
    observe, slow_read, let_it_answer = cli_bridge.observe, asyncio.Event(), asyncio.Event()

    async def slow_first(journey_file):
        result = await observe(journey_file)  # what the Run was when this read began
        if not slow_read.is_set():
            slow_read.set()
            await let_it_answer.wait()
        return result

    async with answering(events, waiting, [step]) as (user, fake):
        await retry(user)
        await fake.started()
        monkeypatch.setattr(cli_bridge, "observe", slow_first)
        append(then(events, "PassStarted", "PassClosed"))  # a pass end: its observe reads the review still pending
        try:
            await asyncio.wait_for(slow_read.wait(), timeout=5)
            fake.release()
            await wait_to_see(user, "answer retry: interrupted")
            await asyncio.sleep(1)  # time for any observe after the exit
        finally:
            let_it_answer.set()
        await wait_to_see(user, "The Run is held: interrupted.")
        await asyncio.sleep(1)  # the slow read has answered: the held state stays
        await user.should_see("The Run is held: interrupted.")
        await user.should_not_see(kind=ui.button, content="Retry")
