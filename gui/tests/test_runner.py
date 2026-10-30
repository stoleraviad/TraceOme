"""The default runner over the existing public replay seam."""

from runs import SAMPLE, grounded_report, journey, snapshot


async def test_the_default_runner_shows_the_whole_collection_with_a_persistent_journey(replay) -> None:
    events = journey("Results")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded_report())
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see(SAMPLE)
        await user.should_see("1 target allele observed.")
        await user.should_see("Results for all 4 requested targets.")
        assert len(user.find(marker="compact-finding").elements) == 4
        assert len(user.find(marker="journey-stage").elements) == 6


async def test_an_in_progress_run_has_a_useful_screen_before_the_first_snapshot(replay) -> None:
    events = journey("Plan")[:-1]
    async with replay(events, {}) as user:
        await user.open("/")
        await user.should_see("Planning in progress")
        await user.should_see("The Journey continues to record the path while this step runs.")


async def test_the_compact_journey_keeps_technical_activity_accessible(replay) -> None:
    events = journey("Plan", "ModelRequestRecorded")
    async with replay(events, {}) as user:
        await user.open("/")
        assert len(user.find(marker="journey-event-title").elements) < len(events)
        user.find("all events").click()
        await user.should_see("model payloads hidden")
        assert len(user.find(marker="journey-event-title").elements) == len(events)
