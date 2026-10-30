"""Readable evidence through the normal replay UI; no backend or scientific decisions."""

from nicegui import ui
from runs import event, journey, snapshot


async def test_a_decision_page_explains_the_choice_without_showing_raw_json(replay) -> None:
    events = journey("Execution")
    n = len(events) + 1
    events.append(
        event(
            n,
            "ExecutionHitlResponded",
            {
                "authority": "user_decision",
                "authorizes_execution": True,
                "scientifically_verified": False,
                "request_event_id": "event-2",
                "response": {"action": "approve", "proposal_id": "proposal:test"},
            },
        )
    )
    async with replay(events, {len(events): snapshot("execution_complete", "working")}) as user:
        await user.open(f"/?section=evidence&record={n}")
        await user.should_see("Researcher decision")
        await user.should_see("Approve")
        await user.should_see("Not established by this decision")
        await user.should_see("Linked records")
        await user.should_not_see(kind=ui.code)
        user.find("Show JSON excerpt").click()
        await user.should_see('"authorizes_execution": true')


async def test_variant_measurements_and_limits_are_taken_from_that_event_only(replay) -> None:
    events = journey("Investigation")
    n = len(events) + 1
    events.append(
        event(
            n,
            "InvestigationEvidenceRecorded",
            {
                "work_item": {
                    "sample_id": "sample-A",
                    "target": {"display_label": "rs42", "contig": "chr2", "one_based_position": 405, "ref": "T", "alt": "C"},
                },
                "sample_observation": {
                    "result": {
                        "record": {
                            "genotypes": [
                                {"sample_name": "sample-A", "genotype": "1/1", "dp": 18, "gq": 7},
                            ]
                        }
                    }
                },
                "assessment": {"outcome": "exact_allele_observed"},
                "admission": {"status": "admitted", "claim_scope": "The observed C allele in this VCF only."},
            },
        )
    )
    async with replay(events, {n: snapshot("investigating", "working")}) as user:
        await user.open(f"/?section=evidence&record={n}")
        await user.should_see("Variant evidence")
        await user.should_see("The observed C allele in this VCF only.")
        await user.should_see("Not recorded in this event")
        table = user.find(kind=ui.table).elements.pop()
        assert table.rows == [{"sample": "sample-A", "genotype": "1/1", "depth": 18, "quality": 7}]


async def test_model_content_stays_hidden_and_json_is_bounded_by_bytes(replay) -> None:
    events = journey("Plan")
    n = len(events) + 1
    events.append(event(n, "ModelRequestRecorded", {"request": "retained-model-content-" + "λ" * 10000}))
    async with replay(events, {n: snapshot("planning", "working")}) as user:
        await user.open(f"/?section=evidence&record={n}")
        await user.should_see("Model payload retained")
        await user.should_not_see("retained-model-content-")
        await user.should_not_see(kind=ui.code)
        user.find("Show JSON excerpt").click()
        await user.should_see("Excerpt limited to the first 4 KB.")
        code = user.find(kind=ui.code).elements.pop()
        assert len(code.content.encode("utf-8")) <= 4000


async def test_evidence_beyond_the_replay_cursor_is_unavailable(replay) -> None:
    events = journey("Plan")
    n = len(events) + 1
    events.append(event(n, "ModelRequestRecorded", {"request": "future-only-content"}))
    async with replay(events, {len(events): snapshot("planning", "working")}) as user:
        await user.open(f"/?section=evidence&record={n}&at={n - 1}")
        await user.should_see("This record is not available at this point in the run.")
        await user.should_not_see("future-only-content")


async def test_replay_movement_refreshes_the_record_and_its_chronological_context(replay) -> None:
    events = journey("Plan")
    n = len(events) + 1
    events.extend(
        [
            event(n, "PlanningHitlResponded", {"response": {"action": "clarify", "instruction": "future-record-sentinel"}}),
            event(n + 1, "PlanningHitlRequested"),
            event(n + 2, "ModelRequestRecorded"),
        ]
    )
    async with replay(events, {n - 1: snapshot("planning", "working")}) as user:
        await user.open(f"/?section=evidence&record={n}")
        await user.should_see("future-record-sentinel")
        await user.should_see("feed says PlanningHitlRequested")
        slider = user.find(kind=ui.slider).elements.pop()
        slider.set_value(n)
        await user.should_not_see("feed says PlanningHitlRequested")
        slider.set_value(n - 1)
        await user.should_see("This record is not available at this point in the run.")
        await user.should_not_see("future-record-sentinel")
        slider.set_value(n)
        await user.should_see("future-record-sentinel")


async def test_window_records_show_the_measurements_and_their_scope(replay) -> None:
    events = journey("Investigation")
    n = len(events) + 1
    events.append(
        event(
            n,
            "InvestigationItemWindowRecorded",
            {
                "result": {
                    "target": {"display_label": "rs42", "contig": "chr2"},
                    "window_start": 305,
                    "window_end": 505,
                    "claim_scope": "Only the retained VCF window.",
                    "records": [
                        {
                            "contig": "chr2",
                            "one_based_position": 405,
                            "ref": "T",
                            "alts": ["C"],
                            "genotypes": [{"sample_name": "sample-A", "genotype": "0/1", "dp": 26, "gq": 99}],
                        }
                    ],
                }
            },
        )
    )
    async with replay(events, {}) as user:
        await user.open(f"/?section=evidence&record={n}")
        await user.should_see("Nearby VCF records")
        await user.should_see("Only the retained VCF window.")
        rows = user.find(kind=ui.table).elements.pop().rows
        assert rows[0]["depth"] == 26
        assert rows[0]["alleles"] == "T>C"


async def test_reading_notes_are_presented_as_agent_readings(replay) -> None:
    events = journey("Investigation")
    n = len(events) + 1
    events.append(
        event(
            n,
            "InvestigationTableChunkRead",
            {
                "readings": [
                    {"work_item_id": "work-42", "reading": "concern", "concerns": ["low_read_depth"], "note": "Depth is low."}
                ]
            },
        )
    )
    async with replay(events, {}) as user:
        await user.open(f"/?section=evidence&record={n}")
        await user.should_see("Agent reading notes")
        await user.should_see("Depth is low.")
        await user.should_see("low read depth")
        await user.should_see("These readings do not independently validate a scientific claim.")
