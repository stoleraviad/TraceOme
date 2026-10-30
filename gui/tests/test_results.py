"""The Results page: the published Grounded report from `observe`, and the Run's files from `run-artifacts`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from conftest import append, wait_to_see
from nicegui import ui
from runs import SAMPLE, grounded_report, journey, run_artifacts, snapshot

import cli_bridge
import report


def published(**report) -> tuple[list[dict], dict[int, dict]]:
    events = journey("Results")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded_report(**report))
    return events, {len(events): snap}


def labels(user, marker: str) -> list[str]:
    """The texts of the labels inside the one element marked `marker`, in page order."""
    return [e.text for e in user.find(marker=marker).elements.pop().descendants() if isinstance(e, ui.label)]


async def test_the_table_variant_shows_one_finding_row_per_work_item_in_the_backends_words(replay) -> None:
    async with replay(*published()) as user:
        await user.open("/?findings=table")
        await user.should_see("Grounded report published")
        rows = user.find(marker="findings").elements.pop().rows
        assert [(r["target"], r["where"], r["genotype"], r["finding"], r["reading"], r["limitation"]) for r in rows] == [
            ("rs10", "NC_000001.11:1000 G>A", "0/1 · GQ 99 · DP 30", f"Sample {SAMPLE} carries the target ALT G>A with genotype 0/1.",
             "no concern · NC_000001.11:1000 G>A: exact_allele_observed", "Exact allele and genotype observation in this VCF only."),
            ("rs10", "NC_000001.11:1000 G>T", "0/1 · GQ 99 · DP 30", f"Sample {SAMPLE} does not carry the target ALT G>T.",
             "concern: nearby variant · NC_000001.11:1000 G>T: target_allele_not_carried",
             "This sample's genotype at the requested REF does not carry the target ALT."),
            ("rs20", "NC_000002.12:2000 T>C", "–", "not admitted: no record is no claim",
             "concern: absence not established · NC_000002.12:2000 T>C: no_record",
             "No record was observed at this locus in this variant-only VCF."),
            ("rs30", "–", "–", "unresolved: refsnp: not_found (approved, checked not verified)", "–", "–"),
            ("rs40", "–", "–", "rejected: the researcher rejected this target", "–", "–"),
        ]  # fmt: skip
        assert len({r["row"] for r in rows}) == len(rows)


async def test_detailed_mode_shows_a_full_card_per_requested_target(replay) -> None:
    """Detailed mode preserves the complete card per requested target; the normal runner uses compact rows."""
    async with replay(*published()) as user:
        await user.open("/?density=detailed")
        await user.should_not_see(marker="findings")
        cards = [
            [e.text for e in card.descendants() if isinstance(e, ui.label)]
            for card in sorted(user.find(marker="finding-card").elements, key=lambda e: e.id)
        ]
        assert [card[0] for card in cards] == ["rs10", "rs20", "rs30", "rs40"]
        assert cards[0][1:] == [
            "NC_000001.11:1000 G>A · 0/1 · GQ 99 · DP 30",
            f"Sample {SAMPLE} carries the target ALT G>A with genotype 0/1.",
            "no concern · NC_000001.11:1000 G>A: exact_allele_observed",
            "Exact allele and genotype observation in this VCF only.",
            "NC_000001.11:1000 G>T · 0/1 · GQ 99 · DP 30",
            f"Sample {SAMPLE} does not carry the target ALT G>T.",
            "concern: nearby variant · NC_000001.11:1000 G>T: target_allele_not_carried",
            "This sample's genotype at the requested REF does not carry the target ALT.",
        ]
        assert cards[2][1:] == ["unresolved: refsnp: not_found (approved, checked not verified)"]


async def test_the_report_lists_its_decisions_what_was_not_carried_through_and_its_limitations(replay) -> None:
    async with replay(*published()) as user:
        await user.open("/")
        decided = labels(user, "decisions")
        assert decided == ["Decisions", "Report basis: approve (finalization review)",
                           "rs30 (refsnp: not_found): approved by researcher:test",
                           "rs40 (refsnp: incompatible): rejected by researcher:test"]  # fmt: skip
        stopped = labels(user, "not-carried")
        assert stopped == ["Not carried through", "rs30: unresolved · refsnp: not_found (approved, checked not verified)",
                           "rs40: rejected · the researcher rejected this target"]  # fmt: skip
        await user.should_see("• Coverage, callability, biological absence, function, causality, and diagnosis are not inferred.")
        await user.should_see("• Conflict: The sample does not carry the requested target ALT in this VCF.")
        # A case keeps no sentence in `grounded-report/7`: its typed origin's fields are listed.
        await user.should_see("multi alt: request rs10; alts A, T")
        await user.should_see("Coverage was not measured for Work item work-item:10A.")


async def test_the_runs_files_are_listed_from_run_artifacts(replay) -> None:
    async with replay(*published(), artifacts=(run_artifacts(), None)) as user:
        await user.open("/")
        rows = user.find(marker="artifact-list").elements.pop().rows
        assert [(r["group"], r["role"], r["location"], r["size"]) for r in rows] == [
            ("outputs", "VCF", "s3://bio-neb-test/execution/test/variants.vcf", "48.8 MB"),
            ("inputs", "read_1", "s3://bio-neb-test/inputs/read_1.fastq.gz", "2.0 GB"),
            ("reference", "reference_fasta", "s3://bio-neb-test/reference/genome.fa", "3.1 GB"),
            ("local", "VCF", "investigation/vcf/variants.vcf.gz", "12.0 MB"),
        ]
        await user.should_see("Bucket bio-neb-test · execution/test")


async def test_a_refused_run_artifacts_keeps_the_report_and_says_why_the_files_are_missing(replay) -> None:
    async with replay(*published(), artifacts=(None, REFUSAL)) as user:
        await user.open("/")
        await user.should_see(marker="compact-finding")
        await user.should_not_see(marker="artifact-list")
        await user.should_see("run-artifacts refused this Run: journey_review_invalid (journey_file, cause:ValueError@result.py:339). "
                              "The report above stands; only its file list is missing.")  # fmt: skip


async def test_before_the_report_is_published_results_says_it_is_not_reached(replay) -> None:
    events = journey("Investigation")
    snap = snapshot("investigation_evidence_ready", "investigation_evidence_ready", grounded_report={"status": "not_published"})
    async with replay(events, {len(events): snap}, artifacts=(run_artifacts(), None)) as user:
        await user.open("/?view=Results")
        await user.should_see("No Grounded report yet: the report is published after you approve the report basis.")
        await user.should_not_see(marker="findings")
        await user.should_not_see(marker="artifacts")
        await user.should_not_see("SAMPLE LAYOUT")


async def wait_for(condition, what: str) -> None:
    """Live mode reads the feed and observes in background tasks; wait up to 5 s for `condition`."""
    for _ in range(50):
        if condition():
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


REFUSAL = {"code": "journey_review_invalid", "fields": ["journey_file", "cause:ValueError@result.py:339"], "status": "failed"}
LIVE_ANSWERS = {
    "document": ((run_artifacts(), None), "Bucket bio-neb-test · execution/test"),
    "refusal": ((None, REFUSAL), "run-artifacts refused this Run: journey_review_invalid"),
}


@pytest.mark.parametrize("overlap", [False, True], ids=["sequential", "overlapping"])
@pytest.mark.parametrize("answer", list(LIVE_ANSWERS))
async def test_live_mode_reads_run_artifacts_once_when_the_report_is_published(run_app, monkeypatch, overlap, answer) -> None:
    """The page reads the Run's files once: a later pass end, also one that closes while the first read is still
    running, starts no second read, and the files card shows the one answer when it lands."""
    events = journey("Results")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded_report())
    (result, shown) = LIVE_ANSWERS[answer]
    calls, observed, release = [], [], asyncio.Event()

    async def observe(journey_file: Path) -> tuple[dict | None, str | None]:
        observed.append(journey_file)
        return snap, None

    async def read_artifacts(journey_file: Path) -> tuple[dict | None, dict | None]:
        calls.append(journey_file)
        await release.wait()
        return result

    monkeypatch.setattr(cli_bridge, "observe", observe)
    monkeypatch.setattr(cli_bridge, "run_artifacts", read_artifacts)
    async with run_app(events[:-2], {}, live=True) as user:
        await user.open("/")
        append(events[-2:])  # the report is published and its pass closes
        await wait_for(lambda: calls, "the first run-artifacts read")
        if not overlap:
            release.set()
            await wait_to_see(user, shown)
        append([{**e, "event_id": f"{e['event_id']}-again"} for e in events[-2:]])  # a later pass ends
        await wait_for(lambda: len(observed) == 2, "the second observe")
        if overlap:
            await user.should_see("Reading the Run's files (run-artifacts)…")
            release.set()
            await wait_to_see(user, shown)
        assert [p.name for p in calls] == ["journey.jsonl"]


def test_an_unanswered_issue_names_no_one() -> None:
    """`decided_by` is null exactly when the Run ended before the researcher answered (`InvestigationIssueItem`)."""
    item = {"request": {"rsid": "rs50", "request_text": "Does it carry rs50?"}, "issue": {"key": "refsnp: not_found"},
            "decision": "unanswered", "decided_by": None, "response_event_id": None}  # fmt: skip
    built = grounded_report()["report"]
    built["issue_items"] = [item]
    assert report.decisions(built) == ["Report basis: approve (finalization review)", "rs50 (refsnp: not_found): unanswered"]


async def test_a_report_without_an_investigation_collection_shows_its_accounting_and_decisions(replay) -> None:
    """A Run that ended before Investigation publishes an accounting-only report: `collection_view` is left out and
    no question, admission or source is retained."""
    grounded = grounded_report()
    built = grounded["report"]
    del built["collection_view"]
    built.update(question=None, admissions=[], issue_items=[], retrieved_sources=[], source_assessments=[], conflicts=[],
                 gaps=[], accounting=[{"membership": {"included": True}, "value": {"accounting_id": f"finalization-accounting:{n}",
                                                                                  "event_type": t}}
                                      for n, t in enumerate(("RunInputsRegistered", "AttemptFailed", "AttemptFailed"))])  # fmt: skip
    events = journey("Results")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded)
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see("Grounded report published")
        await user.should_see("This report has no Investigation collection: no requested target reached Investigation.")
        await user.should_not_see(marker="findings")
        await user.should_not_see(marker="finding-card")
        await user.should_not_see(marker="not-carried")
        assert labels(user, "decisions") == ["Decisions", "Report basis: approve (finalization review)"]
        assert labels(user, "accounting") == ["AttemptFailed: 2", "RunInputsRegistered: 1"]


async def test_members_excluded_at_finalization_are_shown_as_excluded(replay) -> None:
    grounded = grounded_report()
    built = grounded["report"]
    built["admissions"][0]["membership"]["included"] = False
    built["retrieved_sources"][0]["membership"]["included"] = False
    built["report_membership"] = [{"member_id": f"m{n}", "included": n > 1} for n in range(5)]
    events = journey("Results")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded)
    async with replay(events, {len(events): snap}) as user:
        await user.open("/?findings=table")
        rows = user.find(marker="findings").elements.pop().rows
        assert rows[0]["finding"] == f"excluded from the report: Sample {SAMPLE} carries the target ALT G>A with genotype 0/1."
        assert rows[1]["finding"] == f"Sample {SAMPLE} does not carry the target ALT G>T."
        sources = user.find(marker="sources").elements.pop().rows
        assert [s["included"] for s in sources] == ["excluded"]
        assert "Report membership: 2 of 5 retained members excluded" in labels(user, "decisions")


async def test_an_excluded_source_assessment_is_shown_as_excluded_beside_its_included_source(replay) -> None:
    """A source and its assessment are separate members: excluding the assessment leaves the source included."""
    grounded = grounded_report()
    grounded["report"]["source_assessments"][0]["membership"]["included"] = False
    events = journey("Results")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded)
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        (source,) = user.find(marker="sources").elements.pop().rows
        assert (source["included"], source["assessment"]) == (
            "included",
            "excluded from the report: accepted: RefSNP resolved one locus.",
        )
