"""What lane M 028a063a7 adds to a Run, as the simplified workspace shows it: the Run's own report.md (#476), Run
health (#414), the reference choice (#459) and staged S3 inputs (#488)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from runs import execution_proposal, failure_ask, grounded_report, journey, review, snapshot

import run_reading

REPORT_MD = "# Variant report: NA00001\n\n## Conclusion\n\n| # | Target | Result |\n|---|---|---|\n| 1 | rs10 | observed |\n"


def with_report(tmp_path: Path, page: str, *, written: str | None = None) -> tuple[list[dict], dict[int, dict]]:
    """A published Run whose last pass wrote `report.md` into its Run root (the folder above the Journey's), with
    the sha256 of `written` (the page itself by default) in its `ReadableReportWritten`."""
    events = journey("Results", "ReadableReportWritten")
    recorded = (written if written is not None else page).encode()
    events[-2]["payload"] = {"file_name": "report.md", "sha256": hashlib.sha256(recorded).hexdigest(),
                             "byte_count": len(recorded), "report_digest": "grounded-report:1", "explanation_id": "x"}  # fmt: skip
    tmp_path.joinpath("runs").mkdir()
    tmp_path.joinpath("runs", "report.md").write_text(page, encoding="utf-8")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded_report())
    return events, {len(events): snap}


async def test_the_report_page_shows_report_md_as_the_run_wrote_it(replay, tmp_path) -> None:
    async with replay(*with_report(tmp_path, REPORT_MD)) as user:
        await user.open("/?section=report")
        await user.should_see(marker="report-md")
        assert user.find(marker="report-md").elements.pop().default_slot.children[0].content == REPORT_MD
        status = user.find(marker="report-md-status").elements.pop().text
        assert status.startswith("report.md as the Run wrote it at") and "checked" in status


async def test_a_report_md_changed_after_the_run_wrote_it_is_shown_and_flagged(replay, tmp_path) -> None:
    async with replay(*with_report(tmp_path, REPORT_MD + "edited\n", written=REPORT_MD)) as user:
        await user.open("/?section=report")
        await user.should_see(marker="report-md")
        assert "changed after the Run wrote it" in user.find(marker="report-md-status").elements.pop().text


async def test_before_report_md_is_written_the_report_page_says_so_and_keeps_the_grounded_report(replay) -> None:
    events = journey("Results")
    snap = snapshot("grounded_report_published", "report_published", grounded_report=grounded_report())
    async with replay(events, {len(events): snap}) as user:
        await user.open("/?section=report")
        await user.should_see("report.md is not written yet")
        await user.should_not_see(marker="report-md")
        await user.should_see("Grounded report published")


def test_a_paper_drawn_question_reads_as_the_researchers_own_words() -> None:
    import report

    question = {"request_text": "Transcription of the APOE gene is critical…", "research_goals": ["Does HG002 carry them?"]}
    assert report.question_text(question) == "Does HG002 carry them?"
    assert report.question_text({"request_text": "Does NA00001 carry rs10?"}) == "Does NA00001 carry rs10?"


HEALTH = {"collection_event_id": "event-9", "bam_sha256": "1" * 64, "vcf_sha256": "2" * 64,
          "measurement": {"manifest_sha256": "3" * 64, "manifest_byte_count": 10, "reads": 1234567, "mapped_percent": 99.5,
                          "mean_depth": 31.25, "ti_tv": None, "ti_tv_reason": "no_transversion", "covered_contigs": 25,
                          "covered_bases": 3000000, "covered_fraction": 0.95}}  # fmt: skip


def health_run(*facts: tuple[str, dict]) -> list[dict]:
    events = journey("Investigation", *(name for name, _ in facts))
    for event, (_, payload) in zip(events[-1 - len(facts) : -1], facts, strict=True):
        event["payload"] = payload
    return events


async def test_the_health_page_shows_the_runs_measured_health(replay) -> None:
    events = health_run(("RunHealthMeasured", HEALTH))
    async with replay(events, {len(events): snapshot("investigation_running", "running")}) as user:
        await user.open("/?section=health")
        await user.should_see("Run health measured")
        for shown in ("1,234,567", "99.50 %", "31.2x", "95.0%", "not measured (no transversion)"):
            await user.should_see(shown)


async def test_a_failed_health_measurement_is_asked_on_the_health_page(replay) -> None:
    failed = {"collection_event_id": "event-9", "code": "timeout", "detail": "samtools ran past 600 s"}
    events = health_run(("RunHealthMeasurementFailed", failed), ("RunHealthReviewRequested", {"request": failure_ask()}))
    pending = review("run_health", {"request": failure_ask()}, {"action": "retry", "response_schema": {}},
                     {"action": "abort", "response_schema": {}})  # fmt: skip
    snap = snapshot("run_health_review_pending", "review_required", reviews=(pending,))
    async with replay(events, {len(events): snap}) as user:
        await user.open("/")
        await user.should_see("A health measurement failed")
        await user.should_see("timeout: samtools ran past 600 s")
        await user.should_see("Waiting for you · decide on a failed health measurement")
        assert [chip.text for chip in user.find(marker="needs-you").elements] == ["Health"]


def test_the_reference_shows_the_assembly_record_the_choice_and_where_its_sequence_comes_from() -> None:
    import views

    proposal = execution_proposal()
    proposal |= {
        "reference_basis": {"choice": {"quote": "my sample HG002", "reason": "One RefSeq reference genome for Homo sapiens."}},
        "reference_source_admission": {"source_mode": "public_url", "expected_size_bytes": 972_898_531,
                                       "expected_checksum_algorithm": "md5"},
    }  # fmt: skip
    assert views.reference_lines(proposal) == [
        "GRCh38.p14 (GCF_000001405.40)", "Homo sapiens, released 2022-02-03", "chosen from your words “my sample HG002”",
        "One RefSeq reference genome for Homo sapiens.", "sequence: public url · 1.0 GB, md5 checked",
    ]  # fmt: skip


def test_a_staged_input_names_its_object_and_a_missing_one_has_no_size() -> None:
    staged = {"role": "read_1", "registered_path": "s3://bucket/reads/r1.fastq.gz", "size_bytes": 2_000_000_000}
    assert run_reading.input_line(staged) == "read_1: 2.0 GB · staged object s3://bucket/reads/r1.fastq.gz"
    assert (
        run_reading.input_line({**staged, "size_bytes": None})
        == "read_1: not found · staged object s3://bucket/reads/r1.fastq.gz"
    )
    assert (
        run_reading.input_line({"role": "read_2", "registered_path": "/data/r2.fastq.gz", "size_bytes": 10**9})
        == "read_2: 1.0 GB"
    )
