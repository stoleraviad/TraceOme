"""Starting a Run for the GUI to follow: `begin`'s argv, the credentials file on the writing commands only, and the
launcher's wait for the Run's Journey."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import cli_bridge
import start_run

CLI = ["uv", "run", "--locked", "--offline", "python", "-m", "bio_neb.cli"]


def test_begin_answer_and_resume_take_the_credentials_file_and_reads_do_not(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BIONEB_ENV_FILE", "/home/x/controller.env")
    journey = tmp_path / "journey.jsonl"
    with_env = ["uv", "run", "--env-file", "/home/x/controller.env", "--locked", "--offline", "python", "-m", "bio_neb.cli"]
    assert cli_bridge.answer_argv(journey, "request-1", {"action": "approve"})[:10] == [*with_env, "answer"]
    assert cli_bridge.resume_argv(journey) == [*with_env, "resume", "--journey-file", str(journey)]
    assert cli_bridge.observe_argv(journey)[: len(CLI)] == CLI
    assert cli_bridge.journey_feed_argv(journey)[: len(CLI)] == CLI
    monkeypatch.delenv("BIONEB_ENV_FILE")
    assert cli_bridge.resume_argv(journey) == [*CLI, "resume", "--journey-file", str(journey)]


def test_begin_starts_without_interactive_with_every_input(monkeypatch) -> None:
    monkeypatch.setenv("BIONEB_ENV_FILE", "/home/x/controller.env")
    argv = cli_bridge.begin_argv(Path("/runs/r1"), Path("/in/prompt.txt"), ["read_1=s3://b/r1.fq.gz", "read_2=/data/r2.fq.gz"],
                                 paper_files=[Path("/in/paper.pdf")])  # fmt: skip
    assert argv[2:4] == ["--env-file", "/home/x/controller.env"]
    assert argv[argv.index("bio_neb.cli") + 1 :] == [
        "begin", "--run-root", "/runs/r1", "--prompt-file", "/in/prompt.txt", "--fastq", "read_1=s3://b/r1.fq.gz",
        "--fastq", "read_2=/data/r2.fq.gz", "--paper-file", "/in/paper.pdf",
    ]  # fmt: skip
    assert "--interactive" not in argv


def test_a_credentials_file_others_can_read_is_refused(tmp_path) -> None:
    env = tmp_path / "controller.env"
    env.write_text("")
    env.chmod(0o644)
    assert "chmod 600" in start_run.env_file_problem(env)
    env.chmod(0o600)
    assert start_run.env_file_problem(env) is None
    assert "not a file" in start_run.env_file_problem(tmp_path / "missing.env")


def test_the_launcher_follows_the_journey_begin_creates(tmp_path) -> None:
    root = tmp_path / "run"
    script = f"import pathlib,time; p=pathlib.Path({str(root)!r})/'j-1'; time.sleep(0.3); p.mkdir(parents=True); (p/'journey.jsonl').write_text('')"
    with subprocess.Popen([sys.executable, "-c", script + "; time.sleep(5)"]) as process:
        assert start_run.wait_for_journey(root, process, timeout=10, poll=0.05) == root / "j-1"
        process.kill()


def test_the_launcher_stops_waiting_when_begin_ends_without_a_journey(tmp_path) -> None:
    with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
        assert start_run.wait_for_journey(tmp_path / "run", process, timeout=10, poll=0.05) is None


def test_the_launcher_refuses_a_folder_that_exists(tmp_path, capsys) -> None:
    env = tmp_path / "controller.env"
    env.write_text("")
    env.chmod(0o600)
    with pytest.raises(SystemExit):
        start_run.main(["--run-root", str(tmp_path), "--env-file", str(env), "--prompt-file", str(env), "--fastq", "read_1=/r1"])
    assert "every Run takes a new folder" in capsys.readouterr().err


def test_a_detached_command_reads_no_terminal(tmp_path) -> None:
    """A command that would prompt (begin without inputs) gets end-of-file at once instead of waiting on the shell."""
    log = tmp_path / "logs" / "prompt"
    process = cli_bridge.start_detached([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], log)
    assert process.wait(timeout=10) == 0
    assert log.with_suffix(".out").read_text().strip() == "''"
