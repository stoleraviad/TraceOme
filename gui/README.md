# Traceome researcher workspace

A NiceGUI app for following and answering Traceome Runs. Results come first, beside a persistent
six-stage Journey. Each stage opens into its key steps, and each step opens a readable evidence
page. The Simple and Detailed views read the same recorded facts, and every decision form is
built from the Review's own response schema.

The GUI has its own environment. It never imports `bio_neb`; it talks to Traceome only through
the CLI of the checkout it sits in:

- the Journey panel shows `bio_neb.cli journey-feed --all`;
- the centre shows `bio_neb.cli observe` snapshots, taken when a pass closes;
- following a live Run, a decision button runs `bio_neb.cli answer` once the action's schema
  accepts the response, and the page then shows the next `observe`. Interrupt stops the command,
  and a held Run offers Resume (`bio_neb.cli resume`). In a recorded replay, the buttons only show
  the exact command and the schema's verdict.

## Run

From the repository root, after `./scripts/setup-keys.sh`:

```bash
cd gui
uv sync
# Start a new Run and follow it: `begin` stops at the first Review, and the GUI answers every Review after it
uv run python start_run.py --run-root $PWD/../.bio-neb-local/live-runs/<unique-run-id> \
    --env-file ../.bio-neb-local/model.env \
    --prompt-file /abs/path/question.txt --fastq read_1=/abs/path/R1.fastq.gz --fastq read_2=/abs/path/R2.fastq.gz
# Follow an existing Run live
LIVE=1 BIONEB_ENV_FILE=../.bio-neb-local/model.env RUN_DIR=<run root>/<journey id> uv run python app.py
# Replay a finished Run (needs its snapshots, below)
RUN_DIR=<run root>/<journey id> uv run python app.py
```

Open http://localhost:8770 (`PORT` changes it). The app listens on this machine only. `HOST=0.0.0.0`
exposes it, which is safe only for a recorded replay or a trusted network: a live GUI runs
commands with the Run's credentials and has no login. `begin`, `answer` and `resume` contact the
model and the cloud, so they run with the env file (mode 600; the GUI never reads its values).
The reads (`observe`, `journey-feed`, `run-artifacts`) run without it.

Set "Answering as" in the header before answering: every answer records who gave it.

Snapshots for replay: `uv run python make_snapshots.py <run folder>` calls `observe` at every pass
end on temporary copies and writes `gui/data/snapshots/` (git-ignored), then `run-artifacts` once.

Tests: `uv run pytest` runs NiceGUI's `User` simulation over small synthetic Runs, with a fake CLI.

## URLs

Every view has a URL. `section=request|parabricks|health|investigation|findings|report` selects a
stage; `section=evidence&record=<event line>` opens one recorded fact; `section=finding&focus=<target>`
shows a target's results and recorded path; `section=auto` follows the Run. `at=<event line>` moves
the replay cursor, and `density=simple|detailed` switches the view. `uv run --group screenshots
python shoot.py` screenshots every variant (Playwright).

## Files

| File | Role |
| --- | --- |
| `app.py` | The page: status banner, stage chips, drawers, the centre's frame and the replay bar |
| `run_reading.py` | Readers of the Journey items and `observe` snapshots, and the page's `Viewer` |
| `views.py` | The centre per stage and the plain review card |
| `runner.py` | Default workspace, stage navigation, recorded health and per-target path |
| `report.py` | Compact collection and full Grounded report, with the Run's files from `run-artifacts` |
| `evidence.py` | Readable decisions, observations, sources, output checks, VCF windows and reading notes |
| `decision.py` | The decision element: it starts `answer` live, or shows the dry-run command in replay |
| `commands.py` | The `answer`/`resume` the app runs for its Run, and the command bar (Interrupt, refusal, Resume) |
| `responses.py` | A review's response from its `response_schema`, and the schema's verdict on it |
| `journey_panel.py` | The persistent nested Journey and the Timeline/Passes views |
| `journey_feed.py` | The Journey panel's items from `journey-feed`, with the GUI's own narrator as fallback |
| `cli_bridge.py` | The only place that runs the Traceome CLI |
| `start_run.py` | `begin` detached, then the app live on the new Run |
| `make_snapshots.py` | Recorded `observe` snapshots for replay |
| `shoot.py` | Screenshots of every view |
