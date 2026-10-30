# Evals

Eval cases for the planning stage, and the tools that prepare a run and score it. Nothing under
`src/` imports this folder. A result is a review of the recorded Journey; the scores assist
that review.

| Path | What it is |
| --- | --- |
| `cases/planning/<case_id>.yaml` | One case: the public request, prepared Review answers, questions and known bad answers |
| `question-bank.yaml` | Shared questions a case cites by `ref`, with its own `params` |
| `case.py`, `case.schema.json` | The case record (Pydantic), question-bank resolution and compilation |
| `compiled/planning/<case_id>.json` | Each case with the bank inlined: what a run is scored against |
| `core-files.yaml`, `core_files.py` | Each scenario's public reads and reference: source URL, publisher md5, size and sha256 |
| `prepare.py` | One run's prompt, runtime and plan from the core files; no contact |
| `watch.py` | Stops a run at its request, source-read, repeat or stall limits; reports cut-off replies |
| `score.py` | A run's gates, question findings and diagnostics, read from its Journey; no contact |

## Cases

Seven train cases cover pig, human (HG002, HG005), *C. elegans*, *B. subtilis*, rice and dog.
Each case states where it came from in its `donor` field. The corpus's validation and sealed
cases are not published, so public results keep a holdout.

A planning run goes from `begin` to the unanswered Execution Review. Its gates:

- **accepted**: the product recorded the proposal plan;
- **usable**: each asserted sample, taxon and reference value is one identifier in its own format,
  and the planning stage's consumers verified the sample, taxon, reference and companion files. A
  consumer whose stage never ran is listed as `unreached`.

Each case adds typed questions (hard or graded), checked by `score.py`.

## Run one case

```bash
uv run python -m evals.case compile --check          # cases and compiled files agree
uv run python -m evals.prepare --case s2-giab-hg002-clean-L2-snp3 --model nemotron ...   # see --help
uv run python -m evals.watch --run-root <run root>   # beside the run
uv run python -m evals.score --case s2-giab-hg002-clean-L2-snp3 \
    --journey-file <run root>/<journey id>/journey.jsonl --output score.json
```

The core files (reads and reference) are downloaded once per scenario from the URLs in
`core-files.yaml` and checked against the publisher's md5 and the recorded sha256.

## Changing a case

Edit the YAML, run `uv run python -m evals.case compile`, and commit the regenerated files with
it. `tests/test_evals.py` refuses a stale compiled file, a request wrapped across lines and an
unknown question.
