# Tests

The offline test suite. It needs no key, makes no network contact and starts no cloud job:
every model, source and provider answer comes from a fake or a recorded fixture.

## Run

```bash
uv run python -m unittest tests.test_cli tests.test_journey_path          # named modules
ls tests/test_*.py | sed 's|/|.|; s|\.py$||' | xargs uv run python -m unittest   # every module
```

`tests/` has no `__init__.py`, so name the modules (or use the `xargs` line) instead of
`unittest discover`. Some modules take minutes; run several in parallel to go faster.

The GUI has its own suite: `cd gui && uv run pytest`.

## Layout

| Path | What it is |
| --- | --- |
| `test_*.py` | Test modules, one per behaviour or module under test |
| `*_fakes.py`, `*_support.py`, `*_fixture.py` | Shared fakes and builders: model endpoints, providers, Journeys at a given stage |
| `fixtures/` | Recorded inputs: Journeys and replies from live development Runs, source answers, small VCFs and FASTAs |
| `fixtures/source_answers/` | Real public-source answers, recorded once with read-only GETs, replayed beneath every source port |
| `testing_ownership.json` | The test-selection map used by the developer test runner (`bio_neb.devtools.testing`) |
| `value_guard.py`, `value_guard.json` | Refuses one-case values (rsIDs, accessions, sample names) and unreviewed matching rules hard-coded in the product, against a reviewed allowlist |

Fixtures recorded from live Runs stay byte-for-byte as recorded. They contain cloud resource identifiers and run-root paths from the machine that recorded them, but
no credentials.
