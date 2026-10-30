# Paper retrieval tool

A developer entry over the product's paper modules, not product code. It downloads one paper,
reads it for one question in a development Journey, and shows the reader's views of it, so a
person can try the Paper Reader (ADR 0046)
outside a Run. It holds no reading logic: lane P's reader copies were internalized by #383 slice A
and retired (#486), and `src/bio_neb/` does not import this folder. Development Journeys it wrote
before then stay history.

| Module | What it calls |
| --- | --- |
| `tools.py` | `retrieve()`: the product's download (`bio_neb.providers.paper_sources.download_paper`) read into documents by the product converters (`bio_neb.paper_intake.documents.document_texts`) |
| `__main__.py` | the commands below: `answer` records the downloaded files through `paper_intake.intake.record_paper` and reads them with `paper_reading.reading.answer_question`; the reading commands call `paper_reading.reader`, `verify` and the evidence view call `paper_reading.answer`, and `targets` calls `paper_reading.targets` |

The product owns the rest: the download and its routes (providers),
Run paper intake and the reader in a Run (journey path).

## Commands

```text
uv run python -m tools.paper_retrieval retrieve REFERENCE --out DIR              # one paper
uv run python -m tools.paper_retrieval retrieve REFERENCE --replay RAW --out DIR # from a raw capture
uv run --env-file MODEL_ENV python -m tools.paper_retrieval answer --paper DIR --question "..." \
  --runtime-config PATH --run-root ABS_DIR --out DIR \
  [--capture ABS_JSONL [--cloud] [--trace-attribute lane=P ...]]                 # one reading
uv run python -m tools.paper_retrieval outline --paper DIR
uv run python -m tools.paper_retrieval grep --paper DIR "54108297" [--max-hits N]
uv run python -m tools.paper_retrieval read --paper DIR s12 [--offset N] [--length N]
uv run python -m tools.paper_retrieval rows --paper DIR [--table t3] --where POS=54108297 \
  [--column REF ...] [--max-rows N]
uv run python -m tools.paper_retrieval lookup --paper DIR rs1800562 4:54108297 [--ids-file F]
uv run python -m tools.paper_retrieval verify --paper DIR --answer answer.json [--out verify.json]
uv run python -m tools.paper_retrieval targets --answer answer.json --out DIR
```

- `retrieve` takes a DOI, PMCID, PMID or title. It writes `paper.json` (identity and document
  listing), `documents/`, `bundle.json` (what was read from each file), and, as
  `python -m bio_neb.devtools.paper_download` does, `files/` and `download.json`. It exits 1 when a
  failure ended the download. `--replay` reads a raw capture folder (`index.jsonl` and each body by
  its digest) and reaches no host.
- `answer` records a `retrieve --out` folder's downloaded files as one paper of a fresh development
  Journey under `--run-root` and reads it under the role's declared budgets. It writes
  `answer.json`, `answer.md` (the evidence view) and `requests.json` (the model, the requests, each
  item with its Attempt and stop, the Journey file and any capture), and exits 1 when no answer was
  recorded. `--capture` runs the reading inside the development inspection
  (Local traces); `--cloud` also exports it with
  `LOGFIRE_TOKEN`, and each `--trace-attribute` is set on every span.
- `outline`, `grep`, `read`, `rows`, `lookup` and `verify` read a `retrieve --out` folder with the
  product's reader; `targets` writes an answer's findings as `targets.json` (`PaperTargets`). None
  of them makes contact.

`retrieve` (without `--replay`) contacts Europe PMC and the PMC open-data bucket, and `answer` the
configured model; each contact run needs the owner's approved run plan first.

Offline tests: `uv run --locked --offline python -m bio_neb.devtools.testing focused
tests.test_paper_retrieval tests.test_paper_targets tests.test_paper_targets_chain`.
