# `bio_neb`: the Traceome package

One Python package, `bio_neb`, entered through `python -m bio_neb.cli`. The application
coordinator drives a Run from stage to stage. Agent roles propose, deterministic code checks
and executes, and every step is appended to the Run's Journey.

## How a Run flows through the code

```
cli.py ─▶ application/advancement.py      the coordinator: advance until a Review or the end
            │
            ├─ agentic_journey/            shared Agent work: model turns, Tools, role instructions
            ├─ input_inspection/           FASTQ metadata
            ├─ reference_discovery/        reference genome identity and companion files
            ├─ parabricks/                 recipe and proposal → Execution Review → H100 job → verified BAM/VCF
            ├─ investigation/              Work items → Investigation Review → VCF observation, sources
            └─ reporting/                  Finalization basis → Finalization Review → Grounded report
            ▼
journey.py                                 append-only events + content-addressed artifacts
application/queries.py, journey_path.py    the readers behind observe, journey-feed and the GUI
```

## Packages

| Package | Responsibility |
| --- | --- |
| `application/` | Run registration, the coordinator (`advancement.py`), Review answers, forks, Journey readers |
| `agentic_journey/` | Shared Agent work: Attempt lifecycle, model endpoint policy, Tool admission and execution, role instructions (`instructions/*.toml`) |
| `input_inspection/` | Inspecting the registered FASTQ files |
| `reference_discovery/` | Reference identity, companion facts and source admission Tools for planning |
| `prebuilt_index/` | The Galaxy prebuilt BWA index catalogue and contig renaming |
| `parabricks/` | Execution planning, the Planning and Execution Reviews, and the approved H100 execution |
| `investigation/` | Work items, VCF lookup and table reading, sources, run health and coverage |
| `paper_intake/`, `paper_reading/` | A Run's paper files, and the paper reader role |
| `target_request/` | Placing a request's or a paper's targets on the Run's assembly |
| `reporting/` | Finalization, the Grounded report, the explanation and the readable report |
| `failure/` | Failure classes and the correction a model is shown |
| `providers/` | Adapters: NCBI, EVA, Galaxy, Nebius (CLI, auth, object storage, serverless jobs, model endpoint, registry) |
| `devtools/` | Developer tooling: the test runner and fixture exporters. The product never imports it |

## Top-level modules

| Module | Responsibility |
| --- | --- |
| `cli.py` | The command line (`begin`, `observe`, `answer`, `resume`, `fork`, `journey-feed`, ...) |
| `journey.py` | The Journey: append-only events and content-addressed artifacts |
| `journey_path.py`, `journey_contracts.py` | Path facts and the registry of event families |
| `recording.py` | Pass intervals and the contact journal |
| `locations.py` | Every machine-specific location, each with one `BIO_NEB_*` override |
| `content.py`, `contracts.py`, `persistence.py` | Canonical JSON and digests, contract bases, exclusive writes |
| `storage_contracts.py`, `verified_artifacts.py` | Secret-free storage evidence and verified object descriptors |
| `identifier_rule.py`, `fasta_selection.py`, `junction.py`, `observation.py` | Identifier rule, FASTA selection, Review answer decisions, a polling helper |
