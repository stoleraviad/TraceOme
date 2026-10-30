# Run-health runner

A developer tool on lane H of the tools map
([#307](https://github.com/stoleraviad/BioNeb/issues/307)). It runs one pinned bioconda tool
against named local files and returns a run record: the argv that ran, what ran (checked against
the pin, or against the Run's tool probe) and each output's digest. It is the runner behind two
typed callables over a Run's own files (#406): `run_health(<Run Journey>, <cache>, <output dir>)
-> HealthManifest`, one health report per Run
([#317](https://github.com/stoleraviad/BioNeb/issues/317); see
[One health report per Run](#one-health-report-per-run-317)), and `measure_run_items(...) ->
ItemsCoverage`, every Work item in one batch, with option (d)'s per-item checks of the unsure ones
(`check_run_items`). The command line only adapts them. `src/` never imports it
(`tests/test_repository_contract.py`, `test_application_never_imports_developer_tools`).
A Run's own health and coverage are a product step now; see [In the product](#in-the-product-414).

## Parts

- `table.py`: the one tool table. `PINS` holds one `Pin` per bioconda package build (the
  BioContainers image `quay.io/biocontainers/<package>:<version>--<build>` and its local
  executable);
  `TOOLS` holds the entries (what runs, argv template, named inputs and optional inputs, stdout
  file, required outputs). What runs is either a `pin` or `probed`: a tool the Run's probe names (main's
  `ObservationToolsProbe`, `src/bio_neb/investigation/observation_tools.py`). A new tool is a new
  entry, never a new module.
- `runner.py`: the one runner port (`ToolRunner`) and `run_tool`, which turns an entry and named
  files into one invocation, checks what ran against the pin and builds the record.
- `local_process.py`: the local-process adapters. `LocalProcessRunner` runs tools installed in
  user space under `~/.local/opt/bio-neb-run-health/envs/<package>/` (owner, 2026-10-14; the
  devcontainer has no Docker): an explicit executable, its digest, and a `--version` probe.
  `ProbedProcessRunner` runs the executable the Run's probe receipt names (`bio-neb-samtools`,
  `bio-neb-bcftools`) and observes its digest and `--version` the same way.
- `cache.py`: copies a Run's reference FASTA/`.fai` once into a local cache through main's object
  read (see [The local cache](#the-local-cache)). The Run's BAM and VCF are never copied: they are
  the local copies its Investigation keeps beside its Journey (#363).
- `region.py`: coverage and GC over one indexed slice, for a typed range or a Work item (see
  [Coverage and GC of one slice](#coverage-and-gc-of-one-slice-318)); `items.py`: every Work item
  of a Run in one batch (see [Every Work item in one batch](#every-work-item-in-one-batch-405));
  `check.py`: option (d), each flagged or unsure item checked by its own agent loop (see
  [Unsure items checked one by one](#unsure-items-checked-one-by-one-406)); `replay.py`: the
  runner port answered from an earlier run's recorded outputs (see [Replay](#replay-406)).
- `health.py`: `read_run` (the Run's own BAM and VCF with their receipts), `run_health`, every
  whole-Run health entry in order, MultiQC over their summaries, the covered regions and the
  manifest (see [One health report per Run](#one-health-report-per-run-317)).
- `docker_host.py`: the owner-host adapter, `docker run` on a machine that has Docker, for the
  later image route.
- The fake adapters for tests live in `tests/test_run_health.py` and
  `tests/test_run_health_manifest.py`; the kept-probe fake and a Run's local inputs
  (`local_run_inputs`) in `tests/run_health_fakes.py`.

## Running a tool

    uv run --locked --offline python -m tools.run_health run <tool> --runner local|docker|probed \
        [--input NAME=PATH ...] --output-dir <fresh dir> [--prefix <install>] [--docker <cli>] \
        [--probe-dir <dir> [--runtime-config <runtime.json>]]

`--runner probed` holds the run to the Run's tool probe in `--probe-dir`: a receipt kept there is
reused (main's `probe_observation_tools`); with none, this environment is probed once for the tool
section of `--runtime-config` (the Run's runtime file) and the receipt is kept there.

It prints the run record as JSON and exits 0, or prints `{"refused": <code>, "detail": ...}` and
exits 2. An entry's argv is written as a container sees it: `{<input name>}` becomes
`/inputs/<file name>` and `{outputs}` the output directory, which is also the working directory.
An optional input that is left out puts the entry's stand-in value in its place (mosdepth's
`{targets}` becomes the window size). The output directory is created by the run (with any
missing parents) and must not exist yet.
Every input sits side by side under `/inputs`, so an index passed as its own named input lands
beside its data file wherever the two live on the host:

- Docker bind-mounts each input read-only at `/inputs/<file name>` and the output directory as
  the one writable mount, and runs with `--network none` as the calling user.
- The local adapter stages the inputs as symlinks in a private temporary directory that stands
  for `/inputs`, maps `/outputs` to the output directory, and calls
  `<prefix>/envs/<package>/bin/<argv[0]>` by absolute path (a relative `--prefix` is taken from
  the working directory).

## What ran: the pin or the probe

Each runner binds an entry before it runs (`bind`): the facts the run must observe and what the
record keeps. `run_tool` records a run only when every checked fact equals what the runner
observed, and records any other observed fact beside the binding's.

| Runner | Entries | Checked facts | Record `identity` |
| --- | --- | --- | --- |
| `local-process` | `pin` | `version_line` (first stdout line of `--version`, whatever its exit code) | the pinned version line and the observed `executable_sha256` |
| `owner-host-docker` | `pin` | `image_digest` (the repository digest from `docker image inspect`, after running the image by digest) | the pinned facts |
| `probed-process` | `probed` | `executable_sha256` and `tool_version`, held to the probe receipt's | `probe_receipt` and `probe_receipt_sha256` only |

A pinned run is refused before contact while any of its runner's facts is unpinned. A probed run's
record keeps the receipt's place and digest, not the probe's facts, which stay in the receipt.
mosdepth, goleft and MultiQC are pinned by version line from the owner-approved install (#314 run
plan v2): any install of that version runs, whether the lane's own, a Spack build or a Galaxy image
(owner, #406 G3), and each run records the executable digest it observed. samtools and bcftools
are never pinned: they are the Run's own, through its probe (#406). Image digests stay unpinned
until an image route runs, so Docker runs are refused until then.

## The local install

`~/.local/opt/bio-neb-run-health/` holds `micromamba`, its package cache `root/` and one
environment per package under `envs/`, installed from `conda-forge` and `bioconda` only. Nothing is
on `PATH`. Each environment's exact packages (URLs with MD5s) are in `locks/<package>-linux-64.lock`;
a rebuilt devcontainer recreates one with

    env -i HOME=$P/home PATH=/usr/bin:/bin $P/micromamba create --yes --no-rc -r $P/root \
        -p $P/envs/<package> --file tools/run_health/locks/<package>-linux-64.lock

where `P=~/.local/opt/bio-neb-run-health` (micromamba itself: the conda-forge `micromamba` 2.9.0
archive, `bin/micromamba` only). That download is contact and needs the owner's approval, as the
first install did. #405 ran it into a fresh prefix (`~/.local/opt/bio-neb-run-health-rebuild-405`,
all 232 lock URLs from `conda.anaconda.org`, 38 s): every environment built and every
`--version` line equals its pin; some executables' digests differ by the install prefix conda
writes into them, which is why the pins hold the version line only (#406 G3). The locks of the
pinned tools are `mosdepth`, `goleft` and `multiqc`; samtools', bcftools' and bedtools' retired with
their pins (#406).

## Run record (`bio-neb.run-health-record/1`)

`tool`, `runner`, `image_reference` (`quay.io/biocontainers/<package>:<version>`; `null` for a
probed run, which runs no image), `identity` (the binding's recorded identity, above), `argv` (the tool invocation in the container form above, the
same for every runner), `executed_argv` (the command the adapter actually executed: the local
executable's absolute path with the staged input paths, or the full `docker run` command),
`inputs` (name to host path) and
`outputs` (every file the run wrote, by path under the output directory, with `sha256` and
`byte_count`). Outputs are hashed through the product seam `stable_file_observation`. Inputs are
not re-hashed: their identity stays with whoever produced them (ADR 0029). Where the record is
stored is #313's decision; this tool only returns it.

## Refusals

No record is returned when a run is refused:

| Code | When |
| --- | --- |
| `not_pinned` | A fact the runner checks is not pinned, or a pinned-package runner was given a probed entry. Nothing runs. |
| `not_probed` | A probed runner was given a pinned entry, or the probe could not be read or made (the probe's `runtime_config_invalid` and its section paths). Nothing runs. |
| `inputs_mismatch` | A required input is missing, or an input is neither required nor optional for the entry. Nothing runs. |
| `output_dir_exists` | The output directory already exists. Nothing runs or changes. |
| `tool_failed` | The tool (or Docker, or a missing local executable, exit 127) exited non-zero; detail has the exit code and stderr tail. |
| `tool_timed_out` | A process the runner started (the `--version` probe or the tool) ran past `run_tool(..., timeout_seconds=...)`; it is killed and its outputs stay unrecorded. Unset, a run is unbounded. Under Docker only the client is killed. |
| `parameters_mismatch` | A run parameter the entry names is missing, or one it does not name is given. Nothing runs. |
| `identity_mismatch` | What ran differs from the pin; detail names each fact, pinned and ran. Outputs stay unrecorded. |
| `output_missing` | A required output was not written. |

Every refusal of the tool is a `RunRefused` with its code, printed by every command as
`{"refused": <code>, "detail": ...}` with exit 2; the cache's are `cache_exists`, `cache_missing`,
`cache_complete`, `reference_read_failed`, `reference_read_mismatch`, `kept_file_mismatch`,
`manifest_identity_mismatch`, `reference_identity_mismatch` and main's materialization refusals.
A Journey that cannot be read is `run_unreadable`, never a traceback.

## The Run's read and variant health

Five probed entries (#315), run over the Run's BAM and VCF:

| Entry | Inputs | Output |
| --- | --- | --- |
| `samtools_index` | `bam` | `aligned.bam.csi` (`samtools index -c`, into its own fresh directory) |
| `samtools_stats` | `bam` | `samtools-stats.txt` (read counts, mapping rate, properly paired, histograms) |
| `samtools_flagstat` | `bam` | `samtools-flagstat.txt` |
| `samtools_idxstats` | `bam`, `bam_index` | `samtools-idxstats.tsv` (reads per contig) |
| `bcftools_stats` | `vcf` | `bcftools-stats.txt` (`-s -`: counts by type, Ti/Tv, per-sample het/hom, QUAL and depth histograms) |

`bam_index` is `samtools_index`'s output, passed beside the BAM; the index is found because both
sit under `/inputs` as `aligned.bam` and `aligned.bam.csi` (the product names the BAM
`aligned.bam`). The `samtools` and `bcftools` pins and `samtools_version` retired at #406: one
identity per tool, the Run's probe (ADR 0010).

## Whole-genome coverage

Three entries (#316), run over the Run's BAM. All three are viewing only: no result changes a
disposition.

| Entry | Inputs | Outputs |
| --- | --- | --- |
| `mosdepth` (pinned) | `bam`, `bam_index` (the CSI); optional `targets` (a BED) | `coverage.mosdepth.{global,region}.dist.txt`, `coverage.mosdepth.summary.txt`, `coverage.regions.bed.gz` (mean depth per window or target), `coverage.thresholds.bed.gz` (bases ≥ each threshold per window or target), `coverage.quantized.bed.gz` (callable bins), each `.bed.gz` with its `.csi` |
| `samtools_index_bai` (probed) | `bam` | `aligned.bam.bai` (`samtools index -b`, into its own fresh directory) |
| `goleft_indexcov` (pinned) | `bam`, `bam_index` (the `.bai`) | `indexcov/indexcov-indexcov.{bed.gz,ped,roc}`, `indexcov/index.html` and its per-chromosome plots |

mosdepth runs in fixed 500 bp windows (`--by 500`), or `--by` the target BED when `targets` is
given. Its settings come from main's `CoverageDefaults` through `coverage_entry(defaults)` and so
appear in the record's `argv`: the callable bins are `--quantize 0:1:<minimum_depth>:` (no
coverage, below the minimum depth, callable; `0:1:` at minimum depth 1 and `0:` at 0), the thresholds are 1, 20, 30 and
`minimum_depth`, `--mapq` is `minimum_mapping_quality`, and `--flag` is the sum of the
`excluded_flags` bits (1796 for UNMAP, SECONDARY, QCFAIL and DUP). mosdepth's default mode
counts overlapping mates once and leaves deletions out, with no BAQ. `coverage_entry` refuses a
setting it cannot apply (`ValueError`). mosdepth has no base-quality filter, so
`minimum_base_quality` is not applied.

indexcov cannot read a CSI. goleft 0.2.6 opens `<bam>.bai` only and panics without one (#316's
finding), so `samtools_index_bai` builds a `.bai` from the Run's BAM, which is passed beside the
BAM as #315's CSI is. indexcov's `--excludepatt` comes from the Run's reference (see
[Arguments from the Run](#arguments-from-the-run-405)); indexcov's default pattern (`^NC`, …)
would drop every RefSeq chromosome. It infers no sex (`--sex ''`, the `.ped` sex is `-9`): the
health view asks where the data has coverage, not which chromosomes are sex chromosomes (owner,
#406 G2).

## The local cache

    uv run --locked --offline python -m tools.run_health cache --journey <Run Journey> \
        --cache-dir <dir> [--resume]

The cache holds the Run's reference only (`bio-neb.run-health-cache/2`): `reference/` holds the
prepared `reference.fa` and `reference.fa.fai`, one exact read each through main's object read in
the Run's bucket and context, with the identities in the Run's verified execution manifest (the
FASTA's also equal to the Run's reference result). `cache-manifest.json` (written last) records
each file's path, object key, byte count, sha256 and whether it was `kept`, with the Run's
`journey_id` and `journey_file`. Without `--resume` the directory must not exist. The first
refusal stops the copy and no manifest is written; a failed reference read is
`reference_read_failed` with the provider's kind and detail. GETs are contact and need an approved
run plan.

`--resume` continues an earlier attempt that has no manifest yet (`cache_missing`,
`cache_complete` otherwise). A file that is present is kept: it is reread in full and must equal
the Run's identity (`kept_file_mismatch`) and be sealed read-only. Only absent files are read; a
failed read removes its own file (#363), so no operator step precedes a resume. A cache whose
reference files are hard links to another cache's (the same Run's bytes) is built this way with no
contact.

## Limits

- A local executable's digest covers that file, not the shared libraries or Python packages it
  loads; the environment's explicit lock is the install's full record.
- `docker_host.py` needs a POSIX host (it passes the caller's uid and gid).
- Two inputs with the same file name collide at `/inputs/`.
- This is the dev cycle on fixed fixtures (owner, #406 G1/G5): the offline tests use synthetic Runs
  and the HG001 slices; the real use reads each Run's own artifacts, which the live run exercises.

## Coverage and GC of one slice (#318)

Two probed entries read one region through its index. Both are viewing only, and no result
changes a disposition.

| Entry | Inputs | Parameter | Output |
| --- | --- | --- | --- |
| `samtools_depth_region` | `bam`, `bam_index` | `region` | `depth.tsv`: every position of the region (`samtools depth -a -r`) |
| `samtools_faidx_region` | `reference`, `reference_index` (the `.fai`) | `region` | `region.fa` (`samtools faidx`) |

An entry's `parameters` are values each run supplies (`run_tool(..., parameters={"region": ...})`).
They appear in the record's `argv`, and a missing or extra one is refused (`parameters_mismatch`).
`depth_entry(defaults)` applies `CoverageDefaults` exactly:

- `-q` is `minimum_base_quality`, which mosdepth cannot apply, and `-Q` is
  `minimum_mapping_quality`.
- `-g` clears samtools' default excluded flags, and `-G` sets `excluded_flags`.
- `-s` counts overlapping mates once (`count_once`), and `-J` includes deletions (`included`).
- samtools depth has no BAQ, so `applies_baq=True` is refused (`ValueError`).

`region.py` holds the per-item reads:

- `measure_range(files, "contig:start-end", ...)` returns a typed range's `RegionSummary`. The
  range is 1-based and inclusive, uses RefSeq names (`NC_…`), and is at most 1,000,000 bases.
- `measure_item(files, contig, position, flank_bases=...)` returns an item's
  `CoverageMeasurement` (its depth at the position and `meets_minimum_depth`) together with the
  window's summary. The window is kept within the contig, as the item's VCF window is.

`files` is a `RegionFiles`: the BAM and its index, and the reference and its `.fai`. The summary
holds:

- mean, median, minimum and maximum depth;
- the fraction of positions at or above 1, `minimum_depth`, 20 and 30;
- GC over the A, C, G and T bases (`None` when there are none) and the count of other bases;
- the parameters it was read under.

Both reads are bounded by the lane's own `region.TIMEOUT_SECONDS` (`tool_timed_out`). These are
refused before any run, each with its reason: `range_invalid`, `contig_unknown` (not in
the `.fai`), `range_beyond_contig`, `range_too_long`, `index_missing` (`bam_index` or
`reference_index`) and `input_missing`.

    uv run --locked --offline python -m tools.run_health range NC_000019.10:44908584-44908784 \
        --bam <bam> --bam-index <csi> --reference <fa> --reference-index <fai> \
        --probe-dir <dir> --output-dir <fresh dir>

This prints the summary and both run records, or `{"refused": ..., "detail": ...}` with exit 2.
Over #107's cache a typed range answers in about 5 s whatever its length up to 1 Mb, mostly
process start-up and the probe's `--version` checks. A whole-BAM mosdepth run takes 80–120 s.

## One health report per Run (#317)

    uv run --locked --offline python -m tools.run_health health --journey <Run Journey> \
        --cache-dir <the Run's cache> --output-dir <fresh dir> [--prefix <install>] \
        [--timeout-seconds <s>]

`run_health(journey_file, cache_dir, output_dir, pinned=..., probed=None, timeout_seconds=...)`
reads everything it needs from the Run. Probed entries run through `probed`, by default the
Run's own kept probe (`run_probe`, `<Run>/investigation/tools/`, read back without running or
writing anything); pinned ones through `pinned`. Viewing only: no result changes a disposition.

First the Run is read (`read_run`): its completed Execution outcome, and for `aligned_bam` and
`VCF`, the Run's verified outputs as `run-artifacts` lists them, the local copy its Investigation
downloaded (`<Run>/investigation/bam/aligned.bam` and `vcf/variants.vcf`), held to the Run's
output by main's receipt beside it (`CurrentBamAcquisitionReceipt`,
`VcfMaterializationReceipt`). The receipts are read for their meaning; the copies are not reread
(ADR 0029). Then the lane's cache must be this Run's (`cache_other_run` otherwise).
Then, each into `<output_dir>/<entry>/`, in order: `samtools_index` (CSI), `samtools_stats`,
`samtools_flagstat`, `samtools_idxstats`, `bcftools_stats`, `mosdepth` (500 bp windows),
`samtools_index_bai`, `goleft_indexcov`, then `multiqc` over the summaries. The first refused run
stops the rest, and no manifest is written.

`multiqc` (pinned 1.35) runs with `--no-version-check` (no contact) and `--data-format json`. It
writes `multiqc.html` and `multiqc_data/multiqc_data.json` with MultiQC's other data files. Two
limits shape it:

- It reads indexcov's `.ped` but not its `.roc`: MultiQC 1.35's goleft module breaks on the
  RefSeq contig names (`NC_…`). The ROC stays in indexcov's own `index.html`.
- The derived renames (`<output_dir>/multiqc-sample-names.tsv`) rename each file-named sample to
  the `run-artifacts` role of what it summarizes: samtools and mosdepth to `aligned_bam`, bcftools
  to `VCF`. Samples the data names stay as named, such as indexcov's and bcftools' per-sample
  read-group name.

The manifest (`bio-neb.run-health-manifest/2`) is written last, as
`<output_dir>/run-health-manifest.json`, and returned. A reader needs nothing else:

- `journey_id`, `journey_file`, `run_root`, `execution_id` and `outcome_event_id`, as
  `run-artifacts` names them;
- `inputs`: each Run file read, as its `run-artifacts` output entry (`role`, `location`,
  `byte_count`, `sha256`), the local `path` that was read and its `receipt`;
- `files`: every file the runs wrote, with `role`, `path` (relative to the manifest's directory),
  `byte_count`, `sha256` and `record`, the entry whose run wrote it;
- `records`: each run's `bio-neb.run-health-record/1`, by entry, and `entry_seconds`, each run's
  wall-clock seconds;
- `arguments`: the arguments derived from the Run (below);
- `covered`: the regions the reads cover (below).

| Entry | Role (path under the entry's directory) |
| --- | --- |
| `samtools_index` | `bam_csi` (`aligned.bam.csi`) |
| `samtools_stats`, `samtools_flagstat`, `samtools_idxstats`, `bcftools_stats` | the entry's name |
| `mosdepth` | `mosdepth_global_dist`, `mosdepth_region_dist`, `mosdepth_summary`, `mosdepth_regions`, `mosdepth_thresholds`, `mosdepth_quantized`; each `.csi` is `mosdepth_bed_index` |
| `samtools_index_bai` | `bam_bai` (`aligned.bam.bai`) |
| `goleft_indexcov` | `indexcov_report` (`index.html`), `indexcov_bed`, `indexcov_ped`, `indexcov_roc`; the per-chromosome pages and plots are `indexcov_chromosome_plot` |
| `multiqc` | `multiqc_report` (`multiqc.html`), `multiqc_data` (`multiqc_data/multiqc_data.json`); MultiQC's other data files are `multiqc_data_file` |

No health role is a `run-artifacts` role, so a reader can list both together. A file a run writes
that has no role here is refused (`output_unnamed`). The samtools stats, bcftools stats and MultiQC
outputs embed staged input paths and times, so their digests change between runs of the same
files (#315); the mosdepth and indexcov digests are stable (#316).

| Refusal | When |
| --- | --- |
| `run_unreadable` | The Journey cannot be read. Nothing runs. |
| `outcome_unavailable` | The Journey has no completed Execution. Nothing runs. |
| `file_missing` | The Run lists no verified `aligned_bam` or `VCF`, or the cache has no manifest. Nothing runs. |
| `run_file_missing` | The Run holds no local copy of its BAM or VCF (a Run from before #363, or its Investigation has not downloaded it). Nothing runs. |
| `receipt_invalid` | A local copy's receipt is not main's. Nothing runs. |
| `file_changed` | A receipt names another object than the Run's output; detail has both. Nothing runs. |
| `run_file_unreadable` | A local copy's receipt cannot be read (a directory, no permission). Nothing runs. |
| `cache_other_run`, `cache_unreadable` | The cache is another Run's, or not a readable run-health cache. Nothing runs. |
| `not_probed` | The Run kept no tool probe, or one that cannot be read. Nothing runs. |
| `output_dir_exists` | The output directory exists. Nothing runs or changes. |
| `output_unnamed` | A run wrote a file with no health role. No manifest. |
| a run's refusal | Its own code (`tool_failed`, `identity_mismatch`, ...), with the entry named first in the detail. No manifest. |

Where the manifest and files live in the product (Journey facts or a ledgered file set) is #362's
proposal; during the lane they stay in the lane's local evidence directory.

### Arguments from the Run (#405)

`derive_arguments(reference_index, inputs)` takes MultiQC's renames from the Run; there are no fixed values (#406, ADR 0010). The reference `.fai` is the cache's
`reference_fai`.

- `--excludepatt` is `^$`: indexcov leaves out no contig (ADR 0049; no name-shape rule).
- `--sex` is always empty (above).
- MultiQC's renames map each summary's file name (`samtools-stats`, `coverage`, …) and the Run's
  VCF's own name (`variants`) to their `run-artifacts` roles. They are written as
  `<output_dir>/multiqc-sample-names.tsv`, which is not a health file.

### Covered regions (#406)

The data may cover only a region of the reference (`<contig>:<start>-<end>`, any contig), so the
health view finds where it lies (owner, #406 G2). `covered_regions` reads `samtools idxstats` and
mosdepth's 500-base windows (`coverage.regions.bed.gz`): a window is covered at a mean depth of 1
or more, and adjacent covered windows merge into one region of `<output_dir>/covered-regions.bed`.
The manifest's `covered` keeps the BED's path, byte count and sha256, the window and threshold, and
per contig with a mapped read or a covered window: its length, mapped reads, covered bases and the
first and last covered base.

## Every Work item in one batch (#405)

    uv run --locked --offline python -m tools.run_health items --journey <Run Journey> \
        --cache-dir <the Run's cache> --health-dir <the Run's health report> \
        --output-dir <fresh dir> [--flank-bases 100] [--probe <contig:position> ...]

`measure_run_items(journey_file, cache_dir, health_dir, output_dir, ...)` reads the Run's Work
items (`work_item_targets`: `InvestigationWorkItemStarted`, in selected order) and measures them
over the Run's own BAM (`read_run`), the CSI index the Run's health report wrote (its manifest
must have read that same BAM: `health_other_bam` otherwise) and the cached reference, with the
Run's own samtools. It writes `<output_dir>/items-coverage.json` last and returns it.
`measure_items(files, items, flank_bases=..., ...)` measures any items in three probed runs,
whatever their number; the reference read runs beside the slice and depth reads:

| Entry | Inputs | Output |
| --- | --- | --- |
| `samtools_view_windows` | `bam`, `bam_index`, `windows` (BED) | `windows.bam`: every read overlapping a window, through the index (`-M -L`) |
| `samtools_depth_windows` | `bam` (that slice), `windows` | `depth.tsv` (`-b`, the `CoverageDefaults` filters, no `-a`) |
| `samtools_faidx_windows` | `reference`, `reference_index`, `regions` | `windows.fa` (`-r`, one `contig:start-end` per line) |

- An item's span is its REF, or an insertion's anchor base (#368); its window is the span with
  `flank_bases` each side, kept within the contig, as its VCF window is. Work items at one span share
  one `SpanCoverage` (`work_item_ids`).
- `measurement` is main's `CoverageMeasurement`; its `depth` is the least depth over the span. The
  `window` is `range`'s `RegionSummary` of the window; it equals `range` over the same window.
- `uncovered_positions` are the span's positions with no read that passes the filters, and
  `uncovered` says how they combine: `none`, `part` or `whole`.
- An item whose contig the `.fai` does not name (`contig_unknown`) or whose span ends past it
  (`span_beyond_contig`) is listed in `flagged` with its contig, span and reason; the others are
  still measured. With nothing to measure, nothing runs and the three records are `null`; the
  output directory is still created, so an existing one is refused either way.
- A measured item is `unsure`, for option (d), when its span is below the minimum depth while its
  window has reads (`low_depth`), or holds an N base of the reference (`reference_n`). No read in
  the whole window is a plain absence, not a doubt.
- A position with no read that passes the filters has depth 0, whether or not samtools printed it:
  `samtools depth -b` streams its whole input and with `-a` walks every position of every contig, so
  the batch slices first and leaves `-a` out.
- `--probe` adds a one-base position as `probe:<contig>:<position>`, not a Work item.
  `measure_items_alone` reads each span's window by itself through `range`'s path, the path the
  batch replaces.

`ItemsCoverage` (`bio-neb.run-health-items-coverage/2`) also keeps the batch's wall-clock
`seconds` and the read filters it ran under (`parameters`). A batch directory that cannot be read
for its BAM index is refused `health_unreadable`; `items-coverage.json` is written exclusively. An existing output directory is refused (`output_dir_exists`) and nothing runs.
Distinct spans that share one window (both clamped to a short contig) read it once. The offline
fixtures (five rebased HG001 slices around seven Work-item rows, with their original-BAM depths)
are listed with sizes and digests in `docs/development/reviews/issue405-lane-h-spike/fixtures.md`.

## Unsure items checked one by one (#406)

    uv run --locked --offline python -m tools.run_health check --journey <Run Journey> \
        --cache-dir <the Run's cache> --health-dir <the Run's health report> \
        --items-dir <the Run's batch> --output-dir <fresh dir>

Option (d) (owner, #406 G4 d1): after the batch, each span of the flagged and unsure items gets
its own pydantic-ai agent loop, all at once; no check waits on another. Work items at one span,
flagged for one reason, share one check (`flags`), as they share one measurement. `check_unsure(coverage, files,
model=..., ...)` gives the model the item's place, why the batch is in doubt and, when measured,
its window summary, and one Tool, `measure_range`, which reads a region under the batch's read
filters (`ItemsCoverage.parameters`). For a measured item the Tool reads the item's own contig
within 10,000 bases of it; for a flagged item, whose place is in doubt, anywhere on its own
contig, at most 20,001 bases at a time (G4 d1: the item's contig only; a contig the reference
does not name has nothing to read, and the prompt lists the reference's contigs). Any other region goes back to
the model unread, and each read attempt has its own directory (`read-<n>`; a failed read's partial
output stays). The model answers a `CoverageVerdict`: `covered`, `low_coverage`, `no_coverage` or
`cannot_tell`, the region it relies on and a reason. A check answers from its own reading: an
answer before any read, or naming a region it did not read, goes back to the model (counted
against its caps); only an item whose contig the reference lacks, with nothing to read, may
answer unread (#406 live a2). The check's `result` is a `CoverageResult`
built from that read, never from the model's words: the region, the span's
`CoverageMeasurement` (its least depth; `null` when the read does not hold the whole span) and the
region's summary; a verdict naming a region it did not read has no result.

- The model is the one the Run's registered selections name for its agents (`run_model`; the credential
  from the environment variable it names, `model_credential_missing` otherwise), with no client
  retries, so each request is one attempt. Its answer is prompted JSON, not a forced Tool call
  (#403: the Nebius endpoint drops forced-call arguments), and the instructions and the answer's
  schema go in one system message, first (the endpoint refuses a second; #406 live a1). The final
  answer is plain text parsed by the check (a Markdown-fenced JSON object is accepted; other text
  goes back to the model), not the endpoint's JSON mode: under JSON mode the model made no Tool
  call in #406 live a3, and JSON mode asks for a JSON reply (the cause is a hypothesis that a4's
  live run tests).
- Every request is counted before it is sent: each check may send 6 and all together 40 (the run
  plan's caps; `--request-cap` lowers the total). A request past either cap is refused before it
  is sent (`RequestCapReached`, naming the cap), and its check stops with that failure.
  `model_requests` counts requests sent, failed or not.
- A failed check records `<module>.<type>: <message>`; the others go on.
- Each check's model exchange is kept as `<n>/messages.json`. `check --replay <earlier checks
  dir>` (`replay_model`) answers each check with the responses recorded for it, so the checks replay
  without model contact; the Tool's reads run again over the Run's files. A check the earlier run
  has no response for fails as `NotReplayable`. A replay makes no contact, so it runs without the
  request caps: each check replays its own exchange, whatever the others' timing.

`UnsureChecks` (`bio-neb.run-health-unsure-checks/1`) is written last, exclusively, as
`<output_dir>/unsure-checks.json`: each check's flag, verdict, result, the regions it read, its
model requests, seconds and failure, and the model and requests sent against the cap. A batch
result that cannot be read is refused `items_unreadable` before any model request.

## Replay (#406)

    $RH health ... --replay <earlier health report>
    $RH items ... --replay <earlier batch>

`ReplayRunner` is one more runner adapter: it answers each tool run, pinned or probed, with the
outputs the earlier run recorded in the same place. It refuses (`tool_failed`, `replay: ...`) a
run whose argv differs from the recorded one, or whose inputs do: a source input (a Run file, the
reference, another report's file) must be the very file the recorded run read, and an input the
earlier command generated must sit in the same place under the new directory with the same bytes.
Recorded outputs that cannot be copied are a failed run too; nothing is deleted. `run_health` and `measure_run_items` run unchanged,
so their results compare with the earlier ones file by file without the tools or their install;
each replayed record's runner is `replay`. Together with `check --replay`, this is the run plan's
replay stage (Q5): offline fixtures, then replay of recorded outputs, then the online run.

## In the product (#414)

A Run measures its own health and every Work item's coverage as a product step: the table
stage's first steps (`src/bio_neb/investigation/run_measurement.py`; `docs/design/investigation.md`,
"The Run's measurement"). It runs the Run's own probed samtools and the product's one VCF reader,
reads no reference or cache, and records `RunHealthMeasured` and `InvestigationItemCoverageRecorded`
through the default composition. The lane's development composition (`journey`, `LaneMeasurements`)
is retired with it. What stays here is the development extras (mosdepth, goleft indexcov, MultiQC,
the reference cache and GC) and option (d)'s substrate (`items`, `range`, `check`, replay), until
(d)'s follow-up ticket (owner, #414 readiness Q2). No pattern of the lane judges a sequence name by
its shape (ADR 0049): a typed range splits at its last colon and its contig is held to the
reference's own names, and indexcov leaves out no contig.
