# Run ledger tool

Maintained developer tool (outside `src/bio_neb/`) for the repository's run ledger,
`docs/runs/ledger.csv`. Five commands, all `uv run python -m bio_neb.devtools.run_ledger …`:

- `add --run-root <finished-run-root> --ticket <n> [--run-id <id>] [--receipt <location_key>:<name>]…`
  reads `operator/finished.json` and discovers the execution through `approved-input.json`'s
  Journey pointer or `journey.jsonl`, `<id>/journey.jsonl`, `app/<id>/journey.jsonl`,
  `run/<id>/journey.jsonl`, or `runs/<id>/journey.jsonl` beneath the selected root. It uses
  `read_parabricks_result` and `review_execution_contacts`, the public contact-free readers,
  without constructing execution dependencies. Multiple bound Journeys or malformed evidence
  refuse before any CSV write. An unbound approved input records only selected bucket identity,
  marked `resource_evidence=selected_only`; it makes no creation or outcome claim.
  When no current execution is found, historical `harness/result.json`, `harness/input.json`
  or `evidence/input.json` and `harness/journal.jsonl` remain readable without reshaping them. An offline ticket that left only testing receipts
  omits `--run-root` and names its receipts (`add --ticket <n> --receipt testing_run_roots:<dir>…`);
  each row's run id, date and commit come from its own receipt name, so one call may name
  receipts of several commits. `--run-id` and `--source-commit` override them only for receipts
  of one commit; over several commits they refuse before any CSV write.
- `inventory [--update] [--no-contact | --from-receipt <inventory.json>] --project-id <id> [--profile bio-neb-sa] [--open-tickets a,b] [--ticket <n>] [--receipt-root <dir>]`
  lists and diffs. Provider contact is list-only (`storage bucket list` through the accepted
  pinned CLI runner, `list-objects-v2` through the product's boto3 factory), every call journaled
  through the product's execution-contact observer into the receipt's `journal.jsonl`. The
  credentials come from the environment (`uv run --env-file <controller.env> …`); the tool never
  reads a value itself. `--update` fills only empty `size_bytes`, `inventory_sha256` and
  `storage_class` cells. The receipt root is a fresh `<issueN|ledger>-inventory-<UTC>-<sha>-a<N>`
  under `testing_run_roots` (see `python -m bio_neb.locations`).
- `next-run-id --ticket <n> --purpose <purpose> --commit <sha> [--utc <stamp>]` prints a fresh
  `<ticket>-<purpose>-<UTC>-<short-sha>-a<N>`; `a<N>` is one past every run of that ticket and
  purpose in the ledger, `run_roots` or an archive batch, so numbering continues past archived runs.
- `archive --batch <yyyy-mm-dd>-<name> --approved <file> --ticket <n> --authority <locator> [--dry-run]`
  moves owner-approved run roots to `<archive_root>/<batch>/run-roots/`, rekeys their rows to
  `archive:<batch>` with an `archived_to` note (adding a row for an unledgered root) and writes the
  batch's `manifest.json` and `MANIFEST.md`. It validates the whole list first and refuses a
  missing, already-archived or `fixture_use` root. Rules: Owner-approved cleanup.
- `restore --batch <batch> --identity <run root> --ticket <n>` moves one archived root back to
  `run_roots`, returns its rows to `run_roots` with a `restored to run_roots` note and lists it
  under the manifest's `restored`.

Resource matching everywhere uses `(kind, location_key, identity)`, never `run_id` or a
bare basename. Adding another Run citing a checkout preserves the first checkout row,
including owner metadata; other duplicate resources refuse. Inventory report schema
`bio-neb.run-ledger-inventory/2` carries all three identity fields in matched, unledgered,
missing, removed, warning, candidate and filled-cell entries. CLI summaries show the scope too.
The CSV header and existing row values remain unchanged.

Current notes retain the public result disposition/status, contact review status and discrepancies,
all retained pass journal locators, logical attempt/result counts, and the compiler's contact
metrics. A metric with no exact total is `unknown`, with its observed lower bound and reason;
known bucket/CPU/H100 IDs survive even when totals are unknown. `observed_ops` counts only
surviving physical units and is not a total when evidence is incomplete. Logical attempts are
not physical contacts. Job IDs stay in notes under the existing resource schema. No provider
inventory is inferred: bucket size, inventory digest and storage class remain empty. Existing
local directory inventory runs once as before; this tool adds no checksum/revalidation pass of
its own. Shared readers retain their existing evidence checks.

There is no delete, copy, rename, lifecycle, class-change or owner-column path in this tool;
`tests/test_contact.py` pins the allowed contact set and greps the provider module for mutation
names. Tests: `uv run --locked --offline python -m unittest tests.test_run_ledger_add tests.test_run_ledger_contact tests.test_run_ledger_inventory tests.test_run_ledger_ledger`;
the repository contract on the committed CSV is `tests/test_run_ledger_contract.py`.

Module map: `ledger.py` (schema, stable CSV, listing hashes, run-id derivation), `add.py`,
`inventory.py` (local walk, diff, candidates, warnings, fill), `provider.py` (the only module
that lists the provider), `cli.py`.
