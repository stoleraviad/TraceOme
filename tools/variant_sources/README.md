# Variant-source spike (lane Rf, #331; failure path #352; #343's result shape #353)

One variant lookup for any caller: the Investigation's source reading, the reference tool (#330,
#332) and the paper targets the reference tool places. A developer entry over the product's source
operations since #417: `src/` never imports it, every GET is a product operation fetched through the
product's one seam (`bio_neb.providers.eva_investigation.fetch_source`) over its transport family,
and the tool keeps no transport, contact ledger, report rule or source table of its own. Its
records are not Journey evidence.

```sh
uv run --locked --offline python -m tools.variant_sources look --assembly GCF_000001405.40 --source dbsnp --rsid rs429358   # contact: NCBI dbSNP
uv run --locked --offline python -m tools.variant_sources look --assembly GCF_000003025.6 --source eva --rsid rs5334475172  # contact: EVA
uv run --locked --offline python -m tools.variant_sources look --assembly GCF_000002315.4 --source eva \
    --site SV1=NC_006089.4:86770373:A>C                                               # contact: NCBI API, EVA
```

Contact runs only under an owner-approved run plan. Each GET leaves through the product's
`SourceTransport` (`bio_neb/providers/source_transport.py`): its hosts only (any other is refused
before sending), redirects never followed, the surface's byte bound and a 15 s timeout per socket
operation. Outside a Run no contact journal is written (owner, #417); `--max-contacts` and the
cache's contact log are retired with the tool's ledger, and a record counts its GETs
(`contact_count`).

- **The claim** (`VariantClaim`, `tools/variant_sources/contracts.py`): one variant as stated, tied
  to the assembly its coordinates are on: an rs ID, a position (contig as written, 1-based
  position, REF and ALT in VCF form, of any length), HGVS names, or several, and the item's own
  words (`request_text`, the product request item's field, #342). It follows the paper
  prototype's `ReportedVariant`, keyed by an assembly accession.
  #332 replaces #330's `SnvClaim` with it.
- **The source** (`tools/variant_sources/lookup.py`): `--source dbsnp|eva`, the developer's choice,
  as the product's model makes it (#469: no code routes by assembly; dbSNP keeps human variants,
  EVA non-human ones). A claim with a position takes the product's position lookup from the named
  source (below). An HGVS-only claim is `not_checked` with the reason.
- **Failures** (`bio_neb.target_request.source_failure`, the product's since #367): every dbSNP and EVA GET runs through the
  product's failure module, `call_source` → `call_with_failure_policy`, under the policy
  `policy_for` resolves for its own Tool surface (`fetch_refsnp`, and the position lookup's
  `fetch_assembly_report`, `ncbi_sequence_names`, `ncbi_reference_span`, `ncbi_spdi_rsids`; EVA's
  `eva_clustered_variant`, `eva_contig_alias`, `eva_submitted_variants`, `eva_clustered_variants_at`). A transient fault (a 5xx, a 429, a lost
  connection) is retried by that policy (3 attempts, jittered backoff, by default); past the
  allowance the claim's answer is a `SourceFailure` with class `exhausted` and the policy's route
  `ask`. A rejected answer (another 4xx) asks at once. The other claims still run.
- **dbSNP** (`VariantSources.refsnp`): the product's `fetch_refsnp` operation through
  `fetch_source`, `RetrievedSource.create` and `normalize_refsnp`, called in the
  Investigation's order over the product's NCBI transport; the result is the product's
  `RefSnpToolResult`, unchanged, in #368's shape: one candidate per ALT of any class (SNV, MNV,
  insertion, deletion) at one locus, in dbSNP's SPDI form, `ambiguous` for more than one locus, and a claim naming one exact allele keeps only that candidate. A merged rsID is
  followed as the product's Agent does (`investigation_planner` v5): a `merged` record naming
  one current rsID (`ClaimAnswer.merge`) is followed by a lookup of that rsID with the claim's
  rsID as `requested_rsid`, so `ClaimAnswer.dbsnp` carries the requested and the verified rsID
  and the current record's `merged_rsids`. A merge into several rsIDs is the answer itself. A
  test pins the product shapes the route uses (`ProductShapePinTests`, as of main `0ec72245`),
  so a later change in main shows as a red lane test.
- **EVA, rs claims**: the product's EVA lookup, called, not copied (#366 moved it into `src/`):
  `look_up_eva_rsid` over the product's `EvaInvestigationSourceAdapter` and its EVA transport. It
  GETs the clustered
  variant, EVA's contig aliases for each placement sequence (the Run's RefSeq name and the GCA/GCF
  pair) and the submitted variants. The answer (`ClaimAnswer.eva`) is the product's
  `RefSnpToolResult` with `lookup_source: "eva"` and its EVA detail. An rs claim reads no NCBI
  assembly report.
- **Position claims** (`tools/variant_sources/lookup.py`, #393 Q10): the product's
  `look_up_stated_position` (#369), the Run's one mechanism, called with the tool's `fetch`
  (`VariantSources.fetch`, each GET on its own surface). The claim's assembly is its stated
  version. The lookup reads NCBI's assembly and sequence reports (naming a paper's `10` or
  `chr10`), NCBI's reference bases at the stated REF, the rsIDs the routed source lists at the
  stated allele (NCBI's SPDI listing for a human assembly, EVA at its EVA form otherwise), then
  that one rsID's record. The answer (`ClaimAnswer.position`) is its `RefSnpToolResult`
  (`resolved`, `unlisted`, `conflicted` when the reference holds another base, `incompatible`,
  `ambiguous`, ...). The lane's own position lookup (its EVA form and position
  lookup), its NCBI FTP assembly names and its local `samtools` REF check are retired.

The tool creates its cache when it runs, in .bio-neb-local/variant-sources/ under the working
directory (git-ignored; `--cache DIR` moves it): `lookups/<time>-<accession>-<digest>.json` (one
`bio-neb.variant-lookup/7` record per run: answers, failures and this run's GET count). Each claim
fails on its own; the command exits 2 when a claim's failure asks, listing each (#393 Q16). Our
bug stops the run: the record keeps the claim it stopped on, why, and the claims it did not try,
and the command exits 1.

## Gene plus change to position targets (#376, G15)

G15 is the product's since #367: `bio_neb.target_request.ncbi_sources.answer_claim` (with
`gene_change.py` and `annotation.py`) answers a paper's gene plus protein or coding change on the
Run's assembly inside placement, and the one request writer writes its candidates as placed
positions (the target request). The
`targets` command and its `/2` writer are archived
(`.archive/2026-10-22-issue367-target-request/`); `version` stays here.

