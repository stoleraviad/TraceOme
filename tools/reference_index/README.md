# Reference-index spike (lane Rf, #319, #330, #332 and #385)

Finds prebuilt BWA indexes for an assembly accession, checks whether one fits the NCBI FASTA, and
answers one assembly request in one step: pick, choice card or cache hit. A developer tool: `src/`
never imports it, and its records are not Journey evidence (they become Journey facts at the map #4
internalize ticket).

```sh
uv run --locked --offline python -m tools.reference_index ask GCF_000146045.2         # contact: NCBI Datasets, Galaxy, NCBI FTP
uv run --locked --offline python -m tools.reference_index take GCF_000002315.4 prebuilt:galaxy:brc:GCF_016699485.2 \
    --snv SV1=NC_006089.4:86770373:A>C --snv SV2=NC_006089.4:86870271:A>C        # contact: Galaxy, NCBI FTP
uv run --locked --offline python -m tools.reference_index catalogues          # contact: Galaxy
uv run --locked --offline python -m tools.reference_index find GCF_000146045.2
uv run --locked --offline python -m tools.reference_index fetch-bundle GCF_000146045.2 galaxy:brc:GCF_000146045.2  # contact: Galaxy
uv run --locked --offline python -m tools.reference_index fetch-ncbi GCF_000146045.2 R64  # contact: NCBI
uv run --locked --offline python -m tools.reference_index fit GCF_000146045.2 galaxy:brc:GCF_000146045.2
uv run --locked --offline python -m tools.reference_index fetch-sequence NC_001143.4  # contact: NCBI efetch
uv run --locked --offline python -m tools.reference_index move GCF_016699485.2 --choice GCF_000002315.4
uv run --locked --offline python -m tools.reference_index move GCF_000146045.2 --source-assembly "UCSC sacCer1" \
    --source-fasta .bio-neb-local/reference-index/ncbi/sequences/NC_001143.4.fa --snv C4=NC_001143.4:207469:A>C
```

Contact commands run only under an owner-approved run plan. Each GET leaves through the product's
source transport family (`bio_neb/providers/source_transport.py`): downloads through
`SourceTransport.stream` to the hosts the plan names (`PLAN_HOSTS` in
`tools/reference_index/__main__.py`; any other host is refused before sending, and a redirect is
followed by `fetch.py` only as a separately checked hop; a short body continues through the
product's `SourceDownload`, in Range parts each retried by the bound failure policy), NCBI
Datasets reads and lookups as product source operations on their own failure surfaces. Each fetched file keeps a receipt beside
it (`<file>.fetched.json`). No contact ledger and no `--max-contacts` remain (owner, #417):
outside a Run no contact journal is written, and records count their GETs.

- `ask` (`tools/reference_index/choice.py`): a request already chosen is a cache hit with no
  contact. Otherwise it describes the requested assembly and lists its species through the
  product's NCBI Datasets adapter (`bio_neb.providers.ncbi_datasets`, on the `inspect_reference`
  and `discover_assemblies` surfaces in `datasets.py`; an accession NCBI no longer lists as current
  is described by the adapter's `describe_assembly`), lists the Galaxy candidates, and
  picks without asking when an
  exact-accession prebuilt matches every contig by M5 and passes every index check. Anything else
  is a choice card: build the requested assembly (the default), a prebuilt of it (with the fit's
  reasons when it was refused), the species' newer reference genome as a prebuilt or a build
  (naming the SNV chain: #332 moves the SNVs, #331 looks them up), and an older prebuilt (Galaxy's
  UCSC-keyed rows through `UCSC_ALIASES`, confirmed by NCBI). Each option carries its cost and fit.
  A species row keyed by a UCSC build with no `UCSC_ALIASES` row is not offered, and says so.
- `take`: takes a card's prebuilt option: fetch, fit against that assembly's NCBI FASTA, store,
  and record the choice with the SNVs to move.
- Store (`tools/reference_index/store.py`): `store/sha256/<digest>/` is the product's assembly
  of a fitted prebuilt (`bio_neb.prebuilt_index.bundle.assemble_prepared_roles`, #365): the
  eight prepared roles, renamed to NCBI's names after a `fits_after_rename` fit, the `.dict` made
  with pysam's samtools, read back through the product's reader against the fit
  (`assembly_refused` otherwise). The digest covers the fetched FASTA and five BWA files, taken
  from the fit report.
- `find`: the product's candidates (`bio_neb.prebuilt_index.catalogue.series_candidates`, #365)
  of the requested assembly series (same GCA/GCF number), one per publication
  (`galaxy:<brc|vgp>:<accession>`), from the saved datacache catalogues, with naming style,
  contig count and content class read from each fetched `.fai` (or `not_measured`).
  `tools/reference_index/catalogue.py`. No curated or recorded rows: everything is retrieved at
  run time (owner, #365 SP3 a). Tool-only until #450: usegalaxy.org's list and `UCSC_ALIASES`
  feed only the card's older and same-species rows; each alias carries its proof (owner, #393 G4).
- `fit`: the product's fit (`bio_neb.prebuilt_index.fit`, #365): pairs contigs by M5 (names, line
  width and soft-masking do not matter), then checks the candidate's index against its own FASTA
  through the product's one BWA-index reader (`bio_neb.parabricks.bwa_index`, #449): `.fai`,
  `.ann`, `.amb`, `.pac` size and trailer, `.bwt`/`.sa` headers. The fit reports every finding,
  and a file the reader cannot read as its format refuses with the reader's reason. Verdict
  `fits`, `fits_after_rename` or `refused` with reasons. The `.pac` content is not reproduced
  with `bwa fa2pac` (#365 R2 a). The command prints the report with the receipts of the files it
  read (`sources`).
- `auto_pick` is set only when a fit report matched every contig by M5 and every index check
  passed. `tools/reference_index/contracts.py` enforces it, and a `Choice` is auto-picked only
  on such a fit.
- `move` (`tools/reference_index/move.py`, #332): places SNVs stated on assembly A on assembly B,
  whose index is the stored bundle a choice record names for B. An SNV with an rs ID (`--rsid
  LABEL=rsN`) first asks #331's variant source for its placement on B (dbSNP for a human B, EVA
  otherwise; its contacts are logged in that tool's log, to its hosts). A placement counts only
  when its SNV alleles there are the SNV's (B in A's orientation) or their complement (B
  reversed: `strand_flipped`), and B's FASTA has the REF they imply; an A/T or C/G SNV cannot
  show its strand this way. Otherwise, or when the source does not place it on B, 200 bases
  each side are read from A's FASTA and aligned with `bwa mem` to B's index, and the record
  keeps the source's answer and why it was not used. A lookup the source's run plan refuses,
  or an assembly report it cannot read, stops the move (`lookup_refused`, `lookup_failed`). Each SNV gets one result: `unique` (contig, position, strand, MAPQ and
  B's base), `strand_flipped` (unique on B's reverse strand; the alleles are complemented),
  `ref_changed` (B's base is not A's REF on that strand), `several_places` (MAPQ under 30;
  bwa's alternatives are listed) or `not_found` (unaligned, under 0.9 of the flank bases, or the
  SNV's base clipped or inserted). Both coordinates and the method are kept. An SNV whose
  contig, position or REF does not hold on A is refused before anything is aligned or asked.
  `--choice A` moves the choice's SNVs from A's NCBI FASTA; `--source-assembly`, `--source-fasta`
  and `--snv` state them on any FASTA (an assembly NCBI has no record of, as UCSC sacCer1).
- `fetch-sequence`: one sequence's FASTA by accession.version from NCBI efetch, for such an
  assembly; an answer whose header names another sequence is refused and not kept.

## A paper's targets onto the Run's assembly (#385, #393, #367)

Placement and the Run's request file are the product's since #367: `bio_neb.target_request`
(`placement.place`, the resolver order in `resolver.py`, G15, and the one request writer in
`request_document.py`, which writes `bio-neb.investigation-request/3`). Their contract is
the target request. `place-targets` is the
product's developer entry (`bio_neb.devtools.target_placement`) with this tool's bwa flank move
(`stored_move.py`, `StoredMove`) behind the product's `FlankMover` port. The port and its
adapters are development-only (owner, #439 C3; #449): a Run passes no mover, and a target only a
flank move would place waits for its Review as `FlankMoveUnavailable`. #450 moves `move.py` and
`stored_move.py` to `dev/`.

```text
uv run --locked --offline python -m tools.reference_index place-targets --assembly <accession> \
    --paper-targets <targets.json> --request <request.json> [--stated-sequences <table.json>]
uv run --locked --offline python -m tools.reference_index place-targets ... \
    --resume <waiting record> --retry t2 --accept t5                    # the owner's answers
```

The move reads the stated version's sequence from this tool's cache (`fetch-sequence`) and the
Run's stored index; the move's own record goes to `moves/`.

The tool creates its cache when it runs, in .bio-neb-local/reference-index/ under the checkout
(git-ignored; `--cache DIR` moves it): `catalogues/`, `bundles/galaxy/<brc|vgp>/<accession>/`, `ncbi/<accession>/`
(each fetched file with its `.fetched.json` receipt),
`candidates/<accession>.json`, `fit/<accession>/<candidate>-<digest>.json`,
`cards/<accession>/<digest>.json`, `choices/<accession>.json` (written once),
`store/sha256/<digest>/`, `ncbi/sequences/<accession>.fa` and `moves/` (`<target>/<time>-<digest>.json`
records, `bio-neb.reference-index-move/2`, with `queries/` and `alignments/` kept by sha256). `bwa` and `samtools` come from the user-space install under
`~/.local/opt/bio-neb-reference-index/envs/<package>/bin/` (`tools/reference_index/local_tools.py`).
