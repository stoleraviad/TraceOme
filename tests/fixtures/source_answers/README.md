# Recorded source answers

Real public-source answers, recorded once with read-only GETs (#466), replayed in tests
beneath every source port by `tests/recorded_sources.py`. `manifest.json` lists each
fixture with:

- its axis and case;
- the rule it breaks;
- the ticket that owns the fix;
- the expected-red tests that show the break (`tests/test_source_fixture_cases.py`);
- every answer as `SourceTransport` received it: method, URL, status, kept headers,
  body file, byte count, sha256 and the time it was recorded. Each answer is served for its
  own method and URL only.

Where a case is a file a Run reads rather than a source's answer (a tiny VCF, a paper's table), the
fixture holds it under `inputs`, each with its sha256 and declared origin, instead of
`answers`: `biology-ploidy` and `paper-allele-labels` are synthetic, with invented positions.

A body ending `.gz` is stored gzipped and served as received. Bodies are observed source
facts, not expectations: a test states its expected outcome on its own.

## Recording

`uv run --locked --offline python -m tests.recorded_sources record <fixture-id>...` GETs
each request of those fixtures that is not yet recorded, and stores its answer. The GETs
go through the product's transport, at most two per second. `--again` re-records every
request of the named fixtures. Each answer's body and identity are written together after
its GET succeeds, so a failed GET leaves the stored answers as the manifest names them.

To add a fixture:

1. Write its entry (axis, case, breaks, owner_ticket, expected_red and the answers' method
   and URL) in `manifest.json`.
2. Record it.
3. Add its tests.
4. List each new body under `unregistered_fixtures` in `tests/testing_ownership.json`
   (kind `historical`, the owning ticket, the URL it answers).

A recording is source contact. It needs its ticket's contact authority and is counted
against that ticket's ceiling.

## Derived answers

An answer with a `derived` block is not one GET. It names the recorded `originals` it was
cut from and its transformation, and its fixture's `case` says `DERIVED`. Today the only
one is `variant-many-placements`: no real EVA record with more than five placements was
found in 67 probes (owner Q8 a, 2026-10-27).

## Promoting a live Run's answers

Planned live Runs (#383, #114 and other tickets' fresh Runs) retain every source GET's
status and bytes (#458) in their Journey. A Run that meets one of these cases, or a new
one, is the better source.

The ticket whose Run met the case promotes the answer in its own change:

1. Copy the retained body byte for byte from the Run's Journey artifact, checking its
   sha256.
2. Write its entry with the Run id and source event id under `recorded_from`, in place of
   `recorded_at`.
3. Replace a `derived` answer by the real one, and drop its `originals` and the `DERIVED`
   note.
4. Run `tests.test_recorded_sources` and the fixture's expected-red tests.

No new GET is made, so the promotion needs no contact authority.
