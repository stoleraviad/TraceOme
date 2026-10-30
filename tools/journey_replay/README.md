# Journey replay-equivalence oracle

This maintained offline development tool runs one retained Journey bundle through the raw Journey
reader and both public typed projection queries at two exact, clean source revisions. It compares
their normalized meaning with strict equality. It does not import into `src/bio_neb/` or execute a
workflow command.

Run the tool as the child argv of the repository's canonical operator capsule
(guide). Both source checkouts
must be clean and their full SHAs must match the command. Bind every placeholder to a literal
absolute path; from the candidate checkout the outer command is:

```text
uv run python \
  -m bio_neb.devtools.native_ubuntu_operator start \
  --run-root <fresh-external-operator-root> \
  --run-id <operator-root-basename> \
  --checkout <candidate-checkout> \
  --expected-sha <candidate-40-character-sha> -- \
  uv run --locked --offline python \
  -m tools.journey_replay replay \
  --baseline-checkout /absolute/baseline-checkout \
  --baseline-sha <40-character-sha> \
  --candidate-checkout /absolute/candidate-checkout \
  --candidate-sha <40-character-sha> \
  --journey-file /absolute/run-id/journey-id/journey.jsonl \
  --output-directory /absolute/fresh-output-directory
```

Observe that one attempt with the operator's `status` command and retain its receipts. The replay
invocation after `--` is child argv, not a command to run outside the capsule.

The output directory is created exclusively and contains `baseline.json`, `candidate.json` and
`comparison.json`. Each revision reads an independently refreshed replica at the same external
path. The worker installs its contact and filesystem-write audit guard before importing selected
product code, and the parent verifies the retained inventory across each capture. This is a Python
audit boundary, not an operating-system sandbox. Exit zero and `outcome: identical` mean the
normalized raw events, `observe` projection and `review_journey` projection are exactly equal. Any
difference returns exit one and `outcome: difference_requires_investigation`.

Normalization is schema-aware. Journey, event and Attempt identities are mapped only in their
identity/reference fields, preserving repeated aliases and the parent graph. Event identities in
the retained stream use event order; referenced event identities outside that stream use a
separate first-occurrence namespace so adding an event cannot relabel existing external evidence.
`occurred_at` is replaced only on a Journey event envelope. Event order, event type, payload
content, artifact ids and digests, byte counts, content-addressed identities, scientific values and
other timestamps remain exact. Random-looking strings in ordinary payload fields are never
normalized.

Differences identify their JSON Pointer and classify added or removed fields, changed values, and
changed structure. The comparison has no allowlist and no disposition can turn a difference into
equality. Every difference needs retained evidence that disposes it as a regression, an intentional
ticket-authorized change, or an oracle limitation. An unexplained difference blocks acceptance.

For a controlled negative demonstration, use `capture` before and after perturbing a fresh replica
of a retained Journey bundle at the same absolute path, then use `compare` on those captures. Never
perturb the retained Run. The expected detector difference still requires a written disposition.

The oracle covers only reader and projection grammar for the supplied retained prefix. It does not
rerun Journey generation, prove live success, cover unexercised paths, or detect a change that both
revisions make consistently. Workflow-generation changes need a separate replay adapter and the
post-Run examination owned by ticket 254.
