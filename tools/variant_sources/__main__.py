"""`python -m tools.variant_sources look`: look up variant claims on one assembly.

G15's gene-and-change placement and its request file are the product's
(`bio_neb.target_request`, #367); `place-targets` is their developer entry. `version` resolves stated genome version names against the Run's assembly, literally (the
product's `literal_version_form`, #471).

Claims come from `--rsid` and `--site LABEL=CONTIG:POSITION:REF>ALT`. `--source` names the
variant source to ask, dbSNP or EVA, the developer's choice as the product's model makes it
(#469: no code routes by assembly). Each is answered through `VariantSources`;
one record per run (`bio-neb.variant-lookup/7`) goes to `<cache>/lookups/`, with the
answers and the number of GETs this run made.

These are developer entries over the product's source operations and transports: every GET
is a product operation on its own failure surface, and nothing here keeps a contact ledger.
Outside a Run no contact journal is written (owner, #417); a Run's own lookups are journaled.

One rule for `look` (#393 Q16, Q17): a claim whose source contact the failure
module gave up on keeps that failure as its answer and the claims after it still run; when any
failure asks, the command ends waiting for you (exit status 2), listing each. Only an
unclassified error stops the run (exit status 1), keeping the answers before it and naming
the claims not tried.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from bio_neb.content import canonical_json
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter
from bio_neb.providers.source_transport import Opener
from bio_neb.target_request.ncbi_sources import NcbiSources, resolve_version
from bio_neb.target_request.records import LookupStop
from bio_neb.target_request.source_failure import SourceFailed

from .contracts import LookupRecord, VariantClaim
from .lookup import VariantSources

DEFAULT_CACHE = Path(".bio-neb-local") / "variant-sources"
_SITE = re.compile(r"^([^=]+)=([^:]+):([1-9][0-9]*):([ACGTN]+)>([ACGTN]+)$")


def _site(value: str) -> tuple[str, str, int, str, str]:
    match = _SITE.match(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not LABEL=CONTIG:POSITION:REF>ALT"
        )
    label, contig, position, ref, alt = match.groups()
    return label, contig, int(position), ref, alt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.variant_sources")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)
    look = commands.add_parser("look", help="look up claims on one assembly")
    look.add_argument(
        "--assembly", required=True, help="GCF_/GCA_ accession of the claims"
    )
    look.add_argument(
        "--source",
        required=True,
        choices=("dbsnp", "eva"),
        help="the variant source to ask (dbSNP: human variants; EVA: non-human)",
    )
    look.add_argument("--rsid", action="append", default=[])
    look.add_argument("--site", action="append", default=[], type=_site)
    version = commands.add_parser(
        "version", help="resolve stated genome versions against the Run's assembly"
    )
    version.add_argument("--assembly", required=True, help="the Run's assembly")
    version.add_argument("--name", action="append", required=True)
    return parser


def main(argv: list[str] | None = None, *, opener: Opener | None = None) -> int:
    """Run one command; a test supplies the product transports' opener."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "version":
        root = arguments.cache / "gene-change"
        sources = NcbiSources(
            NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=opener)), root
        )
        return _version(sources, arguments)
    claims = [
        VariantClaim(label=rsid, assembly_accession=arguments.assembly, rsid=rsid)
        for rsid in arguments.rsid
    ] + [
        VariantClaim(
            label=label,
            assembly_accession=arguments.assembly,
            contig=contig,
            position=position,
            ref=ref,
            alt=alt,
        )
        for label, contig, position, ref, alt in arguments.site
    ]
    sources = VariantSources.public(opener)
    answers, stopped = [], None
    for index, claim in enumerate(claims):
        try:
            answers.append(sources.answer(claim, arguments.source))
        except Exception as error:  # our bug: kept with the answers before it
            stopped = LookupStop(
                claim_label=claim.label,
                error=f"{type(error).__name__}: {error}",
                unattempted=tuple(later.label for later in claims[index + 1 :]),
            )
            break
    made_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = LookupRecord(
        made_at=made_at,
        assembly_accession=arguments.assembly,
        answers=tuple(answers),
        stopped=stopped,
        contact_count=sources.contact_count,
    )
    content = canonical_json(record.model_dump(mode="json"))
    path = (
        arguments.cache
        / "lookups"
        / f"{made_at.replace(':', '')}-{arguments.assembly}-"
        f"{hashlib.sha256(content.encode()).hexdigest()[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    asks = []
    for answer in answers:
        if answer.failure:
            outcome = f"{answer.failure.failure_class} {answer.failure.route}"
            asks.append(answer)
        else:
            result = answer.dbsnp or answer.eva or answer.position
            outcome = result.status if result else "not_checked"
        if answer.merge:
            outcome += f" (merged into {answer.merge.merged_into[0]})"
        print(f"{answer.claim.label}\t{answer.route}\t{outcome}")
    print(f"record\t{path}")
    for answer in asks:
        failure = answer.failure
        print(
            f"{answer.claim.label} asks: {failure.failure_class} after "
            f"{failure.contact_attempts} attempts on {failure.tool}: {failure.detail}",
            file=sys.stderr,
        )
    if stopped is not None:
        print(f"stopped at {stopped.claim_label}: {stopped.error}", file=sys.stderr)
        return 1
    return 2 if asks else 0


def _version(sources: NcbiSources, arguments) -> int:
    try:
        run = sources.run_assembly(arguments.assembly)
    except SourceFailed as failed:
        print(
            f"the Run's assembly report asks: {failed.failure.detail}", file=sys.stderr
        )
        return 2
    for name in arguments.name:
        resolved = resolve_version(name, run)
        # Unmatched means only that NCBI's report does not give the name: it may name the
        # Run's assembly elsewhere (WBcel235's report gives no UCSC name, so not ce11).
        names = ", ".join(
            name
            for name in (run.name, run.synonym, run.accession, run.paired_accession)
            if name
        )
        matched = (
            resolved.matched or f"not a name NCBI's report gives the Run ({names})"
        )
        print(f"{name}\t{run.name} ({run.accession})\t{matched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
