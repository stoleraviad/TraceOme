"""Command line: `python -m tools.reference_index <command> ... [--cache DIR]`.

- `catalogues`: fetch Galaxy's BWA index catalogues (BRC/VGP `.loc`, usegalaxy.org's list)
  and the BRC `fasta_indexes.loc`; a file already cached costs no request.
- `find <accession>`: list the product's candidates (`galaxy:<brc|vgp>:<accession>`) from
  the saved catalogues, with what their fetched `.fai` files show.
- `fetch-bundle <accession> <candidate id> [--fai-only]`: a candidate's FASTA, `.fai` and
  five BWA files from datacache (its `.fai` only with `--fai-only`).
- `fetch-ncbi <accession> <assembly name>`: the NCBI `_genomic.fna.gz`, checked against
  the folder's `md5checksums.txt`, then unpacked.
- `fit <accession> <candidate id>`: check a fetched candidate against the NCBI FASTA,
  save the report and attach it to the saved candidate list.
- `ask <accession>`: answer one request in one step (`choice.py`): a cache hit, a pick
  on an all-contig M5 match (fetched, fitted and stored), or a choice card.
- `take <accession> <option id> [--snv LABEL=CONTIG:POS:REF>ALT]...`: take a prebuilt
  option of the request's card: fetch, fit, store and record the choice with the SNVs
  the variant chain is to move.
- `place-targets --assembly ACCESSION --paper-targets TARGETS --request REQUEST`: the
  product's developer entry (`bio_neb.devtools.target_placement`) with this tool's bwa
  flank move behind the placement port: a paper's targets placed and written as `/3`.

Records print as JSON (exit 0); a refusal prints its code and detail (exit 2). Every GET
leaves through the product's source transports, to the hosts the command names
(`PLAN_HOSTS`); NCBI's reads are product source operations on their own failure surfaces.
No contact ledger is kept outside a Run (owner, #417).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from bio_neb.content import canonical_json
from bio_neb.devtools import target_placement
from bio_neb.failure import ProviderError
from bio_neb.prebuilt_index.catalogue import CatalogueUnreadable
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.source_transport import Opener, SourceTransport
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError
from tools.reference_index.fit import BundleUnreadable
from tools.variant_sources import VariantSources

from .cache import ReferenceIndexCache
from .choice import ask, take
from .contracts import SnvClaim, SnvToMove
from .datasets import Datasets
from .fetch import Fetcher, FetchFailed, FetchRefused
from .galaxy_rule import assembly_series
from .local_tools import DEFAULT_PREFIX, LocalTools, ToolFailed
from .move import MoveRefused, move
from .operations import (
    BWA_CATALOGUES,
    FASTA_CATALOGUE,
    Refused,
    fetch_bundle,
    fetch_missing,
    fetch_ncbi,
    fetch_sequence,
    find_candidates,
    fit_candidate,
    fit_sources,
)
from .store import StoreRefused
from .stored_move import StoredMove

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[2] / ".bio-neb-local" / "reference-index"
)
PLAN_HOSTS = frozenset(
    {
        "usegalaxy.org",
        "datacache.galaxyproject.org",
        "ftp.ncbi.nlm.nih.gov",
        "api.ncbi.nlm.nih.gov",
        "eutils.ncbi.nlm.nih.gov",
    }
)
"""The approved run plans' hosts: #319 and #330 (Galaxy, NCBI FTP and Datasets) and #332
(NCBI efetch for one sequence). A move's rs ID lookups and `place-targets` read NCBI and
EVA through the product's own transports (`VariantSources`, `NcbiSources`)."""
_SEQUENCE = re.compile(r"[A-Z]{1,2}_?[0-9]{5,9}\.[0-9]+")
VARIANT_CACHE = DEFAULT_CACHE.parent / "variant-sources"


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: Opener | None = None,
    tools: LocalTools | None = None,
) -> int:
    """Run one command; a test supplies the product transports' opener and the local
    tools."""
    parser = argparse.ArgumentParser(prog="python -m tools.reference_index")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalogues", help="fetch Galaxy's BWA index catalogues")
    find_parser = commands.add_parser("find", help="list prebuilt candidates")
    find_parser.add_argument("accession")
    ncbi = commands.add_parser("fetch-ncbi", help="fetch and check the NCBI FASTA")
    ncbi.add_argument("accession")
    ncbi.add_argument("assembly_name")
    sequence = commands.add_parser(
        "fetch-sequence", help="fetch one sequence's FASTA from NCBI efetch"
    )
    sequence.add_argument("accession", help="a sequence accession.version")
    bundle = commands.add_parser(
        "fetch-bundle", help="fetch a candidate's FASTA, .fai and BWA files"
    )
    bundle.add_argument("accession")
    bundle.add_argument("candidate_id")
    bundle.add_argument("--fai-only", action="store_true")
    fit = commands.add_parser("fit", help="check a fetched candidate")
    fit.add_argument("accession")
    fit.add_argument("candidate_id")
    ask_parser = commands.add_parser("ask", help="answer one assembly request")
    ask_parser.add_argument("accession")
    take_parser = commands.add_parser("take", help="take a prebuilt option of a card")
    take_parser.add_argument("accession")
    take_parser.add_argument("option_id")
    take_parser.add_argument("--snv", action="append", default=[])
    move_parser = commands.add_parser(
        "move", help="move SNVs onto an assembly's stored index"
    )
    move_parser.add_argument("accession", help="the target assembly")
    move_parser.add_argument("--choice", help="move this request's SNVs to move")
    move_parser.add_argument("--source-assembly", help="the SNVs' assembly, as stated")
    move_parser.add_argument("--source-fasta", type=Path, help="that assembly's FASTA")
    move_parser.add_argument("--snv", action="append", default=[])
    move_parser.add_argument(
        "--rsid", action="append", default=[], help="LABEL=rsN: look it up on B first"
    )
    move_parser.add_argument(
        "--source",
        choices=("dbsnp", "eva"),
        help="the variant source an --rsid is asked of (dbSNP: human; EVA: non-human)",
    )
    move_parser.add_argument("--variant-cache", type=Path, default=VARIANT_CACHE)
    move_parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    place = commands.add_parser(
        "place-targets",
        help="a paper's targets onto the Run's assembly as a /3 request",
    )
    target_placement.add_arguments(place)
    place.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    for command in commands.choices.values():
        command.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    arguments = parser.parse_args(argv)
    cache = ReferenceIndexCache(arguments.cache)
    fetcher = Fetcher(cache, SourceTransport(PLAN_HOSTS, opener=opener))
    ncbi = Datasets(UrllibNcbiTransport(opener=opener))
    try:
        if arguments.command == "fetch-sequence":
            if not _SEQUENCE.fullmatch(arguments.accession):
                raise Refused("invalid_accession", arguments.accession)
        elif arguments.command != "catalogues" and not assembly_series(
            arguments.accession
        ):
            raise Refused("invalid_accession", arguments.accession)
        if arguments.command == "catalogues":
            result = fetch_missing(
                fetcher,
                [
                    (url, cache.root / "catalogues" / name)
                    for name, url in (*BWA_CATALOGUES.items(), FASTA_CATALOGUE)
                ],
            )
        elif arguments.command == "fetch-bundle":
            result = fetch_bundle(
                cache,
                fetcher,
                arguments.accession,
                arguments.candidate_id,
                fai_only=arguments.fai_only,
            )
        elif arguments.command == "find":
            listed = find_candidates(cache, arguments.accession)
            cache.save_candidates(listed)
            result = listed.model_dump(mode="json")
        elif arguments.command == "ask":
            result = ask(cache, fetcher, ncbi, arguments.accession).model_dump(
                mode="json"
            )
        elif arguments.command == "take":
            result = take(
                cache,
                fetcher,
                ncbi,
                arguments.accession,
                arguments.option_id,
                tuple(_snv(text, arguments.accession) for text in arguments.snv),
            ).model_dump(mode="json")
        elif arguments.command == "move":
            stated = (arguments.source_assembly, arguments.source_fasta, arguments.snv)
            if any(stated) if arguments.choice else not all(stated):
                raise Refused(
                    "invalid_move",
                    "move takes --choice, or --source-assembly, --source-fasta and --snv",
                )
            result = move(
                cache,
                tools or LocalTools(arguments.prefix),
                arguments.accession,
                choice=arguments.choice,
                source_assembly=arguments.source_assembly,
                source_fasta=arguments.source_fasta,
                snvs=tuple(
                    _snv_to_move(text, arguments.source_assembly)
                    for text in arguments.snv
                ),
                rsids=dict(_rsid(text) for text in arguments.rsid),
                lookups=VariantSources.public(opener),
                source=arguments.source,
            ).model_dump(mode="json")
        elif arguments.command == "place-targets":
            # The product's developer entry, with this tool's bwa move behind its port.
            return target_placement.run(
                arguments,
                opener=opener,
                mover=StoredMove(
                    cache,
                    tools or LocalTools(arguments.prefix),
                    VariantSources.public(opener),
                ),
            )
        elif arguments.command == "fetch-sequence":
            result = fetch_sequence(cache, fetcher, arguments.accession)
        elif arguments.command == "fetch-ncbi":
            result = fetch_ncbi(
                cache, fetcher, arguments.accession, arguments.assembly_name
            )
        else:
            report = fit_candidate(cache, arguments.accession, arguments.candidate_id)
            sources = fit_sources(cache, arguments.accession, arguments.candidate_id)
            result = {
                **report.model_dump(mode="json"),
                "sources": [source.model_dump(mode="json") for source in sources],
            }
    except Refused as refusal:
        _print({"refused": refusal.code, "detail": refusal.detail})
        return 2
    except (FetchRefused, FetchFailed, ToolFailed, CatalogueUnreadable) as error:
        _print({"refused": type(error).__name__, "detail": str(error)})
        return 2
    except ProviderError as error:
        # A source's refusal, or a contact the failure module gave up on.
        _print({"refused": _refusal_code(error), "detail": error.detail})
        return 2
    except (StoreRefused, MoveRefused) as error:
        _print({"refused": error.reason, "detail": error.detail})
        return 2
    except BundleUnreadable as error:
        _print({"refused": error.reason, "detail": str(error.path)})
        return 2
    _print(result)
    return 0


def _refusal_code(error: ProviderError) -> str:
    """NCBI Datasets' own code for its refusal; the failure class for any other."""
    if isinstance(error, ReferenceAdapterError):
        return error.failure.code
    return error.failure_class


def _snv(text: str, accession: str) -> SnvClaim:
    """`LABEL=CONTIG:POS:REF>ALT`, stated on the requested assembly."""
    label, contig, position, ref, alt = _snv_fields(text)
    try:
        return SnvClaim(
            label=label,
            assembly_accession=accession,
            contig=contig,
            position=position,
            ref=ref,
            alt=alt,
        )
    except ValidationError as error:
        raise Refused("invalid_snv", text) from error


def _snv_to_move(text: str, assembly: str) -> SnvToMove:
    """`LABEL=CONTIG:POS:REF>ALT`, stated on the source assembly."""
    label, contig, position, ref, alt = _snv_fields(text)
    try:
        return SnvToMove(
            label=label,
            assembly=assembly,
            contig=contig,
            position=position,
            ref=ref,
            alt=alt,
        )
    except ValidationError as error:
        raise Refused("invalid_snv", text) from error


def _rsid(text: str) -> tuple[str, str]:
    """`LABEL=rsN`: the rs ID of the SNV with that label."""
    match = re.fullmatch(r"([^=]+)=(rs[1-9][0-9]*)", text)
    if match is None:
        raise Refused("invalid_rsid", text)
    return match.group(1), match.group(2)


def _snv_fields(text: str) -> tuple[str, str, int, str, str]:
    match = re.fullmatch(r"([^=]+)=(.+):(\d+):([A-Za-z]+)>([A-Za-z]+)", text)
    if match is None:
        raise Refused("invalid_snv", text)
    label, contig, position, ref, alt = match.groups()
    return label, contig, int(position), ref.upper(), alt.upper()


def _print(value: object) -> None:
    sys.stdout.write(json.dumps(json.loads(canonical_json(value)), indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
