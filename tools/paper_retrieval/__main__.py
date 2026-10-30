"""Command line for the paper retrieval tool.

  python -m tools.paper_retrieval retrieve REFERENCE --out DIR [--replay DIR]
  python -m tools.paper_retrieval answer --paper DIR --question Q \
      --runtime-config PATH --run-root DIR --out DIR \
      [--capture ABS_JSONL [--cloud] [--trace-attribute KEY=VALUE ...]]
  python -m tools.paper_retrieval outline|grep|read|rows|lookup --paper DIR ...
  python -m tools.paper_retrieval verify --paper DIR --answer FILE [--out FILE]
  python -m tools.paper_retrieval targets --answer FILE --out DIR

``retrieve`` downloads a paper through the product's source operations
(`bio_neb.providers.paper_sources`): REFERENCE is a DOI, PMCID, PMID or title, and only
Europe PMC and the PMC open-data bucket are contacted, through the product's transport,
its retries and its byte bounds; it writes the paper (``paper.json`` and its documents),
``bundle.json`` (what was read) and, as ``bio_neb.devtools.paper_download`` does, each
downloaded file under ``files/`` and ``download.json`` (each file's sha256 and md5, the
GETs sent and any stop). ``--replay`` reads a raw capture folder (``index.jsonl``
and its bodies by digest) instead and contacts nothing. ``answer`` records the folder's
downloaded files as a paper in a fresh development Journey under ``--run-root``, through the
product's intake (``paper_intake.intake.record_paper``), and reads it with the product's
reading (``paper_reading.reading.answer_question``): the configured model is contacted,
under the role's declared budgets; with ``--cloud`` it also exports its trace to Logfire
cloud. The reading commands (``outline``, ``grep``, ``read``, ``rows``, ``lookup``) and
``verify`` read a ``retrieve --out`` folder with the product's reader and contact nothing.
``targets`` writes an answer's findings as proposed targets (``targets.json``,
``paper_targets``'s record) and contacts nothing. No reading logic lives here.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.error
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import asdict
from email.message import Message
from pathlib import Path

from logfire import set_baggage
from pydantic import BaseModel

from bio_neb.agentic_journey.agent_roles import RuntimeBinding
from bio_neb.agentic_journey.runtime_config import read_runtime_config
from bio_neb.devtools.agent_inspection import LocalInspection
from bio_neb.devtools.paper_download import download_record, write_download
from bio_neb.journey import Journey
from bio_neb.paper_intake.documents import SourceFile
from bio_neb.paper_intake.intake import record_paper
from bio_neb.paper_intake.journey import PaperIds, new_intake_id
from bio_neb.paper_reading.answer import answer_markdown, verify
from bio_neb.paper_reading.contracts import PaperAnswer, PaperDocument, ReadingRefused
from bio_neb.paper_reading.reader import Paper, grep, lookup, outline, read, rows
from bio_neb.paper_reading.reading import answer_question
from bio_neb.paper_reading.targets import paper_targets
from bio_neb.providers.paper_sources import PaperSourceAdapter
from bio_neb.target_request.paper_targets import PaperIdentity
from tools.paper_retrieval.tools import RetrievalBundle, retrieve


def _save_documents(documents: Iterable[PaperDocument], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for document in documents:
        path = out / f"{document.document_id}.txt"
        path.parent.mkdir(
            parents=True, exist_ok=True
        )  # a member path: supp:S2/table.tsv
        path.write_text(document.text, encoding="utf-8")


class _Answer(io.BytesIO):
    def __init__(self, status: int, body: bytes, headers: Message) -> None:
        super().__init__(body)
        self.status, self.headers = status, headers


def replay_opener(raw_dir: Path):
    """An opener answering from a raw capture and never reaching a host: each URL gets
    the status, content type and body first recorded for it; a URL with no recorded
    body answers 404."""

    records: dict[str, dict] = {}
    for line in (raw_dir / "index.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for url in (record["url"], record.get("final_url", record["url"])):
            records.setdefault(url, record)

    def opener(request, timeout):
        del timeout
        record = records.get(request.full_url)
        headers = Message()
        if record is None or record.get("sha256") is None:
            headers["Content-Type"] = "text/plain"
            raise urllib.error.HTTPError(
                request.full_url, 404, "not recorded", headers, io.BytesIO(b"")
            )
        headers["Content-Type"] = record.get("content_type") or ""
        body = (raw_dir / record["sha256"]).read_bytes()
        if record["status"] >= 400:
            raise urllib.error.HTTPError(
                request.full_url, record["status"], "", headers, io.BytesIO(body)
            )
        return _Answer(record["status"], body, headers)

    return opener


def _bundle_summary(bundle: RetrievalBundle) -> dict:
    """What was read from the download (``download.json`` holds what was downloaded)."""
    main = bundle.main
    return {
        "reference": bundle.reference,
        "identity": bundle.identity.model_dump(exclude_none=True)
        if bundle.identity
        else None,
        "main": (
            {"url": main.url, "via": bundle.via, "characters": main.char_count}
            if isinstance(main, PaperDocument)
            else ({"code": main.code, "message": main.message} if main else None)
        ),
        "supplementary": [
            {"document_id": s.document_id, "url": s.url, "characters": s.char_count}
            for s in bundle.supplementary
        ],
        "read": [
            {k: v for k, v in asdict(file).items() if k != "text"}
            for file in bundle.files
        ],
    }


def command_retrieve(args: argparse.Namespace) -> int:
    replay = Path(args.replay) if args.replay else None
    adapter = PaperSourceAdapter(opener=replay_opener(replay) if replay else None)
    bundle = retrieve(
        args.reference, adapter, via="recorded" if replay else "direct_http"
    )
    summary = _bundle_summary(bundle)
    downloaded = download_record(bundle.download, adapter.contact_count)
    if replay:
        summary["replayed_from"] = str(replay)
    documents = [
        d for d in (bundle.main, *bundle.supplementary) if isinstance(d, PaperDocument)
    ]
    paper = Paper(bundle.identity, documents)
    recorded = {
        "identity": summary["identity"],
        "documents": [
            {
                "document_id": d.document_id,
                "kind": d.kind,
                "url": d.url,
                "via": bundle.via,
                "characters": d.char_count,
            }
            for d in paper.documents.values()
        ],
    }
    if args.out:
        out = Path(args.out)
        # The listing's ids: each document under the id the paper holds it by (r9 S10).
        _save_documents(paper.documents.values(), out / "documents")
        write_download(bundle.download, adapter.contact_count, out)
        (out / "bundle.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (out / "paper.json").write_text(
            json.dumps(recorded, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                **{k: summary[k] for k in ("identity", "main", "supplementary")},
                "requests": {k: downloaded[k] for k in ("sent", "stop")},
            },
            indent=2,
        )
    )
    return 1 if bundle.download.stop else 0


def _stored_paper(folder: Path, journey: Journey) -> str:
    """Record a ``retrieve --out`` folder's downloaded files as one paper of a
    development Journey, through the product's intake; its ``PaperRecorded`` id."""

    downloaded = json.loads((folder / "download.json").read_text(encoding="utf-8"))
    resolved = downloaded["resolved"] or {}
    files = [
        SourceFile(
            d["name"],
            d["url"],
            d["role"],
            d["media_type"],
            (folder / "files" / d["name"]).read_bytes(),
        )
        for d in downloaded["documents"]
    ]
    parents = [journey.head_event_id] if journey.head_event_id else []

    def record(fact):
        event = journey.record(
            fact.event_type,
            fact.model_dump(mode="json", exclude={"event_type"}),
            parents,
        )
        parents[:] = [event["event_id"]]
        return event

    def retain(fact, content, media_type):
        event = journey.record_with_artifact(
            fact.event_type,
            fact.model_dump(mode="json", exclude={"event_type"}),
            content,
            media_type,
            parents,
        )
        parents[:] = [event["event_id"]]
        return event

    event = record_paper(
        record,
        retain,
        intake_id=new_intake_id(),
        origin="citation",
        files=files,
        locations=[file.url for file in files],
        reference=downloaded["reference"],
        ids=PaperIds(
            **{
                name: resolved.get(name)
                for name in PaperIds.model_fields
                if name in resolved
            }
        ),
    )
    if event is None:
        raise SystemExit(f"{folder}: no readable document to read")
    return event["event_id"]


def command_answer(args: argparse.Namespace) -> int:
    trace = dict(args.trace_attribute)
    with ExitStack() as scope:
        capture = None
        if args.capture:
            # Development traces only: every span of this reading carries the trace
            # attributes beside main's own correlation, so a query finds the run later.
            capture = scope.enter_context(
                LocalInspection(args.capture, cloud=args.cloud)
            )
            scope.enter_context(set_baggage(**trace))
        journey = Journey.create(Path(args.run_root).absolute())
        paper_event_id = _stored_paper(Path(args.paper), journey)
        selections = read_runtime_config(
            Path(args.runtime_config).absolute()
        ).runtime_selections()
        outcome = answer_question(
            journey,
            paper_event_id=paper_event_id,
            question=args.question,
            runtime=RuntimeBinding(selections, paper_event_id),
            environ=os.environ,
        )
    answer = outcome.answer
    requests = {
        "model": outcome.model,
        **asdict(outcome.requests),
        # The answer's findings by outcome; refused_answers counts invalid output calls.
        "findings": None
        if answer is None
        else {
            "accepted": len(answer.findings),
            "unresolved": len(answer.unresolved),
            "refused": len(answer.refused),
        },
        "items": [
            {
                "kind": o.item.kind,
                "subject": o.item.subject,
                "attempt_id": o.attempt_id,
                "stopped": o.stopped,
            }
            for o in outcome.items
        ],
        "journey_file": outcome.journey_file,
        "capture": None
        if capture is None
        else {
            "path": str(args.capture),
            "cloud": args.cloud,
            "trace": trace,
            "errors": capture.errors,
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if answer is not None:
        (out / "answer.json").write_text(answer.model_dump_json(indent=2), "utf-8")
    (out / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
    (out / "answer.md").write_text(
        answer_markdown(answer, outcome.stopped, Paper.stored(journey, paper_event_id)),
        encoding="utf-8",
    )
    print(json.dumps(requests, indent=2))
    # A stopped item's reading waits for its Review: no answer is recorded yet.
    return 1 if answer is None else 0


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("capture requires an absolute path")
    return path


def _trace_attribute(value: str) -> tuple[str, str]:
    key, sep, attribute = value.partition("=")
    if not (key and sep and attribute):
        raise argparse.ArgumentTypeError("a trace attribute is KEY=VALUE")
    return key, attribute


def load_paper(folder: Path) -> Paper:
    """A ``retrieve --out`` folder as the product's reader reads it: its ``paper.json``
    (identity and document listing) and, from ``documents/``, each listed text."""

    recorded = json.loads((folder / "paper.json").read_text(encoding="utf-8"))
    documents = [
        PaperDocument(
            document_id=d["document_id"],
            kind=d["kind"],
            url=d["url"],
            text=(folder / "documents" / f"{d['document_id']}.txt")
            .read_bytes()
            .decode("utf-8"),
        )
        for d in recorded["documents"]
    ]
    identity = recorded["identity"]
    return Paper(
        PaperIdentity.model_validate(identity) if identity else None, documents
    )


def _paper(args: argparse.Namespace) -> Paper:
    return load_paper(Path(args.paper))


def _printed(result: BaseModel) -> int:
    print(result.model_dump_json(indent=2))
    return 1 if isinstance(result, ReadingRefused) else 0


def command_outline(args: argparse.Namespace) -> int:
    return _printed(outline(_paper(args)))


def command_grep(args: argparse.Namespace) -> int:
    return _printed(grep(_paper(args), args.pattern, args.max_hits))


def command_read(args: argparse.Namespace) -> int:
    return _printed(read(_paper(args), args.at, args.offset, args.length))


def command_rows(args: argparse.Namespace) -> int:
    where = dict(item.split("=", 1) for item in args.where)
    return _printed(rows(_paper(args), args.table, where, args.column, args.max_rows))


def command_lookup(args: argparse.Namespace) -> int:
    ids = list(args.ids)
    if args.ids_file:
        ids += Path(args.ids_file).read_text(encoding="utf-8").split()
    found = lookup(_paper(args), ids, args.max_hits)
    print(json.dumps([f.model_dump(mode="json") for f in found], indent=2))
    return 0


def _kept(args: argparse.Namespace, text: str) -> None:
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def command_verify(args: argparse.Namespace) -> int:
    answer = PaperAnswer.model_validate_json(
        Path(args.answer).read_text(encoding="utf-8")
    )
    result = verify(answer, _paper(args))
    _kept(args, result.model_dump_json(indent=2))
    return 0 if result.verified else 1


def command_targets(args: argparse.Namespace) -> int:
    answer = PaperAnswer.model_validate_json(
        Path(args.answer).read_text(encoding="utf-8")
    )
    targets = paper_targets(answer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "targets.json").write_text(targets.model_dump_json(indent=2), "utf-8")
    summary = {
        "asserted_by": targets.asserted_by,
        "targets": len(targets.targets),
        "not_checked": sum(1 for t in targets.targets if t.not_checked),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.paper_retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("retrieve", help="one paper and its documents")
    p.add_argument("reference", help="a DOI, PMCID, PMID or title")
    p.add_argument("--out")
    p.add_argument("--replay", help="a raw capture folder to read instead of any host")
    p.set_defaults(func=command_retrieve)

    p = sub.add_parser(
        "answer", help="read one paper for one question in reading items"
    )
    p.add_argument("--paper", required=True, help="a retrieve --out folder")
    p.add_argument("--question", required=True)
    p.add_argument(
        "--runtime-config", required=True, help="main's runtime configuration"
    )
    p.add_argument(
        "--run-root", required=True, help="development root for the reading's Journey"
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--capture",
        type=_absolute_path,
        metavar="ABSOLUTE_JSONL_PATH",
        help="trace the reading to a fresh local JSONL file (development inspection)",
    )
    p.add_argument(
        "--cloud",
        action="store_true",
        help="also export the trace to Logfire cloud with LOGFIRE_TOKEN",
    )
    p.add_argument(
        "--trace-attribute",
        type=_trace_attribute,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an attribute on every captured span, such as lane=P (repeatable)",
    )
    p.set_defaults(func=command_answer)

    def paper(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--paper",
            required=True,
            help="a retrieve --out folder (paper.json, documents)",
        )

    p = sub.add_parser("outline", help="a retrieved paper's text sections and tables")
    paper(p)
    p.set_defaults(func=command_outline)

    p = sub.add_parser("grep", help="search a retrieved paper (regular expression)")
    paper(p)
    p.add_argument("pattern")
    p.add_argument("--max-hits", type=int, default=10)
    p.set_defaults(func=command_grep)

    p = sub.add_parser("read", help="read a section, or a document at an offset")
    paper(p)
    p.add_argument("at", help="a section id (s3) or a document id (main)")
    p.add_argument("--offset", type=int)
    p.add_argument("--length", type=int, default=3000)
    p.set_defaults(func=command_read)

    p = sub.add_parser("rows", help="a table's rows by named column")
    paper(p)
    p.add_argument("--table")
    p.add_argument("--where", action="append", default=[], help="COLUMN=VALUE")
    p.add_argument("--column", action="append", default=[], help="a column to keep")
    p.add_argument("--max-rows", type=int, default=20)
    p.set_defaults(func=command_rows)

    p = sub.add_parser("lookup", help="look up a list of ids in code, marking misses")
    paper(p)
    p.add_argument("ids", nargs="*")
    p.add_argument("--ids-file", help="a file of ids, one per line")
    p.add_argument("--max-hits", type=int, default=5)
    p.set_defaults(func=command_lookup)

    p = sub.add_parser(
        "verify", help="rebuild an answer's findings from their pointers"
    )
    paper(p)
    p.add_argument("--answer", required=True, help="an answer.json")
    p.add_argument("--out", help="where to keep the result")
    p.set_defaults(func=command_verify)

    p = sub.add_parser(
        "targets",
        help="an answer's findings as proposed targets (targets.json)",
    )
    p.add_argument("--answer", required=True, help="an answer.json")
    p.add_argument("--out", required=True, help="folder for targets.json")
    p.set_defaults(func=command_targets)

    args = parser.parse_args(argv)
    if args.command == "answer" and not args.capture:
        if args.cloud or args.trace_attribute:
            parser.error("--cloud and --trace-attribute need --capture")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
