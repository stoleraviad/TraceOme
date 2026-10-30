"""Real public-source answers, recorded once, replayed beneath every source port (#466).

`tests/fixtures/source_answers/manifest.json` lists each fixture: its axis, the case it
holds, the current rule it breaks and the ticket that owns the fix, and every GET's
answer as `SourceTransport` received it (status, kept headers and the body's sha256,
the body stored beside the manifest, gzipped when large). `RecordedOpener` is the
`opener=` of a real `SourceTransport`, so a recorded answer runs through the product's
own adapters, readers and source-answer retention (ADR 0020, 0049). It serves only exact
recorded requests and refuses any other, so a test never reaches a source.

Recording is the one contact path: `python -m tests.recorded_sources record <fixture-id>`
GETs each request of that fixture not yet recorded through the product's transport
(every one with ``--again``) and stores its answer. Nothing else here contacts a source.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import io
import json
import sys
import time
import unittest
import urllib.error
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from functools import cache
from pathlib import Path

from bio_neb.agentic_journey.source_contacts import call_source
from bio_neb.investigation.investigation import RetrievedSource
from bio_neb.providers.eva_investigation import (
    EvaInvestigationSourceAdapter,
    PublicInvestigationSources,
    UrllibEvaTransport,
    fetch_source,
)
from bio_neb.providers.galaxy_datacache import GalaxyDatacache
from bio_neb.providers.ncbi_datasets import (
    NcbiDatasetsReferenceAdapter,
    UrllibNcbiTransport,
)
from bio_neb.providers.ncbi_investigation import NcbiInvestigationSourceAdapter
from bio_neb.providers.source_transport import SourceTransport

ROOT = Path(__file__).resolve().parent / "fixtures" / "source_answers"
MANIFEST = ROOT / "manifest.json"
SCHEMA = "bio-neb.recorded-source-answers/1"

# Headers a reader may use: a body's type and length, a redirect's target, and the
# source version NCBI's metadata view records. Session cookies and dates are not kept.
KEPT_HEADERS = (
    "Content-Type",
    "Content-Length",
    "Content-Encoding",
    "Location",
    "X-Datasets-Version",
    "Last-Modified",
    "ETag",
)
# A body larger than this is stored gzipped and served as received.
GZIP_FROM_BYTES = 256 * 1024
# The pause after each recording GET, under the slowest service's rule (NCBI: 3 per
# second without a key).
RECORDING_PAUSE_SECONDS = 0.5


class UnrecordedSourceRequest(Exception):
    """A request no fixture recorded; the test would otherwise reach a source. Not an
    assertion, so an expected-red test that asks for an unrecorded answer is never
    counted as failing for its rule (`expected_red`)."""


@dataclass(frozen=True)
class RecordedAnswer:
    method: str
    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    # How a derived answer was made from the fixture's recorded originals; None when
    # the answer is one GET as received.
    derived: Mapping[str, object] | None = None


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    axis: str
    case: str
    breaks: str
    owner_ticket: int
    answers: tuple[RecordedAnswer, ...]
    # A tiny local file a Run reads where no source answers (a VCF, a paper's table),
    # by its name.
    inputs: Mapping[str, bytes]


def _body(entry: Mapping[str, object]) -> bytes:
    path = ROOT / str(entry["body"])
    stored = path.read_bytes()
    body = gzip.decompress(stored) if path.suffix == ".gz" else stored
    if (len(body), hashlib.sha256(body).hexdigest()) != (
        entry["byte_count"],
        entry["sha256"],
    ):
        raise ValueError(f"{entry['body']}: bytes differ from the manifest")
    return body


def _manifest() -> dict:
    document = json.loads(MANIFEST.read_text())
    if document.get("schema_version") != SCHEMA:
        raise ValueError("source answers manifest has an unknown schema")
    return document


@cache
def fixtures() -> Mapping[str, Fixture]:
    """Every recorded fixture by id, each body (and each original a derived answer was
    made from) checked against its manifest entry."""
    entries = _manifest()["fixtures"]
    for entry in entries.values():
        for original in entry.get("originals", ()):
            _body(original)
    return {
        fixture_id: Fixture(
            fixture_id=fixture_id,
            axis=entry["axis"],
            case=entry["case"],
            breaks=entry["breaks"],
            owner_ticket=entry["owner_ticket"],
            answers=tuple(
                RecordedAnswer(
                    method=answer["method"],
                    url=answer["url"],
                    status=answer["status"],
                    headers=answer["headers"],
                    body=_body(answer),
                    derived=answer.get("derived"),
                )
                for answer in entry["answers"]
            ),
            inputs={item["name"]: _body(item) for item in entry.get("inputs", ())},
        )
        for fixture_id, entry in entries.items()
    }


def fixture(fixture_id: str) -> Fixture:
    return fixtures()[fixture_id]


class _Response:
    def __init__(self, answer: RecordedAnswer, method: str) -> None:
        self.status = answer.status
        self.headers = Message()
        for name, value in answer.headers.items():
            self.headers[name] = value
        self._body = io.BytesIO(b"" if method == "HEAD" else answer.body)

    def read(self, limit: int = -1) -> bytes:
        return self._body.read(limit)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class RecordedOpener:
    """`SourceTransport`'s opener over the answers of the named fixtures.

    Each answer is served for its own method and URL only. A status of 300 or more is
    raised as `urllib.error.HTTPError` with its body, as the transport's no-redirect
    opener raises it. ``sent`` lists each request in order.
    """

    def __init__(self, *fixture_ids: str) -> None:
        if not fixture_ids:
            raise ValueError("a recorded opener serves named fixtures")
        self.answers: dict[tuple[str, str], RecordedAnswer] = {}
        for fixture_id in fixture_ids:
            for answer in fixture(fixture_id).answers:
                self.answers[(answer.method, answer.url)] = answer
        self.sent: list[tuple[str, str]] = []

    def __call__(self, request, timeout: float):
        del timeout
        method, url = request.get_method(), request.full_url
        self.sent.append((method, url))
        answer = self.answers.get((method, url))
        if answer is None:
            raise UnrecordedSourceRequest(f"{method} {url} was never recorded")
        response = _Response(answer, method)
        if answer.status >= 300:
            raise urllib.error.HTTPError(
                url, answer.status, "recorded", response.headers, response._body
            )
        return response


@dataclass(frozen=True)
class RecordedPorts:
    """The product's public-source adapters, each over its own transport with one
    recorded opener: what a Run's Tools read, served from the named fixtures."""

    opener: RecordedOpener

    @property
    def datasets(self):
        return NcbiDatasetsReferenceAdapter(UrllibNcbiTransport(opener=self.opener))

    @property
    def ncbi(self):
        return NcbiInvestigationSourceAdapter(UrllibNcbiTransport(opener=self.opener))

    @property
    def investigation(self):
        return PublicInvestigationSources(
            self.ncbi,
            EvaInvestigationSourceAdapter(UrllibEvaTransport(opener=self.opener)),
        )

    @property
    def galaxy(self):
        return GalaxyDatacache(opener=self.opener)

    def fetch(self):
        """A Run's `SourceFetch` over the Investigation sources: each GET on its own
        operation's failure surface, its answer kept as a `RetrievedSource`."""
        sources = self.investigation

        def fetch(operation: str, key: str):
            response = call_source(
                operation, lambda: fetch_source(sources, operation, key)
            )
            source = RetrievedSource.create(
                operation=operation,
                request_identity=response.request_identity,
                status_code=response.status_code,
                observed_at=response.observed_at,
                elapsed_time_ms=response.elapsed_time_ms,
                raw_response=response.body,
            )
            return source, response.body

        return fetch


def recorded_ports(*fixture_ids: str) -> RecordedPorts:
    return RecordedPorts(RecordedOpener(*fixture_ids))


def expected_red(owner_ticket: int, rule: str):
    """Mark a test of the correct behaviour that today's ``rule`` fails; the owning
    ticket's fix removes the mark (owner, #466 Q3 a).

    Only a failed assertion is the expected failure. Any other error (an unrecorded
    request, a renamed seam) is reported as an unexpected success with its cause, so
    the suite fails instead of hiding a broken test behind the mark.
    """

    def mark(test):
        @functools.wraps(test)
        def run(self) -> None:
            try:
                test(self)
            except AssertionError:
                raise
            except Exception as error:
                print(
                    f"{self.id()}: expected red for #{owner_ticket} but errored: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )

        run.owner_ticket = owner_ticket
        run.breaks = rule
        return unittest.expectedFailure(run)

    return mark


# Recording -------------------------------------------------------------------------


def _stored_name(fixture_id: str, taken: set[str], body: bytes) -> str:
    """The next free ``<fixture>/<NN>.<kind>`` name, gzipped when the body is large."""
    suffix = ".json" if body[:1] in (b"{", b"[") else ".txt"
    if body[:4] == b"PK\x03\x04":
        suffix = ".zip"
    index = 0
    while any(name.startswith(f"{fixture_id}/{index:02d}.") for name in taken):
        index += 1
    name = f"{fixture_id}/{index:02d}{suffix}"
    return name + ".gz" if len(body) > GZIP_FROM_BYTES and suffix != ".zip" else name


def _write_manifest(document: Mapping[str, object]) -> None:
    staged = MANIFEST.with_name(MANIFEST.name + ".partial")
    staged.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    staged.replace(MANIFEST)


def record(fixture_ids: Iterable[str], *, again: bool = False) -> int:
    """GET each unrecorded request of the named fixtures once (every one ``again``); a
    derived answer is never re-recorded, its originals stay as they were cut.

    Each answer's body and its manifest identity are written together after its GET
    succeeds, so a failed GET leaves the stored answer, and every answer recorded before
    it, as the manifest names them.
    """
    document = _manifest()
    transport = SourceTransport()
    sent = 0
    for fixture_id in fixture_ids:
        entry = document["fixtures"][fixture_id]
        (ROOT / fixture_id).mkdir(parents=True, exist_ok=True)
        for answer in entry["answers"]:
            if "derived" in answer or ("sha256" in answer and not again):
                continue
            received = transport.exchange(
                answer["url"],
                method=answer["method"],
                max_bytes=answer.get("max_bytes", 256 * 1024 * 1024),
                accept=answer.get("accept"),
            )
            sent += 1
            previous = answer.get("body")
            taken = {other["body"] for other in entry["answers"] if "body" in other}
            name = _stored_name(fixture_id, taken - {previous}, received.body)
            stored = received.body
            if name.endswith(".gz"):
                stored = gzip.compress(received.body, mtime=0)
            staged = ROOT / (name + ".partial")
            staged.write_bytes(stored)
            staged.replace(ROOT / name)
            if previous not in (None, name):
                (ROOT / previous).unlink(missing_ok=True)
            answer.update(
                status=received.status,
                headers={
                    header: received.headers[header]
                    for header in KEPT_HEADERS
                    if received.headers.get(header) is not None
                },
                body=name,
                byte_count=len(received.body),
                sha256=hashlib.sha256(received.body).hexdigest(),
                recorded_at=received.observed_at.astimezone(UTC)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            )
            _write_manifest(document)
            time.sleep(RECORDING_PAUSE_SECONDS)
    document["recorded_through"] = datetime.now(UTC).date().isoformat()
    _write_manifest(document)
    return sent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.recorded_sources")
    commands = parser.add_subparsers(dest="command", required=True)
    recording = commands.add_parser("record", help="GET and store named fixtures")
    recording.add_argument("fixture_ids", nargs="+")
    recording.add_argument("--again", action="store_true", help="re-record every GET")
    arguments = parser.parse_args(argv)
    print(f"{record(arguments.fixture_ids, again=arguments.again)} GETs recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
