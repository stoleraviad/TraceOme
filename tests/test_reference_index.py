"""The reference-index spike lists prebuilt BWA index candidates and checks whether one fits."""

import contextlib
import email.message
import gzip
import hashlib
import http.client
import http.server
import io
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

from bio_neb.content import canonical_json
from bio_neb.prebuilt_index.catalogue import parse_loc
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.source_transport import SourceTransport, SourceUrlRefused
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError
from tests.prebuilt_index_support import (
    CHROMOSOME_II,
    NCBI,
)
from tests.prebuilt_index_support import fai_text as _fai
from tests.prebuilt_index_support import galaxy_contigs as _galaxy
from tests.prebuilt_index_support import write_bundle as _write_bundle
from tests.reference_index_support import (
    INSTALL,
    INSTALLED,
    SyntheticPair,
    choice_record,
    write_fasta,
)
from tools.reference_index import (
    Candidate,
    Fetcher,
    FetchFailed,
    LocalTools,
    ReferenceIndexCache,
    ToolFailed,
    attach_fit,
    carry_fits,
    find,
    ncbi_md5,
)
from tools.reference_index import bundle as reference_index_bundle
from tools.reference_index import choice as reference_index_choice
from tools.reference_index import store as reference_index_store
from tools.reference_index.__main__ import main
from tools.reference_index.contracts import SnvClaim, SnvToMove
from tools.reference_index.datasets import Datasets
from tools.reference_index.fit import check_fit
from tools.reference_index.move import MoveRefused, move_snvs
from tools.variant_sources import VariantClaim, VariantSources


class FitFixture:
    """An NCBI-style requested FASTA and an empty Galaxy-style candidate folder."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.requested = self.root / "ncbi" / "GCF_000000001.1_genomic.fna"
        self.requested.parent.mkdir()
        write_fasta(self.requested, NCBI, width=80)
        (self.root / "galaxy").mkdir()
        self.candidate = self.root / "galaxy" / "GCF_000000001.1.fa"

    def check(self):
        return check_fit(
            requested=self.requested,
            requested_accession="GCF_000000001.1",
            candidate=self.candidate,
            candidate_id="galaxy:brc:GCF_000000001.1",
        )


def _loc_row(accession, path, *, name=None):
    return f"{accession}\t{accession}\t{name or accession}\t{path}"


BRC_GENOMES = "/cvmfs/brc.galaxyproject.org/data/genomes/"


def _brc(*accessions, names=None, index_names=None):
    """BRC's index and FASTA catalogue rows as datacache publishes them: each
    accession's index under `bwa_mem_index/`, its FASTA under `seq/`."""
    names, index_names = names or {}, index_names or {}
    index = parse_loc(
        "\n".join(
            _loc_row(
                accession,
                f"{BRC_GENOMES}{accession}/bwa_mem_index/v1/{accession}/"
                f"{index_names.get(accession, accession + '.fa')}",
                name=names.get(accession),
            )
            for accession in accessions
        ),
        catalogue="brc-bwa_mem_index.loc",
    )
    fasta = parse_loc(
        "\n".join(
            _loc_row(accession, f"{BRC_GENOMES}{accession}/seq/{accession}.fa")
            for accession in accessions
        ),
        catalogue="brc-fasta_indexes.loc",
    )
    return index, fasta


YEAST_FAI_NAMES = (
    *(f"NC_0011{n}.{v}" for n, v in zip(range(33, 49), [9] + [8] * 15, strict=True)),
    "NC_001224.1",
)
HG38_FAI_NAMES = (
    "chr1",
    "chr1_KI270706v1_random",
    "chr1_KI270762v1_alt",
    "chrUn_GL000195v1",
    "chrM",
)


class FindTests(unittest.TestCase):
    """`find` lists the product's candidates (`series_candidates`, #365) with the
    tool's display; no curated or recorded rows (owner, #365 SP3 a)."""

    def catalogue(self):
        return _brc(
            "GCF_000146045.2",
            "GCA_000146045.2",
            "GCF_000002985.6",
            "GCF_000001405.40",
            names={"GCF_000001405.40": "Human (hg38)"},
            index_names={"GCF_000001405.40": "hg38.fa"},
        )

    def test_find_lists_the_galaxy_candidates_of_an_accession(self):
        listed = find(
            "GCF_000146045.2",
            *self.catalogue(),
            fai_names={"galaxy:brc:GCF_000146045.2": YEAST_FAI_NAMES},
        )

        self.assertEqual(listed.requested_accession, "GCF_000146045.2")
        self.assertEqual(
            [
                (
                    c.candidate_id,
                    c.relation,
                    c.naming_style,
                    c.contig_count,
                    c.content_class,
                )
                for c in listed.candidates
            ],
            [
                ("galaxy:brc:GCF_000146045.2", "exact", "refseq", 17, ("primary",)),
                (
                    "galaxy:brc:GCA_000146045.2",
                    "same_assembly_series",
                    "unknown",
                    None,
                    ("unknown",),
                ),
            ],
        )
        self.assertEqual(
            listed.candidates[0].location,
            "http://datacache.galaxyproject.org/brc/data/genomes/GCF_000146045.2/"
            "bwa_mem_index/v1/GCF_000146045.2/GCF_000146045.2.fa",
        )
        self.assertFalse(any(c.auto_pick for c in listed.candidates))

    def test_find_reads_the_naming_and_content_of_a_fetched_fai(self):
        listed = find(
            "GCF_000001405.40",
            *self.catalogue(),
            fai_names={"galaxy:brc:GCF_000001405.40": HG38_FAI_NAMES},
        )

        (galaxy,) = listed.candidates
        self.assertEqual(
            (
                galaxy.candidate_id,
                galaxy.display_name,
                galaxy.naming_style,
                galaxy.contig_count,
                galaxy.content_class,
            ),
            (
                "galaxy:brc:GCF_000001405.40",
                "Human (hg38)",
                "ucsc",
                5,
                ("primary", "alts"),
            ),
        )

    def test_find_lists_no_row_the_product_rule_leaves_out(self):
        """A row with no datacache path, or no FASTA row, is not a candidate; a human
        request lists the catalogue's own rows, never an analysis set it does not."""
        index, fasta = self.catalogue()
        unlisted = parse_loc(
            _loc_row("GCF_000146045.2", "/brc/GCF_000146045.2/bwa/x.fa"),
            catalogue="vgp-bwa_mem_index.loc",
        )

        for rows, fastas in ((unlisted, fasta), (index, ())):
            with self.subTest(rows=rows[0].catalogue, fastas=len(fastas)):
                self.assertEqual(find("GCF_000146045.2", rows, fastas).candidates, ())
        self.assertEqual(
            [
                (c.candidate_id, c.relation)
                for c in find("GCA_000001405.15", index, fasta).candidates
            ],
            [("galaxy:brc:GCF_000001405.40", "same_assembly_series")],
        )


class AutoPickTests(FitFixture, unittest.TestCase):
    """`auto_pick` follows the fit report: an all-contig M5 match, never a partial one."""

    def listed(self):
        return find("GCF_000000001.1", *_brc("GCF_000000001.1"))

    def test_an_all_contig_match_is_auto_picked(self):
        _write_bundle(self.candidate, _galaxy())
        report = self.check()

        picked = attach_fit(self.listed(), report)

        (candidate,) = picked.candidates
        self.assertTrue(candidate.auto_pick)
        self.assertEqual(candidate.fit.verdict, "fits_after_rename")
        self.assertEqual(
            candidate.fit.report_sha256,
            hashlib.sha256(
                canonical_json(report.model_dump(mode="json")).encode()
            ).hexdigest(),
        )

    def test_a_partial_match_is_never_auto_picked(self):
        _write_bundle(self.candidate, _galaxy()[:2])
        report = self.check()
        self.assertEqual(report.sequence_match, "partial")

        (candidate,) = attach_fit(self.listed(), report).candidates

        self.assertFalse(candidate.auto_pick)
        with self.assertRaisesRegex(ValueError, "auto_pick"):
            candidate.model_validate({**candidate.model_dump(), "auto_pick": True})

    def test_a_fit_summary_short_of_an_all_contig_match_cannot_fit(self):
        _write_bundle(self.candidate, _galaxy())
        (candidate,) = attach_fit(self.listed(), self.check()).candidates
        claimed = candidate.model_dump(mode="json")
        claimed["fit"]["sequence_match"] = "partial"

        with self.assertRaisesRegex(ValueError, "all-contig"):
            Candidate.model_validate(claimed)

    def test_a_refreshed_list_carries_the_retained_fits_of_candidates_still_listed(
        self,
    ):
        _write_bundle(self.candidate, _galaxy())
        previous = attach_fit(self.listed(), self.check())
        rows = _brc("GCF_000000001.1", "GCA_000000001.1")

        refreshed = carry_fits(find("GCF_000000001.1", *rows), previous)

        kept, added = refreshed.candidates
        self.assertEqual((kept.fit, kept.auto_pick), (previous.candidates[0].fit, True))
        self.assertEqual(
            (added.candidate_id, added.fit, added.auto_pick),
            ("galaxy:brc:GCA_000000001.1", None, False),
        )

    def test_a_report_for_another_candidate_is_not_attached(self):
        _write_bundle(self.candidate, _galaxy())
        report = self.check().model_copy(update={"candidate_id": "galaxy:other"})

        with self.assertRaisesRegex(ValueError, "galaxy:other"):
            attach_fit(self.listed(), report)


class FakeOpener:
    """Stands in for the product transports' opener (`SourceTransport(opener=...)`):
    serves scripted bodies by URL, records each request and its timeout."""

    def __init__(self, bodies):
        self.bodies = dict(bodies)
        self.requested = []
        self.timeouts = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.requested.append(url)
        self.timeouts.append(timeout)
        body = self.bodies[url]
        if isinstance(body, Exception):
            raise body
        response = io.BytesIO(body)
        response.status = 200
        response.headers = {"Content-Length": str(len(body))}
        return response


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _local_opener(base):
    """An opener for the product transports that sends each request, already checked
    against the source hosts, to the local server at `base` instead: same path, query,
    method and headers, over a real urllib opener that follows no redirect (a 3xx comes
    back as the `HTTPError` the product's own opener raises)."""
    opener = urllib.request.build_opener(_NoRedirect())

    def open_locally(request, timeout):
        parts = urllib.parse.urlsplit(request.full_url)
        local = base + urllib.parse.urlunsplit(("", "", parts.path, parts.query, ""))
        return opener.open(
            urllib.request.Request(
                local,
                method=request.get_method(),
                headers=dict(request.header_items()),
            ),
            timeout=timeout,
        )

    return open_locally


def _serve(test, routes):
    """A real HTTP server on 127.0.0.1 for one test: path -> (status, headers, body).
    Returns its base URL and the list of paths it was asked for."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            status, headers, body = routes[self.path]
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            if "Content-Length" not in headers:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if "Content-Length" in headers:
                self.close_connection = True

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return f"http://127.0.0.1:{server.server_address[1]}", seen


def _serve_file(test, data, *, cuts, ranges=True, path="/f", others=None):
    """A real HTTP server for one file at `/f` that answers `Range: bytes=a-b` with a
    206 and its `Content-Range`, ending at the file's last byte (unless `ranges` is false: then a 200 with the whole
    file), and closes response number i early after `cuts[i]` bytes while declaring the
    full length; `others` are more paths served whole. Returns its base URL and each
    request's Range header."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            asked = self.headers.get("Range")
            seen.append(asked)
            if self.path != path:
                other = (others or {})[self.path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(other)))
                self.end_headers()
                self.wfile.write(other)
                return
            start, end = 0, len(data) - 1
            if asked and ranges:
                start, end = (int(x) for x in asked.removeprefix("bytes=").split("-"))
                end = min(end, len(data) - 1)  # a range past the end ends at it
            body = data[start : end + 1]
            self.send_response(206 if asked and ranges else 200)
            if asked and ranges:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[: cuts.get(len(seen) - 1, len(body))])
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return f"http://127.0.0.1:{server.server_address[1]}", seen


REPOSITORY = Path(__file__).resolve().parents[1]


class TableRowProofTests(unittest.TestCase):
    """Each row the code still holds carries its proof (owner, #393 Q5, G4): a source in
    this repository and the window it shows, or that nothing was measured."""

    def test_each_ucsc_alias_row_names_its_proof(self):
        rows = [
            (f"UCSC_ALIASES {key}", alias.proof_source, alias.proof_window)
            for key, alias in reference_index_choice.UCSC_ALIASES.items()
        ]
        self.assertEqual(len(rows), 2)
        for row, source, window in rows:
            with self.subTest(row):
                path = REPOSITORY / source.split(" (")[0]
                self.assertTrue(path.is_file(), source)
                self.assertTrue(window.strip())


FTP = "https://ftp.ncbi.nlm.nih.gov"


class FetchTests(unittest.TestCase):
    """`Fetcher` over the product's `SourceTransport` (NCBI's FTP host only): a fake
    opener for scripted answers, a real local server, reached through `_local_opener`,
    for the redirect and Range mechanics."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = ReferenceIndexCache(Path(directory.name) / "cache")

    def transport(self, opener):
        return SourceTransport({"ftp.ncbi.nlm.nih.gov"}, opener=opener)

    def test_a_get_saves_the_whole_file_and_keeps_its_receipt_beside_it(self):
        opener = FakeOpener({f"{FTP}/a": b"alpha"})
        transport = self.transport(opener)
        destination = self.cache.root / "a"

        fetched = Fetcher(self.cache, transport).get(f"{FTP}/a", destination)

        self.assertEqual(opener.requested, [f"{FTP}/a"])
        self.assertEqual(destination.read_bytes(), b"alpha")
        self.assertEqual(
            fetched.model_dump(mode="json"),
            {
                "url": f"{FTP}/a",
                "destination": str(destination),
                "byte_count": 5,
                "sha256": hashlib.sha256(b"alpha").hexdigest(),
                "md5": hashlib.md5(b"alpha").hexdigest(),
            },
        )
        self.assertEqual(
            json.loads((self.cache.root / "a.fetched.json").read_text()),
            fetched.model_dump(mode="json"),
        )
        self.assertEqual(self.cache.receipts_in(self.cache.root), (fetched,))
        self.assertEqual(transport.contact_count, 1)

    def test_a_host_outside_the_transports_hosts_is_refused_before_contact(self):
        opener = FakeOpener({})
        transport = self.transport(opener)
        fetcher = Fetcher(self.cache, transport)

        for url in (
            "https://api.ncbi.nlm.nih.gov/x",  # a source host, not this transport's
            "https://other.test/x",
        ):
            with self.subTest(url), self.assertRaises(SourceUrlRefused) as refused:
                fetcher.get(url, self.cache.root / "x")

            self.assertEqual(
                (refused.exception.failure_class, refused.exception.url),
                ("rejected", url),
            )
        self.assertEqual(opener.requested, [])
        self.assertEqual(transport.contact_count, 0)
        self.assertEqual(list(self.cache.root.iterdir()), [])

    def test_a_failed_get_is_counted_and_leaves_no_file_or_receipt(self):
        error = urllib.error.HTTPError(f"{FTP}/a", 404, "Not Found", {}, None)
        transport = self.transport(FakeOpener({f"{FTP}/a": error}))

        with self.assertRaisesRegex(FetchFailed, "404"):
            Fetcher(self.cache, transport).get(f"{FTP}/a", self.cache.root / "a")

        self.assertEqual(list(self.cache.root.iterdir()), [])
        self.assertEqual(transport.contact_count, 1)

    def test_a_redirect_is_a_counted_hop_through_the_same_checks(self):
        base, seen = _serve(
            self, {"/a": (302, {"Location": "/b"}, b""), "/b": (200, {}, b"beta")}
        )
        transport = self.transport(_local_opener(base))

        fetched = Fetcher(self.cache, transport).get(f"{FTP}/a", self.cache.root / "a")

        self.assertEqual(seen, ["/a", "/b"])
        self.assertEqual(fetched.url, f"{FTP}/b")
        self.assertEqual((self.cache.root / "a").read_bytes(), b"beta")
        self.assertEqual(transport.contact_count, 2)

    def test_a_redirect_off_the_source_hosts_or_past_the_hop_limit_is_not_followed(
        self,
    ):
        base, seen = _serve(
            self,
            {
                "/away": (301, {"Location": "http://other.test/x"}, b""),
                **{f"/r{k}": (302, {"Location": f"/r{k + 1}"}, b"") for k in range(4)},
            },
        )
        cases = (
            ("/away", SourceUrlRefused, "outside the source hosts", ["/away"]),
            ("/r0", FetchFailed, "more than 3 redirects", ["/r0", "/r1", "/r2", "/r3"]),
        )
        for path, error, reason, asked in cases:
            with self.subTest(path):
                del seen[:]
                cache = ReferenceIndexCache(self.cache.root / path.strip("/"))
                transport = self.transport(_local_opener(base))

                with self.assertRaisesRegex(error, reason):
                    Fetcher(cache, transport).get(f"{FTP}{path}", cache.root / "x")

                self.assertEqual(seen, asked)
                self.assertEqual(transport.contact_count, len(asked))
                self.assertEqual(list(cache.root.iterdir()), [])

    def fetcher(self, base):
        self.slept = []
        self.used = self.transport(_local_opener(base))
        return Fetcher(self.cache, self.used, part_bytes=40, sleep=self.slept.append)

    def test_a_short_body_is_a_failure_and_the_rest_comes_in_range_parts(self):
        """Galaxy's datacache closed a GRCg7b FASTA GET after 67,845,940 of 1,074,402,295
        bytes (#330 run, 22:36Z), and http.client returns such a short body without
        error. The rest comes in fixed Range parts, as main's multipart upload sends."""
        data = bytes(range(100))
        base, ranges = _serve_file(self, data, cuts={0: 30})

        fetched = self.fetcher(base).get(f"{FTP}/f", self.cache.root / "f")

        self.assertEqual((self.cache.root / "f").read_bytes(), data)
        self.assertEqual(ranges, [None, "bytes=30-69", "bytes=70-99"])
        self.assertEqual(self.used.contact_count, 3)
        self.assertEqual(
            (fetched.url, fetched.byte_count, fetched.sha256, fetched.md5),
            (
                f"{FTP}/f",
                100,
                hashlib.sha256(data).hexdigest(),
                hashlib.md5(data).hexdigest(),
            ),
        )
        self.assertEqual(self.cache.receipts_in(self.cache.root), (fetched,))
        self.assertFalse((self.cache.root / "f.part").exists())
        self.assertEqual(self.slept, [])

    def test_an_existing_partial_is_continued_in_range_parts(self):
        data = bytes(range(100))
        (self.cache.root).mkdir(parents=True)
        (self.cache.root / "f.part").write_bytes(data[:30])
        base, ranges = _serve_file(self, data, cuts={})

        fetched = self.fetcher(base).get(f"{FTP}/f", self.cache.root / "f")

        self.assertEqual((self.cache.root / "f").read_bytes(), data)
        self.assertEqual(ranges, ["bytes=30-69", "bytes=70-99"])
        self.assertEqual(
            (fetched.byte_count, fetched.sha256),
            (100, hashlib.sha256(data).hexdigest()),
        )

    def test_a_part_received_but_not_saved_stops_without_a_retry(self):
        """A sent part whose bytes cannot be saved is a failure; the fetch stops (no
        retry of a local fault) and keeps the saved prefix."""
        data = bytes(range(100))
        destination = self.cache.root / "f"

        class Response(io.BytesIO):
            def __init__(self, status, headers, body, then=None):
                super().__init__(body)
                self.status, self.headers, self.then = status, headers, then

            def read(self, *args):
                if self.then:
                    self.then()
                return super().read(*args)

        responses = iter(
            [
                Response(200, {"Content-Length": "100"}, data[:30]),
                Response(
                    206,
                    {"Content-Range": "bytes 30-99/100", "Content-Length": "70"},
                    data[30:],
                    then=lambda: destination.mkdir(),
                ),
            ]
        )
        transport = self.transport(lambda request, timeout: next(responses))
        fetcher = Fetcher(self.cache, transport, part_bytes=100, sleep=self.fail)

        with self.assertRaisesRegex(FetchFailed, "received but not saved"):
            fetcher.get(f"{FTP}/f", destination)

        self.assertEqual(transport.contact_count, 2)
        self.assertEqual((self.cache.root / "f.part").read_bytes(), data)
        self.assertEqual(self.cache.receipts_in(self.cache.root), ())

    def test_a_part_gets_the_bound_policys_attempts_then_the_partial_is_kept(self):
        """The parts are the product's `SourceDownload` (#365): three attempts under
        the unbound policy, its backoff of up to 1 s then 2 s jittered."""
        data = bytes(range(100))
        base, ranges = _serve_file(self, data, cuts={0: 30, 1: 5, 2: 5, 3: 5})

        with (
            mock.patch(
                "tools.reference_index.source_download.random.uniform",
                side_effect=lambda low, high: high,
            ),
            self.assertRaisesRegex(
                FetchFailed, "bytes=30-69 failed: short body: 5 of 40 bytes"
            ),
        ):
            self.fetcher(base).get(f"{FTP}/f", self.cache.root / "f")

        self.assertEqual(ranges, [None] + ["bytes=30-69"] * 3)
        self.assertEqual(self.slept, [1.0, 2.0])
        self.assertEqual((self.cache.root / "f.part").read_bytes(), data[:30])
        self.assertFalse((self.cache.root / "f").exists())
        self.assertEqual(self.used.contact_count, 4)

    def test_a_part_answered_without_its_range_is_not_appended(self):
        data = bytes(range(100))
        base, ranges = _serve_file(self, data, cuts={0: 30}, ranges=False)

        with self.assertRaisesRegex(
            FetchFailed, "bytes=30-69 failed: HTTP 200 is not the part"
        ):
            self.fetcher(base).get(f"{FTP}/f", self.cache.root / "f")

        self.assertEqual(ranges, [None, "bytes=30-69"])
        self.assertEqual((self.cache.root / "f.part").read_bytes(), data[:30])
        self.assertEqual(self.used.contact_count, 2)

    def test_an_interrupted_body_fails_and_leaves_no_partial(self):
        class Interrupted(io.BytesIO):
            status = 200
            headers = {}

            def read(self, *args):
                raise http.client.IncompleteRead(b"par", 10)

        transport = self.transport(lambda request, timeout: Interrupted())

        with self.assertRaisesRegex(FetchFailed, "interrupted after 0 bytes"):
            Fetcher(self.cache, transport).get(f"{FTP}/a", self.cache.root / "a")

        self.assertEqual(transport.contact_count, 1)
        self.assertEqual(list(self.cache.root.iterdir()), [])

    def test_a_body_that_cannot_be_put_in_place_fails_and_leaves_no_partial(self):
        """The whole body arrives but cannot be renamed into place (here something
        took its destination meanwhile): the download fails as a `FetchFailed` and no
        partial is left for a later run to continue from."""

        class Blocked(io.BytesIO):
            status = 200
            headers = {}

            def __init__(self, destination):
                super().__init__(b"body")
                self.destination = destination

            def read(self, *args):
                self.destination.mkdir(exist_ok=True)
                return super().read(*args)

        destination = self.cache.root / "b"
        transport = self.transport(lambda request, timeout: Blocked(destination))

        with self.assertRaises(FetchFailed):
            Fetcher(self.cache, transport).get(f"{FTP}/b", destination)

        self.assertEqual(transport.contact_count, 1)
        self.assertEqual(sorted(p.name for p in self.cache.root.iterdir()), ["b"])

    def test_the_ncbi_md5_must_match_md5checksums(self):
        listing = (
            "0123456789abcdef0123456789abcdef  ./GCF_1_R1_genomic.fna.gz\n"
            "fedcba9876543210fedcba9876543210  ./GCF_1_R1_genomic.gff.gz\n"
        )

        self.assertEqual(
            ncbi_md5(listing, "GCF_1_R1_genomic.fna.gz"),
            "0123456789abcdef0123456789abcdef",
        )
        with self.assertRaisesRegex(ValueError, "GCF_1_R1_assembly_report.txt"):
            ncbi_md5(listing, "GCF_1_R1_assembly_report.txt")


SCRIPTED_BWA = """#!/bin/sh
printf '%s\\n' "$@" >> "$0.argv"
if [ "$#" -eq 0 ]; then
  printf '\\nProgram: bwa (alignment via Burrows-Wheeler transformation)\\nVersion: 0.7.17-r1188\\n' >&2
  exit 1
fi
echo "[main] unrecognized command '$1'" >&2
exit 1
"""


class LocalToolsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.prefix = Path(directory.name) / "opt"
        self.bwa = self.prefix / "envs" / "bwa" / "bin" / "bwa"
        self.bwa.parent.mkdir(parents=True)
        self.bwa.write_text(SCRIPTED_BWA)
        self.bwa.chmod(0o755)
        self.tools = LocalTools(self.prefix)

    def argv(self):
        return Path(f"{self.bwa}.argv").read_text().splitlines()

    def test_a_failing_tool_is_refused_with_its_stderr(self):
        with self.assertRaisesRegex(ToolFailed, "unrecognized command 'index'"):
            self.tools.run("bwa", "index", "x.fa")

    def test_bwa_identity_is_its_digest_and_version_line(self):
        identity = self.tools.identity("bwa")

        self.assertEqual(identity.version_line, "Version: 0.7.17-r1188")
        self.assertEqual(
            identity.sha256, hashlib.sha256(SCRIPTED_BWA.encode()).hexdigest()
        )
        self.assertEqual(identity.executable, str(self.bwa))


class StoreTests(FitFixture, unittest.TestCase):
    """`store_bundle` is the product's assembly of a fitted prebuilt (#365)."""

    def test_a_renamed_fit_is_stored_under_ncbis_names_with_true_digests(self):
        _write_bundle(self.candidate, _galaxy())
        report = self.check()
        cache = ReferenceIndexCache(self.root / "cache")

        stored = reference_index_store.store_bundle(cache, report, self.candidate)

        self.assertEqual(report.verdict, "fits_after_rename")
        folder = reference_index_store.bundle_folder(cache, stored.bundle_digest)
        self.assertEqual(
            stored.bundle_digest, reference_index_store.bundle_digest(report)
        )
        self.assertEqual(
            [
                line[1:].split()[0]
                for line in (folder / "reference.fa").read_text().splitlines()
                if line.startswith(">")
            ],
            [name for name, _, _ in NCBI],
        )
        self.assertEqual(
            {(file.name, file.sha256) for file in stored.files},
            {
                (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in folder.iterdir()
                if path.name != "bundle.json"
            },
        )
        library = Path(stored.made_with.executable)
        self.assertEqual(
            stored.made_with.sha256, hashlib.sha256(library.read_bytes()).hexdigest()
        )
        self.assertRegex(stored.made_with.version_line, r"^samtools \S+ \(pysam \S+\)$")
        self.assertEqual(
            reference_index_store.load_bundle(cache, stored.bundle_digest), stored
        )


class CommandLineTests(unittest.TestCase):
    ACCESSION = "GCF_000000001.1"
    BRC = f"/cvmfs/brc.galaxyproject.org/data/genomes/{ACCESSION}/"
    DATACACHE = f"http://datacache.galaxyproject.org/brc/data/genomes/{ACCESSION}/"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "cache"
        (self.cache / "catalogues").mkdir(parents=True)
        (self.cache / "catalogues" / "brc-bwa_mem_index.loc").write_text(
            _loc_row(
                self.ACCESSION,
                self.BRC + f"bwa_mem_index/v1/{self.ACCESSION}/{self.ACCESSION}.fa",
            )
            + "\n"
        )
        (self.cache / "catalogues" / "brc-fasta_indexes.loc").write_text(
            _loc_row(self.ACCESSION, self.BRC + f"seq/{self.ACCESSION}.fa") + "\n"
        )
        self.bundle = self.cache / "bundles" / "galaxy" / "brc" / self.ACCESSION
        self.bundle.mkdir(parents=True)

    def run_main(self, *argv, **effects):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([*argv, "--cache", str(self.cache)], **effects)
        return code, out.getvalue()

    def fetch_ncbi(self, md5):
        folder = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/001/"
            f"{self.ACCESSION}_R1/"
        )
        fasta = io.StringIO()
        for name, description, sequence in NCBI:
            fasta.write(f">{name} {description}\n{sequence}\n")
        packed = gzip.compress(fasta.getvalue().encode(), mtime=0)
        opener = FakeOpener(
            {
                folder + "md5checksums.txt": (
                    f"{md5 or hashlib.md5(packed).hexdigest()}  "
                    f"./{self.ACCESSION}_R1_genomic.fna.gz\n"
                ).encode(),
                folder + f"{self.ACCESSION}_R1_genomic.fna.gz": packed,
            }
        )
        return self.run_main("fetch-ncbi", self.ACCESSION, "R1", opener=opener)

    def test_find_lists_from_the_saved_catalogues_and_saves_the_list(self):
        Path(self.bundle / f"{self.ACCESSION}.fa.fai").write_text(
            _fai(_galaxy(), width=50)
        )

        code, printed = self.run_main("find", self.ACCESSION)

        self.assertEqual(code, 0)
        listed = json.loads(printed)
        self.assertEqual(
            [
                (c["candidate_id"], c["naming_style"], c["contig_count"])
                for c in listed["candidates"]
            ],
            [(f"galaxy:brc:{self.ACCESSION}", "ucsc", 3)],
        )
        saved = self.cache / "candidates" / f"{self.ACCESSION}.json"
        self.assertEqual(json.loads(saved.read_text()), listed)

    def test_a_malformed_accession_is_refused_before_any_contact(self):
        opener = FakeOpener({})
        for argv in (("find", "sacCer3"), ("fetch-ncbi", "GCF_146045.2", "R64")):
            with self.subTest(argv=argv):
                code, printed = self.run_main(*argv, opener=opener)

                self.assertEqual(code, 2)
                self.assertEqual(json.loads(printed)["refused"], "invalid_accession")
        self.assertEqual(opener.requested, [])

    def test_fit_refuses_a_malformed_ann_layout_by_name(self):
        self.fetch_ncbi(md5=None)
        fasta = _write_bundle(self.bundle / f"{self.ACCESSION}.fa", _galaxy())
        ann = Path(f"{fasta}.ann")
        ann.write_text(ann.read_text().replace("\n0 133 0\n", "\n0 133\n", 1))
        self.run_main("find", self.ACCESSION)

        code, printed = self.run_main(
            "fit", self.ACCESSION, f"galaxy:brc:{self.ACCESSION}"
        )

        self.assertEqual(code, 2)
        self.assertEqual(
            json.loads(printed),
            {"refused": "ann_structure_invalid", "detail": str(ann)},
        )

    def test_a_real_commands_redirect_off_the_source_hosts_is_refused_unfollowed(
        self,
    ):
        folder = f"/genomes/all/GCF/000/000/001/{self.ACCESSION}_R1/"
        base, seen = _serve(
            self,
            {
                folder + "md5checksums.txt": (
                    302,
                    {"Location": "http://other.test/md5checksums.txt"},
                    b"",
                )
            },
        )

        code, printed = self.run_main(
            "fetch-ncbi", self.ACCESSION, "R1", opener=_local_opener(base)
        )

        self.assertEqual(code, 2)
        self.assertEqual(
            json.loads(printed),
            {"refused": "rejected", "detail": "The URL is outside the source hosts."},
        )
        self.assertEqual(seen, [folder + "md5checksums.txt"])
        self.assertEqual(list((self.cache / "ncbi" / self.ACCESSION).iterdir()), [])

    def galaxy_opener(self, contigs):
        """Datacache as the BRC repository lays it out: the index under
        `bwa_mem_index/`, the FASTA and its `.fai` under `seq/` (fasta_indexes.loc)."""
        staged = self.cache.parent / "staged" / f"{self.ACCESSION}.fa"
        staged.parent.mkdir()
        _write_bundle(staged, contigs)
        index = (
            self.DATACACHE + f"bwa_mem_index/v1/{self.ACCESSION}/{self.ACCESSION}.fa"
        )
        seq = self.DATACACHE + f"seq/{self.ACCESSION}.fa"
        bodies = {
            seq: staged.read_bytes(),
            seq + ".fai": Path(f"{staged}.fai").read_bytes(),
        }
        for suffix in (".amb", ".ann", ".bwt", ".pac", ".sa"):
            bodies[index + suffix] = Path(f"{staged}{suffix}").read_bytes()
        return FakeOpener(bodies)

    def test_fetch_bundle_reads_the_index_and_the_fasta_from_their_galaxy_paths(self):
        self.fetch_ncbi(md5=None)
        self.run_main("find", self.ACCESSION)
        opener = self.galaxy_opener(_galaxy())

        code, _ = self.run_main(
            "fetch-bundle",
            self.ACCESSION,
            f"galaxy:brc:{self.ACCESSION}",
            opener=opener,
        )
        rerun, _ = self.run_main(
            "fetch-bundle",
            self.ACCESSION,
            f"galaxy:brc:{self.ACCESSION}",
            opener=opener,
        )
        fitted, printed = self.run_main(
            "fit", self.ACCESSION, f"galaxy:brc:{self.ACCESSION}"
        )

        self.assertEqual((code, rerun, fitted), (0, 0, 0))
        self.assertEqual(len(opener.requested), 7)
        self.assertEqual(
            sorted(path.name for path in self.bundle.iterdir()),
            sorted(
                f"{self.ACCESSION}.fa{suffix}{receipt}"
                for suffix in ("", ".fai", ".amb", ".ann", ".bwt", ".pac", ".sa")
                for receipt in ("", ".fetched.json")
            ),
        )
        self.assertEqual(json.loads(printed)["verdict"], "fits_after_rename")
        self.assertEqual(len(json.loads(printed)["sources"]), 9)

    def test_find_leaves_out_usegalaxy_rows(self):
        (self.cache / "catalogues" / "usegalaxy-bwa_mem_indexes.json").write_text(
            json.dumps(
                {
                    "name": "bwa_mem_indexes",
                    "model_class": "TabularToolDataTable",
                    "columns": ["value", "dbkey", "name", "path"],
                    "fields": [
                        [
                            self.ACCESSION,
                            self.ACCESSION,
                            "Synthetic",
                            f"{self.ACCESSION}.fa",
                        ],
                        [
                            "GCF_000000001.2",
                            "GCF_000000001.2",
                            "Newer",
                            "GCF_000000001.2.fa",
                        ],
                    ],
                }
            )
        )

        _, printed = self.run_main("find", self.ACCESSION)

        # usegalaxy.org's paths are not datacache paths: the product rule lists none.
        self.assertEqual(
            [
                (c["candidate_id"], c["catalogue"], c["location"])
                for c in json.loads(printed)["candidates"]
            ],
            [
                (
                    f"galaxy:brc:{self.ACCESSION}",
                    "brc-bwa_mem_index.loc",
                    self.DATACACHE
                    + f"bwa_mem_index/v1/{self.ACCESSION}/{self.ACCESSION}.fa",
                ),
            ],
        )
        opener = FakeOpener({})
        code, refused = self.run_main(
            "fetch-bundle",
            self.ACCESSION,
            "galaxy:usegalaxy:GCF_000000001.2",
            opener=opener,
        )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(refused)["refused"], "not_a_listed_candidate")
        self.assertEqual(opener.requested, [])

    def test_fetch_ncbi_checks_the_published_md5_and_unpacks(self):
        code, _ = self.fetch_ncbi(md5=None)

        self.assertEqual(code, 0)
        plain = (
            self.cache / "ncbi" / self.ACCESSION / f"{self.ACCESSION}_R1_genomic.fna"
        )
        self.assertTrue(plain.read_text().startswith(">NC_000001.1 "))
        folder = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/001/"
            f"{self.ACCESSION}_R1/"
        )
        self.assertEqual(
            [
                receipt.url
                for receipt in ReferenceIndexCache(self.cache).receipts_in(plain.parent)
            ],
            [
                f"{folder}{self.ACCESSION}_R1_genomic.fna.gz",
                f"{folder}md5checksums.txt",
            ],
        )

    def test_fetch_ncbi_checks_the_md5_of_the_whole_file_after_range_parts(self):
        fasta = "".join(f">{n} {d}\n{s}\n" for n, d, s in NCBI)
        packed = gzip.compress(fasta.encode(), mtime=0)
        folder = f"/genomes/all/GCF/000/000/001/{self.ACCESSION}_R1/"
        listing = (
            f"{hashlib.md5(packed).hexdigest()}  ./{self.ACCESSION}_R1_genomic.fna.gz\n"
        )
        base, ranges = _serve_file(
            self,
            packed,
            cuts={1: 20},
            path=folder + f"{self.ACCESSION}_R1_genomic.fna.gz",
            others={folder + "md5checksums.txt": listing.encode()},
        )
        code, printed = self.run_main(
            "fetch-ncbi", self.ACCESSION, "R1", opener=_local_opener(base)
        )

        self.assertEqual(code, 0, printed)
        # The part asked for is main's part size; the server's answer ends at the file.
        self.assertEqual(ranges, [None, None, f"bytes=20-{20 + 32 * 1024 * 1024 - 1}"])
        plain = (
            self.cache / "ncbi" / self.ACCESSION / f"{self.ACCESSION}_R1_genomic.fna"
        )
        self.assertEqual(plain.read_text(), fasta)

    def test_fetch_bundle_refuses_a_fasta_shorter_than_its_fai(self):
        """A FASTA whose size differs from the one its `.fai` gives is refused before
        the fit, even when the server declared that short length."""
        self.fetch_ncbi(md5=None)
        self.run_main("find", self.ACCESSION)
        opener = self.galaxy_opener(_galaxy())
        fasta = self.DATACACHE + f"seq/{self.ACCESSION}.fa"
        opener.bodies[fasta] = opener.bodies[fasta][:-7]

        code, printed = self.run_main(
            "fetch-bundle",
            self.ACCESSION,
            f"galaxy:brc:{self.ACCESSION}",
            opener=opener,
        )

        self.assertEqual(code, 2)
        self.assertEqual(json.loads(printed)["refused"], "fasta_size_differs_from_fai")

    def test_fetch_bundle_refuses_a_malformed_fai_by_name_before_any_fit(self):
        self.fetch_ncbi(md5=None)
        self.run_main("find", self.ACCESSION)
        opener = self.galaxy_opener(_galaxy())
        fai = self.DATACACHE + f"seq/{self.ACCESSION}.fa.fai"
        opener.bodies[fai] = b"chrI\t133\t6\t50\n"

        code, printed = self.run_main(
            "fetch-bundle",
            self.ACCESSION,
            f"galaxy:brc:{self.ACCESSION}",
            opener=opener,
        )

        self.assertEqual(
            (code, json.loads(printed)["refused"]), (2, "fai_structure_invalid")
        )
        self.assertFalse((self.cache / "fit").exists())

    def test_a_complete_fasta_without_a_final_line_end_is_fetched_and_fits(self):
        self.fetch_ncbi(md5=None)
        self.run_main("find", self.ACCESSION)
        opener = self.galaxy_opener(_galaxy())
        fasta = self.DATACACHE + f"seq/{self.ACCESSION}.fa"
        opener.bodies[fasta] = opener.bodies[fasta].rstrip(b"\n")

        code, printed = self.run_main(
            "fetch-bundle",
            self.ACCESSION,
            f"galaxy:brc:{self.ACCESSION}",
            opener=opener,
        )
        fitted, report = self.run_main(
            "fit", self.ACCESSION, f"galaxy:brc:{self.ACCESSION}"
        )

        self.assertEqual((code, fitted), (0, 0), printed)
        self.assertEqual(json.loads(report)["verdict"], "fits_after_rename")

    def test_the_commands_time_out_a_stalled_socket_operation_after_15_seconds(self):
        """The commands' GETs leave through the product's source transport, whose
        timeout is per socket operation, not per answer: a long download fails only
        when it stalls (owner, #417 Q6: 15 s)."""
        folder = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/001/"
            f"{self.ACCESSION}_R1/"
        )
        opener = FakeOpener({folder + "md5checksums.txt": TimeoutError("timed out")})

        code, printed = self.run_main("fetch-ncbi", self.ACCESSION, "R1", opener=opener)

        self.assertEqual(opener.timeouts, [15.0])
        self.assertEqual((code, json.loads(printed)["refused"]), (2, "FetchFailed"))
        self.assertEqual(list((self.cache / "ncbi" / self.ACCESSION).iterdir()), [])

    def test_fetch_ncbi_refuses_a_download_whose_md5_differs(self):
        code, printed = self.fetch_ncbi(md5="0" * 32)

        self.assertEqual(code, 2)
        self.assertIn("md5_mismatch", printed)
        folder = self.cache / "ncbi" / self.ACCESSION
        self.assertFalse((folder / f"{self.ACCESSION}_R1_genomic.fna").exists())

    def test_fit_saves_the_report_and_marks_the_candidate(self):
        self.fetch_ncbi(md5=None)
        _write_bundle(self.bundle / f"{self.ACCESSION}.fa", _galaxy())
        self.run_main("find", self.ACCESSION)

        code, printed = self.run_main(
            "fit", self.ACCESSION, f"galaxy:brc:{self.ACCESSION}"
        )

        self.assertEqual(code, 0)
        printed = json.loads(printed)
        self.assertEqual(printed["verdict"], "fits_after_rename")
        folder = (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/001/"
            f"{self.ACCESSION}_R1/"
        )
        kept = self.cache / "ncbi" / self.ACCESSION
        stem = f"{self.ACCESSION}_R1_genomic.fna.gz"
        self.assertEqual(
            [
                (source["url"], source["destination"], source["sha256"])
                for source in printed["sources"]
            ],
            [
                (
                    folder + stem,
                    str(kept / stem),
                    hashlib.sha256((kept / stem).read_bytes()).hexdigest(),
                ),
                (
                    folder + "md5checksums.txt",
                    str(kept / "md5checksums.txt"),
                    hashlib.sha256(
                        (kept / "md5checksums.txt").read_bytes()
                    ).hexdigest(),
                ),
            ],
        )
        (candidate,) = (
            ReferenceIndexCache(self.cache).load_candidates(self.ACCESSION).candidates
        )
        self.assertTrue(candidate.auto_pick)
        (report,) = (self.cache / "fit" / self.ACCESSION).iterdir()
        self.assertEqual(
            candidate.fit.report_sha256, hashlib.sha256(report.read_bytes()).hexdigest()
        )

        shutil.rmtree(self.cache / "fit")
        self.run_main("find", self.ACCESSION)
        (refound,) = (
            ReferenceIndexCache(self.cache).load_candidates(self.ACCESSION).candidates
        )
        self.assertEqual((refound.auto_pick, refound.fit), (True, candidate.fit))


class CacheTests(FitFixture, unittest.TestCase):
    def test_records_are_saved_and_read_back_typed(self):
        _write_bundle(self.candidate, _galaxy())
        report = self.check()
        cache = ReferenceIndexCache(self.root / "cache")
        listed = find("GCF_000000001.1", *_brc("GCF_000000001.1"))

        fit_path = cache.save_fit(report)
        cache.save_candidates(attach_fit(listed, report))

        self.assertEqual(cache.load_fit(fit_path), report)
        (candidate,) = cache.load_candidates("GCF_000000001.1").candidates
        self.assertTrue(candidate.auto_pick)
        self.assertEqual(
            candidate.fit.report_sha256,
            hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        )


DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2"


def _report(
    accession,
    name,
    *,
    status="current",
    category=None,
    released,
    tax_id=99,
    organism="Synthetica exempli",
):
    """One NCBI Datasets assembly report, in the fields the product adapter reads."""
    info = {
        "assembly_name": name,
        "assembly_status": status,
        "assembly_level": "Chromosome",
        "release_date": released,
    }
    if category:
        info["refseq_category"] = category
    return {
        "accession": accession,
        "current_accession": accession,
        "organism": {"organism_name": organism, "tax_id": tax_id},
        "assembly_info": info,
    }


def _reports(*reports):
    return json.dumps({"reports": list(reports), "total_count": len(reports)}).encode()


class DatasetsTests(unittest.TestCase):
    def test_an_unsolicited_206_to_a_metadata_get_is_refused_as_an_http_error(self):
        """A GET sent without Range that comes back 206 is not a whole response, even
        with a body of its declared length: the product adapter refuses it."""
        body = _reports(_report("GCF_000000001.1", "Syn1.0", released="2020-01-01"))
        sent = []

        def opener(request, timeout):
            sent.append((request.full_url, request.get_header("Range")))
            response = io.BytesIO(body)
            response.status = 206
            response.headers = {"Content-Length": str(len(body))}
            return response

        ncbi = Datasets(UrllibNcbiTransport(opener=opener))

        with self.assertRaises(ReferenceAdapterError) as refused:
            ncbi.describe("GCF_000000001.1")

        self.assertEqual(refused.exception.failure.code, "http_error")
        self.assertIn("HTTP 206", refused.exception.detail)
        self.assertEqual(
            sent,
            [(f"{DATASETS}/genome/accession/GCF_000000001.1/dataset_report", None)],
        )
        self.assertEqual(ncbi.contact_count, 1)


class AskTests(unittest.TestCase):
    """`ask <accession>` against a synthetic species: NCBI Datasets reports, Galaxy's
    catalogues (saved in the cache, as `catalogues` leaves them), Galaxy's datacache
    files and NCBI's genomes FTP, all served by a fake opener."""

    REQUESTED = "GCF_000000001.1"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "cache"
        (self.cache / "catalogues").mkdir(parents=True)
        self.staged = Path(directory.name) / "staged"
        self.bodies = {}
        self.bwa_rows, self.fasta_rows, self.usegalaxy_rows = [], [], []
        self.vgp_rows = []

    def ncbi(self, *reports, species, tax_id=99):
        """Datasets: each report described by accession, and the species listing. As
        NCBI answered on 2026-10-14 (`{}`, 2 bytes, for `GCF_000003025.6`), the plain
        accession report lists current versions only; a suppressed or replaced assembly
        is described only with `filters.assembly_version=all_assemblies`."""
        for report in reports:
            url = f"{DATASETS}/genome/accession/{report['accession']}/dataset_report"
            current = report["assembly_info"]["assembly_status"] == "current"
            self.bodies[url] = _reports(report) if current else b"{}"
            self.bodies[url + "?filters.assembly_version=all_assemblies"] = _reports(
                report
            )
        self.bodies[
            f"{DATASETS}/genome/taxon/{tax_id}/dataset_report?page_size=100"
        ] = _reports(*species)

    def ncbi_fasta(self, accession, name, contigs):
        """NCBI's genomes FTP folder: the gzipped FASTA and its md5checksums.txt."""
        number = accession[4:13]
        folder = (
            f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{accession[:3]}/"
            f"{number[:3]}/{number[3:6]}/{number[6:]}/{accession}_{name}/"
        )
        fasta = "".join(f">{n} {d}\n{s}\n" for n, d, s in contigs)
        packed = gzip.compress(fasta.encode(), mtime=0)
        self.bodies[folder + f"{accession}_{name}_genomic.fna.gz"] = packed
        self.bodies[folder + "md5checksums.txt"] = (
            f"{hashlib.md5(packed).hexdigest()}  ./{accession}_{name}_genomic.fna.gz\n"
        ).encode()

    def galaxy(self, accession, contigs):
        """A BRC row for the accession and its files on datacache: the index under
        `bwa_mem_index/`, the FASTA and `.fai` under `seq/` (fasta_indexes.loc)."""
        brc = f"/cvmfs/brc.galaxyproject.org/data/genomes/{accession}/"
        datacache = f"http://datacache.galaxyproject.org/brc/data/genomes/{accession}/"
        self.bwa_rows.append(
            _loc_row(accession, brc + f"bwa_mem_index/v1/{accession}/{accession}.fa")
        )
        self.fasta_rows.append(_loc_row(accession, brc + f"seq/{accession}.fa"))
        staged = self.staged / accession / f"{accession}.fa"
        staged.parent.mkdir(parents=True)
        _write_bundle(staged, contigs)
        served = {f"{datacache}seq/{accession}.fa{x}": x for x in ("", ".fai")}
        served.update(
            {
                f"{datacache}bwa_mem_index/v1/{accession}/{accession}.fa{x}": x
                for x in (".amb", ".ann", ".bwt", ".pac", ".sa")
            }
        )
        for url, suffix in served.items():
            self.bodies[url] = Path(f"{staged}{suffix}").read_bytes()
        return staged

    def ask(self, accession, opener, catalogues=True):
        for name, rows in (
            ()
            if not catalogues
            else (
                ("brc-bwa_mem_index.loc", self.bwa_rows),
                ("brc-fasta_indexes.loc", self.fasta_rows),
                ("vgp-bwa_mem_index.loc", self.vgp_rows),
            )
        ):
            (self.cache / "catalogues" / name).write_text("\n".join(rows) + "\n")
        if catalogues:  # `catalogues` saves every file, empty or not
            (self.cache / "catalogues" / "usegalaxy-bwa_mem_indexes.json").write_text(
                json.dumps(
                    {
                        "columns": ["value", "dbkey", "name", "path"],
                        "fields": self.usegalaxy_rows,
                    }
                )
            )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["ask", accession, "--cache", str(self.cache)], opener=opener)
        return code, json.loads(out.getvalue())

    def test_an_exact_prebuilt_matching_every_contig_is_picked_and_stored(self):
        self.ncbi(
            _report(self.REQUESTED, "Syn1.0", released="2020-01-01"),
            species=[_report(self.REQUESTED, "Syn1.0", released="2020-01-01")],
        )
        self.ncbi_fasta(self.REQUESTED, "Syn1.0", NCBI)
        staged = self.galaxy(self.REQUESTED, [(n, "", s) for n, _, s in NCBI])
        opener = FakeOpener(self.bodies)

        code, answer = self.ask(self.REQUESTED, opener)

        self.assertEqual(code, 0)
        self.assertEqual(answer["outcome"], "picked")
        choice = answer["choice"]
        self.assertEqual(
            (
                choice["decision"],
                choice["option"]["action"],
                choice["option"]["candidate_id"],
                choice["option"]["fit"]["sequence"],
                choice["requested"]["assembly_status"],
            ),
            (
                "auto_pick",
                "use_prebuilt",
                f"galaxy:brc:{self.REQUESTED}",
                "exact_sequence",
                "current",
            ),
        )
        fetched = {
            role: hashlib.sha256(Path(f"{staged}{suffix}").read_bytes()).hexdigest()
            for role, suffix in (
                ("genomic_fasta", ""),
                ("bwa_amb", ".amb"),
                ("bwa_ann", ".ann"),
                ("bwa_bwt", ".bwt"),
                ("bwa_pac", ".pac"),
                ("bwa_sa", ".sa"),
            )
        }
        digest = hashlib.sha256(canonical_json(fetched).encode()).hexdigest()
        self.assertEqual(answer["bundle"]["bundle_digest"], digest)
        self.assertEqual(choice["bundle_digest"], digest)
        stored = self.cache / "store" / "sha256" / digest
        self.assertEqual(
            sorted(path.name for path in stored.iterdir()),
            [
                "bundle.json",
                "reference.dict",
                "reference.fa",
                "reference.fa.amb",
                "reference.fa.ann",
                "reference.fa.bwt",
                "reference.fa.fai",
                "reference.fa.pac",
                "reference.fa.sa",
            ],
        )
        self.assertEqual((stored / "reference.fa").read_bytes(), staged.read_bytes())
        self.assertEqual(
            (stored / "reference.fa.fai").read_bytes(),
            Path(f"{staged}.fai").read_bytes(),
        )
        self.assertEqual(
            json.loads((self.cache / "choices" / f"{self.REQUESTED}.json").read_text()),
            choice,
        )

    def test_a_second_ask_is_a_cache_hit_with_no_contact(self):
        self.ncbi(
            _report(self.REQUESTED, "Syn1.0", released="2020-01-01"),
            species=[_report(self.REQUESTED, "Syn1.0", released="2020-01-01")],
        )
        self.ncbi_fasta(self.REQUESTED, "Syn1.0", NCBI)
        self.galaxy(self.REQUESTED, [(n, "", s) for n, _, s in NCBI])
        opener = FakeOpener(self.bodies)
        _, first = self.ask(self.REQUESTED, opener)
        sent = list(opener.requested)

        code, second = self.ask(self.REQUESTED, opener)

        self.assertEqual(code, 0)
        self.assertEqual((first["outcome"], first["contacts"]), ("picked", 11))
        self.assertEqual((second["outcome"], second["contacts"]), ("cache_hit", 0))
        self.assertEqual(
            (second["choice"], second["bundle"]), (first["choice"], first["bundle"])
        )
        self.assertEqual(opener.requested, sent)

    def test_one_changed_contig_gives_a_card_naming_it_and_stores_nothing(self):
        self.ncbi(
            _report(self.REQUESTED, "Syn1.0", released="2020-01-01"),
            species=[_report(self.REQUESTED, "Syn1.0", released="2020-01-01")],
        )
        self.ncbi_fasta(self.REQUESTED, "Syn1.0", NCBI)
        changed = [(n, "", s) for n, _, s in NCBI]
        changed[1] = ("NC_000002.1", "", CHROMOSOME_II[:-1] + "C")
        self.galaxy(self.REQUESTED, changed)

        code, answer = self.ask(self.REQUESTED, FakeOpener(self.bodies))

        self.assertEqual(code, 0)
        self.assertEqual(answer["outcome"], "card")
        options = {o["option_id"]: o for o in answer["card"]["options"]}
        self.assertEqual(
            sorted(options),
            [f"build:{self.REQUESTED}", f"prebuilt:galaxy:brc:{self.REQUESTED}"],
        )
        prebuilt = options[f"prebuilt:galaxy:brc:{self.REQUESTED}"]["fit"]
        self.assertEqual(prebuilt["sequence"], "refused")
        self.assertEqual(
            prebuilt["reasons"],
            ["sequence_differs: NC_000002.1 (candidate NC_000002.1)"],
        )
        self.assertEqual(
            options[f"build:{self.REQUESTED}"]["fit"]["sequence"], "requested_assembly"
        )
        self.assertEqual(answer["card"]["default_option_id"], f"build:{self.REQUESTED}")
        self.assertIsNone(answer["choice"])
        self.assertFalse((self.cache / "store").exists())
        self.assertFalse((self.cache / "choices").exists())

    OLD, NEW = "GCF_000000002.1", "GCF_000000003.1"

    def newer_world(self):
        """A suppressed requested assembly whose species now has a newer reference
        genome; Galaxy holds a prebuilt of the newer one only."""
        old = _report(
            self.OLD, "SynOld 1.0", status="suppressed", released="2015-01-01"
        )
        new = _report(
            self.NEW, "SynNew2", category="reference genome", released="2021-01-01"
        )
        # NCBI marks both twins of the pig T2T as the reference genome, GenBank first.
        genbank_twin = _report(
            "GCA_000000003.2",
            "SynNew2",
            category="reference genome",
            released="2021-01-01",
        )
        self.ncbi(old, new, species=[genbank_twin, new])
        return self.galaxy(self.NEW, [(n, "", s) for n, _, s in NCBI])

    def test_a_suppressed_request_is_offered_the_newer_prebuilt_with_the_snv_chain(
        self,
    ):
        staged = self.newer_world()
        opener = FakeOpener(self.bodies)

        code, answer = self.ask(self.OLD, opener)

        self.assertEqual(code, 0)
        self.assertEqual(answer["outcome"], "card")
        card = answer["card"]
        self.assertEqual(
            (card["requested"]["assembly_status"], card["species"]["reference_genome"]),
            ("suppressed", self.NEW),
        )
        self.assertEqual(
            [
                (o["option_id"], o["action"], o["fit"]["sequence"])
                for o in card["options"]
            ],
            [
                (f"build:{self.OLD}", "build_requested", "requested_assembly"),
                (
                    f"prebuilt:galaxy:brc:{self.NEW}",
                    "use_newer_prebuilt",
                    "newer_assembly",
                ),
                (f"build:{self.NEW}", "build_newer", "newer_assembly"),
            ],
        )
        newer = card["options"][1]
        self.assertTrue(any("#332" in step for step in newer["fit"]["snv_chain"]))
        self.assertTrue(any("#331" in step for step in newer["fit"]["snv_chain"]))
        bases = sum(len(sequence) for _, _, sequence in NCBI)
        self.assertEqual(
            (newer["cost"]["basis"], newer["cost"]["download_bytes"]),
            ("estimated_from_fai", staged.stat().st_size + round(1.75 * bases)),
        )
        self.assertEqual(
            sorted(url.rsplit("/", 1)[1] for url in opener.requested),
            [
                f"{self.NEW}.fa.fai",
                "dataset_report",
                "dataset_report?filters.assembly_version=all_assemblies",
                "dataset_report?page_size=100",
            ],
        )

    def take(self, accession, option_id, *extra, opener):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(
                ["take", accession, option_id, *extra, "--cache", str(self.cache)],
                opener=opener,
            )
        return code, json.loads(out.getvalue())

    def test_taking_the_newer_prebuilt_fetches_fits_stores_and_hands_over_the_snvs(
        self,
    ):
        """`take` asks again before it fetches: NCBI's metadata is read anew (no
        Datasets body is replayed from the cache since #417), the card's `.fai` is not."""
        staged = self.newer_world()
        self.ncbi_fasta(self.NEW, "SynNew2", NCBI)
        opener = FakeOpener(self.bodies)
        self.ask(self.OLD, opener)

        code, taken = self.take(
            self.OLD,
            f"prebuilt:galaxy:brc:{self.NEW}",
            "--snv",
            "SV1=NC_000001.1:12:G>T",
            opener=opener,
        )
        sent = list(opener.requested)
        _, again = self.ask(self.OLD, opener)

        self.assertEqual(code, 0)
        self.assertEqual((taken["outcome"], taken["contacts"]), ("taken", 11))
        choice = taken["choice"]
        self.assertEqual(
            (choice["decision"], choice["option"]["assembly_accession"]),
            ("taken", self.NEW),
        )
        self.assertEqual(
            choice["snvs_to_move"],
            [
                {
                    "label": "SV1",
                    "assembly_accession": self.OLD,
                    "contig": "NC_000001.1",
                    "position": 12,
                    "ref": "G",
                    "alt": "T",
                }
            ],
        )
        # Each distinct card is kept: the ask's, and the one take's own ask made (its
        # NCBI sources observed anew); the choice names a kept card that offered it.
        cards = {
            hashlib.sha256(card.read_bytes()).hexdigest(): json.loads(card.read_text())
            for card in (self.cache / "cards" / self.OLD).iterdir()
        }
        self.assertIn(choice["card_sha256"], cards)
        self.assertIn(
            choice["option"],
            cards[choice["card_sha256"]]["options"],
        )
        bundle = taken["bundle"]
        (fit,) = (self.cache / "fit" / self.NEW).iterdir()
        self.assertEqual(
            choice["fit_report_sha256"], hashlib.sha256(fit.read_bytes()).hexdigest()
        )
        self.assertEqual(json.loads(fit.read_text())["verdict"], "fits")
        stored = self.cache / "store" / "sha256" / bundle["bundle_digest"]
        self.assertEqual((stored / "reference.fa").read_bytes(), staged.read_bytes())
        self.assertEqual(
            [url.rsplit("/", 1)[1] for url in sent[4:]],
            [
                "dataset_report",
                "dataset_report?filters.assembly_version=all_assemblies",
                "dataset_report?page_size=100",
                f"{self.NEW}.fa",
                f"{self.NEW}.fa.amb",
                f"{self.NEW}.fa.ann",
                f"{self.NEW}.fa.bwt",
                f"{self.NEW}.fa.pac",
                f"{self.NEW}.fa.sa",
                "md5checksums.txt",
                f"{self.NEW}_SynNew2_genomic.fna.gz",
            ],
        )
        self.assertEqual((again["outcome"], again["contacts"]), ("cache_hit", 0))
        self.assertEqual(again["choice"], choice)
        self.assertEqual(opener.requested, sent)

    def test_a_pig_request_gets_a_card_with_the_older_ucsc_prebuilt_and_fetches_nothing(
        self,
    ):
        pig = {"tax_id": 9823, "organism": "Sus scrofa"}
        requested = _report(
            "GCF_000003025.6",
            "Sscrofa11.1",
            status="suppressed",
            released="2017-02-07",
            **pig,
        )
        newer = _report(
            "GCF_054392235.1",
            "Sscrofa_HxYL_maternal_Yorkshire_Landrace_v2.0",
            category="reference genome",
            released="2026-01-28",
            **pig,
        )
        older = _report(
            "GCF_000003025.5",
            "Sscrofa10.2",
            status="replaced",
            released="2011-09-07",
            **pig,
        )
        self.ncbi(requested, newer, older, species=[newer], tax_id=9823)
        self.usegalaxy_rows = [
            ["susScr3", "susScr3", "Pig (Sus scrofa): susScr3", "susScr3.fa"],
            ["susScr2", "susScr2", "Pig (Sus scrofa): susScr2", "susScr2.fa"],
            [
                "GCA_030704935.1",
                "GCA_030704935.1",
                "Sus scrofa (GCA_030704935.1_ASM3070493v1)",
                "GCA_030704935.1.fa",
            ],
        ]
        opener = FakeOpener(self.bodies)

        code, answer = self.ask("GCF_000003025.6", opener)

        self.assertEqual(code, 0)
        card = answer["card"]
        self.assertEqual(
            [
                (o["option_id"], o["action"], o["fit"]["sequence"], o["cost"]["work"])
                for o in card["options"]
            ],
            [
                (
                    "build:GCF_000003025.6",
                    "build_requested",
                    "requested_assembly",
                    "cpu_build",
                ),
                ("build:GCF_054392235.1", "build_newer", "newer_assembly", "cpu_build"),
                (
                    "prebuilt:galaxy:usegalaxy:susScr3",
                    "use_older_prebuilt",
                    "older_assembly",
                    "not_fetchable",
                ),
            ],
        )
        self.assertEqual(
            (
                card["options"][2]["assembly_accession"],
                card["options"][2]["assembly_name"],
            ),
            ("GCF_000003025.5", "Sscrofa10.2"),
        )
        self.assertEqual(
            [entry.split(":")[2] for entry in card["not_offered"]],
            ["susScr2", "GCA_030704935.1"],
        )
        self.assertEqual(
            card["not_offered"][0],
            "galaxy:usegalaxy:susScr2: another assembly of Sus scrofa; no UCSC_ALIASES "
            "row for "
            "susScr2",
        )
        self.assertNotIn("UCSC_ALIASES", card["not_offered"][1])
        self.assertEqual(
            [url.split("/datasets/v2/")[1] for url in opener.requested],
            [
                "genome/accession/GCF_000003025.6/dataset_report",
                "genome/accession/GCF_000003025.6/dataset_report"
                "?filters.assembly_version=all_assemblies",
                "genome/taxon/9823/dataset_report?page_size=100",
                "genome/accession/GCF_000003025.5/dataset_report",
                "genome/accession/GCF_000003025.5/dataset_report"
                "?filters.assembly_version=all_assemblies",
            ],
        )
        self.assertFalse((self.cache / "bundles").exists())

    def test_a_truncated_listing_without_a_reference_finds_it_among_galaxy_rows(self):
        old = _report(
            self.OLD, "SynOld 1.0", status="suppressed", released="2015-01-01"
        )
        new = _report(
            self.NEW, "SynNew2", category="reference genome", released="2021-01-01"
        )
        other = _report("GCA_000000009.1", "SynOther", released="2022-01-01")
        self.ncbi(old, new, other, species=[other])
        listing = f"{DATASETS}/genome/taxon/99/dataset_report?page_size=100"
        self.bodies[listing] = json.dumps(
            {"reports": [other], "total_count": 250, "next_page_token": "x"}
        ).encode()
        for accession in ("GCA_000000009.1", self.NEW):
            self.galaxy(accession, [(n, "", s) for n, _, s in NCBI])
        self.bwa_rows = [
            row.replace(f"\t{a}\t/cvmfs", f"\tSynthetica exempli ({a})\t/cvmfs")
            for row, a in zip(self.bwa_rows, ("GCA_000000009.1", self.NEW), strict=True)
        ]
        opener = FakeOpener(self.bodies)

        code, answer = self.ask(self.OLD, opener)

        self.assertEqual(code, 0)
        species = answer["card"]["species"]
        self.assertEqual(
            (
                species["truncated"],
                species["reference_genome"],
                species["reference_basis"],
            ),
            (True, self.NEW, "described_galaxy_row"),
        )
        self.assertEqual(
            [
                (a["assembly_accession"], a["assembly_name"], a["assembly_status"])
                for a in species["assemblies"]
            ],
            [("GCA_000000009.1", "SynOther", "current")],
        )
        self.assertEqual(
            [o["option_id"] for o in answer["card"]["options"]],
            [
                f"build:{self.OLD}",
                f"prebuilt:galaxy:brc:{self.NEW}",
                f"build:{self.NEW}",
            ],
        )
        self.assertEqual(
            [
                url.split("/datasets/v2/")[-1]
                for url in opener.requested
                if "datasets" in url
            ],
            [
                f"genome/accession/{self.OLD}/dataset_report",
                f"genome/accession/{self.OLD}/dataset_report"
                "?filters.assembly_version=all_assemblies",
                "genome/taxon/99/dataset_report?page_size=100",
                "genome/accession/GCA_000000009.1/dataset_report",
                f"genome/accession/{self.NEW}/dataset_report",
            ],
        )

    def exact_world(self):
        self.ncbi(
            _report(self.REQUESTED, "Syn1.0", released="2020-01-01"),
            species=[_report(self.REQUESTED, "Syn1.0", released="2020-01-01")],
        )
        self.ncbi_fasta(self.REQUESTED, "Syn1.0", NCBI)
        return self.galaxy(self.REQUESTED, [(n, "", s) for n, _, s in NCBI])

    def test_an_assembly_that_does_not_state_the_fit_stops_and_stores_nothing(self):
        """The product's assembly reads its roles back against the fit (#365): here a
        `.dict` whose first M5 differs, which the reader finds against the FASTA."""
        self.exact_world()
        made = reference_index_bundle.pysam.dict

        def wrong_dict(fasta):
            return made(fasta).replace("M5:", "M5:0", 1)

        with mock.patch.object(reference_index_bundle.pysam, "dict", wrong_dict):
            code, answer = self.ask(self.REQUESTED, FakeOpener(self.bodies))

        self.assertEqual(code, 2)
        self.assertEqual(answer["refused"], "assembly_refused")
        self.assertIn(f"galaxy:brc:{self.REQUESTED}", answer["detail"])
        self.assertEqual(list((self.cache / "store" / "sha256").iterdir()), [])
        self.assertFalse((self.cache / "choices").exists())

    def test_take_refuses_an_option_it_cannot_fetch_before_any_download(self):
        """`take` asks again first, so each refusal follows NCBI's metadata read anew
        (no Datasets body is replayed from the cache since #417), and no file."""
        self.newer_world()
        opener = FakeOpener(self.bodies)
        self.ask(self.OLD, opener)
        sent = len(opener.requested)
        reasked = [
            f"genome/accession/{self.OLD}/dataset_report",
            f"genome/accession/{self.OLD}/dataset_report"
            "?filters.assembly_version=all_assemblies",
            "genome/taxon/99/dataset_report?page_size=100",
        ]

        for option_id, reason in (
            ("prebuilt:galaxy:brc:GCF_000000099.1", "not_an_option"),
            (f"build:{self.NEW}", "not_a_fetchable_prebuilt"),
        ):
            with self.subTest(option_id=option_id):
                code, answer = self.take(self.OLD, option_id, opener=opener)

                self.assertEqual((code, answer["refused"]), (2, reason))
                self.assertEqual(
                    [url.split("/datasets/v2/")[-1] for url in opener.requested[sent:]],
                    reasked,
                )
                sent = len(opener.requested)
        self.assertFalse((self.cache / "choices").exists())
        self.assertEqual(
            sorted(p.name for p in (self.cache / "bundles").glob("*/*/*/*")),
            [f"{self.NEW}.fa.fai", f"{self.NEW}.fa.fai.fetched.json"],
        )

    def test_a_bundle_already_fitted_in_the_cache_is_picked_with_only_ncbi_metadata(
        self,
    ):
        """#319's yeast case: the bundle and NCBI FASTA are cached and the candidate
        already carries its fit, so the pick sends only the two Datasets requests."""
        staged = self.exact_world()
        opener = FakeOpener(self.bodies)
        for argv in (
            ["fetch-ncbi", self.REQUESTED, "Syn1.0"],
            ["find", self.REQUESTED],
            ["fetch-bundle", self.REQUESTED, f"galaxy:brc:{self.REQUESTED}"],
        ):
            self.run_tool(argv, opener)
        self.run_tool(["fit", self.REQUESTED, f"galaxy:brc:{self.REQUESTED}"], opener)
        sent = len(opener.requested)

        code, answer = self.ask(self.REQUESTED, opener)

        self.assertEqual(code, 0)
        self.assertEqual((answer["outcome"], answer["contacts"]), ("picked", 2))
        self.assertEqual(
            [url.split("/datasets/v2/")[1] for url in opener.requested[sent:]],
            [
                f"genome/accession/{self.REQUESTED}/dataset_report",
                "genome/taxon/99/dataset_report?page_size=100",
            ],
        )
        stored = self.cache / "store" / "sha256" / answer["bundle"]["bundle_digest"]
        self.assertEqual((stored / "reference.fa").read_bytes(), staged.read_bytes())

    def run_tool(self, argv, opener):
        for name, rows in (
            ("brc-bwa_mem_index.loc", self.bwa_rows),
            ("brc-fasta_indexes.loc", self.fasta_rows),
        ):
            (self.cache / "catalogues" / name).write_text("\n".join(rows) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([*argv, "--cache", str(self.cache)], opener=opener)
        self.assertEqual(code, 0, argv)

    def test_a_card_from_a_cache_without_catalogues_names_their_fetch(self):
        # With no catalogues `ask` sees no Galaxy prebuilt and offers a build.
        self.ncbi(
            _report(self.REQUESTED, "Syn1.0", released="2020-01-01"),
            species=[_report(self.REQUESTED, "Syn1.0", released="2020-01-01")],
        )

        code, answer = self.ask(
            self.REQUESTED, FakeOpener(self.bodies), catalogues=False
        )

        self.assertEqual(code, 0)
        self.assertEqual(answer["outcome"], "card")
        [missing] = answer["card"]["not_offered"]
        self.assertIn("Galaxy's catalogues are not in the cache", missing)
        self.assertIn(
            "python -m tools.reference_index catalogues, then ask again", missing
        )

    def test_an_exact_row_with_no_download_path_gives_a_card_and_fetches_nothing(self):
        self.ncbi(
            _report(self.REQUESTED, "Syn1.0", released="2020-01-01"),
            species=[_report(self.REQUESTED, "Syn1.0", released="2020-01-01")],
        )
        self.usegalaxy_rows = [
            [self.REQUESTED, self.REQUESTED, "Synthetica", f"{self.REQUESTED}.fa"]
        ]
        opener = FakeOpener(self.bodies)

        code, answer = self.ask(self.REQUESTED, opener)

        self.assertEqual(code, 0)
        card = answer["card"]
        self.assertEqual(
            [o["option_id"] for o in card["options"]], [f"build:{self.REQUESTED}"]
        )
        self.assertEqual(
            card["not_offered"],
            [
                f"galaxy:usegalaxy:{self.REQUESTED}: not fetchable here (Galaxy gives no "
                "datacache "
                "path for its index and FASTA), so its fit cannot be checked"
            ],
        )
        self.assertTrue(all("/datasets/v2/" in url for url in opener.requested))
        self.assertFalse((self.cache / "bundles").exists())
        self.assertFalse((self.cache / "store").exists())

    def test_a_newer_prebuilt_with_no_fasta_path_is_explained_and_builds_offered(self):
        self.newer_world()
        vgp = f"/cvmfs/vgp.galaxyproject.org/data/genomes/{self.NEW}/"
        self.bwa_rows, self.fasta_rows = [], []
        self.vgp_rows = [
            _loc_row(self.NEW, vgp + f"bwa_mem_index/v1/{self.NEW}/{self.NEW}.fa")
        ]
        opener = FakeOpener(self.bodies)

        code, answer = self.ask(self.OLD, opener)

        self.assertEqual(code, 0)
        card = answer["card"]
        self.assertEqual(
            [o["option_id"] for o in card["options"]],
            [f"build:{self.OLD}", f"build:{self.NEW}"],
        )
        self.assertEqual(
            card["not_offered"],
            [
                f"galaxy:vgp:{self.NEW}: not fetchable here (brc-fasta_indexes.loc names no "
                "FASTA for it), so its fit cannot be checked"
            ],
        )
        self.assertTrue(all("/datasets/v2/" in url for url in opener.requested))

    def test_identical_content_is_stored_once_and_each_choice_keeps_its_own_fit(self):
        """Two accessions whose Galaxy bundles are byte-identical: one stored bundle,
        no second `.fai`/`.dict`, and each choice names its own candidate and fit."""
        twin = "GCF_000000004.1"
        for accession, name in ((self.REQUESTED, "Syn1.0"), (twin, "Syn1.1")):
            report = _report(accession, name, released="2020-01-01")
            self.ncbi(report, species=[report])
            self.ncbi_fasta(accession, name, NCBI)
            self.galaxy(accession, [(n, "", s) for n, _, s in NCBI])
        opener = FakeOpener(self.bodies)

        with mock.patch.object(
            reference_index_store,
            "assemble_prepared_roles",
            wraps=reference_index_store.assemble_prepared_roles,
        ) as assembled:
            _, first = self.ask(self.REQUESTED, opener)
            _, second = self.ask(twin, opener)
            _, again = self.ask(twin, opener)

        self.assertEqual(first["bundle"], second["bundle"])
        self.assertEqual(assembled.call_count, 1)
        self.assertNotIn("assembly_accession", second["bundle"])
        fits = {
            accession: next((self.cache / "fit" / accession).iterdir())
            for accession in (self.REQUESTED, twin)
        }
        for answer, accession in (
            (first, self.REQUESTED),
            (second, twin),
            (again, twin),
        ):
            choice = answer["choice"]
            self.assertEqual(
                (
                    choice["option"]["assembly_accession"],
                    choice["option"]["candidate_id"],
                    choice["fit_report_sha256"],
                ),
                (
                    accession,
                    f"galaxy:brc:{accession}",
                    hashlib.sha256(fits[accession].read_bytes()).hexdigest(),
                ),
            )

    def test_an_all_versions_answer_beyond_ten_reports_is_read_whole(self):
        """No cap on NCBI's answer (#471 Q5 a; ADR 0049, #375 C12): an any-version
        report listing eleven reports is copied whole, and the requested accession's
        own report among them is read, where the bound once refused the answer as
        ``invalid_response``. Synthetic reports."""
        report = _report(
            self.OLD, "SynOld 1.0", status="suppressed", released="2015-01-01"
        )
        self.ncbi(report, species=[])
        url = f"{DATASETS}/genome/accession/{self.OLD}/dataset_report"
        others = [
            _report(
                f"GCF_000000002.{version}",
                f"SynOld 1.{version}",
                status="suppressed",
                released="2015-01-01",
            )
            for version in range(2, 12)
        ]
        self.bodies[url + "?filters.assembly_version=all_assemblies"] = _reports(
            report, *others
        )

        code, answer = self.ask(self.OLD, FakeOpener(self.bodies))

        self.assertNotEqual(answer.get("refused"), "invalid_response", answer)
        self.assertNotEqual(code, 2, answer)


class _HttpAnswer:
    """One scripted HTTP 200 answer for the variant source's transport."""

    def __init__(self, body):
        self.status, self.body, self.headers = 200, body, email.message.Message()

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ScriptedSources:
    """#331's `VariantSources` over the product's transports, with scripted answers by
    URL; `sent` lists the requests. `answer` asks the source the developer names (#469):
    these tests name dbSNP for a human assembly, EVA for any other."""

    def __init__(self, answers):
        self.sent = []

        def opener(request, timeout):
            self.sent.append(request.full_url)
            return _HttpAnswer(answers[request.full_url])

        sources = VariantSources.public(opener)
        self.answer = lambda claim: sources.answer(
            claim,
            "dbsnp" if claim.assembly_accession.startswith("GCF_000001405") else "eva",
        )


EVA_RS = (
    "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants/900001"
)
EVA_ALIAS = "https://www.ebi.ac.uk/eva/webservices/contig-alias/v1/chromosomes/genbank"
SYNTHETIC_B_FOLDER = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/900/000/002/"
SYNTHETIC_B_STEM = "GCF_900000002.1_SyntheticB"


def _synthetic_b_answers():
    """B's NCBI folder and assembly report (made up, in NCBI's layout): GenBank twin
    `GCA_900000002.1`, whose `CM900001.1` to `CM900004.1` are `same`, `flipped`,
    `changed` and `twice`."""
    rows = [
        f"{name}\tassembled-molecule\t{k}\tChromosome\t{genbank}\t=\t{name}"
        f"\tPrimary Assembly\t2000\tna"
        for k, name, genbank in (
            (1, "same", "CM900001.1"),
            (2, "flipped", "CM900002.1"),
            (3, "changed", "CM900003.1"),
            (4, "twice", "CM900004.1"),
        )
    ]
    report = "\r\n".join(
        [
            "# Assembly name:  SyntheticB",
            "# Taxid:          1",
            "# GenBank assembly accession: GCA_900000002.1",
            "# RefSeq assembly accession: GCF_900000002.1",
            "#",
            "# Sequence-Name\tSequence-Role\tAssigned-Molecule"
            "\tAssigned-Molecule-Location/Type\tGenBank-Accn\tRelationship\tRefSeq-Accn"
            "\tAssembly-Unit\tSequence-Length\tUCSC-style-name",
            *rows,
        ]
    )
    # EVA's contig aliases for B's sequences (the product EVA lookup names them), and
    # for another assembly's sequence.
    aliases = {
        f"{EVA_ALIAS}/{genbank}": json.dumps(
            {
                "_embedded": {
                    "chromosomeEntities": [
                        {
                            "insdcAccession": genbank,
                            "refseq": refseq,
                            "assembly": {
                                "insdcAccession": gca,
                                "refseq": gcf,
                                "name": name,
                            },
                        }
                    ]
                }
            }
        ).encode()
        for genbank, refseq, gca, gcf, name in (
            *(
                (
                    f"CM90000{k}.1",
                    sequence,
                    "GCA_900000002.1",
                    "GCF_900000002.1",
                    "SyntheticB",
                )
                for k, sequence in enumerate(("same", "flipped", "changed", "twice"), 1)
            ),
            (
                "CM999999.1",
                "NC_999999.1",
                "GCA_900000099.1",
                "GCF_900000099.1",
                "Other",
            ),
        )
    }
    # NCBI's report of B, for the INSDC twin an EVA lookup reads.
    report_json = json.dumps(
        {
            "reports": [
                {
                    "accession": "GCF_900000002.1",
                    "paired_accession": "GCA_900000002.1",
                    "assembly_info": {
                        "assembly_name": "SyntheticB",
                        "assembly_status": "current",
                    },
                }
            ]
        }
    ).encode()
    return aliases | {
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_900000002.1"
        "/dataset_report": report_json,
        SYNTHETIC_B_FOLDER: f'<a href="{SYNTHETIC_B_STEM}/">{SYNTHETIC_B_STEM}/</a>'.encode(),
        f"{SYNTHETIC_B_FOLDER}{SYNTHETIC_B_STEM}/{SYNTHETIC_B_STEM}_assembly_report.txt": (
            report + "\r\n"
        ).encode(),
    }


def _eva_cluster(*placements, variant_type="SNV"):
    """EVA's clustered variant rs900001 (made up) at `(assembly, contig, start)` each."""
    return json.dumps(
        [
            {
                "accession": 900001,
                "version": 1,
                "data": {
                    "assemblyAccession": assembly,
                    "taxonomyAccession": 1,
                    "contig": contig,
                    "start": start,
                    "type": variant_type,
                },
            }
            for assembly, contig, start in placements
        ]
    ).encode()


def _eva_submitted(ref, alt, *places):
    return json.dumps(
        [
            {
                "accession": 900100 + index,
                "version": 1,
                "data": {
                    "referenceSequenceAccession": assembly,
                    "contig": contig,
                    "start": start,
                    "referenceAllele": ref,
                    "alternateAllele": alt,
                    "clusteredVariantAccession": 900001,
                    "projectAccession": "PRJEB900001",
                },
            }
            for index, (assembly, contig, start) in enumerate(places)
        ]
    ).encode()


@unittest.skipUnless(INSTALLED, "needs #319's bwa and samtools install")
class MoveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.pair = SyntheticPair(Path(directory.name))

    def move(self, *snvs, **options):
        return move_snvs(
            snvs,
            source_fasta=self.pair.source,
            target_assembly="GCF_900000002.1",
            target_fasta=self.pair.target,
            tools=INSTALL,
            queries=Path(self.enterContext(tempfile.TemporaryDirectory())),
            **options,
        )

    def test_flanks_found_once_on_the_same_strand_give_a_unique_place(self):
        snv = self.pair.snv(0, "same")

        [moved] = self.move(snv).moves

        self.assertEqual(
            (moved.result, moved.method, moved.snv, moved.target_assembly),
            ("unique", "flank_realignment", snv, "GCF_900000002.1"),
        )
        place = moved.place
        self.assertEqual(
            (place.contig, place.position, place.strand, place.ref, place.mapq),
            ("same", 899, "+", self.pair.refs[0], 60),
        )
        self.assertEqual(
            (moved.target_ref, moved.target_alt), (self.pair.refs[0], self.pair.alts[0])
        )

    def test_flanks_found_on_the_other_strand_flip_the_snv(self):
        snv = self.pair.snv(1, "flipped")
        complement = {"A": "T", "C": "G", "G": "C", "T": "A"}

        [moved] = self.move(snv).moves

        self.assertEqual(
            (moved.result, moved.method), ("strand_flipped", "flank_realignment")
        )
        place = moved.place
        self.assertEqual(
            (place.contig, place.position, place.strand, place.ref),
            ("flipped", 900, "-", complement[self.pair.refs[1]]),
        )
        self.assertEqual(
            (moved.target_ref, moved.target_alt),
            (complement[self.pair.refs[1]], complement[self.pair.alts[1]]),
        )

    def test_a_base_on_b_other_than_the_ref_is_a_changed_ref(self):
        snv = self.pair.snv(2, "changed")

        [moved] = self.move(snv).moves

        self.assertEqual(
            (moved.result, moved.method), ("ref_changed", "flank_realignment")
        )
        place = moved.place
        self.assertEqual(
            (place.contig, place.position, place.strand, place.ref),
            ("changed", 901, "+", self.pair.alts[2]),
        )
        self.assertEqual((moved.target_ref, moved.target_alt), (None, None))

    def test_flanks_found_at_more_than_one_place_are_not_placed(self):
        snv = self.pair.snv(3, "twice")

        [moved] = self.move(snv).moves

        self.assertEqual(
            (moved.result, moved.method, moved.place),
            ("several_places", "flank_realignment", None),
        )
        self.assertEqual(
            {(p.contig, p.position, p.strand, p.ref) for p in moved.places},
            {
                ("twice", 901, "+", self.pair.refs[3]),
                ("twice", 2301, "+", self.pair.refs[3]),
            },
        )

    def test_flanks_b_does_not_hold_are_not_found(self):
        snv = self.pair.snv(4, "absent")

        [moved] = self.move(snv).moves

        self.assertEqual(
            (moved.result, moved.method, moved.place, moved.places),
            ("not_found", "flank_realignment", None, ()),
        )

    def test_flanks_aligned_only_in_part_are_not_found(self):
        snv = self.pair.snv(5, "half")

        [moved] = self.move(snv).moves

        self.assertEqual(
            (moved.result, moved.method, moved.place),
            ("not_found", "flank_realignment", None),
        )
        [partial] = moved.places
        self.assertEqual((partial.contig, partial.position), ("half", 901))
        self.assertLess(partial.score, 0.9 * 401)

    def test_an_snv_base_b_lacks_is_not_found(self):
        snv = self.pair.snv(6, "lacks")

        [moved] = self.move(snv).moves

        self.assertEqual(
            (moved.result, moved.method, moved.place, moved.places),
            ("not_found", "flank_realignment", None, ()),
        )
        self.assertIn("lacks", moved.reason)

    def sources(self, answers):
        return _ScriptedSources(answers)

    def test_an_rs_id_placed_once_on_b_is_a_database_placement(self):
        snv = self.pair.snv(0, "same", rsid="rs900001")
        on_b = ("GCA_900000002.1", "CM900001.1", 899)
        sources = self.sources(
            _synthetic_b_answers()
            | {
                EVA_RS: _eva_cluster(on_b),
                f"{EVA_RS}/submitted": _eva_submitted(
                    self.pair.refs[0], self.pair.alts[0], on_b
                ),
            }
        )

        run = self.move(snv, database=sources.answer)

        [moved] = run.moves
        self.assertEqual((moved.result, moved.method), ("unique", "database_placement"))
        self.assertEqual(
            (moved.place.contig, moved.place.position, moved.place.strand),
            ("same", 899, "+"),
        )
        self.assertEqual(moved.place.ref, self.pair.refs[0])
        self.assertEqual(
            (moved.database.route, moved.database.eva.status), ("eva", "resolved")
        )
        self.assertEqual(moved.database.claim.assembly_accession, "GCF_900000002.1")
        self.assertIsNone(run.query_sha256)

    def test_an_rs_id_the_database_does_not_place_on_b_is_realigned(self):
        snv = self.pair.snv(0, "same", rsid="rs900001")
        sources = self.sources(
            _synthetic_b_answers()
            | {EVA_RS: _eva_cluster(("GCA_900000099.1", "CM999999.1", 4242))}
        )

        [moved] = self.move(snv, database=sources.answer).moves

        self.assertEqual((moved.result, moved.method), ("unique", "flank_realignment"))
        self.assertEqual((moved.place.contig, moved.place.position), ("same", 899))
        self.assertEqual(moved.database.eva.status, "incompatible")
        self.assertIn("eva: incompatible", moved.reason)

    def test_an_rs_id_the_database_places_twice_on_b_is_not_placed(self):
        snv = self.pair.snv(0, "same", rsid="rs900001")
        on_b = (
            ("GCA_900000002.1", "CM900001.1", 899),
            ("GCA_900000002.1", "CM900004.1", 901),
        )
        sources = self.sources(
            _synthetic_b_answers()
            | {
                EVA_RS: _eva_cluster(*on_b),
                f"{EVA_RS}/submitted": _eva_submitted(
                    self.pair.refs[0], self.pair.alts[0], *on_b
                ),
            }
        )

        run = self.move(snv, database=sources.answer)

        [moved] = run.moves
        self.assertEqual(
            (moved.result, moved.method, moved.place),
            ("several_places", "database_placement", None),
        )
        self.assertEqual(
            {(p.contig, p.position) for p in moved.places},
            {("same", 899), ("twice", 901)},
        )
        self.assertIsNone(run.query_sha256)

    def test_a_database_placement_with_swapped_alleles_is_realigned(self):
        # B holds A's ALT where A has its REF; EVA states B's REF>ALT, the SNV's swapped.
        snv = self.pair.snv(2, "changed", rsid="rs900001")
        on_b = ("GCA_900000002.1", "CM900003.1", 901)
        sources = self.sources(
            _synthetic_b_answers()
            | {
                EVA_RS: _eva_cluster(on_b),
                f"{EVA_RS}/submitted": _eva_submitted(
                    self.pair.alts[2], self.pair.refs[2], on_b
                ),
            }
        )

        [moved] = self.move(snv, database=sources.answer).moves

        self.assertEqual(
            (moved.result, moved.method), ("ref_changed", "flank_realignment")
        )
        self.assertEqual(
            (moved.place.contig, moved.place.position, moved.place.ref),
            ("changed", 901, self.pair.alts[2]),
        )
        self.assertEqual(moved.database.eva.status, "resolved")
        self.assertIn("do not support", moved.reason)

    def test_a_database_placement_of_another_variant_is_realigned(self):
        # SNV 0 is G>A; EVA's record at its place on B is another variant.
        snv = self.pair.snv(0, "same", rsid="rs900001")
        on_b = ("GCA_900000002.1", "CM900001.1", 899)
        cases = {
            "another ALT": (_eva_cluster(on_b), ("G", "C")),
            "a deletion": (_eva_cluster(on_b, variant_type="DEL"), ("GT", "")),
        }
        self.assertEqual((snv.ref, snv.alt), ("G", "A"))
        for name, (cluster, (ref, alt)) in cases.items():
            with self.subTest(name):
                sources = self.sources(
                    _synthetic_b_answers()
                    | {
                        EVA_RS: cluster,
                        f"{EVA_RS}/submitted": _eva_submitted(ref, alt, on_b),
                    }
                )

                [moved] = self.move(snv, database=sources.answer).moves

                self.assertEqual(
                    (moved.result, moved.method), ("unique", "flank_realignment")
                )
                self.assertEqual(
                    (moved.place.contig, moved.place.position), ("same", 899)
                )
                self.assertEqual(moved.database.eva.status, "resolved")
                self.assertIn("do not support", moved.reason)

    def test_a_database_placement_b_does_not_hold_is_realigned(self):
        # EVA says G>A two bases past the SNV, where B's FASTA has another base.
        snv = self.pair.snv(0, "same", rsid="rs900001")
        on_b = ("GCA_900000002.1", "CM900001.1", 901)
        b_base = INSTALL.bases(self.pair.target, "same", 901, 901)
        self.assertNotEqual(b_base, snv.ref)
        sources = self.sources(
            _synthetic_b_answers()
            | {
                EVA_RS: _eva_cluster(on_b),
                f"{EVA_RS}/submitted": _eva_submitted(snv.ref, snv.alt, on_b),
            }
        )

        [moved] = self.move(snv, database=sources.answer).moves

        self.assertEqual((moved.result, moved.method), ("unique", "flank_realignment"))
        self.assertEqual((moved.place.contig, moved.place.position), ("same", 899))
        self.assertIn(f"B's FASTA has {b_base} there", moved.reason)

    def test_a_database_placement_with_the_complemented_snv_flips_it(self):
        # Segment 1 is reverse-complemented on B: T>C on A is A>G on B at `flipped` 900.
        self.assertEqual(self.pair.refs[1], "T")
        snv = SnvToMove(
            label="flipped",
            assembly="synthetic A",
            contig="chrA",
            position=self.pair.positions[1],
            ref="T",
            alt="C",
            rsid="rs900001",
        )
        on_b = ("GCA_900000002.1", "CM900002.1", 900)
        sources = self.sources(
            _synthetic_b_answers()
            | {
                EVA_RS: _eva_cluster(on_b),
                f"{EVA_RS}/submitted": _eva_submitted("A", "G", on_b),
            }
        )

        run = self.move(snv, database=sources.answer)

        [moved] = run.moves
        self.assertEqual(
            (moved.result, moved.method), ("strand_flipped", "database_placement")
        )
        self.assertEqual(
            (
                moved.place.contig,
                moved.place.position,
                moved.place.strand,
                moved.place.ref,
            ),
            ("flipped", 900, "-", "A"),
        )
        self.assertEqual((moved.target_ref, moved.target_alt), ("A", "G"))
        self.assertIsNone(run.query_sha256)

    def test_a_strand_ambiguous_snv_is_realigned_when_the_database_has_its_complement(
        self,
    ):
        # T>A complemented is A>T: the same alleles as T>A with B's REF swapped.
        snv = self.pair.snv(1, "flipped", rsid="rs900001")
        self.assertEqual((snv.ref, snv.alt), ("T", "A"))
        on_b = ("GCA_900000002.1", "CM900002.1", 900)
        sources = self.sources(
            _synthetic_b_answers()
            | {
                EVA_RS: _eva_cluster(on_b),
                f"{EVA_RS}/submitted": _eva_submitted("A", "T", on_b),
            }
        )

        [moved] = self.move(snv, database=sources.answer).moves

        self.assertEqual(
            (moved.result, moved.method), ("strand_flipped", "flank_realignment")
        )
        self.assertEqual((moved.place.contig, moved.place.position), ("flipped", 900))
        self.assertIn("strand", moved.reason)

    def human_dbsnp_move(self, *alts):
        """SNV 0 with rs429358 moved to a made-up stretch named as GRCh38's chr19 (A's REF
        at 101, indexed), with dbSNP placing rs429358 there with the given ALTs."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        human = directory / "human.fa"
        write_fasta(
            human,
            [("NC_000019.10", "", "C" * 100 + self.pair.refs[0] + "C" * 99)],
            width=60,
        )
        INSTALL.run("bwa", "index", str(human))
        body = {
            "refsnp_id": "429358",
            "primary_snapshot_data": {
                "placements_with_allele": [
                    {
                        "seq_id": "NC_000019.10",
                        "is_ptlp": True,
                        "placement_annot": {
                            "seq_id_traits_by_assembly": [
                                {
                                    "assembly_accession": "GCF_000001405.40",
                                    "assembly_name": "GRCh38.p14",
                                }
                            ]
                        },
                        "alleles": [
                            {
                                "allele": {
                                    "spdi": {
                                        "seq_id": "NC_000019.10",
                                        "position": 100,
                                        "deleted_sequence": self.pair.refs[0],
                                        "inserted_sequence": base,
                                    }
                                }
                            }
                            for base in (self.pair.refs[0], *alts)
                        ],
                    }
                ]
            },
        }
        sources = self.sources(
            {
                "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358": json.dumps(
                    body
                ).encode()
            }
        )
        return move_snvs(
            (self.pair.snv(0, "same", rsid="rs429358"),),
            source_fasta=self.pair.source,
            target_assembly="GCF_000001405.40",
            target_fasta=human,
            tools=INSTALL,
            queries=directory / "queries",
            database=sources.answer,
        )

    def test_an_rs_id_on_a_human_b_is_placed_by_dbsnp(self):
        run = self.human_dbsnp_move(self.pair.alts[0])

        [moved] = run.moves
        self.assertEqual((moved.result, moved.method), ("unique", "database_placement"))
        self.assertEqual(
            (moved.place.contig, moved.place.position, moved.place.ref),
            ("NC_000019.10", 101, self.pair.refs[0]),
        )
        self.assertEqual(
            (moved.database.route, moved.database.dbsnp.status), ("dbsnp", "resolved")
        )

    def test_a_dbsnp_placement_of_another_alt_is_realigned(self):
        self.assertEqual((self.pair.refs[0], self.pair.alts[0]), ("G", "A"))

        [moved] = self.human_dbsnp_move("C").moves

        self.assertEqual(
            (moved.result, moved.method), ("not_found", "flank_realignment")
        )
        self.assertEqual(moved.database.dbsnp.status, "resolved")
        self.assertIn("do not support", moved.reason)

    def test_an_snv_whose_ref_is_not_on_a_is_refused_before_any_alignment(self):
        snv = SnvToMove(
            label="swapped",
            assembly="synthetic A",
            contig="chrA",
            position=self.pair.positions[0],
            ref=self.pair.alts[0],
            alt=self.pair.refs[0],
        )
        tools = mock.Mock(wraps=INSTALL)

        with self.assertRaises(MoveRefused) as refused:
            move_snvs(
                (snv,),
                source_fasta=self.pair.source,
                target_assembly="GCF_900000002.1",
                target_fasta=self.pair.target,
                tools=tools,
                queries=Path(self.enterContext(tempfile.TemporaryDirectory())),
            )

        self.assertEqual(refused.exception.reason, "ref_not_on_source")
        self.assertIn(
            f"chrA:{self.pair.positions[0]} is {self.pair.refs[0]}",
            refused.exception.detail,
        )
        tools.mem.assert_not_called()

    def test_an_snv_on_a_sequence_a_lacks_is_refused(self):
        snv = SnvToMove(
            label="elsewhere",
            assembly="synthetic A",
            contig="chrZ",
            position=self.pair.positions[0],
            ref=self.pair.refs[0],
            alt=self.pair.alts[0],
        )

        with self.assertRaises(MoveRefused) as refused:
            self.move(snv)

        self.assertEqual(refused.exception.reason, "contig_not_on_source")
        self.assertIn("chrZ", refused.exception.detail)

    def test_an_snv_past_the_end_of_its_sequence_is_refused(self):
        length = len(self.pair.source.read_text().replace("\n", "")) - len(">chrA")
        snv = SnvToMove(
            label="past",
            assembly="synthetic A",
            contig="chrA",
            position=length + 1,
            ref="A",
            alt="C",
        )

        with self.assertRaises(MoveRefused) as refused:
            self.move(snv)

        self.assertEqual(refused.exception.reason, "position_not_on_source")
        self.assertIn(f"chrA has {length} bases", refused.exception.detail)


class SnvToMoveTests(unittest.TestCase):
    """The shared variant claim (#331) and #330's choice SNVs as SNVs to move."""

    def test_a_claimed_snv_keeps_its_assembly_position_alleles_and_rs_id(self):
        claim = VariantClaim(
            label="s16:c1",
            assembly_accession="GCF_000002285.3",
            rsid="rs852365981",
            contig="NC_006592.3",
            position=68587027,
            ref="A",
            alt="G",
        )

        snv = SnvToMove.of_claim(claim)

        self.assertEqual(
            (
                snv.label,
                snv.assembly,
                snv.contig,
                snv.position,
                snv.ref,
                snv.alt,
                snv.rsid,
            ),
            (
                "s16:c1",
                "GCF_000002285.3",
                "NC_006592.3",
                68587027,
                "A",
                "G",
                "rs852365981",
            ),
        )

    def test_a_claim_that_is_not_one_snv_at_a_position_is_refused(self):
        cases = {
            "only an SNV moves": VariantClaim(
                label="pig",
                assembly_accession="GCF_000003025.6",
                contig="NC_010454.4",
                position=3556401,
                ref="GGTGTCCCCCTCGC",
                alt="G",
            ),
            "only a claim with a position moves": VariantClaim(
                label="rs", assembly_accession="GCF_000003025.6", rsid="rs5334475172"
            ),
        }
        for reason, claim in cases.items():
            with self.subTest(reason), self.assertRaisesRegex(ValueError, reason):
                SnvToMove.of_claim(claim)

    def test_a_choice_snv_moves_from_the_requested_assembly(self):
        snv = SnvToMove.of_snv(
            SnvClaim(
                label="SV1",
                assembly_accession="GCF_000002315.4",
                contig="NC_006089.4",
                position=86770373,
                ref="A",
                alt="C",
            )
        )

        self.assertEqual(
            (
                snv.label,
                snv.assembly,
                snv.contig,
                snv.position,
                snv.ref,
                snv.alt,
                snv.rsid,
            ),
            ("SV1", "GCF_000002315.4", "NC_006089.4", 86770373, "A", "C", None),
        )


@unittest.skipUnless(INSTALLED, "needs #319's bwa and samtools install")
class MoveCommandTests(unittest.TestCase):
    """`move` over a cache laid out as #330 leaves it: the choice record, the requested
    assembly's NCBI FASTA, and the chosen bundle in the store."""

    SOURCE, TARGET = "GCF_900000001.1", "GCF_900000002.1"

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.pair = SyntheticPair(Path(directory.name))

    def setUp(self):
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory())) / "cache"
        self.variant_cache = self.cache.parent / "variant-sources"
        self.digest = hashlib.sha256(self.pair.target.read_bytes()).hexdigest()
        store = self.cache / "store" / "sha256" / self.digest
        store.mkdir(parents=True)
        for suffix in ("", ".amb", ".ann", ".bwt", ".pac", ".sa", ".fai"):
            shutil.copy(f"{self.pair.target}{suffix}", store / f"reference.fa{suffix}")
        source = self.cache / "ncbi" / self.SOURCE / f"{self.SOURCE}_SynA_genomic.fna"
        source.parent.mkdir(parents=True)
        shutil.copy(self.pair.source, source)

    def choose(self, *snvs):
        choice = choice_record(
            self.SOURCE,
            self.TARGET,
            self.digest,
            [
                {
                    "label": label,
                    "assembly_accession": self.SOURCE,
                    "contig": "chrA",
                    "position": self.pair.positions[k],
                    "ref": self.pair.refs[k],
                    "alt": self.pair.alts[k],
                }
                for k, label in snvs
            ],
        )
        path = self.cache / "choices" / f"{self.SOURCE}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(choice.model_dump(mode="json")))

    def run_main(self, *argv, opener=None):
        """`move` over this test's caches; the variant source's is its own, empty unless
        the test writes it."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(
                [
                    "move",
                    *argv,
                    "--cache",
                    str(self.cache),
                    "--variant-cache",
                    str(self.variant_cache),
                ],
                tools=INSTALL,
                **({"opener": opener} if opener else {}),
            )
        return code, json.loads(out.getvalue())

    def rsid_move(self, opener):
        """SNV 0 stated on A's FASTA with rs900001, to look up on B first."""
        snv = self.pair.snv(0, "S")
        return self.run_main(
            self.TARGET,
            "--source-assembly",
            "synthetic A",
            "--source-fasta",
            str(self.pair.source),
            "--snv",
            f"S=chrA:{snv.position}:{snv.ref}>{snv.alt}",
            "--rsid",
            "S=rs900001",
            # The synthetic B is no human assembly: the developer names EVA.
            "--source",
            "eva",
            opener=opener,
        )

    def test_an_rs_move_reads_no_ncbi_assembly_report(self):
        # The product's EVA lookup names B's sequences through EVA's contig
        # aliases, so a report that cannot be read is never asked for.
        self.choose()
        opener = FakeOpener(
            {
                SYNTHETIC_B_FOLDER: urllib.error.URLError("reset by peer"),
                EVA_RS: b"[]",
            }
        )

        code, record = self.rsid_move(opener)

        self.assertEqual(code, 0)
        self.assertEqual(opener.requested, [EVA_RS])
        [moved] = record["moves"]
        self.assertEqual(moved["database"]["eva"]["status"], "not_found")

    def test_moving_a_choices_snvs_records_each_result_and_keeps_the_alignment(self):
        self.choose((0, "SV1"), (1, "SV2"))

        code, record = self.run_main(self.TARGET, "--choice", self.SOURCE)

        self.assertEqual(code, 0)
        self.assertEqual(
            [
                (
                    m["snv"]["label"],
                    m["result"],
                    m["place"]["contig"],
                    m["place"]["position"],
                )
                for m in record["moves"]
            ],
            [("SV1", "unique", "same", 899), ("SV2", "strand_flipped", "flipped", 900)],
        )
        self.assertEqual(
            (record["source_assembly"], record["target_assembly"]),
            (self.SOURCE, self.TARGET),
        )
        self.assertEqual(record["target_bundle_digest"], self.digest)
        self.assertEqual(
            record["moves"][0]["snv"],
            {
                "label": "SV1",
                "assembly": self.SOURCE,
                "contig": "chrA",
                "position": self.pair.positions[0],
                "ref": self.pair.refs[0],
                "alt": self.pair.alts[0],
                "rsid": None,
            },
        )
        [saved] = (self.cache / "moves" / self.TARGET).glob("*.json")
        self.assertEqual(json.loads(saved.read_text()), record)
        sam = self.cache / "moves" / "alignments" / f"{record['alignment_sha256']}.sam"
        self.assertEqual(
            hashlib.sha256(sam.read_bytes()).hexdigest(), record["alignment_sha256"]
        )
        self.assertIn(b"\tsame\t", sam.read_bytes())

    def test_snvs_stated_on_a_given_fasta_move_to_the_stored_index_a_choice_names(self):
        self.choose()
        k = 2

        code, record = self.run_main(
            self.TARGET,
            "--source-assembly",
            "synthetic A",
            "--source-fasta",
            str(self.pair.source),
            "--snv",
            f"C4=chrA:{self.pair.positions[k]}:{self.pair.refs[k]}>{self.pair.alts[k]}",
        )

        self.assertEqual(code, 0)
        [moved] = record["moves"]
        self.assertEqual(
            (moved["snv"]["assembly"], moved["result"], moved["place"]["position"]),
            ("synthetic A", "ref_changed", 901),
        )
        self.assertEqual(record["source_fasta"]["path"], str(self.pair.source))
        self.assertEqual(record["target_bundle_digest"], self.digest)

    def test_a_target_no_choice_has_stored_is_refused(self):
        self.choose()

        code, refused = self.run_main(
            "GCF_900000009.1",
            "--source-assembly",
            "synthetic A",
            "--source-fasta",
            str(self.pair.source),
            "--snv",
            f"C4=chrA:{self.pair.positions[0]}:{self.pair.refs[0]}>{self.pair.alts[0]}",
        )

        self.assertEqual((code, refused["refused"]), (2, "no_stored_index"))
        self.assertFalse((self.cache / "moves").exists())

    def test_a_move_the_cache_cannot_serve_is_refused_before_any_record(self):
        self.choose((0, "SV1"))
        snv = (
            f"C4=chrA:{self.pair.positions[0]}:{self.pair.refs[0]}>{self.pair.alts[0]}"
        )
        cases = {
            "invalid_move": [self.TARGET, "--choice", self.SOURCE, "--snv", snv],
            "target_not_chosen": ["GCF_900000009.1", "--choice", self.SOURCE],
            "rsid_label_unknown": [
                self.TARGET,
                "--choice",
                self.SOURCE,
                "--rsid",
                "SV9=rs1",
            ],
            "invalid_rsid": [self.TARGET, "--choice", self.SOURCE, "--rsid", "SV1=1"],
            # The developer names the source an rs ID is asked of.
            "source_missing": [
                self.TARGET,
                "--choice",
                self.SOURCE,
                "--rsid",
                "SV1=rs1",
            ],
        }
        for reason, argv in cases.items():
            with self.subTest(reason):
                code, refused = self.run_main(*argv)

                self.assertEqual((code, refused["refused"]), (2, reason))
        shutil.rmtree(self.cache / "ncbi")
        with self.subTest("source_fasta_missing"):
            code, refused = self.run_main(self.TARGET, "--choice", self.SOURCE)

            self.assertEqual((code, refused["refused"]), (2, "source_fasta_missing"))
        self.assertFalse((self.cache / "moves").exists())

    def test_an_snv_with_an_rs_id_asks_the_variant_source_and_counts_its_contacts(
        self,
    ):
        self.choose()
        on_b = ("GCA_900000002.1", "CM900001.1", 899)
        answers = _synthetic_b_answers() | {
            EVA_RS: _eva_cluster(on_b),
            f"{EVA_RS}/submitted": _eva_submitted(
                self.pair.refs[0], self.pair.alts[0], on_b
            ),
        }
        sent = []

        def opener(request, timeout):
            sent.append(request.full_url)
            return _HttpAnswer(answers[request.full_url])

        variant_cache = self.variant_cache
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(
                [
                    "move",
                    self.TARGET,
                    "--source-assembly",
                    "synthetic A",
                    "--source-fasta",
                    str(self.pair.source),
                    "--snv",
                    f"S=chrA:{self.pair.positions[0]}:{self.pair.refs[0]}>{self.pair.alts[0]}",
                    "--rsid",
                    "S=rs900001",
                    "--source",
                    "eva",
                    "--cache",
                    str(self.cache),
                    "--variant-cache",
                    str(variant_cache),
                ],
                opener=opener,
                tools=INSTALL,
            )
        record = json.loads(out.getvalue())

        self.assertEqual(code, 0)
        [moved] = record["moves"]
        self.assertEqual(
            (moved["snv"]["rsid"], moved["method"], moved["result"]),
            ("rs900001", "database_placement", "unique"),
        )
        # EVA's clustered variant, NCBI's report of B for its INSDC twin, the contig
        # alias and the submitted variants.
        self.assertEqual(
            sent,
            [
                EVA_RS,
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                "GCF_900000002.1/dataset_report",
                f"{EVA_ALIAS}/CM900001.1",
                f"{EVA_RS}/submitted",
            ],
        )
        self.assertEqual(record["contact_count"], 4)
        self.assertFalse(variant_cache.exists())
        self.assertIsNone(record["alignment_sha256"])

    def test_the_same_move_twice_keeps_one_query_and_one_alignment(self):
        self.choose((0, "SV1"), (3, "SV2"))

        _, first = self.run_main(self.TARGET, "--choice", self.SOURCE)
        _, second = self.run_main(self.TARGET, "--choice", self.SOURCE)

        self.assertEqual(
            (first["query_sha256"], first["alignment_sha256"]),
            (second["query_sha256"], second["alignment_sha256"]),
        )
        self.assertEqual(len(list((self.cache / "moves" / "alignments").iterdir())), 1)
        query = self.cache / "moves" / "queries" / f"{first['query_sha256']}.fa"
        sam = self.cache / "moves" / "alignments" / f"{first['alignment_sha256']}.sam"
        self.assertIn(f"bwa mem {self.cache / 'store'}".encode(), sam.read_bytes())
        self.assertIn(str(query).encode(), sam.read_bytes())


EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id=NC_000099.1&rettype=fasta&retmode=text"
)


class FetchSequenceTests(unittest.TestCase):
    """`fetch-sequence <accession.version>`: one NCBI efetch FASTA GET, for an
    assembly NCBI has no assembly record of (#332: the sacCer1-era chromosome XI)."""

    def setUp(self):
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory())) / "cache"

    def fetch(self, accession, opener):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(
                ["fetch-sequence", accession, "--cache", str(self.cache)], opener=opener
            )
        return code, json.loads(out.getvalue())

    def test_a_sequence_is_fetched_as_fasta_with_its_receipt(self):
        body = (
            b">NC_000099.1 Synthetic chromosome XI, complete sequence\nACGTACGT\nGG\n"
        )
        opener = FakeOpener({EFETCH: body})

        code, fetched = self.fetch("NC_000099.1", opener)

        self.assertEqual(code, 0)
        path = self.cache / "ncbi" / "sequences" / "NC_000099.1.fa"
        self.assertEqual(path.read_bytes(), body)
        self.assertEqual(
            (fetched["path"], fetched["byte_count"], fetched["sha256"]),
            (str(path), len(body), hashlib.sha256(body).hexdigest()),
        )
        self.assertEqual(opener.requested, [EFETCH])
        [receipt] = ReferenceIndexCache(self.cache).receipts_in(path.parent)
        self.assertEqual(
            (receipt.url, receipt.destination, receipt.sha256),
            (EFETCH, str(path), hashlib.sha256(body).hexdigest()),
        )
        self.assertEqual(fetched["fetched"], receipt.model_dump(mode="json"))

    def test_an_answer_that_is_not_the_sequences_fasta_is_refused_and_not_kept(self):
        cases = {
            "another sequence": b">NC_000098.1 Synthetic chromosome X\nACGT\n",
            "an error page": b"<html><body>Error: rate limited</body></html>\n",
        }
        for name, body in cases.items():
            with self.subTest(name):
                self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))

                opener = FakeOpener({EFETCH: body})

                code, refused = self.fetch("NC_000099.1", opener)

                self.assertEqual((code, refused["refused"]), (2, "not_the_sequence"))
                # Neither the refused file nor its receipt is kept.
                self.assertEqual(
                    list((self.cache / "ncbi" / "sequences").iterdir()), []
                )
                self.assertEqual(opener.requested, [EFETCH])

    def test_an_accession_that_is_not_a_sequence_is_refused_before_any_request(self):
        opener = FakeOpener({})

        code, refused = self.fetch("GCF_000146045.2", opener)

        self.assertEqual((code, refused["refused"]), (2, "invalid_accession"))
        self.assertEqual(opener.requested, [])


if __name__ == "__main__":
    unittest.main()
