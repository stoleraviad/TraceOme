"""A source file read in Range parts through the one source transport: each part one
GET, a transient fault (a 5xx, a lost connection, a body that stops short) retried by
the bound failure policy, as object storage moves its parts (#363, #365 Q10)."""

from __future__ import annotations

import re
import unittest
from email.message import Message
from unittest.mock import MagicMock

from bio_neb.failure import ProviderError
from bio_neb.providers.source_transport import SourceTransport
from tools.reference_index.source_download import SourceDownload

URL = "http://datacache.galaxyproject.org/brc/data/ce11/bwa_mem/ce11.fa.bwt"
CONTENT = bytes(range(256)) * 40
"""10,240 bytes."""


def _response(status: int, body: bytes = b"", headers=None, fault=None):
    response = MagicMock(status=status)
    message = Message()
    for name, value in (headers or {}).items():
        message[name] = value
    response.headers = message
    response.__enter__.return_value = response
    remaining = [body]

    def read(size=-1):
        if fault is not None and not remaining[0]:
            raise fault
        chunk = remaining[0] if size < 0 else remaining[0][:size]
        remaining[0] = remaining[0][len(chunk) :]
        return chunk

    response.read.side_effect = read
    return response


def _part(start: int, end: int, content: bytes = CONTENT):
    return _response(
        206,
        content[start : end + 1],
        {"Content-Range": f"bytes {start}-{end}/{len(content)}"},
    )


class _Server:
    """Answers each GET from `answers` in order, or as a Range server of CONTENT."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.ranges: list[str | None] = []

    def __call__(self, request, timeout):
        del timeout
        span = request.get_header("Range")
        self.ranges.append(span)
        if self.answers:
            return self.answers.pop(0)
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", span).groups())
        return _part(start, min(end, len(CONTENT) - 1))


def _download(server: _Server, **kwargs) -> tuple[SourceDownload, list[float]]:
    slept: list[float] = []
    download = SourceDownload(
        SourceTransport(opener=server),
        URL,
        part_bytes=4096,
        sleep=slept.append,
        **kwargs,
    )
    return download, slept


def _read_all(download: SourceDownload, block: int = 1000) -> bytes:
    return b"".join(iter(lambda: download.read(block), b""))


class SourceDownloadTests(unittest.TestCase):
    def test_a_file_is_read_in_range_parts(self) -> None:
        server = _Server()
        download, slept = _download(server)
        self.assertEqual(_read_all(download), CONTENT)
        self.assertEqual(
            server.ranges,
            ["bytes=0-4095", "bytes=4096-8191", "bytes=8192-10239"],
        )
        self.assertEqual(download.total, len(CONTENT))
        self.assertEqual(slept, [])

    def test_read_without_a_size_reads_the_rest(self) -> None:
        download, _ = _download(_Server())
        self.assertEqual(download.read(10), CONTENT[:10])
        self.assertEqual(download.read(), CONTENT[10:])
        self.assertEqual(download.read(), b"")

    def test_a_partial_file_continues_from_its_size(self) -> None:
        server = _Server()
        download, _ = _download(server, start=5000)
        self.assertEqual(_read_all(download), CONTENT[5000:])
        self.assertEqual(server.ranges[0], "bytes=5000-9095")

    def test_a_part_that_stops_short_or_fails_is_asked_for_again(self) -> None:
        server = _Server(
            _part(0, 4095),
            # A body that ends before its Content-Range says.
            _response(
                206,
                CONTENT[4096:5000],
                {"Content-Range": f"bytes 4096-8191/{len(CONTENT)}"},
            ),
            _response(503),
            _response(
                206,
                CONTENT[4096:4100],
                {"Content-Range": f"bytes 4096-8191/{len(CONTENT)}"},
                fault=TimeoutError("stalled"),
            ),
        )
        download, slept = _download(server)
        # The default policy allows 3 attempts per part: the 503 and the stall are
        # the second and third, so the short body's part fails exhausted.
        with self.assertRaises(ProviderError) as raised:
            _read_all(download)
        self.assertEqual(raised.exception.failure_class, "transient")
        self.assertEqual(server.ranges[1:], ["bytes=4096-8191"] * 3)
        self.assertEqual(len(slept), 2)

    def test_a_retried_part_completes_the_file(self) -> None:
        server = _Server(_response(502))
        download, slept = _download(server)
        self.assertEqual(_read_all(download), CONTENT)
        self.assertEqual(server.ranges[:2], ["bytes=0-4095", "bytes=0-4095"])
        self.assertEqual(len(slept), 1)
        self.assertLessEqual(slept[0], 1.0)

    def test_an_answer_that_is_not_the_part_is_refused_without_a_retry(self) -> None:
        for name, answer in {
            "not found": _response(404, b"gone"),
            "redirect": _response(302, b"", {"Location": "https://elsewhere/"}),
            "range ignored": _response(200, CONTENT),
            "another range": _response(
                206, CONTENT[:4096], {"Content-Range": "bytes 1-4096/10240"}
            ),
        }.items():
            with self.subTest(name):
                server = _Server(answer)
                download, slept = _download(server)
                with self.assertRaises(ProviderError) as raised:
                    download.read(1)
                self.assertEqual(raised.exception.failure_class, "rejected")
                self.assertEqual(len(server.ranges), 1)
                self.assertEqual(slept, [])

    def test_a_total_that_changes_between_parts_is_refused(self) -> None:
        server = _Server(
            _part(0, 4095),
            _response(
                206, CONTENT[4096:8192], {"Content-Range": "bytes 4096-8191/99999"}
            ),
        )
        download, _ = _download(server)
        with self.assertRaises(ProviderError) as raised:
            _read_all(download)
        self.assertEqual(raised.exception.failure_class, "rejected")

    def test_a_host_outside_the_source_hosts_is_refused_before_sending(self) -> None:
        server = _Server()
        download = SourceDownload(
            SourceTransport(opener=server), "https://example.org/x.bwt"
        )
        with self.assertRaises(ProviderError) as raised:
            download.read(1)
        self.assertEqual(raised.exception.failure_class, "rejected")
        self.assertEqual(server.ranges, [])


if __name__ == "__main__":
    unittest.main()
