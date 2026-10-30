"""Galaxy's datacache view of the source transport (#365): each GET the source
answered is reported as a source answer (#458); a HEAD gives a file's size."""

from __future__ import annotations

import io
import unittest
import urllib.error
from email.message import Message
from unittest.mock import MagicMock

from bio_neb.agentic_journey.source_answers import collected_answers
from bio_neb.failure import ProviderError
from bio_neb.providers.galaxy_datacache import GalaxyDatacache

URL = "http://datacache.galaxyproject.org/brc/config/bwa_mem_index.loc"


def _answer(status=200, body=b"", headers=None):
    response = MagicMock(status=status)
    message = Message()
    for name, value in (headers or {}).items():
        message[name] = value
    response.headers = message
    response.__enter__.return_value = response
    response.read.side_effect = lambda size=-1: body if size < 0 else body[:size]
    return response


def _status(code, body=b""):
    def opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, code, "status", Message(), io.BytesIO(body)
        )

    return opener


class GalaxyDatacacheTests(unittest.TestCase):
    def test_a_get_is_the_answer_and_is_collected(self) -> None:
        galaxy = GalaxyDatacache(opener=lambda request, timeout: _answer(body=b"rows"))
        with collected_answers("describe_companion") as got:
            self.assertEqual(galaxy.get(URL), b"rows")
        self.assertEqual([(a.request_identity, a.body) for a in got], [(URL, b"rows")])

    def test_a_client_status_is_the_source_s_answer_and_is_collected(self) -> None:
        galaxy = GalaxyDatacache(opener=_status(404, b"no such file"))
        with collected_answers("describe_companion") as got:
            with self.assertRaises(ProviderError) as raised:
                galaxy.get(URL)
        self.assertEqual(raised.exception.failure_class, "rejected")
        self.assertEqual(raised.exception.evidence["source_code"], "source_answered")
        self.assertEqual([a.status_code for a in got], [404])

    def test_a_server_status_is_transient_and_not_an_answer(self) -> None:
        galaxy = GalaxyDatacache(opener=_status(503))
        with collected_answers("describe_companion") as got:
            with self.assertRaises(ProviderError) as raised:
                galaxy.get(URL)
        self.assertEqual(raised.exception.failure_class, "transient")
        self.assertEqual(got, [])

    def test_a_head_gives_the_size(self) -> None:
        galaxy = GalaxyDatacache(
            opener=lambda request, timeout: _answer(
                headers={"Content-Length": "100286508"}
            )
        )
        self.assertEqual(galaxy.size(URL), 100286508)
        galaxy = GalaxyDatacache(opener=lambda request, timeout: _answer())
        with self.assertRaises(ProviderError):
            galaxy.size(URL)

    def test_another_host_is_refused_before_sending(self) -> None:
        sent = []
        galaxy = GalaxyDatacache(opener=lambda request, timeout: sent.append(request))
        with self.assertRaises(ProviderError) as raised:
            galaxy.get("https://usegalaxy.org/api/tool_data/bwa_mem_indexes")
        self.assertEqual(raised.exception.failure_class, "rejected")
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
