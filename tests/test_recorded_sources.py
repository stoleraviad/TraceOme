"""The recorded source answers load beneath every source port, as received (#466).

`tests/recorded_sources.py` serves `tests/fixtures/source_answers/` through the one
transport every public-source port uses (`SourceTransport`), so these tests prove each
fixture reaches the ports a Run reads; `tests/test_source_fixture_cases.py` holds the
expected-red tests each fixture names. Recorded answers are observed source facts, not
expectations: no test here asserts what a reader makes of them.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
import unittest
import urllib.request
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from unittest import mock

import tests.recorded_sources as recorded
from bio_neb.providers.source_transport import AnsweredStatus, SourceTransport
from tests.recorded_sources import (
    MANIFEST,
    ROOT,
    RecordedOpener,
    UnrecordedSourceRequest,
    fixture,
    fixtures,
    recorded_ports,
)

AXES = {"naming", "assembly", "variant_source", "biology", "identifiers", "paper"}
OWNERS = {375, 383, 414, 459, 464, 467, 468, 469, 470, 471, 472, 473}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _stored(entry: dict) -> list[dict]:
    """Every file an entry names: its answers, their originals and its inputs."""
    return [*entry["answers"], *entry.get("originals", ()), *entry.get("inputs", ())]


class RecordedOpenerTests(unittest.TestCase):
    def test_each_answer_is_served_through_the_source_transport_as_received(self):
        for fixture_id, recorded_fixture in fixtures().items():
            transport = SourceTransport(opener=RecordedOpener(fixture_id))
            for answer in recorded_fixture.answers:
                with self.subTest(fixture=fixture_id, url=answer.url):
                    try:
                        received = transport.exchange(
                            answer.url,
                            method=answer.method,
                            max_bytes=len(answer.body) or 1,
                            status_fails=True,
                        )
                    except AnsweredStatus as status:
                        received = status.answer
                    self.assertEqual(
                        (received.status, received.body),
                        (
                            answer.status,
                            b"" if answer.method == "HEAD" else answer.body,
                        ),
                    )
                    for header, value in answer.headers.items():
                        self.assertEqual(received.headers[header], value)

    def test_a_port_reads_a_recorded_answer_through_its_own_view(self):
        ports = recorded_ports("variant-eva-retired", "variant-merged")

        eva = ports.investigation.fetch_eva_clustered_variant("rs160719159")
        dbsnp = ports.investigation.fetch_refsnp("rs60037639")

        self.assertEqual(
            (eva.status_code, eva.body),
            (410, fixture("variant-eva-retired").answers[0].body),
        )
        self.assertEqual(dbsnp.body, fixture("variant-merged").answers[0].body)
        self.assertEqual(
            ports.opener.sent,
            [
                (
                    "GET",
                    "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/"
                    "clustered-variants/160719159",
                ),
                ("GET", "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/60037639"),
            ],
        )

    def test_an_unrecorded_request_is_refused_before_any_answer(self):
        opener = RecordedOpener("variant-merged")
        request = urllib.request.Request(
            "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/7412"
        )

        with self.assertRaisesRegex(UnrecordedSourceRequest, "refsnp/7412"):
            opener(request, 1.0)
        self.assertNotIsInstance(UnrecordedSourceRequest("x"), AssertionError)

    def test_a_recorded_url_is_served_for_its_recorded_method_only(self):
        (get,) = fixture("variant-merged").answers
        head = next(
            a for a in fixture("naming-galaxy-unit").answers if a.method == "HEAD"
        )

        with self.assertRaisesRegex(UnrecordedSourceRequest, "HEAD .*refsnp/60037639"):
            SourceTransport(opener=RecordedOpener("variant-merged")).exchange(
                get.url, method="HEAD", max_bytes=0
            )
        served = SourceTransport(opener=RecordedOpener("naming-galaxy-unit")).exchange(
            head.url, method="HEAD", max_bytes=0
        )
        self.assertEqual(
            (served.status, served.headers["Content-Length"]),
            (head.status, head.headers["Content-Length"]),
        )

    def test_a_body_that_differs_from_its_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "source_answers"
            shutil.copytree(ROOT, copy)
            body = (
                copy / _manifest()["fixtures"]["variant-merged"]["answers"][0]["body"]
            )
            body.write_bytes(body.read_bytes().replace(b"4149056", b"4149057"))
            fixtures.cache_clear()
            self.addCleanup(fixtures.cache_clear)
            with (
                mock.patch.object(recorded, "ROOT", copy),
                mock.patch.object(recorded, "MANIFEST", copy / "manifest.json"),
                self.assertRaisesRegex(ValueError, "variant-merged/00.json"),
            ):
                fixtures()


class _Received:
    def __init__(self, body: bytes) -> None:
        self.status, self.body, self.headers = 200, body, Message()
        self.observed_at = datetime(2026, 10, 27, tzinfo=UTC)


class RecordingTests(unittest.TestCase):
    """A refresh that loses its source keeps every stored answer its manifest names."""

    URLS = (
        "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/1",
        "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/2",
    )

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "probe").mkdir()
        answers = []
        for index, url in enumerate(self.URLS):
            body = f'{{"old":{index}}}'.encode()
            (self.root / "probe" / f"0{index}.json").write_bytes(body)
            answers.append(
                {
                    "method": "GET",
                    "url": url,
                    "body": f"probe/0{index}.json",
                    "byte_count": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": recorded.SCHEMA,
                    "fixtures": {"probe": {"answers": answers}},
                }
            )
        )
        self.enterContext(mock.patch.object(recorded, "ROOT", self.root))
        self.enterContext(mock.patch.object(recorded, "MANIFEST", manifest))
        self.enterContext(mock.patch.object(recorded, "RECORDING_PAUSE_SECONDS", 0))
        self.transport = self.enterContext(
            mock.patch.object(recorded, "SourceTransport")
        ).return_value

    def stored(self) -> list[bytes]:
        answers = recorded._manifest()["fixtures"]["probe"]["answers"]
        return [recorded._body(answer) for answer in answers]

    def test_a_failed_first_get_leaves_every_answer_as_stored(self):
        self.transport.exchange.side_effect = TimeoutError("offline")

        with self.assertRaisesRegex(TimeoutError, "offline"):
            recorded.record(["probe"], again=True)

        self.assertEqual(self.stored(), [b'{"old":0}', b'{"old":1}'])

    def test_a_later_failure_keeps_the_answers_refreshed_before_it(self):
        self.transport.exchange.side_effect = [
            _Received(b'{"new":0}'),
            TimeoutError("offline"),
        ]

        with self.assertRaisesRegex(TimeoutError, "offline"):
            recorded.record(["probe"], again=True)

        self.assertEqual(self.stored(), [b'{"new":0}', b'{"old":1}'])
        self.assertEqual(
            sorted(path.name for path in (self.root / "probe").iterdir()),
            ["00.json", "01.json"],
        )


class ManifestTests(unittest.TestCase):
    def test_every_fixture_names_its_axis_case_rule_and_owner(self):
        entries = _manifest()["fixtures"]
        for fixture_id, entry in entries.items():
            with self.subTest(fixture=fixture_id):
                self.assertIn(entry["axis"], AXES)
                self.assertIn(entry["owner_ticket"], OWNERS)
                self.assertTrue(entry["case"].strip() and entry["breaks"].strip())
                self.assertTrue(entry["expected_red"] or entry.get("controls"))
                self.assertTrue(entry["answers"] or entry.get("inputs"))
        self.assertEqual({entry["axis"] for entry in entries.values()}, AXES)

    def test_each_named_test_is_expected_red_for_the_fixtures_owner(self):
        for fixture_id, entry in _manifest()["fixtures"].items():
            for name in entry["expected_red"] + entry.get("controls", []):
                with self.subTest(fixture=fixture_id, test=name):
                    module, case, method = name.rsplit(".", 2)
                    test = getattr(
                        getattr(importlib.import_module(module), case), method
                    )
                    red = getattr(test, "__unittest_expecting_failure__", False)
                    self.assertEqual(red, name in entry["expected_red"])
                    if red:
                        self.assertEqual(test.owner_ticket, entry["owner_ticket"])

    def test_a_derived_answer_names_the_recorded_answers_it_was_made_from(self):
        for fixture_id, entry in _manifest()["fixtures"].items():
            originals = {original["body"] for original in entry.get("originals", ())}
            for answer in entry["answers"]:
                if "derived" in answer:
                    with self.subTest(fixture=fixture_id, url=answer["url"]):
                        self.assertTrue(answer["derived"]["transformation"].strip())
                        self.assertLessEqual(set(answer["derived"]["from"]), originals)
                        self.assertIn("DERIVED", entry["case"])

    def test_every_stored_file_is_the_manifest_or_one_it_lists(self):
        listed = {"manifest.json", "README.md"}
        for entry in _manifest()["fixtures"].values():
            for stored in _stored(entry):
                listed.add(stored["body"])
        stored = {
            str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
        }
        self.assertEqual(stored, listed)

    def test_each_listed_body_holds_the_bytes_its_entry_records(self):
        for entry in _manifest()["fixtures"].values():
            for stored in _stored(entry):
                with self.subTest(body=stored["body"]):
                    self.assertEqual(
                        hashlib.sha256(recorded._body(stored)).hexdigest(),
                        stored["sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
