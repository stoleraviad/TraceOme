"""Planning's source Tools return what NCBI answered, as Investigation's do (#458 Q4).

A planning Run's reference discovery reads NCBI Datasets through the product adapter
and transport; only the opener is replaced. Each Tool's fact retains every GET's bytes
and shows them to the model beside its typed result, and a client status NCBI answers
is the model's to read, not an operational stop.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.parse
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock

from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.journey import Journey
from bio_neb.providers.ncbi_datasets import (
    NcbiDatasetsReferenceAdapter,
    UrllibNcbiTransport,
)
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.planning_assignment_fakes import PlanningComposition

TAXONOMY = "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon/"


def _no_match(term: str) -> bytes:
    """NCBI Taxonomy's explicit invalid-identifier report, in the shape the adapter
    reads as no match (`ncbi_datasets._explicit_taxonomy_no_match`)."""
    return json.dumps(
        {
            "reports": [
                {
                    "query": [term],
                    "errors": [
                        {
                            "invalid_identifiers": [term],
                            "reason": "Not a valid taxonomic identifier.",
                        }
                    ],
                }
            ]
        }
    ).encode()


class _Taxonomy:
    """An opener that answers each taxonomy GET with ``status`` and its body."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.sent: list[str] = []
        self.bodies: list[bytes] = []

    def __call__(self, request, timeout):  # noqa: ANN001, ANN204
        url = request.full_url
        self.sent.append(url)
        term = urllib.parse.unquote(
            url.removeprefix(TAXONOMY).removesuffix("/dataset_report")
        )
        body = (
            _no_match(term)
            if self.status == 200
            else b'{"error":"Bad Request","message":"Invalid taxon %s"}' % term.encode()
        )
        self.bodies.append(body)
        if self.status >= 300:
            raise urllib.error.HTTPError(
                url, self.status, "answer", Message(), io.BytesIO(body)
            )
        response = MagicMock(status=self.status)
        response.headers = {}
        response.read.side_effect = lambda size=-1: body if size < 0 else body[:size]
        response.__enter__.return_value = response
        return response


def _run(opener: _Taxonomy):
    with tempfile.TemporaryDirectory() as directory:
        receipt, paths = _registered(Path(directory))
        composition = PlanningComposition(receipt, paths)
        composition.reference_adapter = NcbiDatasetsReferenceAdapter(
            UrllibNcbiTransport(opener=opener)
        )
        with use_run_advancement_composition(composition):
            result = advance(ResumeRun(journey_file=receipt.journey_file))
        journey = Journey.open(Path(receipt.journey_file))
        events = read_evidence(journey)
        retained = {
            reference["artifact_id"]: journey.read_artifact(reference)
            for event in events
            if event["event_type"] == "ReferenceSourceRetrieved"
            for reference in event["artifact_references"]
        }
    returns = [
        part["content"]
        for event in events
        if event["event_type"] == "ModelRequestRecorded"
        for message in event["payload"]["compatible_adapter_evidence"]["request"][
            "messages"
        ]
        for part in message.get("parts", ())
        if part.get("part_kind") == "tool-return"
        and part.get("tool_name") == "resolve_taxon"
    ]
    return result, events, retained, returns


class PlanningSourceAnswerTests(unittest.TestCase):
    def test_a_taxon_lookup_retains_and_returns_ncbis_answer(self) -> None:
        opener = _Taxonomy(200)

        _result, events, retained, returns = _run(opener)

        [fact] = [e for e in events if e["event_type"] == "TaxonResolutionRecorded"]
        self.assertEqual(fact["payload"]["result"]["resolution"], "not_found")
        [body] = opener.bodies
        [answer] = fact["payload"]["source_answers"]
        self.assertEqual(
            (answer["operation"], answer["status_code"], answer["shown"]),
            ("resolve_taxon", 200, "whole"),
        )
        self.assertEqual(answer["request_identity"], opener.sent[0])
        self.assertEqual(answer["body"], json.loads(body))
        self.assertEqual(answer["sha256"], hashlib.sha256(body).hexdigest())
        # The whole answer is retained under its digest, on its own GET's record,
        # which the fact follows.
        self.assertEqual(retained, {f"sha256:{answer['sha256']}": body})
        [got] = [e for e in events if e["event_type"] == "ReferenceSourceRetrieved"]
        self.assertEqual(fact["parent_ids"], [got["event_id"]])
        self.assertEqual(
            (got["payload"]["request_identity"], got["payload"]["status_code"]),
            (opener.sent[0], 200),
        )
        # The model reads each answer's body, without its digest.
        shown = [
            {
                "operation": item["operation"],
                "request": item["request_identity"],
                "status_code": item["status_code"],
                "shown": item["shown"],
                "body": item["body"],
            }
            for item in fact["payload"]["source_answers"]
        ]
        self.assertEqual(
            [r["source_answers"] for r in returns if "source_answers" in r],
            [shown] * len(returns),
        )
        self.assertTrue(returns)

    def test_a_client_status_ncbi_answers_is_read_and_stops_no_one(self) -> None:
        # Not every non-success is ``http_error``, an operational stop that asks
        # a person; a 400 with a body is NCBI's answer about the term.
        opener = _Taxonomy(400)

        result, events, retained, returns = _run(opener)

        kinds = {event["event_type"] for event in events}
        self.assertNotIn("AgentFailureReviewRequested", kinds)
        self.assertNotEqual(result.status, "operational_review_pending")
        facts = [e for e in events if e["event_type"] == "TaxonResolutionRecorded"]
        self.assertTrue(facts)
        for fact in facts:
            failure = fact["payload"]["result"]["failure"]
            self.assertEqual(failure["code"], "source_answered")
            self.assertIn("HTTP 400", failure["public_message"])
            [answer] = fact["payload"]["source_answers"]
            self.assertEqual(answer["status_code"], 400)
            self.assertEqual(answer["body"]["error"], "Bad Request")
        self.assertEqual(len(retained), len(set(opener.bodies)))
        self.assertTrue(returns)
        self.assertTrue(
            all(r["source_answers"][0]["status_code"] == 400 for r in returns)
        )


class PlanningAnswerBindingTests(unittest.TestCase):
    """Reopen binds what a planning lookup showed to the bytes it retains (#458
    Standards r1)."""

    @classmethod
    def setUpClass(cls) -> None:
        directory = cls.enterClassContext(tempfile.TemporaryDirectory())
        receipt, paths = _registered(Path(directory))
        composition = PlanningComposition(receipt, paths)
        composition.reference_adapter = NcbiDatasetsReferenceAdapter(
            UrllibNcbiTransport(opener=_Taxonomy(200))
        )
        with use_run_advancement_composition(composition):
            advance(ResumeRun(journey_file=receipt.journey_file))
        cls.journey_file = receipt.journey_file
        cls.events = read_evidence(Journey.open(Path(receipt.journey_file)))

    def _forged(self, forge):
        """The taxon fact's answers forged, its return rebound coherently (its digest
        and the requests that carry it), so only the answers' own reading refuses."""
        import copy

        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            tool_model_view,
        )
        from bio_neb.content import canonical_json_digest, shown_to_model

        events = copy.deepcopy(self.events)
        fact = next(e for e in events if e["event_type"] == "TaxonResolutionRecorded")
        forge(fact["payload"]["source_answers"])
        visible = shown_to_model(tool_model_view(model_view_of("resolve_taxon"), fact))
        succeeded = next(
            e
            for e in events
            if e["event_type"] == "ObservedToolExecutionSucceeded"
            and e["payload"]["stage_result_event_id"] == fact["event_id"]
        )
        succeeded["payload"]["model_visible_result_digest"] = canonical_json_digest(
            visible
        )
        call_id = next(
            e["payload"]["tool_call_id"]
            for e in events
            if e["event_id"] == succeeded["payload"]["started_event_id"]
        )
        for event in events:
            if event["event_type"] != "ModelRequestRecorded":
                continue
            request = event["payload"]["compatible_adapter_evidence"]["request"]
            for message in request.get("messages", []):
                for part in message.get("parts", []):
                    if (
                        part.get("part_kind") == "tool-return"
                        and part.get("tool_call_id") == call_id
                    ):
                        part["content"] = copy.deepcopy(visible)
        return events

    def test_a_lookup_shows_exactly_its_own_retained_gets(self) -> None:
        # Standards r1/r2 ST2: no answer dropped or invented, and each answer's
        # request and status are its retained GET's, for every planning reader.
        from bio_neb.application.planning_read import PrefixSnapshot
        from bio_neb.application.queries import project_journey_events

        PrefixSnapshot(self.events)
        project_journey_events(self.events, journey_file=self.journey_file)
        for name, forge in {
            "dropped": lambda answers: answers.clear(),
            "invented": lambda answers: answers.append(dict(answers[0])),
            "another request": lambda answers: answers[0].update(
                request_identity="https://api.ncbi.nlm.nih.gov/never-requested",
                status_code=410,
            ),
        }.items():
            forged = self._forged(forge)
            with self.subTest(answer=name):
                with self.assertRaisesRegex(ValueError, "changed its source answers"):
                    PrefixSnapshot(forged)
                with self.assertRaisesRegex(ValueError, "changed its source answers"):
                    project_journey_events(forged, journey_file=self.journey_file)

    def test_a_broken_parent_is_refused_before_any_get_is_walked(self) -> None:
        # Standards r3 ST3: the GET walk reads only a prefix whose parentage the
        # common check passed; a missing or self parent never reaches it.
        import copy

        from bio_neb.application.planning_read import PrefixSnapshot

        for name, parent in {
            "missing": lambda _got: "ffffffff-ffff-ffff-ffff-ffffffffffff",
            "itself": lambda got: got["event_id"],
        }.items():
            events = copy.deepcopy(self.events)
            got = next(
                e for e in events if e["event_type"] == "ReferenceSourceRetrieved"
            )
            got["parent_ids"] = [parent(got)]
            with (
                self.subTest(parent=name),
                self.assertRaisesRegex(ValueError, "invalid backward parentage"),
            ):
                PrefixSnapshot(events)

    def test_a_shown_body_is_how_its_retained_bytes_show(self) -> None:
        from bio_neb.agentic_journey.source_answers import validate_shown_answers

        validate_shown_answers(
            self.events, journey=Journey.open(Path(self.journey_file))
        )
        forged = self._forged(
            lambda answers: answers[0].update(body={"claim": "never sent"})
        )
        with self.assertRaisesRegex(ValueError, "how its retained bytes show"):
            validate_shown_answers(
                forged, journey=Journey.open(Path(self.journey_file))
            )


if __name__ == "__main__":
    unittest.main()


class PrebuiltIndexSourceAnswerTests(unittest.TestCase):
    """The genomic FASTA's index Tools keep every GET as a source answer (#458, #464):
    a search's two catalogues, a read's `.ann`, each retained on its own
    `ReferenceSourceRetrieved`; each file's size is a HEAD, nothing is downloaded."""

    def test_each_index_tool_shows_and_retains_its_own_gets(self) -> None:
        from bio_neb.prebuilt_index.catalogue import REPOSITORIES
        from bio_neb.providers.galaxy_datacache import GalaxyDatacache
        from tests.prebuilt_index_support import (
            GALAXY_INDEX,
            GalaxyOpener,
            offline_galaxy_files,
        )

        with tempfile.TemporaryDirectory() as directory:
            opener = GalaxyOpener(files=offline_galaxy_files(Path(directory) / "g"))
            receipt, paths = _registered(Path(directory))
            composition = PlanningComposition(receipt, paths)
            composition.galaxy_adapter = GalaxyDatacache(opener=opener)
            composition.companion_adapter.answers = True
            with use_run_advancement_composition(composition):
                advance(ResumeRun(journey_file=receipt.journey_file))
            journey = Journey.open(Path(receipt.journey_file))
            events = read_evidence(journey)

        [listed] = [e for e in events if e["event_type"] == "PrebuiltIndexesListed"]
        [read] = [e for e in events if e["event_type"] == "PrebuiltIndexRead"]
        brc = REPOSITORIES["brc"]
        shown = {
            fact["event_type"]: [
                a["request_identity"] for a in fact["payload"]["source_answers"]
            ]
            for fact in (listed, read)
        }
        self.assertEqual(
            shown,
            {
                "PrebuiltIndexesListed": [brc.index_catalogue, brc.fasta_catalogue],
                "PrebuiltIndexRead": [f"{GALAXY_INDEX}.ann"],
            },
        )
        gets = [url for method, url in opener.sent if method == "GET"]
        self.assertEqual(gets, [*shown["PrebuiltIndexesListed"], f"{GALAXY_INDEX}.ann"])
        retrieved = [
            e["payload"]["request_identity"]
            for e in events
            if e["event_type"] == "ReferenceSourceRetrieved"
        ]
        self.assertLessEqual(set(gets), set(retrieved))
        # Every one of the candidate's seven files is sized by a HEAD.
        self.assertEqual(sum(1 for method, _ in opener.sent if method == "HEAD"), 7)
