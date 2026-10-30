"""Every Run is offered both lookup Tools, dbSNP's and EVA's, each with metadata that
steers the choice; the model chooses and no code routes by assembly (#469, owner RC2 1,
ADR 0049; it supersedes #366's one-Tool route).

The planner runs through the real Investigation Attempt (`run_attempt`) with a scripted
model and recorded source bodies; the fakes only answer the GETs the product asks for.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.investigation.investigation import (
    InvestigationInput,
    InvestigationVcfIdentity,
    UserAssertion,
)
from bio_neb.journey import Journey
from bio_neb.providers.ncbi_investigation import InvestigationSourceResponse
from bio_neb.recording import read_evidence
from tests.eva_retained_bodies import (
    PIG_CM000823_5_ALIAS,
    PIG_RS5334475172,
    PIG_RS5334475172_SUBMITTED,
    RS160719159_GONE,
)
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    run_attempt,
)

SSCROFA11 = "GCF_000003025.6"
SSCROFA11_REPORT = (
    b'{"reports":[{"accession":"GCF_000003025.6","paired_accession":"GCA_000003025.6",'
    b'"assembly_info":{"assembly_name":"Sscrofa11.1","assembly_status":"current"}}]}'
)
"""Synthetic NCBI Datasets report in the shape `normalize_assembly_report` reads, with the
paired INSDC accession an EVA lookup reads (#468, owner R2 a)."""


SYNTHETIC_RS = "rs81000001"
"""A second pig rsID, synthetic: the recorded deletion's bodies under another number."""


def _as(rsid: str, body: bytes) -> bytes:
    return body.replace(b"5334475172", rsid.removeprefix("rs").encode())


class PigSources(SourceResponsesOnly):
    """NCBI's assembly report and EVA's recorded pig answers; dbSNP is never asked."""

    def __init__(self, failures: dict[str, list[int]] | None = None) -> None:
        super().__init__(assembly_body=SSCROFA11_REPORT)
        # Statuses to answer, in order, before the recorded body (a 503 retries),
        # by operation or by "operation:key".
        self.failures = failures or {}

    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        raise AssertionError("a non-human Run never reaches dbSNP")

    def _eva(self, operation: str, key: str, url: str, body: bytes):
        self.operations.append(f"{operation}:{key}")
        pending = self.failures.get(f"{operation}:{key}") or self.failures.get(
            operation, []
        )
        status = pending.pop(0) if pending else 200
        return InvestigationSourceResponse(
            request_identity=url,
            status_code=status,
            body=body if status == 200 else b'{"error":"unavailable"}',
            observed_at="2026-10-17T12:00:00Z",
            elapsed_time_ms=3,
        )

    def fetch_eva_clustered_variant(self, rsid: str):
        return self._eva(
            "eva_clustered_variant",
            rsid,
            "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants/"
            + rsid.removeprefix("rs"),
            _as(rsid, PIG_RS5334475172),
        )

    def fetch_eva_submitted_variants(self, rsid: str):
        return self._eva(
            "eva_submitted_variants",
            rsid,
            "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants/"
            + rsid.removeprefix("rs")
            + "/submitted",
            _as(rsid, PIG_RS5334475172_SUBMITTED),
        )

    def fetch_eva_contig_alias(self, insdc_contig: str):
        return self._eva(
            "eva_contig_alias",
            insdc_contig,
            "https://www.ebi.ac.uk/eva/webservices/contig-alias/v1/chromosomes/genbank/CM000823.5",
            PIG_CM000823_5_ALIAS,
        )


def pig_input() -> tuple[InvestigationInput, UserAssertion]:
    investigation_input = InvestigationInput.create(
        vcf=InvestigationVcfIdentity.create(
            bucket="accepted-run-bucket",
            object_key="outputs/PIG01.vcf.gz",
            byte_count=1234,
            sha256="c" * 64,
        ),
        sample_id="PIG01",
        assembly_accession=SSCROFA11,
    )
    assertion = UserAssertion.create(
        asserted_by="researcher:owner",
        request_text="Investigate rs5334475172 in sample PIG01.",
        rsid="rs5334475172",
        predecessor_ids=(investigation_input.input_id,),
    )
    return investigation_input, assertion


class _OfferedTools(SequentialAdaptiveModel):
    def __init__(self) -> None:
        super().__init__()
        self.offered: list[set[str]] = []

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        self.offered.append({tool.name for tool in info.function_tools})
        return await super().respond(messages, info)


def _pig_run(directory: str, sources: PigSources):
    investigation_input, assertion = pig_input()
    model = _OfferedTools()
    projection, journey = run_attempt(
        Path(directory),
        model,
        sources,
        investigation_input=investigation_input,
        assertion=assertion,
    )
    return projection, read_evidence(journey), model


class NonHumanRouteTests(unittest.TestCase):
    def test_a_pig_run_looks_up_eva_and_proposes_work_on_its_own_contig(self) -> None:
        sources = PigSources()
        with tempfile.TemporaryDirectory() as directory:
            projection, events, model = _pig_run(directory, sources)

        self.assertEqual(projection.status, "review_pending")
        # Both Tools are offered; the model chose EVA for a pig Run.
        for tools in model.offered:
            self.assertLessEqual({"fetch_refsnp", "fetch_eva_variant"}, tools)
        self.assertEqual(
            sources.operations,
            [
                "eva_clustered_variant:rs5334475172",
                # The Run's INSDC twin, as NCBI pairs it.
                f"fetch_assembly_report:{SSCROFA11}",
                "eva_contig_alias:CM000823.5",
                "eva_submitted_variants:rs5334475172",
                f"fetch_assembly_report:{SSCROFA11}",
            ],
        )
        retained = [
            (e["payload"]["source"]["operation"], e["payload"]["source"]["authority"])
            for e in events
            if e["event_type"] == "InvestigationSourceRetrieved"
        ]
        self.assertEqual(
            retained,
            [
                ("eva_clustered_variant", "eva"),
                ("fetch_assembly_report", "ncbi"),
                ("eva_contig_alias", "eva"),
                ("eva_submitted_variants", "eva"),
                ("fetch_assembly_report", "ncbi"),
            ],
        )
        [proposal] = projection.proposals
        [item] = proposal.work_items
        self.assertEqual(
            (
                item.target.assembly_accession,
                item.target.contig,
                item.target.one_based_position,
                item.target.ref,
                item.target.alt,
            ),
            (SSCROFA11, "NC_010454.4", 3556402, "GTGTCCCCCTCGC", ""),
        )
        eva_sources = {
            e["payload"]["source"]["source_id"]
            for e in events
            if e["event_type"] == "InvestigationSourceRetrieved"
            and e["payload"]["source"]["authority"] == "eva"
        }
        self.assertTrue(eva_sources <= set(item.source_ids))

    def test_the_run_feed_names_eva_for_a_pig_runs_lookups(self) -> None:
        from bio_neb.application.run_reader import read_journey_feed

        investigation_input, assertion = pig_input()
        with tempfile.TemporaryDirectory() as directory:
            _projection, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                PigSources(),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            feed = read_journey_feed(str(journey.path), everything=True)

        texts = [item.text for item in feed.items]
        self.assertIn("rs5334475172: found in EVA", texts)
        self.assertIn("EVA variant lookup for rs5334475172 started", texts)
        self.assertIn("EVA contig names received", texts)
        self.assertFalse(any("dbSNP" in text for text in texts))

    def test_a_transient_eva_fault_retries_on_its_own_surface(self) -> None:
        sources = PigSources(failures={"eva_contig_alias": [503]})
        with tempfile.TemporaryDirectory() as directory:
            projection, events, _model = _pig_run(directory, sources)

        self.assertEqual(projection.status, "review_pending")
        [failure] = [
            e["payload"]
            for e in events
            if e["event_type"] == "SourceOperationFailureRecorded"
        ]
        self.assertEqual(
            (failure["policy"]["surface"]["name"], failure["failure_class"]),
            ("eva_contig_alias", "transient"),
        )
        self.assertEqual(sources.operations.count("eva_contig_alias:CM000823.5"), 2)

    def test_an_exhausted_eva_fault_asks_and_the_other_member_still_runs(self) -> None:
        from bio_neb.investigation.investigation import InvestigationRequestItem

        sources = PigSources(
            failures={"eva_submitted_variants:rs5334475172": [503, 503, 503]}
        )
        investigation_input, _ = pig_input()
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=tuple(
                InvestigationRequestItem(
                    request_text=f"Investigate {rsid} in sample PIG01.", rsid=rsid
                )
                for rsid in ("rs5334475172", SYNTHETIC_RS)
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(
            sources.operations.count("eva_submitted_variants:rs5334475172"), 3
        )
        [failed] = [
            e["payload"]
            for e in events
            if e["event_type"] == "InvestigationSourceContactFailed"
        ]
        self.assertEqual(
            (failed["operation"], failed["request_key"]),
            ("eva_submitted_variants", "rs5334475172"),
        )
        [fault] = [
            e["payload"]
            for e in events
            if e["event_type"] == "OperationalFailureRecorded"
        ]
        self.assertEqual(
            (fault["policy"]["surface"]["name"], fault["failure_class"]),
            ("eva_submitted_variants", "exhausted"),
        )
        # The other member was still looked up and resolved.
        resolved = [
            e["payload"]["query"]
            for e in events
            if e["event_type"] == "RefSnpToolResultRecorded"
            and e["payload"]["result"]["status"] == "resolved"
        ]
        self.assertEqual(resolved, [SYNTHETIC_RS])
        self.assertIn("pending", projection.status)


WBCEL235 = "GCF_000002985.6"


class GoneRecordSources(SourceResponsesOnly):
    """EVA as #457's C. elegans check met it: rs160719159 answered HTTP 410 with a
    record on another taxon's assembly (#457 F1)."""

    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        raise AssertionError("a non-human Run never reaches dbSNP")

    def fetch_eva_clustered_variant(self, rsid: str):
        self.operations.append(f"eva_clustered_variant:{rsid}")
        return InvestigationSourceResponse(
            request_identity=(
                "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/"
                "clustered-variants/160719159"
            ),
            status_code=410,
            body=RS160719159_GONE,
            observed_at="2026-10-25T20:10:00Z",
            elapsed_time_ms=40,
        )


class _ReadsReturns(SequentialAdaptiveModel):
    """The sequential planner, keeping each lookup return it was sent."""

    def __init__(self) -> None:
        super().__init__()
        self.returns: list[dict] = []

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        parts = [
            part
            for message in ModelMessagesTypeAdapter.dump_python(messages, mode="json")
            for part in message["parts"]
        ]
        self.returns = [
            part["content"]
            for part in parts
            if part["part_kind"] == "tool-return"
            and part["tool_name"] == "fetch_eva_variant"
        ]
        return await super().respond(messages, info)


def _rebind_return(timeline, fact) -> None:
    """Keep a forged fact's return coherent (its model view's digest and the
    requests that carry it, #478 G1.3), so only the fact's own reading can refuse
    it."""
    from bio_neb.agentic_journey.tool_execution import (
        model_view_of,
        model_visible,
        tool_model_view,
    )
    from bio_neb.content import canonical_json_digest

    call_id = fact["payload"]["tool_call_id"]
    [tool_name] = {
        event["payload"]["tool_name"]
        for event in timeline
        if event["event_type"] == "ObservedToolExecutionStarted"
        and event["payload"]["tool_call_id"] == call_id
    }
    visible, guarded = model_visible(tool_model_view(model_view_of(tool_name), fact))
    for event in timeline:
        payload = event["payload"]
        if (
            event["event_type"] == "ObservedToolExecutionSucceeded"
            and payload["stage_result_event_id"] == fact["event_id"]
        ):
            payload["model_visible_result_digest"] = canonical_json_digest(visible)
            payload["result_guarded"] = guarded
        if event["event_type"] != "ModelRequestRecorded":
            continue
        request = payload["compatible_adapter_evidence"]["request"]
        for message in request.get("messages", []):
            for part in message.get("parts", []):
                if (
                    part.get("part_kind") == "tool-return"
                    and part.get("tool_call_id") == call_id
                ):
                    part["content"] = visible


class GoneRecordTests(unittest.TestCase):
    def test_a_410_is_read_as_the_sources_answer_and_stops_no_one(self) -> None:
        # The 410 was rejected before the lookup or the model saw it, and the Attempt
        # stopped twice for a person. Now EVA's answer is retained, typed with its
        # status named, and shown whole to the model beside the typed result.
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/CE01.vcf.gz",
                byte_count=1234,
                sha256="d" * 64,
            ),
            sample_id="CE01",
            assembly_accession=WBCEL235,
        )
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs160719159 in sample CE01.",
            rsid="rs160719159",
            predecessor_ids=(investigation_input.input_id,),
        )
        model, sources = _ReadsReturns(), GoneRecordSources()
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory),
                model,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        kinds = {event["event_type"] for event in events}
        self.assertFalse(
            kinds
            & {
                "InvestigationSourceContactFailed",
                "SourceOperationFailureRecorded",
                "OperationalFailureRecorded",
            }
        )
        self.assertEqual(projection.status, "review_pending")
        self.assertEqual(sources.operations, ["eva_clustered_variant:rs160719159"])
        [retained] = [
            e["payload"]["source"]
            for e in events
            if e["event_type"] == "InvestigationSourceRetrieved"
        ]
        self.assertEqual(
            (retained["status_code"], retained["raw_response_byte_count"]), (410, 232)
        )
        [returned] = model.returns
        [fact] = [e for e in events if e["event_type"] == "RefSnpToolResultRecorded"]
        # EVA's own answer, with no verdict on it (ADR 0049): the statement is
        # what EVA answered, in its own words, and no instruction; the next step
        # is the view's own field.
        gone = (
            "EVA's clustered variant answered HTTP 410 for rs160719159: "
            + RS160719159_GONE.decode()
            + "."
        )
        self.assertEqual(returned["status"], "source_answered")
        self.assertEqual(returned["statement"], gone)
        self.assertEqual(fact["payload"]["result"]["conflicts"], [gone])
        self.assertNotIn("conflicts", returned["result"])
        self.assertNotIn("source_answers", returned["statement"])
        self.assertEqual(returned["cite"], {"refsnp_event_id": fact["event_id"]})
        self.assertIn("source_answers", returned["next"])
        [answer] = returned["source_answers"]
        self.assertEqual(
            (answer["operation"], answer["status_code"], answer["shown"]),
            ("eva_clustered_variant", 410, "whole"),
        )
        self.assertEqual(answer["body"][0]["data"]["taxonomyAccession"], 9940)
        self.assertEqual(
            answer["body"][0]["data"]["assemblyAccession"], "GCA_000298735.2"
        )
        # The view shows the retained bytes' body, without their hash.
        self.assertEqual(answer["body"], json.loads(RS160719159_GONE))
        self.assertNotIn("sha256", answer)
        self.assertEqual(
            retained["raw_response_sha256"],
            hashlib.sha256(RS160719159_GONE).hexdigest(),
        )
        # The item Review shows the researcher the same statement of EVA's.
        question = projection.pending_request.question
        self.assertEqual(question.issue.conflicts, (gone,))
        self.assertEqual(question.missing_information, gone)

    def test_the_run_feed_says_eva_answered_with_no_verdict(self) -> None:
        """The feed names what EVA's 410 is: its answer for the assistant to read, no
        verdict of code's on it (ADR 0049; owner #468 R1 b)."""
        from bio_neb.application.run_reader import read_journey_feed

        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/CE01.vcf.gz",
                byte_count=1234,
                sha256="d" * 64,
            ),
            sample_id="CE01",
            assembly_accession=WBCEL235,
        )
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs160719159 in sample CE01.",
            rsid="rs160719159",
            predecessor_ids=(investigation_input.input_id,),
        )
        with tempfile.TemporaryDirectory() as directory:
            _projection, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                GoneRecordSources(),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            feed = read_journey_feed(str(journey.path), everything=True)

        self.assertIn(
            "rs160719159: EVA answered with something other than a record; the "
            "assistant reads its reply",
            [item.text for item in feed.items],
        )

    def test_a_dbsnp_success_that_is_no_record_is_retained_and_read(self) -> None:
        # Spec r1 SP3: a 2xx body that is not a JSON object was rejected before
        # retention, an operational stop with nothing for the model to read.
        from tests.sequential_investigation_support import SourceResponsesOnly

        raw = b"The request key was not understood; see our new identifier."
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                SourceResponsesOnly(refsnp_body=raw),
            )
            events = read_evidence(journey)

        kinds = {event["event_type"] for event in events}
        self.assertFalse(
            kinds & {"OperationalFailureRecorded", "AgentFailureReviewRequested"}
        )
        self.assertEqual(projection.status, "review_pending")
        [fact] = [e for e in events if e["event_type"] == "RefSnpToolResultRecorded"]
        self.assertEqual(fact["payload"]["result"]["status"], "malformed")
        [answer] = fact["payload"]["source_answers"]
        self.assertEqual(
            (answer["status_code"], answer["media"], answer["body"]),
            (200, "text", raw.decode()),
        )

    def test_reopen_refuses_an_answer_the_attempt_never_retained(self) -> None:
        import copy

        from bio_neb.investigation.investigation import project_investigation

        investigation_input, assertion = pig_input()
        with tempfile.TemporaryDirectory() as directory:
            _projection, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                PigSources(),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)
        project_investigation(events)
        for forge in (
            lambda answers: answers[0].update(status_code=200, sha256="0" * 64),
            # The result's own source left out of what the Tool showed.
            lambda answers: answers.pop(0),
        ):
            forged = copy.deepcopy(events)
            fact = next(
                e for e in forged if e["event_type"] == "RefSnpToolResultRecorded"
            )
            forge(fact["payload"]["source_answers"])
            _rebind_return(forged, fact)
            with (
                self.subTest(forge=forge),
                self.assertRaisesRegex(ValueError, "changed its source answers"),
            ):
                project_investigation(forged)


class ShownAnswerBindingTests(unittest.TestCase):
    """Reopen binds what a Tool showed to its own retained GETs (#458 Standards r1)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = cls.enterClassContext(tempfile.TemporaryDirectory())
        investigation_input, assertion = pig_input()
        _projection, cls.journey = run_attempt(
            Path(cls.directory),
            SequentialAdaptiveModel(),
            PigSources(),
            investigation_input=investigation_input,
            assertion=assertion,
        )
        cls.events = read_evidence(cls.journey)

    def _forged(self, forge):
        import copy

        events = copy.deepcopy(self.events)
        facts = {e["event_type"]: e for e in events if "source_answers" in e["payload"]}
        forge(facts)
        for fact in facts.values():
            _rebind_return(events, fact)
        return events

    def test_a_tool_shows_exactly_its_own_gets_in_call_order(self) -> None:
        # ST2: the EVA lookup's three GETs, each once, in the order it made them.
        from bio_neb.investigation.investigation import project_investigation

        project_investigation(self.events)
        lookup = "RefSnpToolResultRecorded"
        for name, forge in {
            "dropped": lambda f: f[lookup]["payload"]["source_answers"].pop(),
            "reordered": lambda f: f[lookup]["payload"]["source_answers"].reverse(),
            "duplicated": lambda f: f[lookup]["payload"]["source_answers"].append(
                f[lookup]["payload"]["source_answers"][0]
            ),
            "another call's": lambda f: f[lookup]["payload"]["source_answers"].append(
                f["AssemblyReportToolResultRecorded"]["payload"]["source_answers"][0]
            ),
        }.items():
            with (
                self.subTest(answer=name),
                self.assertRaisesRegex(ValueError, "changed its source answers"),
            ):
                project_investigation(self._forged(forge))

    def test_a_shown_body_is_how_its_retained_bytes_show(self) -> None:
        # ST1: the body, its media and its cut re-render from the retained bytes;
        # the owner's checked read refuses any other, a detached fold reads none.
        from bio_neb.agentic_journey.source_answers import validate_shown_answers
        from bio_neb.application.planning_read import PrefixSnapshot

        validate_shown_answers(self.events, journey=self.journey)
        lookup = "RefSnpToolResultRecorded"

        def answer(facts):
            return facts[lookup]["payload"]["source_answers"][0]

        for name, forge in {
            "body": lambda f: answer(f).update(body={"claim": "never sent"}),
            "media": lambda f: answer(f).update(media="text"),
            "cut": lambda f: answer(f).update(shown="excerpt"),
        }.items():
            forged = self._forged(forge)
            with self.subTest(changed=name):
                with self.assertRaisesRegex(ValueError, "how its retained bytes show"):
                    validate_shown_answers(
                        forged, journey=Journey.open(self.journey.path)
                    )
                with self.assertRaisesRegex(ValueError, "how its retained bytes show"):
                    PrefixSnapshot(forged, owner=Journey.open(self.journey.path))


class _NotCurrentReport(PigSources):
    """The Run's assembly answered first with NCBI's ``{}`` for a no-longer-current
    report, then with its any-version report, in one operation."""

    def fetch_assembly_report(self, assembly_accession: str):
        final = super().fetch_assembly_report(assembly_accession)
        plain = InvestigationSourceResponse(
            request_identity=final.request_identity,
            status_code=200,
            body=b"{}",
            observed_at=final.observed_at,
            elapsed_time_ms=1,
        )
        return InvestigationSourceResponse(
            request_identity=f"{final.request_identity}?filters.assembly_version="
            "all_assemblies",
            status_code=200,
            body=final.body,
            observed_at=final.observed_at,
            elapsed_time_ms=2,
            preceding=(plain,),
        )


class EveryGetTests(unittest.TestCase):
    def test_both_gets_of_a_no_longer_current_report_are_retained_and_shown(self):
        from bio_neb.investigation.investigation import project_investigation

        investigation_input, assertion = pig_input()
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                _NotCurrentReport(),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(projection.status, "review_pending")
        retained = [
            e["payload"]["source"]
            for e in events
            if e["event_type"] == "InvestigationSourceRetrieved"
            and e["payload"]["source"]["operation"] == "fetch_assembly_report"
        ]
        [fact] = [
            e for e in events if e["event_type"] == "AssemblyReportToolResultRecorded"
        ]
        answers = fact["payload"]["source_answers"]
        self.assertEqual(
            [a["body"] for a in answers], [{}, json.loads(SSCROFA11_REPORT)]
        )
        # The EVA lookup read the same two GETs first; the assembly Tool
        # shows its own two.
        self.assertEqual(len(retained), 4)
        self.assertEqual(
            [a["sha256"] for a in answers],
            [source["raw_response_sha256"] for source in retained[2:]],
        )
        self.assertEqual(
            fact["payload"]["result"]["source"]["request_identity"],
            answers[1]["request_identity"],
        )
        project_investigation(events)


class _EvaRefetching:
    """A pig Run's planner that only sends the same EVA lookup."""

    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        del messages, info
        self.calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "fetch_eva_variant",
                    {"rsid": "rs5334475172"},
                    tool_call_id=f"fetch-{self.calls}",
                )
            ]
        )


class _MovedRecordSources(GoneRecordSources):
    """EVA answering rs160719159 with HTTP 410, its body (synthetic: the recorded body
    with a ``mergedInto`` field added) naming another clustered variant, which EVA does
    not hold."""

    def fetch_eva_clustered_variant(self, rsid: str):
        if rsid == "rs160719159":
            moved = json.loads(RS160719159_GONE)
            moved[0]["mergedInto"] = 7089734242
            gone = super().fetch_eva_clustered_variant(rsid)
            return InvestigationSourceResponse(
                request_identity=gone.request_identity,
                status_code=410,
                body=json.dumps(moved).encode(),
                observed_at=gone.observed_at,
                elapsed_time_ms=1,
            )
        self.operations.append(f"eva_clustered_variant:{rsid}")
        return InvestigationSourceResponse(
            request_identity=(
                "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/"
                f"clustered-variants/{rsid[2:]}"
            ),
            status_code=404,
            body=b'{"message": "Accession could not be found"}',
            observed_at="2026-10-27T12:00:00Z",
            elapsed_time_ms=1,
        )


class _FollowsTheAnswer:
    """A C. elegans planner that looks the stated rsID up, then an rsID no answer names,
    then the one EVA's answer names, then repeats it until the Attempt stops."""

    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        del messages, info
        self.calls += 1
        rsid = {1: "rs160719159", 2: "rs999"}.get(self.calls, "rs7089734242")
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "fetch_eva_variant", {"rsid": rsid}, tool_call_id=f"f-{self.calls}"
                )
            ]
        )


class FollowUpTests(unittest.TestCase):
    def test_a_lookup_follows_only_an_rsid_the_sources_answer_names(self) -> None:
        """Through the real Tool and admission (owner #468 Q3 a, ADR 0049): EVA's 410
        names a record, and the model's lookup of it is admitted on membership in that
        retained answer; an rsID no answer names is refused before contact."""
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/CE01.vcf.gz",
                byte_count=1234,
                sha256="d" * 64,
            ),
            sample_id="CE01",
            assembly_accession=WBCEL235,
        )
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs160719159 in sample CE01.",
            rsid="rs160719159",
            predecessor_ids=(investigation_input.input_id,),
        )
        sources = _MovedRecordSources()
        with tempfile.TemporaryDirectory() as directory:
            _projection, journey = run_attempt(
                Path(directory),
                _FollowsTheAnswer(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(
            sources.operations,
            ["eva_clustered_variant:rs160719159", "eva_clustered_variant:rs7089734242"],
        )
        self.assertEqual(
            [
                (e["payload"]["query"], e["payload"]["result"]["status"])
                for e in events
                if e["event_type"] == "RefSnpToolResultRecorded"
            ],
            [("rs160719159", "source_answered"), ("rs7089734242", "source_answered")],
        )
        refused = [
            json.dumps(e["payload"])
            for e in events
            if "eva_query_not_requested" in json.dumps(e["payload"])
        ]
        self.assertTrue(refused)
        self.assertTrue(all("rs999" in text for text in refused[:1]))


class SharedLoopControlTests(unittest.TestCase):
    def test_an_identical_eva_lookup_is_stopped_before_its_contacts(self) -> None:
        # The shared loop's repeated-call guard (ADR 0044) covers the EVA Tool as it
        # covers fetch_refsnp (`tests/test_repeated_calls.py`): only the first
        # lookup reaches EVA; its repeats are refused, each a failed turn, and the
        # 4th failed turn in a row (the default failed_turns) stops.
        model = _EvaRefetching()
        sources = PigSources()
        investigation_input, assertion = pig_input()
        with tempfile.TemporaryDirectory() as directory:
            projection, journey = run_attempt(
                Path(directory),
                model,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(projection.attempts[-1].stop_reason, "failed_turns")
        self.assertEqual(
            sources.operations,
            [
                "eva_clustered_variant:rs5334475172",
                f"fetch_assembly_report:{SSCROFA11}",
                "eva_contig_alias:CM000823.5",
                "eva_submitted_variants:rs5334475172",
            ],
        )
        self.assertEqual(model.calls, 5)
        self.assertEqual(
            [
                e["payload"]["errors"]
                for e in events
                if e["event_type"] == "ToolCallValidated"
            ],
            [[], *[["repeated_call"]] * 4],
        )


class RouteContractTests(unittest.TestCase):
    def test_every_run_is_offered_both_lookup_tools_and_neither_is_refused(
        self,
    ) -> None:
        """No Tool is bound to the Run's assembly (#469, owner RC2 1, ADR 0049)."""
        from types import SimpleNamespace

        from bio_neb.investigation.investigation_agent import (
            INVESTIGATION_TOOLS,
            investigation_tool_values,
        )
        from bio_neb.investigation.investigation_request import InvestigationRequestItem

        self.assertLessEqual(
            {"fetch_refsnp", "fetch_eva_variant"}, set(INVESTIGATION_TOOLS)
        )
        # The reading Tool's values are the member's unread words, so it reads a
        # member with words still to read (an untyped item's own words, a member with
        # none left waits); no other Tool reads one.
        reading = SimpleNamespace(
            request=InvestigationRequestItem(request_text="Investigate rs1."),
            raw_answers=(),
            target_reads=[],
            started_event_id="started",
        )
        from bio_neb.agentic_journey.action_policy import PreconditionUnmet
        from bio_neb.investigation.investigation_agent import (
            DECISION_TOOLS,
            MEMBER_HAS_NO_PAPER,
            READING_TOOLS,
        )

        bound = {"search_assemblies", *DECISION_TOOLS, *READING_TOOLS}
        for name in set(INVESTIGATION_TOOLS) - bound:
            with self.subTest(tool=name):
                deps = reading if name == "record_target_reading" else None
                self.assertEqual(investigation_tool_values(deps, name).constants, {})
        # The paper's reading Tools read the member's paper: a Run whose member no
        # paper proposed (the dependencies' default) has them wait with their
        # reason, never fail on the missing paper.
        from dataclasses import fields

        from bio_neb.investigation.investigation_agent import (
            InvestigationPlannerDependencies,
        )

        (paper_field,) = (
            f for f in fields(InvestigationPlannerDependencies) if f.name == "paper"
        )
        for name in READING_TOOLS:
            with self.subTest(tool=name):
                with self.assertRaises(PreconditionUnmet) as unmet:
                    investigation_tool_values(
                        SimpleNamespace(paper=paper_field.default), name
                    )
                self.assertEqual(unmet.exception.code, MEMBER_HAS_NO_PAPER.code)
        # The assembly search alone is bound: to the Run's organism, never to its
        # assembly.
        self.assertEqual(
            investigation_tool_values(
                SimpleNamespace(run_taxid=6239), "search_assemblies"
            ).constants,
            {"taxid": 6239},
        )
        # A decision Tool waits until a result of the current target asks for
        # it, and is callable once one does.
        asked = {
            "event_id": "lookup",
            "event_type": "RefSnpToolResultRecorded",
            "payload": {"result": {"status": "placement_unnamed"}},
        }
        for lookups, replaced, waits in (
            ([], frozenset(), True),
            ([asked], frozenset({"lookup"}), True),
            ([asked], frozenset(), False),
        ):
            deps = SimpleNamespace(source_events=lookups, replaced_result_ids=replaced)
            with self.subTest(lookups=len(lookups), replaced=bool(replaced)):
                if waits:
                    with self.assertRaises(PreconditionUnmet) as unmet:
                        investigation_tool_values(deps, "name_placement")
                    self.assertEqual(unmet.exception.code, "decision_none_asked")
                else:
                    self.assertEqual(
                        investigation_tool_values(deps, "name_placement").constants, {}
                    )
        self.assertEqual(
            investigation_tool_values(reading, "record_target_reading").enums,
            {"source_event_id": ("started",)},
        )

    def test_admission_admits_either_sources_lookup_tool_on_any_assembly(self) -> None:
        """Rule 2 B8 migrated (#375): which source to ask is the model's choice (#469,
        owner RC2 1, ADR 0049), so a pig Run's dbSNP lookup and a human Run's EVA lookup
        of the stated rsID are both admitted."""
        from types import SimpleNamespace

        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.investigation.investigation_agent import (
            investigation_action_errors,
        )
        from bio_neb.investigation.investigation_request import InvestigationRequestItem

        for assembly, tool in (
            (SSCROFA11, "fetch_refsnp"),
            ("GCF_000001405.40", "fetch_eva_variant"),
        ):
            with self.subTest(assembly=assembly, tool=tool):
                deps = SimpleNamespace(
                    investigation_input=SimpleNamespace(assembly_accession=assembly),
                    request=InvestigationRequestItem(
                        request_text="Investigate rs1.", rsid="rs1"
                    ),
                    researcher_answers=(),
                    raw_answers=(),
                    target_reads=[],
                    started_event_id="started",
                    source_events=[],
                )
                action = ProposedAgentAction(
                    tool_name=tool, tool_call_id="call-1", arguments={"rsid": "rs1"}
                )

                self.assertEqual(investigation_action_errors(deps, action), ())

    def test_locate_variant_names_the_variant_source_a_position_is_listed_by(
        self,
    ) -> None:
        """locate_variant's source is the model's choice (#469, owner Q1 a, ADR 0049):
        a position's call names one, a region's names none (it lists no rsID, #367), and
        EVA's own variant_type and contig_naming go only with EVA (R1 a, SP4 a)."""
        from types import SimpleNamespace

        from bio_neb.agentic_journey.action_policy import ProposedAgentAction
        from bio_neb.investigation.investigation_agent import (
            investigation_action_errors,
        )
        from bio_neb.investigation.investigation_request import (
            InvestigationRequestItem,
            StatedPosition,
            StatedRegion,
        )

        position = InvestigationRequestItem(
            request_text="Check chr12:3556401 GGTGTCCCCCTCGC>G.",
            position=StatedPosition(
                contig="chr12", position=3556401, ref="GGTGTCCCCCTCGC", alt="G"
            ),
        )
        region = InvestigationRequestItem(
            request_text="Check chr12:100-200.",
            region=StatedRegion(contig="chr12", start=100, end=200),
            normalized_question_action="observe_region",
        )
        for request, arguments, errors in (
            (position, {}, ("locate_variant_source_missing",)),
            (position, {"source": "eva"}, ()),
            (position, {"source": "eva", "variant_type": "INDEL"}, ()),
            (
                position,
                {"source": "ncbi_dbsnp", "variant_type": "SNV"},
                ("eva_option_without_eva",),
            ),
            (region, {}, ()),
            (region, {"source": "eva"}, ("region_names_no_source",)),
            (region, {"variant_type": "SNV"}, ("region_names_no_source",)),
            (position, {"source": "eva", "contig_naming": "REFSEQ"}, ()),
            (
                position,
                {"source": "ncbi_dbsnp", "contig_naming": "REFSEQ"},
                ("eva_option_without_eva",),
            ),
            (region, {"contig_naming": "REFSEQ"}, ("region_names_no_source",)),
        ):
            with self.subTest(arguments=arguments, region=request.region is not None):
                deps = SimpleNamespace(
                    investigation_input=SimpleNamespace(assembly_accession=SSCROFA11),
                    request=request,
                    researcher_answers=(),
                    raw_answers=(),
                    target_reads=[],
                    started_event_id="started",
                    source_events=[],
                )
                action = ProposedAgentAction(
                    tool_name="locate_variant",
                    tool_call_id="call-1",
                    arguments=arguments,
                )

                self.assertEqual(investigation_action_errors(deps, action), errors)

    def test_only_the_lookup_tools_call_a_variant_source(self) -> None:
        """No second lookup path: in `src/`, only the planner's two lookup Tools reach
        dbSNP's or EVA's rsID GETs, and nothing in `src/` imports lane tooling."""
        import re

        root = Path(__file__).resolve().parents[1] / "src" / "bio_neb"
        callers = {}
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if re.search(r"\.fetch_refsnp\(|look_up_eva_rsid\(|fetch_source\(", text):
                callers[str(path.relative_to(root))] = text
        self.assertEqual(
            set(callers),
            {
                "investigation/investigation_agent.py",
                "investigation/investigation_sources.py",
                "providers/eva_investigation.py",
                # A paper's download fetches through the same seam, no rsID GET.
                "providers/paper_sources.py",
                # Placement reads NCBI's G15 and resolver operations there too.
                "target_request/ncbi_sources.py",
                # The suggestion of another assembly checks the Run's known targets with
                # the Investigation's own lookup, through the same seam.
                "reference_discovery/assembly_suggestion.py",
                "reference_discovery/reference_tools.py",
            },
        )
        self.assertEqual(
            callers["reference_discovery/reference_tools.py"].count(
                "fetch_source(adapter, operation, key)"
            ),
            1,
        )
        self.assertEqual(
            callers["reference_discovery/assembly_suggestion.py"].count(
                "look_up_eva_rsid("
            ),
            1,
        )
        for no_rsid_lookup in (
            "providers/paper_sources.py",
            "target_request/ncbi_sources.py",
        ):
            self.assertIsNone(
                re.search(
                    r"fetch_refsnp|look_up_eva_rsid|\"eva_", callers[no_rsid_lookup]
                ),
                no_rsid_lookup,
            )
        agent = callers["investigation/investigation_agent.py"]
        # Each GET goes through the one source-fetch seam.
        self.assertEqual(
            agent.count('fetch_source(dependencies.adapter, "fetch_refsnp"'), 1
        )
        self.assertEqual(agent.count("look_up_eva_rsid("), 1)


class _TransportPigSources(PigSources):
    """Each recorded EVA answer passes the product's physical-contact seam."""

    def _eva(self, operation: str, key: str, url: str, body: bytes):
        from bio_neb.agentic_journey.source_contacts import (
            observe_source_transport,
            source_physical_attempt,
        )

        observe_source_transport()
        with source_physical_attempt(url, method="GET"):
            return super()._eva(operation, key, url, body)


class CompositeToolLimitTests(unittest.TestCase):
    """A count belongs to the Tool the Agent called, not its operations (#416)."""

    def _breached(self, directory, *, declared=None, failure=None):
        from unittest.mock import patch

        from tests.sequential_investigation_support import runtime_configuration
        from tests.source_limits_support import declaring

        configuration = runtime_configuration()
        if failure is not None:
            configuration["failure"] = failure
        investigation_input, assertion = pig_input()
        with (
            declaring(declared or {}),
            patch(
                "tests.sequential_investigation_support.runtime_configuration",
                return_value=configuration,
            ),
        ):
            projection, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                _TransportPigSources(),
                investigation_input=investigation_input,
                assertion=assertion,
            )
        return projection, read_evidence(journey)

    def test_a_composite_tools_breach_names_it_and_reads_back(self) -> None:
        from bio_neb.agentic_journey.source_contacts import source_contact_units
        from bio_neb.failure import ContactLimitOverrides

        for limit, settings in (
            (
                "contacts",
                {"declared": {"fetch_eva_variant": ContactLimitOverrides(contacts=1)}},
            ),
            ("run_contacts", {"failure": {"limits": {"run_contacts": 1}}}),
        ):
            with (
                self.subTest(limit=limit),
                tempfile.TemporaryDirectory() as directory,
            ):
                projection, events = self._breached(directory, **settings)
                (breach,) = [
                    e["payload"]
                    for e in events
                    if e["event_type"] == "SourceContactLimitReached"
                ]
                self.assertEqual(
                    (breach["limit"], breach["limits"]["surface"]["name"]),
                    (limit, "fetch_eva_variant"),
                )
                self.assertEqual(
                    sum(
                        e["event_type"] == "SourcePhysicalContactStarted"
                        for e in events
                    ),
                    1,
                )
                # Its operation's policy routed the refusal; the Run reads back.
                (refused,) = [
                    e["payload"]
                    for e in events
                    if e["event_type"] == "SourceOperationFailureRecorded"
                ]
                self.assertEqual(refused["failure_class"], "exhausted")
                self.assertEqual(
                    refused["policy"]["surface"]["name"], "eva_contig_alias"
                )
                self.assertEqual(projection.status, "operational_review_pending")
                source_contact_units(events)

    def test_a_count_breach_on_an_operation_surface_is_refused(self) -> None:
        import copy

        from bio_neb.agentic_journey.source_contacts import source_contact_units
        from bio_neb.failure import ContactLimitOverrides

        with tempfile.TemporaryDirectory() as directory:
            _projection, events = self._breached(
                directory,
                declared={"fetch_eva_variant": ContactLimitOverrides(contacts=1)},
            )
        hostile = copy.deepcopy(events)
        (breach,) = [
            e for e in hostile if e["event_type"] == "SourceContactLimitReached"
        ]
        breach["payload"]["limits"]["surface"]["name"] = "eva_submitted_variants"
        with self.assertRaisesRegex(ValueError, "lacks its exact open scope"):
            source_contact_units(hostile)


class HumanRouteTests(unittest.TestCase):
    def test_a_human_run_is_offered_both_and_its_dbsnp_choice_never_contacts_eva(
        self,
    ) -> None:
        """Both Tools on a human Run too (#469); the model chose dbSNP, so no EVA GET."""
        sources = SourceResponsesOnly()
        model = _OfferedTools()
        with tempfile.TemporaryDirectory() as directory:
            projection, _journey = run_attempt(Path(directory), model, sources)

        self.assertEqual(projection.status, "review_pending")
        for tools in model.offered:
            self.assertLessEqual({"fetch_refsnp", "fetch_eva_variant"}, tools)
        self.assertEqual(
            sources.operations,
            ["fetch_refsnp:rs429358", "fetch_assembly_report:GCF_000001405.40"],
        )


if __name__ == "__main__":
    unittest.main()
