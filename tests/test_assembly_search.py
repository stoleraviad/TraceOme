"""The Investigation's assembly search and the stated version the model names from it
(#471, owner Q3 a, R3 a; ADR 0049): a search is NCBI's own answer, bound to its member
and call on reopen; an answered error reaches the model; a member's searches survive its
continuation; a version's quote holds the version (review r1 SP1, SP2, ST1, ST2)."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.agentic_journey.effort import (
    ExtendAgentEffortResponse,
    project_effort_review,
    respond_effort_review,
)
from bio_neb.investigation.investigation import (
    UserAssertion,
    VersionChoice,
    project_investigation,
)
from bio_neb.investigation.investigation_agent import version_naming
from bio_neb.investigation.investigation_assignment import investigation_assignment
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedAssembly,
    StatedPosition,
)
from bio_neb.providers.ncbi_datasets import genome_report_url
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.sequential_investigation_support import (
    SourceResponsesOnly,
    input_and_assertion,
    run_attempt,
    run_taxon,
)

# The TaxId is the Run organism's, bound by code: never sent.
GRCH37_SEARCH = {
    "search_text": ["GRCh37"],
    "assembly_version": "all_assemblies",
}


def _position(contig, position, ref=None, alt=None, *, rsid=None, text=None):
    return InvestigationRequestItem(
        request_text=text or f"The paper reports {contig}:{position}.",
        rsid=rsid,
        position=StatedPosition(contig=contig, position=position, ref=ref, alt=alt),
    )


def _assertion(request: InvestigationRequestItem):
    investigation_input, _ = input_and_assertion()
    return investigation_input, UserAssertion.create_collection(
        asserted_by="paper:doi:10.1000/example",
        requests=(request,),
        predecessor_ids=(investigation_input.input_id,),
    )


def _scripted(calls):
    """A model that makes ``calls`` in order, keeping each request it was sent."""
    seen: list[str] = []

    async def respond(messages, info):
        seen.append(json.dumps([repr(message) for message in messages]))
        name, arguments = calls[len(seen) - 1]
        if callable(arguments):
            arguments = arguments(messages)
        return ModelResponse(
            parts=[ToolCallPart(name, arguments, tool_call_id=f"search-{len(seen)}")]
        )

    return SimpleNamespace(model=ScriptedModelEndpoint(respond)), seen


class _AnsweredError(SourceResponsesOnly):
    """NCBI answering a query it cannot run (synthetic: a plain-text 400) and
    #466's recorded GRCh37 search to the corrected query."""

    def fetch_taxon_assemblies(self, taxon_query):
        if "sort.field" not in taxon_query:
            return super().fetch_taxon_assemblies(taxon_query)
        return self._answer(
            "ncbi_taxon_assemblies",
            taxon_query,
            genome_report_url(taxon_query),
            400,
            b"Bad request: the requested sort field is not supported.",
        )


class VersionQuoteTests(unittest.TestCase):
    def deps(self, cited_line):
        request = _position("chr19", 44908684, "T", "C").model_copy(
            update={
                "stated_assembly": StatedAssembly(name="GRCh38", cited_line=cited_line)
            }
        )
        return SimpleNamespace(
            request=request,
            researcher_answers=(),
            raw_answers=(),
            target_reads=[],
            started_event_id="started",
            assembly_searches=[],
            source_events=[],
            investigation_input=SimpleNamespace(assembly_accession="GCF_000001405.40"),
        )

    def test_a_version_quote_holds_the_stated_version(self):
        """The quote stands in the line that states the version and holds the
        version as written: a span of that line without it cites no version
        (ADR 0049; review r1 SP1)."""
        deps = self.deps("Methods: aligned on GRCh38.")
        refused = version_naming(
            deps, VersionChoice(assembly_accession="GCF_000001405.40", quote="Methods")
        )
        named = version_naming(
            deps,
            VersionChoice(assembly_accession="GCF_000001405.40", quote="on GRCh38"),
        )

        self.assertEqual(refused, "version_quote_ungrounded")
        self.assertEqual(
            (named.assembly_accession, named.quote, named.match),
            ("GCF_000001405.40", "on GRCh38", "model_matched"),
        )


def _latest_cite(messages) -> str:
    """The cite of the latest lookup result the scripted model read."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    returns = [
        part["content"]
        for message in ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        for part in message["parts"]
        if part["part_kind"] == "tool-return"
        and part["tool_name"] == "locate_variant"
        and isinstance(part["content"], dict)
        and "cite" in part["content"]
    ]
    return returns[-1]["cite"]["refsnp_event_id"]


@run_taxon(9606)
class SearchBoundaryTests(unittest.TestCase):
    def test_an_answered_search_error_reaches_the_model_and_a_corrected_query_runs(
        self,
    ):
        """A 400 NCBI answers is its answer, not a mechanical fault: the search is
        recorded with its status and no rows, its body reaches the next model
        request, and the model's corrected query runs (ADR 0049; review r1 ST2)."""
        model, seen = _scripted(
            [
                ("search_assemblies", {**GRCH37_SEARCH, "sort_field": "nothing"}),
                ("search_assemblies", GRCH37_SEARCH),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                model,
                _AnsweredError(),
                invocation_overrides={"model_requests": 2},
            )
            searches = [
                event["payload"]["result"]
                for event in read_evidence(journey)
                if event["event_type"] == "AssembliesSearched"
            ]

        self.assertEqual(result.status, "effort_review_pending")
        self.assertEqual(
            [(page["status_code"], len(page["rows"])) for page in searches],
            [(400, 0), (200, 6)],
        )
        self.assertIn("the requested sort field is not supported", seen[1])

    def test_a_search_is_bound_to_its_member_call_and_source_on_reopen(self):
        """A search belongs to its exact member, call and GET: substituting any of
        them is refused when the Journey is read again (ADR 0034; review r1 ST1)."""
        model, _ = _scripted([("search_assemblies", GRCH37_SEARCH)])
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                model,
                SourceResponsesOnly(),
                invocation_overrides={"model_requests": 1},
            )
            events = read_evidence(journey)

        self.assertEqual(project_investigation(events).status, result.status)
        # Each substitution is refused for its own relationship: the search's member
        # and call by the Investigation reader's binding to its admitted call, its
        # GET by the shared reader's retained-artifact identity.
        call_binding = "^assembly search changed its member, query or call identity$"
        for field, replacement, reason in (
            ("collection_member_id", "another-member", call_binding),
            ("tool_call_id", "another-call", call_binding),
            (
                "source_event_id",
                str(uuid.uuid4()),
                "^observed Tool execution changed its retained artifact identity$",
            ),
        ):
            with self.subTest(field=field):
                hostile = copy.deepcopy(events)
                search = next(
                    e for e in hostile if e["event_type"] == "AssembliesSearched"
                )
                search["payload"][field] = replacement
                with self.assertRaisesRegex(ValueError, reason):
                    project_investigation(hostile)

    def test_a_members_searches_survive_its_continuation(self):
        """After an effort extension the member names a version from the search its
        earlier Attempt retained, with no second listing GET (review r1 SP2; the
        one item history, #479 G2.3)."""
        model, _ = _scripted(
            [
                ("search_assemblies", GRCH37_SEARCH),
                ("locate_variant", {"source": "ncbi_dbsnp"}),
                (
                    "name_version",
                    lambda messages: {
                        "refsnp_event_id": _latest_cite(messages),
                        "assembly_accession": "GCF_000001405.25",
                        "quote": "GRCh37",
                    },
                ),
            ]
        )
        request = _position(
            "chr19",
            45411941,
            "T",
            "C",
            rsid="rs429358",
            text="rs429358 at chr19:45411941 T>C on GRCh37.",
        ).model_copy(
            update={
                "stated_assembly": StatedAssembly(
                    name="GRCh37", cited_line="Methods: GRCh37"
                )
            }
        )
        investigation_input, assertion = _assertion(request)
        sources = SourceResponsesOnly()
        with tempfile.TemporaryDirectory() as directory:
            stopped, journey = run_attempt(
                Path(directory),
                model,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
                invocation_overrides={"model_requests": 1},
            )
            self.assertEqual(stopped.status, "effort_review_pending")
            pending = project_effort_review(read_evidence(journey)).pending_request
            receipt = respond_effort_review(
                journey,
                pending.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=2),
            )
            investigation_assignment(
                model_factory=model.model, source_adapter_factory=lambda: sources
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                effort_review_response_event_id=receipt.response_event_id,
            )
            validation = next(
                e["payload"]
                for e in read_evidence(journey)
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_call_id"] == "search-3"
            )

        self.assertEqual(validation["errors"], [])
        self.assertEqual(
            sum(op.startswith("ncbi_taxon_assemblies") for op in sources.operations),
            1,
        )


if __name__ == "__main__":
    unittest.main()
