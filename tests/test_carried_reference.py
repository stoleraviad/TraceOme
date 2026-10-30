"""Planning's reference facts carried into the Investigation (owner, #476 alignment G;
#479 Q4 A).

The Investigation's start records which of planning's retained GETs describe the Run's
assembly; an item on that assembly reads them with no new NCBI contact, each shown as
a cited prior source and checked against the cited GET on reopen. NCBI is contacted
only for other assemblies, or when planning holds no report.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.investigation.investigation import UserAssertion, project_investigation
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedPosition,
)
from bio_neb.providers.ncbi_datasets import assembly_report_url
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.refsnp_retained_bodies import RS429358
from tests.sequential_investigation_support import (
    ASSEMBLY_BODY,
    GRCH38_NAMES,
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    input_and_assertion,
    run_attempt,
)

RUN = "GCF_000001405.40"
NAMES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    f"{RUN}_GRCh38.p14/{RUN}_GRCh38.p14_assembly_report.txt"
)
PLANNING_GETS = (
    (assembly_report_url(RUN), ASSEMBLY_BODY, "application/json"),
    (NAMES_URL, GRCH38_NAMES, "text/plain"),
)


def _of(events, event_type):
    return [event for event in events if event["event_type"] == event_type]


class CarriedReferenceTests(unittest.TestCase):
    def test_the_runs_assembly_report_is_read_from_planning_with_no_contact(self):
        sources = SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358})
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                planning_gets=PLANNING_GETS,
            )
            events = read_evidence(journey)

        planning = _of(events, "ReferenceSourceRetrieved")
        (start,) = _of(events, "InvestigationStarted")
        # The start carries planning's two GETs of the Run's assembly, once.
        self.assertEqual(
            start["payload"]["run_reference_sources"],
            [event["event_id"] for event in planning],
        )
        # The item checked its rsID on the Run's assembly with no NCBI report GET.
        self.assertEqual(sources.operations, ["fetch_refsnp:rs429358"])
        self.assertTrue(result.pending_request.proposal.work_items)
        (carried,) = _of(events, "InvestigationSourceCarried")
        self.assertEqual(carried["payload"]["cited_event_id"], planning[0]["event_id"])
        (report,) = _of(events, "AssemblyReportToolResultRecorded")
        (answer,) = report["payload"]["source_answers"]
        # Shown as the cited prior source, the planning GET's own answer.
        self.assertEqual(answer["cited_event_id"], planning[0]["event_id"])
        self.assertEqual(answer["request_identity"], assembly_report_url(RUN))
        self.assertEqual(report["payload"]["source_event_id"], carried["event_id"])
        self.assertEqual(project_investigation(events), result)

    def test_reopen_checks_the_cited_prior_source(self):
        sources = SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358})
        with tempfile.TemporaryDirectory() as directory:
            _result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                planning_gets=PLANNING_GETS,
            )
            events = read_evidence(journey)

        index = next(
            i
            for i, event in enumerate(events)
            if event["event_type"] == "InvestigationSourceCarried"
        )
        names = next(
            event["event_id"]
            for event in events
            if event["event_type"] == "ReferenceSourceRetrieved"
            and event["payload"]["request_identity"] == NAMES_URL
        )
        hostile = copy.deepcopy(events)
        # The carried report claims planning's sequence-names GET instead.
        hostile[index]["payload"]["cited_event_id"] = names
        with self.assertRaisesRegex(ValueError, "carried source changed its cited GET"):
            project_investigation(hostile)
        start = next(
            i
            for i, event in enumerate(events)
            if event["event_type"] == "InvestigationStarted"
        )
        hostile = copy.deepcopy(events)
        hostile[start]["payload"]["run_reference_sources"] = [names]
        with self.assertRaisesRegex(ValueError, "carried reference facts"):
            project_investigation(hostile)

    def test_a_position_on_the_runs_assembly_searches_planning_names_first(self):
        # Planning's sequence names are the Run assembly's: searchable before any
        # lookup, and the position lookup reads both carried GETs.
        calls = iter(
            [
                ("search_sequence_names", {"assembly_accession": RUN, "text": "19"}),
                ("locate_variant", {"source": "ncbi_dbsnp"}),
            ]
        )

        async def respond(messages, info):
            name, arguments = next(
                calls,
                (
                    "request_investigation_information",
                    {
                        "question": "Which variant?",
                        "missing_information": "Stop here.",
                        "asks_about": "target",
                    },
                ),
            )
            return ModelResponse(
                parts=[ToolCallPart(name, json.dumps(arguments), tool_call_id=name)]
            )

        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="paper:doi:10.1000/example",
            requests=(
                InvestigationRequestItem(
                    request_text="The paper reports chr19:44908684.",
                    position=StatedPosition(
                        contig="chr19", position=44908684, ref="T", alt="C"
                    ),
                ),
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        sources = SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358})
        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
                planning_gets=PLANNING_GETS,
            )
            events = read_evidence(journey)

        (search,) = _of(events, "SequenceNamesRead")
        self.assertTrue(search["payload"]["result"]["rows"])
        self.assertNotIn(f"fetch_assembly_report:{RUN}", sources.operations)
        self.assertNotIn(f"ncbi_sequence_names:{RUN}", sources.operations)
        self.assertEqual(
            {
                e["payload"]["source"]["operation"]
                for e in _of(events, "InvestigationSourceCarried")
            },
            {"fetch_assembly_report", "ncbi_sequence_names"},
        )
        self.assertEqual(project_investigation(events), result)


class RunTaxonTests(unittest.TestCase):
    """The Run organism's taxon for `search_assemblies` (owner #479 Q4 A): the Run
    reference's recorded taxid, at species level when planning's taxon answer ranks
    it below species."""

    def events(self, rank, species=None):
        taxonomy = {"tax_id": 3702001, "rank": rank}
        if species is not None:
            taxonomy["classification"] = {"species": {"id": species}}
        return [
            {
                "event_id": "taxon",
                "event_type": "TaxonResolutionRecorded",
                "payload": {
                    "source_answers": [{"body": {"reports": [{"taxonomy": taxonomy}]}}]
                },
            },
            {
                "event_id": "description",
                "event_type": "ReferenceDescriptionRecorded",
                "payload": {
                    "result": {
                        "references": [
                            {
                                "assembly_accession": RUN,
                                "record": {"ncbi_tax_id": 3702001},
                            }
                        ]
                    }
                },
            },
        ]

    def test_a_strain_is_bound_at_its_species(self):
        from bio_neb.investigation.run_reference import run_species_taxid

        self.assertEqual(run_species_taxid(self.events("strain", 3702), RUN), 3702)

    def test_a_species_or_an_unnamed_species_keeps_the_recorded_taxid(self):
        from bio_neb.investigation.run_reference import run_species_taxid

        self.assertEqual(run_species_taxid(self.events("species"), RUN), 3702001)
        self.assertEqual(run_species_taxid(self.events("strain"), RUN), 3702001)
        self.assertIsNone(run_species_taxid(self.events("species")[:1], RUN))


if __name__ == "__main__":
    unittest.main()
