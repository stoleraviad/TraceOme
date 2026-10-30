"""A member's typed gene change placed in the Investigation's agentic loop (#473).

The model types a gene change from the researcher's words; `place_gene_change` reads it on
the Run's own annotation and copies every transcript reading, every candidate SNV and NCBI's
HGVS answer; the model picks one candidate (`pick_gene_change_candidate`), checked by
membership, and the member's lookup then reads that candidate as any position (owner Q1 a,
Q3 a, Q11; ADR 0049). A human and a non-human change run through the same Tools; NCBI's
HGVS service answers human sequences only, and its 400 is copied, never skipped.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

from bio_neb.failure import ProviderError
from bio_neb.investigation.investigation import (
    InvestigationInput,
    InvestigationVcfIdentity,
    UserAssertion,
    project_investigation,
)
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedChange,
)
from bio_neb.journey import canonical_json
from bio_neb.recording import read_evidence
from tests.gene_change_support import YEAST, YEAST_GFF_TEXT, dataset_report, gff_zip
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.sequential_investigation_support import (
    SourceResponsesOnly,
    input_and_assertion,
    run_attempt,
)

# APOE's MANE Select transcript as GRCh38's own GFF3 writes it (as in
# tests/test_gene_change_flow.py): one CDS segment 501-530, so codon 2 is 504-506, TGC.
APOE_GFF_TEXT = "\n".join(
    [
        "##gff-version 3",
        "NC_000019.10\tBestRefSeq\tgene\t501\t530\t.\t+\t.\tID=gene-APOE;Name=APOE;"
        "gene=APOE",
        "NC_000019.10\tBestRefSeq\tmRNA\t501\t530\t.\t+\t.\tID=rna-NM_000041.4;"
        "Parent=gene-APOE;gene=APOE;tag=MANE Select;transcript_id=NM_000041.4",
        "NC_000019.10\tBestRefSeq\tCDS\t501\t530\t.\t+\t0\tID=cds-NP_000032.1;"
        "Parent=rna-NM_000041.4;gene=APOE;protein_id=NP_000032.1;tag=MANE Select",
        "",
    ]
)
APOE_WORDS = "Does HG001 carry APOE c.4T>C (p.Cys2Arg) on NM_000041.4?"
APOE = StatedChange(
    gene="APOE",
    change="c.4T>C (p.Cys2Arg)",
    transcript="NM_000041.4",
    coding={"cds_position": 4, "ref": "T", "alt": "C"},
    protein={"ref": "C", "codon": 2, "alt": "R"},
)
CODING = {
    "seq_id": "NM_000041.4",
    "position": 3,
    "deleted_sequence": "T",
    "inserted_sequence": "C",
}
ON_RUN = {
    "seq_id": "NC_000019.10",
    "position": 503,
    "deleted_sequence": "T",
    "inserted_sequence": "C",
}
NOT_HUMAN = {"error": {"code": 400, "message": "The submitted sequence is not human."}}
_ASK = {
    "question": "Which target do you mean?",
    "missing_information": "The intended target.",
    "asks_about": "target",
}


class GeneChangeSources(SourceResponsesOnly):
    """The shared Investigation stand-in, with an assembly's annotation and NCBI's
    HGVS and equivalents answers, as `NcbiInvestigationSourceAdapter` returns them."""

    def __init__(self, gff_text, hgvs, equivalents=None, **options):
        super().__init__(**options)
        self.gff_text = gff_text
        self.hgvs = hgvs
        self.equivalents = equivalents or {}

    def fetch_genome_annotation(self, assembly_accession):  # noqa: ANN001, ANN201
        url = (
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            f"{assembly_accession}/download?include_annotation_type=GENOME_GFF"
        )
        answer = self._answer(
            "ncbi_genome_annotation",
            assembly_accession,
            url,
            200,
            gff_zip(self.gff_text),
        )
        return type(answer)(**{**answer.__dict__, "media_type": "application/zip"})

    def fetch_hgvs_contextuals(self, name):  # noqa: ANN001, ANN201
        status, body = self.hgvs[name]
        url = f"https://api.ncbi.nlm.nih.gov/variation/v0/hgvs/{name}/contextuals"
        return self._answer(
            "ncbi_hgvs_contextuals", name, url, status, canonical_json(body).encode()
        )

    def fetch_spdi_equivalents(self, spdi):  # noqa: ANN001, ANN201
        url = (
            f"https://api.ncbi.nlm.nih.gov/variation/v0/spdi/{spdi}/"
            "all_equivalent_contextual"
        )
        body = {"data": {"spdis": self.equivalents[spdi]}}
        return self._answer(
            "ncbi_spdi_equivalents", spdi, url, 200, canonical_json(body).encode()
        )


def _returns(messages):
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _recorded(part):
    """A Tool return showing its recorded fact's model view (#478 G1.3), not a
    refusal's view."""
    return isinstance(part.content, dict) and part.content.get("status") != "refused"


def _loop(pick=lambda candidates: candidates[0], locate=True):
    """A scripted model: place the member's gene change, pick a candidate the result
    lists, locate it as any position, then ask (the test's stand-in for the model)."""
    sent = []

    async def respond(messages, info):
        sent.append(info)
        # A Tool's recorded result; a refused call returns its refusal instead.
        returns = [part for part in _returns(messages) if _recorded(part)]
        done = {part.tool_name for part in returns}
        if "place_gene_change" not in done:
            name, arguments = "place_gene_change", {}
        elif "pick_gene_change_candidate" not in done:
            (placed,) = [
                part.content
                for part in returns
                if part.tool_name == "place_gene_change"
            ]
            candidate = pick(placed["result"]["candidates"])
            name, arguments = (
                ("request_investigation_information", _ASK)
                if candidate is None
                else (
                    "pick_gene_change_candidate",
                    {
                        "placement_event_id": placed["cite"]["placement_event_id"],
                        **{
                            key: candidate[key]
                            for key in ("contig", "position", "ref", "alt")
                        },
                        "reason": "The one candidate on the MANE Select transcript.",
                    },
                )
            )
        elif locate and "locate_variant" not in done:
            name, arguments = "locate_variant", {"source": "ncbi_dbsnp"}
        else:
            name, arguments = "request_investigation_information", _ASK
        return ModelResponse(
            parts=[ToolCallPart(name, arguments, tool_call_id=f"call-{len(sent)}")]
        )

    return SimpleNamespace(model=ScriptedModelEndpoint(respond))


def _run(directory, adapter, item, *, investigation_input=None, model=None):
    default_input, _ = input_and_assertion()
    selected = investigation_input or default_input
    assertion = UserAssertion.create_collection(
        asserted_by="researcher",
        requests=(item,),
        predecessor_ids=(selected.input_id,),
    )
    result, journey = run_attempt(
        Path(directory),
        model or _loop(),
        adapter,
        investigation_input=selected,
        assertion=assertion,
        invocation_overrides={"model_requests": 8},
    )
    return result, read_evidence(journey)


def _of(events, event_type):
    return [event["payload"] for event in events if event["event_type"] == event_type]


def _human(**options):
    return GeneChangeSources(
        APOE_GFF_TEXT,
        {"NM_000041.4:c.4T>C": (200, {"data": {"spdis": [CODING]}})},
        {"NM_000041.4:3:T:C": [CODING, ON_RUN]},
        assembly_body=canonical_json(
            dataset_report(
                "GCF_000001405.40", "GRCh38.p14", "hg38", "GCA_000001405.29", 9606
            )
        ).encode(),
        reference_bases={"NC_000019.10:503:3": "TGC", "NC_000019.10:503:1": "T"},
        **options,
    )


class HumanGeneChangeTests(unittest.TestCase):
    """The default case: a human Run (GRCh38, #141's HG001)."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = _human()
        with tempfile.TemporaryDirectory() as directory:
            cls.result, cls.events = _run(
                directory,
                cls.adapter,
                InvestigationRequestItem(request_text=APOE_WORDS, gene_change=APOE),
            )

    def test_the_placement_copies_the_annotations_candidates_and_ncbis_answer(self):
        (placed,) = _of(self.events, "InvestigationGeneChangePlaced")
        result = placed["result"]
        self.assertEqual(
            [
                (c["contig"], c["position"], c["ref"], c["alt"], c["transcripts"])
                for c in result["candidates"]
            ],
            [("NC_000019.10", 504, "T", "C", ["NM_000041.4"])],
        )
        [read] = result["transcripts"]
        self.assertEqual(
            (read["accession"], read["tags"], read["genetic_code_origin"]),
            ("NM_000041.4", ["MANE Select"], "default"),
        )
        hgvs = result["hgvs"]
        self.assertEqual(
            (hgvs["hgvs"], hgvs["status_code"], hgvs["on_run"], hgvs["candidates"]),
            (
                "NM_000041.4:c.4T>C",
                200,
                ["NC_000019.10:503:T:C"],
                ["NC_000019.10:503:T:C"],
            ),
        )
        # Every GET is retained and shown, in call order.
        self.assertEqual(
            [answer["operation"] for answer in placed["source_answers"]],
            [
                "fetch_assembly_report",
                "ncbi_genome_annotation",
                "ncbi_reference_span",
                "ncbi_hgvs_contextuals",
                "ncbi_spdi_equivalents",
                "ncbi_sequence_names",
            ],
        )

    def test_the_picked_candidate_is_the_members_placed_position_for_its_lookup(self):
        (picked,) = _of(self.events, "InvestigationGeneChangePicked")
        self.assertEqual(
            picked["placed"],
            {"contig": "NC_000019.10", "position": 504, "ref": "T", "alt": "C"},
        )
        # locate_variant reads the picked candidate as any placed position: the
        # Run's own reference base at 504, then dbSNP's listing there.
        (lookup,) = _of(self.events, "RefSnpToolResultRecorded")
        self.assertIn("ncbi_spdi_rsids:NC_000019.10:503:T:C", self.adapter.operations)
        location = lookup["result"]["location"]
        self.assertEqual(
            (location["placed"], location["stated"], lookup["query"]),
            (picked["placed"], None, "NC_000019.10:503:T:C"),
        )

    def test_the_journey_reopens_with_its_placement_and_pick(self):
        project_investigation(self.events)


class GeneChangeMembershipTests(unittest.TestCase):
    """The model's choices are checked by membership only (ADR 0049, #473 Q4 a, Q11):
    a candidate the placement did not list, or a transcript no result lists, is
    refused as its argument and the model repairs it."""

    def test_a_candidate_the_placement_does_not_list_is_refused(self):
        calls = []

        def pick(candidates):
            calls.append(candidates)
            listed = candidates[0]
            # First a candidate one base off, then the listed one.
            return (
                {**listed, "position": listed["position"] + 1}
                if len(calls) == 1
                else listed
            )

        # A refused pick returns no result, so the scripted model picks again.
        with tempfile.TemporaryDirectory() as directory:
            _result, events = _run(
                directory,
                _human(),
                InvestigationRequestItem(request_text=APOE_WORDS, gene_change=APOE),
                model=_loop(pick=pick, locate=False),
            )
        refused = [
            v["errors"]
            for v in _of(events, "ToolCallValidated")
            if v["tool_name"] == "pick_gene_change_candidate"
            and v["status"] == "invalid"
        ]
        self.assertEqual(refused, [["gene_change_candidate_not_listed"]])
        (picked,) = _of(events, "InvestigationGeneChangePicked")
        self.assertEqual(picked["placed"]["position"], 504)

    def test_a_transcript_is_named_from_a_result_and_its_candidate_picked(self):
        """The model's transcript pick (#473 Q4 a): a transcript no result of this
        member lists is refused; one it lists reads the change on it alone, and a
        candidate of that placement is picked for the member's gene change."""
        words = "Does HG001 carry APOE p.Cys2Arg?"
        change = StatedChange(
            gene="APOE",
            change="p.Cys2Arg",
            protein={"ref": "C", "codon": 2, "alt": "R"},
        )
        steps = iter(
            [
                ("place_gene_change", {}),
                ("place_gene_change", {"transcript": "NM_NOPE.1"}),
                ("place_gene_change", {"transcript": "NM_000041.4"}),
            ]
        )

        sent = []

        async def respond(messages, info):
            sent.append(info)
            step = next(steps, None)
            if step is None:
                placed = [
                    part.content
                    for part in _returns(messages)
                    if part.tool_name == "place_gene_change" and _recorded(part)
                ]
                picked = any(
                    part.tool_name == "pick_gene_change_candidate"
                    for part in _returns(messages)
                )
                candidate = placed[-1]["result"]["candidates"][0]
                step = (
                    ("request_investigation_information", _ASK)
                    if picked
                    else (
                        "pick_gene_change_candidate",
                        {
                            "placement_event_id": placed[-1]["cite"][
                                "placement_event_id"
                            ],
                            **{
                                key: candidate[key]
                                for key in ("contig", "position", "ref", "alt")
                            },
                            "reason": "The MANE Select transcript's one-base change.",
                        },
                    )
                )
            name, arguments = step
            return ModelResponse(
                parts=[ToolCallPart(name, arguments, tool_call_id=f"t-{len(sent)}")]
            )

        adapter = _human()
        adapter.hgvs["NP_000032.1:p.C2R"] = (400, {"error": {"message": "unread"}})
        with tempfile.TemporaryDirectory() as directory:
            _result, events = _run(
                directory,
                adapter,
                InvestigationRequestItem(request_text=words, gene_change=change),
                model=SimpleNamespace(model=ScriptedModelEndpoint(respond)),
            )
        refused = [
            v["errors"]
            for v in _of(events, "ToolCallValidated")
            if v["tool_name"] == "place_gene_change" and v["status"] == "invalid"
        ]
        self.assertEqual(refused, [["gene_change_transcript_not_listed"]])
        first, named = _of(events, "InvestigationGeneChangePlaced")
        self.assertEqual(
            (first["result"]["hgvs"], named["result"]["claim"]["change"]["transcript"]),
            (None, "NM_000041.4"),
        )
        # Both place the member's own change; the pick reads the named placement.
        self.assertEqual(first["gene_change"], named["gene_change"])
        (picked,) = _of(events, "InvestigationGeneChangePicked")
        self.assertEqual(picked["placed"]["position"], 504)
        project_investigation(events)

    def test_a_pick_moved_to_another_candidate_is_refused_at_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            _result, events = _run(
                directory,
                _human(),
                InvestigationRequestItem(request_text=APOE_WORDS, gene_change=APOE),
                model=_loop(locate=False),
            )
        moved = json.loads(json.dumps(events))
        (event,) = [
            e for e in moved if e["event_type"] == "InvestigationGeneChangePicked"
        ]
        event["payload"]["placed"]["position"] = 505
        with self.assertRaisesRegex(ValueError, "names no candidate"):
            project_investigation(moved)


class PickBindingTests(unittest.TestCase):
    """A retained pick is its own admitted call's, and yields to the researcher's
    newer intent (ADR 0034 call binding; #473 review r1 ST3, SP1)."""

    def test_a_pick_moved_to_another_listed_candidate_is_refused_at_reopen(self):
        # C2S: TGC to AGC, TCC or TCT, so the placement lists several candidates.
        change = StatedChange(
            gene="APOE", change="C2S", protein={"ref": "C", "codon": 2, "alt": "S"}
        )
        with tempfile.TemporaryDirectory() as directory:
            _result, events = _run(
                directory,
                _human(),
                InvestigationRequestItem(
                    request_text="Does HG001 carry APOE C2S?", gene_change=change
                ),
                model=_loop(locate=False),
            )
        project_investigation(events)
        moved = json.loads(json.dumps(events))
        (placed,) = [
            e for e in moved if e["event_type"] == "InvestigationGeneChangePlaced"
        ]
        (pick,) = [
            e for e in moved if e["event_type"] == "InvestigationGeneChangePicked"
        ]
        other = next(
            {key: c[key] for key in ("contig", "position", "ref", "alt")}
            for c in placed["payload"]["result"]["candidates"]
            if {key: c[key] for key in ("contig", "position", "ref", "alt")}
            != pick["payload"]["placed"]
        )
        pick["payload"]["placed"] = other
        with self.assertRaisesRegex(ValueError, "names no candidate"):
            project_investigation(moved)

    def _answers(self, picks, *answers):
        from bio_neb.investigation.investigation_request import typed_answers

        item = InvestigationRequestItem(
            request_text="Does HG001 carry rs429358, APOE c.4T>C?",
            rsid="rs429358",
            gene_change=APOE,
        )
        return item, typed_answers(item, answers, (), picks)

    PICK = {
        "gene_change": APOE.model_dump(mode="json"),
        "placed": {"contig": "NC_000019.10", "position": 504, "ref": "T", "alt": "C"},
        "after_answer_event_ids": [],
    }

    def test_a_pick_keeps_the_researchers_stated_rsid(self):
        from bio_neb.investigation.investigation_request import (
            effective_location,
            researcher_rsids,
        )

        item, answers = self._answers((self.PICK,))
        self.assertEqual(researcher_rsids(item, answers), ("rs429358",))
        self.assertEqual(
            effective_location(item, answers).placed.model_dump(),
            self.PICK["placed"],
        )

    def test_a_pick_never_displaces_a_site_the_researcher_states(self):
        """A model pick does not amend the researcher's target (ADR 0038, 0049; #473
        review r2 SP1): with a stated site, the site is the member's location."""
        from bio_neb.investigation.investigation_request import (
            effective_location,
            typed_answers,
        )

        item = InvestigationRequestItem(
            request_text="Does HG001 carry APOE c.4T>C at NC_000019.10:600 T>C?",
            position={
                "contig": "NC_000019.10",
                "position": 600,
                "ref": "T",
                "alt": "C",
            },
            gene_change=APOE,
        )
        location = effective_location(item, typed_answers(item, (), (), (self.PICK,)))
        self.assertEqual((location.placed, location.stated.position), (None, 600))

    def test_a_member_stating_its_own_site_is_not_placed(self):
        """The placement Tool is refused for a member that states a site, and the
        member's lookup reads the stated site, never the annotation (#473 review r2
        SP1, r3 ST4)."""
        words = "Does HG001 carry APOE c.4T>C at NC_000019.10:600 T>C?"
        site = {"contig": "NC_000019.10", "position": 600, "ref": "T", "alt": "C"}
        # The ordinary lookup's sequence_unnamed result asks for the written contig's
        # sequence, named with name_sequence and read by the next call.
        steps = iter(
            [
                ("place_gene_change", {}),
                ("locate_variant", {"source": "ncbi_dbsnp"}),
                (
                    "name_sequence",
                    lambda messages: {
                        "refsnp_event_id": _latest_cite(messages),
                        "assembly_accession": "GCF_000001405.40",
                        "record": "NC_000019.10",
                        "quote": "NC_000019.10",
                    },
                ),
                ("locate_variant", {"source": "ncbi_dbsnp"}),
            ]
        )
        sent = []

        async def respond(messages, info):
            sent.append(info)
            name, arguments = next(steps, ("request_investigation_information", _ASK))
            if callable(arguments):
                arguments = arguments(messages)
            return ModelResponse(
                parts=[ToolCallPart(name, arguments, tool_call_id=f"s-{len(sent)}")]
            )

        adapter = _human()
        adapter.reference_bases["NC_000019.10:599:1"] = "T"
        with tempfile.TemporaryDirectory() as directory:
            _result, events = _run(
                directory,
                adapter,
                InvestigationRequestItem(
                    request_text=words, position=site, gene_change=APOE
                ),
                model=SimpleNamespace(model=ScriptedModelEndpoint(respond)),
            )
        refused = [
            v["errors"]
            for v in _of(events, "ToolCallValidated")
            if v["tool_name"] == "place_gene_change"
        ]
        self.assertEqual(refused, [["gene_change_site_stated"]])
        self.assertEqual(_of(events, "InvestigationGeneChangePlaced"), [])
        lookup = _of(events, "RefSnpToolResultRecorded")[-1]
        # The location keeps the stated site and holds no placed one.
        location = lookup["result"]["location"]
        self.assertEqual(
            (location["stated"]["position"], location.get("placed")), (600, None)
        )
        self.assertIn("ncbi_reference_span:NC_000019.10:599:1", adapter.operations)
        self.assertFalse(
            any(op.startswith("ncbi_genome_annotation:") for op in adapter.operations)
        )

    def test_a_newer_answer_outranks_an_earlier_pick(self):
        from bio_neb.investigation.investigation_request import effective_location

        newer = {
            "action": "clarify",
            "instruction": "I meant chr19:44908684.",
            "target": {"position": {"contig": "chr19", "position": 44908684}},
            "response_event_id": "answer-2",
        }
        item, answers = self._answers((self.PICK,), newer)
        location = effective_location(item, answers)
        self.assertEqual((location.placed, location.stated.position), (None, 44908684))


class GeneChangeFailureTests(unittest.TestCase):
    """A placement's source fault is routed once by the failure module, at the GET or
    reading its answer, and opens the member's scoped Review; the Journey reopens with
    the retained answer beside its fault (ADR 0031, #473 review r1 ST1, ST2, SP4)."""

    def _failed(self, adapter):
        with tempfile.TemporaryDirectory() as directory:
            result, events = _run(
                directory,
                adapter,
                InvestigationRequestItem(request_text=APOE_WORDS, gene_change=APOE),
            )
        project_investigation(events)
        self.assertEqual(result.status, "operational_review_pending")
        self.assertEqual(_of(events, "InvestigationGeneChangePlaced"), [])
        (failed,) = _of(events, "InvestigationSourceContactFailed")
        return failed, _of(events, "SourceOperationFailureRecorded")

    def test_an_annotation_that_is_not_a_zip_is_a_semantic_fault(self):
        adapter = _human()
        fetch = adapter.fetch_genome_annotation

        def malformed(accession):
            answer = fetch(accession)
            return type(answer)(**{**answer.__dict__, "body": b"not a zip"})

        adapter.fetch_genome_annotation = malformed
        failed, failures = self._failed(adapter)
        self.assertEqual(failed["operation"], "ncbi_genome_annotation")
        self.assertEqual(
            [(f["failure_class"], f["original_failure_class"]) for f in failures],
            [("semantic", "semantic")],
        )

    def test_an_hgvs_answer_that_does_not_read_is_a_semantic_fault(self):
        adapter = _human()
        adapter.hgvs["NM_000041.4:c.4T>C"] = (200, {"data": {"spdis": [{"x": 1}]}})
        failed, failures = self._failed(adapter)
        self.assertEqual(failed["operation"], "ncbi_hgvs_contextuals")
        self.assertEqual([f["failure_class"] for f in failures], ["semantic"])

    def test_a_rejected_contact_is_classified_once(self):
        adapter = _human()

        def rejected(_accession):
            raise ProviderError("rejected", "annotation access rejected")

        adapter.fetch_genome_annotation = rejected
        _failed, failures = self._failed(adapter)
        self.assertEqual(
            [(f["failure_class"], f["original_failure_class"]) for f in failures],
            [("rejected", "rejected")],
        )

    def test_an_exhausted_transient_keeps_its_original_class(self):
        adapter = _human()

        def busy(_accession):
            raise ProviderError("transient", "NCBI answered 503.")

        adapter.fetch_genome_annotation = busy
        _failed, failures = self._failed(adapter)
        self.assertEqual(
            (failures[-1]["failure_class"], failures[-1]["original_failure_class"]),
            ("exhausted", "transient"),
        )


class NonHumanGeneChangeTests(unittest.TestCase):
    """The non-default case: a yeast Run's own annotation (#469, every organism alike).
    NCBI's HGVS service answers its transcript name 400 "not human": the answer is
    copied and the annotation's candidates stand (owner 2026-10-28: not human-only)."""

    @classmethod
    def setUpClass(cls):
        yeast_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/yeast.vcf.gz",
                byte_count=1234,
                sha256="b" * 64,
            ),
            sample_id="S288C",
            assembly_accession=YEAST,
        )
        adapter = GeneChangeSources(
            YEAST_GFF_TEXT,
            {"NM_1.3:c.6C>A": (400, NOT_HUMAN)},
            assembly_body=canonical_json(
                dataset_report(YEAST, "R64", "sacCer3", "GCA_000146045.2", 559292)
            ).encode(),
            reference_bases={"NC_001134.8:1003:3": "TAC"},
        )
        # Codon 2 is TAC at 1004-1006: c.6C>A makes TAA, a stop.
        words = "Does S288C carry PBP2 c.6C>A (p.Tyr2*) on NM_1.3?"
        change = StatedChange(
            gene="PBP2",
            change="c.6C>A (p.Tyr2*)",
            transcript="NM_1.3",
            coding={"cds_position": 6, "ref": "C", "alt": "A"},
            protein={"ref": "Y", "codon": 2, "alt": "*"},
        )
        with tempfile.TemporaryDirectory() as directory:
            cls.result, cls.events = _run(
                directory,
                adapter,
                InvestigationRequestItem(request_text=words, gene_change=change),
                investigation_input=yeast_input,
                model=_loop(),
            )

    def test_ncbis_not_human_answer_is_copied_beside_the_annotations_candidates(self):
        (placed,) = _of(self.events, "InvestigationGeneChangePlaced")
        hgvs = placed["result"]["hgvs"]
        self.assertEqual(
            (hgvs["status_code"], json.loads(hgvs["body"]), hgvs["on_run"]),
            (400, NOT_HUMAN, []),
        )
        self.assertEqual(placed["result"]["outcome"], "candidates")
        self.assertIn(
            400,
            [answer["status_code"] for answer in placed["source_answers"]],
        )
        (picked,) = _of(self.events, "InvestigationGeneChangePicked")
        self.assertEqual(picked["placed"]["contig"], "NC_001134.8")
        # The pick reaches the member's lookup as any placed position (Q11).
        (lookup,) = _of(self.events, "RefSnpToolResultRecorded")
        self.assertEqual(lookup["result"]["location"]["placed"], picked["placed"])
        project_investigation(self.events)


def _latest_cite(messages) -> str:
    """The cite of the latest lookup result a scripted model read."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    parts = [
        part
        for message in ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        for part in message["parts"]
        if part["part_kind"] == "tool-return"
        and part["tool_name"] == "locate_variant"
        and isinstance(part["content"], dict)
        and "cite" in part["content"]
    ]
    return parts[-1]["content"]["cite"]["refsnp_event_id"]


if __name__ == "__main__":
    unittest.main()
