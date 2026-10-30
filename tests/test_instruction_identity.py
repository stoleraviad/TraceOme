"""Public role instructions and their retained Attempt provenance."""

import copy
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb.agentic_journey.agent_roles import (
    role_instructions,
)
from bio_neb.agentic_journey.instruction_evidence import read_attempt_instructions
from bio_neb.agentic_journey.planning_assignment import (
    planning_assignment,
)
from bio_neb.agentic_journey.role_instructions import stage_text
from bio_neb.application.queries import project_journey_events
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.input_correction_fakes import prepare_input_correction
from tests.proposal_information_fakes import InspectionModel, registered_run


class InstructionIdentityTests(unittest.TestCase):
    def test_an_archived_or_unknown_version_and_an_unknown_role_are_unknown(self):
        # Only the selected versions ship: an archived version is unknown, as one
        # never published is; the archive manifest keeps the retired files' bytes
        # and digests.
        for role, version in (
            ("planning_agent", "v1"),
            ("planning_agent", "v8"),
            ("planning_agent", "v9"),
            ("planning_agent", "v10"),
            ("planning_agent", "v11"),
            ("planning_agent", "v12"),
            ("investigation_planner", "v10"),
            ("investigation_planner", "v12"),
            ("investigation_planner", "v13"),
            ("investigation_planner", "v14"),
            ("investigation_planner", "v15"),
            ("investigation_planner", "v16"),
            ("investigation_planner", "v17"),
            ("investigation_planner", "v18"),
            ("investigation_planner", "v19"),
            ("investigation_planner", "v20"),
            ("investigation_planner", "v21"),
            ("investigation_planner", "v22"),
            ("investigation_planner", "v23"),
            ("planning_agent", "v13"),
            ("planning_agent", "v14"),
            ("planning_agent", "v15"),
            ("planning_agent", "v16"),
            ("planning_agent", "v17"),
            ("planning_agent", "v18"),
            ("planning_agent", "v19"),
            ("planning_agent", "v20"),
            ("planning_agent", "v21"),
            ("report_explainer", "v6"),
            ("report_explainer", "v10"),
            ("report_explainer", "v11"),
            ("report_explainer", "v12"),
            ("report_explainer", "v13"),
            ("report_explainer", "v14"),
            ("report_explainer", "v15"),
            ("report_explainer", "v16"),
            ("report_explainer", "v17"),
            ("report_explainer", "v18"),
            ("report_explainer", "v19"),
            ("paper_reader", "v3"),
            ("paper_reader", "v12"),
            ("paper_reader", "v13"),
            ("planning_agent", "v999999"),
        ):
            for form in ("qwen_3_5", "nemotron_3_5_super"):
                with (
                    self.subTest(role=role, version=version, form=form),
                    self.assertRaisesRegex(
                        ValueError, f"^unknown instruction version: {role}:{version}$"
                    ),
                ):
                    role_instructions(role, version=version, form=form)
        with self.assertRaisesRegex(ValueError, "unknown instruction role"):
            role_instructions("../unknown")

    def test_public_lookup_exposes_planning_assignment_and_retires_inspection_role(
        self,
    ):
        selected = role_instructions("planning_agent")
        self.assertEqual(selected.identity, "planning_agent:v22:nemotron_3_5_super")
        self.assertEqual(
            hashlib.sha256(selected.content).hexdigest(), _NEMOTRON_PLANNING
        )
        # v9: each registered request file is cited by its own item.
        self.assertIn(
            "for each file record one scientific_investigation", selected.text
        )
        # v18: the target-less item is for a request with no file; beside a file,
        # its own item stands for the targets.
        self.assertIn(
            "When the request names no target and the context lists no "
            "registered_requests",
            selected.text,
        )
        self.assertIn(
            "beside a file record no other to_find scientific_investigation",
            selected.text,
        )
        self.assertIn("currently exposed capabilities", selected.text)
        self.assertIn("Never dispatch execution", selected.text)
        # v11: a lookup's source_answers are NCBI's own answer.
        self.assertIn("Each lookup also returns source_answers", selected.text)
        # v12: a site names the version its quote writes, and a target answer
        # is read in its own stage.
        self.assertIn("stated_version names the version exactly", selected.text)
        self.assertIn(
            "record the targets it names with record_answered_targets", selected.text
        )
        # The typed-slot schema: no nullable-argument advice, no quote extraction.
        self.assertNotIn("JSON null", selected.text)
        self.assertNotIn("exact_text", selected.text)
        for slot_example in ('"value": 9606', '"state": "to_find"', "search_journey"):
            self.assertIn(slot_example, selected.text)
        # The reference stage's split Tools with agent-chosen terms (K4).
        for tool in (
            "attest_sample",
            "resolve_taxon",
            "discover_assemblies",
            "inspect_reference",
            "describe_companion",
        ):
            self.assertIn(tool, selected.text)
        self.assertNotIn("discover_references", selected.text)
        self.assertIn("A lookup never changes the recorded choice", selected.text)
        for retired in ("registered_input_inspector", "execution_planning_agent"):
            with self.assertRaisesRegex(ValueError, "unknown instruction role"):
                role_instructions(retired)

    def test_selected_planning_text_teaches_both_forms_and_forbids_placeholders(self):
        # A site the request states, even only by coordinates, is asserted with
        # its quote as its value; an rsID the request does not give is to_find,
        # never a placeholder.
        text = role_instructions("planning_agent").text
        # The site is typed, never the quote its value.
        self.assertIn(
            '{"state": "asserted", "quote": "V:2129317 C>T", "site": {"contig": "V", '
            '"position": 2129317, "ref": "C", "alt": "T"}}',
            text,
        )
        self.assertIn(
            '{"state": "asserted", "value": "rs429358", "quote": "rs429358"}', text
        )
        self.assertIn("an rsID the request does not give is to_find", text)
        self.assertIn("never a placeholder", text)
        self.assertNotIn("leave the value out", text)
        self.assertNotIn("omit", text.lower())

    def test_selected_planning_text_teaches_a_non_human_case_and_no_output(self):
        # A non-human example beside the human one, before the selected text is
        # frozen; BAM + VCF is fixed, so no output is taught, and the targets
        # come from the request or its registered file.
        text = role_instructions("planning_agent").text
        for human, other in (
            ("HG001", "MM-04"),
            ('"value": 9606', '"value": 10090'),
            ("GCF_000001405.40", "GCF_000001635.27"),
        ):
            with self.subTest(human=human):
                self.assertIn(human, text)
                self.assertIn(other, text)
        self.assertNotIn("proposal_output", text)
        self.assertIn("registered_request", text)
        self.assertIn("When the request names no target", text)

    def test_selected_investigation_text_shows_a_multi_alt_and_a_no_record_case(self):
        # A recorded planner stop followed a schema that marks the locus fields
        # optional and a v5 text with no worked case; each example v6 shows must be
        # a draft the planner's output type accepts, in every model's words.
        for form in ("qwen_3_5", "nemotron_3_5_super"):
            with self.subTest(form=form):
                self._investigation_text_teaches_its_cases(form)

    def _investigation_text_teaches_its_cases(self, form):
        from bio_neb.investigation.investigation import InvestigationPlanDraft

        selected = role_instructions("investigation_planner", form=form)
        self.assertEqual(selected.identity, f"investigation_planner:v24:{form}")
        # v11: a placed position is looked up, an unplaced one is not.
        self.assertIn("user_assertion.placement.position", selected.text)
        # v21 (G1.1): every Tool's one answer shape, and the Tools a request
        # lists as callable now and waiting.
        self.assertIn("Every Tool answers in one shape", selected.text)
        self.assertIn("callable_now", selected.text)
        decoder = json.JSONDecoder()
        examples = []
        marker = "compile_proposal {"
        start = selected.text.find(marker)
        while start != -1:
            example, _end = decoder.raw_decode(selected.text, start + len(marker) - 1)
            examples.append(example)
            start = selected.text.find(marker, start + 1)
        unresolved, region, complete = examples
        for example in examples:
            InvestigationPlanDraft.model_validate_json(json.dumps(example))
            # The model re-types nothing code holds: an example states the outcome
            # and the results it cites, nothing copied.
            self.assertLessEqual(
                set(example), {"outcome", "refsnp_event_id", "assembly_event_id"}
            )
        self.assertEqual(unresolved["outcome"], "unresolved")
        self.assertEqual(
            (region["outcome"], "assembly_event_id" in region), ("complete", False)
        )
        self.assertEqual(
            (complete["outcome"], "assembly_event_id" in complete), ("complete", True)
        )
        self.assertIn("three candidate Work items", selected.text)
        self.assertIn(
            "fetch_eva_variant (the European Variation Archive)", selected.text
        )
        # A stated position is placed first, and the intent check compares what the
        # request states, from the variant source the model chooses.
        self.assertIn(
            "call locate_variant first, naming in source the variant source you choose",
            selected.text,
        )
        # Both lookup Tools on every Run, the model choosing; a source's no record
        # leads it to the other; no route remains in the text.
        self.assertIn("Both lookup Tools are offered on every Run", selected.text)
        self.assertIn("ask the other source within your budget", selected.text)
        self.assertNotIn("follows the Run's assembly", selected.text)
        self.assertIn("locate_variant itself follows a dbSNP merge", selected.text)
        self.assertIn(
            "a stated position, its alleles and its genome version must be the "
            "source's",
            selected.text,
        )
        # Every listed allele is a candidate; the retired typed outcome is no
        # longer named.
        self.assertIn("(SNV, MNV, insertion or deletion)", selected.text)
        self.assertNotIn("unsupported_alleles", selected.text)
        # A result is cited by its view's cite, its stage_findings id.
        self.assertIn(
            "the lookup result's cite (its source_event_id in stage_findings)",
            selected.text,
        )

    def test_completed_attempt_retains_its_readable_identity_and_exact_instruction_bytes(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = registered_run(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            input_set_id = next(
                e["payload"]["input_set_id"]
                for e in journey.read()
                if e["event_type"] == "RunInputsRegistered"
            )
            model = InspectionModel(input_set_id)
            completed = planning_assignment(model_factory=model.model).run(
                prepare_input_correction(journey),
                environ={"LOCAL_MODEL_KEY": "offline"},
            )
            self.assertEqual(completed.status, "inspection_succeeded")
            start = next(
                e for e in journey.read() if e["event_type"] == "AttemptStarted"
            )
            self.assertEqual(
                start["payload"]["instruction_identity"],
                "planning_agent:v22:qwen_3_5",
            )
            self.assertEqual(len(start["artifact_references"]), 1)
            reference = start["artifact_references"][0]
            retained = read_attempt_instructions(
                journey, start["payload"]["attempt_id"]
            )
            self.assertEqual(retained.identity, "planning_agent:v22:qwen_3_5")
            self.assertEqual(journey.read_artifact(reference), retained.content)
            self.assertEqual(reference["source_event_id"], start["event_id"])
            self.assertTrue(reference["artifact_id"].startswith("sha256:"))
            self.assertEqual(
                retained, role_instructions("planning_agent", form="qwen_3_5")
            )
            self.assertEqual(
                completed.attempts[0].instruction_identity,
                "planning_agent:v22:qwen_3_5",
            )
            with patch(
                "bio_neb.agentic_journey.role_instructions.role_instructions",
                side_effect=AssertionError(
                    "retained readback consulted today's selection"
                ),
            ):
                self.assertEqual(
                    read_attempt_instructions(
                        Journey.open(Path(receipt.journey_file)),
                        start["payload"]["attempt_id"],
                    ),
                    retained,
                )
            for mutation in (
                "missing_identity",
                "different_role",
                "different_role_without_form",
                "same_role_without_form",
                "empty_form",
                "missing_artifact",
                "wrong_owner",
            ):
                changed = copy.deepcopy(read_evidence(journey))
                attempt = next(
                    e for e in changed if e["event_type"] == "AttemptStarted"
                )
                if mutation == "missing_identity":
                    del attempt["payload"]["instruction_identity"]
                elif mutation == "different_role":
                    attempt["payload"]["instruction_identity"] = (
                        "investigation_planner:v1:plain"
                    )
                elif mutation == "different_role_without_form":
                    attempt["payload"]["instruction_identity"] = (
                        "investigation_planner:v1"
                    )
                elif mutation == "same_role_without_form":
                    # A 2-part identity is refused.
                    attempt["payload"]["instruction_identity"] = "planning_agent:v9"
                elif mutation == "empty_form":
                    attempt["payload"]["instruction_identity"] = "planning_agent:v8:"
                elif mutation == "missing_artifact":
                    attempt["artifact_references"] = []
                else:
                    attempt["artifact_references"][0]["source_event_id"] = changed[0][
                        "event_id"
                    ]
                with (
                    self.subTest(mutation=mutation),
                    self.assertRaisesRegex(ValueError, "instruction"),
                ):
                    project_journey_events(changed, journey_file=receipt.journey_file)

    def test_a_retained_identity_needs_its_form_but_never_its_installed_version(self):
        """A 2-part identity is refused; a retired form or an archived version
        still reads, since a retained Attempt reads its own artifact (#441, owner
        Q1 and Q4 R2)."""
        from bio_neb.agentic_journey.instruction_evidence import (
            retained_instruction_reference,
        )

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = registered_run(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            input_set_id = next(
                e["payload"]["input_set_id"]
                for e in journey.read()
                if e["event_type"] == "RunInputsRegistered"
            )
            completed = planning_assignment(
                model_factory=InspectionModel(input_set_id).model
            ).run(
                prepare_input_correction(journey),
                environ={"LOCAL_MODEL_KEY": "offline"},
            )
            self.assertEqual(completed.status, "inspection_succeeded")
            start = next(
                e for e in read_evidence(journey) if e["event_type"] == "AttemptStarted"
            )
        reference = retained_instruction_reference(start)

        def recorded(identity):
            changed = copy.deepcopy(start)
            changed["payload"]["instruction_identity"] = identity
            return changed

        # Recorded Journeys recorded v8 and v10 under the retired form name
        # nemotron.
        for identity in (
            "planning_agent:v8:nemotron",
            "planning_agent:v9:plain",
            "planning_agent:v1:nemotron_3_5_super",
        ):
            with self.subTest(identity=identity):
                self.assertEqual(
                    retained_instruction_reference(recorded(identity)), reference
                )
        for identity in (
            "planning_agent:v9",
            "planning_agent:v9:",
            "planning_agent:v9:3_5",
            "planning_agent:v9:Plain",
            "planning_agent:v9:qwen_3_5:extra",
        ):
            with (
                self.subTest(identity=identity),
                self.assertRaisesRegex(
                    ValueError,
                    "Attempt instruction identity requires its exact role and artifact",
                ),
            ):
                retained_instruction_reference(recorded(identity))

    def test_planning_and_investigation_requests_retain_their_fixed_instruction_prefix(
        self,
    ):
        from tests.application_fakes import _ready_journey
        from tests.sequential_investigation_support import (
            SourceResponsesOnly,
            _AdaptiveInvestigationModel,
            run_attempt,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner_root = root / "runner"
            investigation_root = root / "investigation"
            runner_root.mkdir()
            investigation_root.mkdir()
            receipt, _ = _ready_journey(runner_root)
            _, investigation = run_attempt(
                investigation_root, _AdaptiveInvestigationModel(), SourceResponsesOnly()
            )
            seen = set()
            for journey in (Journey.open(Path(receipt.journey_file)), investigation):
                events = journey.read()
                for start in (e for e in events if e["event_type"] == "AttemptStarted"):
                    role = start["payload"]["agent_role"]
                    seen.add(role)
                    retained = read_attempt_instructions(
                        journey, start["payload"]["attempt_id"]
                    )
                    self.assertEqual(
                        retained.identity,
                        {
                            "planning_agent": "planning_agent:v22:qwen_3_5",
                            "investigation_planner": "investigation_planner:v24:qwen_3_5",
                        }[role],
                    )
                    self.assertEqual(retained, role_instructions(role, form="qwen_3_5"))
                    requests = [
                        e
                        for e in events
                        if e["event_type"] == "ModelRequestRecorded"
                        and e["payload"]["attempt_id"] == start["payload"]["attempt_id"]
                    ]
                    self.assertTrue(requests)
                    for request in requests:
                        messages = request["payload"]["compatible_adapter_evidence"][
                            "request"
                        ]["messages"]
                        rendered = [
                            m["instructions"] for m in messages if m.get("instructions")
                        ][-1]
                        # Each request reads the retained text's shared part and
                        # its own stage's section (ADR 0044).
                        self.assertTrue(
                            rendered.startswith(
                                stage_text(retained.text, request["payload"]["stage"])
                            ),
                            role,
                        )
            self.assertEqual(
                seen,
                {"planning_agent", "investigation_planner"},
            )


_PUBLISHED = Path(__file__).parents[1] / "src/bio_neb/agentic_journey/instructions"
# The selected versions' rendered bytes in each model's form (Nemotron's text became the block
# source and Qwen's the overlay). Every pull that selects a version pins it here; a role with
# one text for every model pins the same digest in every form. v12: v11's text with one
# sentence and one example on a site's stated_version, and the target_answer stage's block (v11
# added a lookup's source_answers). v13 rendered v12's text: only select_recipe_candidate's
# purpose changed. v14: v13's text with reference_lookups for NCBI's own query, the copied page
# and the model's choice of a row with the researcher's words. v15: v14's text with a
# coordinate site typed (site) in asserted_values, the examples and target_answer, and the
# sample name and supplied accession typed with their quote in reference_lookups. v16: v15's
# text with a gene and its change typed (gene_change) in the example, target_answer and the
# record_answered_targets purpose. v17: v16's text with each request's callable_now and waiting
# Tools named in place of offered_tools. v18: v17's text with the reference stage's line on a
# cited paper (find_paper, a confirming question when several rows fit or one disagrees,
# retrieve_paper). v19: v18's text with the plan stage's cited_papers block (find and retrieve
# a cited paper first, then declare it), validation's cited_paper line and the paper stage's
# paper_lookups block; reference_lookups loses its paper line. v20: v19's text with
# reference_lookups' sentence on a value to find in a cited paper, taken from its reading. v21:
# v20's text with three sentences developed in parallel on v17: the "request names no target"
# item only where no registered request file is listed, a file's item standing for its targets;
# the targets sentence keeps a cited paper's target beside a file. v22: v21's text with a
# companion_lookups paragraph developed in parallel on v17: the genomic FASTA's Galaxy index
# work (search_prebuilt_indexes, read_prebuilt_index and record_prebuilt_index_choice) before
# the companions, an index fitted on its assembled molecules, a read against another assembly,
# suggest_assembly_with_index, and an Execution Review revise reopening the choice.
_NEMOTRON_PLANNING = "23d01a1d91849ff14cf7d46948fb3c2b93e46457db3652d2933c77a64dbb8c81"
# v20: v19's text with the gene change typed in target_reading, gene_change_lookups (place_gene_change,
# then pick_gene_change_candidate) and the two Tools named in assignment. v19: v18's text with
# target_reading (record_target_reading reads a member's words first) and asks_about in
# researcher_questions. v18: v17's text with the version_unnamed sentence (search_assemblies, the
# version argument). v17: v16's text with both lookup Tools offered on every Run, the model choosing
# the source from their metadata, locate_variant's source and EVA's variant_type, pinned from this head
# (v16, added copied placements, placement_unnamed, listed_none, follow-ups a source's answer names,
# EVA's contig_naming and a verdict-free source_answered, v15, naming a written contig's sequence, v14,
# a source Tool's source_answers). v21: v20's text with every Tool's one answer shape, the request's
# callable_now and waiting Tools, and compile_proposal as the outcome and its cited results only. v22:
# v21's text with the target question citing the lookup that places the target elsewhere, each lookup
# decision its own Tool, the bound TaxId and the replaced_target message. v23: v22's text with a
# body-less listing answer cited by its status and a position with neither alleles nor an rsID deferred
# out of the Run. v24: v23's text with paper_target (a member a paper's reading proposed receives the
# paper and reads it with the reading Tools).
_NEMOTRON_INVESTIGATION = (
    "3bd331c0240a83363519912a7fe9de61c71ae082a028c9cb2894a519b2b34c7b"
)
_ONE_TEXT = {
    # v4, v10 and v12 render the text v3, v9 and v11 rendered: only the shared
    # researcher question's purpose was added.
    ("table_reader", "v4"): (
        "6a23cad781ce9fd52bb05cc9d7d4b51403a2e27abfa2c5a5c95c3b2b99d507bf"
    ),
    # v20: v19's text with the combination writing the report's title. v19: v18's text with a paper's request named
    # as the paper's, never the researcher's. v18 reads each target's plain words and covers every result of a
    # multi-request conclusion: a new text, pinned from this head. v17 listed each genotype with its target and
    # quote.
    ("report_explainer", "v20"): (
        "c7e886b14e4ec68da34ec5e3069d232fb6529549dcb88a4a8a6802a460bc5a1c"
    ),
    # Lane P's paper reader, selected at the pull of M that brought it; main's v9 at
    # the last pull before PR 1. v13 (Q7) adds the paper item's organism and the
    # 50,000-character read; v14 names the ReadingFindingJudged findings
    # search_journey finds.
    ("paper_reader", "v14"): (
        "1cc92553654b8f3bd5dd338310cca4f02334a6b17887b57f3b29ebeaf405c478"
    ),
}
_SELECTED_DIGESTS = {
    ("planning_agent", "v22", "qwen_3_5"): (
        "0cb717f13f947b5e94e20e835ab07af4ef72177a16157911c58ca9263b99459e"
    ),
    ("planning_agent", "v22", "nemotron_3_5_super"): _NEMOTRON_PLANNING,
    ("planning_agent", "v22", "nemotron_3_ultra"): _NEMOTRON_PLANNING,
    ("investigation_planner", "v24", "qwen_3_5"): (
        "00709393c46d305ecd6889f23d25e8437f6f1209adac25be23d70c8793cc40bf"
    ),
    ("investigation_planner", "v24", "nemotron_3_5_super"): _NEMOTRON_INVESTIGATION,
    ("investigation_planner", "v24", "nemotron_3_ultra"): _NEMOTRON_INVESTIGATION,
    **{
        (role, version, form): digest
        for (role, version), digest in _ONE_TEXT.items()
        for form in ("qwen_3_5", "nemotron_3_5_super", "nemotron_3_ultra")
    },
}


# The words a model reads outside the instruction text, by role and form: each
# message sentence as the role fills it and every function Tool's purpose. Pinned
# from what the Python constants and registry rows gave before these words moved
# into the instruction blocks, so no model reads a new word.
_NEMOTRON_WORDING = {
    # Since then: planning's target_answer_task message and the record_answered_targets
    # purpose (a target answer goes back to planning); the ploidy judgment in
    # select_recipe_candidate's purpose; rewritten discover_assemblies,
    # record_reference_choice and request_planning_hitl purposes; the gene change in
    # record_answered_targets' purpose; the find_paper and retrieve_paper purposes and
    # the paper named in request_planning_hitl's; the subject_key in find_paper's and
    # the job stated in the paper stage; a cited paper's kept words in
    # record_reference_choice's; the three Galaxy index Tools' purposes; rewritten
    # read_prebuilt_index and record_prebuilt_index_choice purposes and
    # suggest_assembly_with_index's.
    "planning_agent": "e673f3a7c25fb5b94ce242727b08197e75d9482f2b6f615e93a18ecf121c16e6",
    # Since then: locate_variant's purpose rewritten and search_sequence_names' added;
    # the fetch_refsnp, fetch_eva_variant and locate_variant purposes rewritten, then
    # again (what each source keeps, locate_variant's source) with the rsID job
    # sentence; the version argument in locate_variant's purpose and search_assemblies'
    # purpose; the gene change in record_target_reading's purpose, with
    # place_gene_change's and pick_gene_change_candidate's added; a body-less listing
    # answer's status in locate_variant's and quote_listed_none's purposes; the four
    # reading Tools' purposes.
    "investigation_planner": (
        "0a39a2aeff130e42ad5c4a6335e61abcb5a307a222dbd12c6327aedbef170b0b"
    ),
}
_ONE_WORDING = {
    # Each declared job sentence: the table reader's and the explainer's as
    # their Python task texts stated them, the paper reader's as approved for
    # it. Each also states the shared researcher question's purpose, in its
    # approved words.
    "table_reader": "68d13945cf036ae90423dcbf6e0f760de8995099a30b8831cfb71da860441668",
    # The explainer's two job sentences, rewritten for the why and conclusion,
    # name targets and positions.
    "report_explainer": (
        "62f21f7371a42fb775ae92229677de0e11ca3eb0520fb1532a177d4f0d98964a"
    ),
    # Read_paper's purpose states the 50,000-character read.
    "paper_reader": "d7255816532d3a195b70ed9c71c8a47fbdc45d43c402cd2922b770d9d0866aab",
}
_WORDING_DIGESTS = {
    ("planning_agent", "qwen_3_5"): (
        # The same two new sentences as Nemotron's, and the same purposes.
        "fb97e8c620a30665f6f482ee2741d0caf57908f274f267c5b02ee34a11ae72ea"
    ),
    ("investigation_planner", "qwen_3_5"): (
        # The same purposes as Nemotron's, the decision Tools' purposes and the
        # replaced_target message, and the reading Tools' purposes.
        "4d024b0de0e13b6fe763b83cd24c0269ae362fdd8220a34ffd25719859bed2a5"
    ),
    **{
        (role, form): digest
        for role, digest in _NEMOTRON_WORDING.items()
        for form in ("nemotron_3_5_super", "nemotron_3_ultra")
    },
    **{
        (role, form): digest
        for role, digest in _ONE_WORDING.items()
        for form in ("qwen_3_5", "nemotron_3_5_super", "nemotron_3_ultra")
    },
}


def _shown_wording(selected):
    """Each message sentence as its role fills it, and every Tool's purpose."""
    from bio_neb.agentic_journey.tool_execution import function_tool_contracts
    from bio_neb.investigation.member_context import member_job as _member_job

    messages = {}
    if selected.agent_role == "planning_agent":
        messages = {
            name: selected.message(name)
            for name in ("objective", "job", "correction_task")
        }
    if selected.agent_role == "investigation_planner":
        messages = {
            "job_positioned": _member_job(selected, positioned=True),
            "job_rsid": _member_job(selected, positioned=False),
            "next_step": selected.message(
                "next_step", assembly="GCF_X", lookup="lookup_X"
            ),
            # The replaced target's last lookup, marked replaced.
            "replaced_target": selected.message(
                "replaced_target",
                by="BY_X",
                call="CALL_X",
                result="RESULT_X",
                status="STATUS_X",
            ),
        }
    if not messages:
        # A role whose job has no fields states each declared job as written.
        from bio_neb.agentic_journey.role_declaration import role

        messages = {
            name: selected.message(name)
            for name in role(selected.agent_role).jobs.values()
        }
    purposes = {
        contract.tool_name: selected.tool_purpose(contract.tool_name)
        for contract in function_tool_contracts(selected.agent_role)
    }
    return json.dumps({"messages": messages, "purposes": purposes}, sort_keys=True)


class InstructionFormTests(unittest.TestCase):
    def test_each_selected_role_renders_its_pinned_bytes_in_every_models_form(self):
        from bio_neb.agentic_journey.role_declaration import ROLES
        from bio_neb.agentic_journey.role_instructions import (
            _FORMS,
            instruction_blocks,
        )

        # A newly selected version or form without a pin fails here, not in a Run.
        self.assertEqual(
            set(_SELECTED_DIGESTS),
            {
                (role, version, form)
                for role, version in (
                    (name, item.instruction_version) for name, item in ROLES.items()
                )
                for form in _FORMS
            },
        )
        self.assertEqual(
            set(_WORDING_DIGESTS),
            {(role, form) for role, _version, form in _SELECTED_DIGESTS},
        )
        for (role, version, form), digest in _SELECTED_DIGESTS.items():
            with self.subTest(role=role, form=form):
                selected = role_instructions(role, form=form)
                self.assertEqual(selected.identity, f"{role}:{version}:{form}")
                self.assertEqual(hashlib.sha256(selected.content).hexdigest(), digest)
                self.assertEqual(
                    hashlib.sha256(_shown_wording(selected).encode()).hexdigest(),
                    _WORDING_DIGESTS[(role, form)],
                )
                blocks = instruction_blocks(role, version)
                names = [block.name for block in blocks]
                self.assertEqual(len(names), len(set(names)), names)
                stages = re.findall(
                    r"^\[stage ([a-z_]+)\]$", selected.text, re.MULTILINE
                )
                self.assertEqual(
                    stages, list(dict.fromkeys(b.stage for b in blocks if b.stage))
                )

    def test_message_and_tool_blocks_never_enter_the_instruction_text(self):
        from bio_neb.agentic_journey.role_instructions import instruction_blocks

        # A message sentence or a Tool's purpose reaches the model in its own
        # place, never in the recorded instructions.
        for role in ("planning_agent", "investigation_planner", "paper_reader"):
            selected = role_instructions(role)
            fed = [b for b in instruction_blocks(role) if b.use is not None]
            with self.subTest(role=role):
                self.assertTrue(fed)
                for block in fed:
                    self.assertNotIn(block.text.strip(), selected.text)

    def test_an_unknown_form_is_refused(self):
        # The retired form names plain and nemotron render nothing new.
        for form in ("unlisted", "plain", "nemotron"):
            with (
                self.subTest(form=form),
                self.assertRaisesRegex(
                    ValueError, f"^unknown instruction form: {form}$"
                ),
            ):
                role_instructions("planning_agent", form=form)

    def test_a_block_source_that_breaks_its_layout_is_refused(self):
        from bio_neb.agentic_journey.role_instructions import instruction_blocks

        def block(name, stage=None):
            own = f'stage = "{stage}"\n' if stage else ""
            return f'[[block]]\nname = "{name}"\n{own}text = """\nx\n"""\n'

        cases = {
            "repeated name": (block("a") + block("a"), "repeats a block name"),
            "shared after a stage": (
                block("a", "s") + block("b"),
                "shared block after a stage",
            ),
            "split stage": (
                block("a", "s") + block("b", "t") + block("c", "s"),
                "splits a stage's blocks",
            ),
            "unknown field": (block("a") + 'tag = "x"\n', "unknown field"),
        }
        for case, (source, reason) in cases.items():
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "planning_agent.v8.blocks.toml"
                path.write_text(source, encoding="utf-8")
                with (
                    self.subTest(case=case),
                    patch(
                        "bio_neb.agentic_journey.role_instructions._published",
                        lambda _name, path=path: path,
                    ),
                    self.assertRaisesRegex(ValueError, reason),
                ):
                    instruction_blocks("planning_agent", "v8")

    def test_each_model_reads_the_form_named_after_it_and_unlisted_has_none(self):
        from bio_neb.agentic_journey.model_settings import (
            BUILD_MODEL_REGISTRY,
            NEMOTRON_SUPER_VL_ENDPOINT_POLICY,
            NEMOTRON_ULTRA_ENDPOINT_POLICY,
            QWEN_ENDPOINT_POLICY,
            ModelUnlisted,
            model_endpoint_policy,
        )

        # Forms name their model.
        for policy, form in (
            (QWEN_ENDPOINT_POLICY, "qwen_3_5"),
            (model_endpoint_policy("Qwen/Qwen3.5-397B-A17B"), "qwen_3_5"),
            (NEMOTRON_ULTRA_ENDPOINT_POLICY, "nemotron_3_ultra"),
            (
                model_endpoint_policy("nvidia/Nemotron-3-Ultra-550b-a55b"),
                "nemotron_3_ultra",
            ),
            (NEMOTRON_SUPER_VL_ENDPOINT_POLICY, "nemotron_3_5_super"),
            (
                model_endpoint_policy("nvidia/Nemotron-3.5-Super-VL"),
                "nemotron_3_5_super",
            ),
        ):
            with self.subTest(model=policy.settings_selection.model):
                self.assertEqual(policy.instruction_form, form)
        # A render that names no form reads the build's default model's.
        self.assertEqual(
            BUILD_MODEL_REGISTRY.default, NEMOTRON_SUPER_VL_ENDPOINT_POLICY
        )
        self.assertEqual(role_instructions("paper_reader").form, "nemotron_3_5_super")
        # No fallback form stands in for an unlisted model.
        with self.assertRaises(ModelUnlisted) as refused:
            model_endpoint_policy("unlisted/model")
        self.assertEqual(refused.exception.model, "unlisted/model")

    def test_an_attempt_reads_and_records_its_models_form(self):
        from dataclasses import replace

        from bio_neb.agentic_journey import model_settings

        # The Run's Qwen entry names another form: the Attempt reads that form's
        # instructions and its job sentence, and records the form it read.
        qwen = model_settings.QWEN_ENDPOINT_POLICY
        with (
            tempfile.TemporaryDirectory() as directory,
            model_settings.use_model_registry(
                model_settings.model_registry().replacing(
                    replace(qwen, instruction_form="nemotron_3_ultra")
                )
            ),
        ):
            receipt, _ = registered_run(Path(directory))
            journey = Journey.open(Path(receipt.journey_file))
            input_set_id = next(
                e["payload"]["input_set_id"]
                for e in journey.read()
                if e["event_type"] == "RunInputsRegistered"
            )
            completed = planning_assignment(
                model_factory=InspectionModel(input_set_id).model
            ).run(
                prepare_input_correction(journey),
                environ={"LOCAL_MODEL_KEY": "offline"},
            )
            self.assertEqual(completed.status, "inspection_succeeded")
            events = journey.read()
            start = next(e for e in events if e["event_type"] == "AttemptStarted")
            self.assertEqual(
                start["payload"]["instruction_identity"],
                "planning_agent:v22:nemotron_3_ultra",
            )
            retained = read_attempt_instructions(
                journey, start["payload"]["attempt_id"]
            )
            self.assertEqual(retained.form, "nemotron_3_ultra")
            self.assertEqual(
                hashlib.sha256(retained.content).hexdigest(), _NEMOTRON_PLANNING
            )
            published = role_instructions("planning_agent", form="qwen_3_5").content
            self.assertNotEqual(retained.content, published)
            requests = [
                e
                for e in events
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] == start["payload"]["attempt_id"]
            ]
            self.assertTrue(requests)
            for request in requests:
                messages = request["payload"]["compatible_adapter_evidence"]["request"][
                    "messages"
                ]
                rendered = [
                    m["instructions"] for m in messages if m.get("instructions")
                ][-1]
                # Each request reads the recorded form's shared part and its stage.
                self.assertTrue(
                    rendered.startswith(
                        stage_text(retained.text, request["payload"]["stage"])
                    )
                )


def _route_lines(text):
    """Each per-kind line of a routes list: kinds, states and the shapes it teaches."""
    shapes = re.compile(
        r'acquisition_route "([a-z_]+)", validation \{"kind": "([a-z_]+)"'
    )
    for line in text.splitlines():
        head = re.match(
            r"- ([a-z_, ]+): ((?:asserted|observed|to_find)[a-z_ ]*),", line
        )
        if head and shapes.search(line):
            yield (
                [kind.strip() for kind in head.group(1).split(",")],
                line,
                shapes.findall(line),
            )


class InstructionLoadingTests(unittest.TestCase):
    def test_rendering_a_role_twice_reads_its_files_once(self):
        # Defaults and rendered instructions load once per process: published
        # files are immutable, so a second rendering reads none.
        from bio_neb.agentic_journey import role_instructions as module

        module.role_instructions.cache_clear()
        self.addCleanup(module.role_instructions.cache_clear)
        with patch.object(module, "_published", wraps=module._published) as reads:
            first = module.role_instructions("planning_agent", form="qwen_3_5")
            once = reads.call_count
            second = module.role_instructions("planning_agent", form="qwen_3_5")
        self.assertGreater(once, 0)
        self.assertEqual(reads.call_count, once)
        self.assertEqual(second, first)


class ModelFormTests(unittest.TestCase):
    """The block source holds Nemotron's words; Qwen's form text replaces its
    named blocks (owner, #439 A4)."""

    def test_qwen_form_changes_only_its_named_proposal_blocks(self):
        from bio_neb.agentic_journey.role_instructions import instruction_blocks

        nemotron = role_instructions("planning_agent", form="nemotron_3_5_super")
        qwen = role_instructions("planning_agent", form="qwen_3_5")
        self.assertEqual(qwen.identity, "planning_agent:v22:qwen_3_5")
        self.assertEqual(
            role_instructions("planning_agent", form="nemotron_3_ultra").content,
            nemotron.content,
        )
        self.assertNotEqual(qwen.content, nemotron.content)
        # Reference and companion requests read exactly the same text.
        for stage in ("reference", "companions", "execution_planning"):
            with self.subTest(stage=stage):
                self.assertEqual(
                    stage_text(qwen.text, stage), stage_text(nemotron.text, stage)
                )
        # Nemotron reads every instruction block as written; Qwen every one it
        # does not rewrite, in order.
        blocks = [b for b in instruction_blocks("planning_agent") if b.use is None]
        rewritten = {
            "asserted_values",
            "to_find_values",
            "validation",
        }
        position = 0
        for block in blocks:
            with self.subTest(block=block.name):
                self.assertIn(block.text, nemotron.text)
                found = qwen.text.find(block.text, position)
                if block.name in rewritten:
                    self.assertEqual(block.stage, "proposal_requirements")
                    self.assertEqual(found, -1)
                else:
                    self.assertGreaterEqual(found, position)
                    position = found + len(block.text)
        # The roles with no words of their own read one text in every form; the
        # explainer's v8 is one text for both models.
        for role in ("table_reader", "report_explainer", "paper_reader"):
            with self.subTest(role=role):
                own = role_instructions(role, form="qwen_3_5")
                self.assertEqual(own.identity.rsplit(":", 1)[1], "qwen_3_5")
                self.assertEqual(own.content, role_instructions(role).content)

    def test_explainer_steps_read_their_own_sections(self):
        # A request's member reads the claim rule and the lookup kept apart from
        # what it found; the combination reads only its own call.
        from bio_neb.agentic_journey.role_instructions import stage_text

        selected = role_instructions("report_explainer")
        self.assertEqual(selected.identity, "report_explainer:v20:nemotron_3_5_super")
        member = stage_text(selected.text, "report_explanation")
        combination = stage_text(selected.text, "report_combination")
        # Each target shows its claim; the model explains it, never writes it,
        # and is not asked which target a qualification applies to.
        self.assertIn("Each target shows the claim its recorded state reads as", member)
        self.assertNotIn("apply to target", member)
        self.assertIn(
            "the VCF was searched at this position and had no record there", member
        )
        # A member says why its request ended so, naming what was looked up
        # and what the source or the VCF answered; the combination's
        # conclusion sits under the counts the report shows.
        self.assertIn("what was looked up, and what the source or the VCF", member)
        # Each target's words say what happened; a source is named as asked only
        # when they or its evidence show its answer.
        self.assertIn("its words saying what happened to it", member)
        self.assertIn("When they show a source answered", member)
        self.assertIn("say that no source was asked", member)
        # A conclusion over several requests covers every result and leads with no
        # single target.
        self.assertIn("it covers every result", combination)
        self.assertIn("with no single target leading", combination)
        self.assertIn("counts of each result above your conclusion", combination)
        self.assertIn("what new evidence would change the answer", combination)
        # The combination titles the readable report.
        self.assertIn("Give the report a short plain title", combination)
        self.assertNotIn("title", member)
        self.assertNotIn("record_combined_answer", member)
        self.assertIn("Call record_combined_answer once", combination)
        self.assertNotIn("record_explanation", combination)
        for text in (member, combination):
            # The task says the request and the position searched, never a row
            # or Work item, and no word ban is asked for: ADR 0049 retires it.
            self.assertNotRegex(text, r"(?i)\brows?\b|work items?")
            self.assertNotIn("Never write the words coverage", text)
            self.assertIn("the report's own plain words", text)
            # Each genotype the text writes is listed with its target and the
            # words that write it, checked by membership (ADR 0049).
            self.assertIn("list each genotype", text.replace("List each", "list each"))

    def test_nemotron_investigation_form_treats_an_unlisted_lookup_as_admissible(self):
        # A recorded C. elegans Journey: Qwen's v10 told the model to check an assembly
        # report "only after a resolved lookup result" and to return unresolved when
        # "not resolved", while admission takes an unlisted lookup, with its assembly
        # report, as complete. Nemotron read it literally.
        from bio_neb.agentic_journey.role_instructions import instruction_blocks
        from bio_neb.investigation.investigation import ADMISSIBLE_LOOKUP_STATUSES

        qwen = role_instructions("investigation_planner", form="qwen_3_5")
        nemotron = role_instructions("investigation_planner", form="nemotron_3_5_super")
        self.assertEqual(
            nemotron.identity, "investigation_planner:v24:nemotron_3_5_super"
        )
        self.assertEqual(
            hashlib.sha256(nemotron.content).hexdigest(), _NEMOTRON_INVESTIGATION
        )
        self.assertEqual(
            hashlib.sha256(qwen.content).hexdigest(),
            _SELECTED_DIGESTS[("investigation_planner", "v24", "qwen_3_5")],
        )
        self.assertEqual(ADMISSIBLE_LOOKUP_STATUSES, {"resolved", "unlisted"})
        self.assertIn(
            "Only after a resolved or unlisted lookup result, call fetch_assembly_report",
            nemotron.text,
        )
        # Main's v12 locates a region and proposes it complete without an assembly
        # report, so the unresolved rule names region_located too.
        self.assertIn(
            "If the selected lookup result is neither resolved, unlisted nor "
            "region_located",
            nemotron.text,
        )
        self.assertNotIn("Only after a resolved lookup result,", nemotron.text)
        # S2 attempt 2 (Journey 54276d89): with "a versioned assembly accession you
        # choose", Nemotron took the examples' accessions and values, and sent
        # rsid "" for an unlisted site.
        self.assertIn(
            "call fetch_assembly_report with the lookup result's "
            "expected_assembly_accession",
            nemotron.text,
        )
        self.assertNotIn("a versioned assembly accession you choose", nemotron.text)
        # The model re-types no locus field, so an unlisted site's missing
        # rsID is code's to copy, and the examples show a shape only.
        self.assertIn("The example shows the output's shape only", nemotron.text)
        self.assertIn(
            "its one candidate is the stated allele with no rsID, and code copies none",
            nemotron.text,
        )
        self.assertNotIn("omits the rsid field", nemotron.text)
        # S3 attempt 1 (Journey b82fe399): Nemotron cited a result's own id
        # (refsnp-result:...) as refsnp_event_id, and tried to read HG001's genotype
        # itself or asked the researcher for it.
        self.assertIn(
            "refsnp_event_id is that lookup's source_event_id in stage_findings, the "
            "RefSnpToolResultRecorded event id, never a result id such as "
            "refsnp-result:",
            nemotron.text,
        )
        self.assertIn(
            "The Run reads this sample's VCF itself after the researcher selects the "
            "Work items; never try to read the genotype, and never ask the researcher "
            "for it.",
            nemotron.text,
        )
        # A recorded Journey: five members listed the RefSNP record's merged rsIDs
        # as aliases; rs182549 did it twice and stopped. Code now copies the
        # aliases, and the text still says which.
        self.assertIn(
            "(a record's merged_rsids are never aliases)",
            nemotron.text,
        )
        # The form carries main's v11 placement and v12 region wording.
        self.assertIn("(user_assertion.placement.position)", nemotron.text)
        self.assertIn(
            "a placement with step none means no step placed the position",
            nemotron.text,
        )
        self.assertIn(
            "code copies the Run's assembly, the region's sequence", nemotron.text
        )
        self.assertIn("do not call fetch_assembly_report for it", nemotron.text)
        # The block source is Nemotron's text; Qwen's form rewrites only this
        # instruction block (v21 reads the output blocks as the source does).
        rewritten = {"source_lookups"}
        for block in instruction_blocks("investigation_planner"):
            if block.use is not None:
                continue
            with self.subTest(block=block.name):
                self.assertIn(block.text, nemotron.text)
                self.assertEqual(block.text in qwen.text, block.name not in rewritten)

    def test_every_selected_role_renders_in_every_form(self):
        from bio_neb.agentic_journey.role_declaration import ROLES
        from bio_neb.agentic_journey.role_instructions import _FORMS

        # A pull that selects a version with no block source fails here.
        for role, version in (
            (name, item.instruction_version) for name, item in ROLES.items()
        ):
            for form in _FORMS:
                with self.subTest(role=role, form=form):
                    rendered = role_instructions(role, form=form)
                    self.assertEqual(rendered.identity, f"{role}:{version}:{form}")
                    self.assertTrue(rendered.text.strip())

    def test_nemotron_proposal_routes_teach_only_shapes_admission_accepts(self):
        from bio_neb.parabricks.slot_satisfaction import slot_rows

        text = stage_text(
            role_instructions("planning_agent", form="nemotron_3_5_super").text,
            "proposal_requirements",
        )
        taught = {}
        for kinds, line, shapes in _route_lines(text):
            states = re.findall(r"\b(asserted|observed|to_find)\b", line)
            for kind in kinds:
                taught[kind] = line
                for route, validation in shapes:
                    self.assertTrue(
                        any(
                            slot_rows(kind, state, validation, route)
                            for state in states
                        ),
                        (kind, route, validation, line),
                    )
        # One route line per kind, the S4 per-kind list (tagged-d) on v8's kinds,
        # and v19's optional cited_paper.
        self.assertEqual(
            set(taught),
            {
                "cited_paper",
                "research_goal",
                "registered_input",
                "sample_identity",
                "taxon_identity",
                "reference_assembly_identity",
                "companion_asset",
                "analysis_recipe",
                "resource_profile",
                "scientific_investigation",
            },
        )
        identity = taught["sample_identity"]
        self.assertIn("Never validation none", identity)
        # A stated identity is taught both routes admission admits for it, as
        # the Tool's own example uses use_supplied_assertion.
        self.assertEqual(
            {
                route
                for _kinds, line, shapes in _route_lines(text)
                for route, _ in shapes
                if line == identity
            },
            {"use_supplied_assertion", "validate_with_tool"},
        )
        # resource_profile has its own line: always recorded, chosen by a Tool.
        self.assertIn("always record it", taught["resource_profile"])
        self.assertNotIn("sample_identity", taught["resource_profile"])
        self.assertNotIn("proposal_output", text)

    def test_nemotron_quote_rules_agree_with_the_shared_rsid_rule(self):
        text = stage_text(
            role_instructions("planning_agent", form="nemotron_3_5_super").text,
            "proposal_requirements",
        )
        # The tightened quote rule (S4 tagged-c).
        self.assertIn("the shortest span of the request's own words", text)
        self.assertIn("copied character for character", text)
        # No placeholder rsID; a coordinate-only site is asserted with its site
        # typed; an rsID the request does not give is to_find.
        self.assertIn(
            "an rsID the request does not give is to_find, never a placeholder", text
        )
        self.assertIn(
            "A site the request states only by coordinates is asserted too", text
        )
        # The site is typed, never the quote its value.
        self.assertIn(
            '{"state": "asserted", "quote": "V:2129317 C>T", "site": {"contig": "V", '
            '"position": 2129317, "ref": "C", "alt": "T"}}',
            text,
        )
        self.assertNotRegex(text.lower(), r"leave (the )?(value|quote) out")

    def test_a_form_text_that_breaks_its_block_source_is_refused(self):
        published = _PUBLISHED / "planning_agent.v22.blocks.toml"

        def row(name, extra=""):
            return f'[[block]]\nname = "{name}"\n{extra}text = """\nx\n"""\n'

        cases = {
            "names no block": (row("absent"), "replaces no block: absent"),
            "repeats a block": (
                row("validation") + row("validation"),
                "repeats a block name",
            ),
            "unknown field": (
                row("validation", 'stage = "reference"\n'),
                "unknown field",
            ),
        }
        # A cached rendering would answer instead of the broken file.
        role_instructions.cache_clear()
        self.addCleanup(role_instructions.cache_clear)
        for case, (source, reason) in cases.items():
            with tempfile.TemporaryDirectory() as directory:
                own = Path(directory) / "planning_agent.v22.qwen_3_5.toml"
                own.write_text(source, encoding="utf-8")
                paths = {own.name: own, published.name: published}
                with (
                    self.subTest(case=case),
                    patch(
                        "bio_neb.agentic_journey.role_instructions._published",
                        lambda name, paths=paths, directory=directory: paths.get(
                            name, Path(directory) / name
                        ),
                    ),
                    self.assertRaisesRegex(ValueError, reason),
                ):
                    role_instructions("planning_agent", form="qwen_3_5")


class UnrenderableInstructionTests(unittest.TestCase):
    """Instructions that cannot render are refused with one typed code (I-65)."""

    def _refused(self, reason, files, render):
        from bio_neb.agentic_journey import role_instructions as module

        module.role_instructions.cache_clear()
        self.addCleanup(module.role_instructions.cache_clear)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in files:
                source = _PUBLISHED / name
                (root / name).write_text(
                    files[name] or source.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            with (
                patch.object(module, "_published", lambda name: root / name),
                self.assertRaisesRegex(
                    module.InstructionsUnrenderable, reason
                ) as raised,
            ):
                render(module)
        self.assertEqual(raised.exception.code, "instructions_unrenderable")
        self.assertEqual(raised.exception.violations, (str(raised.exception),))

    def test_a_missing_block_source_is_refused(self):
        self._refused(
            "^unknown instruction version: table_reader:v4$",
            {},
            lambda module: module.role_instructions("table_reader"),
        )

    def test_a_declared_form_text_that_is_missing_is_refused(self):
        # Qwen never reads Nemotron's words because its own file went missing.
        self._refused(
            "^form text planning_agent:v22:qwen_3_5 is declared but missing$",
            {"planning_agent.v22.blocks.toml": None},
            lambda module: module.role_instructions("planning_agent", form="qwen_3_5"),
        )

    def test_a_form_text_no_declaration_names_is_refused(self):
        self._refused(
            "^form text table_reader:v4:qwen_3_5 is published but not declared$",
            {
                "table_reader.v4.blocks.toml": None,
                "table_reader.v4.qwen_3_5.toml": (
                    '[[block]]\nname = "assignment"\ntext = "x"\n'
                ),
            },
            lambda module: module.role_instructions("table_reader", form="qwen_3_5"),
        )

    def test_a_block_with_an_unknown_use_is_refused(self):
        self._refused(
            "^block source table_reader:v4 has an unknown use$",
            {
                "table_reader.v4.blocks.toml": (
                    '[[block]]\nname = "a"\nuse = "footnote"\ntext = "x"\n'
                )
            },
            lambda module: module.role_instructions("table_reader"),
        )

    def test_a_message_its_role_does_not_state_or_cannot_fill_is_refused(self):
        source = _PUBLISHED / "investigation_planner.v24.blocks.toml"
        text = source.read_text(encoding="utf-8")
        for reason, changed in (
            (
                "^instructions table_reader:v4:nemotron_3_5_super state messages "
                r"\['job', 'note'\], not \['job'\]$",
                None,
            ),
            (
                "^instructions investigation_planner:v24:nemotron_3_5_super message "
                r"next_step names fields its role does not fill: \['sample'\]$",
                text.replace("on the reference {assembly}", "for {sample}"),
            ),
        ):
            with self.subTest(reason=reason):
                if changed is None:
                    files = {
                        "table_reader.v4.blocks.toml": (
                            _PUBLISHED / "table_reader.v4.blocks.toml"
                        ).read_text(encoding="utf-8")
                        + '\n[[block]]\nname = "note"\nuse = "message"\ntext = "x"\n'
                    }
                    role = "table_reader"
                else:
                    self.assertNotEqual(changed, text)
                    files = {source.name: changed}
                    role = "investigation_planner"
                self._refused(
                    reason,
                    files,
                    lambda module, role=role: module.role_instructions(role),
                )

    def test_a_message_field_is_a_bare_name_so_every_message_renders(self):
        # A conversion, format specification or nested field would fail only
        # when the member fills its job, after the model is acquired.
        source = _PUBLISHED / "investigation_planner.v24.blocks.toml"
        text = source.read_text(encoding="utf-8")
        formats = "message job formats a field: a field is a bare name$"
        for field, reason in (
            ("{lookup!x}", formats),
            ("{lookup!z}", formats),
            ("{lookup!r}", formats),
            ("{lookup:d}", formats),
            ("{lookup:{sample}}", formats),
            (
                "{lookup.upper}",
                r"message job names fields its role does not fill: "
                r"\['lookup.upper'\]$",
            ),
            (
                "{lookup[0]}",
                r"message job names fields its role does not fill: "
                r"\['lookup\[0\]'\]$",
            ),
            ("{}", r"message job names fields its role does not fill: \[''\]$"),
        ):
            changed = text.replace("{lookup}", field, 1)
            self.assertNotEqual(changed, text)
            with self.subTest(field=field):
                self._refused(
                    "^instructions investigation_planner:v24:nemotron_3_5_super "
                    + reason,
                    {source.name: changed},
                    lambda module: module.role_instructions("investigation_planner"),
                )
        # Every shipped message renders with its role's fields as text; Qwen's job
        # names no field.
        for form, fills in (("qwen_3_5", False), ("nemotron_3_5_super", True)):
            selected = role_instructions("investigation_planner", form=form)
            job = selected.message("job", lookup="lookup_X")
            self.assertEqual("lookup_X" in job, fills)
            self.assertNotIn("{", job)
            self.assertNotIn(
                "{", selected.message("next_step", assembly="GCF_X", lookup="l")
            )

    def test_purposes_must_name_exactly_the_roles_tools(self):
        from bio_neb.agentic_journey.role_instructions import InstructionsUnrenderable
        from bio_neb.agentic_journey.tool_execution import function_tool_names

        selected = role_instructions("table_reader")
        selected.require_tool_purposes(function_tool_names("table_reader"))
        with self.assertRaisesRegex(
            InstructionsUnrenderable,
            r"state no purpose for \['fetch_refsnp'\] or one for no Tool of the role: "
            r"\['ask_researcher', 'search_journey'\]$",
        ):
            selected.require_tool_purposes({"fetch_refsnp"})

    def test_an_unknown_message_or_an_unfilled_field_is_refused(self):
        selected = role_instructions("investigation_planner")
        from bio_neb.agentic_journey.role_instructions import InstructionsUnrenderable

        for name, fields in (("absent", {}), ("next_step", {"assembly": "GCF_X"})):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    InstructionsUnrenderable,
                    f"^instructions investigation_planner:v24:nemotron_3_5_super "
                    f"have no message {name} or leave its field",
                ),
            ):
                selected.message(name, **fields)
