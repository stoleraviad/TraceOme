from __future__ import annotations

import copy
import tempfile
import unittest
import uuid
from pathlib import Path

from pydantic import ValidationError

from bio_neb.application.queries import (
    ReviewJourneyQuery,
    review_journey,
)
from bio_neb.input_inspection.local_files import LocalFilesAdapter
from bio_neb.input_inspection.tool import (
    InputMetadataInspectionTool,
    InspectInputMetadataInput,
    RegisteredInputPath,
)
from bio_neb.parabricks.proposal_information import (
    ProposalInformationPlanDraft,
    ProposalPlanGroundingError,
    ground_proposal_information_plan,
)
from tests.content_identity_vectors import expected_digest, fields
from tests.proposal_information_fakes import proposal_draft, registered_run


def grounding_fixture(root: Path):
    receipt, paths = registered_run(root)
    registered = review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
    tool_input = InspectInputMetadataInput(
        input_set_id=registered.registered_input_set.input_set_id,
        inputs=tuple(
            RegisteredInputPath(role=role, registered_path=str(path))
            for role, path in paths.items()
        ),
    )
    result = InputMetadataInspectionTool().inspect(tool_input, LocalFilesAdapter())
    inspection_event_id = str(uuid.uuid4())
    draft_value = proposal_draft(
        type(
            "Review",
            (),
            {
                "request": registered.request,
                "registered_input_set": registered.registered_input_set,
                "inspection": type(
                    "Inspection", (), {"source_event_id": inspection_event_id}
                )(),
            },
        )(),
        path_by_role={role: str(path) for role, path in paths.items()},
    )
    return registered, result, inspection_event_id, draft_value


def ground_draft(
    registered,
    result,
    inspection_event_id,
    draft_value,
    *,
    prompt=None,
    registered_request_event_ids=(),
    citation_words=(),
    recorded_papers=None,
):
    return ground_proposal_information_plan(
        ProposalInformationPlanDraft.model_validate(draft_value),
        journey_id=registered.journey_id,
        research_request=prompt or registered.request.prompt,
        request_event_id=registered.request.source_event_id,
        registration_event_id=registered.registered_input_set.source_event_id,
        inspection_event_id=inspection_event_id,
        input_set_id=registered.registered_input_set.input_set_id,
        inspection_result=result,
        registered_request_event_ids=registered_request_event_ids,
        citation_words=citation_words,
        recorded_papers=recorded_papers,
    )


FIXED_PROMPT = (
    "Sample HG001, human (NCBI Taxonomy 9606), whole-genome paired-end reads "
    "registered, reference GCF_000001405.40.\n"
    "Tell me whether HG001 carries rs429358, rs7412 and rs4149056."
)


def requirement(draft_value, kind):
    return next(item for item in draft_value["requirements"] if item["kind"] == kind)


def grounded(plan, kind):
    return next(item for item in plan.requirements if item.kind == kind)


class ProposalInformationTests(unittest.TestCase):
    def test_scientific_investigation_is_required_collection_shaped_information(self):
        # Every Run names what the Investigation is about, before Execution; a
        # plan without a target is refused, and items are a collection.
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            without_target = copy.deepcopy(draft_value)
            without_target["requirements"] = [
                item
                for item in without_target["requirements"]
                if item["kind"] != "scientific_investigation"
            ]
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(registered, result, inspection_event_id, without_target)
            self.assertEqual(
                raised.exception.errors, ("proposal_requirement_kinds_incomplete",)
            )
            value = "Why might rs1234 affect the phenotype?"
            draft_value["requirements"].append(
                {
                    "kind": "scientific_investigation",
                    "subject_key": "variant_hypothesis",
                    "domain": "scientific",
                    "meaning": "A post-execution question, not an execution consequence.",
                    "knowledge": {
                        "state": "to_find",
                        "evidence_event_ids": [registered.request.source_event_id],
                    },
                    "acquisition_route": "validate_with_tool",
                    "validation": {
                        "kind": "tool_evidence",
                        "requirement": value,
                    },
                    "investigation_purpose": "hypothesis",
                }
            )
            plan = ground_draft(registered, result, inspection_event_id, draft_value)
            self.assertEqual(
                [
                    item.subject_key
                    for item in plan.requirements
                    if item.kind == "scientific_investigation"
                ],
                ["requested_variant_rs1234", "variant_hypothesis"],
            )
            investigation = next(
                item
                for item in plan.requirements
                if item.subject_key == "variant_hypothesis"
            )
            self.assertEqual(investigation.investigation_purpose, "hypothesis")
            self.assertIsNone(investigation.value)
            self.assertIsNone(investigation.verification)

    def test_a_plan_records_no_output_because_execution_always_makes_bam_and_vcf(
        self,
    ):
        # BAM + VCF is the fixed product contract, so an output is no requirement; a
        # plan without one grounds and one that records it is refused.
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            self.assertNotIn(
                "proposal_output",
                {item["kind"] for item in draft_value["requirements"]},
            )
            plan = ground_draft(registered, result, inspection_event_id, draft_value)
            self.assertNotIn(
                "proposal_output", {item.kind for item in plan.requirements}
            )
            with_output = copy.deepcopy(draft_value)
            with_output["requirements"].append(
                {
                    "kind": "proposal_output",
                    "subject_key": "requested_outputs",
                    "domain": "scientific",
                    "meaning": "The output the request names.",
                    "knowledge": {"state": "asserted", "value": "VCF", "quote": "VCF"},
                    "acquisition_route": "use_supplied_assertion",
                    "validation": {"kind": "none"},
                }
            )
            with self.assertRaises(ValidationError) as raised:
                ProposalInformationPlanDraft.model_validate(with_output)
            self.assertEqual(raised.exception.errors()[0]["type"], "union_tag_invalid")

    def test_schema_errors_also_report_missing_required_kinds(self):
        invalid = {
            "requirements": [
                {
                    "kind": "research_goal",
                    "subject_key": "research_goal",
                    "domain": "scientific",
                    "meaning": "Requested analysis.",
                    "knowledge": {"state": "asserted", "quote": "requested analysis\n"},
                    "acquisition_route": "use_supplied_assertion",
                    "validation": {"kind": "none"},
                }
            ],
        }

        with self.assertRaises(ValidationError) as raised:
            ProposalInformationPlanDraft.model_validate(invalid)

        types = {item["type"] for item in raised.exception.errors()}
        # The quote's rule is in its schema.
        self.assertIn("string_pattern_mismatch", types)
        self.assertIn("proposal_requirement_kinds_incomplete", types)

    def test_a_refusal_names_each_missing_kind_in_both_paths(self):
        # One, two or all but one kind missing; the grounding refusal holds them
        # as a typed fact and its registry sentence names them, and a
        # field-error refusal names the same kinds in its own message.
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts

        sentence = next(
            item.explanation
            for contract in function_tool_contracts("planning_agent")
            for item in contract.preconditions
            if item.code == "proposal_requirement_kinds_incomplete"
        )
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            kinds = [item["kind"] for item in draft_value["requirements"]]
            for dropped, missing in (
                ({"taxon_identity"}, ("taxon_identity",)),
                (
                    {"resource_profile", "companion_asset"},
                    ("companion_asset", "resource_profile"),
                ),
                (
                    set(kinds) - {"research_goal"},
                    (
                        "registered_input",
                        "sample_identity",
                        "taxon_identity",
                        "reference_assembly_identity",
                        "companion_asset",
                        "analysis_recipe",
                        "resource_profile",
                        "scientific_investigation",
                    ),
                ),
            ):
                with self.subTest(missing=missing):
                    partial = copy.deepcopy(draft_value)
                    partial["requirements"] = [
                        item
                        for item in partial["requirements"]
                        if item["kind"] not in dropped
                    ]
                    with self.assertRaises(ProposalPlanGroundingError) as raised:
                        ground_draft(registered, result, inspection_event_id, partial)
                    self.assertIn(
                        "proposal_requirement_kinds_incomplete", raised.exception.errors
                    )
                    facts = raised.exception.facts[
                        "proposal_requirement_kinds_incomplete"
                    ]
                    self.assertEqual(facts, {"missing": missing})
                    self.assertIn(
                        "still missing: "
                        + ", ".join(f"`{kind}`" for kind in missing)
                        + ".",
                        sentence_with_facts(sentence, facts),
                    )
                    # The same draft with a field error too: the field-error
                    # message names the same kinds.
                    partial["requirements"][0]["knowledge"] = {"quote": "x"}
                    with self.assertRaises(ValidationError) as invalid:
                        ProposalInformationPlanDraft.model_validate(partial)
                    (message,) = [
                        item["msg"]
                        for item in invalid.exception.errors()
                        if item["type"] == "proposal_requirement_kinds_incomplete"
                    ]
                    self.assertEqual(
                        message,
                        "add one requirement of each missing kind: "
                        + ", ".join(missing),
                    )

    def test_a_refusal_names_the_inputs_and_request_file_it_lacks(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            inputs = [
                item
                for item in draft_value["requirements"]
                if item["kind"] == "registered_input"
            ]
            dropped = inputs[-1]
            draft_value["requirements"].remove(dropped)
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    draft_value,
                    registered_request_event_ids=(inspection_event_id,),
                )
            facts = raised.exception.facts
            self.assertEqual(
                facts["registered_inputs_not_fully_represented"],
                {"roles": (dropped["knowledge"]["input_role"],)},
            )
            self.assertEqual(
                facts["registered_request_uncited"],
                {"event_ids": (inspection_event_id,)},
            )

    def test_an_ungrounded_item_is_told_what_it_may_cite(self):
        # Each ungrounded item's refusal names the allowed values, and a target
        # only the registered request file lists is told the file's targets are
        # one to_find item (24 such refusals in one reply).
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts

        rows = {
            item.code: item.explanation
            for contract in function_tool_contracts("planning_agent")
            if contract.tool_name == "record_proposal_requirements"
            for item in contract.preconditions
        }
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            file_event_id = str(uuid.uuid4())
            draft_value["requirements"].extend(
                (
                    {
                        "kind": "scientific_investigation",
                        "subject_key": "rs1801133",
                        "domain": "scientific",
                        "meaning": "Whether the sample carries the variant.",
                        "knowledge": {
                            "state": "asserted",
                            "value": "rs1801133",
                            "quote": "rs1801133",
                        },
                        "acquisition_route": "validate_with_tool",
                        "validation": {
                            "kind": "tool_evidence",
                            "requirement": "Observe it.",
                        },
                        "investigation_purpose": "known_variant",
                    },
                    {
                        "kind": "scientific_investigation",
                        "subject_key": "file_targets",
                        "domain": "scientific",
                        "meaning": "The targets the request file lists.",
                        "knowledge": {
                            "state": "to_find",
                            "evidence_event_ids": ["not-an-event"],
                        },
                        "acquisition_route": "validate_with_tool",
                        "validation": {
                            "kind": "tool_evidence",
                            "requirement": "Observe them.",
                        },
                        "investigation_purpose": "known_variant",
                    },
                )
            )
            observed = next(
                item
                for item in draft_value["requirements"]
                if item["knowledge"]["state"] == "observed"
            )
            observed["knowledge"]["input_role"] = "no_such_role"
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    draft_value,
                    registered_request_event_ids=(file_event_id,),
                )
        facts = raised.exception.facts
        self.assertEqual(
            sentence_with_facts(
                rows["request_assertion_ungrounded"],
                facts["rs1801133:request_assertion_ungrounded"],
            ),
            "Its quote must copy words of the research request exactly, starting "
            "and ending on whole words. A target only a registered request file "
            "lists is not quoted: each file's targets are the one to_find "
            f"scientific_investigation item that cites it: `{file_event_id}`.",
        )
        self.assertEqual(
            sentence_with_facts(
                rows["context_evidence_ungrounded"],
                facts["file_targets:context_evidence_ungrounded"],
            ),
            "to_find knowledge may cite only the research request, the registered "
            "request file, the input registration, the inspection or a recorded "
            "paper's events: "
            f"`{registered.request.source_event_id}`, "
            f"`{registered.registered_input_set.source_event_id}`, "
            f"`{inspection_event_id}`, `{file_event_id}`.",
        )
        roles = ", ".join(f"`{item.role}`" for item in result.inputs)
        self.assertEqual(
            sentence_with_facts(
                rows["input_observation_ungrounded"],
                facts[f"{observed['subject_key']}:input_observation_ungrounded"],
            ),
            "The observed input role and fact must name a value the inspection "
            f"recorded: input_role one of {roles}; fact one of `role`, "
            "`registered_path`, `exists`, `regular_file`, `size_bytes`.",
        )

    def test_an_unquoted_value_without_a_request_file_is_told_only_the_rule(self):
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts

        (row,) = [
            item.explanation
            for contract in function_tool_contracts("planning_agent")
            for item in contract.preconditions
            if item.code == "request_assertion_ungrounded"
        ]
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            sample = requirement(draft_value, "sample_identity")
            sample["knowledge"]["quote"] = "a sample the request never names"
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(registered, result, inspection_event_id, draft_value)
        (code,) = raised.exception.errors
        self.assertEqual(
            sentence_with_facts(row, raised.exception.facts[code]),
            "Its quote must copy words of the research request exactly, starting "
            "and ending on whole words.",
        )

    def test_repeated_subject_keys_are_named(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
        repeated = copy.deepcopy(draft_value)
        repeated["requirements"].append(copy.deepcopy(repeated["requirements"][0]))
        with self.assertRaises(ValidationError) as raised:
            ProposalInformationPlanDraft.model_validate(repeated)
        key = repeated["requirements"][0]["subject_key"]
        self.assertIn(
            f"requirement subject keys must be unique; repeated: {key}",
            raised.exception.errors()[0]["msg"],
        )

    def test_grounded_plan_has_deterministic_unique_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            plan = ground_draft(registered, result, inspection_event_id, draft_value)
            scope = {
                "journey_id": registered.journey_id,
                "request_event_id": registered.request.source_event_id,
                "registration_event_id": registered.registered_input_set.source_event_id,
                "inspection_event_id": inspection_event_id,
                "input_set_id": registered.registered_input_set.input_set_id,
            }
            # The matcher is recorded for a typed value; the verification for a fact slot.
            requirements = [
                fields(
                    dump,
                    "requirement_id kind subject_key domain meaning knowledge_state value evidence acquisition_route validation investigation_purpose"
                    + "".join(
                        " " + name
                        for name in ("identifier_rule", "verification")
                        if name in dump
                    ),
                )
                for dump in (item.model_dump(mode="json") for item in plan.requirements)
            ]
            self.assertEqual(
                plan.plan_id,
                "plan_"
                + expected_digest({"scope": scope, "requirements": requirements}),
            )
            for item in plan.requirements:
                self.assertEqual(
                    item.requirement_id,
                    "requirement_"
                    + expected_digest(
                        [expected_digest(scope), item.kind, item.subject_key]
                    )[:32],
                )
            self.assertEqual(
                len({item.requirement_id for item in plan.requirements}),
                len(plan.requirements),
            )

    def test_prose_quote_is_its_own_value_with_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            knowledge = draft_value["requirements"][0]["knowledge"]
            self.assertEqual(knowledge, {"state": "asserted", "quote": "rs1234"})
            plan = ground_draft(registered, result, inspection_event_id, draft_value)
            goal = grounded(plan, "research_goal")
            expected_offset = registered.request.prompt.index("rs1234")
            self.assertEqual(goal.value, "rs1234")
            self.assertIsNone(goal.identifier_rule)
            self.assertIsNone(goal.verification)
            self.assertEqual(
                fields(
                    goal.evidence[0].model_dump(mode="json"),
                    "start_offset end_offset exact_text",
                ),
                {
                    "start_offset": expected_offset,
                    "end_offset": expected_offset + len("rs1234"),
                    "exact_text": "rs1234",
                },
            )

    def test_typed_value_must_be_one_its_quote_states(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            prompt = (
                registered.request.prompt
                + " Also check RS4321 (NCBI taxon 9606), sample HG001_ at 1.2."
            )
            cases = (
                ("sample_identity", "sample-002", "sample-001"),
                ("sample_identity", "HG001", "HG001_"),
                ("taxon_identity", 10090, "NCBI taxon 9606"),
                ("taxon_identity", 1, "1.2"),
            )
            for kind, value, quote in cases:
                changed = copy.deepcopy(draft_value)
                item = requirement(changed, kind)
                item["knowledge"] = {
                    "state": "asserted",
                    "value": value,
                    "quote": quote,
                }
                item["acquisition_route"] = "use_supplied_assertion"
                with self.subTest(kind=kind, value=value, quote=quote):
                    with self.assertRaises(ProposalPlanGroundingError) as raised:
                        ground_draft(
                            registered,
                            result,
                            inspection_event_id,
                            changed,
                            prompt=prompt,
                        )
                    self.assertEqual(
                        raised.exception.errors,
                        (f"{item['subject_key']}:asserted_value_not_in_quote",),
                    )
            investigation = {
                "kind": "scientific_investigation",
                "subject_key": "rs1234",
                "domain": "scientific",
                "meaning": "Whether the sample carries the variant.",
                "knowledge": {
                    "state": "asserted",
                    "value": "rs4321",
                    "quote": "RS4321",
                },
                "acquisition_route": "validate_with_tool",
                "validation": {"kind": "tool_evidence", "requirement": "Observe it."},
                "investigation_purpose": "known_variant",
            }
            matching = copy.deepcopy(draft_value)
            matching["requirements"].append(investigation)
            plan = ground_draft(
                registered, result, inspection_event_id, matching, prompt=prompt
            )
            item = next(
                item for item in plan.requirements if item.subject_key == "rs1234"
            )
            self.assertEqual(
                (item.value, item.identifier_rule, item.evidence[0].exact_text),
                ("rs4321", "rsid", "RS4321"),
            )
            other = copy.deepcopy(matching)
            other["requirements"][-1]["knowledge"]["value"] = "rs1234"
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(
                    registered, result, inspection_event_id, other, prompt=prompt
                )
            self.assertEqual(
                raised.exception.errors, ("rs1234:asserted_value_not_in_quote",)
            )

    def test_a_coordinate_only_site_is_typed_and_checked_against_its_quote(self):
        """#472 Q1 a (ADR 0049 membership): the model types the site, and code checks
        only that the quote holds its contig, position and bases; a part the quote
        does not write is refused with `site_not_in_quote`. Owner D1 (#305), #326: with
        no rsID in the request, the quote stays the item's value."""
        from bio_neb.investigation.investigation_request import StatedPosition

        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            prompt = registered.request.prompt + " Does it carry V:2129317 C>T?"
            draft_value["requirements"].append(
                {
                    "kind": "scientific_investigation",
                    "subject_key": "site_v_2129317",
                    "domain": "scientific",
                    "meaning": "Whether the sample carries the site.",
                    "knowledge": {
                        "state": "asserted",
                        "quote": "V:2129317 C>T",
                        "site": {
                            "contig": "V",
                            "position": 2129317,
                            "ref": "C",
                            "alt": "T",
                        },
                    },
                    "acquisition_route": "validate_with_tool",
                    "validation": {
                        "kind": "tool_evidence",
                        "requirement": "Observe it.",
                    },
                    "investigation_purpose": "known_variant",
                }
            )
            plan = ground_draft(
                registered, result, inspection_event_id, draft_value, prompt=prompt
            )
            site = next(
                item
                for item in plan.requirements
                if item.subject_key == "site_v_2129317"
            )
            self.assertEqual(
                (
                    site.knowledge_state,
                    site.value,
                    site.identifier_rule,
                    site.verification,
                ),
                ("asserted", "V:2129317 C>T", None, None),
            )
            self.assertEqual(
                site.site,
                StatedPosition(contig="V", position=2129317, ref="C", alt="T"),
            )
            offset = len(registered.request.prompt) + len(" Does it carry ")
            self.assertEqual(
                fields(
                    site.evidence[0].model_dump(mode="json"),
                    "start_offset end_offset exact_text",
                ),
                {
                    "start_offset": offset,
                    "end_offset": offset + len("V:2129317 C>T"),
                    "exact_text": "V:2129317 C>T",
                },
            )
            # A site the quote does not hold is refused, each part on its own.
            for part, wrong in (
                ("contig", "IV"),
                ("position", 2129318),
                ("ref", "G"),
                ("alt", "A"),
            ):
                with self.subTest(part=part):
                    changed = copy.deepcopy(draft_value)
                    changed["requirements"][-1]["knowledge"]["site"][part] = wrong
                    with self.assertRaises(ProposalPlanGroundingError) as raised:
                        ground_draft(
                            registered,
                            result,
                            inspection_event_id,
                            changed,
                            prompt=prompt,
                        )
                    self.assertEqual(
                        raised.exception.errors, ("site_v_2129317:site_not_in_quote",)
                    )
            # The same site with a placeholder rsID is refused, not admitted.
            draft_value["requirements"][-1]["knowledge"]["value"] = "rs429358"
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(
                    registered, result, inspection_event_id, draft_value, prompt=prompt
                )
            self.assertEqual(
                raised.exception.errors, ("site_v_2129317:asserted_value_not_in_quote",)
            )

    def test_a_gene_change_is_typed_and_checked_against_its_quote(self):
        """#473 Q1 a, Q2 a (ADR 0049 membership): the model types the gene change, and
        code checks only that the quote writes its gene, its change and the number
        it names; a part the quote does not write is refused with
        `gene_change_not_in_quote`."""
        from bio_neb.investigation.investigation_request import StatedChange

        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            prompt = registered.request.prompt + " Does it carry BRAF V600E?"
            change = {
                "gene": "BRAF",
                "change": "V600E",
                "protein": {"ref": "V", "codon": 600, "alt": "E"},
            }
            draft_value["requirements"].append(
                {
                    "kind": "scientific_investigation",
                    "subject_key": "braf_v600e",
                    "domain": "scientific",
                    "meaning": "Whether the sample carries BRAF V600E.",
                    "knowledge": {
                        "state": "asserted",
                        "quote": "BRAF V600E",
                        "gene_change": change,
                    },
                    "acquisition_route": "validate_with_tool",
                    "validation": {
                        "kind": "tool_evidence",
                        "requirement": "Observe it.",
                    },
                    "investigation_purpose": "known_variant",
                }
            )
            plan = ground_draft(
                registered, result, inspection_event_id, draft_value, prompt=prompt
            )
            (item,) = [i for i in plan.requirements if i.subject_key == "braf_v600e"]
            self.assertEqual(
                (item.value, item.gene_change),
                ("BRAF V600E", StatedChange.model_validate(change)),
            )
            for part, wrong in (
                ("gene", "BRAFF"),
                ("change", "V600K"),
                ("protein", {"ref": "V", "codon": 601, "alt": "E"}),
                ("transcript", "NM_004333.6"),
            ):
                with self.subTest(part=part):
                    changed = copy.deepcopy(draft_value)
                    changed["requirements"][-1]["knowledge"]["gene_change"][part] = (
                        wrong
                    )
                    with self.assertRaises(ProposalPlanGroundingError) as raised:
                        ground_draft(
                            registered,
                            result,
                            inspection_event_id,
                            changed,
                            prompt=prompt,
                        )
                    self.assertEqual(
                        raised.exception.errors,
                        ("braf_v600e:gene_change_not_in_quote",),
                    )

    def test_a_sites_stated_version_is_kept_as_the_model_names_it(self):
        # The model names the version the request states for the site, and code does
        # not check it ("remove this"): "hg38" may resolve to GRCh38.p14, which only
        # the placement lookup knows.
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            quote = "chr19:45411941 T>C on CanFam 3.1"
            prompt = registered.request.prompt + f" Check {quote}."

            def site(version):
                knowledge = {"state": "asserted", "quote": quote}
                if version is not None:
                    knowledge["stated_version"] = version
                changed = copy.deepcopy(draft_value)
                changed["requirements"].append(
                    {
                        "kind": "scientific_investigation",
                        "subject_key": "site_chr19",
                        "domain": "scientific",
                        "meaning": "Whether the sample carries the site.",
                        "knowledge": knowledge,
                        "acquisition_route": "validate_with_tool",
                        "validation": {
                            "kind": "tool_evidence",
                            "requirement": "Observe it.",
                        },
                        "investigation_purpose": "known_variant",
                    }
                )
                return changed

            def stated(version):
                plan = ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    site(version),
                    prompt=prompt,
                )
                return _site(plan).stated_version

            def _site(plan):
                return next(
                    item
                    for item in plan.requirements
                    if item.subject_key == "site_chr19"
                )

            for version in ("CanFam 3.1", "CanFam3.1", "GRCh38.p14"):
                with self.subTest(version=version):
                    self.assertEqual(stated(version), version)
            self.assertIsNone(stated(None))
            # Left out when empty, so a plan with no stated version keeps its digest.
            plan = ground_draft(
                registered, result, inspection_event_id, site(None), prompt=prompt
            )
            self.assertNotIn("stated_version", _site(plan).model_dump(mode="json"))

    def test_an_answers_targets_are_grounded_like_the_requests(self):
        """#462 (owner, 2026-10-26), #472 Q1 a (ADR 0049): an answer is read as the
        request is, under the same quote rules, so a quote, a value or a site the
        answer does not hold is refused by its own code; the version is the model's,
        unchecked."""
        from bio_neb.investigation.investigation_request import StatedPosition
        from bio_neb.parabricks.proposal_information import (
            AnsweredTarget,
            AnsweredTargetsDraft,
            ground_answered_targets,
        )

        answer = "Check RS429358, and chr19:45,411,941 T>C on hg38."
        quote = "chr19:45,411,941 T>C on hg38"
        site = {"contig": "chr19", "position": 45411941, "ref": "T", "alt": "C"}

        def read(*targets):
            return ground_answered_targets(
                AnsweredTargetsDraft.model_validate({"targets": list(targets)}), answer
            )

        self.assertEqual(
            read(
                {"state": "asserted", "value": "rs429358", "quote": "RS429358"},
                {
                    "state": "asserted",
                    "quote": quote,
                    "site": site,
                    "stated_version": "GRCh38.p14",
                },
            ),
            (
                AnsweredTarget(quote="RS429358", start_offset=6, rsid="rs429358"),
                AnsweredTarget(
                    quote=quote,
                    start_offset=20,
                    stated_version="GRCh38.p14",
                    site=StatedPosition(
                        contig="chr19", position=45411941, ref="T", alt="C"
                    ),
                ),
            ),
        )
        # An HGVS g. deletion writes no bases: its site is its contig and first
        # position (its change stays in the quote).
        hgvs = "NC_049231.1:g.69888722_69888723del"
        self.assertEqual(
            ground_answered_targets(
                AnsweredTargetsDraft.model_validate(
                    {
                        "targets": [
                            {
                                "state": "asserted",
                                "quote": hgvs,
                                "site": {
                                    "contig": "NC_049231.1",
                                    "position": 69888722,
                                },
                            }
                        ]
                    }
                ),
                f"Check {hgvs}.",
            )[0].site,
            StatedPosition(contig="NC_049231.1", position=69888722),
        )
        self.assertEqual(read(), ())
        for target, code in (
            ({"state": "asserted", "quote": "RS42935"}, "answer_quote_ungrounded"),
            (
                {"state": "asserted", "value": "rs7412", "quote": "RS429358"},
                "asserted_value_not_in_quote",
            ),
            (
                {"state": "asserted", "quote": quote, "site": {**site, "alt": "G"}},
                "site_not_in_quote",
            ),
        ):
            with self.subTest(code=code):
                with self.assertRaises(ProposalPlanGroundingError) as raised:
                    read(target)
                self.assertEqual(raised.exception.errors, (f"target_0:{code}",))

    def test_a_site_the_stored_position_cannot_hold_is_refused_as_an_argument(self):
        """#472 Q1 a, review r1 ST1/SP1 (ADR 0031: an invalid model action gets its
        bounded repair): the plan's and an answer's typed site hold the stored
        position's rules, so a site with one allele, equal alleles or a padded
        sequence name is refused by the schema at its own field, before grounding."""
        from pydantic import ValidationError

        from bio_neb.parabricks.proposal_information import (
            AnsweredTargetsDraft,
            AssertedInvestigationDraft,
        )

        for site, message in (
            ({"ref": "T"}, "a position states both alleles or neither"),
            ({"alt": "C"}, "a position states both alleles or neither"),
            ({"ref": "T", "alt": "T"}, "REF and ALT must differ"),
            ({"contig": " chr19"}, "contig must be one non-empty canonical line"),
        ):
            target = {
                "state": "asserted",
                "quote": "chr19:44908684 T>C",
                "site": {"contig": "chr19", "position": 44908684, **site},
            }
            for model, value, loc in (
                (AssertedInvestigationDraft, target, ("site",)),
                (AnsweredTargetsDraft, {"targets": [target]}, ("targets", 0, "site")),
            ):
                with (
                    self.subTest(site=site, loc=loc),
                    self.assertRaises(ValidationError) as refused,
                ):
                    model.model_validate(value)
                (error,) = refused.exception.errors()
                self.assertEqual(error["loc"][: len(loc)], loc)
                self.assertIn(message, error["msg"])

    def test_sent_null_value_is_not_a_typed_value(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            requirement(draft_value, "sample_identity")["knowledge"]["value"] = None
            with self.assertRaises(ValidationError) as raised:
                ProposalInformationPlanDraft.model_validate(draft_value)
            self.assertEqual(
                [item["loc"][-1] for item in raised.exception.errors()], ["value"]
            )

    def test_truncated_identifier_quote_is_ungrounded(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            prompt = registered.request.prompt + " Sample HG001 is the reference."
            requirement(draft_value, "sample_identity")["knowledge"] = {
                "state": "asserted",
                "value": "HG00",
                "quote": "HG00",
            }
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(
                    registered, result, inspection_event_id, draft_value, prompt=prompt
                )
            self.assertEqual(
                raised.exception.errors,
                ("sample_identity:request_assertion_ungrounded",),
            )

    def test_quote_must_start_and_end_on_word_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            for quote in ("rs123", "s1234", "sample-00"):
                truncated = copy.deepcopy(draft_value)
                truncated["requirements"][0]["knowledge"]["quote"] = quote
                with self.subTest(quote=quote):
                    self.assertIn(quote, registered.request.prompt)
                    with self.assertRaises(ProposalPlanGroundingError) as raised:
                        ground_draft(registered, result, inspection_event_id, truncated)
                    self.assertEqual(
                        raised.exception.errors,
                        ("requested_variant:request_assertion_ungrounded",),
                    )

    def test_each_typed_value_records_its_matcher_and_each_fact_slot_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            prompt = (
                registered.request.prompt
                + " Organism: Homo sapiens (NCBI taxon 9606); produce germline VCF."
            )
            for kind, value, quote in (
                ("taxon_identity", 9606, "Homo sapiens (NCBI taxon 9606)"),
            ):
                item = requirement(draft_value, kind)
                item["knowledge"] = {
                    "state": "asserted",
                    "value": value,
                    "quote": quote,
                }
                item["acquisition_route"] = "use_supplied_assertion"
            plan = ground_draft(
                registered, result, inspection_event_id, draft_value, prompt=prompt
            )
            recorded = {
                item.kind: (
                    item.knowledge_state,
                    item.value,
                    item.evidence[-1].exact_text
                    if item.knowledge_state == "asserted"
                    else None,
                    item.identifier_rule,
                    None if item.verification is None else item.verification.source,
                )
                for item in plan.requirements
                if item.kind != "registered_input"
            }
            self.assertEqual(
                recorded,
                {
                    "research_goal": ("asserted", "rs1234", "rs1234", None, None),
                    "sample_identity": (
                        "asserted",
                        "sample-001",
                        "sample-001",
                        "sample_identifier",
                        None,
                    ),
                    "taxon_identity": (
                        "asserted",
                        "9606",
                        "Homo sapiens (NCBI taxon 9606)",
                        "ncbi_taxon_id",
                        "NCBI Taxonomy",
                    ),
                    "reference_assembly_identity": (
                        "to_find",
                        None,
                        None,
                        None,
                        "NCBI Datasets",
                    ),
                    # The rsID's matcher is recorded; its source is the
                    # Investigation model's choice.
                    "scientific_investigation": (
                        "asserted",
                        "rs1234",
                        "rs1234",
                        "rsid",
                        None,
                    ),
                    "companion_asset": ("to_find", None, None, None, None),
                    "analysis_recipe": ("to_find", None, None, None, None),
                    "resource_profile": ("to_find", None, None, None, None),
                },
            )
            sample = grounded(plan, "sample_identity")
            self.assertEqual(
                sample.model_dump(mode="json")["verification"], {"source": None}
            )
            observed = grounded(plan, "registered_input")
            self.assertIsNone(observed.identifier_rule)
            self.assertIsNone(observed.verification)

    def test_a_value_the_request_does_not_state_is_to_find_with_its_clue(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            prompt = registered.request.prompt + " The organism is Homo sapiens."
            taxon = requirement(draft_value, "taxon_identity")
            taxon["knowledge"]["quote"] = "Homo sapiens"
            plan = ground_draft(
                registered, result, inspection_event_id, draft_value, prompt=prompt
            )
            found = grounded(plan, "taxon_identity")
            self.assertEqual((found.knowledge_state, found.value), ("to_find", None))
            self.assertEqual(found.evidence[-1].exact_text, "Homo sapiens")
            taxon["knowledge"]["quote"] = "Mus musculus"
            with self.assertRaises(ProposalPlanGroundingError) as raised:
                ground_draft(
                    registered, result, inspection_event_id, draft_value, prompt=prompt
                )
            self.assertEqual(
                raised.exception.errors,
                ("taxon_identity:request_assertion_ungrounded",),
            )
            asserted_unstated = copy.deepcopy(draft_value)
            asserted_unstated["requirements"][0]["acquisition_route"] = (
                "use_supplied_assertion"
            )
            taxon = requirement(asserted_unstated, "taxon_identity")
            taxon["knowledge"] = {
                "state": "asserted",
                "value": "Homo sapiens",
                "quote": "Homo sapiens",
            }
            with self.assertRaises(ValidationError):
                ProposalInformationPlanDraft.model_validate(asserted_unstated)

    def test_fixed_prompt_records_typed_slots_and_a_repeated_value_grounds(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            asserted = {
                "research_goal": (
                    None,
                    "Tell me whether HG001 carries rs429358, rs7412 and rs4149056.",
                ),
                "sample_identity": ("HG001", "HG001"),
                "taxon_identity": (9606, "human (NCBI Taxonomy 9606)"),
                "reference_assembly_identity": (
                    "GCF_000001405.40",
                    "GCF_000001405.40",
                ),
            }
            for item in draft_value["requirements"]:
                if item["kind"] not in asserted:
                    continue
                value, quote = asserted[item["kind"]]
                item["knowledge"] = {"state": "asserted", "quote": quote}
                if value is not None:
                    item["knowledge"]["value"] = value
                item["acquisition_route"] = "use_supplied_assertion"
            draft_value["requirements"] = [
                item
                for item in draft_value["requirements"]
                if item["kind"] != "scientific_investigation"
            ]
            for rsid in ("rs429358", "rs7412", "rs4149056"):
                draft_value["requirements"].append(
                    {
                        "kind": "scientific_investigation",
                        "subject_key": rsid,
                        "domain": "scientific",
                        "meaning": f"Whether HG001 carries {rsid}.",
                        "knowledge": {
                            "state": "asserted",
                            "value": rsid,
                            "quote": rsid,
                        },
                        "acquisition_route": "validate_with_tool",
                        "validation": {
                            "kind": "tool_evidence",
                            "requirement": "Observe the variant after execution.",
                        },
                        "investigation_purpose": "known_variant",
                    }
                )
            plan = ground_draft(
                registered,
                result,
                inspection_event_id,
                draft_value,
                prompt=FIXED_PROMPT,
            )
            slots = {
                item.subject_key: (
                    item.knowledge_state,
                    item.value,
                    item.verification.source,
                )
                for item in plan.requirements
                if item.verification is not None
            }
            self.assertEqual(
                slots,
                {
                    "sample_identity": ("asserted", "HG001", None),
                    "taxon_identity": ("asserted", "9606", "NCBI Taxonomy"),
                    "reference_assembly": (
                        "asserted",
                        "GCF_000001405.40",
                        "NCBI Datasets",
                    ),
                    # An rsID's source is the Investigation model's choice:
                    # planning names none.
                    "rs429358": ("asserted", "rs429358", None),
                    "rs7412": ("asserted", "rs7412", None),
                    "rs4149056": ("asserted", "rs4149056", None),
                },
            )
            # HG001 is written twice; the quote records its first occurrence.
            sample = grounded(plan, "sample_identity")
            self.assertEqual(
                sample.evidence[0].start_offset, FIXED_PROMPT.index("HG001")
            )

    def test_targets_keep_the_order_the_researcher_wrote_them_in(self):
        # A recorded Run asked for rs429358, rs7412 and rs4149056 in that order.
        # Neither the model's order (reversed here) nor its subject keys
        # (alphabetical order rs4149056, rs429358, rs7412) may reorder them; a
        # target with no quote follows the quoted ones.
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            draft_value["requirements"] = [
                item
                for item in draft_value["requirements"]
                if item["kind"] != "scientific_investigation"
            ]
            for key, rsid in (
                ("target_c", "rs7412"),
                ("target_a", "rs4149056"),
                ("target_b", "rs429358"),
            ):
                draft_value["requirements"].append(
                    {
                        "kind": "scientific_investigation",
                        "subject_key": key,
                        "domain": "scientific",
                        "meaning": f"Whether HG001 carries {rsid}.",
                        "knowledge": {
                            "state": "asserted",
                            "value": rsid,
                            "quote": rsid,
                        },
                        "acquisition_route": "validate_with_tool",
                        "validation": {
                            "kind": "tool_evidence",
                            "requirement": "Observe the variant after execution.",
                        },
                        "investigation_purpose": "known_variant",
                    }
                )
            draft_value["requirements"].insert(
                0,
                {
                    "kind": "scientific_investigation",
                    "subject_key": "a_further_question",
                    "domain": "scientific",
                    "meaning": "A further question the request implies.",
                    "knowledge": {
                        "state": "to_find",
                        "evidence_event_ids": [registered.request.source_event_id],
                    },
                    "acquisition_route": "ask_planning_hitl",
                    "validation": {
                        "kind": "researcher_decision",
                        "requirement": "The researcher names it.",
                    },
                    "investigation_purpose": "open_question",
                },
            )
            prompt = (
                registered.request.prompt
                + " Tell me whether HG001 carries rs429358, rs7412 and rs4149056."
            )
            plan = ground_draft(
                registered, result, inspection_event_id, draft_value, prompt=prompt
            )

        self.assertEqual(
            [
                item.subject_key
                for item in plan.requirements
                if item.kind == "scientific_investigation"
            ],
            ["target_b", "target_c", "target_a", "a_further_question"],
        )

    def test_each_registered_file_is_cited_by_its_own_target_item(self):
        # A Run takes several files; one item stands for one file, and a
        # second file left uncited is refused.
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts

        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            files = (str(uuid.uuid4()), str(uuid.uuid4()))
            together = copy.deepcopy(draft_value)
            target = requirement(together, "scientific_investigation")
            target["knowledge"] = {
                "state": "to_find",
                "evidence_event_ids": list(files),
            }
            target["acquisition_route"] = "validate_with_tool"
            with self.assertRaises(ProposalPlanGroundingError) as cited_together:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    together,
                    registered_request_event_ids=files,
                )
            one = copy.deepcopy(together)
            requirement(one, "scientific_investigation")["knowledge"][
                "evidence_event_ids"
            ] = [files[0]]
            with self.assertRaises(ProposalPlanGroundingError) as one_uncited:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    one,
                    registered_request_event_ids=files,
                )
            each = copy.deepcopy(one)
            second = copy.deepcopy(requirement(each, "scientific_investigation"))
            second["subject_key"] = f"{second['subject_key']}_second_file"
            second["knowledge"]["evidence_event_ids"] = [files[1]]
            each["requirements"].append(second)
            plan = ground_draft(
                registered,
                result,
                inspection_event_id,
                each,
                registered_request_event_ids=files,
            )

        together_code = f"{target['subject_key']}:registered_requests_cited_together"
        self.assertIn(together_code, cited_together.exception.errors)
        self.assertEqual(one_uncited.exception.errors, ("registered_request_uncited",))
        # Each refusal names its files: the ones cited together, the ones uncited.
        rows = {
            item.code: item.explanation
            for contract in function_tool_contracts("planning_agent")
            if contract.tool_name == "record_proposal_requirements"
            for item in contract.preconditions
        }
        self.assertEqual(
            sentence_with_facts(
                rows["registered_requests_cited_together"],
                cited_together.exception.facts[together_code],
            ),
            "One scientific_investigation item stands for one registered request "
            f"file, and this item cites `{files[0]}`, `{files[1]}`: cite each file "
            "by its own item.",
        )
        self.assertEqual(
            sentence_with_facts(
                rows["registered_request_uncited"],
                one_uncited.exception.facts["registered_request_uncited"],
            ),
            "Each registered request file lists targets: record one "
            "scientific_investigation to_find whose evidence_event_ids cite that "
            f"file's event id; still uncited: `{files[1]}`.",
        )
        self.assertEqual(
            sorted(
                event_id
                for item in plan.requirements
                if item.kind == "scientific_investigation"
                for evidence in item.evidence
                for event_id in [getattr(evidence, "source_event_id", None)]
                if event_id in files
            ),
            sorted(files),
        )

    def test_a_registered_request_file_is_cited_by_a_target_or_refused(self):
        # Planning places the researcher's request file; its targets never reach
        # the Investigation beside the plan unseen.
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            file_event_id = str(uuid.uuid4())
            with self.assertRaises(ProposalPlanGroundingError) as uncited:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    draft_value,
                    registered_request_event_ids=(file_event_id,),
                )
            citing = copy.deepcopy(draft_value)
            target = requirement(citing, "scientific_investigation")
            target["knowledge"] = {
                "state": "to_find",
                "evidence_event_ids": [file_event_id],
            }
            target["acquisition_route"] = "validate_with_tool"
            plan = ground_draft(
                registered,
                result,
                inspection_event_id,
                citing,
                registered_request_event_ids=(file_event_id,),
            )
            # Without a registered request the same citation names no context event.
            with self.assertRaises(ProposalPlanGroundingError) as unknown:
                ground_draft(registered, result, inspection_event_id, citing)
            # The citing item stands for the file; a target the request names
            # beside it would be dropped, so a quote on it is refused.
            quoted = copy.deepcopy(citing)
            requirement(quoted, "scientific_investigation")["knowledge"]["quote"] = (
                "rs1234"
            )
            with self.assertRaises(ProposalPlanGroundingError) as quoted_citation:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    quoted,
                    registered_request_event_ids=(file_event_id,),
                )
            # The file stands for the targets only: another to_find item citing
            # it is refused at admission, since no later reader can find that
            # sibling of the registration in its ancestry.
            elsewhere = copy.deepcopy(citing)
            companion = requirement(elsewhere, "companion_asset")
            companion["knowledge"]["evidence_event_ids"] = [
                *companion["knowledge"]["evidence_event_ids"],
                file_event_id,
            ]
            with self.assertRaises(ProposalPlanGroundingError) as outside_targets:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    elsewhere,
                    registered_request_event_ids=(file_event_id,),
                )

        self.assertEqual(
            outside_targets.exception.errors,
            (f"{companion['subject_key']}:registered_request_not_a_target",),
        )
        self.assertEqual(uncited.exception.errors, ("registered_request_uncited",))
        self.assertEqual(
            quoted_citation.exception.errors, ("registered_request_quoted",)
        )
        self.assertEqual(
            grounded(plan, "scientific_investigation")
            .evidence[0]
            .model_dump(mode="json"),
            {"source_kind": "planning_context", "source_event_id": file_event_id},
        )
        self.assertEqual(
            unknown.exception.errors,
            ("requested_variant_rs1234:context_evidence_ungrounded",),
        )

    def test_no_item_asks_for_targets_a_registered_request_file_lists(self):
        """#482 (owner Q2 a; ADR 0049: code checks citation, never words): beside a
        registered request file, a to_find scientific_investigation citing no file
        would ask the researcher to name targets the file lists, so it is refused,
        however it is routed or quoted. The request's own asserted target stays
        beside the file (#462 F2), and with no file the same item is admitted."""
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts

        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            file_event_id = str(uuid.uuid4())
            request_event_id = registered.request.source_event_id
            beside = copy.deepcopy(draft_value)
            own_target = requirement(beside, "scientific_investigation")
            file_item = copy.deepcopy(own_target)
            file_item["subject_key"] = "targets_from_request_file"
            file_item["knowledge"] = {
                "state": "to_find",
                "evidence_event_ids": [file_event_id],
            }
            file_item["acquisition_route"] = "validate_with_tool"
            beside["requirements"].append(file_item)
            plan = ground_draft(
                registered,
                result,
                inspection_event_id,
                beside,
                registered_request_event_ids=(file_event_id,),
            )
            open_item = copy.deepcopy(own_target)
            open_item["subject_key"] = "targets_from_request"
            open_item["investigation_purpose"] = "open_question"
            open_item["knowledge"] = {
                "state": "to_find",
                "evidence_event_ids": [request_event_id],
            }
            open_item["acquisition_route"] = "ask_planning_hitl"
            open_item["validation"] = {
                "kind": "researcher_decision",
                "requirement": "the targets to investigate",
            }
            quoted = copy.deepcopy(open_item)
            quoted["knowledge"]["quote"] = "rs1234"
            tool_found = copy.deepcopy(open_item)
            tool_found["acquisition_route"] = "validate_with_tool"
            tool_found["validation"] = {
                "kind": "tool_evidence",
                "requirement": "Find the targets after Execution.",
            }
            refusals = {}
            for name, item in (
                ("target-less", open_item),
                ("quoted", quoted),
                ("tool-found", tool_found),
            ):
                shadowed = copy.deepcopy(beside)
                shadowed["requirements"].append(item)
                with self.assertRaises(ProposalPlanGroundingError) as refused:
                    ground_draft(
                        registered,
                        result,
                        inspection_event_id,
                        shadowed,
                        registered_request_event_ids=(file_event_id,),
                    )
                refusals[name] = refused.exception
            alone = copy.deepcopy(draft_value)
            alone["requirements"].append(open_item)
            without_file = ground_draft(registered, result, inspection_event_id, alone)

        code = "targets_from_request:target_question_beside_registered_request"
        self.assertEqual(
            {name: refusal.errors for name, refusal in refusals.items()},
            {name: (code,) for name in ("target-less", "quoted", "tool-found")},
        )
        rows = {
            item.code: item.explanation
            for contract in function_tool_contracts("planning_agent")
            if contract.tool_name == "record_proposal_requirements"
            for item in contract.preconditions
        }
        self.assertEqual(
            sentence_with_facts(
                rows["target_question_beside_registered_request"],
                refusals["target-less"].facts[code],
            ),
            f"A registered request file lists this request's targets (`{file_event_id}`), "
            "so no scientific_investigation asks the researcher to name them: the "
            "file's own to_find item stands for its targets, and each target the "
            "request itself names is its own asserted scientific_investigation; a "
            "target the researcher says a cited paper gives is to_find from that "
            "paper, validate_with_tool.",
        )
        self.assertEqual(
            [
                (item.subject_key, item.knowledge_state)
                for item in plan.requirements
                if item.kind == "scientific_investigation"
            ],
            [
                ("requested_variant_rs1234", "asserted"),
                ("targets_from_request_file", "to_find"),
            ],
        )
        self.assertIn(
            "targets_from_request",
            [item.subject_key for item in without_file.requirements],
        )

    def test_a_target_a_cited_paper_gives_stays_beside_a_registered_request_file(self):
        """Lane P landing on M (owner, 2026-10-30: "a yes this papaer is a valide
        target"): beside a registered request file, a to_find scientific_investigation
        the researcher says a cited paper gives (validate_with_tool, citing the
        paper's PaperRecorded, or the request while the declared paper is still to
        find) is admitted; the target question (ask_planning_hitl), and a target citing
        neither route, stay refused beside the file even when the plan cites a paper
        (#482; landing Spec r1 SP1)."""
        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            file_event_id = str(uuid.uuid4())
            paper_event_id = str(uuid.uuid4())
            request_event_id = registered.request.source_event_id
            citation = "PMID 12345678"
            prompt = f"{registered.request.prompt}\nTake the targets from {citation}."
            beside = copy.deepcopy(draft_value)
            own_target = requirement(beside, "scientific_investigation")
            file_item = copy.deepcopy(own_target)
            file_item["subject_key"] = "targets_from_request_file"
            file_item["knowledge"] = {
                "state": "to_find",
                "evidence_event_ids": [file_event_id],
            }
            file_item["acquisition_route"] = "validate_with_tool"
            beside["requirements"].append(file_item)
            cited_paper = {
                "kind": "cited_paper",
                "subject_key": "cited_paper",
                "domain": "scientific",
                "meaning": "The paper the researcher cites.",
                "knowledge": {"state": "asserted", "quote": citation},
                "acquisition_route": "validate_with_tool",
                "validation": {
                    "kind": "tool_evidence",
                    "requirement": "the cited paper retrieved",
                },
            }
            paper_target = copy.deepcopy(own_target)
            paper_target["subject_key"] = "paper_targets"
            paper_target["investigation_purpose"] = "literature_claim"
            paper_target["acquisition_route"] = "validate_with_tool"
            paper_target["validation"] = {
                "kind": "tool_evidence",
                "requirement": "the targets cited_paper gives",
            }

            def plan(target_evidence, paper_evidence, route="validate_with_tool"):
                draft = copy.deepcopy(beside)
                paper = copy.deepcopy(cited_paper)
                if paper_evidence:
                    paper["knowledge"]["evidence_event_ids"] = paper_evidence
                target = copy.deepcopy(paper_target)
                target["knowledge"] = {
                    "state": "to_find",
                    "evidence_event_ids": target_evidence,
                }
                target["acquisition_route"] = route
                if route == "ask_planning_hitl":
                    target["validation"] = {
                        "kind": "researcher_decision",
                        "requirement": "the targets to investigate",
                    }
                draft["requirements"] += [paper, target]
                return ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    draft,
                    prompt=prompt,
                    registered_request_event_ids=(file_event_id,),
                    citation_words=((request_event_id, prompt),),
                    recorded_papers=(
                        {paper_event_id: "citation"} if paper_evidence else {}
                    ),
                )

            retrieved = plan([paper_event_id], [paper_event_id])
            unretrieved = plan([request_event_id], [])
            with self.assertRaises(ProposalPlanGroundingError) as question:
                plan([request_event_id], [], route="ask_planning_hitl")
            # Neither route: the request once the paper is retrieved, or a context
            # event that is neither the request nor a paper (landing Spec r1 SP1).
            outside = {}
            for name, target_evidence, paper_evidence in (
                ("request_after_retrieval", [request_event_id], [paper_event_id]),
                (
                    "registration",
                    [registered.registered_input_set.source_event_id],
                    [],
                ),
                ("inspection", [inspection_event_id], []),
            ):
                with self.assertRaises(ProposalPlanGroundingError) as refused:
                    plan(target_evidence, paper_evidence)
                outside[name] = refused.exception.errors

        for admitted in (retrieved, unretrieved):
            self.assertCountEqual(
                [
                    (item.subject_key, item.knowledge_state)
                    for item in admitted.requirements
                    if item.kind == "scientific_investigation"
                ],
                [
                    ("paper_targets", "to_find"),
                    ("requested_variant_rs1234", "asserted"),
                    ("targets_from_request_file", "to_find"),
                ],
            )
        code = "paper_targets:target_question_beside_registered_request"
        self.assertEqual(question.exception.errors, (code,))
        self.assertEqual(
            outside,
            {
                name: (code,)
                for name in ("request_after_retrieval", "registration", "inspection")
            },
        )

    def test_one_registered_request_file_is_cited_by_one_item(self):
        """#482 (owner Q2 a), from #367's one item per file: a second item citing the
        same file would stand for its targets twice, so it is refused."""
        from bio_neb.agentic_journey.tool_execution import function_tool_contracts
        from bio_neb.failure import sentence_with_facts

        with tempfile.TemporaryDirectory() as directory:
            registered, result, inspection_event_id, draft_value = grounding_fixture(
                Path(directory).resolve()
            )
            file_event_id = str(uuid.uuid4())
            twice = copy.deepcopy(draft_value)
            target = requirement(twice, "scientific_investigation")
            target["knowledge"] = {
                "state": "to_find",
                "evidence_event_ids": [file_event_id],
            }
            target["acquisition_route"] = "validate_with_tool"
            second = copy.deepcopy(target)
            second["subject_key"] = "targets_from_request_file_again"
            twice["requirements"].append(second)
            with self.assertRaises(ProposalPlanGroundingError) as refused:
                ground_draft(
                    registered,
                    result,
                    inspection_event_id,
                    twice,
                    registered_request_event_ids=(file_event_id,),
                )

        self.assertEqual(refused.exception.errors, ("registered_request_cited_twice",))
        rows = {
            item.code: item.explanation
            for contract in function_tool_contracts("planning_agent")
            if contract.tool_name == "record_proposal_requirements"
            for item in contract.preconditions
        }
        self.assertEqual(
            sentence_with_facts(
                rows["registered_request_cited_twice"],
                refused.exception.facts["registered_request_cited_twice"],
            ),
            "One scientific_investigation item stands for each registered request "
            f"file, and more than one item cites `{file_event_id}`: cite each file "
            "by one item.",
        )

    def test_requirement_kind_and_extra_fields_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            mislabeled = copy.deepcopy(draft_value)
            for item in mislabeled["requirements"]:
                if item["kind"] == "registered_input":
                    item["kind"] = "research_goal"
            with self.assertRaises(ValidationError):
                ProposalInformationPlanDraft.model_validate(mislabeled)
            invalid = copy.deepcopy(draft_value)
            invalid["requirements"][0]["extra"] = True
            with self.assertRaises(ValidationError):
                ProposalInformationPlanDraft.model_validate(invalid)
            unknown_state = copy.deepcopy(draft_value)
            requirement(unknown_state, "analysis_recipe")["knowledge"]["state"] = (
                "unknown"
            )
            with self.assertRaises(ValidationError):
                ProposalInformationPlanDraft.model_validate(unknown_state)

    def test_to_find_observed_and_identity_knowledge_require_lawful_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            to_find_without_validation = copy.deepcopy(draft_value)
            requirement(to_find_without_validation, "analysis_recipe")["validation"] = {
                "kind": "none"
            }
            with self.assertRaises(ValidationError):
                ProposalInformationPlanDraft.model_validate(to_find_without_validation)

            reconstructed_tool_fact = copy.deepcopy(draft_value)
            requirement(reconstructed_tool_fact, "companion_asset")[
                "acquisition_route"
            ] = "derive_deterministically"
            with self.assertRaises(ValidationError):
                ProposalInformationPlanDraft.model_validate(reconstructed_tool_fact)

            asserted_identity_without_validation = copy.deepcopy(draft_value)
            reference = requirement(
                asserted_identity_without_validation, "reference_assembly_identity"
            )
            reference["knowledge"] = {
                "state": "asserted",
                "value": "GCF_000001405.40",
                "quote": "GCF_000001405.40",
            }
            reference["acquisition_route"] = "ask_planning_hitl"
            reference["validation"] = {"kind": "none"}
            with self.assertRaises(ValidationError):
                ProposalInformationPlanDraft.model_validate(
                    asserted_identity_without_validation
                )

    def test_a_target_no_source_can_supply_is_refused_naming_its_routes(self):
        # A target the request does not state is found after Execution or asked of
        # the researcher; the request's own words cannot supply it, so admission
        # refuses that route and names the lawful ones.
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            draft = copy.deepcopy(draft_value)
            target = requirement(draft, "scientific_investigation")
            target["knowledge"] = {"state": "to_find", "evidence_event_ids": ["e"]}
            target["acquisition_route"] = "use_supplied_assertion"
            with self.assertRaises(ValidationError) as refused:
                ProposalInformationPlanDraft.model_validate(draft)
            message = str(refused.exception)
            self.assertIn("no Tool, fact or Review can ever supply", message)
            self.assertIn("ask_planning_hitl", message)
            # The lawful shapes stay admitted.
            target["acquisition_route"] = "ask_planning_hitl"
            target["validation"] = {
                "kind": "researcher_decision",
                "requirement": "The researcher names the targets.",
            }
            ProposalInformationPlanDraft.model_validate(draft)
            ProposalInformationPlanDraft.model_validate(draft_value)

    def test_no_kind_uses_an_accepted_pre_parabricks_fact_nothing_produces(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            for index, item in enumerate(draft_value["requirements"]):
                with self.subTest(kind=item["kind"]):
                    draft = copy.deepcopy(draft_value)
                    draft["requirements"][index]["acquisition_route"] = (
                        "use_accepted_pre_parabricks_fact"
                    )
                    with self.assertRaises(ValidationError) as refused:
                        ProposalInformationPlanDraft.model_validate(draft)
                    self.assertIn(
                        "acquisition_route='use_accepted_pre_parabricks_fact'",
                        str(refused.exception),
                    )

    def test_an_input_nobody_registered_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            draft = copy.deepcopy(draft_value)
            draft["requirements"].append(
                {
                    "kind": "registered_input",
                    "subject_key": "additional_input",
                    "domain": "operational",
                    "meaning": "A further FASTQ input.",
                    "knowledge": {"state": "to_find", "evidence_event_ids": ["e"]},
                    "acquisition_route": "validate_with_tool",
                    "validation": {
                        "kind": "tool_evidence",
                        "requirement": "Find the further FASTQ input.",
                    },
                }
            )
            with self.assertRaises(ValidationError) as refused:
                ProposalInformationPlanDraft.model_validate(draft)
            self.assertIn("registered_input cannot be to_find", str(refused.exception))

    def test_a_kind_whose_tool_binds_one_requirement_is_recorded_once(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            for kind in (
                "sample_identity",
                "taxon_identity",
                "reference_assembly_identity",
                "companion_asset",
            ):
                with self.subTest(kind=kind):
                    draft = copy.deepcopy(draft_value)
                    second = copy.deepcopy(requirement(draft, kind))
                    second["subject_key"] = "second_" + kind
                    draft["requirements"].append(second)
                    with self.assertRaises(ValidationError) as refused:
                        ProposalInformationPlanDraft.model_validate(draft)
                    self.assertIn(
                        f"more than one {kind} requirement", str(refused.exception)
                    )
            # Several targets are one collection: each is its own item.
            targets = copy.deepcopy(draft_value)
            second = copy.deepcopy(requirement(targets, "scientific_investigation"))
            second["subject_key"] = "second_target"
            targets["requirements"].append(second)
            ProposalInformationPlanDraft.model_validate(targets)

    def test_only_a_tool_or_downstream_fact_derives_a_value(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, draft_value = grounding_fixture(Path(directory).resolve())
            for kind in ("research_goal",):
                with self.subTest(kind=kind):
                    draft = copy.deepcopy(draft_value)
                    requirement(draft, kind)["acquisition_route"] = (
                        "derive_deterministically"
                    )
                    with self.assertRaises(ValidationError) as refused:
                        ProposalInformationPlanDraft.model_validate(draft)
                    self.assertIn(
                        "acquisition_route='derive_deterministically'",
                        str(refused.exception),
                    )
            # The inspection Tool supplies an observed input's value.
            observed = copy.deepcopy(draft_value)
            requirement(observed, "registered_input")["acquisition_route"] = (
                "derive_deterministically"
            )
            ProposalInformationPlanDraft.model_validate(observed)


if __name__ == "__main__":
    unittest.main()
