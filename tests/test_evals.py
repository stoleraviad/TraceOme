"""The eval package: case records and compilation, the planning scorer and run preparation.

Rule owners: ADR 0042 (scores assist the review of recorded evidence) and
`docs/agents/testing.md#live-separation` (eval runs are eval evidence). The scorer is
qualified against a real planning Journey written by the product's own offline path, with
deliberately wrong controls (eval contract section 7).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from bio_neb import cli
from bio_neb.devtools import live_readiness
from bio_neb.parabricks.proposal_information import (
    REQUIRED_PROPOSAL_INFORMATION_KINDS,
    ProposalInformationKind,
)
from bio_neb.recording import ContactJournal, ContactScope
from evals import case as eval_case
from evals import score as eval_score
from evals.core_files import load_ledger, scenario_files
from evals.prepare import MODELS, prepare
from evals.score import ScorerError, replies, score_planning
from evals.watch import Limits, cut_replies, reply_rows, source_gets, trips, watch
from tests.clock_check_fakes import steady_clock, write_clock_record
from tests.planning_assignment_fakes import execution_configuration
from tests.proposal_information_fakes import runtime_configuration
from tests.run_prerequisites import RunPrerequisites

REPOSITORY = Path(__file__).resolve().parents[1]
DONOR = REPOSITORY / "work/research/2026-09-24-nemotron/eval"
# Beyond the human cases: a haploid bacterium (s20), a plant (s7) and a
# non-human mammal named by words (s16).
DEV_SUBSET = (
    "s1-pig-clean-L2",
    "s12-human-hg005-clean-L3",
    "s16-dog-asprv1-clean-L3",
    "s2-giab-hg002-clean-L2-snp3",
    "s20-bsubtilis-t30r7-clean-L2-snp3",
    "s5-celegans-ma1a-clean-L2-snp4",
    "s7-rice-albatros-clean-L2-snp5",
)


def _case_document(case_id: str) -> dict:
    path = eval_case.CASES / "planning" / f"{case_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class CaseRecordTests(unittest.TestCase):
    def test_the_dev_subset_is_the_owner_selected_train_cases(self):
        splits = json.loads((DONOR / "corpus/splits.json").read_text())["scenarios"]
        cases = [eval_case.load_case(path) for path in eval_case.case_paths()]

        self.assertEqual(tuple(c.case_id for c in cases), DEV_SUBSET)
        for case in cases:
            self.assertEqual(splits[case.scenario], "train", case.case_id)

    def test_generated_schema_and_compiled_cases_match_the_case_files(self):
        for path, text in eval_case.expected_outputs().items():
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(encoding="utf-8"), text)
        written = set(eval_case.COMPILED.rglob("*.json"))
        self.assertEqual(
            written, set(eval_case.expected_outputs()) - {eval_case.SCHEMA}
        )

    def test_a_request_wrapped_inside_a_sentence_is_refused(self):
        # The donor s1 request, as its YAML block delivers it.
        donor = yaml.safe_load(
            (DONOR / "batch1/cases/s1-pig-clean-L2.yaml").read_text()
        )
        wrapped = donor["public"]["research_request"]
        self.assertIn("germline\nvariant", wrapped)
        document = _case_document("s1-pig-clean-L2")
        document["public"]["research_request"] = wrapped

        with self.assertRaisesRegex(ValidationError, "line break inside a sentence"):
            eval_case.EvalCase.model_validate(document)

    def test_each_dev_request_states_the_donor_words_on_unbroken_sentences(self):
        for case_id in DEV_SUBSET:
            case = eval_case.load_case(eval_case.CASES / "planning" / f"{case_id}.yaml")
            donor_path = REPOSITORY / case.donor.path
            donor = yaml.safe_load(donor_path.read_text(encoding="utf-8"))
            with self.subTest(case=case_id):
                self.assertEqual(
                    case.public.research_request.split(),
                    donor["public"]["research_request"].split(),
                )
                self.assertEqual(case.public.research_request.count("\n"), 1)

    def test_a_case_label_in_the_request_is_refused(self):
        document = _case_document("s2-giab-hg002-clean-L2-snp3")
        document["public"]["research_request"] = "Plan s2-giab-hg002 for HG002.\n"

        with self.assertRaisesRegex(ValidationError, "leaks into the request"):
            eval_case.EvalCase.model_validate(document)

    def test_a_validation_or_sealed_case_is_refused(self):
        for split in ("validation", "sealed"):
            document = _case_document("s1-pig-clean-L2")
            document["split"] = split
            with self.subTest(split=split), self.assertRaises(ValidationError):
                eval_case.EvalCase.model_validate(document)

    def test_bank_kinds_are_the_products_required_kinds(self):
        bank = eval_case.load_bank()
        self.assertEqual(
            set(bank["B-ALL-KINDS"].spec["kinds"]), REQUIRED_PROPOSAL_INFORMATION_KINDS
        )
        kinds = set(get_args(ProposalInformationKind))
        for path in eval_case.case_paths():
            compiled = eval_case.compile_case(eval_case.load_case(path), bank)
            named = {answer.review for answer in compiled.planning.prepared_answers} & (
                kinds | {"proposal_output"}
            )
            for question in compiled.planning.questions:
                spec = question.spec or {}
                named |= set(spec.get("asserted", ())) | set(spec.get("to_find", ()))
                named |= {spec["kind"]} if "kind" in spec else set()
            with self.subTest(case=compiled.case_id):
                self.assertLessEqual(named, kinds)

    def test_compiling_inlines_the_bank_with_the_case_params(self):
        bank = eval_case.load_bank()
        compiled = eval_case.compile_case(
            eval_case.load_case(
                eval_case.CASES / "planning" / "s12-human-hg005-clean-L3.yaml"
            ),
            bank,
        )
        supplied = next(
            q for q in compiled.planning.questions if q.id == "B-SUPPLIED-KEPT"
        )

        self.assertEqual(supplied.ask, bank["B-SUPPLIED-KEPT"].ask)
        self.assertEqual(supplied.spec, {"asserted": ["sample_identity"]})
        self.assertEqual(compiled.planning.gates, ("accepted", "usable"))

    def test_an_unknown_reference_missing_params_or_unknown_bad_answer_id_is_refused(
        self,
    ):
        bank = eval_case.load_bank()
        edits = {
            "unknown bank question B-NOPE": lambda q: q.append({"ref": "B-NOPE"}),
            "B-TERMINAL needs params": lambda q: q.append({"ref": "B-TERMINAL"}),
        }
        for message, edit in edits.items():
            document = _case_document("s1-pig-clean-L2")
            document["planning"]["questions"] = [
                q
                for q in document["planning"]["questions"]
                if q.get("ref") != "B-TERMINAL"
            ]
            edit(document["planning"]["questions"])
            case = eval_case.EvalCase.model_validate(document)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                eval_case.compile_case(case, bank)
        document = _case_document("s1-pig-clean-L2")
        document["planning"]["bad_answers"][0]["must_fail"] = ["q9"]
        with self.assertRaisesRegex(ValueError, r"bad answer names \['q9'\]"):
            eval_case.compile_case(eval_case.EvalCase.model_validate(document), bank)


def _baseline_case(**planning_overrides) -> eval_case.CompiledCase:
    """A case for the offline planning baseline's request, judged by its own words."""
    questions = [
        {"ref": "B-ALL-KINDS"},
        {
            "ref": "B-SUPPLIED-KEPT",
            "params": {"asserted": ["sample_identity"]},
        },
        {
            "ref": "B-UNSUPPLIED-TO-FIND",
            "params": {"to_find": ["taxon_identity", "reference_assembly_identity"]},
        },
        {"ref": "B-FIRST-ATTEMPT"},
        {"ref": "B-NO-INVENT"},
        {"ref": "B-TERMINAL", "params": {"terminal": "execution_review"}},
        {"ref": "B-REVIEWS-RAISED", "params": {"reviews": ["analysis_recipe"]}},
        {
            "id": "sample",
            "ask": "Is sample-001 asserted?",
            "check": "plan_value",
            "spec": {
                "kind": "sample_identity",
                "state": "asserted",
                "value": "sample-001",
            },
            "gate": "hard",
        },
        {
            "id": "taxon",
            "ask": "Is 9606 resolved?",
            "check": "taxon_resolved",
            "spec": {"ncbi_tax_id": 9606},
            "gate": "hard",
        },
        {
            "id": "reference",
            "ask": "Is GCF_000001405.40 verified and taken from discovery?",
            "check": "reference_from_discovery",
            "spec": {"accession": "GCF_000001405.40"},
            "gate": "hard",
        },
        {
            "id": "verified",
            "ask": "Is GCF_000001405.40 the current verified reference?",
            "check": "reference_verified",
            "spec": {"accession": "GCF_000001405.40"},
            "gate": "hard",
        },
    ]
    document = {
        "schema_version": "bio-neb.eval-case/1",
        "case_id": "s0-offline-baseline",
        "family": "offline",
        "scenario": "offline-baseline",
        "split": "train",
        "level": "L3",
        "evidence_shape": "clean",
        "donor": {
            "path": "tests/run_prerequisites.py",
            "case_id": "none",
            "version": 1,
        },
        "refresh": ["test case"],
        "public": {
            "research_request": "Analyze rs1234 for sample-001 Homo sapiens and produce VCF.",
            "fastq_roles": ["read_1", "read_2"],
        },
        "planning": {
            "prepared_answers": [
                {
                    "review": "analysis_recipe",
                    "answer": "gDNA / paired",
                    "reason": "test",
                }
            ],
            "questions": questions,
            "good_answer": "test",
            **planning_overrides,
        },
    }
    return eval_case.compile_case(
        eval_case.EvalCase.model_validate(document), eval_case.load_bank()
    )


def _verdicts(score: dict) -> dict[str, str]:
    found = {q["id"]: q["verdict"] for q in score["questions"]}
    found.update({name: gate["verdict"] for name, gate in score["gates"].items()})
    return found


class ScorePlanningTests(RunPrerequisites, unittest.TestCase):
    """The product's offline planning path writes the Journey; the scripted model only chooses."""

    def setUp(self):
        journey_file = self.prerequisite("review-pending").journey_file
        self.events = [
            json.loads(line) for line in journey_file.read_text().splitlines()
        ]
        requirements = self._plan(self.events)["payload"]["plan"]["requirements"]
        if not any(r["kind"] == "scientific_investigation" for r in requirements):
            requirements.append(
                {
                    "requirement_id": "requirement_test_rs1234",
                    "kind": "scientific_investigation",
                    "knowledge_state": "asserted",
                    "value": "rs1234",
                    "investigation_purpose": "known_variant",
                    "evidence": [
                        {"source_kind": "research_request", "exact_text": "rs1234"}
                    ],
                }
            )
        self.case = _baseline_case()

    def _without_items(self, events):
        plan = self._plan(events)["payload"]["plan"]
        plan["requirements"] = [
            r for r in plan["requirements"] if r["kind"] != "scientific_investigation"
        ]
        return events

    def _plan(self, events):
        return next(
            e for e in events if e["event_type"] == "ProposalRequirementsIdentified"
        )

    def test_a_clean_planning_run_passes_both_gates_and_every_question(self):
        score = score_planning(self.case, self.events)

        self.assertEqual(set(_verdicts(score).values()), {"pass"}, _verdicts(score))
        self.assertEqual(score["diagnostics"]["terminal"], "execution_review")
        self.assertEqual(score["diagnostics"]["planning_reviews"], ["analysis_recipe"])
        self.assertEqual(
            score["gates"]["usable"]["detail"]["atomic"], {"sample_identity": True}
        )

    def test_the_ploidy_notice_is_scored_by_its_cited_record_and_its_question(self):
        """#470 (owner Q7 a): the Review states diploid calling with the model's
        judgment citing an organism record by words it holds, and asks exactly when the
        judgment is not diploid."""
        question = {
            "id": "ploidy",
            "ask": "Does the Review state ploidy and ask only for a non-diploid organism?",
            "check": "ploidy_notice",
            "spec": {"asks": False},
            "gate": "hard",
        }

        def verdict(events, asks):
            case = _baseline_case(questions=[{**question, "spec": {"asks": asks}}])
            return _verdicts(score_planning(case, events))["ploidy"]

        def ploidy(events):
            return next(
                e for e in events if e["event_type"] == "ExecutionProposalSubmitted"
            )["payload"]["proposal"]["ploidy"]

        def changed(change):
            events = copy.deepcopy(self.events)
            change(ploidy(events), events)
            return events

        def asking(notice, _):
            notice["organism"]["judgment"] = "likely_non_diploid"
            notice["question"] = "Is this organism diploid?"

        self.assertEqual(ploidy(self.events)["organism"]["judgment"], "diploid")
        self.assertEqual(verdict(self.events, False), "pass")
        self.assertEqual(verdict(self.events, True), "fail")
        asked = changed(asking)
        self.assertEqual(verdict(asked, True), "pass")
        self.assertEqual(verdict(asked, False), "fail")
        wrong = {
            "a diploid judgment that asks": lambda notice, _: notice.update(
                question="Is this organism diploid?"
            ),
            "no statement": lambda notice, _: notice.update(statement=""),
            "another calling ploidy": lambda notice, _: notice.update(calling_ploidy=4),
            "words the record does not hold": lambda notice, _: notice[
                "organism"
            ].update(quote="not a word in this record"),
            "no organism record": lambda notice, events: notice["organism"].update(
                record_event_id=self._plan(events)["event_id"]
            ),
        }
        for name, change in wrong.items():
            with self.subTest(name):
                self.assertEqual(verdict(changed(change), False), "fail")
        with self.subTest("a non-diploid judgment with a null question"):
            unasked = changed(asking)
            ploidy(unasked)["question"] = None
            self.assertEqual(verdict(unasked, True), "fail")
        unstated = copy.deepcopy(self.events)
        next(e for e in unstated if e["event_type"] == "ExecutionProposalSubmitted")[
            "payload"
        ]["proposal"].pop("ploidy")
        self.assertEqual(verdict(unstated, False), "fail")

    def test_a_widened_sample_value_fails_usable_and_the_sample_question(self):
        events = copy.deepcopy(self.events)
        sample = next(
            r
            for r in self._plan(events)["payload"]["plan"]["requirements"]
            if r["kind"] == "sample_identity"
        )
        sample["value"] = "the sample sample-001"

        verdicts = _verdicts(score_planning(self.case, events))

        self.assertEqual((verdicts["usable"], verdicts["sample"]), ("fail", "fail"))
        self.assertEqual(verdicts["accepted"], "pass")

    def test_a_failed_companion_lookup_fails_usable(self):
        # Nemotron s2 (run 244-eval-nemotron-s2-...-9e817c4b-a1): NCBI returned HTTP
        # 500 three times; the failed result carries no outcome.
        events = copy.deepcopy(self.events)
        for event in events:
            if event["event_type"] == "CompanionObservationRecorded":
                result = event["payload"]["result"]
                event["payload"]["result"] = {
                    "evidence_id": "companion-failure:x",
                    "status": "failed",
                    "summary": "Official NCBI metadata returned HTTP 500.",
                    "work_item": result["work_item"],
                    "failure": {"code": "http_error", "failure_class": "exhausted"},
                }

        score = score_planning(self.case, events)

        self.assertEqual(_verdicts(score)["usable"], "fail")
        self.assertEqual(_verdicts(score)["accepted"], "pass")

    def test_a_plan_whose_consumers_never_ran_is_not_reached_for_usable(self):
        # Nemotron s5 control (run 244-eval-nemotron-s5-...-4bbde994-a2) stopped in
        # reference before any Tool; its values were never consumed, not refused.
        consumers = {
            "SampleAttestationRecorded",
            "TaxonResolutionRecorded",
            "ReferenceDescriptionRecorded",
            "CompanionObservationRecorded",
        }
        events = [
            e for e in copy.deepcopy(self.events) if e["event_type"] not in consumers
        ]

        verdicts = _verdicts(score_planning(self.case, events))

        self.assertEqual(verdicts["usable"], "not_reached")
        self.assertEqual(verdicts["accepted"], "pass")

    def test_a_plan_whose_companion_stage_never_ran_is_not_reached_for_usable(self):
        # A recorded Nemotron s2 Journey: sample, taxon and reference were consumed,
        # then the Attempt held before the companions stage.
        events = [
            e
            for e in copy.deepcopy(self.events)
            if e["event_type"] != "CompanionObservationRecorded"
        ]

        score = score_planning(self.case, events)

        self.assertEqual(_verdicts(score)["usable"], "not_reached")
        self.assertEqual(score["gates"]["usable"]["detail"]["unreached"], ["companion"])

    def test_a_refused_sample_fails_usable_though_the_companion_never_ran(self):
        events = [
            e
            for e in copy.deepcopy(self.events)
            if e["event_type"] != "CompanionObservationRecorded"
        ]
        for event in events:
            if event["event_type"] == "SampleAttestationRecorded":
                event["payload"]["result"]["status"] = "refused"

        self.assertEqual(_verdicts(score_planning(self.case, events))["usable"], "fail")

    def test_an_accession_no_source_returned_fails_no_invent_and_discovery(self):
        events = copy.deepcopy(self.events)
        call = next(
            e
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "discover_assemblies"
        )
        # The model types an accession from memory before any discovery result.
        call["payload"]["arguments"]["note"] = "use GCA_000001405.15"
        call["payload"]["arguments"]["also"] = "GCF_000001405.40"

        score = score_planning(self.case, events)
        verdicts = _verdicts(score)

        self.assertEqual(
            (verdicts["B-NO-INVENT"], verdicts["reference"]), ("fail", "fail")
        )
        invented = next(q for q in score["questions"] if q["id"] == "B-NO-INVENT")
        self.assertEqual(
            {i["value"] for i in invented["detail"]["invented"]},
            {"GCA_000001405.15", "GCF_000001405.40"},
        )

    def test_identifiers_are_read_with_the_products_rules(self):
        # The product's rsID rule reads `rsID:987654321` as rs987654321 (Standards r1 S2).
        events = copy.deepcopy(self.events)
        call = next(
            e
            for e in events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "discover_assemblies"
        )
        call["payload"]["arguments"]["note"] = "check rsID:987654321 first"

        invented = next(
            q
            for q in score_planning(self.case, events)["questions"]
            if q["id"] == "B-NO-INVENT"
        )

        self.assertEqual(invented["verdict"], "fail")
        self.assertEqual(
            invented["detail"]["invented"],
            [{"tool": "discover_assemblies", "value": "rs987654321"}],
        )

    def test_a_refused_proposal_turn_fails_first_attempt_only(self):
        events = copy.deepcopy(self.events)
        index = events.index(self._plan(events))
        events.insert(
            index,
            {
                "event_type": "AgentActionTurnDispositionRecorded",
                "event_id": "refused-turn",
                "payload": {
                    "disposition": "singleton_refused",
                    "correction_key": "record_proposal_requirements: proposal_requirement_kinds_incomplete",
                },
            },
        )

        score = score_planning(self.case, events)
        failed = {k for k, v in _verdicts(score).items() if v != "pass"}

        self.assertEqual(failed, {"B-FIRST-ATTEMPT"})

    def test_a_refused_proposal_turn_with_no_plan_fails_first_attempt(self):
        # Nemotron s5 (run 244-eval-nemotron-s5-...-4bbde994-a1) was refused twice
        # and stopped before any plan; the first attempt still failed.
        index = self.events.index(self._plan(self.events))
        events = copy.deepcopy(self.events[:index])
        events.append(
            {
                "event_type": "AgentActionTurnDispositionRecorded",
                "event_id": "refused-turn",
                "payload": {
                    "disposition": "singleton_refused",
                    "correction_key": "record_proposal_requirements: tool_arguments_invalid",
                },
            },
        )

        verdicts = _verdicts(score_planning(self.case, events))

        self.assertEqual(verdicts["B-FIRST-ATTEMPT"], "fail")
        self.assertEqual(verdicts["B-ALL-KINDS"], "not_reached")
        self.assertEqual(
            _verdicts(score_planning(self.case, copy.deepcopy(self.events[:index])))[
                "B-FIRST-ATTEMPT"
            ],
            "not_reached",
        )

    def test_reviews_are_named_by_requirement_kind_not_the_models_subject_key(self):
        # Nemotron s2 (Journey of run 282-eval-nemotron-s2-...-a1) named its recipe
        # requirement `germline_variant_calling`; the Review is still the recipe Review.
        events = copy.deepcopy(self.events)
        plan = self._plan(events)["payload"]["plan"]
        recipe = next(r for r in plan["requirements"] if r["kind"] == "analysis_recipe")
        recipe["subject_key"] = "germline_variant_calling"
        for event in events:
            if event["event_type"] == "PlanningHitlRequested":
                event["payload"]["subject_key"] = "germline_variant_calling"

        score = score_planning(self.case, events)

        self.assertEqual(_verdicts(score)["B-REVIEWS-RAISED"], "pass")
        self.assertEqual(score["diagnostics"]["planning_reviews"], ["analysis_recipe"])

    def test_a_review_without_its_plan_binding_is_refused(self):
        for name, edit in {
            "missing": lambda payload: payload.pop("requirement_id"),
            "unknown": lambda payload: payload.update(requirement_id="requirement_x"),
        }.items():
            events = copy.deepcopy(self.events)
            review = next(
                e for e in events if e["event_type"] == "PlanningHitlRequested"
            )
            edit(review["payload"])
            with self.subTest(name), self.assertRaisesRegex(ScorerError, "requirement"):
                score_planning(self.case, events)

    def test_a_stop_before_the_execution_review_names_the_pending_review(self):
        cut = next(
            i
            for i, e in enumerate(self.events)
            if e["event_type"] == "PlanningHitlResponded"
        )

        score = score_planning(self.case, self.events[:cut])
        terminal = next(q for q in score["questions"] if q["id"] == "B-TERMINAL")

        self.assertEqual(terminal["verdict"], "fail")
        self.assertEqual(
            terminal["detail"], {"observed": "planning_review:analysis_recipe"}
        )

    def test_without_an_accepted_plan_the_plan_questions_are_not_reached(self):
        # A run that stopped before the product accepted any plan.
        events = self.events[: self.events.index(self._plan(self.events))]

        verdicts = _verdicts(score_planning(self.case, events))

        self.assertEqual(verdicts["accepted"], "fail")
        for name in (
            "usable",
            "B-ALL-KINDS",
            "B-SUPPLIED-KEPT",
            "B-FIRST-ATTEMPT",
            "sample",
        ):
            self.assertEqual(verdicts[name], "not_reached", name)

    def _planning_attempt(self, events) -> str:
        return next(
            e["payload"]["attempt_id"]
            for e in events
            if e["event_type"] == "AttemptStarted"
            and e["payload"]["agent_role"] == "planning_agent"
        )

    def test_a_cut_off_reply_before_the_plan_counts_in_the_ledger_and_fails_first_attempt(
        self,
    ):
        # The product's fault ledger counts it (`model_output_faults`); a refusing turn
        # fault before the plan fails B-FIRST-ATTEMPT.
        events = copy.deepcopy(self.events)
        events.insert(
            events.index(self._plan(events)),
            {
                "event_type": "ModelCompletionTruncated",
                "event_id": "cut-reply",
                "payload": {
                    "attempt_id": self._planning_attempt(events),
                    "correction_count": 1,
                    "request_number": 2,
                    "completion_tokens": 230158,
                    "requested_max_tokens": 245760,
                    "response_event_id": "response-2",
                    "feedback": "cut",
                },
            },
        )

        score = score_planning(self.case, events)
        failed = {k for k, v in _verdicts(score).items() if v != "pass"}
        first = next(q for q in score["questions"] if q["id"] == "B-FIRST-ATTEMPT")

        self.assertEqual(failed, {"B-FIRST-ATTEMPT"})
        self.assertEqual(
            first["detail"],
            {
                "refused_before_plan": [],
                "turn_faults_before_plan": ["output_truncated"],
            },
        )
        self.assertIn(
            {
                "attempt": self._planning_attempt(events),
                "key": "output_truncated",
                "count": 1,
                "event_id": "cut-reply",
                "source": "ModelCompletionTruncated",
            },
            score["diagnostics"]["turn_faults"],
        )
        self.assertEqual(score["schema_version"], "bio-neb.eval-score/2")

    def test_a_held_run_reads_run_held(self):
        # A recorded Journey: the interrupted Run closed its Pass `held`.
        events = self.events[: self.events.index(self._plan(self.events))] + [
            {
                "event_type": "PassClosed",
                "event_id": "held-pass",
                "payload": {
                    "disposition": "held",
                    "failure": {"failure_class": "interrupted"},
                },
            }
        ]

        score = score_planning(self.case, events)
        terminal = next(q for q in score["questions"] if q["id"] == "B-TERMINAL")

        self.assertEqual(terminal["detail"], {"observed": "run_held:interrupted"})
        self.assertEqual(score["diagnostics"]["terminal"], "run_held:interrupted")

    def test_scoring_never_uses_the_checked_reader(self):
        # Old Journeys stop reading on main's checked reader; the raw scorer must still
        # score them. Shapes main refuses: a checkpoint with `output_hold_event_id`, a
        # `proposal_output` item, a cut-off retry.
        events = copy.deepcopy(self.events)
        self._plan(events)["payload"]["plan"]["requirements"].append(
            {
                "requirement_id": "requirement_output",
                "kind": "proposal_output",
                "knowledge_state": "asserted",
                "value": "VCF",
                "evidence": [],
            }
        )
        events.insert(
            events.index(self._plan(events)),
            {
                "event_type": "ModelCompletionTruncated",
                "event_id": "cut-reply",
                "payload": {
                    "attempt_id": self._planning_attempt(events),
                    "correction_count": 1,
                    "request_number": 2,
                    "response_event_id": "response-2",
                },
            },
        )
        events.append(
            {
                "event_type": "SliceCheckpointReached",
                "event_id": "checkpoint",
                "payload": {"output_hold_event_id": None},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            journey = Path(directory) / "journey.jsonl"
            journey.write_bytes(
                b"".join(json.dumps(e).encode() + b"\n" for e in events)
            )
            output = Path(directory) / "score.json"
            refuse = AssertionError("the checked reader was used")
            with (
                patch("bio_neb.application.queries.observe", side_effect=refuse),
                patch("bio_neb.journey.Journey.open", side_effect=refuse),
                patch.object(eval_score, "load_compiled", return_value=self.case),
            ):
                code, _, _ = _invoke(
                    eval_score.main,
                    [
                        "--case",
                        "s2-giab-hg002-clean-L2-snp3",
                        "--journey-file",
                        str(journey),
                        "--output",
                        str(output),
                    ],
                )
            score = json.loads(output.read_text())
            content = journey.read_bytes()

        compiled = (
            REPOSITORY / "evals/compiled/planning/s2-giab-hg002-clean-L2-snp3.json"
        )
        self.assertEqual(code, 0)
        self.assertEqual(score["journey"]["sha256"], _sha256(content))
        # The compiled case's digest keeps a score readable after a gold refresh (RO3).
        self.assertEqual(score["case"], {"sha256": _sha256(compiled.read_bytes())})
        self.assertEqual(
            [f["key"] for f in score["diagnostics"]["turn_faults"]],
            ["output_truncated"],
        )

    def test_positional_items_whose_value_is_their_quote_pass(self):
        # The product types a prose item's value as its quote (Nemotron s5 run, Journey
        # 1e0e8314): four sites, four items, no rsID.
        sites = ["V:2129317 C>T", "X:765242 C>A"]
        events = self._without_items(copy.deepcopy(self.events))
        requirements = self._plan(events)["payload"]["plan"]["requirements"]
        for site in sites:
            requirements.append(
                {
                    "requirement_id": f"requirement_test_{site}",
                    "kind": "scientific_investigation",
                    "knowledge_state": "asserted",
                    "value": site,
                    "investigation_purpose": "known_variant",
                    "evidence": [
                        {"source_kind": "research_request", "exact_text": site}
                    ],
                }
            )
        question = {
            "id": "items",
            "ask": "Are the sites separate known_variant items with no rsID?",
            "check": "investigation_items",
            "spec": {"purpose": "known_variant", "quotes": sites},
            "gate": "hard",
        }
        case = _baseline_case(questions=[question])

        self.assertEqual(_verdicts(score_planning(case, events))["items"], "pass")
        requirements[-1]["value"] = "rs12345"
        self.assertEqual(_verdicts(score_planning(case, events))["items"], "fail")

    def test_merged_or_dropped_investigation_items_fail(self):
        question = {
            "id": "items",
            "ask": "Is rs1234 one known_variant item?",
            "check": "investigation_items",
            "gate": "hard",
        }
        cases = {
            "values": ({"purpose": "known_variant", "values": ["rs1234"]}, "fail"),
            "quotes": ({"purpose": "known_variant", "quotes": ["rs1234 for"]}, "fail"),
        }
        # A plan that drops the rs1234 item.
        dropped = self._without_items(copy.deepcopy(self.events))
        for name, (spec, expected) in cases.items():
            case = _baseline_case(questions=[{**question, "spec": spec}])
            with self.subTest(name):
                self.assertEqual(
                    _verdicts(score_planning(case, dropped))["items"], expected
                )

        events = self.events
        for name, spec in {
            "values": {"purpose": "known_variant", "values": ["rs1234"]},
            "positional with an rsID": {
                "purpose": "known_variant",
                "quotes": ["rs1234"],
            },
        }.items():
            case = _baseline_case(questions=[{**question, "spec": spec}])
            with self.subTest(name):
                expected = "pass" if name == "values" else "fail"
                self.assertEqual(
                    _verdicts(score_planning(case, events))["items"], expected
                )


def _invoke(main, argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        patch("socket.socket", side_effect=AssertionError("network contact")),
        patch("subprocess.Popen", side_effect=AssertionError("process contact")),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PrepareTests(unittest.TestCase):
    RUN_ID = "282-eval-nemotron-s12-20261013T120000Z-f7e61de5-a1"

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.home = self.root / "home"
        (self.home / "live-runs").mkdir(parents=True)
        # Readiness reads names only: the donor's credential variable.
        (self.home / "model.env").write_text("LOCAL_MODEL_KEY=\n", encoding="utf-8")
        write_clock_record(self.home)
        self.enterContext(steady_clock())
        configuration = execution_configuration(self.root)
        self.donor = runtime_configuration()
        self.donor["parabricks"] = configuration.model_dump(mode="json")
        # The donor's roster: Qwen only, with its recorded settings selection.
        del self.donor["model_roster"]["nemotron"]
        self.donor["model_roster"]["qwen"]["settings_selection"] = {
            "accepted_settings": [
                *("parallel_tool_calls", "max_tokens", "temperature"),
                *("top_p", "top_k", "seed"),
            ],
            "baseline": {
                "max_tokens": 245760,
                "parallel_tool_calls": False,
                **dict.fromkeys(("seed", "temperature", "top_k", "top_p")),
            },
            "context_budget": {
                "context_window": 262144,
                "input_reserve": 16384,
                "serialized_body_safety_margin": 8192,
                "source": "preserved_product_configuration",
            },
            "model": "Qwen/Qwen3.5-397B-A17B",
        }
        # The donor is older than the current runtime: it still names a reference
        # identity in the runtime, which the current runtime refuses.
        self.donor["parabricks"]["reference_source"] = {
            "assembly_accession": "GCF_000001405.40",
            "source_locator": "https://example.test/GRCh38.fna.gz",
        }
        self.donor["parabricks"]["launch"]["accepted_plain_fasta"] = {
            "byte_count": 1,
            "sha256": "0" * 64,
        }
        self.donor_file = self.root / "donor.json"
        self.donor_file.write_text(json.dumps(self.donor), encoding="utf-8")
        self.state = self.root / "state"
        self.ledger = self.root / "core-files.yaml"
        self.ledger.write_text(yaml.safe_dump(self._core_rows()), encoding="utf-8")

    def _core_rows(self) -> list[dict]:
        """Synthetic core files for scenario s12-human-hg005, written under the state root."""
        plain = b">chr1\nACGTACGTAC\n"
        files = {
            "reference": ("eval-core/s12/ref.fna.gz", gzip.compress(plain, mtime=0)),
            "read_1": ("eval-core/s12/r1.fastq.gz", gzip.compress(b"@a\nAC\n+\nII\n")),
            "read_2": ("eval-core/s12/r2.fastq.gz", gzip.compress(b"@b\nGT\n+\nII\n")),
        }
        rows = []
        for name, (relative, data) in files.items():
            path = self.state / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            row = {
                "id": f"s12-human-hg005.{name}",
                "scenario": "s12-human-hg005",
                "kind": "reference_fasta" if name == "reference" else "fastq",
                "source_url": f"https://example.test/{relative}",
                "source_checksum": "md5:" + hashlib.md5(data).hexdigest(),
                "source_checksum_locator": "https://example.test/md5checksums.txt",
                "path": relative,
                "bytes": len(data),
                "sha256": _sha256(data),
                "retrieved_at": "2026-10-13T00:00:00Z",
                "retrieved_by": "test",
            }
            if name == "reference":
                row.update(
                    assembly_accession="GCF_000001405.40",
                    reference_asset_id="all_genomic_fasta",
                    plain_bytes=len(plain),
                    plain_sha256=_sha256(plain),
                )
            else:
                row["role"] = name
            rows.append(row)
        return rows

    def _prepare(self, model: str, case_id: str = "s12-human-hg005-clean-L3") -> dict:
        return prepare(
            eval_case.load_compiled(case_id),
            model=model,
            donor_runtime=self.donor_file,
            run_id=self.RUN_ID,
            run_root=self.home / "live-runs" / self.RUN_ID,
            inputs_dir=self.root / "inputs",
            local_state=self.state,
            ledger=self.ledger,
        )

    def test_prepared_inputs_pass_contact_free_readiness_with_no_decisions(self):
        plan = self._prepare("nemotron")
        files = plan["files"]

        with patch.dict("os.environ", {"BIO_NEB_CONTROLLER_HOME": str(self.home)}):
            code, stdout, stderr = _invoke(
                live_readiness.main,
                [
                    "--run-id",
                    self.RUN_ID,
                    "--runtime-config",
                    files["runtime"]["path"],
                    "--prompt-file",
                    files["prompt"]["path"],
                    "--fastq",
                    f"read_1={files['read_1']['path']}",
                    "--fastq",
                    f"read_2={files['read_2']['path']}",
                ],
            )
        readiness = json.loads(stdout)

        self.assertEqual((code, stderr), (0, ""))
        self.assertTrue(readiness["ready"], readiness["blockers"])
        self.assertEqual(readiness["researcher_decisions"], [])

    def test_runtime_takes_the_core_reference_and_no_prepared_index(self):
        plan = self._prepare("nemotron")
        runtime = json.loads(Path(plan["files"]["runtime"]["path"]).read_text())
        reference = next(r for r in self._core_rows() if r["kind"] == "reference_fasta")

        launch = runtime["parabricks"]["launch"]
        self.assertEqual(
            launch["local_reference_source"],
            {"path": str(self.state / reference["path"])},
        )
        # It names no prepared index: the current launch contract, which has no
        # prepared-candidate field, reads it whole.
        from bio_neb.parabricks.configuration import ParabricksLaunch

        self.assertEqual(
            ParabricksLaunch.model_validate(launch).local_reference_source.path,
            str(self.state / reference["path"]),
        )
        self.assertNotIn("reference_source", runtime["parabricks"])

    def test_runtime_changes_only_this_runs_selections_the_model_and_the_cap(self):
        plan = self._prepare("nemotron")
        runtime = json.loads(Path(plan["files"]["runtime"]["path"]).read_text())

        launch = runtime["parabricks"]["launch"]
        self.assertEqual(
            launch["object_storage"]["bucket"], "bio-neb-s282-eval-nemotron-s12-a1"
        )
        self.assertEqual(launch["object_scope"], "issue282/eval-nemotron-s12-a1")
        invocation = runtime["agent_invocation"]
        self.assertEqual(invocation["model_reference"], "nemotron")
        self.assertEqual(
            invocation["scopes"]["role"]["planning_agent"]["model_requests"], 30
        )
        entry = runtime["model_roster"]["nemotron"]
        self.assertEqual(entry["model"], "nvidia/Nemotron-3.5-Super-VL")
        self.assertEqual(entry["settings_selection"]["model"], MODELS["nemotron"])
        # Super-VL is served on us-north1 only.
        self.assertEqual(
            entry["endpoint"], "https://api.tokenfactory.us-north1.nebius.com/v1"
        )
        # The entry carries the product's Super-VL size entry: the provider
        # decides sampling and thinking, and the reply limit stays 245,760.
        selection = entry["settings_selection"]
        self.assertEqual(
            {
                name: selection["baseline"][name]
                for name in (
                    "temperature",
                    "top_p",
                    "reasoning",
                    "force_nonempty_content",
                    "max_tokens",
                )
            },
            {
                "temperature": None,
                "top_p": None,
                "reasoning": None,
                "force_nonempty_content": None,
                "max_tokens": 245760,
            },
        )
        # Nemotron's own window; the donor's Qwen entry keeps 262,144.
        qwen = self.donor["model_roster"]["qwen"]["settings_selection"]
        self.assertEqual(qwen["context_budget"]["context_window"], 262144)
        self.assertEqual(
            selection["context_budget"],
            {**qwen["context_budget"], "context_window": 1048576},
        )
        for key in ("parabricks", "model_roster", "agent_invocation"):
            runtime.pop(key)
            self.donor.pop(key, None)
        self.assertEqual(runtime, self.donor)

    def test_begin_arguments_parse_and_register_the_core_reads(self):
        plan = self._prepare("qwen")
        parser = cli._parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for command in action.choices.values():
                    command.allow_abbrev = False

        parsed = parser.parse_args(plan["begin_arguments"])
        prompt = Path(plan["files"]["prompt"]["path"]).read_text(encoding="utf-8")

        self.assertEqual(parsed.command, "begin")
        self.assertTrue(parsed.interactive)
        self.assertEqual(
            parsed.fastq,
            [
                f"read_1={self.state / 'eval-core/s12/r1.fastq.gz'}",
                f"read_2={self.state / 'eval-core/s12/r2.fastq.gz'}",
            ],
        )
        self.assertEqual(
            prompt,
            eval_case.load_compiled("s12-human-hg005-clean-L3").public.research_request,
        )

    def test_a_changed_core_file_is_refused_before_any_write(self):
        read = self.state / "eval-core/s12/r1.fastq.gz"
        read.write_bytes(read.read_bytes() + b"x")

        with self.assertRaisesRegex(
            ValueError, "core file s12-human-hg005.read_1 changed"
        ):
            self._prepare("qwen")
        self.assertFalse((self.root / "inputs").exists())

    def test_a_donor_without_investigation_tools_is_refused_before_any_write(self):
        # Without them a Run holds at its first Investigation grant.
        del self.donor["investigation_tools"]
        self.donor_file.write_text(json.dumps(self.donor), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "investigation_tools"):
            self._prepare("nemotron")
        self.assertFalse((self.root / "inputs").exists())

    def test_runtime_drops_the_global_parallel_tool_calls(self):
        # Each model's endpoint policy sends its own value.
        self.donor["agent_invocation"] = {
            "model_settings": {"parallel_tool_calls": True, "max_tokens": 245760}
        }
        self.donor_file.write_text(json.dumps(self.donor), encoding="utf-8")

        runtime = json.loads(
            Path(self._prepare("nemotron")["files"]["runtime"]["path"]).read_text()
        )

        self.assertEqual(
            runtime["agent_invocation"]["model_settings"], {"max_tokens": 245760}
        )

    def test_a_scenario_without_core_files_or_a_bad_run_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "s1-pig-duroc07 has no core reference"):
            self._prepare("qwen", case_id="s1-pig-clean-L2")
        with self.assertRaisesRegex(ValueError, "is not <ticket>-<purpose>"):
            prepare(
                eval_case.load_compiled("s12-human-hg005-clean-L3"),
                model="qwen",
                donor_runtime=self.donor_file,
                run_id="eval-s12",
                run_root=self.home / "live-runs" / "eval-s12",
                inputs_dir=self.root / "inputs",
                local_state=self.state,
                ledger=self.ledger,
            )
        self.assertFalse((self.root / "inputs").exists())

    def test_the_committed_ledger_holds_valid_core_files(self):
        rows = load_ledger()
        self.assertTrue(rows)
        for scenario in {row.scenario for row in rows}:
            core = scenario_files(rows, scenario)
            self.assertEqual({r.role for r in core.reads}, {"read_1", "read_2"})


_EVENT_IDS = itertools.count(1)


def _event(kind: str, at: str = "2026-10-13T12:00:00Z", **payload) -> dict:
    """One event shaped as the Journey writes it; tests name an id they refer to."""
    return {
        "event_id": f"event-{next(_EVENT_IDS)}",
        "event_type": kind,
        "occurred_at": at,
        "parent_ids": [],
        "payload": payload,
    }


def _search(kinds: list[str]) -> list[dict]:
    """One admitted search turn as the Journey records it."""
    return [
        _event("ModelRequestRecorded"),
        _event("ModelTransportRequestRecorded"),
        _event("ModelResponseRecorded"),
        _event(
            "ToolCallValidated", tool_name="search_journey", arguments={"kinds": kinds}
        ),
        _event("AgentActionTurnDispositionRecorded", disposition="singleton_admitted"),
        _event("JourneySearchRead"),
    ]


class WatchTests(unittest.TestCase):
    NOW = datetime(2026, 10, 13, 12, 0, 30, tzinfo=timezone.utc)

    def _trips(self, events, gets=0, **limits):
        found = trips(events, Limits(**limits), self.NOW, gets)
        return {t["limit"]: t for t in found}

    def test_identical_calls_with_no_new_fact_trip_at_the_repeat_limit(self):
        events = [e for _ in range(5) for e in _search(["PlanningCandidateSelected"])]

        self.assertNotIn("repeated_call", self._trips(events[:24]))
        self.assertEqual(self._trips(events)["repeated_call"]["count"], 5)

    def test_a_new_fact_or_different_arguments_reset_the_streak(self):
        same = _search(["PlanningCandidateSelected"])
        fact = [_event("ResourceObservationRecorded")]
        other = _search(["RecipeObservationRecorded"])

        self.assertNotIn("repeated_call", self._trips(same * 3 + fact + same * 3))
        self.assertNotIn("repeated_call", self._trips(same * 4 + other + same * 4))

    def test_request_ceiling_trips_at_its_limit(self):
        requests = [_event("ModelRequestRecorded")] * 50

        self.assertEqual(self._trips(requests[:49]), {})
        self.assertEqual(self._trips(requests)["model_requests"]["count"], 50)

    def test_source_gets_count_every_physical_get_in_the_contact_journal(self):
        # Product-written journals: failed GETs count, and one operation may make several.
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-1"
            scope = ContactScope(attempt_id="a1", request_id="r1")
            first = ContactJournal(run / "contacts/p1/journal.jsonl", "run-1", "p1")
            for _ in range(6):
                first.finish(
                    first.start(scope, "source.get"), "failed", cause="timeout"
                )
            first.finish(first.start(scope, "model.request"), "succeeded")
            first.close()
            second = ContactJournal(run / "contacts/p2/journal.jsonl", "run-1", "p2")
            for _ in range(4):  # one discovery making several lineage GETs
                second.finish(second.start(scope, "source.get"), "succeeded")

            passes = [_event("PassStarted", pass_id=name) for name in ("p1", "p2")]
            gets = source_gets(run, passes)

        self.assertEqual(gets, 10)
        self.assertEqual(self._trips([], gets=9), {})
        self.assertEqual(self._trips([], gets=gets)["source_gets"]["count"], 10)

    def test_unreadable_contact_accounting_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-1"
            journal = run / "contacts/p1/journal.jsonl"
            journal.parent.mkdir(parents=True)
            journal.write_text('{"kind": "start"}\n', encoding="utf-8")

            with self.assertRaises(ValueError):
                source_gets(run, [_event("PassStarted", pass_id="p1")])
        self.assertIn("source_accounting_unreadable", self._trips([], gets=None))

    def _watch_run(self, root: Path, passes: list[str], journals: list[str], gets=1):
        """A capsule root whose Journey records `passes`; journals exist for `journals`."""
        run = root / "capsule" / "runs" / "run-1"
        run.mkdir(parents=True)
        (root / "capsule" / "operator").mkdir()
        events = [_event("PassStarted", pass_id=name) for name in passes]
        (run / "journey.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
        )
        scope = ContactScope(attempt_id="a1", request_id="r1")
        for name in journals:
            journal = ContactJournal(
                run / f"contacts/{name}/journal.jsonl", "run-1", name
            )
            for _ in range(gets):
                journal.finish(journal.start(scope, "source.get"), "failed")
        return root / "capsule"

    def _watch_once(self, capsule: Path, **limits) -> tuple[int, list[dict]]:
        printed = io.StringIO()
        with (
            patch("evals.watch.subprocess.run") as interrupt,
            patch("evals.watch.time.sleep", side_effect=StopIteration),
            redirect_stdout(printed),
        ):
            interrupt.return_value.returncode = 0
            try:
                code = watch(capsule, Limits(**limits))
            except StopIteration:  # one poll ended without a trip
                code = None
        trips_seen = [json.loads(line) for line in printed.getvalue().splitlines()]
        self.assertEqual(interrupt.called, code == 2)
        return code, trips_seen

    def test_a_missing_journal_of_a_recorded_pass_trips_through_the_watcher(self):
        for name, journals in {"one of two": ["p1"], "all": []}.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                capsule = self._watch_run(Path(directory), ["p1", "p2"], journals)

                code, printed = self._watch_once(capsule)

            self.assertEqual(code, 2)
            self.assertEqual(
                [t["limit"] for t in printed[0]["trips"]],
                ["source_accounting_unreadable"],
            )

    def test_the_watcher_counts_gets_across_recorded_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            capsule = self._watch_run(
                Path(directory), ["p1", "p2"], ["p1", "p2"], gets=5
            )
            code, printed = self._watch_once(capsule)
        self.assertEqual(code, 2)
        self.assertEqual(printed[0]["trips"], [{"limit": "source_gets", "count": 10}])

        with tempfile.TemporaryDirectory() as directory:
            capsule = self._watch_run(Path(directory), [], [])  # before any Pass
            self.assertEqual(self._watch_once(capsule), (None, []))

    def test_cut_off_replies_are_reported(self):
        cut = _event(
            "ModelCompletionTruncated",
            attempt_id="a1",
            request_number=18,
            completion_tokens=21596,
            requested_max_tokens=245760,
        )

        self.assertEqual(
            cut_replies([_event("ModelResponseRecorded"), cut]),
            [
                {
                    "request_number": 18,
                    "completion_tokens": 21596,
                    "requested_max_tokens": 245760,
                    "final_max_completion_tokens": None,
                }
            ],
        )

    def _reply(self, output_tokens: int, calls: list[tuple[str, dict]]) -> list[dict]:
        """One request, its send and its reply, as the Journey records them."""
        request = _event(
            "ModelRequestRecorded",
            attempt_id="a1",
            request_number=5,
            compatible_adapter_evidence={
                "request": {"model_settings": {"max_tokens": 245760}}
            },
        )
        request["event_id"] = "request-5"
        response = _event(
            "ModelResponseRecorded",
            attempt_id="a1",
            request_number=5,
            compatible_adapter_evidence={
                "response": {
                    "message": {
                        "finish_reason": "length",
                        "usage": {"output_tokens": output_tokens},
                        "parts": [
                            {"part_kind": "thinking"},
                            *(
                                {
                                    "part_kind": "tool-call",
                                    "tool_name": n,
                                    "args": a,
                                    "tool_call_id": f"call-{i}",
                                }
                                for i, (n, a) in enumerate(calls, 1)
                            ),
                        ],
                    }
                }
            },
        )
        response["event_id"] = "response-5"
        return [
            request,
            _event(
                "ModelTransportRequestRecorded",
                attempt_id="a1",
                request_number=5,
                logical_request_event_id="request-5",
                requested_max_tokens=245760,
                final_max_completion_tokens=230158,
            ),
            response,
        ]

    def test_reply_rows_use_the_ceiling_sent(self):
        # A recorded live reply: cut at 230,158, the ceiling sent, not the 245,760
        # configured.
        search = ("search_journey", {"kinds": ["PlanningCandidateSelected"]})
        events = self._reply(230158, [search, search, ("pick", {"id": "x"})])
        cut = _event(
            "ModelCompletionTruncated",
            attempt_id="a1",
            request_number=5,
            response_event_id="response-5",
            completion_tokens=230158,
            requested_max_tokens=245760,
        )
        routed = _event(
            "AgentTurnRouted",
            attempt_id="a1",
            response_event_id="response-5",
            copies={"call-2": "call-1"},
            unoffered_validation_event_ids=["v3"],
            dropped_call_id="call-4",
            faults=[{"key": "copied_calls", "count": 1}],
        )

        (row,) = reply_rows([*events, routed, cut])
        (small,) = reply_rows(self._reply(1000, [search]))

        self.assertEqual(
            row,
            {
                "response_event_id": "response-5",
                "attempt": "a1",
                "request_number": 5,
                "output_tokens": 230158,
                "allowance": 230158,
                "finish_reason": "length",
                "cut_off": True,
                "calls": 3,
                "distinct_calls": 2,
                "copies": 1,
                "unoffered": 1,
                "refused_outputs": 0,
                "dropped_call": "call-4",
                "faults": {"copied_calls": 1},
                "large": True,
            },
        )
        self.assertEqual((small["large"], small["cut_off"]), (False, False))
        self.assertEqual(
            cut_replies([*events, cut])[0]["final_max_completion_tokens"], 230158
        )
        self.assertEqual(replies(events)[0]["allowance"], 230158)

    def test_distinct_calls_use_the_products_call_identity(self):
        # A plain and a fenced call with the same arguments are one call to the
        # router (`action_policy.call_identity`), so one distinct call here too.
        events = self._reply(10, [("pick", {"id": "x", "n": 1})])
        message = events[-1]["payload"]["compatible_adapter_evidence"]["response"][
            "message"
        ]
        message["parts"].append(
            {
                "part_kind": "tool-call",
                "tool_name": "pick",
                "args": '```json\n{"n": 1, "id": "x"}\n```',
            }
        )

        (row,) = replies(events)

        self.assertEqual((row["calls"], row["distinct_calls"]), (2, 1))

    def _watch_polls(self, capsule: Path, *later: list[dict]) -> list[dict]:
        """Watch ``capsule``; at each sleep append the next batch of events. The
        finish record arrives in the same sleep as the last batch."""
        journey = next(capsule.glob("runs/*/journey.jsonl"))
        batches = list(later)

        def sleep(_):
            with journey.open("a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(e) + "\n" for e in batches.pop(0))
            if not batches:
                (capsule / "operator" / "finished.json").write_text("{}")

        printed = io.StringIO()
        with (
            patch("evals.watch.subprocess.run") as interrupt,
            patch("evals.watch.time.sleep", side_effect=sleep),
            redirect_stdout(printed),
        ):
            code = watch(capsule, Limits())
        self.assertEqual((code, interrupt.called), (0, False))
        return [json.loads(line) for line in printed.getvalue().splitlines()]

    def _routed_reply(self) -> tuple[list[dict], list[dict]]:
        """A cut-off reply as the product routes it, then its later facts.

        Its calls: a search, an exact copy of it, and a partial last call that
        reached the ceiling. The copy is collapsed onto the first call; the
        partial call is dropped before copy detection (ADR 0044).
        """
        search = ("search_journey", {"kinds": ["PlanningCandidateSelected"]})
        partial = ("pick", {"id": "x"})
        started = _event("AttemptStarted", attempt_id="a1")
        started["event_id"] = "start"
        reply = [started, *self._reply(230158, [search, search, partial])]
        note = "Copied calls: you sent `search_journey` 2 times."
        cut = _event(
            "ModelCompletionTruncated",
            attempt_id="a1",
            request_number=5,
            response_event_id="response-5",
            completion_tokens=230158,
            requested_max_tokens=245760,
            correction_count=1,
            feedback=note,
        )
        cut["event_id"] = "cut-5"
        routed = _event(
            "AgentTurnRouted",
            attempt_id="a1",
            response_event_id="response-5",
            run_order=["call-1"],
            copies={"call-2": "call-1"},
            unoffered_validation_event_ids=[],
            dropped_call_id="call-3",
            faults=[
                {"key": "output_truncated", "count": 1},
                {"key": "copied_calls", "count": 1},
            ],
            feedback=note,
        )
        routed["event_id"] = "routed-5"
        return reply, [cut, routed]

    def _assert_each_fault_once(self, printed: list[dict]) -> None:
        self.assertNotIn("turn_faults_unreadable", [p["event"] for p in printed])
        self.assertEqual(
            [
                (p["source"], p["key"], p["count"])
                for p in printed
                if p["event"] == "turn_fault"
            ],
            [
                ("ModelCompletionTruncated", "output_truncated", 1),
                ("AgentTurnRouted", "copied_calls", 1),
            ],
        )

    def test_a_reply_row_is_reported_again_when_later_facts_enrich_it(self):
        reply, facts = self._routed_reply()
        with tempfile.TemporaryDirectory() as directory:
            capsule = self._watch_run(Path(directory), [], [])

            printed = self._watch_polls(capsule, reply, facts, [])

        rows = [p for p in printed if p["event"] == "reply"]
        self.assertEqual(
            [
                (r["calls"], r["cut_off"], r["copies"], r["dropped_call"], r["faults"])
                for r in rows
            ],
            [
                (3, False, 0, None, {}),
                (
                    3,
                    True,
                    1,
                    "call-3",
                    {"output_truncated": 1, "copied_calls": 1},
                ),
            ],
        )
        self.assertEqual([r.get("updated") for r in rows], [None, True])
        self._assert_each_fault_once(printed)

    def test_the_watcher_reads_the_journey_once_more_before_finished(self):
        reply, facts = self._routed_reply()
        with tempfile.TemporaryDirectory() as directory:
            capsule = self._watch_run(Path(directory), [], [])

            # The reply, its facts and the finish record arrive in one sleep.
            printed = self._watch_polls(capsule, reply + facts)

        self.assertEqual(
            [p["event"] for p in printed],
            ["truncated", "turn_fault", "turn_fault", "reply", "finished"],
        )
        self.assertTrue(printed[3]["cut_off"])
        self._assert_each_fault_once(printed)

    def test_routing_records_do_not_reset_the_repeat_streak(self):
        # A routed or cut-off reply adds no fact (repeated_calls' loop records).
        same = _search(["PlanningCandidateSelected"])
        routing = [_event("AgentTurnRouted"), _event("ModelCompletionTruncated")]

        events = same + routing + same + routing + same + same + routing + same

        self.assertEqual(self._trips(events)["repeated_call"]["count"], 5)

    def test_the_loops_own_records_do_not_reset_the_repeat_streak(self):
        # The product's rule (`repeated_calls.is_loop_record`): a search's execution
        # receipts and its contact evidence are the loop's own records, never a fact.
        same = _search(["PlanningCandidateSelected"]) + [
            _event("ObservedToolExecutionStarted"),
            _event("ObservedToolExecutionSucceeded"),
        ]

        self.assertEqual(self._trips(same * 5)["repeated_call"]["count"], 5)
        self.assertNotIn(
            "repeated_call",
            self._trips(same * 3 + [_event("ResourceObservationRecorded")] + same * 3),
        )

    def test_turn_faults_are_reported_not_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            capsule = self._watch_run(Path(directory), [], [])
            journey = next(capsule.glob("runs/*/journey.jsonl"))
            started = _event("AttemptStarted", attempt_id="a1")
            started["event_id"] = "start"
            # A cut-off belongs to the reply of its request.
            request = _event("ModelRequestRecorded", attempt_id="a1", request_number=1)
            cut = _event(
                "ModelCompletionTruncated",
                attempt_id="a1",
                request_number=1,
                response_event_id="response-1",
                correction_count=1,
                completion_tokens=9,
                requested_max_tokens=10,
            )
            cut["event_id"] = "cut"
            journey.write_text(
                "".join(json.dumps(e) + "\n" for e in (started, request, cut)),
                encoding="utf-8",
            )

            code, printed = self._watch_once(capsule)

        self.assertIsNone(code)
        self.assertEqual([p["event"] for p in printed], ["truncated", "turn_fault"])
        self.assertEqual(
            {k: printed[1][k] for k in ("key", "count", "source")},
            {
                "key": "output_truncated",
                "count": 1,
                "source": "ModelCompletionTruncated",
            },
        )

    def test_an_unanswered_request_trips_after_the_stall_limit(self):
        sent = [_event("ModelTransportRequestRecorded", at="2026-10-13T11:57:00Z")]
        answered = sent + [_event("ModelResponseRecorded")]

        self.assertEqual(self._trips(sent)["stalled_request"]["seconds"], 210)
        self.assertNotIn("stalled_request", self._trips(sent, stall_seconds=300))
        self.assertNotIn("stalled_request", self._trips(answered))


class RoutedReplayScoreTests(unittest.TestCase):
    """Main's #326 reply-4 replay (2,626 copies, one unoffered call, a cut reply)
    scored and watched: each turn fault of the product's ledger is seen once."""

    def test_the_326_replay_shows_its_three_turn_faults_and_its_reply_row(self):
        from tests.turn_routing_replays import recorded_reply, replay_326

        with tempfile.TemporaryDirectory() as directory:
            events, response, routed, _, _ = replay_326(
                Path(directory), recorded_reply(4)
            )
        score = score_planning(_baseline_case(), events)
        rows = reply_rows(events)
        limits = trips(events, Limits(), datetime.now(timezone.utc), 0)

        # The cut-off counts on its own truncation fact, recorded before the routing.
        self.assertEqual(
            [
                (f["source"], f["key"], f["count"])
                for f in score["diagnostics"]["turn_faults"]
                if f["event_id"] == routed["event_id"]
                or f["source"] == "ModelCompletionTruncated"
            ],
            [
                ("ModelCompletionTruncated", "output_truncated", 1),
                ("AgentTurnRouted", "copied_calls", 1),
                ("AgentTurnRouted", "unoffered_calls", 1),
            ],
        )
        (row,) = [
            r
            for r in rows
            if r["request_number"] == response["payload"]["request_number"]
            and r["attempt"] == response["payload"]["attempt_id"]
        ]
        self.assertEqual(
            {k: row[k] for k in ("calls", "copies", "unoffered", "cut_off", "large")},
            {
                "calls": 2632,
                "copies": 2625,
                "unoffered": 1,
                "cut_off": True,
                "large": True,
            },
        )
        self.assertEqual(row["dropped_call"], routed["payload"]["dropped_call_id"])
        self.assertEqual(limits, [])


if __name__ == "__main__":
    unittest.main()
