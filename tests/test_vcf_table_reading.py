"""The post-approval VCF table stage (ADR 0047), through its public seams.

Each approved item's deterministic lookup runs first; the coordinator's step then
reads every item's fixed window of the Run's prepared VCF in one model-free batch,
gives each chunk of the VCF table one Attempt of the table reader, which records
one reading per row, and folds the collection's item evidence before
Finalization. Each test names the item query guarantee it migrates (#361).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from bio_neb.agentic_journey.agent_work_review import (
    RejectAgentWorkResponse,
    project_agent_work_review,
    respond_agent_work_review,
)
from bio_neb.agentic_journey.role_declaration import role as declared_role
from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration
from bio_neb.agentic_journey.tool_execution import (
    function_tool_names,
    stage_tool_names,
)
from bio_neb.application.queries import read_completion, read_table_reading_stage
from bio_neb.investigation.investigation_observation import (
    validate_investigation_collection_evidence_fact,
    validate_investigation_work_item_started_fact,
)
from bio_neb.investigation.item_evidence import (
    CONCERN_CODES,
    CoverageMeasurement,
    InvestigationItemWindowRecorded,
    InvestigationTableChunkBound,
    InvestigationTableChunkFailed,
    InvestigationTableChunkRead,
    ItemCoverageField,
    ItemReading,
    ItemReadingField,
    coverage_span,
    fold_item_evidence,
    validate_item_evidence_fact,
)
from bio_neb.investigation.observation_defaults import (
    CoverageDefaults,
    VcfTableDefaults,
)
from bio_neb.investigation.run_health import validate_run_health_fact
from bio_neb.investigation.vcf_table_reading import (
    advance_table_reading,
    validate_table_chunk_facts,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.recording import read_evidence
from bio_neb.reporting.finalization_commands import (
    CompileFinalizationCommand,
    compile_finalization,
)
from tests.item_state_runs import FINALIZATION_PENDING
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.proposal_information_fakes import runtime_configuration
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    ObservationAdapter,
    observed_collection,
)
from tests.vcf_lookup_fakes import (
    TableReaderModel,
    chunk_rows,
    lookup_command,
    query_region,
    query_row,
)

ENVIRON = {"LOCAL_MODEL_KEY": "secret-value-not-retained"}


class _WindowAdapter(ObservationAdapter):
    """Answers the lookup with no record, then the item's window with two records."""

    def run(self, argv, *, timeout_seconds, stdout_path=None):
        if lookup_command(argv) == "variant_query":
            contig, start, end = query_region(self.files, argv)
            self.calls.append(tuple(argv))
            if start == end:
                return type(super().run(argv, timeout_seconds=timeout_seconds))(
                    0, b"", b""
                )
            return type(super().run(argv, timeout_seconds=timeout_seconds))(
                0,
                query_row(contig, start, "CA", "C", self.sample_id, "0/1", gq="40")
                + query_row(contig, end, "G", "A", self.sample_id, "1/1"),
                b"",
            )
        return super().run(
            argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
        )


def _observed(root: Path, *, adapter_type=ObservationAdapter, **choices):
    return observed_collection(root, adapter_type=adapter_type, **choices)


def _run_stage(journey: Journey, model_factory, adapter) -> int:
    steps = 0
    while advance_table_reading(
        journey, environ=ENVIRON, model_factory=model_factory, adapter=adapter
    ):
        steps += 1
        journey = Journey.open(journey.path)
    return steps


def _measure(journey: Journey, adapter) -> Journey:
    """The stage's first two steps: the Run's health, then every item's coverage
    (#414); these Runs hold no local BAM, so neither runs a process."""
    for _ in range(2):
        advance_table_reading(
            journey,
            environ=ENVIRON,
            model_factory=TableReaderModel().model,
            adapter=adapter,
        )
        journey = Journey.open(journey.path)
    return journey


def _facts(journey: Journey, event_type: str):
    return [
        (event, validate_item_evidence_fact(event))
        for event in read_evidence(journey)
        if event["event_type"] == event_type
    ]


def _text_reply(messages, info):
    return ModelResponse(parts=[TextPart("I will not read the table.")])


class VcfTableStageTests(unittest.TestCase):
    def test_one_model_free_batch_reads_every_window_with_server_bound_item_and_vcf(
        self,
    ) -> None:
        # Migrates "each item is queried once with server-bound item and VCF".
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve())
            journey = _measure(journey, adapter)
            stage = read_table_reading_stage(journey)
            assert stage is not None
            model = TableReaderModel()
            lookups = len(adapter.calls)

            self.assertTrue(
                advance_table_reading(
                    journey, environ=ENVIRON, model_factory=model.model, adapter=adapter
                )
            )

            journey = Journey.open(journey.path)
            windows = _facts(journey, "InvestigationItemWindowRecorded")
            after = read_table_reading_stage(journey)

        self.assertEqual(model.calls, 0)
        self.assertEqual(len(adapter.calls) - lookups, 3)
        self.assertEqual(
            [fact.work_item_id for _, fact in windows],
            [item.work_item_id for item in stage.collection.item_states],
        )
        for _, fact in windows:
            argv = fact.result.query_invocation.argv
            self.assertEqual(argv[0], adapter.command.bcftools_executable)
            self.assertTrue(argv[-1].endswith("variants.vcf.gz"))
            # The Run's selected flank, named default ±100.
            self.assertEqual(fact.flank_bases, 100)
            self.assertEqual(fact.collection_event_id, stage.collection_event_id)
        assert after is not None
        self.assertEqual(after.windows_pending, ())
        self.assertEqual(len(after.unread), 3)

    def test_items_are_read_in_chunks_of_the_runs_size_each_with_its_own_reading(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(), investigation_table={"chunk_size": 2}
            )
            stage = read_table_reading_stage(journey)
            assert stage is not None
            model = TableReaderModel(
                readings={
                    1: {
                        "row": 1,
                        "reading": "concern",
                        "concerns": ["absence_not_established"],
                        "note": "No record; the reference call is not established.",
                    }
                }
            )

            steps = _run_stage(journey, model.model, adapter)

            journey = Journey.open(journey.path)
            bound = _facts(journey, "InvestigationTableChunkBound")
            read = _facts(journey, "InvestigationTableChunkRead")
            ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")

        selected = [item.work_item_id for item in stage.collection.item_states]
        # Health, coverage, windows, both chunks as one batch under the default
        # bound, then the fold.
        self.assertEqual(steps, 5)
        self.assertEqual(model.calls, 2)
        self.assertEqual([len(rows) for rows in model.chunks], [2, 1])
        self.assertEqual(
            [fact.work_item_ids for _, fact in bound],
            [tuple(selected[:2]), tuple(selected[2:])],
        )
        self.assertEqual({fact.chunk_size for _, fact in bound}, {2})
        # Every item has its own reading, bound to its own chunk's Attempt.
        attempts = {
            item: fact.attempt_id for _, fact in bound for item in fact.work_item_ids
        }
        self.assertEqual([item.work_item_id for item in fold.items], selected)
        for item in fold.items:
            self.assertEqual(item.reading.status, "recorded")
            self.assertEqual(item.reading.attempt_id, attempts[item.work_item_id])
            self.assertEqual(item.reading.reading.work_item_id, item.work_item_id)
            self.assertEqual(item.window.status, "recorded")
            self.assertEqual(item.coverage.status, "not_measured")
        self.assertEqual(
            [item.reading.reading.reading for item in fold.items],
            ["concern", "no_concern", "concern"],
        )
        # Both chunks run at once, so their reads land in the order they finish;
        # each read is matched to its bound chunk by the chunk's Attempt.
        self.assertEqual(len(read), len(bound))
        reads = {fact.attempt_id: fact for _, fact in read}
        self.assertEqual(
            [reads[fact.attempt_id].readings[0].work_item_id for _, fact in bound],
            [selected[0], selected[2]],
        )

    def test_chunks_run_concurrently_under_the_runs_bound_and_reopen(self) -> None:
        # Each chunk is its own Attempt; up to the Run's bound run at once and
        # are collected, and the interleaved Journey reopens.
        import asyncio

        class Overlapping(TableReaderModel):
            def __init__(self, expected: int) -> None:
                super().__init__()
                self.expected, self.in_flight, self.peak = expected, 0, 0

            async def respond(self, messages, info):  # noqa: ANN001, ANN201
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                for _ in range(200):
                    if self.peak >= self.expected:
                        break
                    await asyncio.sleep(0.01)
                try:
                    return await super().respond(messages, info)
                finally:
                    self.in_flight -= 1

        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(),
                investigation_table={"chunk_size": 1},
                investigation_concurrency={"planner_members": 1, "table_chunks": 2},
            )
            stage = read_table_reading_stage(journey)
            assert stage is not None
            model = Overlapping(expected=2)

            steps = _run_stage(journey, model.model, adapter)

            journey = Journey.open(journey.path)
            bound = _facts(journey, "InvestigationTableChunkBound")
            read = _facts(journey, "InvestigationTableChunkRead")
            ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")
            # The interleaved chunks reopen through the checked reader and reach
            # the one Finalization path.
            candidate = read_completion(Journey.open(journey.path)).candidate
            basis = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            )

        selected = [item.work_item_id for item in stage.collection.item_states]
        # Health, coverage, windows, a batch of two chunks, a batch of one, then
        # the fold.
        self.assertEqual(steps, 6)
        self.assertEqual(model.peak, 2)
        self.assertEqual(model.calls, 3)
        self.assertEqual(
            sorted(fact.work_item_ids for _, fact in bound),
            sorted((item,) for item in selected),
        )
        # Each reading stays bound to its own chunk's Attempt, whatever the order.
        attempts = {fact.work_item_ids[0]: fact.attempt_id for _, fact in bound}
        self.assertEqual(len({fact.attempt_id for _, fact in read}), 3)
        self.assertEqual([item.work_item_id for item in fold.items], selected)
        for item in fold.items:
            self.assertEqual(item.reading.status, "recorded")
            self.assertEqual(item.reading.attempt_id, attempts[item.work_item_id])
        self.assertIsNotNone(candidate)
        self.assertEqual(
            [item.work_item_id for item in basis.pending_request.basis.item_evidence],
            sorted(selected),
        )

    def _two_stopping_chunks(self, directory, second_reply):
        """Two concurrent chunks: the first ends in text, the second as scripted."""
        import asyncio
        from unittest.mock import patch

        import tests.sequential_investigation_support as support
        from tests.vcf_lookup_fakes import chunk_rows

        original = support.runtime_configuration

        def configured():
            raw = original()
            raw["agent_invocation"] = {
                "failed_turns": 1,
                "scopes": {"role": {"table_reader": {"model_requests": 1}}},
            }
            return raw

        seen: dict = {}

        async def respond(messages, info):
            key = canonical_json(chunk_rows(messages))
            seen.setdefault(key, len(seen))
            for _ in range(500):
                if len(seen) == 2:
                    break
                await asyncio.sleep(0.01)
            if seen[key] == 0:
                return ModelResponse(parts=[TextPart("No reading supplied.")])
            return second_reply()

        with patch.object(support, "runtime_configuration", configured):
            journey, adapter = _observed(
                directory,
                cardinality=2,
                investigation_table={"chunk_size": 1},
                investigation_concurrency={"planner_members": 1, "table_chunks": 2},
            )
        _run_stage(journey, ScriptedModelEndpoint(respond), adapter)
        return Journey.open(journey.path), adapter

    def test_each_concurrent_chunks_answer_reenters_it_beside_a_siblings_review(
        self,
    ) -> None:
        # Each chunk holds its own scope, so either chunk's answer re-enters
        # it while the other chunk's Review still pends.
        from bio_neb.agentic_journey.agent_work_review import (
            ContinueAgentWorkResponse,
        )
        from bio_neb.agentic_journey.effort import (
            ExtendAgentEffortResponse,
            project_effort_review,
            respond_effort_review,
        )

        def search():
            return ModelResponse(
                parts=[ToolCallPart("search_journey", {}, tool_call_id="search")]
            )

        for order in ("effort_first", "work_first"):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as temporary:
                journey, adapter = self._two_stopping_chunks(
                    Path(temporary).resolve(), search
                )
                events = read_evidence(journey)
                (work,) = project_agent_work_review(events).requests
                (effort,) = project_effort_review(events).requests
                answers = {
                    "effort": lambda r=effort, p=journey.path: respond_effort_review(
                        Journey.open(p),
                        r.request_event_id,
                        ExtendAgentEffortResponse(
                            action="extend", additional_requests=2
                        ),
                    ),
                    "work": lambda r=work, p=journey.path: respond_agent_work_review(
                        Journey.open(p),
                        r.request_event_id,
                        ContinueAgentWorkResponse(action="continue"),
                    ),
                }
                first, second = (
                    ("effort", "work")
                    if order == "effort_first"
                    else ("work", "effort")
                )
                answers[first]()
                model = TableReaderModel()
                # The answered chunk re-enters and reads; its sibling still waits.
                self.assertTrue(
                    advance_table_reading(
                        Journey.open(journey.path),
                        environ=ENVIRON,
                        model_factory=model.model,
                        adapter=adapter,
                    )
                )
                self.assertEqual(model.calls, 1)
                self.assertEqual(
                    len(
                        _facts(
                            Journey.open(journey.path), "InvestigationTableChunkRead"
                        )
                    ),
                    1,
                )
                self.assertFalse(
                    advance_table_reading(
                        Journey.open(journey.path),
                        environ=ENVIRON,
                        model_factory=model.model,
                        adapter=adapter,
                    )
                )
                answers[second]()
                _run_stage(Journey.open(journey.path), model.model, adapter)
                journey = Journey.open(journey.path)
                ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")
                self.assertEqual(
                    [item.reading.status for item in fold.items],
                    ["recorded", "recorded"],
                )
                self.assertEqual(model.calls, 2)

    def test_a_chunks_availability_stop_reads_and_holds_the_stage(self) -> None:
        # A table reader's availability exhaustion closes its Attempt on the
        # shared exact terminal; the stage reads and holds.
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )

        def unavailable():
            raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})

        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = self._two_stopping_chunks(
                Path(temporary).resolve(), unavailable
            )
            events = read_evidence(journey)
            (availability,) = project_model_availability_review(events).requests
            stage = read_table_reading_stage(journey)

        self.assertTrue(availability.pending)
        self.assertEqual(availability.agent_role, "table_reader")
        assert stage is not None
        self.assertEqual(stage.unread, ())
        self.assertEqual(
            [
                e["payload"]["code"]
                for e in events
                if e["event_type"] == "AttemptFailed"
            ],
            ["model_availability_exhausted"],
        )

    def test_a_chunks_start_scope_must_be_its_own_rows(self) -> None:
        # Standards r2 S2: the start's hold scope is checked against the
        # chunk's bound rows.
        import copy

        from bio_neb.investigation.investigation import project_investigation
        from bio_neb.investigation.vcf_table_reading import validate_table_reading

        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(),
                cardinality=2,
                investigation_table={"chunk_size": 1},
                investigation_concurrency={"planner_members": 1, "table_chunks": 2},
            )
            _run_stage(journey, TableReaderModel().model, adapter)
            events = read_evidence(Journey.open(journey.path))
        validate_table_reading(events, project_investigation(events))
        starts = [
            e
            for e in events
            if e["event_type"] == "AttemptStarted"
            and e["payload"]["agent_role"] == "table_reader"
        ]
        sibling = starts[1]["payload"]["member_id"]
        for scope in (sibling, "table-chunk:work-item:" + "0" * 64):
            hostile = copy.deepcopy(events)
            next(e for e in hostile if e["event_id"] == starts[0]["event_id"])[
                "payload"
            ]["member_id"] = scope
            with (
                self.subTest(scope=scope),
                self.assertRaisesRegex(
                    ValueError, "table reading Attempt changed its hold scope"
                ),
            ):
                validate_table_reading(hostile, project_investigation(hostile))

    def test_one_chunk_reads_the_collection_with_no_vcf_tool(self) -> None:
        # Contract (ADR 0047): the stage never drifts back to one Agent per item,
        # and the model reads the table, never the VCF.
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve())
            model = TableReaderModel()
            _run_stage(journey, model.model, adapter)
            journey = Journey.open(journey.path)
            bound = _facts(journey, "InvestigationTableChunkBound")
            body = model.model.bodies[0]

        self.assertEqual(len(bound), 1)
        self.assertEqual(model.calls, 1)
        # No VCF Tool: the search, the shared researcher question and the
        # output.
        self.assertEqual(
            function_tool_names("table_reader"), {"search_journey", "ask_researcher"}
        )
        self.assertEqual(
            stage_tool_names("table_reader", "vcf_table_reading"), {"ask_researcher"}
        )
        self.assertEqual(
            declared_role("table_reader").output_tools, {"record_item_readings"}
        )
        sent = {tool["function"]["name"]: tool["function"] for tool in body["tools"]}
        self.assertEqual(
            set(sent), {"search_journey", "ask_researcher", "record_item_readings"}
        )
        # No regex pattern reaches the model, and no item identity is asked.
        output = json.dumps(sent["record_item_readings"]["parameters"])
        self.assertNotIn('"pattern"', output)
        self.assertNotIn("work_item", output)
        rows = json.loads(body["messages"][-1]["content"])["rows"]
        self.assertEqual([row["row"] for row in rows], [1, 2, 3])
        self.assertTrue(all("work_item_id" not in row for row in rows))
        # Each request names what the model can call now and what waits.
        view = json.loads(body["messages"][0]["content"].rpartition("\n")[2])
        self.assertEqual(
            (view["callable_now"], view["waiting"]),
            (["ask_researcher", "record_item_readings", "search_journey"], []),
        )

    def test_a_reading_never_changes_the_lookup_disposition_or_admission(
        self,
    ) -> None:
        # Migrates "a query never changes the lookup disposition or admission".
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(), cardinality=1, adapter_type=_WindowAdapter
            )
            before = [
                event
                for event in read_evidence(journey)
                if event["event_type"]
                in {
                    "InvestigationEvidenceRecorded",
                    "InvestigationCollectionEvidenceRecorded",
                }
            ]
            self.assertIsNone(read_completion(journey).candidate)
            model = TableReaderModel(
                readings={
                    1: {
                        "row": 1,
                        "reading": "concern",
                        "concerns": ["nearby_variant", "absence_not_established"],
                        "note": "A deletion upstream may hide this call.",
                    }
                }
            )

            _run_stage(journey, model.model, adapter)

            journey = Journey.open(journey.path)
            ((_, window),) = _facts(journey, "InvestigationItemWindowRecorded")
            pending = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            ).pending_request
            after = [
                event
                for event in read_evidence(journey)
                if event["event_type"]
                in {
                    "InvestigationEvidenceRecorded",
                    "InvestigationCollectionEvidenceRecorded",
                }
            ]

        assert pending is not None
        basis = pending.basis
        # The window saw two records where the lookup saw none.
        self.assertEqual(window.result.outcome, "window_observed")
        self.assertEqual(len(window.result.records), 2)
        (row,) = model.chunks[0]
        self.assertEqual(row["outcome"], "no_record")
        self.assertEqual(len(row["window"]["records"]), 2)
        self.assertEqual(after, before)
        (observation,) = basis.sample_observations
        self.assertEqual(observation.result.outcome, "no_record")
        self.assertEqual(
            [item.status for item in basis.admissions],
            [
                event["payload"]["admission"]["status"]
                for event in before
                if event["event_type"] == "InvestigationEvidenceRecorded"
            ],
        )
        (item,) = basis.item_evidence
        self.assertEqual(item.lookup_outcome, "no_record")
        self.assertEqual(item.reading.reading.reading, "concern")

    def test_an_item_reads_alone_from_its_own_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(), investigation_table={"chunk_size": 2}
            )
            _run_stage(journey, TableReaderModel().model, adapter)
            journey = Journey.open(journey.path)
            basis = compile_finalization(
                CompileFinalizationCommand(journey_file=str(journey.path))
            ).pending_request.basis

        # The last item, read in a chunk of its own, keeps a complete record:
        # its lookup, its window, its reading and its coverage, on its own row.
        last = basis.collection_item_states[-1].work_item_id
        (record,) = [i for i in basis.item_evidence if i.work_item_id == last]
        (row,) = [
            item
            for snp in basis.collection_view.snps
            for item in snp.items
            if item.work_item_id == last
        ]
        self.assertEqual(record.window.status, "recorded")
        self.assertEqual(record.reading.status, "recorded")
        self.assertEqual(record.lookup_outcome, row.outcome)
        self.assertEqual(row.window.flank_bases, 100)
        self.assertEqual(row.coverage, "not_measured")
        self.assertEqual(row.reading, record.reading.reading.reading)
        self.assertEqual(row.reading_note, record.reading.reading.note)

    def test_an_output_that_misses_or_repeats_a_row_is_refused_by_its_code(
        self,
    ) -> None:
        # Migrates "a second call in one reply is refused" and "a recorded query
        # refuses another": every row gets exactly one reading, or the output is
        # refused with a typed code naming the rows, and repaired.
        cases = {
            "missing": ([1], "chunk_rows_missing:2"),
            "repeated": ([1, 1, 2], "chunk_rows_repeated:1"),
            "unknown": ([1, 2, 3], "chunk_rows_unknown:3"),
        }
        for name, (rows, code) in cases.items():
            sent = []

            async def respond(messages, info, rows=rows, sent=sent):
                complete = [row["row"] for row in chunk_rows(messages)]
                sent.append(complete)
                numbers = rows if len(sent) == 1 else complete
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "record_item_readings",
                            canonical_json(
                                {
                                    "readings": [
                                        {
                                            "row": number,
                                            "reading": "no_concern",
                                            "concerns": [],
                                            "note": "Clean call.",
                                        }
                                        for number in numbers
                                    ]
                                }
                            ),
                            tool_call_id=f"readings-{len(sent)}",
                        )
                    ]
                )

            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                journey, adapter = _observed(Path(temporary).resolve(), cardinality=2)
                _run_stage(journey, ScriptedModelEndpoint(respond), adapter)
                journey = Journey.open(journey.path)
                validations = [
                    event["payload"]
                    for event in read_evidence(journey)
                    if event["event_type"] == "AgentOutputValidated"
                    and event["payload"]["agent_role"] == "table_reader"
                ]
                ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")

                self.assertEqual(
                    [item["status"] for item in validations], ["invalid", "valid"]
                )
                self.assertEqual(validations[0]["errors"], [code])
                self.assertIn(code, validations[0]["feedback"])
                # The correction gives the chunk's row range, so an unknown
                # row is told where the rows end.
                self.assertIn(
                    "This chunk has rows 1 to 2; record exactly one reading for "
                    f"each row: {code}",
                    validations[0]["feedback"],
                )
                self.assertEqual(
                    {item.reading.status for item in fold.items}, {"recorded"}
                )

    def test_an_incoherent_reading_is_refused_by_its_own_type_and_repaired(
        self,
    ) -> None:
        # Standards r1 S1 / Spec r1 S2: a reading the recorded item reading would
        # refuse is refused as the output's own schema error, by its type, and
        # repaired; the item then keeps its corrected reading.
        cases = {
            "repeated concern": (
                {"reading": "concern", "concerns": ["other", "other"]},
                "readings.0:reading_concerns_repeated",
            ),
            "concern without a code": (
                {"reading": "concern", "concerns": []},
                "readings.0:reading_concerns_mismatch",
            ),
        }
        for name, (bad, code) in cases.items():
            sent = []

            async def respond(messages, info, bad=bad, sent=sent):
                sent.append(1)
                reading = (
                    bad
                    if len(sent) == 1
                    else {"reading": "concern", "concerns": ["other"]}
                )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "record_item_readings",
                            canonical_json(
                                {
                                    "readings": [
                                        {"row": 1, "note": "Look again.", **reading}
                                    ]
                                }
                            ),
                            tool_call_id=f"readings-{len(sent)}",
                        )
                    ]
                )

            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                journey, adapter = _observed(Path(temporary).resolve(), cardinality=1)
                _run_stage(journey, ScriptedModelEndpoint(respond), adapter)
                journey = Journey.open(journey.path)
                validations = [
                    event["payload"]
                    for event in read_evidence(journey)
                    if event["event_type"] == "AgentOutputValidated"
                    and event["payload"]["agent_role"] == "table_reader"
                ]
                failed = _facts(journey, "InvestigationTableChunkFailed")
                ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")

                self.assertEqual(len(sent), 2)
                self.assertEqual(
                    [item["status"] for item in validations], ["invalid", "valid"]
                )
                self.assertEqual(validations[0]["errors"], [code])
                self.assertTrue(validations[0]["feedback"])
                self.assertEqual(failed, [])
                (item,) = fold.items
                self.assertEqual(item.reading.status, "recorded")
                self.assertEqual(item.reading.reading.concerns, ("other",))

    def test_a_changed_default_never_changes_or_refuses_a_runs_recorded_table(
        self,
    ) -> None:
        # Spec r1 S1: a Run that names no investigation_table freezes the named
        # defaults its first window and first chunk used; a later default change
        # applies only to what the Run has not used yet.
        from unittest import mock

        def defaults(chunk_size, window_flank_bases):
            def changed(**values):
                return VcfTableDefaults(
                    **{
                        "chunk_size": chunk_size,
                        "window_flank_bases": window_flank_bases,
                        **values,
                    }
                )

            return mock.patch(
                "bio_neb.investigation.vcf_table_reading.VcfTableDefaults", changed
            )

        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve())
            journey = _measure(journey, adapter)
            model = TableReaderModel()
            # The window batch runs under today's defaults (±100) ...
            advance_table_reading(
                journey, environ=ENVIRON, model_factory=model.model, adapter=adapter
            )
            # ... then the defaults change before any chunk.
            with defaults(1, 50):
                _run_stage(Journey.open(journey.path), model.model, adapter)
                journey = Journey.open(journey.path)
                windows = _facts(journey, "InvestigationItemWindowRecorded")
                bound = _facts(journey, "InvestigationTableChunkBound")
                self.assertIsNotNone(read_completion(journey).candidate)
            # Back to today's defaults: the Run's recorded values still read.
            with defaults(7, 25):
                pending = compile_finalization(
                    CompileFinalizationCommand(journey_file=str(journey.path))
                ).pending_request

        self.assertEqual({fact.flank_bases for _, fact in windows}, {100})
        # The chunk size was first used after the change, so the new default holds.
        self.assertEqual([fact.chunk_size for _, fact in bound], [1, 1, 1])
        assert pending is not None
        self.assertEqual(len(pending.basis.item_evidence), 3)

    def test_a_stopped_chunk_holds_the_rest_and_a_reject_folds_them_not_run(
        self,
    ) -> None:
        # Migrates "a stopped query holds the rest and a reject folds them not run".
        # One chunk at a time, so the stop holds chunks not yet dispatched.
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(),
                investigation_table={"chunk_size": 1},
                investigation_concurrency={"planner_members": 1, "table_chunks": 1},
            )
            model = ScriptedModelEndpoint(_text_reply)
            _run_stage(journey, model, adapter)
            journey = Journey.open(journey.path)
            held = read_table_reading_stage(journey)
            request = project_agent_work_review(read_evidence(journey)).pending_request
            self.assertIsNone(read_completion(journey).candidate)

            respond_agent_work_review(
                journey,
                request.request_event_id,
                RejectAgentWorkResponse(
                    action="reject",
                    reason="Skip the reading.",
                    asserted_by="researcher",
                ),
            )
            journey = Journey.open(journey.path)
            _run_stage(journey, model, adapter)
            journey = Journey.open(journey.path)
            ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")
            candidate = read_completion(journey).candidate

        assert held is not None and request is not None
        self.assertEqual(request.agent_role, "table_reader")
        # The reject declares the stage's own consequence: the rest stop, folded
        # not_run (ADR 0047); a chunk is no Investigation member.
        reject_effect = next(
            item.effect for item in request.legal_actions if item.action == "reject"
        )
        self.assertIn("no further Attempt of this role starts", reject_effect)
        self.assertNotIn("other members continue", reject_effect)
        self.assertEqual(
            [item.reading.status for item in fold.items],
            ["stopped", "not_run", "not_run"],
        )
        self.assertEqual(
            [item.reading.reason for item in fold.items[1:]],
            ["stage_stopped", "stage_stopped"],
        )
        # Every lookup result and window is kept and Finalization proceeds.
        self.assertEqual(
            [item.lookup_outcome for item in fold.items],
            [state.outcome for state in held.collection.item_states],
        )
        self.assertEqual({item.window.status for item in fold.items}, {"recorded"})
        self.assertIsNotNone(candidate)


def _held():
    """A reading owner's store: each piece derived once, a refusal never held."""
    held = {}

    def once(key, check):
        if key not in held:
            held[key] = check()
        return held[key]

    return held, once


def _checked(timeline, checked_once):
    from bio_neb.investigation.investigation import project_investigation
    from bio_neb.investigation.vcf_table_reading import validate_table_reading

    try:
        validate_table_reading(
            timeline, project_investigation(timeline), checked_once=checked_once
        )
    except ValueError as error:
        return ("refused", str(error))
    return ("read", None)


class HeldTableCheckTests(RunPrerequisites, unittest.TestCase):
    """Each window's check and chunk's rows are held by their admitted events
    (#437): a held check equals the fresh check, refusals included."""

    def events(self):
        # A Run whose table the coordinator read: windows, chunks and the fold.
        return read_evidence(
            Journey.open(self.prerequisite(FINALIZATION_PENDING).journey_file)
        )

    def test_held_checks_equal_fresh_checks_at_every_prefix_of_a_read_table(
        self,
    ) -> None:
        import copy

        events = self.events()
        window = next(
            e for e in events if e["event_type"] == "InvestigationItemWindowRecorded"
        )
        # A window moved before the lookup it binds: the whole timeline reads,
        # the prefix ending at the window lacks the lookup and refuses.
        moved = [
            e for e in copy.deepcopy(events) if e["event_id"] != window["event_id"]
        ]
        lookup = next(
            i
            for i, e in enumerate(moved)
            if e["event_id"] == window["payload"]["evidence_event_id"]
        )
        moved.insert(
            lookup,
            {**copy.deepcopy(window), "parent_ids": [moved[lookup - 1]["event_id"]]},
        )
        compared = {}
        for name, timeline in (("retained", events), ("moved window", moved)):
            held, once = _held()
            # The owner read the whole Run first, as a command does.
            self.assertEqual(_checked(timeline, once), ("read", None), name)
            for index in range(1, len(timeline) + 1):
                prefix = timeline[:index]
                with self.subTest(timeline=name, prefix=index):
                    expected = _checked(prefix, None)
                    self.assertEqual(_checked(prefix, once), expected)
                compared[(name, expected[0])] = compared.get((name, expected[0]), 0) + 1
            self.assertIn(("table_window", window["event_id"]), {k[:2] for k in held})
        self.assertGreater(compared.get(("retained", "read"), 0), 0, compared)
        self.assertGreater(compared.get(("moved window", "refused"), 0), 0, compared)

    def test_a_refused_window_or_chunk_check_is_never_held(self) -> None:
        import copy

        events = self.events()
        window = next(
            e for e in events if e["event_type"] == "InvestigationItemWindowRecorded"
        )
        chunk = next(
            e for e in events if e["event_type"] == "InvestigationTableChunkBound"
        )
        held, once = _held()
        self.assertEqual(_checked(events, once), ("read", None))
        self.assertEqual(
            sorted({key[0] for key in held}), ["table_rows", "table_window"]
        )
        for name, change, message in (
            (
                "window",
                lambda event: event["event_id"] == window["event_id"]
                and event["payload"].update(evidence_event_id=chunk["event_id"]),
                "item window binds no retained lookup",
            ),
            (
                "chunk",
                lambda event: event["event_id"] == chunk["event_id"]
                and event["payload"]["rows"][0].update(target="changed:1"),
                "table chunk changed its VCF table rows",
            ),
        ):
            hostile = copy.deepcopy(events)
            for event in hostile:
                change(event)
            refused, refusing = _held()
            with self.subTest(change=name):
                for checked_once in (refusing, None):
                    self.assertEqual(
                        _checked(hostile, checked_once), ("refused", message)
                    )
                if name == "window":
                    self.assertNotIn(window["event_id"], {key[1] for key in refused})


class _MissingExecutableAdapter(ObservationAdapter):
    """The lookup runs; once ``executable_gone`` is set, a query finds none."""

    executable_gone = False

    def run(self, argv, *, timeout_seconds, stdout_path=None):
        if self.executable_gone and lookup_command(argv) == "variant_query":
            raise FileNotFoundError(argv[0])
        return super().run(
            argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
        )


def _attempt_window(timeline, attempt_id):
    start = next(
        index
        for index, event in enumerate(timeline)
        if event["event_type"] == "AttemptStarted"
        and event["payload"]["attempt_id"] == attempt_id
    )
    end = next(
        index
        for index, event in enumerate(timeline)
        if event["event_type"] == "InvestigationTableChunkRead"
        and event["payload"]["attempt_id"] == attempt_id
    )
    return timeline[start : end + 1]


def _hostile_fold(journey, statuses):
    """Append a fold that claims the given reading statuses (#344 r1 S1/Spec 2)."""
    stage = read_table_reading_stage(journey)
    assert stage is not None
    items = []
    for item, status in zip(
        fold_item_evidence(
            stage.collection_event_id, stage.collection, read_evidence(journey)
        ),
        statuses,
        strict=True,
    ):
        field = (
            ItemReadingField(status="not_run", reason="stage_stopped")
            if status == "not_run"
            else ItemReadingField(
                status=status, attempt_id=item.reading.attempt_id or "attempt"
            )
        )
        items.append(item.model_copy(update={"reading": field}))
    journey.record(
        "InvestigationItemEvidenceCollected",
        {
            "collection_event_id": stage.collection_event_id,
            "collection_state_id": stage.collection.collection_state_id,
            "items": [item.model_dump(mode="json") for item in items],
        },
        [journey.head_event_id],
    )
    return Journey.open(journey.path)


def _edit_journey(journey: Journey, event_type: str, change) -> bytes:
    edited = []
    for line in journey.path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") == event_type:
            change(event["payload"])
            line = json.dumps(event, separators=(",", ":"), sort_keys=True)
        edited.append(line)
    journey.path.write_text("\n".join(edited) + "\n", encoding="utf-8")
    return journey.path.read_bytes()


class VcfTableClosureTests(unittest.TestCase):
    """The fold accounts a closed stage only; each fact rests on its own evidence."""

    def test_a_fold_before_any_reading_or_stop_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve(), cardinality=1)
            journey = _measure(journey, adapter)
            # The window batch runs; no chunk has read the item.
            advance_table_reading(
                journey,
                environ=ENVIRON,
                model_factory=TableReaderModel().model,
                adapter=adapter,
            )
            hostile = _hostile_fold(Journey.open(journey.path), ["not_run"])
            with self.assertRaisesRegex(ValueError, "fold changed its items"):
                read_table_reading_stage(hostile)
            with self.assertRaisesRegex(ValueError, "fold changed its items"):
                read_completion(hostile)

    def test_a_fold_beside_a_pending_review_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(),
                cardinality=2,
                investigation_table={"chunk_size": 1},
            )
            _run_stage(journey, ScriptedModelEndpoint(_text_reply), adapter)
            journey = Journey.open(journey.path)
            self.assertIsNotNone(
                project_agent_work_review(read_evidence(journey)).pending_request
            )
            hostile = _hostile_fold(journey, ["stopped", "not_run"])
            with self.assertRaisesRegex(ValueError, "precedes its stage's Review"):
                read_completion(hostile)

    def test_a_chunk_without_its_terminal_reads_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve(), cardinality=1)
            _run_stage(journey, TableReaderModel().model, adapter)
            timeline = read_evidence(Journey.open(journey.path))
            bound = next(
                index
                for index, event in enumerate(timeline)
                if event["event_type"] == "InvestigationTableChunkBound"
            )
            validated = next(
                index
                for index, event in enumerate(timeline)
                if event["event_type"] == "AgentOutputValidated"
                and event["payload"]["agent_role"] == "table_reader"
            )
            collection_event_id, collection = next(
                (event["event_id"], fact)
                for event in timeline
                if (fact := validate_investigation_collection_evidence_fact(event))
                is not None
            )
            (unfinished,) = fold_item_evidence(
                collection_event_id, collection, timeline[: bound + 1]
            )
            # A valid output is settled only by its chunk's terminal.
            (unread,) = fold_item_evidence(
                collection_event_id, collection, timeline[: validated + 1]
            )
            (finished,) = fold_item_evidence(collection_event_id, collection, timeline)

        self.assertEqual(unfinished.reading.status, "pending")
        self.assertEqual(unread.reading.status, "pending")
        self.assertEqual(finished.reading.status, "recorded")

    def test_a_chunk_reads_its_accepted_output_over_its_own_table_rows(self) -> None:
        # Migrates "a query fact names its own admitted call".
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve(), cardinality=1)
            _run_stage(journey, TableReaderModel().model, adapter)
            timeline = read_evidence(Journey.open(journey.path))
        ((event, read),) = [
            (event, validate_item_evidence_fact(event))
            for event in timeline
            if event["event_type"] == "InvestigationTableChunkRead"
        ]
        validate_table_chunk_facts(timeline, _attempt_window(timeline, read.attempt_id))
        hostile = json.loads(json.dumps(timeline))
        changed = next(e for e in hostile if e["event_id"] == event["event_id"])
        changed["payload"]["readings"][0]["note"] = "A note no model wrote."
        with self.assertRaisesRegex(ValueError, "changed its accepted output"):
            validate_table_chunk_facts(
                hostile, _attempt_window(hostile, read.attempt_id)
            )

    def test_finalization_reads_no_window_or_chunk_its_lineage_refuses(self) -> None:
        # Migrates "Finalization reads no query fact its lineage refuses": the
        # public reader and command refuse and write nothing.
        from bio_neb.reporting.finalization_commands import (
            FinalizationApplicationRefused,
        )

        seen = []

        def wider_window(payload):
            # A self-consistent second window, read one base wider upstream than
            # the Run's first window froze.
            seen.append(payload["work_item_id"])
            if len(seen) != 2:
                return
            result = payload["result"]
            # The window's regions file, as its receipt keeps it.
            regions = result["query_invocation"]["regions_file"]
            contig, _, end = regions["text"].rstrip("\n").split("\t")
            start = result["target"]["one_based_position"] - 101
            regions["text"] = f"{contig}\t{start}\t{end}\n"
            regions["sha256"] = hashlib.sha256(regions["text"].encode()).hexdigest()
            payload["flank_bases"] = result["flank_bases"] = 101
            if "window_start" in result:
                result["window_start"] = start

        def changed_row(payload):
            payload["rows"][0]["target"] = "NC_000019.10:1 A>C"

        def changed_authority(payload):
            payload["authority_id"] = "investigation-authority:" + "f" * 64

        def changed_reading(payload):
            payload["readings"][0]["reading"] = "concern"
            payload["readings"][0]["concerns"] = ["other"]

        for name, event_type, change, reason in (
            (
                "window",
                "InvestigationItemWindowRecorded",
                wider_window,
                "window changed its lookup",
            ),
            (
                "row",
                "InvestigationTableChunkBound",
                changed_row,
                "changed its VCF table rows",
            ),
            (
                "authority",
                "InvestigationTableChunkBound",
                changed_authority,
                "changed its collection authority",
            ),
            (
                "reading",
                "InvestigationTableChunkRead",
                changed_reading,
                "changed its accepted output",
            ),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                journey, adapter = _observed(Path(temporary).resolve(), cardinality=2)
                _run_stage(journey, TableReaderModel().model, adapter)
                self.assertIsNotNone(
                    read_completion(Journey.open(journey.path)).candidate
                )
                hostile = _edit_journey(journey, event_type, change)
                with self.assertRaisesRegex(ValueError, reason):
                    read_completion(Journey.open(journey.path))
                with self.assertRaises((ValueError, FinalizationApplicationRefused)):
                    compile_finalization(
                        CompileFinalizationCommand(journey_file=str(journey.path))
                    )
                # Refused before any Review was written.
                self.assertEqual(journey.path.read_bytes(), hostile)

    def test_a_window_fault_with_no_process_result_is_recorded_on_its_item(
        self,
    ) -> None:
        # Migrates "a query fault with no process result is recorded on its item".
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(
                Path(temporary).resolve(),
                cardinality=1,
                adapter_type=_MissingExecutableAdapter,
            )
            adapter.executable_gone = True
            model = TableReaderModel()
            _run_stage(journey, model.model, adapter)
            journey = Journey.open(journey.path)
            ((_, window),) = _facts(journey, "InvestigationItemWindowRecorded")
            ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")

        self.assertEqual(
            (window.result.outcome, window.result.code, window.result.violations),
            ("refused", "tool_failure", ("FileNotFoundError",)),
        )
        self.assertIsNone(window.result.query_invocation)
        (item,) = fold.items
        self.assertEqual(
            (item.window.status, item.window.outcome), ("recorded", "refused")
        )
        # The table shows the refusal; the item is still read.
        self.assertEqual(
            model.chunks[0][0]["window"]["refusal"], "tool_failure:FileNotFoundError"
        )
        self.assertEqual(item.reading.status, "recorded")


def _read_to_fold(journey: Journey, adapter) -> Journey:
    """Run the stage's windows and chunk steps, stopping before its fold."""
    model = TableReaderModel()
    while (stage := read_table_reading_stage(journey)) is not None and (
        stage.windows_pending or stage.unread
    ):
        advance_table_reading(
            journey, environ=ENVIRON, model_factory=model.model, adapter=adapter
        )
        journey = Journey.open(journey.path)
    return journey


def _coverage_fact(journey: Journey, work_item_id: str, **outcome) -> dict:
    """One item's coverage fact payload over its own target's span."""
    stage = read_table_reading_stage(journey)
    assert stage is not None
    (target,) = (
        fact.target
        for event in read_evidence(journey)
        if (fact := validate_investigation_work_item_started_fact(event)) is not None
        and fact.work_item_id == work_item_id
    )
    first, last = coverage_span(target)
    return {
        "schema_version": "bio-neb.item-coverage/2",
        "authority_id": stage.collection.authority.authority_id,
        "collection_event_id": stage.collection_event_id,
        "work_item_id": work_item_id,
        "contig": target.contig,
        "span_start": first,
        "span_end": last,
        **outcome,
    }


def _measured(depth: int) -> dict:
    return {
        "measurement": {
            "parameters": CoverageDefaults().model_dump(mode="json"),
            "depth": depth,
            "mean_depth": float(depth),
            "meets_minimum_depth": depth >= CoverageDefaults().minimum_depth,
        }
    }


class ItemCoverageTests(unittest.TestCase):
    """Each item's own coverage fact fills its field in the fold (#414, ADR 0047)."""

    def test_the_measurement_step_covers_every_item_before_the_windows(
        self,
    ) -> None:
        # These Runs hold no local BAM (their inputs are supplied, never
        # downloaded), so health and every item record why none was measured.
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve(), forked=True)
            lookups = len(adapter.calls)
            _run_stage(journey, TableReaderModel().model, adapter)
            journey = Journey.open(journey.path)
            types = [event["event_type"] for event in read_evidence(journey)]
            coverage = _facts(journey, "InvestigationItemCoverageRecorded")
            ((_, health),) = [
                (event, validate_run_health_fact(event))
                for event in read_evidence(journey)
                if event["event_type"] == "RunHealthMeasured"
            ]
            ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")

        first_window = types.index("InvestigationItemWindowRecorded")
        self.assertLess(types.index("RunHealthMeasured"), first_window)
        self.assertEqual(len(coverage), 3)
        self.assertLess(
            max(
                index
                for index, kind in enumerate(types)
                if kind == "InvestigationItemCoverageRecorded"
            ),
            first_window,
        )
        self.assertEqual(health.not_measured.code, "bam_unavailable")
        # No samtools ran: the adapter saw only its VCF windows.
        self.assertFalse(any("samtools" in argv[0] for argv in adapter.calls[lookups:]))
        self.assertEqual(
            [
                (item.coverage.status, item.coverage.not_measured.code)
                for item in fold.items
            ],
            [("not_measured", "bam_unavailable")] * 3,
        )
        self.assertEqual(
            [item.coverage.coverage_event_id for item in fold.items],
            [event["event_id"] for event, _ in coverage],
        )

    def test_a_recorded_coverage_fact_fills_its_items_field_once(self) -> None:
        # A restart after an item's fact repeats no fact: the step covers only
        # the items still without one.
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve())
            stage = read_table_reading_stage(journey)
            assert stage is not None
            first = stage.collection.item_states[0].work_item_id
            journey.record(
                "InvestigationItemCoverageRecorded",
                _coverage_fact(journey, first, **_measured(7)),
                [journey.head_event_id],
            )
            _run_stage(journey, TableReaderModel().model, adapter)
            journey = Journey.open(journey.path)
            facts = _facts(journey, "InvestigationItemCoverageRecorded")
            ((_, fold),) = _facts(journey, "InvestigationItemEvidenceCollected")

        self.assertEqual(len(facts), 3)
        coverage = [item.coverage for item in fold.items]
        self.assertEqual(
            [field.status for field in coverage],
            ["measured", "not_measured", "not_measured"],
        )
        self.assertEqual(coverage[0].measurement.depth, 7)
        self.assertFalse(coverage[0].measurement.meets_minimum_depth)

    def test_a_coverage_fact_its_item_cannot_support_is_refused_by_its_reason(
        self,
    ) -> None:
        def facts(journey, items):
            first, second = items
            return {
                "repeated": [
                    _coverage_fact(journey, first, **_measured(12)),
                    _coverage_fact(journey, first, **_measured(12)),
                ],
                "unknown item": [
                    {
                        **_coverage_fact(journey, first, **_measured(12)),
                        "work_item_id": "work-item:" + "f" * 64,
                    }
                ],
                "span": [
                    {
                        **_coverage_fact(journey, second, **_measured(12)),
                        "span_start": 1,
                    }
                ],
                "contig": [
                    {
                        **_coverage_fact(journey, second, **_measured(12)),
                        "contig": "NC_000001.11",
                    }
                ],
                "authority": [
                    {
                        **_coverage_fact(journey, first, **_measured(12)),
                        "authority_id": "investigation-authority:" + "f" * 64,
                    }
                ],
                "collection": [
                    {
                        **_coverage_fact(journey, first, **_measured(12)),
                        "collection_event_id": "another-collection",
                    }
                ],
            }

        reasons = {
            "repeated": "item coverage repeated its item",
            "unknown item": "item coverage names no collection item",
            "span": "item coverage changed its item's span",
            "contig": "item coverage changed its item's span",
            "authority": "item coverage changed its collection authority",
            "collection": "item coverage names no exact collection",
        }
        for name, reason in reasons.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                journey, _ = _observed(Path(temporary).resolve(), cardinality=2)
                stage = read_table_reading_stage(journey)
                assert stage is not None
                items = [item.work_item_id for item in stage.collection.item_states]
                for fact in facts(journey, items)[name]:
                    journey.record(
                        "InvestigationItemCoverageRecorded",
                        fact,
                        [journey.head_event_id],
                    )
                with self.assertRaisesRegex(ValueError, reason):
                    read_completion(Journey.open(journey.path))

    def test_a_coverage_fact_after_its_collections_fold_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve(), cardinality=2)
            stage = read_table_reading_stage(journey)
            assert stage is not None
            first = stage.collection.item_states[0].work_item_id
            fact = _coverage_fact(journey, first, **_measured(12))
            _run_stage(journey, TableReaderModel().model, adapter)
            journey = Journey.open(journey.path)
            self.assertIsNotNone(read_completion(journey).candidate)
            journey.record(
                "InvestigationItemCoverageRecorded", fact, [journey.head_event_id]
            )

            with self.assertRaisesRegex(ValueError, "item coverage follows its fold"):
                read_completion(Journey.open(journey.path))


class RunHealthFactTests(unittest.TestCase):
    """The Run's health is recorded once, at a collection whose stage is still
    open: a fact after its collection's fold is refused (ADR 0047; the "Item
    fold" and "The Run's measurement" bullets of docs/design/investigation.md)."""

    def test_a_health_fact_the_run_cannot_hold_is_refused_by_its_reason(
        self,
    ) -> None:
        def record(journey, payload):
            journey.record("RunHealthMeasured", payload, [journey.head_event_id])

        def fact(journey, **update):
            stage = read_table_reading_stage(journey)
            return {
                "schema_version": "bio-neb.run-health/2",
                "collection_event_id": stage.collection_event_id,
                "not_measured": {"code": "bam_unavailable", "detail": "no BAM"},
                **update,
            }

        # A repeated fact is refused as it is appended (the run_health family);
        # one naming no open collection when the Run is read.
        for name, reason in (
            ("repeated", "run health repeated"),
            ("collection", "run health names no open collection"),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                journey, adapter = _observed(Path(temporary).resolve(), cardinality=2)
                if name == "collection":
                    record(journey, fact(journey, collection_event_id="another"))
                    with self.assertRaisesRegex(ValueError, reason):
                        read_completion(Journey.open(journey.path))
                    continue
                payload = fact(journey)
                if name == "repeated":
                    record(journey, payload)
                else:
                    # The stage records the Run's one health fact itself.
                    _run_stage(journey, TableReaderModel().model, adapter)
                    journey = Journey.open(journey.path)
                before = journey.path.read_bytes()
                with self.assertRaisesRegex(ValueError, reason):
                    record(journey, payload)
                self.assertEqual(journey.path.read_bytes(), before)

    def test_a_health_fact_after_its_collections_fold_is_refused(self) -> None:
        # A folded Run's timeline with its one health fact moved past the fold:
        # the read-time check refuses it, whichever single fact it is.
        with tempfile.TemporaryDirectory() as temporary:
            journey, adapter = _observed(Path(temporary).resolve(), cardinality=2)
            _run_stage(journey, TableReaderModel().model, adapter)
            timeline = read_evidence(Journey.open(journey.path))
        health = next(e for e in timeline if e["event_type"] == "RunHealthMeasured")
        moved = [e for e in timeline if e is not health] + [health]
        self.assertGreater(
            moved.index(health),
            next(
                i
                for i, e in enumerate(moved)
                if e["event_type"] == "InvestigationItemEvidenceCollected"
            ),
        )
        from bio_neb.investigation.investigation import project_investigation
        from bio_neb.investigation.vcf_table_reading import validate_table_reading

        with self.assertRaisesRegex(ValueError, "run health names no open collection"):
            validate_table_reading(moved, project_investigation(timeline))


class ItemEvidenceFieldTests(unittest.TestCase):
    def test_coverage_is_declared_unfilled_and_a_measurement_needs_its_payload(
        self,
    ) -> None:
        self.assertEqual(ItemCoverageField().status, "not_measured")
        with self.assertRaises(ValueError):
            ItemCoverageField(status="measured")
        measurement = CoverageMeasurement(
            parameters=CoverageDefaults(),
            depth=12,
            mean_depth=12.5,
            meets_minimum_depth=True,
        )
        with self.assertRaises(ValueError):
            ItemCoverageField(measurement=measurement)
        with self.assertRaisesRegex(ValueError, "coverage measurement is incoherent"):
            # Its minimum met or not, as its depth and minimum state it.
            CoverageMeasurement(
                parameters=CoverageDefaults(),
                depth=12,
                mean_depth=12.5,
                meets_minimum_depth=False,
            )
        measured = ItemCoverageField(
            status="measured",
            coverage_event_id="event-1",
            measurement=measurement,
        )
        self.assertEqual(measured.measurement.parameters.minimum_depth, 10)

    def test_the_four_facts_stay_adjustable(self) -> None:
        # Each fact names its shape version, the concern codes are a closed set,
        # and the chunk size and window are the Run's selection.
        for fact, version in (
            (InvestigationItemWindowRecorded, "bio-neb.item-window/1"),
            (InvestigationTableChunkBound, "bio-neb.table-chunk-bound/1"),
            (InvestigationTableChunkRead, "bio-neb.table-chunk-read/1"),
            (InvestigationTableChunkFailed, "bio-neb.table-chunk-failed/1"),
        ):
            with self.subTest(fact.__name__):
                self.assertEqual(fact.model_fields["schema_version"].default, version)
        work_item = "work-item:" + "a" * 64
        self.assertIn("nearby_variant", CONCERN_CODES)
        with self.assertRaises(ValidationError):
            ItemReading(
                work_item_id=work_item,
                reading="concern",
                concerns=("not_a_code",),
                note="x",
            )
        with self.assertRaisesRegex(ValueError, "only when it has one"):
            ItemReading(
                work_item_id=work_item,
                reading="no_concern",
                concerns=("other",),
                note="x",
            )
        self.assertEqual(
            (VcfTableDefaults().chunk_size, VcfTableDefaults().window_flank_bases),
            (10, 100),
        )
        omitted = RuntimeConfiguration.model_validate(runtime_configuration())
        self.assertNotIn(
            "investigation_table",
            omitted.runtime_selections().model_dump(mode="json"),
        )
        self.assertIsNone(omitted.runtime_selections().investigation_table)
        selected = RuntimeConfiguration.model_validate(
            {
                **runtime_configuration(),
                "investigation_table": {"chunk_size": 25, "window_flank_bases": 50},
            }
        ).runtime_selections()
        self.assertEqual(
            selected.investigation_table,
            VcfTableDefaults(chunk_size=25, window_flank_bases=50),
        )


if __name__ == "__main__":
    unittest.main()
