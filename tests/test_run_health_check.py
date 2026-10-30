"""Option (d): each unsure item of a batch checked by itself by an agent loop (lane H,
#406 G4 d1).

The batch runs over the HG001 fixture slices (#405) with this environment's samtools;
the model is scripted (`FunctionModel`), so every verdict and request below is the
script's, and every read is the per-item region read the Tool performs.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
import pysam
from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from bio_neb.agentic_journey.runtime_config import read_runtime_config
from bio_neb.application.input_journey import RunInputsRegistered
from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.journey import Journey
from tests.run_health_fakes import pysam_probed_runner
from tests.run_prerequisites import prerequisite
from tools.run_health import RunRefused, RunResult
from tools.run_health.check import (
    CHECKS_NAME,
    UnsureChecks,
    check_unsure,
    replay_model,
    run_model,
)
from tools.run_health.items import ItemFlag, ItemTarget, measure_items
from tools.run_health.region import RegionFiles

FIXTURE = Path(__file__).parent / "fixtures" / "run_health" / "hg001-slices"
FLANK = 100


class _Scripted(Exception):
    """A scripted model fault."""


class _FailFirstDepth:
    """This environment's samtools, except that the first depth read exits 1."""

    def __init__(self, runner):
        self.runner = runner
        self.name = runner.name
        self.failed = False

    def bind(self, entry):
        return self.runner.bind(entry)

    def run(self, invocation):
        if invocation.argv[1] == "depth" and not self.failed:
            self.failed = True
            return RunResult(
                exit_code=1,
                identity={},
                executed_argv=invocation.argv,
                stderr=b"scripted failure",
            )
        return self.runner.run(invocation)


def _targets(rows):
    return tuple(
        ItemTarget(
            work_item_id=row["work_item_id"],
            contig=row["original"]["contig"],
            one_based_position=row["rebased_position"],
            ref=row["original"]["ref"],
        )
        for row in rows
    )


def _verdict(verdict, region, reason="scripted"):
    return ModelResponse(
        parts=[
            TextPart(
                json.dumps({"verdict": verdict, "region": region, "reason": reason})
            )
        ]
    )


def _read(region, call_id="read-1"):
    return ModelResponse(
        parts=[ToolCallPart("measure_range", {"region": region}, tool_call_id=call_id)]
    )


class UnsureCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        run = cls.root / "run"
        shutil.copytree(FIXTURE, run)
        pysam.index("-c", str(run / "aligned.bam"))
        pysam.faidx(str(run / "reference.fa"))
        cls.files = RegionFiles(
            bam=run / "aligned.bam",
            bam_index=run / "aligned.bam.csi",
            reference=run / "reference.fa",
            reference_index=run / "reference.fa.fai",
        )
        cls.rows = json.loads((FIXTURE / "items.json").read_text())["items"]
        cls.runner = pysam_probed_runner(cls.root / "tools")
        cls.coverage = measure_items(
            cls.files,
            tuple(
                ItemTarget(
                    work_item_id=row["work_item_id"],
                    contig=row["original"]["contig"],
                    one_based_position=row["rebased_position"],
                    ref=row["original"]["ref"],
                )
                for row in cls.rows
            ),
            flank_bases=FLANK,
            runner=cls.runner,
            output_dir=cls.root / "batch",
        )
        # The batch's one unsure item: rs7412, below the minimum with reads nearby.
        (cls.flag,) = cls.coverage.unsure
        (cls.span,) = [
            span
            for span in cls.coverage.spans
            if cls.flag.work_item_id in span.work_item_ids
        ]
        cls.window = f"{cls.span.contig}:{cls.span.window.start}-{cls.span.window.end}"

    def setUp(self):
        self.output = self.root / f"checks-{self._testMethodName}"

    def _check(self, respond, coverage=None, **caps):
        return check_unsure(
            coverage or self.coverage,
            self.files,
            model=FunctionModel(respond),
            runner=self.runner,
            output_dir=self.output,
            **caps,
        )

    def _second_unsure(self):
        """The batch with a second unsure item: rs429358, as if it were in doubt."""
        (other,) = [
            row["work_item_id"]
            for row in self.rows
            if row["display_label"] == "rs429358"
        ]
        (span,) = [span for span in self.coverage.spans if other in span.work_item_ids]
        flag = ItemFlag(
            work_item_id=other,
            contig=span.contig,
            span_start=span.span_start,
            span_end=span.span_end,
            code="low_depth",
            detail="scripted",
        )
        return self.coverage.model_copy(
            update={"unsure": (*self.coverage.unsure, flag)}
        ), other

    def test_each_unsure_item_is_checked_by_itself_with_its_reads_and_verdict(self):
        def respond(messages, info):
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(self.window)
            return _verdict("low_coverage", self.window)

        checks = self._check(respond)

        (check,) = checks.checks
        self.assertEqual(check.flags, (self.flag,))
        self.assertEqual(
            (check.verdict.verdict, check.verdict.region),
            ("low_coverage", self.window),
        )
        (read,) = check.reads
        # The Tool's region read equals the batch's own window summary, and the result
        # is that read measured over the item's span, as the batch measured it.
        self.assertEqual(read.summary, self.span.window)
        self.assertEqual(
            (check.result.region, check.result.measurement, check.result.window),
            (self.window, self.span.measurement, self.span.window),
        )
        self.assertEqual((check.model_requests, check.failure), (2, None))
        self.assertEqual((checks.model_requests, checks.request_cap), (2, 40))
        self.assertTrue((self.output / check.messages).is_file())
        self.assertEqual(
            UnsureChecks.model_validate_json((self.output / CHECKS_NAME).read_text()),
            checks,
        )

    def test_a_region_off_the_items_contig_or_far_from_it_is_sent_back_unread(self):
        far = f"{self.span.contig}:{self.span.span_end + 20_000}-"
        far += str(self.span.span_end + 20_100)
        returned = []

        def respond(messages, info):
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            returned.extend(
                part.content
                for message in messages[-1:]
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            )
            if sent == 0:
                return _read("NC_000001.11:1-10")
            if sent == 1:
                return _read(far, call_id="read-2")
            if sent == 2:
                return _read(self.window, call_id="read-3")
            return _verdict("low_coverage", self.window)

        (check,) = self._check(respond).checks

        # Only the allowed region was read; the two others went back unread.
        self.assertEqual([read.summary for read in check.reads], [self.span.window])
        refusals = [content for content in returned if "within" in str(content)]
        self.assertEqual(len(refusals), 2)
        for content in refusals:
            self.assertIn(f"within 10000 bases of {self.span.span_start}", content)
        self.assertEqual(check.verdict.verdict, "low_coverage")

    def test_a_failing_check_records_its_module_and_the_other_check_finishes(self):
        coverage, other = self._second_unsure()

        def respond(messages, info):
            if other in str(messages[0]):
                raise _Scripted("the endpoint broke")
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(self.window)
            return _verdict("low_coverage", self.window)

        checks = self._check(respond, coverage=coverage)

        verdicts = {check.flags[0].work_item_id: check for check in checks.checks}
        failed = verdicts[other]
        self.assertIsNone(failed.verdict)
        # The failed request was sent, so it counts.
        self.assertEqual(failed.model_requests, 1)
        self.assertEqual(
            failed.failure, "tests.test_run_health_check._Scripted: the endpoint broke"
        )
        self.assertEqual(
            verdicts[self.flag.work_item_id].verdict.verdict, "low_coverage"
        )

    def test_checks_stop_at_their_own_and_the_total_request_caps(self):
        coverage, other = self._second_unsure()

        def respond(messages, info):
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            return _read(self.window, call_id=f"read-{sent + 1}")

        checks = self._check(
            respond, coverage=coverage, item_requests=2, total_requests=3
        )

        # Both checks start; together they send exactly the total, each at most its
        # own cap, and each stops at the cap it met.
        self.assertEqual(
            [check.flags[0].work_item_id for check in checks.checks],
            [self.flag.work_item_id, other],
        )
        self.assertEqual(checks.model_requests, 3)
        self.assertEqual(sum(check.model_requests for check in checks.checks), 3)
        self.assertTrue(all(check.model_requests <= 2 for check in checks.checks))
        failures = [check.failure for check in checks.checks]
        self.assertTrue(all("RequestCapReached" in failure for failure in failures))
        self.assertTrue(any("checks' 3 requests" in failure for failure in failures))
        self.assertIsNone(checks.checks[0].verdict)

    def test_work_items_at_one_span_share_one_check(self):
        (same,) = [
            row["work_item_id"]
            for row in self.rows
            if row["display_label"] == "rs4149056"
        ][1:]
        (span,) = [span for span in self.coverage.spans if same in span.work_item_ids]
        shared = tuple(
            ItemFlag(
                work_item_id=work_item_id,
                contig=span.contig,
                span_start=span.span_start,
                span_end=span.span_end,
                code="low_depth",
                detail="scripted",
            )
            for work_item_id in span.work_item_ids
        )
        prompts = []

        def respond(messages, info):
            prompts.append(str(messages[0]))
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(window)
            return _verdict("covered", window)

        window = f"{span.contig}:{span.window.start}-{span.window.end}"
        checks = self._check(
            respond, coverage=self.coverage.model_copy(update={"unsure": shared})
        )

        (check,) = checks.checks
        self.assertEqual(check.flags, shared)
        self.assertEqual(checks.model_requests, 2)
        for work_item_id in span.work_item_ids:
            self.assertIn(work_item_id, prompts[0])

    def test_the_wire_request_holds_one_system_message_first(self):
        bodies = []

        def answer(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(400, json={"message": "scripted"})

        client = AsyncOpenAI(
            base_url="http://endpoint.invalid/v1",
            api_key="scripted",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
        )
        model = OpenAIChatModel(
            "scripted", provider=OpenAIProvider(openai_client=client)
        )

        checks = check_unsure(
            self.coverage,
            self.files,
            model=model,
            runner=self.runner,
            output_dir=self.output,
        )

        (body,) = bodies
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        # The Tool can be called: no JSON mode, which constrains every reply to JSON.
        self.assertNotIn("response_format", body)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(
            [tool["function"]["name"] for tool in body["tools"]], ["measure_range"]
        )
        system = body["messages"][0]["content"]
        self.assertIn("You check the read coverage", system)
        self.assertIn('"verdict"', system)
        (check,) = checks.checks
        self.assertIn("ModelHTTPError", check.failure)
        self.assertEqual(check.model_requests, 1)

    def test_an_answer_before_reading_goes_back_until_the_check_reads(self):
        returned = []

        def respond(messages, info):
            returned.extend(
                str(part.content)
                for part in messages[-1].parts
                if isinstance(part, RetryPromptPart)
            )
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _verdict("covered", self.window)
            if sent == 1:
                return _read(self.window)
            if sent == 2:
                return _verdict("low_coverage", "none")
            return _verdict("low_coverage", self.window)

        (check,) = self._check(respond).checks

        self.assertEqual(len(returned), 2)
        for content in returned:
            self.assertIn("answer from your own reading", content)
        self.assertEqual(
            (check.verdict.verdict, check.result.region, check.model_requests),
            ("low_coverage", self.window, 4),
        )
        self.assertEqual(check.result.measurement, self.span.measurement)

    def _flagged(self, target):
        return measure_items(
            self.files,
            (target,),
            flank_bases=FLANK,
            runner=self.runner,
            output_dir=self.output.with_name(self.output.name + "-batch"),
        )

    def test_a_flagged_item_on_a_contig_the_reference_lacks_reads_nothing(self):
        coverage = self._flagged(
            ItemTarget(
                work_item_id="work-item:unknown",
                contig="chr19",
                one_based_position=5,
                ref="A",
            )
        )
        prompts, returned = [], []

        def respond(messages, info):
            prompts.append(str(messages[0]))
            returned.extend(
                part.content
                for part in messages[-1].parts
                if isinstance(part, RetryPromptPart)
            )
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(self.window)
            return _verdict("cannot_tell", "none")

        (check,) = self._check(respond, coverage=coverage).checks

        self.assertEqual(
            (check.flags[0].code, check.flags[0].contig), ("contig_unknown", "chr19")
        )
        self.assertIn(f"{self.span.contig} (", prompts[0])
        self.assertEqual((check.reads, check.result), ((), None))
        (refusal,) = returned
        self.assertIn("read only chr19", refusal)
        self.assertEqual(check.verdict.verdict, "cannot_tell")

    def test_a_flagged_item_reads_its_own_contig_only(self):
        coverage = self._flagged(
            ItemTarget(
                work_item_id="work-item:beyond",
                contig=self.span.contig,
                one_based_position=10_000_000,
                ref="A",
            )
        )
        other = next(
            row for row in self.rows if row["original"]["contig"] != self.span.contig
        )
        elsewhere = f"{other['original']['contig']}:1-100"
        returned = []

        def respond(messages, info):
            returned.extend(
                part.content
                for part in messages[-1].parts
                if isinstance(part, RetryPromptPart)
            )
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(elsewhere)
            if sent == 1:
                return _read(self.window, call_id="read-2")
            return _verdict("cannot_tell", self.window)

        (check,) = self._check(respond, coverage=coverage).checks

        self.assertEqual(check.flags[0].code, "span_beyond_contig")
        (refusal,) = returned
        self.assertIn(f"read only {self.span.contig}", refusal)
        (read,) = check.reads
        self.assertEqual(read.summary, self.span.window)
        # The read does not hold the item's span: no measurement of it.
        self.assertEqual(
            (check.result.window, check.result.measurement), (self.span.window, None)
        )

    def test_a_checks_reads_keep_the_batchs_read_filters(self):
        strict = CoverageDefaults(minimum_depth=100)
        coverage = measure_items(
            self.files,
            _targets(self.rows),
            flank_bases=FLANK,
            runner=self.runner,
            output_dir=self.output.with_name(self.output.name + "-batch"),
            defaults=strict,
        )
        flag = next(
            flag
            for flag in coverage.unsure
            if flag.work_item_id == self.flag.work_item_id
        )

        def respond(messages, info):
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(self.window)
            return _verdict("low_coverage", self.window)

        checks = self._check(
            respond, coverage=coverage.model_copy(update={"unsure": (flag,)})
        )

        (check,) = checks.checks
        (read,) = check.reads
        self.assertEqual(read.summary.parameters, strict)
        self.assertEqual(check.result.measurement.parameters, strict)

    def test_a_failed_read_keeps_its_output_and_the_next_read_gets_its_own(self):
        runner = _FailFirstDepth(self.runner)

        def respond(messages, info):
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent < 2:
                return _read(self.window, call_id=f"read-{sent + 1}")
            return _verdict("low_coverage", self.window)

        checks = check_unsure(
            self.coverage,
            self.files,
            model=FunctionModel(respond),
            runner=runner,
            output_dir=self.output,
        )

        (check,) = checks.checks
        self.assertEqual(len(check.reads), 1)
        self.assertTrue((self.output / "0" / "read-1" / "depth").is_dir())
        self.assertEqual(check.reads[0].depth_record.inputs["bam"], str(self.files.bam))
        self.assertTrue((self.output / "0" / "read-2" / "reference").is_dir())

    def test_the_final_answer_is_parsed_from_text_and_other_text_goes_back(self):
        returned = []

        def respond(messages, info):
            returned.extend(
                str(part.content)
                for part in messages[-1].parts
                if isinstance(part, RetryPromptPart)
            )
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(self.window)
            if sent == 1:
                return ModelResponse(parts=[TextPart("I think it is low.")])
            fenced = json.dumps(
                {"verdict": "low_coverage", "region": self.window, "reason": "read"}
            )
            return ModelResponse(parts=[TextPart(f"```json\n{fenced}\n```")])

        (check,) = self._check(respond).checks

        (refusal,) = returned
        self.assertIn("one JSON object compatible with the schema", refusal)
        self.assertEqual(
            (check.verdict.verdict, check.result.region), ("low_coverage", self.window)
        )

    def test_a_failed_exchange_replays_as_not_replayable(self):
        def respond(messages, info):
            raise _Scripted("the endpoint broke")

        self._check(respond)
        replayed = check_unsure(
            self.coverage,
            self.files,
            model=replay_model(self.output),
            runner=self.runner,
            output_dir=self.output.with_name(self.output.name + "-replay"),
        )

        (check,) = replayed.checks
        self.assertIsNone(check.verdict)
        self.assertIn("NotReplayable", check.failure)
        self.assertIn("request 1 has none to replay", check.failure)

    def test_a_cut_short_exchange_does_not_take_a_complete_ones_replay(self):
        coverage, other = self._second_unsure()
        alone = {
            work_item_id: coverage.model_copy(
                update={
                    "unsure": tuple(
                        flag
                        for flag in coverage.unsure
                        if flag.work_item_id == work_item_id
                    )
                }
            )
            for work_item_id in (self.flag.work_item_id, other)
        }

        def complete(messages, info):
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(self.window)
            return _verdict("low_coverage", self.window)

        def cut_short(messages, info):
            return ModelResponse(parts=[TextPart("not a verdict")])

        recorded = self.output.with_name(self.output.name + "-recorded")
        recorded.mkdir()
        exchanges = []
        for index, (work_item_id, respond, requests) in enumerate(
            ((self.flag.work_item_id, complete, 6), (other, cut_short, 1))
        ):
            self.output = self.output.with_name(f"{self.output.name}-{index}")
            (check,) = self._check(
                respond, coverage=alone[work_item_id], item_requests=requests
            ).checks
            exchanges.append(check)
            (recorded / str(index)).mkdir()
            shutil.copy(
                self.output / check.messages, recorded / str(index) / "messages.json"
            )
        self.assertIn("RequestCapReached", exchanges[1].failure)

        replayed = check_unsure(
            coverage,
            self.files,
            model=replay_model(recorded),
            runner=self.runner,
            output_dir=recorded.with_name(recorded.name + "-replay"),
            item_requests=None,
            total_requests=None,
        )

        first, second = replayed.checks
        self.assertEqual((first.verdict, first.failure), (exchanges[0].verdict, None))
        self.assertIsNone(second.verdict)
        self.assertIn("NotReplayable", second.failure)
        self.assertIsNone(replayed.request_cap)

    def test_kept_exchanges_replay_to_the_same_checks_without_contact(self):
        coverage, other = self._second_unsure()

        def respond(messages, info):
            sent = sum(isinstance(message, ModelResponse) for message in messages)
            if sent == 0:
                return _read(self.window)
            verdict = "covered" if other in str(messages[0]) else "low_coverage"
            return _verdict(verdict, self.window)

        recorded = self._check(respond, coverage=coverage)
        replayed = check_unsure(
            coverage,
            self.files,
            model=replay_model(self.output),
            runner=self.runner,
            output_dir=self.output.with_name(self.output.name + "-replay"),
        )

        self.assertEqual(
            [check.verdict.verdict for check in recorded.checks],
            ["low_coverage", "covered"],
        )
        self.assertEqual(
            [(check.flags, check.verdict) for check in replayed.checks],
            [(check.flags, check.verdict) for check in recorded.checks],
        )
        self.assertEqual(
            [[read.summary for read in check.reads] for check in replayed.checks],
            [[read.summary for read in check.reads] for check in recorded.checks],
        )


class RunModelTests(unittest.TestCase):
    """The check's model is the one the Run's runtime file names."""

    @classmethod
    def setUpClass(cls):
        cls.journey_file = prerequisite("executed").journey_file
        registration = next(
            event
            for event in Journey.open(cls.journey_file).events_view()
            if event["event_type"] == "RunInputsRegistered"
        )
        fact = RunInputsRegistered.model_validate(
            {"event_type": registration["event_type"], **registration["payload"]}
        )
        config = read_runtime_config(Path(fact.runtime_config_file))
        cls.selected = config.model_roster[config.agent_invocation.model_reference]

    def test_the_runs_model_on_its_endpoint_without_contact(self):
        with mock.patch.dict(os.environ, {self.selected.credential_env: "secret"}):
            model = run_model(self.journey_file)

        self.assertEqual(model.model_name, self.selected.model)
        # Each counted request is one attempt on the wire: no hidden client retries.
        self.assertEqual(model.client.max_retries, 0)
        self.assertEqual(model.base_url.rstrip("/"), self.selected.endpoint.rstrip("/"))

    def test_a_missing_credential_is_refused_by_its_variable(self):
        with mock.patch.dict(os.environ, {self.selected.credential_env: ""}):
            with self.assertRaises(RunRefused) as refused:
                run_model(self.journey_file)

        self.assertEqual(refused.exception.code, "model_credential_missing")
        self.assertIn(self.selected.credential_env, refused.exception.detail)


if __name__ == "__main__":
    unittest.main()
