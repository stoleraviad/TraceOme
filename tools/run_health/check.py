"""Option (d): each unsure item of a batch checked by itself (owner, #406 G4 d1).

The batch (`items.measure_items`) measures every Work item at once, flags the ones it
cannot place and marks the measured ones it is unsure of. Each of these then gets its own
agent loop, all of them at once: the model sees the item, why it is in doubt and, when
measured, its batch window; it may read regions of the reference through one Tool
(`measure_range`, the per-item read, under the batch's read filters), and answers with a
verdict. The check's result is the read the verdict names, measured, never the model's
words. No item waits on another; a check that fails records its failure, with the module
that raised it, and the others go on. Every model request is counted before it is sent,
against the check's and all checks' caps. Each check's model exchange is kept as
`messages.json` beside its reads, so the checks can be replayed without contact
(`replay_model`). Lane only: no Journey is written.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelRetry, TextOutput, capture_run_messages
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.openai import OpenAIProvider

from bio_neb.application.input_journey import RunInputsRegistered
from bio_neb.investigation.item_evidence import CoverageMeasurement
from bio_neb.investigation.observation_defaults import CoverageDefaults
from bio_neb.journey import Journey

from .health import run_probe
from .items import (
    ITEMS_NAME,
    ItemFlag,
    ItemsCoverage,
    SpanCoverage,
    fresh_directory,
    run_region_files,
)
from .local_process import ProbedProcessRunner
from .region import (
    RangeCoverage,
    RegionFiles,
    RegionSummary,
    parse_range,
    read_range,
    reference_lengths,
)
from .runner import RunRefused, ToolRunner
from .table import StrictModel

CHECKS_SCHEMA = "bio-neb.run-health-unsure-checks/1"
CHECKS_NAME = "unsure-checks.json"
# How far from its span a check may read, and the run plan's request caps (G5).
SEARCH_BASES = 10_000
ITEM_REQUESTS = 6
TOTAL_REQUESTS = 40

INSTRUCTIONS = """\
You check the read coverage of one Work item of a sequencing Run. A batch pass measured
every item at once and was in doubt about this one; you look at it by itself.

You get the item's place on the reference and why the batch is in doubt: either it could
not place the item (its contig is not a contig of the reference, or its span ends past
the contig), or it measured the item and is unsure of the measurement (you then also get
the batch's summary of the window around it). You may read the depth and GC of regions of
the item's own contig with `measure_range` (contig:start-end, 1-based and inclusive); the
Tool tells you which regions it allows. Read at least one region before you answer (the
span's window, or wider), and only what you need.

Answer with one verdict for the item's own span:
- covered: reads cover the span at or above the minimum depth;
- low_coverage: reads reach the span, but below the minimum depth;
- no_coverage: no read reaches the span;
- cannot_tell: the reads do not settle it, or the item cannot be placed (say why).
Name the region you read whose reading you rely on (exactly as you asked for it; "none"
only when the Tool allows no region at all), and give a one-sentence reason.
"""


class CoverageVerdict(StrictModel):
    """What a check's model concludes about its item's span, and from which region."""

    verdict: Literal["covered", "low_coverage", "no_coverage", "cannot_tell"]
    region: str
    reason: str


# The whole system message: the instructions and the final answer's shape. The answer is
# plain text parsed here, not the endpoint's JSON mode, which asks for a JSON reply: under
# it the model made no Tool call in a live run (the cause is not yet confirmed).
_SYSTEM = (
    INSTRUCTIONS
    + "\nWhen you have read what you need, give your final answer as one JSON object, and "
    "nothing else in that message, compatible with this schema:\n\n"
    + json.dumps(CoverageVerdict.model_json_schema())
)


class CoverageResult(StrictModel):
    """The read a verdict relies on, as measured: the region, the item's span's
    `CoverageMeasurement` (its least depth; None when the region does not hold the
    whole span) and the region's summary."""

    region: str
    measurement: CoverageMeasurement | None
    window: RegionSummary


class UnsureCheck(StrictModel):
    """One span's check, for every Work item the batch flagged or was unsure of there
    for one reason (`flags`, sharing the span as they share its measurement): its
    verdict and the measured `result` the verdict names
    (None when it failed, did not run, or named a region it did not read), every region
    it read, the model requests it sent (counted before each is sent, failed or not), its
    seconds, and its failure as `<module>.<type>: <message>`. `messages` is its model
    exchange as far as it went, relative to the checks' directory (None when no request
    was made)."""

    flags: tuple[ItemFlag, ...]
    verdict: CoverageVerdict | None
    result: CoverageResult | None
    reads: tuple[RangeCoverage, ...]
    model_requests: int
    seconds: float
    failure: str | None
    messages: str | None


class UnsureChecks(StrictModel):
    """Every flagged and unsure item's check, the model that made them and the requests
    they sent against the cap."""

    schema_version: Literal["bio-neb.run-health-unsure-checks/1"] = CHECKS_SCHEMA
    model: str
    checks: tuple[UnsureCheck, ...]
    model_requests: int
    # None for a replay, which makes no contact and so has no cap.
    request_cap: int | None
    seconds: float


class NotReplayable(Exception):
    """A replayed check asked for a response the earlier run did not record."""


class RequestCapReached(Exception):
    """A model request refused before it was sent: the check's or all checks' cap."""


class _Budget:
    """The model requests all checks together may still send."""

    def __init__(self, cap: int | None) -> None:
        self.cap = cap
        self.sent = 0


class _CountedModel(WrapperModel):
    """The check's model, counting each request before it is sent: a request past the
    check's `item_requests` or the shared budget is refused instead."""

    def __init__(
        self, wrapped: Model, budget: _Budget, item_requests: int | None
    ) -> None:
        super().__init__(wrapped)
        self.budget = budget
        self.item_requests = item_requests
        self.sent = 0

    def _admit(self) -> None:
        if self.item_requests is not None and self.sent >= self.item_requests:
            raise RequestCapReached(
                f"the check's {self.item_requests} requests are sent"
            )
        if self.budget.cap is not None and self.budget.sent >= self.budget.cap:
            raise RequestCapReached(f"the checks' {self.budget.cap} requests are sent")
        self.sent += 1
        self.budget.sent += 1

    async def request(self, messages, model_settings, model_request_parameters):
        self._admit()
        return await super().request(messages, model_settings, model_request_parameters)

    def request_stream(self, *args, **kwargs):
        self._admit()
        return super().request_stream(*args, **kwargs)


class _Item(StrictModel):
    flags: tuple[ItemFlag, ...]
    span: SpanCoverage | None


def check_unsure(
    coverage: ItemsCoverage,
    files: RegionFiles,
    *,
    model: Model,
    runner: ToolRunner,
    output_dir: Path,
    item_requests: int | None = ITEM_REQUESTS,
    total_requests: int | None = TOTAL_REQUESTS,
) -> UnsureChecks:
    """Check each span of the batch's flagged and unsure items by itself, all at once:
    one check per span and reason, for every Work item flagged there.

    Each check may send `item_requests` model requests, and all of them together
    `total_requests`: every request is counted against both caps before it is sent,
    and one past either is refused (`RequestCapReached`), so a check that meets the
    total stops with that failure. A replay (`replay_model`) makes no contact and passes
    no caps (None): each check replays its own recorded exchange, whatever the others'
    timing. `output_dir` must not exist; check `n` reads into
    `<output_dir>/<n>/` and the result is written last as `unsure-checks.json`.
    """
    started = time.monotonic()
    output_dir = fresh_directory(output_dir)
    spans = {
        work_item_id: span
        for span in coverage.spans
        for work_item_id in span.work_item_ids
    }
    shared: dict[tuple, list[ItemFlag]] = {}
    for flag in (*coverage.flagged, *coverage.unsure):
        key = (flag.contig, flag.span_start, flag.span_end, flag.code)
        shared.setdefault(key, []).append(flag)
    items = [
        _Item(flags=tuple(flags), span=spans.get(flags[0].work_item_id))
        for flags in shared.values()
    ]
    budget = _Budget(total_requests)
    contigs = reference_lengths(files)

    async def every() -> list[UnsureCheck]:
        return list(
            await asyncio.gather(
                *(
                    _check(
                        item,
                        files,
                        model=_CountedModel(model, budget, item_requests),
                        runner=runner,
                        directory=output_dir / str(index),
                        defaults=coverage.parameters,
                        contigs=contigs,
                    )
                    for index, item in enumerate(items)
                )
            )
        )

    checks = asyncio.run(every())
    result = UnsureChecks(
        model=model.model_name,
        checks=tuple(checks),
        model_requests=budget.sent,
        request_cap=total_requests,
        seconds=round(time.monotonic() - started, 3),
    )
    with (output_dir / CHECKS_NAME).open("x") as written:
        written.write(result.model_dump_json(indent=2))
    return result


def check_run_items(
    journey_file: Path,
    cache_dir: Path,
    health_dir: Path,
    items_dir: Path,
    output_dir: Path,
    *,
    model: Model | None = None,
    runner: ToolRunner | None = None,
    item_requests: int | None = ITEM_REQUESTS,
    total_requests: int | None = TOTAL_REQUESTS,
) -> UnsureChecks:
    """Check the unsure items of a Run's batch (`items_dir`'s `items-coverage.json`)
    over the Run's own files (`items.run_region_files`), with the Run's own samtools
    and model (`run_model`) unless `runner` or `model` is given."""
    outcome, files = run_region_files(journey_file, cache_dir, health_dir)
    coverage_file = items_dir / ITEMS_NAME
    try:
        coverage = ItemsCoverage.model_validate_json(coverage_file.read_bytes())
    except FileNotFoundError:
        raise RunRefused(
            "file_missing", f"no batch result at {coverage_file}"
        ) from None
    except (OSError, ValueError) as error:
        raise RunRefused(
            "items_unreadable",
            f"{coverage_file} is not a readable batch result: {error}",
        ) from error
    if coverage.slice_record is not None and coverage.slice_record.inputs["bam"] != str(
        files.bam
    ):
        raise RunRefused(
            "items_other_bam",
            f"{coverage_file} read {coverage.slice_record.inputs['bam']}; the Run's "
            f"BAM is {files.bam}",
        )
    return check_unsure(
        coverage,
        files,
        model=model or run_model(journey_file),
        runner=runner or ProbedProcessRunner(run_probe(outcome)),
        output_dir=output_dir,
        item_requests=item_requests,
        total_requests=total_requests,
    )


def run_model(journey_file: Path) -> Model:
    """The model the Run's registered selections name for its agents
    (`agent_invocation`'s `model_reference`), on its endpoint, with the credential
    from the environment variable they name; no runtime file is read (#440)."""
    registration = next(
        event
        for event in Journey.open(journey_file).events_view()
        if event["event_type"] == "RunInputsRegistered"
    )
    fact = RunInputsRegistered.model_validate(
        {"event_type": registration["event_type"], **registration["payload"]}
    )
    selections = fact.runtime_selections
    selected = selections.model_roster[selections.agent_invocation.model_reference]
    credential = os.environ.get(selected.credential_env)
    if not credential:
        raise RunRefused(
            "model_credential_missing",
            f"{selected.credential_env} is not set for {selected.model}",
        )
    # No hidden retries: each request the check counts is one attempt on the wire.
    client = AsyncOpenAI(base_url=selected.endpoint, api_key=credential, max_retries=0)
    return OpenAIChatModel(
        selected.model,
        provider=OpenAIProvider(openai_client=client),
        settings={"parallel_tool_calls": False, "max_tokens": 16_384},
    )


def replay_model(checks_dir: Path) -> FunctionModel:
    """A model that answers each check with the responses an earlier run of the checks
    (`checks_dir`) recorded for the same item, in order: the checks replayed without
    contact. An item the earlier run did not exchange about is refused by the model."""
    recorded = {}
    for messages_file in sorted(checks_dir.glob("*/messages.json")):
        exchange = ModelMessagesTypeAdapter.validate_json(messages_file.read_bytes())
        if exchange:
            recorded[_asked(exchange)] = [
                message for message in exchange if isinstance(message, ModelResponse)
            ]

    def respond(messages, info: AgentInfo) -> ModelResponse:
        sent = sum(isinstance(message, ModelResponse) for message in messages)
        responses = recorded.get(_asked(messages))
        if responses is None:
            raise NotReplayable("the earlier checks hold no exchange for this check")
        if sent >= len(responses):
            raise NotReplayable(
                f"the earlier check received {len(responses)} responses; "
                f"request {sent + 1} has none to replay"
            )
        return responses[sent]

    return FunctionModel(respond, model_name=f"replay:{checks_dir}")


def _asked(messages) -> str:
    """The prompt an exchange opened with: what identifies its item."""
    (prompt,) = [
        part.content for part in messages[0].parts if isinstance(part, UserPromptPart)
    ]
    return prompt


async def _check(
    item: _Item,
    files: RegionFiles,
    *,
    model: _CountedModel,
    runner: ToolRunner,
    directory: Path,
    defaults: CoverageDefaults,
    contigs: dict[str, int],
) -> UnsureCheck:
    started = time.monotonic()
    flag = item.flags[0]
    reads: list[tuple[str, RangeCoverage, CoverageMeasurement | None]] = []
    attempts = 0
    # One system message, first: the Run's endpoint refuses a second (L1).
    agent = Agent(
        model,
        instructions=_SYSTEM,
        output_type=TextOutput(_verdict_text),
        retries=model.item_requests or ITEM_REQUESTS,
    )
    readable = flag.contig in contigs

    # A check answers from its own reading (L5): a verdict before any read, or naming
    # a region it did not read, goes back to the model. Only an item whose contig the
    # reference lacks, which has nothing to read, may answer unread.
    @agent.output_validator
    def answers_from_its_reading(verdict: CoverageVerdict) -> CoverageVerdict:
        if readable and verdict.region not in {region for region, _, _ in reads}:
            raise ModelRetry(
                "answer from your own reading: read the region with measure_range, "
                "then name it exactly as you asked for it"
            )
        return verdict

    # A refused region goes back to the model as often as its requests allow.
    @agent.tool_plain(
        name="measure_range", retries=model.item_requests or ITEM_REQUESTS
    )
    def read_region(region: str) -> dict:
        """Depth and GC of `region` (contig:start-end, 1-based, inclusive) on the
        Run's reference, under the batch's read filters."""
        nonlocal attempts
        try:
            contig, start, end = parse_range(region)
        except RunRefused as refused:
            raise ModelRetry(refused.detail) from None
        if not _allowed(flag, item.span is not None, contig, start, end, contigs):
            raise ModelRetry(_allowance(flag, item.span is not None))
        attempts += 1
        try:
            read, depths = read_range(
                files,
                region,
                runner=runner,
                output_dir=directory / f"read-{attempts}",
                defaults=defaults,
            )
        except RunRefused as refused:
            raise ModelRetry(f"{refused.code}: {refused.detail}") from None
        reads.append(
            (region, read, _span_measurement(flag, contig, start, depths, defaults))
        )
        return read.summary.model_dump(mode="json", exclude={"parameters"})

    directory.mkdir()
    messages = directory / "messages.json"
    verdict = failure = None
    with capture_run_messages() as exchanged:
        try:
            verdict = (await agent.run(_prompt(item, contigs))).output
        except Exception as error:  # every failure is this check's record
            failure = f"{type(error).__module__}.{type(error).__qualname__}: {error}"
    messages.write_bytes(ModelMessagesTypeAdapter.dump_json(exchanged))
    # The last read of the region the verdict names.
    relied = next(
        (read for read in reversed(reads) if verdict and read[0] == verdict.region),
        None,
    )
    return UnsureCheck(
        flags=item.flags,
        verdict=verdict,
        result=None
        if relied is None
        else CoverageResult(
            region=verdict.region, measurement=relied[2], window=relied[1].summary
        ),
        reads=tuple(read for _, read, _ in reads),
        model_requests=model.sent,
        seconds=round(time.monotonic() - started, 3),
        failure=failure,
        messages=f"{directory.name}/{messages.name}",
    )


def _verdict_text(text: str) -> CoverageVerdict:
    """The final answer, a JSON `CoverageVerdict` (a Markdown fence around it is
    dropped); anything else goes back to the model."""
    answer = (
        text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    )
    try:
        return CoverageVerdict.model_validate_json(answer.strip())
    except ValueError as error:
        raise ModelRetry(
            "your final answer must be one JSON object compatible with the schema, "
            f"and nothing else: {error}"
        ) from None


def _allowed(
    flag: ItemFlag,
    measured: bool,
    contig: str,
    start: int,
    end: int,
    contigs: dict[str, int],
) -> bool:
    """Whether the check may read this region: only the item's own contig (G4 d1),
    near the item when the batch measured it, and anywhere on it, in reads of a bounded
    length, when the batch could not place it (a contig the reference does not name has
    nothing to read)."""
    if contig != flag.contig or contig not in contigs:
        return False
    if measured:
        return (
            flag.span_start - SEARCH_BASES <= start
            and end <= flag.span_end + SEARCH_BASES
        )
    return end - start + 1 <= 2 * SEARCH_BASES + 1


def _allowance(flag: ItemFlag, measured: bool) -> str:
    if measured:
        return (
            f"read {flag.contig} within {SEARCH_BASES} bases of "
            f"{flag.span_start}-{flag.span_end}"
        )
    return (
        f"read only {flag.contig}, at most {2 * SEARCH_BASES + 1} bases at a time "
        "(nothing, when the reference does not name it)"
    )


def _span_measurement(
    flag: ItemFlag,
    contig: str,
    start: int,
    depths: list[int],
    defaults: CoverageDefaults,
) -> CoverageMeasurement | None:
    """The item's span's least depth in a read that holds the whole span."""
    first, last = flag.span_start - start, flag.span_end - start
    if contig != flag.contig or first < 0 or last >= len(depths):
        return None
    over_span = depths[first : last + 1]
    least = min(over_span)
    return CoverageMeasurement(
        parameters=defaults,
        depth=least,
        mean_depth=sum(over_span) / len(over_span),
        meets_minimum_depth=least >= defaults.minimum_depth,
    )


def _prompt(item: _Item, contigs: dict[str, int]) -> str:
    flag, span = item.flags[0], item.span
    named_items = ", ".join(each.work_item_id for each in item.flags)
    place = (
        f"Work items {named_items}: span {flag.contig}:{flag.span_start}-"
        f"{flag.span_end}.\n"
        f"Why the batch is in doubt ({flag.code}): {flag.detail}.\n"
    )
    if span is None:
        named = ", ".join(
            f"{name} ({length} bases)" for name, length in contigs.items()
        )
        return (
            place + f"The batch could not place it. The reference's contigs: {named}."
        )
    window = span.window.model_dump(mode="json", exclude={"parameters"})
    return place + (
        f"Least depth over the span: {span.measurement.depth} (minimum "
        f"{span.measurement.parameters.minimum_depth}); positions with no read: "
        f"{list(span.uncovered_positions)}.\n"
        f"Batch window summary: {window}"
    )
