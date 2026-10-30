"""#326's two runaway Nemotron replies, replayed through the real planning loop (#336).

Shared by the turn-routing tests and the eval scorer's tests. The replay keeps each
reply's calls, their sent order, multiplicities, argument text and ending. Its
identities are mapped onto the test registration's request, so the same calls are
lawful there: the TaxId term ``6239`` becomes ``9606`` and the accession
``GCF_000002985.6`` becomes the supplied ``GCF_000001405.40``; every other argument is
replayed as recorded. #472 (Q2 a) has the model type the sample and the supplied accession
from the researcher's words, which the replies predate: each complete call of those Tools
gains them, quoting the prompt, and the supplied choice cites the prompt's event.
"""

import json
from pathlib import Path

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.usage import RequestUsage

from bio_neb.application.advancement import (
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.queries import ReviewJourneyQuery, review_journey
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.application_fakes import _registered
from tests.planning_assignment_fakes import PlanningComposition

FIXTURE = Path(__file__).parent / "fixtures" / "issue326" / "runaway-replies.json"
PROMPT = (
    "Analyze rs1234 for sample-001 Homo sapiens (TaxId 9606) using reference "
    "GCF_000001405.40 and produce VCF."
)
_MAPPED = {"6239": "9606", "GCF_000002985.6": "GCF_000001405.40"}
_TYPED = {
    "attest_sample": {"sample_identifier": "sample-001", "quote": "sample-001"},
    "record_supplied_reference": {
        "assembly_accession": "GCF_000001405.40",
        "quote": "GCF_000001405.40",
    },
}


def _typed(reply: ModelResponse, info) -> ModelResponse:
    """The reply with #472's typed arguments added to each complete call (above)."""
    (statement,) = next(
        tool.parameters_json_schema["properties"]["assertion_event_id"]["enum"]
        for tool in info.function_tools
        if tool.name == "record_supplied_reference"
    )
    parts = []
    for part in reply.parts:
        typed = _TYPED.get(getattr(part, "tool_name", None))
        try:
            arguments = json.loads(part.args) if typed else None
        except json.JSONDecodeError:
            arguments = None  # A cut-off call stays as it was sent.
        if arguments is not None:
            arguments.update(typed)
            if part.tool_name == "record_supplied_reference":
                arguments["assertion_event_id"] = statement
            part = ToolCallPart(
                part.tool_name, json.dumps(arguments), tool_call_id=part.tool_call_id
            )
        parts.append(part)
    return ModelResponse(
        parts=parts, finish_reason=reply.finish_reason, usage=reply.usage
    )


def recorded_reply(index: int, *, finish_reason: str | None = None) -> ModelResponse:
    """#326 reply ``index`` (4 or 5) as the model sent it, its identities mapped."""
    reply = json.loads(FIXTURE.read_text())["replies"][index - 4]
    text = {}
    parts = []
    for run in reply["calls"]:
        arguments = run["args"]
        for recorded, mapped in _MAPPED.items():
            arguments = arguments.replace(recorded, mapped)
        for _ in range(run["sent"]):
            parts.append(
                ToolCallPart(
                    run["tool_name"], arguments, tool_call_id=f"r{index}-{len(parts)}"
                )
            )
        text[run["tool_name"]] = arguments
    assert len(parts) == reply["call_count"]
    return ModelResponse(
        parts=parts,
        finish_reason=finish_reason or reply["finish_reason"],
        usage=RequestUsage(output_tokens=reply["output_tokens"]),
    )


class Replays(PlanningComposition):
    """Works as scripted, but answers the first reference-stage request with a
    recorded #326 reply."""

    def __init__(self, receipt, paths, reply):
        super().__init__(receipt, paths)
        self.reply = reply
        self.replayed = False
        self.sent_after = []

    async def respond(self, messages, info):
        offered = {tool.name for tool in info.function_tools}
        if self.replayed and not self.sent_after:
            self.sent_after.append(messages)
        if "attest_sample" in offered and not self.replayed:
            self.replayed = True
            return _typed(self.reply, info)
        return await super().respond(messages, info)


def of(events, event_type):
    return [e for e in events if e["event_type"] == event_type]


def replay_326(directory: Path, reply: ModelResponse, **registration):
    """Register a Run in ``directory``, answer its first reference-stage request with
    ``reply`` and advance it to its stop; the Journey is reopened by the checked reader.

    Returns the events, the replayed reply's response and routing records, the turn's
    events up to the next request, and that request (``None`` if none followed).
    """
    receipt, paths = _registered(directory, prompt_text=PROMPT, **registration)
    composition = Replays(receipt, paths, reply)
    with use_run_advancement_composition(composition):
        advance(ResumeRun(journey_file=receipt.journey_file))
    # Reopen reads the routed turn against the raw reply.
    review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
    events = read_evidence(Journey.open(Path(receipt.journey_file)))
    (response,) = [
        e
        for e in of(events, "ModelResponseRecorded")
        if len(
            e["payload"]["compatible_adapter_evidence"]["response"]["message"]["parts"]
        )
        == len(reply.parts)
    ]
    (routed,) = [
        e
        for e in of(events, "AgentTurnRouted")
        if e["payload"]["response_event_id"] == response["event_id"]
    ]
    turn = events[events.index(routed) + 1 :]
    later = of(turn, "ModelRequestRecorded")
    turn = turn[: turn.index(later[0])] if later else turn
    return events, response, routed, turn, later[0] if later else None
