"""In-memory Journey and activity fakes for contact-free admission tests."""

from __future__ import annotations

import uuid

from tests.reference_fakes import (
    append,
    payload,
)


class FakeJourney:
    def __init__(self, timeline):
        self.timeline = timeline

    def read(self):
        return self.timeline


class FakeActivity:
    def __init__(self, journey: FakeJourney):
        self.journey = journey
        self.counter = 0
        self.closed = False

    def record(self, fact):
        self.counter += 1
        return append(
            self.journey.timeline,
            f"agent-record-{self.counter}-{uuid.uuid4()}",
            fact.event_type,
            payload(fact),
        )

    def start_child(self, **_kwargs):
        return self

    def succeeded(self, event):
        return event

    def failed(self, event):
        self.closed = True
        return event


def forge_request_binding(timeline, validation):
    """Rebind a forged call's request to its forged effective arguments.

    The request the call answers records each server-bound value; a coherent forgery
    rewrites it so the later relationship check must reject the claim.
    """
    by_id = {event["event_id"]: event for event in timeline}
    response = next(
        event
        for event in reversed(timeline[: timeline.index(validation)])
        if event["event_type"] == "ModelResponseRecorded"
    )
    contact = by_id[response["payload"]["physical_contact_started_event_id"]]
    request = by_id[contact["payload"]["logical_request_event_id"]]
    payload = validation["payload"]
    tool = next(
        item
        for item in request["payload"]["compatible_adapter_evidence"]["request"][
            "function_tools"
        ]
        if item["name"] == payload["tool_name"]
    )
    tool["metadata"]["bound_arguments"] = {
        name: payload["arguments"][name] for name in payload["bound_argument_names"]
    }


def received_reply(actions, *, cut_off=False, output_tokens=None, resends=0):
    """One reply as admission receives it: its retained identity and its Model turn.

    ``actions`` are the reply's calls in the order sent, each with its arguments
    as the model wrote them (``{"invalid_json": raw}`` for no JSON object), for
    contact-free routing tests.
    """
    import json

    from bio_neb.agentic_journey.model_observation import ReceivedReply
    from bio_neb.agentic_journey.model_turn import ModelCall, ModelTurn, read_arguments

    reply = ReceivedReply(
        response_event_id=str(uuid.uuid4()),
        request_number=1,
        requested_max_tokens=None,
        resends=resends,
    )
    turn = ModelTurn(
        calls=tuple(
            ModelCall(
                action.tool_name,
                action.tool_call_id,
                read_arguments(
                    action.arguments["invalid_json"]
                    if action.arguments_invalid_json
                    else json.dumps(action.arguments)
                ),
            )
            for action in actions
        ),
        cut_off=cut_off,
        output_tokens=output_tokens,
    )
    return reply, turn
