"""Retained provider recovery grammar fixture; no adapter contact."""

import copy
import uuid

from bio_neb.failure import ModelContactPolicy

ATTEMPT_ID = "00000000-0000-4000-8000-000000000001"
ENDPOINT = {
    "provider": "nebius",
    "model": "Qwen/Qwen3.5-397B-A17B",
    "contacted_url": "https://api.tokenfactory.nebius.com/v1",
}


def _uuid(name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ticket-86-test:{name}"))


def _pending_review_timeline(*, two_contacts=False, server_status=503):
    timeline = []

    def add(name, event_type, payload, parents=None):
        event = {
            "event_id": _uuid(name),
            "event_type": event_type,
            "parent_ids": (
                list(parents)
                if parents is not None
                else ([timeline[-1]["event_id"]] if timeline else [])
            ),
            "payload": copy.deepcopy(payload),
        }
        timeline.append(event)
        return event

    frontier = add("frontier", "InputsRegistered", {})
    lifecycle = add(
        "attempt-lifecycle",
        "activity_started",
        {"activity_id": _uuid("activity")},
        [frontier["event_id"]],
    )
    add(
        "attempt-started",
        "AttemptStarted",
        {
            "attempt_id": ATTEMPT_ID,
            "agent_role": "planning_agent",
            "input_set_id": "inputs",
            "invocation_overrides": {},
        },
        [lifecycle["event_id"]],
    )
    request = add(
        "logical-request",
        "ModelRequestRecorded",
        {
            "attempt_id": ATTEMPT_ID,
            "request_number": 1,
            "nebius_model_endpoint": ENDPOINT,
            # The window this timeline records, stated whole: its 1-second delay is
            # a 2-second base at 0.5 jitter, whatever today's defaults are.
            "contact_policy": ModelContactPolicy(
                contacts=2,
                base_delay_seconds=2,
                max_delay_seconds=30,
                max_wait_seconds=180,
            ).model_dump(),
            "compatible_adapter_evidence": {
                "model_name": ENDPOINT["model"],
                "model_system": "openai",
                "request": {
                    "messages": [],
                    "model_settings": {},
                    "function_tools": [],
                    "output_tools": [],
                    "allow_text_output": True,
                    "output_mode": "text",
                    "instruction_parts": None,
                },
            },
        },
    )
    contact = add(
        "contact-started",
        "ModelPhysicalContactStarted",
        {
            "attempt_id": ATTEMPT_ID,
            "request_number": 1,
            "logical_request_event_id": request["event_id"],
            "contact_ordinal": 1,
            "completion_certainty": "completion_unknown",
            "policy_disposition": "pending",
        },
    )
    failed_contact = add(
        "contact-failed",
        "ModelPhysicalContactFailed",
        {
            "attempt_id": ATTEMPT_ID,
            "request_number": 1,
            "logical_request_event_id": request["event_id"],
            "contact_started_event_id": contact["event_id"],
            "contact_ordinal": 1,
            "failure_class": "transient",
            "error_type": "ModelHTTPError",
            "contact_seconds": 1.0,
            "http_status_code": server_status if two_contacts else 429,
            "failure_reason": "provider_failure" if two_contacts else "rate_limited",
            "completion_certainty": "response_received",
            "policy_disposition": (
                "automatic_retry_scheduled"
                if two_contacts
                else "availability_review_requested"
            ),
            "retry_after_disposition": "missing" if two_contacts else "over_cap",
        },
    )
    terminal_contact = failed_contact
    physical_contact_count = 1
    exhaustion_reason = "retry_after_over_cap"
    if two_contacts:
        scheduled = add(
            "retry-scheduled",
            "ModelAvailabilityRetryScheduled",
            {
                "attempt_id": ATTEMPT_ID,
                "request_number": 1,
                "logical_request_event_id": request["event_id"],
                "failed_contact_event_id": failed_contact["event_id"],
                "failed_contact_ordinal": 1,
                "next_contact_ordinal": 2,
                "delay_seconds": 1,
                "delay_source": "exponential_jitter",
            },
        )
        completed = add(
            "retry-completed",
            "ModelAvailabilityRetryDelayCompleted",
            {
                "attempt_id": ATTEMPT_ID,
                "request_number": 1,
                "logical_request_event_id": request["event_id"],
                "retry_scheduled_event_id": scheduled["event_id"],
                "failed_contact_ordinal": 1,
                "next_contact_ordinal": 2,
                "delay_seconds": 1,
            },
        )
        second_contact = add(
            "second-contact-started",
            "ModelPhysicalContactStarted",
            {
                "attempt_id": ATTEMPT_ID,
                "request_number": 1,
                "logical_request_event_id": request["event_id"],
                "contact_ordinal": 2,
                "retry_delay_completed_event_id": completed["event_id"],
                "completion_certainty": "completion_unknown",
                "policy_disposition": "pending",
            },
        )
        terminal_contact = add(
            "second-contact-failed",
            "ModelPhysicalContactFailed",
            {
                "attempt_id": ATTEMPT_ID,
                "request_number": 1,
                "logical_request_event_id": request["event_id"],
                "contact_started_event_id": second_contact["event_id"],
                "contact_ordinal": 2,
                "failure_class": "transient",
                "error_type": "ModelHTTPError",
                "contact_seconds": 1.0,
                "http_status_code": server_status,
                "failure_reason": "provider_failure",
                "completion_certainty": "response_received",
                "policy_disposition": "availability_review_requested",
                "retry_after_disposition": "missing",
            },
        )
        physical_contact_count = 2
        exhaustion_reason = "automatic_retry_exhausted"
    logical_failure = add(
        "logical-failure",
        "ModelRequestFailed",
        {
            "attempt_id": ATTEMPT_ID,
            "request_number": 1,
            "physical_contact_count": physical_contact_count,
            "last_physical_contact_event_id": terminal_contact["event_id"],
            "availability_exhaustion_reason": exhaustion_reason,
            "nebius_model_endpoint": ENDPOINT,
            "error_type": "ModelHTTPError",
            "http_status_code": server_status if two_contacts else 429,
            "failure_reason": "provider_failure" if two_contacts else "rate_limited",
        },
    )
    review = add(
        "review-requested",
        "ModelAvailabilityReviewRequested",
        {
            "attempt_id": ATTEMPT_ID,
            "agent_role": "planning_agent",
            "retry_frontier_event_id": frontier["event_id"],
            "logical_request_event_id": request["event_id"],
            "logical_failure_event_id": logical_failure["event_id"],
            "request_number": 1,
            "nebius_model_endpoint": ENDPOINT,
            "physical_contact_count": physical_contact_count,
            "exhaustion_reason": exhaustion_reason,
            # No answer body: the cause's key is the exhaustion.
            "fault_key": f"model endpoint: {exhaustion_reason}",
        },
    )
    add(
        "attempt-failed",
        "AttemptFailed",
        {
            "attempt_id": ATTEMPT_ID,
            "code": "model_availability_exhausted",
            "source_event_id": logical_failure["event_id"],
        },
    )
    return timeline, review["event_id"]


def model_request_payload(attempt_id):
    """Use the valid retained request shape while varying its owning Attempt."""
    events, _ = _pending_review_timeline()
    payload = next(
        e["payload"] for e in events if e["event_type"] == "ModelRequestRecorded"
    )
    return {**payload, "attempt_id": attempt_id}
