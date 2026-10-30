import asyncio
import copy
import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
from bio_neb.agentic_journey.invocation import invoke_agent
from bio_neb.agentic_journey.model_availability import (
    ModelAvailabilityReviewPending,
    ModelAvailabilityReviewResponseRefused,
    RetryModelAvailabilityResponse,
    StopModelAvailabilityResponse,
    evaluate_model_availability_review_response,
    project_model_availability_review,
)
from bio_neb.agentic_journey.model_observation import JourneyObservedModel
from bio_neb.agentic_journey.planning_assignment import planning_assignment
from bio_neb.application.agent_reentry import AgentReentryRefused
from bio_neb.application.commands import (
    RespondModelAvailabilityReviewCommand,
    respond_model_availability_review,
)
from bio_neb.failure import (
    Invariant,
    ModelContactPolicy,
    ModelResponseMalformed,
    model_retry_after,
)
from bio_neb.journey import Journey, canonical_json
from tests.application_fakes import _registered
from tests.input_correction_fakes import prepare_input_correction
from tests.model_availability_fakes import (
    ATTEMPT_ID,
    ENDPOINT,
    _pending_review_timeline,
    _uuid,
)
from tests.model_endpoint_fakes import ScriptedModelEndpoint


def _http_error(status_code, retry_after=None):
    headers = {
        "authorization": "Bearer forbidden-header-secret",
        "x-should-retry": "true",
        "retry-after-ms": "1",
    }
    if retry_after is not None:
        headers["retry-after"] = retry_after
    return ModelHTTPError(
        status_code,
        ENDPOINT["model"],
        body={
            "message": "retryable forbidden-body-secret",
            "url": "https://provider.invalid/?token=forbidden-query-secret",
        },
        headers=headers,
    )


class _RecordedActivity:
    def __init__(self):
        self.facts = []
        self.events = []
        self.children = []
        self.closed = False

    def start_child(self, **kwargs):
        del kwargs
        child = _RecordedActivity()
        self.children.append(child)
        return child

    def record(self, fact):
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": fact.event_type,
            "payload": fact.model_dump(mode="json", exclude={"event_type"}),
        }
        self.facts.append(fact)
        self.events.append(event)
        return copy.deepcopy(event)

    def record_contact(self, fact):
        return self.record(fact)

    def succeeded(self, event):
        del event
        self.closed = True

    def failed(self, event):
        del event
        self.closed = True

    def cancelled(self, event):
        del event
        self.closed = True


class _ScriptedModel(Model):
    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = list(outcomes)
        self.calls = 0
        self.sent_messages = []

    @property
    def model_name(self):
        return ENDPOINT["model"]

    @property
    def system(self):
        return "openai"

    async def request(self, messages, model_settings, model_request_parameters):
        del model_settings, model_request_parameters
        self.sent_messages.append(messages)
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


def _parameters():
    return ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
        output_mode="text",
    )


def _observer(
    outcomes, *, availability_sleep=None, contact_policy=None, continuation_request=None
):
    parent = _RecordedActivity()
    model = _ScriptedModel(outcomes)
    kwargs = {}
    if availability_sleep is not None:
        kwargs["availability_sleep"] = availability_sleep
    if continuation_request is not None:
        kwargs["continuation_request"] = continuation_request
    observed = JourneyObservedModel(
        model,
        parent_activity=parent,
        attempt_id=ATTEMPT_ID,
        failure_source=AttemptFailureSource(),
        nebius_model_endpoint=ENDPOINT,
        availability_jitter=lambda: 0.5,
        # The Attempt supplies its role's resolved window; no observer default.
        contact_policy=ModelContactPolicy()
        if contact_policy is None
        else contact_policy,
        **kwargs,
    )
    return observed, model, parent


async def _pending_request(instructions, *, parameters=None):
    """One unavailable logical request, retained as a retry would load it."""
    observed, _, parent = _observer([_http_error(503, _OVER_WINDOW)])
    messages = [ModelRequest(parts=[UserPromptPart("plan")], instructions=instructions)]
    try:
        await observed.request(messages, {}, parameters or _parameters())
    except ModelAvailabilityReviewPending:
        pass
    return parent.children[0].events[0]


def _event_types(parent):
    return [event["event_type"] for event in parent.children[0].events]


# A Retry-After beyond the whole default window opens the Review at once.
_OVER_WINDOW = str(int(ModelContactPolicy().max_wait_seconds) + 1)


class DefaultContactWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_default_window_sends_once_more_within_a_minute_then_asks(self):
        # Budgets are per call, and a short failure is sent once more after a
        # jittered wait of at most a minute, then asks the availability Review.
        slept = []

        async def wait(seconds):
            slept.append(seconds)

        observed, model, parent = _observer(
            [_http_error(502), _http_error(503)], availability_sleep=wait
        )
        with self.assertRaises(ModelAvailabilityReviewPending) as pending:
            await observed.request([], {}, _parameters())
        self.assertEqual(model.calls, 2)
        # The observer's jitter is 0.5: half of the 30 s base.
        self.assertEqual(slept, [15.0])
        self.assertEqual(
            pending.exception.exhaustion_reason, "automatic_retry_exhausted"
        )
        failed = [
            fact
            for fact in parent.children[0].facts
            if fact.event_type == "ModelPhysicalContactFailed"
        ]
        self.assertEqual(
            [fact.policy_disposition for fact in failed],
            ["automatic_retry_scheduled", "availability_review_requested"],
        )


class MixedWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_malformed_reply_on_the_second_send_asks_instead_of_a_third(self):
        # The window's two sends bound every retry, a malformed reply's
        # included.
        async def wait(seconds):
            del seconds

        observed, model, parent = _observer(
            [
                _http_error(502),
                ModelResponseMalformed(),
                ModelResponse(parts=[TextPart("too late")]),
            ],
            availability_sleep=wait,
        )
        with self.assertRaises(ModelAvailabilityReviewPending) as pending:
            await observed.request([], {}, _parameters())
        self.assertEqual(model.calls, 2)
        self.assertEqual(
            pending.exception.exhaustion_reason, "automatic_retry_exhausted"
        )


class RetryAfterPolicyTests(unittest.TestCase):
    def test_http_date_retry_after_uses_remaining_time(self):
        from datetime import UTC, datetime

        now = datetime(2026, 9, 30, 12, 0, tzinfo=UTC).timestamp()
        self.assertEqual(
            model_retry_after(
                {"Retry-After": "Wed, 30 Sep 2026 12:00:45 GMT"}, now=now
            ),
            ("valid", 45),
        )

    def test_strict_ascii_retry_after_matrix(self):
        cases = {
            None: ("missing", None),
            "1": ("valid", 1),
            "30": ("valid", 30),
            "0001": ("valid", 1),
            "": ("unusable", None),
            "0": ("valid", 0),
            "-1": ("unusable", None),
            "1.0": ("unusable", None),
            " 1": ("valid", 1),
            "1 ": ("valid", 1),
            "\t1": ("valid", 1),
            "١": ("unusable", None),
            "Wed, 21 Oct 2015 07:28:00 GMT": ("valid", 0),
            "31": ("valid", 31),
            "999999999999999999999999999999999999": ("over_cap", None),
            "9" * 10_000: ("over_cap", None),
            "0" * 10_000: ("valid", 0),
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(
                    model_retry_after(_http_error(429, value).headers, now=1760000000),
                    expected,
                )


class ObservedAvailabilityRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_invocation_keeps_retry_history_equal_to_received_request(
        self,
    ):
        observed, _, parent = _observer([_http_error(503, _OVER_WINDOW)])
        with self.assertRaises(ModelAvailabilityReviewPending):
            await Agent(observed, instructions="Retained instruction.").run("plan")
        pending = parent.children[0].events[0]
        history = ModelMessagesTypeAdapter.validate_python(
            pending["payload"]["compatible_adapter_evidence"]["request"]["messages"]
        )
        observed, model, _ = _observer(
            [ModelResponse(parts=[TextPart("done")])], continuation_request=pending
        )

        result = await invoke_agent(
            Agent(observed, instructions="Retained instruction."),
            None,
            deps=None,
            policy=SimpleNamespace(
                model_requests=10,
                model_settings=SimpleNamespace(request_settings=lambda: {}),
            ),
            journey_id=str(uuid.uuid4()),
            attempt_id=ATTEMPT_ID,
            message_history=history,
        )

        self.assertEqual(result.output, "done")
        self.assertEqual(
            ModelMessagesTypeAdapter.dump_python(
                result.all_messages()[:-1], mode="json"
            ),
            ModelMessagesTypeAdapter.dump_python(model.sent_messages[0], mode="json"),
        )
        self.assertEqual(
            ModelMessagesTypeAdapter.dump_python(model.sent_messages[0], mode="json"),
            pending["payload"]["compatible_adapter_evidence"]["request"]["messages"],
        )

    async def test_six_contact_window_preserves_one_logical_request(self):
        slept = []

        async def wait(seconds):
            slept.append(seconds)

        response = ModelResponse(parts=[TextPart("recovered")])
        observed, model, parent = _observer(
            [_http_error(503, "2")] * 5 + [response],
            availability_sleep=wait,
            contact_policy=ModelContactPolicy(contacts=6),
        )
        self.assertIs(await observed.request([], {}, _parameters()), response)
        self.assertEqual(model.calls, 6)
        self.assertEqual(slept, [2] * 5)
        self.assertEqual(_event_types(parent).count("ModelRequestRecorded"), 1)
        self.assertEqual(parent.children[0].facts[-1].physical_contact_ordinal, 6)

    async def test_adapter_sends_the_exact_retained_pending_request(self):
        pending = await _pending_request('{"open_questions": []}')
        response = ModelResponse(parts=[TextPart("done")])
        observed, model, parent = _observer([response], continuation_request=pending)

        returned = await observed.request(
            ModelMessagesTypeAdapter.validate_python(
                pending["payload"]["compatible_adapter_evidence"]["request"]["messages"]
            ),
            {},
            _parameters(),
        )

        self.assertIs(returned, response)
        self.assertEqual(
            model.sent_messages[0][-1].instructions, '{"open_questions": []}'
        )
        resent = parent.children[0].events[0]["payload"]
        self.assertEqual(resent["continuation_request_event_id"], pending["event_id"])
        self.assertEqual(
            resent["compatible_adapter_evidence"]["request"],
            pending["payload"]["compatible_adapter_evidence"]["request"],
        )

    async def test_retry_with_changed_tools_messages_or_instructions_refuses_before_contact(
        self,
    ):
        pending = await _pending_request('{"open_questions": []}')
        retained = ModelMessagesTypeAdapter.validate_python(
            pending["payload"]["compatible_adapter_evidence"]["request"]["messages"]
        )
        changed_tools = ModelRequestParameters(
            function_tools=[ToolDefinition(name="added_after_failure")],
            output_tools=[],
            allow_text_output=True,
            output_mode="text",
        )
        changed_messages = [
            *retained[:-1],
            replace(retained[-1], parts=[UserPromptPart("changed after failure")]),
        ]
        changed_instructions = [
            *retained[:-1],
            replace(retained[-1], instructions="Changed after failure"),
        ]
        for label, messages, parameters in (
            ("tools", retained, changed_tools),
            ("messages", changed_messages, _parameters()),
            ("instructions", changed_instructions, _parameters()),
        ):
            with self.subTest(changed=label):
                observed, model, parent = _observer(
                    [ModelResponse(parts=[TextPart("done")])],
                    continuation_request=pending,
                )
                with self.assertRaises(Invariant) as held:
                    await observed.request(messages, {}, parameters)
                self.assertEqual(held.exception.invariant_id, "pending_request_changed")
                self.assertEqual(model.calls, 0)
                self.assertEqual(parent.children, [])

    async def test_retry_refuses_changed_instruction_parts_before_contact(self):
        parameters = replace(
            _parameters(),
            instruction_parts=[
                InstructionPart(content="First", dynamic=False),
                InstructionPart(content="Second", dynamic=True),
            ],
        )
        pending = await _pending_request("First\n\nSecond", parameters=parameters)
        retained = ModelMessagesTypeAdapter.validate_python(
            pending["payload"]["compatible_adapter_evidence"]["request"]["messages"]
        )
        for label, parts in (
            ("content", [InstructionPart(content="Changed", dynamic=True)]),
            (
                "origin",
                [
                    InstructionPart(content="First", dynamic=True),
                    InstructionPart(content="Second", dynamic=True),
                ],
            ),
            ("order", list(reversed(parameters.instruction_parts))),
            ("joined", [InstructionPart(content="First\n\nSecond", dynamic=True)]),
            ("absent", None),
        ):
            with self.subTest(changed=label):
                observed, model, parent = _observer(
                    [ModelResponse(parts=[TextPart("done")])],
                    continuation_request=pending,
                )
                with self.assertRaises(Invariant) as held:
                    await observed.request(
                        retained, {}, replace(parameters, instruction_parts=parts)
                    )
                self.assertEqual(held.exception.invariant_id, "pending_request_changed")
                self.assertEqual(model.calls, 0)
                self.assertEqual(parent.children, [])

    async def test_retry_after_may_exceed_backoff_cap_but_not_cumulative_window(self):
        slept = []

        async def wait(seconds):
            slept.append(seconds)

        # A role's wider window: each wait is above the backoff cap; together they
        # exceed the window's wait budget.
        observed, model, parent = _observer(
            [_http_error(503, "400"), _http_error(503, "400")],
            availability_sleep=wait,
            contact_policy=ModelContactPolicy(contacts=3, max_wait_seconds=600),
        )
        with self.assertRaises(ModelAvailabilityReviewPending) as raised:
            await observed.request([], {}, _parameters())
        self.assertEqual(model.calls, 2)
        self.assertEqual(slept, [400])
        self.assertEqual(raised.exception.exhaustion_reason, "retry_after_over_cap")
        self.assertEqual(parent.children[0].facts[-1].physical_contact_count, 2)

    async def test_permanent_errors_and_defects_never_retry(self):
        for error in (
            _http_error(400),
            _http_error(401),
            _http_error(402),
            _http_error(403),
            _http_error(413),
            RuntimeError("bug"),
            ValueError("bug"),
        ):
            with self.subTest(error=type(error).__name__):
                observed, model, parent = _observer([error])
                with self.assertRaises(
                    ModelAvailabilityReviewPending
                    if isinstance(error, ModelHTTPError)
                    else type(error)
                ):
                    await observed.request([], {}, _parameters())
                self.assertEqual(model.calls, 1)
                self.assertNotIn(
                    "ModelAvailabilityRetryScheduled", _event_types(parent)
                )
                self.assertEqual(
                    parent.children[0].facts[-1].availability_exhaustion_reason,
                    "request_rejected" if isinstance(error, ModelHTTPError) else None,
                )

    async def test_ordinary_success_is_one_logical_request_and_one_contact(self):
        response = ModelResponse(parts=[TextPart("done")])
        observed, model, parent = _observer([response])

        returned = await observed.request([], {}, _parameters())

        self.assertIs(returned, response)
        self.assertEqual(model.calls, 1)
        self.assertEqual(
            _event_types(parent),
            [
                "ModelRequestRecorded",
                "ModelPhysicalContactStarted",
                "ModelResponseRecorded",
            ],
        )
        response_fact = parent.children[0].facts[-1]
        started_event = parent.children[0].events[1]
        self.assertEqual(response_fact.physical_contact_ordinal, 1)
        self.assertEqual(
            response_fact.physical_contact_started_event_id,
            started_event["event_id"],
        )

    async def test_a_rate_limited_contact_keeps_its_rate_limit_headers_only(self):
        # A 429 or 5xx keeps the provider's x-ratelimit-* headers on its failed
        # contact, so a live Run shows the account's real limit; no other header,
        # credentials among them, is retained.
        error = _http_error(429, "1")
        error.headers.update(
            {
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "0",
                "X-RateLimit-Limit-Tokens": "400000",
            }
        )

        async def sleep(seconds):
            del seconds

        observed, _model, parent = _observer(
            [error, ModelResponse(parts=[TextPart("done")])],
            availability_sleep=sleep,
        )
        await observed.request([], {}, _parameters())
        (failed,) = [
            event
            for event in parent.children[0].events
            if event["event_type"] == "ModelPhysicalContactFailed"
        ]
        self.assertEqual(
            failed["payload"]["rate_limit_headers"],
            {
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-limit-tokens": "400000",
                "x-ratelimit-remaining-requests": "0",
            },
        )
        plain = _http_error(503, "1")
        observed, _model, parent = _observer(
            [plain, ModelResponse(parts=[TextPart("done")])],
            availability_sleep=sleep,
        )
        await observed.request([], {}, _parameters())
        (failed,) = [
            event
            for event in parent.children[0].events
            if event["event_type"] == "ModelPhysicalContactFailed"
        ]
        self.assertNotIn("rate_limit_headers", failed["payload"])

    async def test_first_contact_failure_then_success_has_exact_correlation_and_no_prompt(
        self,
    ):
        response = ModelResponse(parts=[TextPart("done")])
        # Without a usable Retry-After the one wait is the default 30-second
        # base times the observer's 0.5 jitter.
        cases = (
            (429, "7", 7, "retry_after"),
            (429, None, 15.0, None),
            (429, "not-an-integer", 15.0, None),
            (500, "2", 2, "retry_after"),
            (500, None, 15.0, "exponential_jitter"),
            (500, "not-an-integer", 15.0, "exponential_jitter"),
            (502, "2", 2, "retry_after"),
            (503, "2", 2, "retry_after"),
            (503, None, 15.0, "exponential_jitter"),
            (503, "not-an-integer", 15.0, "exponential_jitter"),
            (503, "0", 0, "retry_after"),
            (503, "-1", 15.0, "exponential_jitter"),
            (503, "1.0", 15.0, "exponential_jitter"),
            (503, " 1", 1, "retry_after"),
            (503, "١", 15.0, "exponential_jitter"),
            (
                503,
                "Wed, 21 Oct 2015 07:28:00 GMT",
                0,
                "retry_after",
            ),
            (504, "2", 2, "retry_after"),
        )
        for status, header, delay, source in cases:
            with self.subTest(status=status, header=header):
                slept = []

                async def sleep(seconds):
                    slept.append(seconds)  # noqa: B023 - Existing immediately awaited test callback; import-only relocation.

                messages = []
                observed, model, parent = _observer(
                    [_http_error(status, header), response],
                    availability_sleep=sleep,
                )
                returned = await observed.request(messages, {}, _parameters())
                events = parent.children[0].events
                facts = parent.children[0].facts

                self.assertIs(returned, response)
                self.assertEqual((model.calls, slept, messages), (2, [delay], []))
                self.assertEqual(
                    _event_types(parent),
                    [
                        "ModelRequestRecorded",
                        "ModelPhysicalContactStarted",
                        "ModelPhysicalContactFailed",
                        "ModelAvailabilityRetryScheduled",
                        "ModelAvailabilityRetryDelayCompleted",
                        "ModelPhysicalContactStarted",
                        "ModelResponseRecorded",
                    ],
                )
                logical_request_id = events[0]["event_id"]
                self.assertTrue(
                    all(
                        fact.logical_request_event_id == logical_request_id
                        for fact in (facts[1], facts[2], facts[3], facts[4], facts[5])
                    )
                )
                self.assertEqual({fact.request_number for fact in facts}, {1})
                self.assertEqual(
                    facts[2].contact_started_event_id, events[1]["event_id"]
                )
                self.assertEqual(
                    facts[3].failed_contact_event_id, events[2]["event_id"]
                )
                if source is not None:
                    self.assertEqual(facts[3].delay_source, source)
                self.assertEqual(
                    facts[4].retry_scheduled_event_id, events[3]["event_id"]
                )
                self.assertEqual(
                    facts[5].retry_delay_completed_event_id,
                    events[4]["event_id"],
                )
                self.assertEqual(
                    facts[6].physical_contact_started_event_id, events[5]["event_id"]
                )
                self.assertFalse(
                    any("RetryPrompt" in kind for kind in _event_types(parent))
                )
                self.assertFalse(
                    any(kind.startswith("Tool") for kind in _event_types(parent))
                )

    async def test_over_cap_retry_after_opens_review_without_second_contact(self):
        review_cases = (
            (429, _OVER_WINDOW, "retry_after_over_cap"),
            (
                429,
                "999999999999999999999999999999999999",
                "retry_after_over_cap",
            ),
            (500, _OVER_WINDOW, "retry_after_over_cap"),
            (502, _OVER_WINDOW, "retry_after_over_cap"),
            (503, _OVER_WINDOW, "retry_after_over_cap"),
            (
                503,
                "999999999999999999999999999999999999",
                "retry_after_over_cap",
            ),
            (504, _OVER_WINDOW, "retry_after_over_cap"),
        )
        for status, header, reason in review_cases:
            with self.subTest(status=status, header=header):
                observed, model, parent = _observer([_http_error(status, header)])
                with self.assertRaises(ModelAvailabilityReviewPending) as raised:
                    await observed.request([], {}, _parameters())
                self.assertEqual(model.calls, 1)
                self.assertEqual(raised.exception.exhaustion_reason, reason)
                self.assertEqual(raised.exception.physical_contact_count, 1)
                self.assertNotIn(
                    "ModelAvailabilityRetryScheduled", _event_types(parent)
                )

    async def test_explicit_two_contact_override_exhausts_its_window(self):
        # A usable Retry-After waits as asked; a completion-unknown fault waits
        # the default 30-second base times the observer's 0.5 jitter.
        cases = (
            (_http_error(502, "1"), _http_error(504, "1"), [502, 504], 1),
            (
                TimeoutError("completion unknown timeout"),
                ConnectionError("completion unknown connection reset"),
                [None, None],
                15.0,
            ),
        )
        for first, second, statuses, delay in cases:
            with self.subTest(statuses=statuses, error_type=type(first).__name__):
                slept = []

                async def sleep(seconds):
                    slept.append(seconds)  # noqa: B023 - Existing immediately awaited test callback; import-only relocation.

                observed, model, parent = _observer(
                    [first, second],
                    availability_sleep=sleep,
                    contact_policy=ModelContactPolicy(contacts=2),
                )
                with self.assertRaises(ModelAvailabilityReviewPending) as raised:
                    await observed.request([], {}, _parameters())

                self.assertEqual((model.calls, slept), (2, [delay]))
                self.assertEqual(
                    raised.exception.exhaustion_reason,
                    "automatic_retry_exhausted",
                )
                self.assertEqual(raised.exception.physical_contact_count, 2)
                kinds = _event_types(parent)
                self.assertEqual(kinds.count("ModelRequestRecorded"), 1)
                self.assertEqual(kinds.count("ModelAvailabilityRetryScheduled"), 1)
                self.assertEqual(kinds.count("ModelRequestFailed"), 1)
                self.assertEqual(kinds.count("ModelPhysicalContactStarted"), 2)
                self.assertEqual(kinds.count("ModelPhysicalContactFailed"), 2)
                failed_contacts = [
                    fact
                    for fact in parent.children[0].facts
                    if fact.event_type == "ModelPhysicalContactFailed"
                ]
                self.assertEqual(
                    [fact.contact_ordinal for fact in failed_contacts], [1, 2]
                )
                self.assertEqual(
                    [fact.http_status_code for fact in failed_contacts],
                    statuses,
                )
                self.assertEqual(
                    [fact.completion_certainty for fact in failed_contacts],
                    ["response_received", "response_received"]
                    if statuses[0] is not None
                    else ["completion_unknown", "completion_unknown"],
                )
                expected_failure_class = (
                    "provider_failure"
                    if statuses[0] is not None
                    else "non_http_failure"
                )
                self.assertEqual(
                    [fact.failure_reason for fact in failed_contacts],
                    [expected_failure_class, expected_failure_class],
                )
                self.assertEqual(parent.children[0].facts[-1].physical_contact_count, 2)
                self.assertEqual(
                    parent.children[0].facts[-1].failure_reason,
                    expected_failure_class,
                )

    async def test_completion_unknown_failure_uses_fallback_then_returns_response(self):
        response = ModelResponse(parts=[TextPart("done")])
        for outcome in (
            TimeoutError("completion unknown timeout"),
            ConnectionError("completion unknown connection reset"),
        ):
            with self.subTest(error_type=type(outcome).__name__):
                slept = []

                async def sleep(seconds):
                    slept.append(seconds)  # noqa: B023 - Existing immediately awaited test callback; import-only relocation.

                observed, model, parent = _observer(
                    [outcome, response], availability_sleep=sleep
                )
                returned = await observed.request([], {}, _parameters())

                self.assertIs(returned, response)
                self.assertEqual((model.calls, slept), (2, [15.0]))
                self.assertEqual(
                    _event_types(parent),
                    [
                        "ModelRequestRecorded",
                        "ModelPhysicalContactStarted",
                        "ModelPhysicalContactFailed",
                        "ModelAvailabilityRetryScheduled",
                        "ModelAvailabilityRetryDelayCompleted",
                        "ModelPhysicalContactStarted",
                        "ModelResponseRecorded",
                    ],
                )
                failed_contact = parent.children[0].facts[2]
                self.assertEqual(failed_contact.error_type, type(outcome).__name__)
                self.assertIsNone(failed_contact.http_status_code)
                self.assertEqual(
                    failed_contact.failure_reason,
                    "non_http_failure",
                )
                self.assertEqual(
                    failed_contact.completion_certainty, "completion_unknown"
                )
                self.assertEqual(
                    failed_contact.policy_disposition,
                    "automatic_retry_scheduled",
                )

    async def test_malformed_response_resends_once_then_waits_for_review(self):
        observed, model, parent = _observer([object(), object()])

        with self.assertRaises(ModelAvailabilityReviewPending) as raised:
            await observed.request([], {}, _parameters())

        self.assertEqual(raised.exception.exhaustion_reason, "malformed_response")
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.sent_messages[0], model.sent_messages[1])
        self.assertEqual(
            _event_types(parent).count("ModelAvailabilityRetryScheduled"), 1
        )
        self.assertEqual(_event_types(parent)[-1], "ModelRequestFailed")
        failed_contact = parent.children[0].facts[-2]
        request_failed = parent.children[0].facts[-1]
        self.assertEqual(failed_contact.completion_certainty, "response_received")
        self.assertEqual(failed_contact.failure_reason, "malformed_response")
        self.assertEqual(
            failed_contact.policy_disposition, "availability_review_requested"
        )
        self.assertEqual(request_failed.failure_reason, "malformed_response")

    async def test_provider_body_headers_url_and_arbitrary_retryable_text_are_not_evidence(
        self,
    ):
        response = ModelResponse(parts=[TextPart("done")])
        observed, model, parent = _observer(
            [_http_error(502, "1"), response],
            availability_sleep=lambda _: asyncio.sleep(0),
        )

        returned = await observed.request([], {}, _parameters())

        self.assertIs(returned, response)
        self.assertEqual(model.calls, 2)

        retained = json.dumps(parent.children[0].events)
        for forbidden in (
            "forbidden-header-secret",
            "forbidden-body-secret",
            "forbidden-query-secret",
            "authorization",
            "x-should-retry",
            "retry-after-ms",
            "retryable",
        ):
            self.assertNotIn(forbidden, retained)

    async def test_cancellation_during_contact_awaits_inner_and_never_contacts_again(
        self,
    ):
        entered = asyncio.Event()
        inner_cancelled = asyncio.Event()
        inner_finished = asyncio.Event()

        async def blocked_contact():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                inner_cancelled.set()
                raise RuntimeError("inner translated cancellation")  # noqa: B904 - Existing cancellation translation scenario.
            finally:
                inner_finished.set()

        observed, model, parent = _observer([blocked_contact])
        task = asyncio.create_task(observed.request([], {}, _parameters()))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(inner_cancelled.is_set())
        self.assertTrue(inner_finished.is_set())
        self.assertEqual(model.calls, 1)
        self.assertEqual(
            _event_types(parent),
            [
                "ModelRequestRecorded",
                "ModelPhysicalContactStarted",
                "ModelPhysicalContactCancelled",
            ],
        )
        self.assertTrue(parent.children[0].closed)

    async def test_cancellation_during_backoff_closes_delay_and_never_contacts_again(
        self,
    ):
        delay_entered = asyncio.Event()
        delay_cancelled = asyncio.Event()

        async def blocked_sleep(seconds):
            self.assertEqual(seconds, 15)
            delay_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                delay_cancelled.set()
                raise

        observed, model, parent = _observer(
            [_http_error(503), ModelResponse(parts=[TextPart("too late")])],
            availability_sleep=blocked_sleep,
        )
        task = asyncio.create_task(observed.request([], {}, _parameters()))
        await delay_entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(delay_cancelled.is_set())
        self.assertEqual(model.calls, 1)
        self.assertEqual(
            _event_types(parent)[-1], "ModelAvailabilityRetryDelayCancelled"
        )
        self.assertNotIn("ModelAvailabilityRetryDelayCompleted", _event_types(parent))
        self.assertTrue(parent.children[0].closed)


class AvailabilityReviewReplayTests(unittest.TestCase):
    def test_projection_binds_exact_request_contact_failure_and_pending_review(self):
        timeline, request_id = _pending_review_timeline()

        projection = project_model_availability_review(timeline)

        self.assertEqual(projection.pending_request.request_event_id, request_id)
        self.assertEqual(
            projection.pending_request.exhaustion_reason, "retry_after_over_cap"
        )
        self.assertEqual(len(projection.pending_request.contacts), 1)
        self.assertEqual(projection.pending_request.contacts[0].ordinal, 1)
        self.assertEqual(projection.pending_request.delays, ())
        retried = project_model_availability_review(
            _pending_review_timeline(two_contacts=True)[0]
        )
        self.assertEqual(
            retried.pending_request.exhaustion_reason,
            "automatic_retry_exhausted",
        )
        self.assertEqual(len(retried.pending_request.contacts), 2)
        five_hundred = project_model_availability_review(
            _pending_review_timeline(two_contacts=True, server_status=500)[0]
        )
        self.assertEqual(
            [contact.status_code for contact in five_hundred.pending_request.contacts],
            [500, 500],
        )
        self.assertEqual(
            five_hundred.pending_request.delays[0].source,
            "exponential_jitter",
        )

    def test_hostile_missing_duplicate_reorder_mismatch_and_role_drift_are_rejected(
        self,
    ):
        timeline, _ = _pending_review_timeline()
        cases = {}
        cases["missing contact"] = [
            event
            for event in copy.deepcopy(timeline)
            if event["event_type"] != "ModelPhysicalContactStarted"
        ]
        duplicate = copy.deepcopy(timeline)
        duplicate.append(copy.deepcopy(duplicate[-1]))
        cases["duplicate identity"] = duplicate
        reordered = copy.deepcopy(timeline)
        start = next(
            i
            for i, event in enumerate(reordered)
            if event["event_type"] == "ModelPhysicalContactStarted"
        )
        failure = next(
            i
            for i, event in enumerate(reordered)
            if event["event_type"] == "ModelPhysicalContactFailed"
        )
        reordered[start], reordered[failure] = reordered[failure], reordered[start]
        cases["reordered contact"] = reordered
        mismatched = copy.deepcopy(timeline)
        next(
            event
            for event in mismatched
            if event["event_type"] == "ModelPhysicalContactFailed"
        )["payload"]["logical_request_event_id"] = _uuid("hostile-request")
        cases["mismatched request"] = mismatched
        mismatched_start = copy.deepcopy(timeline)
        next(
            event
            for event in mismatched_start
            if event["event_type"] == "ModelPhysicalContactFailed"
        )["payload"]["contact_started_event_id"] = _uuid("hostile-contact")
        cases["mismatched contact start"] = mismatched_start
        wrong_role = copy.deepcopy(timeline)
        next(
            event
            for event in wrong_role
            if event["event_type"] == "ModelAvailabilityReviewRequested"
        )["payload"]["agent_role"] = "investigation_planner"
        cases["wrong role"] = wrong_role

        for name, hostile in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                project_model_availability_review(hostile)

    def test_hostile_logical_diagnostics_exhaustion_and_delay_are_rejected(self):
        one_contact, _ = _pending_review_timeline()
        relabeled_header = copy.deepcopy(one_contact)
        next(
            event
            for event in relabeled_header
            if event["event_type"] == "ModelRequestFailed"
        )["payload"]["availability_exhaustion_reason"] = "automatic_retry_exhausted"
        next(
            event
            for event in relabeled_header
            if event["event_type"] == "ModelAvailabilityReviewRequested"
        )["payload"]["exhaustion_reason"] = "automatic_retry_exhausted"

        changed_diagnostics = copy.deepcopy(one_contact)
        next(
            event
            for event in changed_diagnostics
            if event["event_type"] == "ModelRequestFailed"
        )["payload"]["failure_reason"] = "provider_failure"

        changed_physical_error = copy.deepcopy(one_contact)
        next(
            event
            for event in changed_physical_error
            if event["event_type"] == "ModelPhysicalContactFailed"
        )["payload"]["error_type"] = "TimeoutError"

        detached_physical_failure = copy.deepcopy(one_contact)
        detached_request = next(
            event
            for event in detached_physical_failure
            if event["event_type"] == "ModelRequestRecorded"
        )
        next(
            event
            for event in detached_physical_failure
            if event["event_type"] == "ModelPhysicalContactFailed"
        )["payload"]["contact_started_event_id"] = detached_request["event_id"]

        detached_logical_failure = copy.deepcopy(one_contact)
        detached_logical_request = next(
            event
            for event in detached_logical_failure
            if event["event_type"] == "ModelRequestRecorded"
        )
        next(
            event
            for event in detached_logical_failure
            if event["event_type"] == "ModelRequestFailed"
        )["payload"]["last_physical_contact_event_id"] = detached_logical_request[
            "event_id"
        ]

        two_contacts, _ = _pending_review_timeline(two_contacts=True)
        relabeled_retry = copy.deepcopy(two_contacts)
        next(
            event
            for event in relabeled_retry
            if event["event_type"] == "ModelRequestFailed"
        )["payload"]["availability_exhaustion_reason"] = "retry_after_missing"
        next(
            event
            for event in relabeled_retry
            if event["event_type"] == "ModelAvailabilityReviewRequested"
        )["payload"]["exhaustion_reason"] = "retry_after_missing"

        changed_delay = copy.deepcopy(two_contacts)
        next(
            event
            for event in changed_delay
            if event["event_type"] == "ModelAvailabilityRetryDelayCompleted"
        )["payload"]["delay_seconds"] = 2
        detached_delay = copy.deepcopy(two_contacts)
        detached_request = next(
            event
            for event in detached_delay
            if event["event_type"] == "ModelRequestRecorded"
        )
        next(
            event
            for event in detached_delay
            if event["event_type"] == "ModelAvailabilityRetryDelayCompleted"
        )["payload"]["retry_scheduled_event_id"] = detached_request["event_id"]

        for name, hostile in {
            "one-contact over-cap relabeled retry exhausted": relabeled_header,
            "logical diagnostics changed": changed_diagnostics,
            "physical HTTP error type changed": changed_physical_error,
            "physical failure detached from start": detached_physical_failure,
            "logical failure detached from final contact": detached_logical_failure,
            "two-contact exhaustion relabeled missing": relabeled_retry,
            "completed delay changed": changed_delay,
            "completed delay detached": detached_delay,
        }.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                project_model_availability_review(hostile)

    def test_response_is_idempotent_but_conflict_stale_and_mismatch_are_refused(self):
        timeline, request_id = _pending_review_timeline()
        response_id = _uuid("review-response")
        timeline.append(
            {
                "event_id": response_id,
                "event_type": "ModelAvailabilityReviewResponded",
                "parent_ids": [request_id, timeline[-1]["event_id"]],
                "payload": {
                    "request_event_id": request_id,
                    "response": {"action": "retry"},
                    "authority": "researcher_direction",
                    "scientifically_verified": False,
                    "authorizes_execution": False,
                },
            }
        )
        projection = project_model_availability_review(timeline)
        retained = evaluate_model_availability_review_response(
            projection, request_id, RetryModelAvailabilityResponse(action="retry")
        )
        self.assertEqual(retained.response_event_id, response_id)

        with self.assertRaises(ModelAvailabilityReviewResponseRefused) as conflict:
            evaluate_model_availability_review_response(
                projection,
                request_id,
                StopModelAvailabilityResponse(action="stop", reason="stop"),
            )
        self.assertEqual(
            conflict.exception.code, "model_availability_review_response_conflict"
        )
        self.assertEqual(
            conflict.exception.disposition.kind,
            "caller_correction_and_reissue",
        )

        with self.assertRaises(ModelAvailabilityReviewResponseRefused) as stale:
            evaluate_model_availability_review_response(
                projection,
                _uuid("unrelated-review"),
                RetryModelAvailabilityResponse(action="retry"),
            )
        self.assertEqual(
            stale.exception.code, "model_availability_review_response_stale"
        )
        self.assertEqual(
            stale.exception.disposition.kind,
            "caller_correction_and_reissue",
        )

        mismatched = copy.deepcopy(timeline)
        mismatched[-1]["payload"]["request_event_id"] = _uuid("wrong-review")
        with self.assertRaises(ValueError):
            project_model_availability_review(mismatched)

    def test_reason_bearing_stop_projects_terminal_direction(self):
        timeline, request_id = _pending_review_timeline()
        response_id = _uuid("reject-response")
        timeline.extend(
            [
                {
                    "event_id": response_id,
                    "event_type": "ModelAvailabilityReviewResponded",
                    "parent_ids": [request_id, timeline[-1]["event_id"]],
                    "payload": {
                        "request_event_id": request_id,
                        "response": {
                            "action": "stop",
                            "reason": "stop contacting the model",
                        },
                        "authority": "researcher_direction",
                        "scientifically_verified": False,
                        "authorizes_execution": False,
                    },
                },
                {
                    "event_id": _uuid("rejected"),
                    "event_type": "ModelAvailabilityStopped",
                    "parent_ids": [response_id],
                    "payload": {
                        "request_event_id": request_id,
                        "response_event_id": response_id,
                        "reason": "stop contacting the model",
                    },
                },
            ]
        )

        projection = project_model_availability_review(timeline)

        self.assertIsNone(projection.pending_request)
        self.assertEqual(projection.stop_event_id, _uuid("rejected"))
        self.assertEqual(
            projection.receipts[0].response.reason, "stop contacting the model"
        )

    def test_the_folds_stop_id_is_left_out_only_for_a_role_settling_its_member(
        self,
    ):
        """#485 Q4 a (the #383 r3 rule, ``ROLES[role].settles_member_stops``): the
        availability fold names a stop answer's ``ModelAvailabilityStopped`` as its
        ``stop_event_id`` for every role but one whose declaration settles its
        stopped member, the Paper Reader's reading item. This pins the fold only;
        what a stop then does is each caller's (#485 r1 SP1): a reading item settles
        alone (``test_paper_planning.MemberScopedStopTests``), an explainer member
        settles ``not_explained`` (``test_report_explanation...test_an_unavailable_
        model_stopped_settles_only_its_member_not_explained``), and the
        Investigation's answer-all stop answers every member
        (``test_investigation_questions...test_one_model_stop_ends_every_member_
        with_the_reviews_cause``)."""
        from bio_neb.agentic_journey.role_declaration import ROLES

        def stop_event(role, member_id):
            timeline, request_id = _pending_review_timeline()
            for event in timeline:
                if event["payload"].get("agent_role") == "planning_agent":
                    event["payload"]["agent_role"] = role
                    if event["event_type"] == "AttemptStarted" and member_id:
                        event["payload"]["member_id"] = member_id
            response_id = _uuid("stop-response")
            timeline += [
                {
                    "event_id": response_id,
                    "event_type": "ModelAvailabilityReviewResponded",
                    "parent_ids": [request_id, timeline[-1]["event_id"]],
                    "payload": {
                        "request_event_id": request_id,
                        "response": {"action": "stop", "reason": "Skip it."},
                        "authority": "researcher_direction",
                        "scientifically_verified": False,
                        "authorizes_execution": False,
                    },
                },
                {
                    "event_id": _uuid("stopped"),
                    "event_type": "ModelAvailabilityStopped",
                    "parent_ids": [response_id],
                    "payload": {
                        "request_event_id": request_id,
                        "response_event_id": response_id,
                        "reason": "Skip it.",
                    },
                },
            ]
            return project_model_availability_review(timeline).stop_event_id

        cases = {
            "planning_agent": (None, _uuid("stopped")),
            "investigation_planner": ("request-1", _uuid("stopped")),
            "report_explainer": ("request-1", _uuid("stopped")),
            "paper_reader": ("paper-reading:reading-1:area-1", None),
        }
        self.assertEqual(
            {role for role, (member, _) in cases.items() if member},
            {name for name, item in ROLES.items() if item.member_scope == "member"},
        )
        for role, (member_id, expected) in cases.items():
            with self.subTest(role=role):
                self.assertEqual(stop_event(role, member_id), expected)
                self.assertEqual(expected is None, ROLES[role].settles_member_stops)


class AvailabilityReviewCommandPreflightTests(unittest.TestCase):
    """A retry the command cannot route is refused before any Journey write.

    Each case stops a real planning Attempt at its availability Review (a
    rejected request needs no automatic retry), optionally rewrites the retained
    lines canonically, and compares the Journey bytes around the refusal.
    """

    def _pending(self, rewrite=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        receipt, _ = _registered(Path(temporary.name).resolve())
        journey = Journey.open(Path(receipt.journey_file))
        prepare_input_correction(journey)

        async def unavailable(messages, info):
            del messages, info
            raise ModelHTTPError(401, "offline")

        stopped = planning_assignment(
            model_factory=ScriptedModelEndpoint(unavailable)
        ).run(journey, environ={"LOCAL_MODEL_KEY": "secret"})
        self.assertEqual(stopped.status, "model_availability_review_pending")
        request_id = stopped.model_availability_review.pending_request.request_event_id
        path = Path(receipt.journey_file)
        if rewrite is not None:
            events = [json.loads(line) for line in path.read_text().splitlines()]
            rewrite(events, request_id)
            path.write_text("".join(canonical_json(event) + "\n" for event in events))
        command = RespondModelAvailabilityReviewCommand(
            journey_file=str(path),
            request_event_id=request_id,
            response=RetryModelAvailabilityResponse(action="retry"),
        )
        return path, command

    def _assert_refused_before_any_write(self, rewrite, agent_role, error, message):
        path, command = self._pending(rewrite)
        before = path.read_bytes()
        attempt = Mock(agent_role=agent_role)
        with self.assertRaisesRegex(error, message):
            respond_model_availability_review(command, environ={}, attempt=attempt)
        attempt.run.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_retry_by_work_of_another_assignment_refuses_before_any_write(self):
        self._assert_refused_before_any_write(
            None,
            "investigation_planner",
            AgentReentryRefused,
            "no Agent work re-enters the recorded assignment 'planning_agent'",
        )

    def test_retry_frontier_not_the_stopped_attempts_start_refuses_before_any_write(
        self,
    ):
        # The checked request binds its retry frontier to the stopped Attempt's
        # own start, so a retry needs no per-role frontier status table.
        def move_frontier(events, request_id):
            started = next(e for e in events if e["event_type"] == "AttemptStarted")
            request = next(e for e in events if e["event_id"] == request_id)
            self.assertNotEqual(
                request["payload"]["retry_frontier_event_id"], started["event_id"]
            )
            request["payload"]["retry_frontier_event_id"] = started["event_id"]

        self._assert_refused_before_any_write(
            move_frontier,
            "planning_agent",
            ValueError,
            "availability request changed its logical request",
        )

    def test_review_owned_by_a_retired_role_is_refused_before_any_write(self):
        def retire_role(events, request_id):
            del request_id
            for event in events:
                if event["event_type"] in {
                    "AttemptStarted",
                    "ModelAvailabilityReviewRequested",
                }:
                    event["payload"]["agent_role"] = "execution_planning_agent"

        self._assert_refused_before_any_write(
            retire_role,
            "execution_planning_agent",
            ValueError,
            r"agent_role\n.*input_value='execution_planning_agent'",
        )


if __name__ == "__main__":
    unittest.main()
