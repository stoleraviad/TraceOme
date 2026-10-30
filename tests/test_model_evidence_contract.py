import unittest
import uuid
from dataclasses import asdict, replace

from pydantic import ValidationError
from pydantic_ai.messages import InstructionPart, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from bio_neb.agentic_journey.attempt_journey import (
    AttemptFailureSource,
    ModelCompletionTruncated,
    ModelRequestFailed,
    ModelRequestRecorded,
    ModelResponseRecorded,
    NebiusModelEndpointEvidence,
    validate_attempt_fact,
)
from bio_neb.agentic_journey.model_observation import (
    JourneyObservedModel,
    ReceivedReply,
)
from bio_neb.failure import ModelContactPolicy

ATTEMPT_ID = "00000000-0000-4000-8000-000000000001"
ENDPOINT = {
    "provider": "nebius",
    "model": "Qwen/Qwen3.5-397B-A17B",
    "contacted_url": "https://api.tokenfactory.nebius.com/v1",
}


def compatible_request():
    return {
        "model_name": ENDPOINT["model"],
        "model_system": "openai",
        "request": {
            "messages": [{"kind": "request", "parts": []}],
            "model_settings": {"parallel_tool_calls": False},
            "function_tools": [
                {
                    "name": "resolve_taxon",
                    "parameters_json_schema": {"type": "object"},
                }
            ],
            "output_tools": [],
            "allow_text_output": False,
            "output_mode": "tool",
            "instruction_parts": None,
        },
    }


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
        self.facts.append(fact)
        event = {"event_id": str(uuid.uuid4()), "event_type": fact.event_type}
        self.events.append(event)
        return event

    def record_contact(self, fact):
        # Mechanical contact evidence; the fake keeps one ordered fact list.
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


class _PreparingModel(Model):
    def __init__(self, response: ModelResponse | None = None):
        super().__init__()
        self.actual_request = None
        self.response = (
            response
            if response is not None
            else ModelResponse(parts=[TextPart("done")])
        )

    @property
    def model_name(self):
        return ENDPOINT["model"]

    @property
    def system(self):
        return "openai"

    def prepare_request(self, model_settings, model_request_parameters):
        return (
            {
                **(model_settings or {}),
                "parallel_tool_calls": False,
            },
            replace(
                model_request_parameters,
                instruction_parts=[
                    *(model_request_parameters.instruction_parts or []),
                    InstructionPart(content="Adapter output guidance."),
                ],
            ),
        )

    async def request(self, messages, model_settings, model_request_parameters):
        self.actual_request = (messages, model_settings, model_request_parameters)
        self.effective_settings, self.effective_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        return self.response


class ModelEvidenceContractTests(unittest.IsolatedAsyncioTestCase):
    def test_product_identity_is_exact_and_secret_free(self):
        self.assertEqual(
            NebiusModelEndpointEvidence.model_validate(ENDPOINT).model_dump(
                mode="json"
            ),
            ENDPOINT,
        )
        for contacted_url in (
            "https://api.tokenfactory.nebius.com",
            "https://api.tokenfactory.nebius.com/v1/",
            "https://user:secret@api.tokenfactory.nebius.com/v1",
            "https://api.tokenfactory.nebius.com/v1?token=secret",
            "https://api.tokenfactory.nebius.com/v1#fragment",
            "https://api.tokenfactory.nebius.com/v1/chat/completions",
        ):
            with (
                self.subTest(contacted_url=contacted_url),
                self.assertRaises(ValidationError),
            ):
                NebiusModelEndpointEvidence.model_validate(
                    {**ENDPOINT, "contacted_url": contacted_url}
                )

    def test_compatible_normalization_is_nested_without_old_top_level_reader(self):
        request = ModelRequestRecorded(
            attempt_id=ATTEMPT_ID,
            request_number=1,
            nebius_model_endpoint=ENDPOINT,
            compatible_adapter_evidence=compatible_request(),
        )
        dumped = request.model_dump(mode="json")
        self.assertEqual(dumped["nebius_model_endpoint"], ENDPOINT)
        self.assertEqual(
            dumped["compatible_adapter_evidence"]["model_system"], "openai"
        )
        self.assertNotIn("request", dumped)
        self.assertNotIn("model_name", dumped)
        with self.assertRaises(ValidationError):
            ModelRequestRecorded.model_validate(
                {
                    "attempt_id": ATTEMPT_ID,
                    "request_number": 1,
                    "model_name": ENDPOINT["model"],
                    "model_system": "openai",
                    "request": compatible_request()["request"],
                }
            )

    def test_response_replays_when_optional_provider_fields_are_absent(self):
        response = ModelResponseRecorded(
            attempt_id=ATTEMPT_ID,
            request_number=1,
            physical_contact_ordinal=1,
            physical_contact_started_event_id="00000000-0000-4000-8000-000000000002",
            completion_certainty="response_received",
            policy_disposition="response_admitted",
            nebius_model_endpoint=ENDPOINT,
            compatible_adapter_evidence={
                "model_name": ENDPOINT["model"],
                "model_system": "openai",
                "response": {
                    "message": {
                        "kind": "response",
                        "parts": [
                            {
                                "part_kind": "tool-call",
                                "tool_name": "resolve_taxon",
                                "tool_call_id": "call-1",
                                "args": {"query": "human"},
                            }
                        ],
                    }
                },
            },
        )
        event = {
            "event_type": response.event_type,
            "payload": response.model_dump(mode="json", exclude={"event_type"}),
        }
        replayed = validate_attempt_fact(event)
        self.assertEqual(replayed, response)
        retained = response.compatible_adapter_evidence.response.message
        for optional in (
            "provider_response_id",
            "provider_name",
            "provider_url",
            "finish_reason",
            "usage",
            "provider_details",
        ):
            self.assertNotIn(optional, retained)
        call = retained["parts"][0]
        self.assertEqual(
            (call["tool_name"], call["tool_call_id"], call["args"]),
            ("resolve_taxon", "call-1", {"query": "human"}),
        )

    def test_http_and_legacy_non_http_failures_replay_with_safe_shapes(self):
        failure = ModelRequestFailed(
            attempt_id=ATTEMPT_ID,
            request_number=2,
            physical_contact_count=1,
            last_physical_contact_event_id="00000000-0000-4000-8000-000000000003",
            nebius_model_endpoint=ENDPOINT,
            error_type="ModelHTTPError",
            http_status_code=429,
            failure_reason="rate_limited",
        )
        dumped = failure.model_dump(mode="json")
        self.assertEqual(
            set(dumped),
            {
                "event_type",
                "attempt_id",
                "request_number",
                "physical_contact_count",
                "last_physical_contact_event_id",
                "nebius_model_endpoint",
                "error_type",
                "http_status_code",
                "failure_reason",
            },
        )
        replayed = validate_attempt_fact(
            {
                "event_type": failure.event_type,
                "payload": failure.model_dump(mode="json", exclude={"event_type"}),
            }
        )
        self.assertEqual(replayed, failure)

        legacy_payload = {
            "attempt_id": "00000000-0000-4000-8000-000000000001",
            "request_number": 1,
            "physical_contact_count": 1,
            "last_physical_contact_event_id": "00000000-0000-4000-8000-000000000002",
            "nebius_model_endpoint": {
                "provider": "nebius",
                "model": "selected/model",
                "contacted_url": "https://api.example.test/v1",
            },
            "error_type": "RuntimeError",
        }
        legacy = ModelRequestFailed(**legacy_payload)
        expected = {"event_type": "ModelRequestFailed", **legacy_payload}
        self.assertEqual(legacy.model_dump(mode="json"), expected)
        replayed_legacy = validate_attempt_fact(
            {"event_type": "ModelRequestFailed", "payload": legacy_payload}
        )
        self.assertIsNotNone(replayed_legacy)
        self.assertEqual(replayed_legacy.model_dump(mode="json"), expected)

    async def test_observer_records_prepared_policy_without_double_preparing_dispatch(
        self,
    ):
        from bio_neb.agentic_journey.model_settings import model_endpoint_policy

        selection = model_endpoint_policy(ENDPOINT["model"]).settings_selection
        wrapped = _PreparingModel()
        parent = _RecordedActivity()
        observed = JourneyObservedModel(
            wrapped,
            contact_policy=ModelContactPolicy(),
            parent_activity=parent,
            attempt_id=ATTEMPT_ID,
            failure_source=AttemptFailureSource(),
            nebius_model_endpoint=ENDPOINT,
            settings_selection=selection,
        )
        messages = []
        original_settings = {"max_tokens": 245760}
        parameters = ModelRequestParameters(
            function_tools=[],
            output_tools=[],
            allow_text_output=True,
            output_mode="text",
            instruction_parts=[
                InstructionPart(content="Agent instruction.", dynamic=True)
            ],
        )

        await observed.request(messages, original_settings, parameters)

        request_fact = parent.children[0].facts[0]
        self.assertEqual(request_fact.settings_selection, selection)
        self.assertEqual(
            request_fact.settings_selection.context_budget.input_reserve, 16384
        )
        self.assertEqual(
            request_fact.settings_selection.context_budget.output_ceiling, 245760
        )
        retained_settings = (
            request_fact.compatible_adapter_evidence.request.model_settings
        )
        self.assertEqual(
            retained_settings,
            {"max_tokens": 245760, "parallel_tool_calls": False},
        )
        # Observation preparation does not replace the provider's dispatch inputs.
        self.assertIs(wrapped.actual_request[0], messages)
        self.assertIs(wrapped.actual_request[1], original_settings)
        self.assertIs(wrapped.actual_request[2], parameters)
        expected_parts = [
            {
                "content": "Agent instruction.",
                "dynamic": True,
                "part_kind": "instruction",
            },
            {
                "content": "Adapter output guidance.",
                "dynamic": False,
                "part_kind": "instruction",
            },
        ]
        self.assertEqual(
            [asdict(part) for part in wrapped.effective_parameters.instruction_parts],
            expected_parts,
        )
        self.assertEqual(
            request_fact.compatible_adapter_evidence.request.instruction_parts,
            expected_parts,
        )
        self.assertEqual(wrapped.effective_settings, retained_settings)

    async def test_observer_retains_but_does_not_admit_function_actions(self):
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    "wrong_tool",
                    {"input_set_id": "registered-inputs"},
                    tool_call_id="call-1",
                )
            ]
        )
        wrapped = _PreparingModel(response)
        parent = _RecordedActivity()
        failure_source = AttemptFailureSource()
        observed = JourneyObservedModel(
            wrapped,
            contact_policy=ModelContactPolicy(),
            parent_activity=parent,
            attempt_id=ATTEMPT_ID,
            failure_source=failure_source,
            nebius_model_endpoint=ENDPOINT,
        )
        parameters = ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="inspect_input_metadata",
                    parameters_json_schema={"type": "object"},
                )
            ],
            output_tools=[],
            allow_text_output=False,
            output_mode="tool",
        )

        returned = await observed.request([], {}, parameters)

        self.assertIs(returned, response)
        self.assertEqual(
            [fact.event_type for fact in parent.children[0].facts],
            [
                "ModelRequestRecorded",
                "ModelPhysicalContactStarted",
                "ModelResponseRecorded",
            ],
        )
        self.assertEqual(parent.facts, [])
        self.assertIsNone(failure_source.event_id)

    async def test_ceiling_stop_reaches_admission_with_its_request_facts(self):
        # A ceiling stop leaves a partial call behind. The observer records the
        # reply as received and hands it on; admission reads every reply, a
        # cut-off one included, and records its truncation (ADR 0044).
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    "inspect_input_metadata",
                    '{"input_set_id": "registered-inp',
                    tool_call_id="call-1",
                )
            ],
            usage=RequestUsage(input_tokens=11701, output_tokens=245760),
            finish_reason="length",
        )
        wrapped = _PreparingModel(response)
        parent = _RecordedActivity()
        failure_source = AttemptFailureSource()
        observed = JourneyObservedModel(
            wrapped,
            contact_policy=ModelContactPolicy(),
            parent_activity=parent,
            attempt_id=ATTEMPT_ID,
            failure_source=failure_source,
            nebius_model_endpoint=ENDPOINT,
        )
        parameters = ModelRequestParameters(
            function_tools=[],
            output_tools=[
                ToolDefinition(
                    name="inspect_input_metadata",
                    parameters_json_schema={"type": "object"},
                )
            ],
            allow_text_output=False,
            output_mode="tool",
        )

        returned = await observed.request([], {"max_tokens": 245760}, parameters)

        self.assertIs(returned, response)
        self.assertEqual(parent.facts, [])
        self.assertIsNone(failure_source.event_id)
        response_event = parent.children[0].events[-1]
        self.assertEqual(response_event["event_type"], "ModelResponseRecorded")
        self.assertEqual(
            observed.last_reply,
            ReceivedReply(
                response_event_id=response_event["event_id"],
                request_number=1,
                requested_max_tokens=245760,
                resends=0,
            ),
        )

    async def test_completion_above_the_retired_pin_is_admitted_unchanged(self):
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    "inspect_input_metadata",
                    {"input_set_id": "registered-inputs"},
                    tool_call_id="call-1",
                )
            ],
            usage=RequestUsage(input_tokens=11701, output_tokens=50611),
            finish_reason="stop",
        )
        wrapped = _PreparingModel(response)
        parent = _RecordedActivity()
        failure_source = AttemptFailureSource()
        observed = JourneyObservedModel(
            wrapped,
            contact_policy=ModelContactPolicy(),
            parent_activity=parent,
            attempt_id=ATTEMPT_ID,
            failure_source=failure_source,
            nebius_model_endpoint=ENDPOINT,
        )
        parameters = ModelRequestParameters(
            function_tools=[],
            output_tools=[
                ToolDefinition(
                    name="inspect_input_metadata",
                    parameters_json_schema={"type": "object"},
                )
            ],
            allow_text_output=False,
            output_mode="tool",
        )

        returned = await observed.request([], {"max_tokens": 245760}, parameters)

        self.assertIs(returned, response)
        self.assertEqual(parent.facts, [])
        self.assertIsNone(failure_source.event_id)

    def test_truncation_replays_as_its_own_retained_fact(self):
        truncation = ModelCompletionTruncated(
            attempt_id=ATTEMPT_ID,
            request_number=2,
            response_event_id="00000000-0000-4000-8000-000000000002",
            requested_max_tokens=245760,
            completion_tokens=245760,
        )
        event = {
            "event_type": truncation.event_type,
            "payload": truncation.model_dump(mode="json", exclude={"event_type"}),
        }

        replayed = validate_attempt_fact(event)

        self.assertEqual(replayed, truncation)
        with self.assertRaises(ValidationError):
            ModelCompletionTruncated(
                attempt_id=ATTEMPT_ID,
                request_number=2,
                response_event_id="00000000-0000-4000-8000-000000000002",
                finish_reason="stop",
            )


if __name__ == "__main__":
    unittest.main()
