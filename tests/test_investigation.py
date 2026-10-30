from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
import urllib.error
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from bio_neb.agentic_journey.agent_work_review import (
    ContinueAgentWorkResponse,
    project_agent_work_review,
    respond_agent_work_review,
)
from bio_neb.agentic_journey.continuation_policy import ContinuationPolicy
from bio_neb.agentic_journey.model_settings import model_endpoint_policy
from bio_neb.agentic_journey.runtime_config import (
    LoadedRuntimeConfig,
    RuntimeConfiguration,
)
from bio_neb.agentic_journey.structured_output import StructuredOutputLineage
from bio_neb.application.advancement import use_run_advancement_dependencies
from bio_neb.application.queries import ObserveRun, observe, read_investigation
from bio_neb.cli import main
from bio_neb.content import canonical_json_digest, sha256_bytes
from bio_neb.failure import CorrectionWording, Invariant, ModelContextExhausted
from bio_neb.investigation.investigation import (
    AddInvestigationMember,
    AdjustInvestigationSearch,
    AmendInvestigationMember,
    ClarifyInvestigationMember,
    DeferInvestigationMember,
    InvestigationInput,
    InvestigationPlanDraft,
    InvestigationProposal,
    InvestigationRequestItem,
    InvestigationStarted,
    InvestigationVcfIdentity,
    KeepInvestigationMember,
    RemoveInvestigationMember,
    RetrievedSource,
    ReviseInvestigationResponse,
    UserAssertion,
    compile_investigation_proposal,
    investigation_output_member_id,
    project_investigation,
)
from bio_neb.investigation.investigation_agent import (
    InvestigationPlannerDependencies,
    assemble_investigation_planner,
)
from bio_neb.investigation.investigation_assignment import investigation_assignment
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
    InvestigationApplicationRefused,
    RespondInvestigationReviewCommand,
    respond_investigation_review,
)
from bio_neb.investigation.investigation_sources import (
    normalize_assembly_report,
    normalize_refsnp,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.providers.ncbi_datasets import MetadataResponse
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceResponse,
    NcbiInvestigationSourceAdapter,
)
from bio_neb.providers.nebius_model_endpoint import NebiusModelEndpointAdapter
from bio_neb.recording import processing_pass, read_evidence
from bio_neb.reference_discovery.reference_identity_tool import ReferenceAdapterError
from tests.content_identity_vectors import assert_json_identity, expected_digest
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.observation_input_fakes import RUN_VCF, TABLE_READING_ENVIRON, RunOutputStore
from tests.proposal_information_fakes import runtime_configuration
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    SourceResponsesOnly as _SourceResponsesOnly,
)
from tests.sequential_investigation_support import (
    _AdaptiveInvestigationModel,
    _IndependentCorrectionsModel,
    _MalformedOutputModel,
    _RepeatedActionModel,
    begin_in_pass,
    collection_request,
    forked_child,
    members_one_at_a_time,
    source_basis,
)
from tests.sequential_investigation_support import (
    input_and_assertion as _input_and_assertion,
)
from tests.sequential_investigation_support import (
    run_attempt as _run_attempt,
)
from tests.vcf_lookup_fakes import TableReaderModel

# The B023 suppressions below mark pre-existing loop-variable captures: every flagged
# closure is built and consumed inside the same iteration, so none of them is a live defect,
# and binding 35 of them by default argument would change signatures that the scripted model
# and the review fakes inspect.
REFSNP_BODY = json.dumps(
    {
        "refsnp_id": "429358",
        "primary_snapshot_data": {
            "placements_with_allele": [
                {
                    "seq_id": "NC_000019.10",
                    "is_ptlp": True,
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [
                            {
                                "assembly_accession": "GCF_000001405.40",
                                "assembly_name": "GRCh38.p14",
                            }
                        ]
                    },
                    "alleles": [
                        {
                            "allele": {
                                "spdi": {
                                    "seq_id": "NC_000019.10",
                                    "position": 44908683,
                                    "deleted_sequence": "T",
                                    "inserted_sequence": "T",
                                }
                            }
                        },
                        {
                            "allele": {
                                "spdi": {
                                    "seq_id": "NC_000019.10",
                                    "position": 44908683,
                                    "deleted_sequence": "T",
                                    "inserted_sequence": "C",
                                }
                            }
                        },
                    ],
                }
            ]
        },
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()

ASSEMBLY_BODY = json.dumps(
    {
        "reports": [
            {
                "accession": "GCF_000001405.40",
                "assembly_info": {
                    "assembly_name": "GRCh38.p14",
                    "assembly_status": "current",
                },
            }
        ]
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()


def _plain_view(tool_name):
    """A registered Tool's view replaced by a plain one, for the generic Tool session
    tests whose stage result is synthetic (#478 G1.3)."""
    from unittest.mock import patch

    from bio_neb.agentic_journey import tool_execution

    return patch.dict(
        tool_execution._model_views(),
        {
            tool_name: tool_execution.ModelView(
                next_step={"*": "Read it."},
                status=lambda _payload: "recorded",
                result=lambda payload: payload["result"],
            )
        },
    )


class _LateRepairModel(_AdaptiveInvestigationModel):
    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        if self.calls == 3:
            arguments = response.parts[0].args_as_dict()
            arguments["one_based_position"] = 1
            arguments["contig"] = "NC_000019405.10"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        canonical_json(arguments),
                        tool_call_id="late-invalid",
                    )
                ]
            )
        return response


class _TextOnlyModel:
    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        self.calls += 1
        return ModelResponse(
            parts=[TextPart(f"Still considering the request {self.calls}")]
        )


class _LiveDuplicateFinalOutputModel(_AdaptiveInvestigationModel):
    """Replay the live model's duplicate final output unless explicitly constrained."""

    def __init__(
        self,
        *,
        honor_exactly_one_instruction: bool = True,
        invalid_before_duplicate: bool = False,
    ) -> None:
        super().__init__()
        self.honor_exactly_one_instruction = honor_exactly_one_instruction
        self.invalid_before_duplicate = invalid_before_duplicate
        self.final_output_call_counts: list[int] = []

    async def respond(self, messages, info):
        response = await super().respond(messages, info)
        if self.calls < 3:
            return response

        if self.invalid_before_duplicate and self.calls == 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        # A compatible pair marked unresolved is refused.
                        {**source_basis(info), "outcome": "unresolved"},
                        tool_call_id="invalid-unresolved-identity",
                    )
                ]
            )

        visible = self.visible_messages[-1].lower()
        exactly_one_final_output = (
            "exactly one" in visible
            and "compile_proposal" in visible
            and "final" in visible
            and "call" in visible
        )
        if exactly_one_final_output and self.honor_exactly_one_instruction:
            self.final_output_call_counts.append(1)
            return response

        complete = {"outcome": "complete", **source_basis(info)}
        # Two distinct outputs: an identical copy would collapse (ADR 0044).
        first = {**complete, "outcome": "unresolved"}
        self.final_output_call_counts.append(2)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    canonical_json(first),
                    tool_call_id="live-duplicate-final-1",
                ),
                ToolCallPart(
                    info.output_tools[0].name,
                    canonical_json(complete),
                    tool_call_id="live-duplicate-final-2",
                ),
            ]
        )


class _RefusedThenAdaptiveInvestigationModel(_AdaptiveInvestigationModel):
    async def respond(self, messages, info):
        if self.calls == 0:
            self.visible_messages.append(
                ModelMessagesTypeAdapter.dump_json(messages).decode()
            )
            self.calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_refsnp",
                        canonical_json({"rsid": "not-an-rsid"}),
                        tool_call_id="refused-1",
                    )
                ]
            )
        return await super().respond(messages, info)


class _RepeatedAssemblyActionModel:
    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        del messages
        self.calls += 1
        name = "fetch_refsnp" if self.calls == 1 else "fetch_assembly_report"
        arguments = (
            {"rsid": "rs429358"}
            if name == "fetch_refsnp"
            else {
                "assembly_accession": "GCF_000001405.40",
                "refsnp_event_id": source_basis(info)["refsnp_event_id"]
                if self.calls == 2
                else "missing",
            }
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name,
                    canonical_json(arguments),
                    tool_call_id=f"repeated-assembly-{self.calls}",
                )
            ]
        )


class _RecordingMetadataTransport:
    def __init__(self) -> None:
        self.contact_count = 0
        self.requests: list[tuple[str, int]] = []

    def get(self, url: str, *, max_bytes: int) -> MetadataResponse:
        self.contact_count += 1
        self.requests.append((url, max_bytes))
        return MetadataResponse(
            request_identity=url,
            citation_url=url,
            status_code=200,
            body=b"{}",
            source_version=None,
            observed_at=datetime(2026, 8, 27, tzinfo=UTC),
            elapsed_time_ms=1,
        )


class _HttpErrorOpener:
    def __init__(self, *, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.body = body
        self.requests: list[tuple[str, str, float]] = []

    def open(self, request, *, timeout: float):  # noqa: ANN001
        self.requests.append((request.full_url, request.get_method(), timeout))
        raise urllib.error.HTTPError(
            request.full_url,
            self.status_code,
            "not found",
            {},
            BytesIO(self.body),
        )


def _transport_failure(cause: Exception) -> ReferenceAdapterError:
    """The NCBI transport's classified error for a failed GET, chained to its cause."""
    error = ReferenceAdapterError(
        "transport_error",
        f"Official NCBI metadata request failed: {type(cause).__name__}.",
        retryable=True,
    )
    error.__cause__ = cause
    return error


class _FailingSourceResponses(_SourceResponsesOnly):
    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        self.operations.append(f"fetch_refsnp:{rsid}")
        raise _transport_failure(OSError("forbidden-source-error-detail"))


class _FailingAssemblySourceResponses(_SourceResponsesOnly):
    def fetch_assembly_report(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        self.operations.append(f"fetch_assembly_report:{assembly_accession}")
        raise _transport_failure(TimeoutError("forbidden-assembly-source-error-detail"))


class _NegativeSourceResponses(_SourceResponsesOnly):
    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        self.operations.append(f"fetch_refsnp:{rsid}")
        return InvestigationSourceResponse(
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
                + rsid.removeprefix("rs")
            ),
            status_code=404,
            body=b'{"error":"rsID not found"}',
            observed_at="2026-08-26T00:00:00Z",
            elapsed_time_ms=11,
        )


class _NegativeAssemblySourceResponses(_SourceResponsesOnly):
    def fetch_assembly_report(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        self.operations.append(f"fetch_assembly_report:{assembly_accession}")
        return InvestigationSourceResponse(
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                f"{assembly_accession}/dataset_report"
            ),
            status_code=404,
            body=b'{"error":"assembly report not found"}',
            observed_at="2026-08-26T00:00:01Z",
            elapsed_time_ms=9,
        )


class _MalformedSourceResponse(_SourceResponsesOnly):
    def fetch_refsnp(self, rsid: str):  # noqa: ANN201
        self.operations.append(f"fetch_refsnp:{rsid}")
        return object()


class _MalformedAssemblySourceResponse(_SourceResponsesOnly):
    def fetch_assembly_report(self, assembly_accession: str):  # noqa: ANN201
        self.operations.append(f"fetch_assembly_report:{assembly_accession}")
        return object()


class _FailingInvestigationModel:
    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        del messages, info
        self.calls += 1
        raise ModelHTTPError(
            503,
            "offline-investigation-failure",
            body={"message": "offline failure"},
            headers={},
        )


class _PreOutputFailureAdapter(NebiusModelEndpointAdapter):
    def prepare_request(self, model_settings, model_request_parameters):
        del model_settings, model_request_parameters
        raise ValueError("pre-output-secret-sentinel")


class _OwnReturnsInvestigationModel(_AdaptiveInvestigationModel):
    """Fetches sources until this Attempt's own Tool returns hold both.

    Retained sources from an earlier Attempt are visible context, not this
    Attempt's typed results, so a fresh Attempt fetches them again.
    """

    async def respond(self, messages, info):
        returned = {
            part.tool_name
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        for tool_name, arguments in (
            ("fetch_refsnp", {"rsid": "rs429358"}),
            (
                "fetch_assembly_report",
                {
                    "assembly_accession": "GCF_000001405.40",
                    "refsnp_event_id": "missing",
                },
            ),
        ):
            if tool_name not in returned:
                if tool_name == "fetch_assembly_report":
                    arguments["refsnp_event_id"] = source_basis(info)["refsnp_event_id"]
                self.calls += 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name,
                            canonical_json(arguments),
                            tool_call_id=f"own-{self.calls}",
                        )
                    ]
                )
        return await super().respond(messages, info)


class InvestigationContractTests(RunPrerequisites, unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0.0,
            )
        )

    # The scripted model answers by global call order.
    @members_one_at_a_time()
    def test_provider_retry_preserves_completed_member_and_output_repair(self):
        from bio_neb.agentic_journey.model_availability import (
            RetryModelAvailabilityResponse,
        )
        from bio_neb.application.commands import (
            RespondModelAvailabilityReviewCommand,
            respond_model_availability_review,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from tests.sequential_investigation_support import (
            SecondOutputRepairOutageModel,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "requests": [
                    {
                        "request": f"Investigate {rsid} in sample HG001.",
                        "target": {"rsid": rsid},
                    }
                    for rsid in SequentialAdaptiveModel.rsids
                ],
            }
            child = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "request.json"
            request_file.write_text(canonical_json(request))
            source = SequentialSourceResponses()
            first = SecondOutputRepairOutageModel()
            pending = begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(child.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                model_factory=first.model,
                source_adapter_factory=lambda: source,
            )
            self.assertEqual(pending.status, "operational_review_pending")
            # The outage persists, so the second member's repair and the third
            # member's first request each hold their member in one Review for
            # their one cause; each item is answered on its own.
            retry = SequentialAdaptiveModel()
            answered = []
            while (
                request := observe(
                    ObserveRun(journey_file=str(child.path))
                ).model_availability_review.pending_request
            ) is not None:
                command = RespondModelAvailabilityReviewCommand(
                    journey_file=str(child.path),
                    request_event_id=request.request_event_id,
                    response=RetryModelAvailabilityResponse(action="retry"),
                )
                answer = respond_model_availability_review(
                    command,
                    environ={"LOCAL_MODEL_KEY": "secret"},
                    attempt=investigation_assignment(
                        model_factory=retry.model,
                        source_adapter_factory=lambda: source,
                    ),
                )
                answered.append((request.member_id, command, answer))
            self.assertEqual(len({member for member, _, _ in answered}), 2)
            receipts = observe(
                ObserveRun(journey_file=str(child.path))
            ).model_availability_review.receipts
            result = observe(ObserveRun(journey_file=str(child.path)))
            self.assertEqual(result.status, "investigation_review_pending")
            self.assertEqual(
                source.operations,
                [
                    operation
                    for rsid in SequentialAdaptiveModel.rsids
                    for operation in (
                        f"fetch_refsnp:{rsid}",
                        "fetch_assembly_report:GCF_000001405.40",
                    )
                ],
            )
            self.assertEqual(retry.calls, 4)
            # The collection keeps each member's own lineage.
            lineages = result.investigation.collection_lineages
            self.assertEqual(
                [len(lineage.validation_attempts) for lineage in lineages], [1, 2, 1]
            )
            # One Attempt per member, and one more per retried member.
            self.assertEqual(
                len(result.investigation.attempts),
                len(SequentialAdaptiveModel.rsids) + len(receipts),
            )
            before = child.path.read_bytes()
            for _, command, answer in answered:
                self.assertEqual(
                    respond_model_availability_review(command, environ={}), answer
                )
            self.assertEqual(child.path.read_bytes(), before)

    def test_an_unavailable_provider_retry_keeps_the_frontier_it_retried_from(
        self,
    ):
        # Each retry answer is recorded as the command records it and consumed by
        # the same shared Attempt lifecycle the command and the coordinator call.
        # Members, sources and output repair across a retry are pinned at the
        # public command by test_provider_retry_preserves_completed_member_and_output_repair.
        from bio_neb.agentic_journey.continuation import review_answer_parents
        from bio_neb.agentic_journey.model_availability import (
            ModelAvailabilityReviewResponded,
            RetryModelAvailabilityResponse,
            project_model_availability_review,
        )
        from bio_neb.investigation.investigation import InvestigationRequestItem
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from bio_neb.recording import read_evidence
        from tests.sequential_investigation_support import (
            SecondOutputRepairOutageModel,
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        async def still_unavailable(messages, info):
            raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})

        def answer_retry(journey):
            request = project_model_availability_review(
                read_evidence(journey)
            ).pending_request
            answer = ModelAvailabilityReviewResponded(
                request_event_id=request.request_event_id,
                response=RetryModelAvailabilityResponse(action="retry"),
            )
            return request, journey.record(
                answer.event_type,
                answer.model_dump(mode="json", exclude={"event_type"}),
                review_answer_parents(journey.read(), request.request_event_id),
            )

        investigation_input, single = _input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by=single.asserted_by,
            requests=tuple(
                InvestigationRequestItem(
                    request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
                )
                for rsid in SequentialAdaptiveModel.rsids
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = SequentialSourceResponses()
            pending, journey = _run_attempt(
                Path(directory).resolve(),
                SecondOutputRepairOutageModel(),
                source,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            self.assertEqual(pending.status, "operational_review_pending")
            started = next(
                e for e in journey.read() if e["event_type"] == "InvestigationStarted"
            )
            first, first_answer = answer_retry(journey)
            self.assertEqual(first.retry_frontier_event_id, started["event_id"])
            unavailable = investigation_assignment(
                model_factory=ScriptedModelEndpoint(still_unavailable),
                source_adapter_factory=lambda: source,
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                availability_review_response_event_id=first_answer["event_id"],
            )
            self.assertEqual(unavailable.status, "operational_review_pending")
            # The failed retry opens a new Review for the same member from the retry
            # Attempt (the third member's original Review is still pending beside it).
            retried = [
                request
                for request in project_model_availability_review(
                    read_evidence(journey)
                ).requests
                if request.pending
                and request.member_id == first.member_id
                and request.attempt_id != first.attempt_id
            ]
            self.assertEqual(len(retried), 1)
            # A retry that is itself unavailable keeps the frontier it retried from.
            # Known product bug: the member-held Review records the retry answer as
            # its frontier instead.
            self.assertEqual(retried[0].retry_frontier_event_id, started["event_id"])

    def test_rejected_model_request_holds_its_member_for_availability_review(
        self,
    ):
        # A rejected (4xx) model request enters the member's availability
        # Review with its request source instead of failing the item.
        async def refused(messages, info):
            raise ModelHTTPError(400, "offline", body={"message": "bad request"})

        model = SimpleNamespace(model=ScriptedModelEndpoint(refused))
        with tempfile.TemporaryDirectory() as temporary:
            source = _SourceResponsesOnly()
            projected, journey = _run_attempt(Path(temporary).resolve(), model, source)
            self.assertEqual(
                (projected.status, projected.stop_reason),
                ("operational_review_pending", None),
            )
            self.assertEqual(source.operations, [])
            events = journey.read()
            [review] = [
                e
                for e in events
                if e["event_type"] == "ModelAvailabilityReviewRequested"
            ]
            self.assertEqual(review["payload"]["exhaustion_reason"], "request_rejected")
            source_event = next(
                e
                for e in events
                if e["event_id"] == review["payload"]["logical_failure_event_id"]
            )
            self.assertEqual(source_event["event_type"], "ModelRequestFailed")
            self.assertEqual(source_event["payload"]["http_status_code"], 400)
            [completed] = [
                e for e in events if e["event_type"] == "InvestigationAttemptCompleted"
            ]
            self.assertEqual(
                completed["payload"]["held_review_event_ids"], [review["event_id"]]
            )
            self.assertFalse(
                any(
                    e["event_type"] in {"InvestigationAttemptFailed", "RunFailed"}
                    for e in events
                )
            )

    def test_members_refused_for_one_cause_share_one_review_answered_per_item(self):
        """Owner, #476 alignment G4.1, with the orchestrator's answer and the
        owner's amendment under the grant (#477 6483312280): the first item to
        hit a model endpoint fault opens one Review; every other item joins it at
        its next model call and sends nothing. Each item is answered on its own:
        a retried item re-sends once and proceeds, a stopped one ends with the
        Review's cause, nothing else is sent while the Review is open, and the
        Review closes with its last item. Answer-all gives one answer to all."""
        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )
        from bio_neb.application.commands import (
            RespondModelAvailabilityReviewCommand,
            respond_model_availability_review,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        calls, outage = [], []

        async def endpoint(messages, info):
            calls.append(messages)
            if outage:
                raise ModelHTTPError(400, "offline", body={"message": "bad request"})
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which allele for this request?",
                            "missing_information": "The allele was not specified.",
                            "asks_about": "target",
                        },
                        tool_call_id=f"question-{len(calls)}",
                    )
                ]
            )

        def read(journey):
            return project_model_availability_review(
                read_evidence(Journey.open(journey.path))
            )

        def answer(journey, source, request, response, *, every_item=False):
            # This Journey is the Investigation's alone, so the whole-Run read
            # the command repeats after writing is left out; the writes and the
            # members' Attempts are real.
            with patch("bio_neb.application.queries.review_journey"):
                respond_model_availability_review(
                    RespondModelAvailabilityReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=request.request_event_id,
                        response=response,
                        every_item=every_item,
                    ),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                    attempt=investigation_assignment(
                        model_factory=ScriptedModelEndpoint(endpoint),
                        source_adapter_factory=lambda: source,
                    ),
                )

        investigation_input, single = _input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by=single.asserted_by,
            requests=tuple(
                InvestigationRequestItem(
                    request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
                )
                for rsid in SequentialAdaptiveModel.rsids
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        members = len(SequentialAdaptiveModel.rsids)
        stop = {"action": "stop", "reason": "The endpoint is down."}
        for answer_all in (False, True):
            with (
                self.subTest(answer_all=answer_all),
                tempfile.TemporaryDirectory() as directory,
            ):
                calls.clear()
                outage[:] = [True]
                source = SequentialSourceResponses()
                _, journey = _run_attempt(
                    Path(directory).resolve(),
                    SimpleNamespace(model=ScriptedModelEndpoint(endpoint)),
                    source,
                    investigation_input=investigation_input,
                    assertion=assertion,
                )
                projection = read(journey)
                first, *joined = projection.requests
                # The first item's refusal is shown once and the same one asks;
                # every other item joined at its next call without sending.
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    (first.joins_event_id, first.exhaustion_reason),
                    (None, "request_rejected"),
                )
                self.assertEqual(len(joined), members - 1)
                for item in joined:
                    self.assertEqual(
                        (
                            item.joins_event_id,
                            item.joined_before_sending,
                            item.fault_key,
                            item.provider_statement,
                        ),
                        (
                            first.request_event_id,
                            True,
                            first.fault_key,
                            first.provider_statement,
                        ),
                    )
                self.assertEqual(
                    len({item.member_id for item in projection.requests}), members
                )
                if answer_all:
                    answer(journey, source, joined[0], stop, every_item=True)
                    self.assertEqual(len(calls), 2)
                else:
                    # The outage is over: the retried item re-sends once, past the
                    # hold of the still-open Review, and proceeds to its question.
                    outage.clear()
                    answer(journey, source, first, {"action": "retry"})
                    self.assertEqual(len(calls), 3)
                    for item in joined:
                        answer(journey, source, item, stop)
                    # A stopped item sends nothing.
                    self.assertEqual(len(calls), 3)
                answered = read(journey)
                self.assertIsNone(answered.pending_request)
                actions = {
                    receipt.request_event_id: receipt.response.action
                    for receipt in answered.receipts
                }
                expected = {
                    item.request_event_id: "stop" for item in projection.requests
                }
                if not answer_all:
                    expected[first.request_event_id] = "retry"
                self.assertEqual(
                    {item: actions.get(item) for item in expected}, expected
                )
                stopped = [
                    e["payload"]["reason"]
                    for e in read_evidence(Journey.open(journey.path))
                    if e["event_type"] == "ModelAvailabilityStopped"
                ]
                # Each stopped item ends with the Review's own reason.
                self.assertEqual(
                    stopped, [stop["reason"]] * (members if answer_all else members - 1)
                )

    def test_continue_whose_configuration_no_longer_resolves_records_the_fault_and_holds(
        self,
    ):
        # Investigation records a configuration fault and raises the invariant
        # the coordinator holds on, exactly as planning does; resume after the
        # fix continues. An Attempt resolves only the registered selections, so
        # the fault is the missing credential, and the retry ignores an edited
        # runtime file.
        from bio_neb.failure import Invariant
        from bio_neb.investigation.investigation import RetryInvestigationMember
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            continue_investigation,
        )
        from tests.sequential_investigation_support import SequentialAdaptiveModel

        with tempfile.TemporaryDirectory() as temporary:
            model, source = _AdaptiveInvestigationModel(), _SourceResponsesOnly()
            projected, journey = _run_attempt(Path(temporary).resolve(), model, source)
            proposal = projected.proposals[-1]
            revised = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=projected.review_requests[-1].request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=proposal.proposal_id,
                        asserted_by="researcher:owner",
                        operations=(
                            RetryInvestigationMember(
                                operation="retry",
                                work_item_id=proposal.work_items[0].work_item_id,
                            ),
                        ),
                    ),
                )
            )
            self.assertEqual(revised.status, "investigation_work_ready")
            command = ContinueInvestigationCommand(journey_file=str(journey.path))
            before = Journey.open(journey.path).read()
            with self.assertRaises(Invariant) as held:
                continue_investigation(
                    command,
                    environ={},
                    model_factory=model.model,
                    source_adapter_factory=lambda: source,
                )
            after = Journey.open(journey.path).read()
            self.assertEqual(after[: len(before)], before)
            started, failure, closed = after[len(before) :]
            self.assertEqual(
                [e["event_type"] for e in (started, failure, closed)],
                ["PassStarted", "RuntimeConfigurationFailed", "PassClosed"],
            )
            self.assertEqual(
                (failure["payload"]["code"], failure["payload"]["violations"]),
                ("model_credential_unavailable", ["model_roster.qwen.credential_env"]),
            )
            # Recorded at the Investigation's causal head, not after the interval.
            self.assertEqual(failure["parent_ids"], [before[-1]["event_id"]])
            self.assertEqual(
                held.exception.invariant_id, "model_credential_unavailable"
            )
            self.assertEqual(
                held.exception.evidence,
                {
                    "configuration_failure_event_id": failure["event_id"],
                    "violations": failure["payload"]["violations"],
                },
            )
            self.assertRegex(held.exception.cause, r"^RuntimeConfigError@")
            # The file is not consulted: one naming another heartbeat and an
            # endpoint no adapter serves changes nothing about the retry.
            runtime_file = journey.run_directory / "investigation-runtime.json"
            changed = json.loads(runtime_file.read_text(encoding="utf-8"))
            changed["heartbeat_interval_seconds"] += 1
            changed["model_roster"]["qwen"]["endpoint"] = "https://edited.invalid/v1"
            runtime_file.write_text(json.dumps(changed), encoding="utf-8")
            retry = SequentialAdaptiveModel()
            continued = continue_investigation(
                command,
                environ={"LOCAL_MODEL_KEY": "secret"},
                model_factory=retry.model,
                source_adapter_factory=lambda: source,
            )
            self.assertEqual(
                (continued.status, continued.stop_reason), ("review_pending", None)
            )
            self.assertEqual([p.revision_number for p in continued.proposals], [1, 2])
            self.assertEqual(len(continued.attempts), 2)
            events = Journey.open(journey.path).read()
            [registration] = [
                e for e in events if e["event_type"] == "RunInputsRegistered"
            ]
            bound = [
                e["payload"]["runtime_selections"]
                for e in events
                if e["event_type"] == "InvestigationAttemptBound"
            ]
            self.assertEqual(len(bound), 2)
            for selections in bound:
                self.assertEqual(
                    selections, registration["payload"]["runtime_selections"]
                )

    def test_reentry_whose_configuration_no_longer_resolves_records_the_fault_and_holds(
        self,
    ):
        from bio_neb.agentic_journey.effort import (
            ExtendAgentEffortResponse,
            pending_effort_work,
            project_effort_review,
            respond_effort_review,
        )
        from bio_neb.failure import Invariant
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from bio_neb.recording import read_evidence

        with tempfile.TemporaryDirectory() as temporary:
            model, source = _AdaptiveInvestigationModel(), _SourceResponsesOnly()
            stopped, journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                source,
                invocation_overrides={"model_requests": 1},
            )
            self.assertEqual(stopped.status, "effort_review_pending")
            [request] = project_effort_review(read_evidence(journey)).requests
            receipt = respond_effort_review(
                journey,
                request.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=5),
            )
            # The registered selections are the only configuration read: the
            # reentry's fault is its missing credential.
            before = journey.read()
            with self.assertRaises(Invariant) as held:
                investigation_assignment(
                    model_factory=model.model,
                    source_adapter_factory=lambda: source,
                ).run(
                    journey,
                    environ={},
                    effort_review_response_event_id=receipt.response_event_id,
                )
            self.assertEqual(
                held.exception.invariant_id, "model_credential_unavailable"
            )
            added = [e["event_type"] for e in journey.read()[len(before) :]]
            self.assertEqual(
                added, ["PassStarted", "RuntimeConfigurationFailed", "PassClosed"]
            )
            # The extension stays pending for the retry after the fix.
            self.assertEqual(
                pending_effort_work(
                    journey,
                    receipt.response_event_id,
                    agent_role="investigation_planner",
                )[0].request_event_id,
                request.request_event_id,
            )
            # The retry resolves the registration, not the file: removing the
            # file faults nothing, and the extension is spent.
            (journey.run_directory / "investigation-runtime.json").unlink()
            fixed = journey.read()
            investigation_assignment(
                model_factory=model.model,
                source_adapter_factory=lambda: source,
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret"},
                effort_review_response_event_id=receipt.response_event_id,
            )
            retried = journey.read()[len(fixed) :]
            self.assertIn("AttemptStarted", [e["event_type"] for e in retried])
            self.assertNotIn(
                "RuntimeConfigurationFailed", [e["event_type"] for e in retried]
            )
            with self.assertRaisesRegex(ValueError, "exact unused authority"):
                pending_effort_work(
                    journey,
                    receipt.response_event_id,
                    agent_role="investigation_planner",
                )

    def test_causal_rework_provider_review_takes_precedence_over_completed_evidence(
        self,
    ):
        from bio_neb.application.commands import (
            RespondModelAvailabilityReviewCommand,
            respond_model_availability_review,
        )
        from bio_neb.investigation.investigation import RetryInvestigationMember
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            continue_investigation,
        )
        from bio_neb.investigation.investigation_observation_commands import (
            investigation_observe,
        )
        from tests.sequential_investigation_support import (
            ObservationEvidence,
            SequentialAdaptiveModel,
            SequentialObservationAdapter,
            SequentialSourceResponses,
            approved_collection_observation,
        )

        async def outage(messages, info):
            raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})

        with tempfile.TemporaryDirectory() as directory:
            inp, journey, _, descriptor, command = approved_collection_observation(
                Path(directory).resolve(),
                cardinality=1,
                forked=True,
                parent_file=self.prerequisite(
                    "executed", investigation_request=collection_request(1)
                ).journey_file,
            )
            investigation_observe(
                command,
                adapter=SequentialObservationAdapter(
                    command, descriptor, inp.sample_id
                ),
                evidence_reader=ObservationEvidence(descriptor),
            )
            original = project_investigation(read_evidence(Journey.open(journey.path)))
            proposal = original.proposals[-1]
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=original.review_requests[-1].request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=proposal.proposal_id,
                        asserted_by="researcher:owner",
                        operations=(
                            RetryInvestigationMember(
                                operation="retry",
                                work_item_id=proposal.work_items[0].work_item_id,
                            ),
                        ),
                    ),
                )
            )
            continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "secret"},
                model_factory=ScriptedModelEndpoint(outage),
                source_adapter_factory=SequentialSourceResponses,
            )
            pending = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(pending.status, "operational_review_pending")
            request = pending.model_availability_review.pending_request
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=request.request_event_id,
                    response={"action": "retry"},
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=investigation_assignment(
                    model_factory=lambda loaded,
                    selected: SequentialAdaptiveModel().model(loaded, selected),
                    source_adapter_factory=SequentialSourceResponses,
                ),
            )
            result = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertIsNone(result.model_availability_review.pending_request)
            self.assertEqual(len(result.investigation.proposals), 2)
            self.assertEqual(result.investigation.proposals[-1].revision_number, 2)

    def test_transient_model_outage_preserves_completed_source_tools(self):
        class OutageModel(_AdaptiveInvestigationModel):
            failed = False

            async def respond(self, messages, info):
                if self.calls == 1 and not self.failed:
                    self.failed = True
                    raise ModelHTTPError(503, "offline", headers={"retry-after": "0"})
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as temporary:
            model, source = OutageModel(), _SourceResponsesOnly()
            projected, journey = _run_attempt(Path(temporary).resolve(), model, source)
            self.assertEqual(projected.status, "review_pending")
            self.assertEqual(
                source.operations,
                ["fetch_refsnp:rs429358", "fetch_assembly_report:GCF_000001405.40"],
            )
            self.assertEqual(projected.model_contact_count, 4)
            self.assertEqual(
                len(
                    [
                        e
                        for e in journey.read()
                        if e["event_type"] == "ModelRequestRecorded"
                    ]
                ),
                3,
            )
            self.assertEqual(
                project_investigation(read_evidence(Journey.open(journey.path))),
                projected,
            )

    def test_incompatible_assembly_requires_repair_of_false_complete_output(
        self,
    ) -> None:
        class FalseCompletionModel(_AdaptiveInvestigationModel):
            async def respond(self, messages, info):  # noqa: ANN001,ANN201
                response = await super().respond(messages, info)
                if self.calls != 3:
                    return response
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            info.output_tools[0].name,
                            canonical_json(
                                {**source_basis(info), "outcome": "complete"}
                            ),
                            tool_call_id="false-completion",
                        )
                    ]
                )

        body = json.loads(ASSEMBLY_BODY)
        body["reports"][0]["accession"] = "GCF_000001405.39"
        with tempfile.TemporaryDirectory() as temporary:
            model = FalseCompletionModel()
            source = _SourceResponsesOnly(assembly_body=json.dumps(body).encode())
            projected, journey = _run_attempt(Path(temporary).resolve(), model, source)
            lineage = projected.attempts[0].structured_output_lineages[0]
            self.assertEqual(
                [item.status for item in lineage.validation_attempts],
                ["invalid", "valid"],
            )
            self.assertIn(
                "unresolved_sources_marked_complete",
                str(lineage.validation_attempts[0].errors),
            )
            self.assertEqual(lineage.accepted_output["outcome"], "unresolved")
            # The repaired finding is the member's issue, keyed with its verdict.
            self.assertEqual(projected.proposals, ())
            issue = projected.pending_request.question.issue
            self.assertEqual(
                (issue.source, issue.verdict), ("assembly_report", "incompatible")
            )
            # Code copied the cited report's conflicts.
            (report,) = [
                e["payload"]["result"]
                for e in read_evidence(journey)
                if e["event_type"] == "AssemblyReportToolResultRecorded"
            ]
            self.assertEqual(tuple(report["conflicts"]), issue.conflicts)
            self.assertNotIn("unresolved_conflicts", lineage.accepted_output)
            self.assertEqual(
                issue.finding_event_id,
                lineage.validation_attempts[-1].validation_event_id,
            )
            self.assertEqual(model.calls, 4)
            self.assertEqual(len(source.operations), 2)
            self.assertEqual(
                project_investigation(read_evidence(Journey.open(journey.path))),
                projected,
            )

    def test_fenced_output_is_accepted_and_reopens_with_raw_response(self) -> None:
        class FencedOutput(_AdaptiveInvestigationModel):
            raw = None

            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                part = response.parts[0]
                if part.tool_name == "compile_proposal":
                    self.raw = (
                        "```json\n" + canonical_json(part.args_as_dict()) + "\n```"
                    )
                    part.args = self.raw
                return response

        with tempfile.TemporaryDirectory() as temporary:
            model = FencedOutput()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, _SourceResponsesOnly()
            )
            self.assertEqual(projection.status, "review_pending")
            lineage = projection.attempts[0].structured_output_lineages[0]
            self.assertEqual([v.status for v in lineage.validation_attempts], ["valid"])
            self.assertEqual(lineage.accepted_output["outcome"], "complete")
            events = read_evidence(Journey.open(journey.path))
            self.assertEqual(project_investigation(events), projection)
            recorded = next(
                e for e in events if e["event_type"] == "StructuredOutputCallRecorded"
            )
            response = next(
                e
                for e in events
                if e["event_id"] == recorded["payload"]["response_event_id"]
            )
            parts = response["payload"]["compatible_adapter_evidence"]["response"][
                "message"
            ]["parts"]
            self.assertEqual(parts[0]["args"], model.raw)
            self.assertEqual(
                recorded["payload"]["schema_normalized_arguments"],
                lineage.accepted_output,
            )
            # The Model turn stripped the fence, and the call records it.
            self.assertIs(recorded["payload"]["arguments_fenced"], True)
            self.assertEqual(
                recorded["payload"]["arguments"],
                json.loads(model.raw.removeprefix("```json\n").removesuffix("\n```")),
            )
            hostile = copy.deepcopy(events)
            changed = next(e for e in hostile if e["event_id"] == response["event_id"])
            changed["payload"]["compatible_adapter_evidence"]["response"]["message"][
                "parts"
            ][0]["args"] = model.raw.replace('"complete"', '"unresolved"')
            with self.assertRaisesRegex(
                ValueError, "structured-output call changed its arguments"
            ):
                project_investigation(hostile)

    def test_invalid_fenced_output_repairs_without_losing_its_raw_evidence(
        self,
    ) -> None:
        class InvalidThenValid(_AdaptiveInvestigationModel):
            def __init__(self, invalid):
                super().__init__()
                self.invalid = invalid

            async def respond(self, messages, info):
                response = await super().respond(messages, info)
                if self.calls == 3:
                    response.parts[0].args = self.invalid
                return response

        for invalid in (
            "```json\n{broken}\n```",
            '```json\n{"outcome":"nope"}\n```',
            "[]",
        ):
            with (
                self.subTest(invalid=invalid),
                tempfile.TemporaryDirectory() as temporary,
            ):
                model = InvalidThenValid(invalid)
                source = _SourceResponsesOnly()
                projection, journey = _run_attempt(
                    Path(temporary).resolve(), model, source
                )
                lineage = projection.attempts[0].structured_output_lineages[0]
                self.assertEqual(
                    [v.status for v in lineage.validation_attempts],
                    ["invalid", "valid"],
                )
                self.assertEqual(lineage.accepted_output["outcome"], "complete")
                self.assertEqual(source.contact_count, 2)
                self.assertEqual(model.calls, 4)
                events = read_evidence(Journey.open(journey.path))
                self.assertEqual(project_investigation(events), projection)
                calls = [
                    e
                    for e in events
                    if e["event_type"] == "StructuredOutputCallRecorded"
                ]
                response = next(
                    e
                    for e in events
                    if e["event_id"] == calls[0]["payload"]["response_event_id"]
                )
                self.assertEqual(
                    response["payload"]["compatible_adapter_evidence"]["response"][
                        "message"
                    ]["parts"][0]["args"],
                    invalid,
                )

    def test_a_late_output_that_retypes_the_locus_is_refused_by_its_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _LateRepairModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )
            self.assertEqual(projection.status, "review_pending")
            self.assertEqual(model.calls, 4)
            self.assertEqual(adapter.contact_count, 2)
            lineage = projection.attempts[0].structured_output_lineages[0]
            self.assertEqual(
                tuple(item.status for item in lineage.validation_attempts),
                ("invalid", "valid"),
            )
            # The re-typed locus is refused by its fields, never overwritten, and
            # code copies the locus from the cited record.
            self.assertEqual(
                set(lineage.validation_attempts[0].errors),
                # One derived validator names each field and its error.
                {"contig:extra_forbidden", "one_based_position:extra_forbidden"},
            )
            self.assertNotIn("contig", lineage.accepted_output)
            (item,) = {
                (w.target.contig, w.target.one_based_position)
                for w in projection.proposals[0].work_items
            }
            self.assertEqual(item, ("NC_000019.10", 44908684))
            self.assertEqual(project_investigation(read_evidence(journey)), projection)

    def test_configured_corrections_are_independent_and_reopen_uses_bound_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # Two refused calls, two admitted fetches, two refused outputs, then
            # the valid one: an admitted call resets the failed turns in a row, so
            # each pair counts on its own under three.
            model = _IndependentCorrectionsModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                adapter,
                continuation_policy={"failed_turns": 2},
                invocation_overrides={"failed_turns": 3},
            )
            self.assertEqual(projection.status, "review_pending")
            self.assertEqual(model.calls, 7)
            effective = projection.attempts[0].effective_invocation_policy
            self.assertEqual(effective.agent_role, "investigation_planner")
            # The Attempt's amendment replaces the Run's number.
            self.assertEqual(effective.continuation.failed_turns, 3)
            self.assertEqual(effective.continuation.stage_failed_turns, {})
            self.assertEqual(effective.exhaustion.model_availability, "review")
            self.assertEqual(
                effective.model_settings.request_settings(),
                {"parallel_tool_calls": False, "max_tokens": 245760},
            )
            self.assertEqual(adapter.contact_count, 2)
            lineage = projection.attempts[0].structured_output_lineages[0]
            self.assertEqual(
                tuple(item.status for item in lineage.validation_attempts),
                ("invalid", "invalid", "valid"),
            )
            self.assertEqual(lineage.max_retries, 3)
            self.assertEqual(lineage.accepted_output["outcome"], "complete")
            runtime_path = journey.run_directory / "investigation-runtime.json"
            changed_config = json.loads(runtime_path.read_text())
            changed_config["agent_invocation"] = {
                **changed_config.get("agent_invocation", {}),
                "failed_turns": 1,
            }
            runtime_path.write_text(canonical_json(changed_config))
            events = read_evidence(Journey.open(journey.path))
            self.assertEqual(project_investigation(events), projection)
            for missing in ("agent_invocation", "max_tokens", "model_availability"):
                hostile = copy.deepcopy(events)
                selections = next(
                    event["payload"]["runtime_selections"]
                    for event in hostile
                    if event["event_type"] == "InvestigationAttemptBound"
                )
                if missing == "agent_invocation":
                    del selections[missing]
                elif missing == "max_tokens":
                    del selections["model_roster"]["qwen"]["settings_selection"][
                        "baseline"
                    ][missing]
                else:
                    del selections["agent_invocation"]["exhaustion"][missing]
                with (
                    self.subTest(missing_default=missing),
                    self.assertRaises(ValueError),
                ):
                    project_investigation(hostile)
            self.assertEqual(
                projection.pending_request.review_policy.model_dump(),
                {"stage": "investigation", "approval": "mandatory"},
            )
            for value in (
                {},
                {"stage": "investigation"},
                {"stage": "execution", "approval": "mandatory"},
                {"stage": "investigation", "approval": "optional"},
            ):
                hostile = copy.deepcopy(events)
                next(
                    e
                    for e in hostile
                    if e["event_type"] == "InvestigationReviewRequested"
                )["payload"]["review_policy"] = value
                with self.subTest(review_policy=value), self.assertRaises(ValueError):
                    project_investigation(hostile)
            for attack in (
                "policy",
                "identity",
                "overrides",
                "request",
                "registration",
            ):
                with self.subTest(invocation=attack):
                    hostile = copy.deepcopy(events)
                    started = next(
                        e["payload"]
                        for e in hostile
                        if e["event_type"] == "AttemptStarted"
                    )
                    if attack == "policy":
                        started["effective_invocation_policy"]["agent_role"] = (
                            "planning_agent"
                        )
                    elif attack == "identity":
                        started["invocation_policy_id"] = (
                            "invocation-policy:" + "0" * 64
                        )
                    elif attack == "overrides":
                        started["invocation_overrides"] = {"failed_turns": 1}
                    elif attack == "request":
                        request = next(
                            e["payload"]
                            for e in hostile
                            if e["event_type"] == "ModelRequestRecorded"
                        )
                        request["compatible_adapter_evidence"]["request"][
                            "model_settings"
                        ]["max_tokens"] = 32768
                    else:
                        bound = next(
                            e["payload"]
                            for e in hostile
                            if e["event_type"] == "InvestigationAttemptBound"
                        )
                        bound["runtime_selections"]["agent_invocation"][
                            "model_settings"
                        ]["max_tokens"] = 32768
                    with self.assertRaises(ValueError):
                        project_investigation(hostile)
            for field in ("continuation_policy", "continuation_policy_id"):
                for mutation in ("missing", "changed"):
                    with self.subTest(field=field, mutation=mutation):
                        hostile = copy.deepcopy(events)
                        payload = next(
                            event["payload"]
                            for event in hostile
                            if event["event_type"] == "InvestigationAttemptBound"
                        )
                        if mutation == "missing":
                            payload.pop(field)
                        elif field == "continuation_policy":
                            payload[field]["failed_turns"] = 9
                        else:
                            payload[field] = "continuation-policy:" + "0" * 64
                        with self.assertRaises(ValueError):
                            project_investigation(hostile)
            # Even a self-consistent replacement identity cannot authorize a
            # request whose preceding reply reached that retained policy's limit.
            # The output lineage records the policy's most failed turns, so the
            # replacement is refused there first.
            hostile = copy.deepcopy(events)
            payload = next(
                event["payload"]
                for event in hostile
                if event["event_type"] == "InvestigationAttemptBound"
            )
            payload["continuation_policy"]["failed_turns"] = 2
            payload["continuation_policy_id"] = (
                "continuation-policy:"
                + hashlib.sha256(
                    canonical_json(payload["continuation_policy"]).encode()
                ).hexdigest()
            )
            with self.assertRaisesRegex(
                ValueError, "Investigation proposal changed its exact Attempt lineage"
            ):
                project_investigation(hostile)
            self.assertEqual(adapter.contact_count, 2)

    def test_configured_output_and_action_exhaustion_stop_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_model = _MalformedOutputModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve() / "output",
                output_model,
                adapter,
                continuation_policy={"failed_turns": 3},
            )
            self.assertEqual(
                (projection.status, projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(projection.attempts[-1].stop_reason, "failed_turns")
            self.assertEqual(output_model.calls, 3)
            self.assertEqual(adapter.contact_count, 0)
            events = read_evidence(journey)
            request = project_agent_work_review(events).pending_request
            self.assertEqual(
                (request.agent_role, request.cause),
                ("investigation_planner", "failed_turns"),
            )
            self.assertEqual(
                [item.event_type for item in request.faults],
                ["AgentOutputValidated"] * 3,
            )
            self.assertFalse(
                any(
                    e["event_type"] in {"InvestigationAttemptFailed", "RunFailed"}
                    for e in events
                )
            )
            self.assertEqual(project_investigation(events), projection)
            action_model = _IndependentCorrectionsModel()
            projection, journey = _run_attempt(
                Path(temporary).resolve() / "action",
                action_model,
                adapter,
                continuation_policy={"failed_turns": 1},
            )
            self.assertEqual(
                (projection.status, projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(projection.attempts[-1].stop_reason, "failed_turns")
            self.assertEqual(action_model.calls, 1)
            self.assertEqual(adapter.contact_count, 0)
            events = read_evidence(journey)
            (refused,) = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            self.assertEqual(refused["payload"]["disposition"], "singleton_refused")
            request = project_agent_work_review(events).pending_request
            self.assertEqual(
                (request.cause, request.reason, request.source_event_id),
                (
                    "failed_turns",
                    "the model's replies failed their limit of turns in a row",
                    refused["event_id"],
                ),
            )
            self.assertEqual(project_investigation(events), projection)

    def test_an_output_beside_a_source_fetch_is_refused_and_the_fetch_result_is_sent(
        self,
    ) -> None:
        class FetchWithOutput:
            def __init__(self) -> None:
                self.calls = 0
                self.model = ScriptedModelEndpoint(self.respond)

            async def respond(self, messages, info):
                del messages
                self.calls += 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            canonical_json({"rsid": "rs429358"}),
                            tool_call_id=f"fetch-{self.calls}",
                        ),
                        ToolCallPart(
                            info.output_tools[0].name,
                            canonical_json({"outcome": "complete"}),
                            tool_call_id=f"output-{self.calls}",
                        ),
                    ]
                )

        with tempfile.TemporaryDirectory() as temporary:
            model = FetchWithOutput()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                adapter,
                continuation_policy={"failed_turns": 1},
            )
            # An output must come alone (ADR 0044): the first reply's output is
            # refused at routing; its fetch runs, so the reply is no failed turn,
            # and its result reaches the next request. The same reply again has its
            # output refused and its fetch refused as a repeat: nothing of it
            # worked, so its failed turn, the first in a row, stops the Attempt at
            # this limit of one before its fetch runs.
            self.assertEqual(
                (projection.status, model.calls, adapter.contact_count),
                ("agent_work_review_pending", 2, 1),
            )
            events = read_evidence(journey)
            request = project_agent_work_review(events).pending_request
            (refused,) = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "multi_call_refused"
            ]
            self.assertEqual(
                (request.cause, request.source_event_id),
                ("failed_turns", refused["event_id"]),
            )
            first, second = [e for e in events if e["event_type"] == "AgentTurnRouted"]
            # Each reply is the first in its row: the admitted fetch reset it.
            self.assertEqual(
                [first["payload"]["faults"], second["payload"]["faults"]],
                [
                    [{"key": "output_not_alone", "count": 1}],
                    [{"key": "output_not_alone", "count": 1}],
                ],
            )
            (result,) = [
                e for e in events if e["event_type"] == "RefSnpToolResultRecorded"
            ]
            (later,) = [
                e
                for e in events[events.index(result) :]
                if e["event_type"] == "ModelRequestRecorded"
            ]
            returned = later["payload"]["compatible_adapter_evidence"]["request"][
                "messages"
            ][-1]["parts"]
            # The fetch's own result, then the turn's note.
            self.assertEqual(
                [(p["part_kind"], p.get("content")) for p in returned][1:],
                [("retry-prompt", first["payload"]["feedback"])],
            )
            self.assertEqual(
                (returned[0]["part_kind"], returned[0]["tool_call_id"]),
                ("tool-return", "fetch-1"),
            )
            self.assertFalse(
                any(e["event_type"] == "StructuredOutputCallRecorded" for e in events)
            )
            self.assertEqual(project_investigation(events), projection)

    def test_context_exhaustion_after_a_source_fetch_stops_for_review_and_continues(
        self,
    ) -> None:
        class OverflowAfterFetch:
            """The request carrying the fetched source overflows its window.

            It raises the typed error the Nebius Adapter raises before sending.
            """

            def __init__(self) -> None:
                self.calls = 0
                self.model = ScriptedModelEndpoint(self.respond)

            async def respond(self, messages, info):
                del messages, info
                self.calls += 1
                if self.calls > 1:
                    raise ModelContextExhausted(
                        input_estimate=270000,
                        safety_margin=8192,
                        context_window=262144,
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            canonical_json({"rsid": "rs429358"}),
                            tool_call_id="fetch-1",
                        )
                    ]
                )

        with tempfile.TemporaryDirectory() as temporary:
            adapter = _SourceResponsesOnly()
            model = OverflowAfterFetch()
            stopped, journey = _run_attempt(Path(temporary).resolve(), model, adapter)
            self.assertEqual(
                (
                    stopped.status,
                    stopped.attempts[-1].stop_reason,
                    model.calls,
                    adapter.contact_count,
                ),
                ("agent_work_review_pending", "context_exhausted", 2, 1),
            )
            events = read_evidence(journey)
            (failure,) = [e for e in events if e["event_type"] == "ModelRequestFailed"]
            request = project_agent_work_review(events).pending_request
            self.assertEqual(
                (request.agent_role, request.cause, request.source_event_id),
                ("investigation_planner", "context_exhausted", failure["event_id"]),
            )
            self.assertFalse(
                any(
                    e["event_type"] in {"InvestigationAttemptFailed", "RunFailed"}
                    for e in events
                )
            )
            self.assertEqual(project_investigation(events), stopped)
            answer = respond_agent_work_review(
                journey,
                request.request_event_id,
                ContinueAgentWorkResponse(action="continue"),
            )
            resumed_model = _OwnReturnsInvestigationModel()
            resumed = investigation_assignment(
                model_factory=resumed_model.model,
                source_adapter_factory=lambda: adapter,
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                work_review_response_event_id=answer.response_event_id,
            )
            # The member's Attempt ends its own work; the collection's proposal
            # closes beside it once every member settles.
            self.assertEqual(
                [item.outcome for item in resumed.attempts],
                ["agent_work_review_pending", "work_completed"],
            )
            self.assertEqual(resumed.status, "review_pending")
            self.assertEqual(project_investigation(read_evidence(journey)), resumed)

    def test_second_truncation_after_a_source_fetch_stops_for_review_and_continues(
        self,
    ) -> None:
        class CutOffAfterFetch:
            """Every reply after the fetch stops on its output ceiling."""

            def __init__(self) -> None:
                self.calls = 0
                self.seen = []
                self.model = ScriptedModelEndpoint(self.respond)

            async def respond(self, messages, info):
                self.seen.append((messages, info))
                self.calls += 1
                if self.calls > 1:
                    return ModelResponse(
                        parts=[TextPart("partial")],
                        usage=RequestUsage(input_tokens=10, output_tokens=32768),
                        finish_reason="length",
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            canonical_json({"rsid": "rs429358"}),
                            tool_call_id="fetch-1",
                        )
                    ]
                )

        with tempfile.TemporaryDirectory() as temporary:
            adapter = _SourceResponsesOnly()
            model = CutOffAfterFetch()
            # The second failed turn in a row stops.
            stopped, journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                adapter,
                continuation_policy={"failed_turns": 2},
            )
            self.assertEqual(
                (
                    stopped.status,
                    stopped.attempts[-1].stop_reason,
                    model.calls,
                    adapter.contact_count,
                ),
                ("agent_work_review_pending", "failed_turns", 3, 1),
            )
            # One retry: it carries the truncation correction, never the cut-off reply.
            retry_messages, _ = model.seen[2]
            offered = sorted(
                tool.name
                for tool in (
                    *model.seen[1][1].function_tools,
                    *model.seen[1][1].output_tools,
                )
            )
            self.assertNotIn(
                "partial", ModelMessagesTypeAdapter.dump_json(retry_messages).decode()
            )
            retry = retry_messages[-1].parts[-1]
            self.assertEqual(
                (retry.part_kind, retry.content),
                (
                    "retry-prompt",
                    "Your reply was cut at 32768 tokens before any tool call.\n"
                    "Allowed form: a short reply that calls one of "
                    + ", ".join(f"`{name}`" for name in offered)
                    + ".\nCount: failed turn 1 of 2 in a row. At 2 the Attempt stops and "
                    "the researcher reviews it.\nLast correction: one more failed turn "
                    "stops the Attempt for review.",
                ),
            )
            events = read_evidence(journey)
            retried, truncation = [
                e for e in events if e["event_type"] == "ModelCompletionTruncated"
            ]
            request = project_agent_work_review(events).pending_request
            self.assertEqual(
                (
                    request.agent_role,
                    request.cause,
                    request.source_event_id,
                    request.truncation.request_number,
                    request.truncation.completion_tokens,
                ),
                (
                    "investigation_planner",
                    "failed_turns",
                    truncation["event_id"],
                    3,
                    32768,
                ),
            )
            self.assertEqual(
                [
                    (item.event_id, item.correction_key, item.correction_count)
                    for item in request.faults
                ],
                [
                    (retried["event_id"], "output_truncated", 1),
                    (truncation["event_id"], "output_truncated", 2),
                ],
            )
            self.assertFalse(
                any(
                    e["event_type"] in {"InvestigationAttemptFailed", "RunFailed"}
                    for e in events
                )
            )
            self.assertEqual(project_investigation(events), stopped)
            answer = respond_agent_work_review(
                journey,
                request.request_event_id,
                ContinueAgentWorkResponse(action="continue"),
            )
            resumed_model = _OwnReturnsInvestigationModel()
            resumed = investigation_assignment(
                model_factory=resumed_model.model,
                source_adapter_factory=lambda: adapter,
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                work_review_response_event_id=answer.response_event_id,
            )
            # The member's Attempt ends its own work; the collection's proposal
            # closes beside it once every member settles.
            self.assertEqual(
                [item.outcome for item in resumed.attempts],
                ["agent_work_review_pending", "work_completed"],
            )
            self.assertEqual(resumed.status, "review_pending")
            self.assertEqual(project_investigation(read_evidence(journey)), resumed)

    def test_repeated_missing_source_basis_stops_for_review_and_continues_in_a_new_attempt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = _SourceResponsesOnly()
            stopped, journey = _run_attempt(
                Path(temporary).resolve(), _RepeatedActionModel(), adapter
            )
            self.assertEqual(
                (stopped.status, stopped.stop_reason),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(adapter.contact_count, 1)
            events = read_evidence(journey)
            refused = [
                e
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["status"] == "invalid"
            ]
            # Each refused repeat is a failed turn: three corrections, then the
            # default's fourth failed turn in a row stops.
            self.assertEqual(
                [e["payload"]["tool_call_id"] for e in refused],
                [f"repeated-{call}" for call in range(2, 6)],
            )
            dispositions = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "singleton_refused"
            ]
            request = project_agent_work_review(events).pending_request
            self.assertEqual(
                (request.cause, request.source_event_id),
                ("failed_turns", dispositions[-1]["event_id"]),
            )
            self.assertEqual(
                [
                    (item.correction_key.split(" @ ")[0], item.correction_count)
                    for item in request.faults
                ],
                [
                    # The lookup ran, so the refusal names it.
                    (
                        "fetch_assembly_report: fetch_assembly_report_refsnp_event_unknown",
                        count,
                    )
                    for count in range(1, 5)
                ],
            )
            instruction = "Fetch both sources again in this Attempt, then compile."
            answer = respond_agent_work_review(
                journey,
                request.request_event_id,
                ContinueAgentWorkResponse(action="continue", instruction=instruction),
            )
            model = _OwnReturnsInvestigationModel()
            resumed = investigation_assignment(
                model_factory=model.model,
                source_adapter_factory=lambda: adapter,
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                work_review_response_event_id=answer.response_event_id,
            )
            self.assertEqual(resumed.status, "review_pending")
            # The member's Attempt ends its own work; the collection's proposal
            # closes beside it once every member settles.
            self.assertEqual(
                [item.outcome for item in resumed.attempts],
                ["agent_work_review_pending", "work_completed"],
            )
            self.assertEqual(resumed.status, "review_pending")
            # The new Attempt inherits no member state: it fetches the sources
            # again under its shared Attempt policy.
            self.assertEqual(adapter.contact_count, 3)
            events = read_evidence(journey)
            second = resumed.attempts[-1].attempt_id
            first_request = next(
                e
                for e in events
                if e["event_type"] == "ModelRequestRecorded"
                and e["payload"]["attempt_id"] == second
            )
            self.assertIn(instruction, json.dumps(first_request["payload"]))
            self.assertFalse(
                any(e["event_type"] == "InvestigationAttemptFailed" for e in events)
            )
            self.assertEqual(project_investigation(events), resumed)
            with self.assertRaisesRegex(ValueError, "exact unused authority"):
                investigation_assignment(
                    model_factory=model.model,
                    source_adapter_factory=lambda: adapter,
                ).run(
                    journey,
                    environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
                    work_review_response_event_id=answer.response_event_id,
                )

    def test_the_same_refused_call_stops_at_the_configured_failed_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _IndependentCorrectionsModel(wrong_actions=5, wrong_outputs=0)
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                adapter,
                continuation_policy={"failed_turns": 3},
            )
            self.assertEqual(
                (projection.status, projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            # The first action and exactly two corrections; the third failed turn
            # in a row stops.
            self.assertEqual(model.calls, 3)
            self.assertEqual(
                [
                    event["payload"]["errors"]
                    for event in journey.read()
                    if event["event_type"] == "ToolCallValidated"
                ],
                [["fetch_assembly_report_requires_typed_refsnp_result"]] * 3,
            )
            self.assertEqual(adapter.contact_count, 0)

    def test_changed_invalid_output_does_not_replenish_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _IndependentCorrectionsModel(wrong_actions=0, wrong_outputs=20)
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                adapter,
                continuation_policy={"failed_turns": 3},
            )
            self.assertEqual(
                (projection.status, projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(model.calls, 5)
            self.assertEqual(adapter.contact_count, 2)
            lineage = projection.attempts[0].structured_output_lineages[0]
            self.assertEqual(
                tuple(
                    item.output["one_based_position"]
                    for item in lineage.validation_attempts
                ),
                (1, 2, 3),
            )
            self.assertIsNone(lineage.accepted_output)
            self.assertEqual(project_investigation(read_evidence(journey)), projection)

    def test_different_refused_calls_are_failed_turns_in_a_row(self) -> None:
        # One rule: a different refused call spends no allowance of its own;
        # each reply in which nothing worked is the next failed turn in a row,
        # and the default's fourth stops before the valid work.
        class DifferentRefusals(_AdaptiveInvestigationModel):
            """Four different refused calls, then the scripted valid work."""

            refusals = (
                (
                    "fetch_assembly_report",
                    {
                        "assembly_accession": "GCF_000001405.40",
                        "refsnp_event_id": "missing",
                    },
                ),
                ("fetch_refsnp", {"rsid": "not-an-rsid"}),
                ("select_companion_asset", {}),
                ("search_journey", {"before_event_id": "None"}),
            )

            async def respond(self, messages, info):
                if self.calls < len(self.refusals):
                    name, arguments = self.refusals[self.calls]
                    self.calls += 1
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                name,
                                canonical_json(arguments),
                                tool_call_id=f"different-{self.calls}",
                            )
                        ]
                    )
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as temporary:
            model = DifferentRefusals()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )
            self.assertEqual(
                (projection.status, projection.attempts[-1].stop_reason),
                ("agent_work_review_pending", "failed_turns"),
            )
            self.assertEqual(model.calls, 4)
            events = read_evidence(journey)
            refused = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "singleton_refused"
            ]
            self.assertEqual(
                [turn["payload"]["correction_count"] for turn in refused], [1, 2, 4]
            )
            self.assertEqual(
                [
                    turn["payload"]["correction_key"].partition(":")[0]
                    for turn in refused
                ],
                ["fetch_assembly_report", "fetch_refsnp", "search_journey"],
            )
            # A Tool outside the role is an unoffered call: refused on its own
            # while its reply is routed, under the reply's own fault key and its
            # place in the row.
            (routed,) = [
                e["payload"] for e in events if e["event_type"] == "AgentTurnRouted"
            ]
            self.assertEqual(routed["faults"], [{"key": "unoffered_calls", "count": 3}])
            self.assertEqual(adapter.contact_count, 0)
            request = project_agent_work_review(events).pending_request
            self.assertEqual(
                (request.cause, request.source_event_id),
                ("failed_turns", refused[-1]["event_id"]),
            )
            self.assertEqual(project_investigation(events), projection)

    def test_text_ending_is_corrected_then_investigation_continues(self) -> None:
        class TextFirst(_AdaptiveInvestigationModel):
            async def respond(self, messages, info):
                if not self.calls:
                    self.calls += 1
                    return ModelResponse(parts=[TextPart("Let me think first.")])
                return await super().respond(messages, info)

        with tempfile.TemporaryDirectory() as temporary:
            model = TextFirst()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )
            self.assertEqual(projection.status, "review_pending")
            events = read_evidence(journey)
            (turn,) = [
                e["payload"]
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["disposition"] == "text_refused"
            ]
            self.assertEqual(
                (turn["correction_key"], turn["correction_count"]), ("text_ending", 1)
            )
            self.assertIn("Let me think first.", model.visible_messages[1])
            # The visible messages are serialized JSON, line breaks escaped.
            self.assertIn(json.dumps(turn["feedback"])[1:-1], model.visible_messages[1])
            self.assertEqual(project_investigation(events), projection)

    def test_text_only_output_exhaustion_stops_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _TextOnlyModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )
            self.assertEqual(
                (projection.status, projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(projection.attempts[-1].stop_reason, "failed_turns")
            # Three corrections of the text ending; the default's fourth failed
            # turn in a row stops.
            self.assertEqual(model.calls, 4)
            self.assertEqual(adapter.contact_count, 0)
            self.assertEqual(
                project_investigation(read_evidence(Journey.open(journey.path))),
                projection,
            )

    def test_review_revision_response_expresses_every_collection_operation(
        self,
    ) -> None:
        proposal_id = "investigation-proposal:" + "a" * 64
        work_item_ids = tuple("work-item:" + value * 64 for value in "bcdef")

        response = ReviseInvestigationResponse(
            action="revise",
            proposal_id=proposal_id,
            asserted_by="researcher:owner",
            operations=(
                KeepInvestigationMember(
                    operation="keep", work_item_id=work_item_ids[0]
                ),
                AddInvestigationMember(
                    operation="add",
                    request=InvestigationRequestItem(
                        request_text="Investigate rs28503599 in sample HG001.",
                        rsid="rs28503599",
                    ),
                ),
                AmendInvestigationMember(
                    operation="amend",
                    work_item_id=work_item_ids[1],
                    request=InvestigationRequestItem(
                        request_text="Investigate rs7412 with the amended scope.",
                        rsid="rs7412",
                    ),
                ),
                RemoveInvestigationMember(
                    operation="remove", work_item_id=work_item_ids[2]
                ),
                DeferInvestigationMember(
                    operation="defer", work_item_id=work_item_ids[3]
                ),
                ClarifyInvestigationMember(
                    operation="clarify",
                    work_item_id=work_item_ids[4],
                    request=InvestigationRequestItem(
                        request_text="Investigate clarified rs4149056 evidence.",
                        rsid="rs4149056",
                    ),
                ),
                AdjustInvestigationSearch(
                    operation="search_adjustment",
                    work_item_id=work_item_ids[0],
                    request=InvestigationRequestItem(
                        request_text="Investigate rs429358 with the adjusted search.",
                        rsid="rs429358",
                    ),
                ),
            ),
        )

        self.assertEqual(
            tuple(item.operation for item in response.operations),
            (
                "keep",
                "add",
                "amend",
                "remove",
                "defer",
                "clarify",
                "search_adjustment",
            ),
        )
        self.assertEqual(response.proposal_id, proposal_id)

    def test_compile_proposal_takes_the_outcome_and_its_citations_only(self) -> None:
        # The type requires what cites the source and picks the branch; code copies
        # every locus field and conflict from the cited facts.
        schema = InvestigationPlanDraft.model_json_schema()
        self.assertEqual(set(schema["required"]), {"outcome", "refsnp_event_id"})
        self.assertEqual(
            set(schema["properties"]),
            {"outcome", "refsnp_event_id", "assembly_event_id"},
        )
        InvestigationPlanDraft.model_validate_json(
            json.dumps({"outcome": "unresolved", "refsnp_event_id": "retained-ref"})
        )
        InvestigationPlanDraft.model_validate_json(
            json.dumps(
                {
                    "outcome": "complete",
                    "refsnp_event_id": "retained-ref",
                    "assembly_event_id": "retained-assembly",
                }
            )
        )

    def test_observed_tool_session_correlates_one_admitted_success(self) -> None:
        # A synthetic stage result under a registered name reads a plain view.
        self.enterContext(_plain_view("fetch_refsnp"))
        from typing import Literal

        from pydantic import BaseModel, ConfigDict

        from bio_neb.agentic_journey.action_admission import (
            ActionAdmissionSession,
            ProposedAgentAction,
        )
        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from bio_neb.agentic_journey.lifecycle import (
            ActivityLifecycle,
        )
        from bio_neb.agentic_journey.tool_execution import (
            ObservedToolSession,
            ToolExecutionAdapter,
        )

        class TypedResult(BaseModel):
            model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

            status: Literal["resolved"] = "resolved"

        class SourceRetained(BaseModel):
            model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

            event_type: Literal["SourceRetained"] = "SourceRetained"
            attempt_id: str

        class StageResultRecorded(BaseModel):
            model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

            event_type: Literal["StageResultRecorded"] = "StageResultRecorded"
            attempt_id: str
            source_event_id: str
            result: TypedResult

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary).resolve())
            seed = journey.record("TestStarted", {})
            with processing_pass(journey):
                activity = ActivityLifecycle(
                    journey, causal_head=seed["event_id"]
                ).start(
                    adapter="pydantic_ai",
                    stage="investigation_planning_attempt",
                )
                failure_source = AttemptFailureSource()
                admission = ActionAdmissionSession(
                    wording=CorrectionWording(),
                    attempt_id="attempt-191",
                    parent_activity=activity,
                    failure_source=failure_source,
                    eligible_tool_names=frozenset({"fetch_refsnp"}),
                    action_validator=lambda _action: (),
                    continuation=ContinuationPolicy(failed_turns=1),
                )
                session = ObservedToolSession(
                    attempt_id="attempt-191",
                    parent_activity=activity,
                    admission_session=admission,
                    failure_source=failure_source,
                    observed_tool_names=frozenset({"fetch_refsnp"}),
                )
                response_event_id = "00000000-0000-4000-8000-000000000191"
                disposition = session.admit_turn(
                    response_event_id,
                    (
                        ProposedAgentAction(
                            tool_name="fetch_refsnp",
                            tool_call_id="call-191",
                            arguments={"rsid": "rs429358"},
                        ),
                    ),
                    frozenset({"fetch_refsnp"}),
                )
                retained_result: dict[str, object] = {}

                artifact = b'{"refsnp_id":"429358"}'

                def operation() -> TypedResult:
                    result = TypedResult()
                    source = activity.record_with_artifact(
                        SourceRetained(attempt_id="attempt-191"),
                        artifact,
                        media_type="application/json",
                    )
                    retained_result.update(
                        activity.record(
                            StageResultRecorded(
                                attempt_id="attempt-191",
                                source_event_id=source["event_id"],
                                result=result,
                            )
                        )
                    )
                    return result

                result = session.execute(
                    "call-191",
                    operation,
                    ToolExecutionAdapter(
                        identity="offline_refsnp",
                        stage_result_event=lambda _result: retained_result,
                        artifact_digests=lambda _result: (sha256_bytes(artifact),),
                    ),
                )

                self.assertEqual(disposition, "singleton_admitted")
                # The model is shown the row's view; the fact's id rides beside it.
                self.assertEqual(
                    (result.view["status"], result.event_id),
                    ("recorded", retained_result["event_id"]),
                )
                events = read_evidence(journey)
                started = next(
                    event
                    for event in events
                    if event["event_type"] == "ObservedToolExecutionStarted"
                )
                succeeded = next(
                    event
                    for event in events
                    if event["event_type"] == "ObservedToolExecutionSucceeded"
                )
                validation = next(
                    event
                    for event in events
                    if event["event_type"] == "ToolCallValidated"
                )
                self.assertEqual(
                    started["payload"],
                    {
                        "attempt_id": "attempt-191",
                        "response_event_id": response_event_id,
                        "validation_event_id": validation["event_id"],
                        "tool_call_id": "call-191",
                        "tool_name": "fetch_refsnp",
                        "adapter_identity": "offline_refsnp",
                        # A first call continues no earlier request.
                        "continuation_request_event_id": None,
                    },
                )
                self.assertEqual(
                    succeeded["payload"]["started_event_id"], started["event_id"]
                )
                self.assertEqual(
                    succeeded["payload"]["stage_result_event_id"],
                    retained_result["event_id"],
                )
                self.assertEqual(
                    succeeded["payload"]["artifact_digests"], [sha256_bytes(artifact)]
                )

    def test_observed_tool_session_preserves_mixed_call_cardinality(self) -> None:
        from pydantic_ai import ToolFailed

        from bio_neb.agentic_journey.action_admission import (
            ActionAdmissionSession,
            ProposedAgentAction,
        )
        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from bio_neb.agentic_journey.lifecycle import (
            ActivityLifecycle,
        )
        from bio_neb.agentic_journey.tool_execution import ObservedToolSession

        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary).resolve())
            seed = journey.record("TestStarted", {})
            activity = ActivityLifecycle(journey, causal_head=seed["event_id"]).start(
                adapter="pydantic_ai",
                stage="investigation_planning_attempt",
            )
            failure_source = AttemptFailureSource()
            session = ObservedToolSession(
                attempt_id="attempt-191",
                parent_activity=activity,
                admission_session=ActionAdmissionSession(
                    wording=CorrectionWording(),
                    attempt_id="attempt-191",
                    parent_activity=activity,
                    failure_source=failure_source,
                    eligible_tool_names=frozenset({"fetch_refsnp"}),
                    action_validator=lambda _action: (),
                    continuation=ContinuationPolicy(failed_turns=1),
                ),
                failure_source=failure_source,
                observed_tool_names=frozenset({"fetch_refsnp"}),
            )

            # A multi-call response records nothing until each call's turn; each
            # call is admitted just before it runs, in the order sent.
            self.assertIsNone(
                session.admit_turn(
                    "00000000-0000-4000-8000-000000000192",
                    (
                        ProposedAgentAction(
                            tool_name="fetch_refsnp",
                            tool_call_id="call-refsnp",
                            arguments={"rsid": "rs429358"},
                        ),
                        ProposedAgentAction(
                            tool_name="fetch_assembly_report",
                            tool_call_id="call-assembly",
                            arguments={"assembly_accession": "GCF_000001405.40"},
                        ),
                    ),
                    frozenset({"fetch_refsnp", "fetch_assembly_report"}),
                )
            )
            self.assertFalse(
                any(e["event_type"] == "ToolCallValidated" for e in journey.read())
            )
            # Out of the sent order is our bug, held under its own id.
            with self.assertRaises(Invariant) as held:
                session.admit_call("call-assembly")
            self.assertEqual(held.exception.invariant_id, "multi_call_order_lost")
            self.assertEqual(session.admit_call("call-refsnp"), {"rsid": "rs429358"})
            # The refused second call is refused with its correction; the stop is
            # the loop's, not admission's.
            with self.assertRaises(ToolFailed):
                session.admit_call("call-assembly")

            events = journey.read()
            validations = [
                event for event in events if event["event_type"] == "ToolCallValidated"
            ]
            self.assertEqual(
                [event["payload"]["tool_name"] for event in validations],
                ["fetch_refsnp", "fetch_assembly_report"],
            )
            dispositions = [
                event["payload"]
                for event in events
                if event["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            self.assertEqual(
                [d["validation_event_ids"] for d in dispositions],
                [[event["event_id"]] for event in validations],
            )
            self.assertEqual(
                [d["disposition"] for d in dispositions],
                ["multi_call_admitted", "multi_call_refused"],
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionStarted"
                    for event in events
                )
            )

    def test_observed_tool_session_closes_post_start_bridge_failures(self) -> None:
        # A synthetic stage result under a registered name reads a plain view.
        self.enterContext(_plain_view("fetch_refsnp"))
        from typing import Literal

        from pydantic import BaseModel, ConfigDict

        from bio_neb.agentic_journey.action_admission import (
            ActionAdmissionSession,
            ProposedAgentAction,
        )
        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from bio_neb.agentic_journey.lifecycle import (
            ActivityLifecycle,
        )
        from bio_neb.agentic_journey.tool_execution import (
            ObservedToolSession,
            ToolExecutionAdapter,
        )

        class TypedResult(BaseModel):
            model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

            status: Literal["resolved"] = "resolved"

        class SourceRetained(BaseModel):
            model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

            event_type: Literal["SourceRetained"] = "SourceRetained"
            attempt_id: str

        class StageResultRecorded(BaseModel):
            model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

            event_type: Literal["StageResultRecorded"] = "StageResultRecorded"
            attempt_id: str
            source_event_id: str
            result: TypedResult

        cases = (
            ("stage result bridge", LookupError, False),
            ("artifact digest bridge", ArithmeticError, False),
            ("stage failure bridge", KeyError, True),
        )
        for label, expected_error, operation_fails in cases:
            with self.subTest(bridge=label):
                with tempfile.TemporaryDirectory() as temporary:
                    journey = Journey.create(Path(temporary).resolve())
                    seed = journey.record("TestStarted", {})
                    with processing_pass(journey):
                        activity = ActivityLifecycle(
                            journey, causal_head=seed["event_id"]
                        ).start(
                            adapter="pydantic_ai",
                            stage="investigation_planning_attempt",
                        )
                        failure_source = AttemptFailureSource()
                        session = ObservedToolSession(
                            attempt_id="attempt-191",
                            parent_activity=activity,
                            admission_session=ActionAdmissionSession(
                                wording=CorrectionWording(),
                                attempt_id="attempt-191",
                                parent_activity=activity,
                                failure_source=failure_source,
                                eligible_tool_names=frozenset({"fetch_refsnp"}),
                                action_validator=lambda _action: (),
                                continuation=ContinuationPolicy(failed_turns=1),
                            ),
                            failure_source=failure_source,
                            observed_tool_names=frozenset({"fetch_refsnp"}),
                        )
                        session.admit_turn(
                            "00000000-0000-4000-8000-000000000193",
                            (
                                ProposedAgentAction(
                                    tool_name="fetch_refsnp",
                                    tool_call_id="call-bridge",
                                    arguments={"rsid": "rs429358"},
                                ),
                            ),
                            frozenset({"fetch_refsnp"}),
                        )
                        retained_result: dict[str, object] = {}
                        artifact = b'{"refsnp_id":"429358"}'

                        def operation() -> TypedResult:
                            if operation_fails:  # noqa: B023
                                raise ValueError("typed operation failed")
                            result = TypedResult()
                            source = activity.record_with_artifact(  # noqa: B023
                                SourceRetained(attempt_id="attempt-191"),
                                artifact,  # noqa: B023
                                media_type="application/json",
                            )
                            retained_result.update(  # noqa: B023
                                activity.record(  # noqa: B023
                                    StageResultRecorded(
                                        attempt_id="attempt-191",
                                        source_event_id=source["event_id"],
                                        result=result,
                                    )
                                )
                            )
                            return result

                        def stage_result(_result: TypedResult) -> dict[str, object]:
                            if expected_error is LookupError:  # noqa: B023
                                raise LookupError("stage result bridge failed")
                            return retained_result  # noqa: B023

                        def artifact_digests(_result: TypedResult) -> tuple[str, ...]:
                            if expected_error is ArithmeticError:  # noqa: B023
                                raise ArithmeticError("artifact digest bridge failed")
                            return (sha256_bytes(artifact),)  # noqa: B023

                        def stage_failure(_error: BaseException) -> None:
                            if expected_error is KeyError:  # noqa: B023
                                raise KeyError("stage failure bridge failed")

                        with self.assertRaises(expected_error):
                            session.execute(
                                "call-bridge",
                                operation,
                                ToolExecutionAdapter(
                                    identity="offline_refsnp",
                                    stage_result_event=stage_result,
                                    artifact_digests=artifact_digests,
                                    stage_failure_event=stage_failure,
                                ),
                            )

                        lifecycle = [
                            event
                            for event in read_evidence(journey)
                            if event["event_type"].startswith("ObservedToolExecution")
                        ]
                        self.assertEqual(
                            [event["event_type"] for event in lifecycle],
                            [
                                "ObservedToolExecutionStarted",
                                "ObservedToolExecutionFailed",
                            ],
                        )
                        self.assertEqual(
                            lifecycle[1]["payload"]["started_event_id"],
                            lifecycle[0]["event_id"],
                        )
                        self.assertEqual(
                            lifecycle[1]["payload"]["mechanical_error_type"],
                            expected_error.__name__,
                        )
                        self.assertIsNone(
                            lifecycle[1]["payload"]["stage_failure_event_id"]
                        )

    def test_refsnp_normalizer_accepts_live_seq_id_traits_metadata(self) -> None:
        source = RetrievedSource.create(
            operation="fetch_refsnp",
            request_identity="https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358",
            status_code=200,
            observed_at="2026-08-27T15:19:37.982374Z",
            elapsed_time_ms=994,
            raw_response=REFSNP_BODY,
        )

        result = normalize_refsnp(
            rsid="rs429358",
            assembly_accession="GCF_000001405.40",
            source=source,
            raw_response=REFSNP_BODY,
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual(
            [candidate.model_dump(mode="json") for candidate in result.candidates],
            [
                {
                    "assembly_accession": "GCF_000001405.40",
                    "assembly_name": "GRCh38.p14",
                    "contig": "NC_000019.10",
                    "one_based_position": 44908684,
                    "ref": "T",
                    "alt": "C",
                    "aliases": ["rs429358"],
                }
            ],
        )

    def test_refsnp_adapter_uses_a7_accepted_get_once_without_query(self) -> None:
        transport = _RecordingMetadataTransport()
        adapter = NcbiInvestigationSourceAdapter(transport)

        response = adapter.fetch_refsnp("rs429358")

        expected_url = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358"
        self.assertEqual([request[0] for request in transport.requests], [expected_url])
        self.assertEqual(transport.contact_count, 1)
        self.assertEqual(response.request_identity, expected_url)

    def test_refsnp_adapter_retains_404_body_as_dbsnps_answer(self) -> None:
        raw_response = b'{"error":"rsID not found"}'
        opener = _HttpErrorOpener(status_code=404, body=raw_response)
        with patch(
            "bio_neb.providers.ncbi_datasets.urllib.request.build_opener",
            return_value=opener,
        ):
            adapter = NcbiInvestigationSourceAdapter()
            response = adapter.fetch_refsnp("rs429358")

        expected_url = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358"
        self.assertEqual(opener.requests, [(expected_url, "GET", 15.0)])
        self.assertEqual(adapter.contact_count, 1)
        self.assertEqual(response.request_identity, expected_url)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.body, raw_response)
        source = RetrievedSource.create(
            operation="fetch_refsnp",
            request_identity=response.request_identity,
            status_code=response.status_code,
            observed_at=response.observed_at,
            elapsed_time_ms=response.elapsed_time_ms,
            raw_response=response.body,
        )

        result = normalize_refsnp(
            rsid="rs429358",
            assembly_accession="GCF_000001405.40",
            source=source,
            raw_response=response.body,
        )

        # dbSNP's 404 is its own answer, with no verdict (ADR 0049).
        self.assertEqual(result.status, "source_answered")
        self.assertEqual(result.source, source)

    def test_configured_planner_inherits_resolved_model_maximum(self) -> None:
        from bio_neb.agentic_journey.attempt_journey import AttemptFailureSource
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.lifecycle import (
            ActivityLifecycle,
        )
        from bio_neb.agentic_journey.model_observation import JourneyObservedModel
        from bio_neb.agentic_journey.role_instructions import role_instructions

        runtime = RuntimeConfiguration.model_validate(runtime_configuration())
        policy = runtime.agent_invocation_policy(
            "investigation_planner", AgentInvocationOverrides()
        )
        adapter = NebiusModelEndpointAdapter(
            LoadedRuntimeConfig(
                selections=runtime,
                credential="dummy-key",
                invocation_policy=policy,
                model=model_endpoint_policy(policy.settings_selection.model),
            ),
            runtime.model_for_policy(policy),
        )
        investigation_input, assertion = _input_and_assertion()
        request = InvestigationRequestItem(
            request_text=assertion.request_text, rsid=assertion.rsid
        )
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary).resolve())
            seed = journey.record("TestStarted", {})
            activity = ActivityLifecycle(journey, causal_head=seed["event_id"]).start(
                adapter="pydantic_ai",
                stage="investigation_planning_attempt",
            )
            failure_source = AttemptFailureSource()
            attempt_id = str(uuid.uuid4())
            # The planner is assembled over the observed model, as an Attempt does.
            observed_model = JourneyObservedModel(
                adapter,
                parent_activity=activity,
                attempt_id=attempt_id,
                failure_source=failure_source,
                nebius_model_endpoint=adapter.nebius_model_endpoint.model_dump(),
                settings_selection=policy.settings_selection,
                agent_role="investigation_planner",
                contact_policy=policy.model_contact,
            )
            planner, _deps = assemble_investigation_planner(
                observed_model,
                instructions=role_instructions("investigation_planner"),
                policy=policy,
                model_policy=adapter.endpoint_policy,
                journey=journey,
                attempt_id=attempt_id,
                parent_activity=activity,
                failure_source=failure_source,
                dependencies=lambda observed: InvestigationPlannerDependencies(
                    parent_activity=activity,
                    attempt_id=attempt_id,
                    investigation_input=investigation_input,
                    assertion=assertion,
                    request=request,
                    collection_member_id=investigation_output_member_id(request),
                    revision_number=1,
                    supersedes_proposal_id=None,
                    adapter=_SourceResponsesOnly(),
                    failure_source=failure_source,
                    observed_tool_session=observed,
                ),
            )

        settings, _ = adapter.prepare_request(
            planner.model_settings,
            ModelRequestParameters(
                function_tools=[],
                output_tools=[],
                allow_text_output=False,
                output_mode="tool",
            ),
        )

        self.assertNotIn("max_tokens", planner.model_settings or {})
        self.assertEqual(settings, {"parallel_tool_calls": False, "max_tokens": 245760})

    def test_partial_start_reopens_as_failed_instead_of_not_started(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journey = Journey.create(Path(temporary).resolve())
            investigation_input, assertion = _input_and_assertion()
            started = InvestigationStarted(
                continuation_id="scientific-continuation:" + "b" * 64,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            journey.record(
                started.event_type,
                started.model_dump(mode="json", exclude={"event_type"}),
            )

            projection = project_investigation(Journey.open(journey.path).read())

            self.assertEqual(projection.status, "failed")
            self.assertEqual(projection.stop_reason, "attempt_not_bound")
            self.assertEqual(projection.investigation_input, investigation_input)
            self.assertEqual(projection.assertions, (assertion,))

    def test_recorded_sources_compile_exact_a7_one_item_proposal(self) -> None:
        refsnp_source = RetrievedSource.create(
            operation="fetch_refsnp",
            request_identity="https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358",
            status_code=200,
            observed_at="2026-08-26T00:00:00Z",
            elapsed_time_ms=11,
            raw_response=REFSNP_BODY,
        )
        assembly_source = RetrievedSource.create(
            operation="fetch_assembly_report",
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                "GCF_000001405.40/dataset_report"
            ),
            status_code=200,
            observed_at="2026-08-26T00:00:01Z",
            elapsed_time_ms=9,
            raw_response=ASSEMBLY_BODY,
        )
        refsnp = normalize_refsnp(
            rsid="rs429358",
            assembly_accession="GCF_000001405.40",
            source=refsnp_source,
            raw_response=REFSNP_BODY,
        )
        assembly = normalize_assembly_report(
            expected_assembly_accession="GCF_000001405.40",
            expected_assembly_name="GRCh38.p14",
            source=assembly_source,
            raw_response=ASSEMBLY_BODY,
        )
        investigation_input = InvestigationInput.create(
            vcf=InvestigationVcfIdentity.create(
                bucket="accepted-run-bucket",
                object_key="outputs/HG001.vcf.gz",
                byte_count=1234,
                sha256="a" * 64,
            ),
            sample_id="HG001",
            assembly_accession="GCF_000001405.40",
        )
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs429358 in sample HG001.",
            rsid="rs429358",
            predecessor_ids=(investigation_input.input_id,),
        )

        proposal = compile_investigation_proposal(
            investigation_input=investigation_input,
            assertion=assertion,
            refsnp=refsnp,
            assembly_report=assembly,
            revision_number=1,
            supersedes_proposal_id=None,
            predecessor_ids=(refsnp.result_id, assembly.result_id),
        )

        self.assertEqual(len(proposal.work_items), 1)
        item = proposal.work_items[0]
        for value, identity_field, namespace, names in (
            (
                investigation_input.vcf,
                "vcf_id",
                "investigation-vcf:",
                "bucket object_key byte_count sha256",
            ),
            (
                investigation_input,
                "input_id",
                "investigation-input:",
                "vcf sample_id assembly_accession",
            ),
            (
                assertion,
                "assertion_id",
                "user-assertion:",
                "asserted_by request_text rsid predecessor_ids scientifically_verified",
            ),
            (
                refsnp_source,
                "source_id",
                "retrieved-source:",
                "operation authority request_identity status_code observed_at elapsed_time_ms raw_artifact_id raw_response_sha256 raw_response_byte_count",
            ),
            (
                assembly_source,
                "source_id",
                "retrieved-source:",
                "operation authority request_identity status_code observed_at elapsed_time_ms raw_artifact_id raw_response_sha256 raw_response_byte_count",
            ),
            (
                refsnp.assessment,
                "assessment_id",
                "evidence-assessment:",
                "source_id verdict rationale",
            ),
            (
                assembly.assessment,
                "assessment_id",
                "evidence-assessment:",
                "source_id verdict rationale",
            ),
            (
                refsnp,
                "result_id",
                "refsnp-result:",
                # Every lookup result names its source; dbSNP's says EVA does
                # not apply. An rsID lookup says no position applies, and every
                # placement is copied with the record's state.
                "requested_rsid verified_rsid expected_assembly_accession status candidates conflicts placements record_state source assessment lookup_source eva location",
            ),
            (
                assembly,
                "result_id",
                "assembly-report-result:",
                # The read report's NCBI status is part of the identity.
                "expected_assembly_accession expected_assembly_name status observed_assembly_accession observed_assembly_name observed_assembly_status conflicts source assessment",
            ),
            (
                item.target,
                "target_id",
                "variant-target:",
                "assembly_accession assembly_name contig one_based_position ref alt aliases assessment_ids",
            ),
            (
                proposal,
                "proposal_id",
                "investigation-proposal:",
                "revision_number investigation_input_id assertion_id work_items request_members unresolved_conflicts completeness supersedes_proposal_id predecessor_ids",
            ),
        ):
            with self.subTest(identity_namespace=namespace):
                assert_json_identity(
                    self, getattr(value, identity_field), value, names, namespace
                )
        self.assertEqual(
            item.work_item_id,
            "work-item:"
            + expected_digest(
                {
                    "assembly_accession": "GCF_000001405.40",
                    "contig": "NC_000019.10",
                    "one_based_position": 44908684,
                    "ref": "T",
                    "alt": "C",
                    "normalized_question_action": "observe_variant",
                }
            ),
        )
        for ordinal, member in enumerate(proposal.request_members):
            output_member_id = "request-member:" + expected_digest(
                {
                    "request": member.request.model_dump(mode="json"),
                    "ordinal": ordinal,
                }
            )
            self.assertEqual(
                member.request_member_id,
                "request-member:"
                + expected_digest(
                    {
                        "assertion_id": assertion.assertion_id,
                        "output_member_id": output_member_id,
                    }
                ),
            )
        self.assertEqual(
            refsnp_source.raw_response_sha256, hashlib.sha256(REFSNP_BODY).hexdigest()
        )
        self.assertEqual(
            assembly_source.raw_response_sha256,
            hashlib.sha256(ASSEMBLY_BODY).hexdigest(),
        )
        self.assertEqual(item.sample_id, "HG001")
        self.assertEqual(item.intended_action, "observe_variant")
        self.assertEqual(
            item.target.model_dump(mode="json"),
            {
                "target_id": item.target.target_id,
                "assembly_accession": "GCF_000001405.40",
                "assembly_name": "GRCh38.p14",
                "contig": "NC_000019.10",
                "one_based_position": 44908684,
                "ref": "T",
                "alt": "C",
                "aliases": ["rs429358"],
                "assessment_ids": [
                    refsnp.assessment.assessment_id,
                    assembly.assessment.assessment_id,
                ],
            },
        )
        self.assertEqual(proposal.completeness, "complete")
        self.assertEqual(proposal.unresolved_conflicts, ())
        work_items_schema = InvestigationProposal.model_json_schema()["properties"][
            "work_items"
        ]
        self.assertEqual(work_items_schema["type"], "array")
        self.assertNotIn("minItems", work_items_schema)
        self.assertNotIn("maxItems", work_items_schema)

    def test_real_agent_receives_each_typed_tool_result_before_next_action(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _AdaptiveInvestigationModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )

            self.assertEqual(projection.status, "review_pending")
            self.assertEqual(model.calls, 3)
            self.assertEqual(
                adapter.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            self.assertNotIn("refsnp-result:", model.visible_messages[0])
            self.assertIn("refsnp-result:", model.visible_messages[1])
            self.assertNotIn("assembly-report-result:", model.visible_messages[1])
            self.assertIn("refsnp-result:", model.visible_messages[2])
            self.assertIn("assembly-report-result:", model.visible_messages[2])
            self.assertEqual(projection.model_contact_count, 3)
            self.assertEqual(projection.source_contact_count, 2)
            self.assertEqual(len(projection.proposals[0].work_items), 1)
            self.assertEqual(
                projection.proposals[0].work_items[0].target.one_based_position,
                44908684,
            )
            source_events = [
                event
                for event in journey.read()
                if event["event_type"] == "InvestigationSourceRetrieved"
            ]
            self.assertEqual(len(source_events), 2)
            for event, source in zip(source_events, projection.sources, strict=True):
                reference = event["artifact_references"][0]
                self.assertEqual(reference["artifact_id"], source.raw_artifact_id)
                self.assertEqual(
                    len(journey.read_artifact(reference)),
                    source.raw_response_byte_count,
                )
            retained = canonical_json(projection.model_dump(mode="json"))
            self.assertNotIn("secret-value-not-retained", retained)
            self.assertNotIn("vcf_observation", canonical_json(adapter.operations))

    def test_exact_request_disambiguates_multiallelic_refsnp_candidate(self) -> None:
        multiallelic = json.loads(REFSNP_BODY)
        multiallelic["primary_snapshot_data"]["placements_with_allele"][0][
            "alleles"
        ].append(
            {
                "allele": {
                    "spdi": {
                        "seq_id": "NC_000019.10",
                        "position": 44908683,
                        "deleted_sequence": "T",
                        "inserted_sequence": "G",
                    }
                }
            }
        )
        investigation_input, _assertion = _input_and_assertion()
        request = InvestigationRequestItem(
            request_text=(
                "For rs429358 on GCF_000001405.40 / GRCh38.p14 in sample "
                "HG001, observe the exact allele NC_000019.10:44908684 T>C."
            ),
            rsid="rs429358",
        )
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=(request,),
            predecessor_ids=(investigation_input.input_id,),
        )

        with tempfile.TemporaryDirectory() as temporary:
            model = _AdaptiveInvestigationModel()
            adapter = _SourceResponsesOnly(
                refsnp_body=json.dumps(multiallelic).encode()
            )
            projection, _journey = _run_attempt(
                Path(temporary).resolve(),
                model,
                adapter,
                investigation_input=investigation_input,
                assertion=assertion,
            )

        self.assertEqual(projection.status, "review_pending")
        self.assertEqual(
            adapter.operations,
            [
                "fetch_refsnp:rs429358",
                "fetch_assembly_report:GCF_000001405.40",
            ],
        )
        target = projection.proposals[0].work_items[0].target
        self.assertEqual(
            (
                target.contig,
                target.one_based_position,
                target.ref,
                target.alt,
            ),
            ("NC_000019.10", 44908684, "T", "C"),
        )

    # Known pre-existing length; see the B023 note above.
    def test_observed_fetch_refsnp_correlates_success_negative_refusal_failure_and_no_progress(  # noqa: PLR0915 - 122 statements; see above
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _AdaptiveInvestigationModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )

            self.assertEqual(projection.status, "review_pending")
            events = read_evidence(journey)
            refsnp_result = next(
                event
                for event in events
                if event["event_type"] == "RefSnpToolResultRecorded"
            )
            validation = next(
                event
                for event in events
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["tool_name"] == "fetch_refsnp"
            )
            disposition = next(
                event
                for event in events
                if event["event_type"] == "AgentActionTurnDispositionRecorded"
                and event["payload"]["validation_event_ids"] == [validation["event_id"]]
            )
            response = next(
                event
                for event in events
                if event["event_id"] == disposition["payload"]["response_event_id"]
            )
            started = next(
                event
                for event in events
                if event["event_type"] == "ObservedToolExecutionStarted"
            )
            succeeded = next(
                event
                for event in events
                if event["event_type"] == "ObservedToolExecutionSucceeded"
            )
            self.assertEqual(validation["payload"]["status"], "valid")
            self.assertEqual(validation["payload"]["tool_call_id"], "investigation-1")
            self.assertEqual(
                disposition["payload"]["disposition"], "singleton_admitted"
            )
            self.assertEqual(
                started["payload"],
                {
                    "attempt_id": validation["payload"]["attempt_id"],
                    "response_event_id": response["event_id"],
                    "validation_event_id": validation["event_id"],
                    "tool_call_id": "investigation-1",
                    "tool_name": "fetch_refsnp",
                    "adapter_identity": "ncbi_investigation",
                    # A first call continues no earlier request.
                    "continuation_request_event_id": None,
                },
            )
            self.assertEqual(
                succeeded["payload"]["stage_result_event_id"],
                refsnp_result["event_id"],
            )
            self.assertEqual(
                succeeded["payload"]["artifact_digests"],
                [refsnp_result["payload"]["result"]["source"]["raw_response_sha256"]],
            )
            self.assertLess(events.index(response), events.index(validation))
            self.assertLess(events.index(validation), events.index(started))
            self.assertLess(events.index(started), events.index(refsnp_result))
            self.assertLess(events.index(refsnp_result), events.index(succeeded))
            next_request = next(
                event
                for event in events[events.index(succeeded) + 1 :]
                if event["event_type"] == "ModelRequestRecorded"
            )
            visible_returns = [
                part
                for message in next_request["payload"]["compatible_adapter_evidence"][
                    "request"
                ]["messages"]
                for part in message["parts"]
                if part.get("part_kind") == "tool-return"
                and part.get("tool_call_id") == "investigation-1"
            ]
            self.assertEqual(len(visible_returns), 1)
            self.assertEqual(
                hashlib.sha256(
                    canonical_json(visible_returns[0]["content"]).encode()
                ).hexdigest(),
                succeeded["payload"]["model_visible_result_digest"],
            )
            self.assertEqual(
                [
                    event["event_type"]
                    for event in events
                    if event["event_id"] == started["event_id"]
                    or event["payload"].get("started_event_id") == started["event_id"]
                ],
                ["ObservedToolExecutionStarted", "ObservedToolExecutionSucceeded"],
            )

            hostile_reopens = []

            hostile_adapter = copy.deepcopy(events)
            next(
                event
                for event in hostile_adapter
                if event["event_type"] == "ObservedToolExecutionStarted"
            )["payload"]["adapter_identity"] = "hostile-adapter"
            hostile_reopens.append(("adapter identity", hostile_adapter))

            hostile_response = copy.deepcopy(events)
            next(
                event
                for event in hostile_response
                if event["event_type"] == "ObservedToolExecutionStarted"
            )["payload"]["response_event_id"] = validation["event_id"]
            hostile_reopens.append(("response identity", hostile_response))

            hostile_validation = copy.deepcopy(events)
            next(
                event
                for event in hostile_validation
                if event["event_type"] == "ObservedToolExecutionStarted"
            )["payload"]["validation_event_id"] = response["event_id"]
            hostile_reopens.append(("validation identity", hostile_validation))

            hostile_stage_result = copy.deepcopy(events)
            next(
                event
                for event in hostile_stage_result
                if event["event_type"] == "ObservedToolExecutionSucceeded"
            )["payload"]["stage_result_event_id"] = validation["event_id"]
            hostile_reopens.append(("stage result", hostile_stage_result))

            hostile_digest = copy.deepcopy(events)
            next(
                event
                for event in hostile_digest
                if event["event_type"] == "ObservedToolExecutionSucceeded"
            )["payload"]["model_visible_result_digest"] = "0" * 64
            hostile_reopens.append(("model-visible digest", hostile_digest))

            hostile_artifact = copy.deepcopy(events)
            next(
                event
                for event in hostile_artifact
                if event["event_type"] == "ObservedToolExecutionSucceeded"
            )["payload"]["artifact_digests"] = ["0" * 64]
            hostile_reopens.append(("artifact digest", hostile_artifact))

            hostile_missing_artifact = copy.deepcopy(events)
            next(
                event
                for event in hostile_missing_artifact
                if event["event_type"] == "ObservedToolExecutionSucceeded"
            )["payload"]["artifact_digests"] = []
            hostile_reopens.append(
                ("missing artifact digest", hostile_missing_artifact)
            )

            hostile_return = copy.deepcopy(events)
            hostile_request = next(
                event
                for event in hostile_return
                if event["event_id"] == next_request["event_id"]
            )
            hostile_part = next(
                part
                for message in hostile_request["payload"][
                    "compatible_adapter_evidence"
                ]["request"]["messages"]
                for part in message["parts"]
                if part.get("part_kind") == "tool-return"
                and part.get("tool_call_id") == "investigation-1"
            )
            hostile_part["content"]["status"] = "malformed"
            hostile_reopens.append(("same-call return", hostile_return))

            hostile_missing_terminal = copy.deepcopy(events)
            hostile_missing_terminal = [
                event
                for event in hostile_missing_terminal
                if event["event_type"] != "ObservedToolExecutionSucceeded"
            ]
            hostile_reopens.append(("missing terminal", hostile_missing_terminal))

            hostile_duplicate_terminal = copy.deepcopy(events)
            hostile_duplicate_terminal.append(copy.deepcopy(succeeded))
            hostile_reopens.append(("duplicate terminal", hostile_duplicate_terminal))

            for label, hostile in hostile_reopens:
                with self.subTest(hostile_reopen=label):
                    with self.assertRaises(ValueError):
                        project_investigation(hostile)

            failed_projection, failed_journey = _run_attempt(
                Path(temporary).resolve() / "source-failure",
                _AdaptiveInvestigationModel(),
                _FailingSourceResponses(),
            )
            # The exhausted retry window holds the member for its operational
            # Review instead of failing it.
            self.assertEqual(
                (failed_projection.status, failed_projection.stop_reason),
                ("operational_review_pending", None),
            )
            failed_events = read_evidence(failed_journey)
            source_failure = next(
                event
                for event in failed_events
                if event["event_type"] == "InvestigationSourceContactFailed"
            )
            failure_start = next(
                event
                for event in failed_events
                if event["event_type"] == "ObservedToolExecutionStarted"
            )
            execution_failure = next(
                event
                for event in failed_events
                if event["event_type"] == "ObservedToolExecutionFailed"
            )
            self.assertEqual(
                execution_failure["payload"],
                {
                    "attempt_id": failure_start["payload"]["attempt_id"],
                    "started_event_id": failure_start["event_id"],
                    "mechanical_error_type": "ReferenceAdapterError",
                    "stage_failure_event_id": source_failure["event_id"],
                },
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionSucceeded"
                    for event in failed_events
                )
            )

            hostile_failure_type = copy.deepcopy(failed_events)
            source_contact_start = next(
                event
                for event in hostile_failure_type
                if event["event_type"] == "InvestigationSourceContactStarted"
            )
            next(
                event
                for event in hostile_failure_type
                if event["event_type"] == "ObservedToolExecutionFailed"
            )["payload"]["stage_failure_event_id"] = source_contact_start["event_id"]
            with self.assertRaises(ValueError):
                project_investigation(hostile_failure_type)

            hostile_failure_error = copy.deepcopy(failed_events)
            next(
                event
                for event in hostile_failure_error
                if event["event_type"] == "ObservedToolExecutionFailed"
            )["payload"]["mechanical_error_type"] = "TimeoutError"
            with self.assertRaisesRegex(ValueError, "observed Tool execution"):
                project_investigation(hostile_failure_error)

            # A malformed adapter return is checked inside the source contact: the
            # contact fails with the unclassified error, the Attempt records its
            # terminal, and the error propagates for the coordinator's hold
            # instead of a retry or a member Review.
            mechanical_root = Path(temporary).resolve() / "mechanical-failure"
            with self.assertRaises(AttributeError):
                _run_attempt(
                    mechanical_root,
                    _AdaptiveInvestigationModel(),
                    _MalformedSourceResponse(),
                )
            (mechanical_file,) = mechanical_root.rglob("journey.jsonl")
            mechanical_events = read_evidence(Journey.open(mechanical_file))
            mechanical_failure = next(
                event
                for event in mechanical_events
                if event["event_type"] == "ObservedToolExecutionFailed"
            )
            mechanical_source_failure = next(
                event
                for event in mechanical_events
                if event["event_type"] == "InvestigationSourceContactFailed"
            )
            self.assertEqual(
                (
                    mechanical_failure["payload"]["mechanical_error_type"],
                    mechanical_source_failure["payload"]["error_type"],
                ),
                ("AttributeError", "AttributeError"),
            )
            self.assertEqual(
                mechanical_failure["payload"]["stage_failure_event_id"],
                mechanical_source_failure["event_id"],
            )
            mechanical_terminal = next(
                event
                for event in mechanical_events
                if event["event_type"] == "InvestigationAttemptFailed"
            )
            self.assertEqual(
                (
                    mechanical_terminal["payload"]["code"],
                    mechanical_terminal["payload"]["source_event_id"],
                ),
                ("source_contact_failed", mechanical_source_failure["event_id"]),
            )
            # A fault after the source result fails the Tool mechanically with no
            # stage failure. The model is shown it once; the same fault again
            # holds, and the Attempt terminal names that journal failure.
            anchored_root = Path(temporary).resolve() / "mechanical-after-result"
            with (
                patch(
                    "bio_neb.investigation.investigation_agent.normalize_refsnp",
                    side_effect=AttributeError("controlled normalizer fault"),
                ),
                self.assertRaisesRegex(Invariant, "fetch_refsnp: AttributeError"),
            ):
                _run_attempt(
                    anchored_root,
                    _AdaptiveInvestigationModel(),
                    _SourceResponsesOnly(),
                )
            (anchored_file,) = anchored_root.rglob("journey.jsonl")
            anchored_events = read_evidence(Journey.open(anchored_file))
            shown_failure, anchored_failure = (
                event
                for event in anchored_events
                if event["event_type"] == "ObservedToolExecutionFailed"
            )
            (shown,) = (
                event
                for event in anchored_events
                if event["event_type"] == "ToolExecutionRefused"
            )
            self.assertEqual(
                shown["payload"]["failed_event_id"], shown_failure["event_id"]
            )
            for failure in (shown_failure, anchored_failure):
                self.assertEqual(
                    (
                        failure["payload"]["mechanical_error_type"],
                        failure["payload"]["stage_failure_event_id"],
                    ),
                    ("AttributeError", None),
                )
            anchored_terminal = next(
                event
                for event in anchored_events
                if event["event_type"] == "InvestigationAttemptFailed"
            )
            self.assertEqual(
                (
                    anchored_terminal["payload"]["code"],
                    anchored_terminal["payload"]["source_event_id"],
                ),
                ("agent_run_failed", anchored_failure["event_id"]),
            )
            # The journal failure is bound by its anchor, never as a parent.
            for hostile_parent in (
                next(
                    event["event_id"]
                    for event in anchored_events
                    if event["event_type"] == "InvestigationStarted"
                ),
                anchored_failure["event_id"],
            ):
                hostile = [dict(event) for event in anchored_events]
                next(
                    event
                    for event in hostile
                    if event["event_type"] == "InvestigationAttemptFailed"
                )["parent_ids"] = [hostile_parent]
                with self.assertRaisesRegex(
                    ValueError,
                    "Attempt failure changed its exact mechanical Tool source",
                ):
                    project_investigation(hostile)

            negative_adapter = _NegativeSourceResponses()
            negative_projection, negative_journey = _run_attempt(
                Path(temporary).resolve() / "typed-negative",
                _AdaptiveInvestigationModel(),
                negative_adapter,
            )
            self.assertEqual(negative_projection.status, "review_pending")
            # The negative finding is the member's keyed issue.
            self.assertTrue(
                negative_projection.pending_request.question.issue.conflicts
            )
            self.assertEqual(negative_adapter.operations, ["fetch_refsnp:rs429358"])
            negative_events = read_evidence(negative_journey)
            negative_result = next(
                event
                for event in negative_events
                if event["event_type"] == "RefSnpToolResultRecorded"
            )
            negative_success = next(
                event
                for event in negative_events
                if event["event_type"] == "ObservedToolExecutionSucceeded"
            )
            self.assertEqual(
                negative_result["payload"]["result"]["status"], "source_answered"
            )
            self.assertEqual(
                negative_success["payload"]["stage_result_event_id"],
                negative_result["event_id"],
            )

            refused_adapter = _NegativeSourceResponses()
            refused_projection, refused_journey = _run_attempt(
                Path(temporary).resolve() / "pre-contact-refusal",
                _RefusedThenAdaptiveInvestigationModel(),
                refused_adapter,
            )
            self.assertEqual(refused_projection.status, "review_pending")
            self.assertEqual(refused_adapter.operations[0], "fetch_refsnp:rs429358")
            refused_events = read_evidence(refused_journey)
            refused_validation = next(
                event
                for event in refused_events
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["tool_call_id"] == "refused-1"
            )
            self.assertEqual(refused_validation["payload"]["status"], "invalid")
            refused_disposition = next(
                event
                for event in refused_events
                if event["event_type"] == "AgentActionTurnDispositionRecorded"
                and event["payload"]["validation_event_ids"]
                == [refused_validation["event_id"]]
            )
            self.assertEqual(
                refused_disposition["payload"]["disposition"], "singleton_refused"
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionStarted"
                    and event["payload"]["tool_call_id"] == "refused-1"
                    for event in refused_events
                )
            )

            repeated_adapter = _SourceResponsesOnly()
            repeated_projection, repeated_journey = _run_attempt(
                Path(temporary).resolve() / "no-progress",
                _RepeatedActionModel(),
                repeated_adapter,
            )
            self.assertEqual(
                (repeated_projection.status, repeated_projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(repeated_adapter.operations, ["fetch_refsnp:rs429358"])
            repeated_events = read_evidence(repeated_journey)
            # The refused repeat is corrected three times; the default's fourth
            # failed turn in a row stops.
            repeated_validations = [
                event
                for event in repeated_events
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["status"] == "invalid"
            ]
            self.assertEqual(
                [event["payload"]["tool_call_id"] for event in repeated_validations],
                [f"repeated-{call}" for call in range(2, 6)],
            )
            repeated_turns = [
                event
                for event in repeated_events
                if event["event_type"] == "AgentActionTurnDispositionRecorded"
                and event["payload"]["disposition"] == "singleton_refused"
            ]
            repeated_request = project_agent_work_review(
                read_evidence(repeated_journey)
            ).pending_request
            self.assertEqual(
                (repeated_request.cause, repeated_request.source_event_id),
                ("failed_turns", repeated_turns[-1]["event_id"]),
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionStarted"
                    and event["payload"]["tool_call_id"] != "repeated-1"
                    for event in repeated_events
                )
            )

    # Known pre-existing length; see the B023 note above.
    def test_observed_fetch_assembly_report_correlates_success_negative_refusal_failure_and_no_progress(  # noqa: PLR0915 - 137 statements; see above
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _AdaptiveInvestigationModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )

            self.assertEqual(projection.status, "review_pending")
            events = read_evidence(journey)
            assembly_result = next(
                event
                for event in events
                if event["event_type"] == "AssemblyReportToolResultRecorded"
            )
            validation = next(
                event
                for event in events
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["tool_name"] == "fetch_assembly_report"
            )
            started = next(
                event
                for event in events
                if event["event_type"] == "ObservedToolExecutionStarted"
                and event["payload"]["tool_name"] == "fetch_assembly_report"
            )
            succeeded = next(
                event
                for event in events
                if event["event_type"] == "ObservedToolExecutionSucceeded"
                and event["payload"]["stage_result_event_id"]
                == assembly_result["event_id"]
            )
            self.assertEqual(validation["payload"]["status"], "valid")
            self.assertEqual(validation["payload"]["tool_call_id"], "investigation-2")
            self.assertEqual(
                started["payload"]["validation_event_id"], validation["event_id"]
            )
            self.assertEqual(started["payload"]["tool_call_id"], "investigation-2")
            self.assertEqual(
                started["payload"]["adapter_identity"], "ncbi_investigation"
            )
            self.assertEqual(
                succeeded["payload"]["artifact_digests"],
                [assembly_result["payload"]["result"]["source"]["raw_response_sha256"]],
            )
            self.assertLess(events.index(validation), events.index(started))
            self.assertLess(events.index(started), events.index(assembly_result))
            self.assertLess(events.index(assembly_result), events.index(succeeded))
            next_request = next(
                event
                for event in events[events.index(succeeded) + 1 :]
                if event["event_type"] == "ModelRequestRecorded"
            )
            visible_returns = [
                part
                for message in next_request["payload"]["compatible_adapter_evidence"][
                    "request"
                ]["messages"]
                for part in message["parts"]
                if part.get("part_kind") == "tool-return"
                and part.get("tool_call_id") == "investigation-2"
            ]
            self.assertEqual(len(visible_returns), 1)
            self.assertEqual(
                hashlib.sha256(
                    canonical_json(visible_returns[0]["content"]).encode()
                ).hexdigest(),
                succeeded["payload"]["model_visible_result_digest"],
            )

            hostile_reopens = []

            hostile_adapter = copy.deepcopy(events)
            next(
                event
                for event in hostile_adapter
                if event["event_id"] == started["event_id"]
            )["payload"]["adapter_identity"] = "hostile-assembly-adapter"
            hostile_reopens.append(("adapter identity", hostile_adapter))

            hostile_stage_result = copy.deepcopy(events)
            next(
                event
                for event in hostile_stage_result
                if event["event_id"] == succeeded["event_id"]
            )["payload"]["stage_result_event_id"] = validation["event_id"]
            hostile_reopens.append(("stage result", hostile_stage_result))

            hostile_artifact = copy.deepcopy(events)
            next(
                event
                for event in hostile_artifact
                if event["event_id"] == succeeded["event_id"]
            )["payload"]["artifact_digests"] = ["0" * 64]
            hostile_reopens.append(("artifact digest", hostile_artifact))

            hostile_return = copy.deepcopy(events)
            hostile_request = next(
                event
                for event in hostile_return
                if event["event_id"] == next_request["event_id"]
            )
            hostile_part = next(
                part
                for message in hostile_request["payload"][
                    "compatible_adapter_evidence"
                ]["request"]["messages"]
                for part in message["parts"]
                if part.get("part_kind") == "tool-return"
                and part.get("tool_call_id") == "investigation-2"
            )
            hostile_part["content"]["status"] = "malformed"
            hostile_reopens.append(("same-call return", hostile_return))

            hostile_missing_terminal = [
                event
                for event in copy.deepcopy(events)
                if event["event_id"] != succeeded["event_id"]
            ]
            hostile_reopens.append(("missing terminal", hostile_missing_terminal))

            hostile_duplicate_terminal = copy.deepcopy(events)
            hostile_duplicate_terminal.append(copy.deepcopy(succeeded))
            hostile_reopens.append(("duplicate terminal", hostile_duplicate_terminal))

            hostile_missing_request = [
                event
                for event in copy.deepcopy(events)
                if event["event_id"] != next_request["event_id"]
            ]
            hostile_reopens.append(
                ("missing continued request", hostile_missing_request)
            )

            operations_before_reopen = list(adapter.operations)
            for label, hostile in hostile_reopens:
                with self.subTest(hostile_assembly_reopen=label):
                    with self.assertRaises(ValueError):
                        project_investigation(hostile)
            self.assertEqual(adapter.operations, operations_before_reopen)

            repeated_adapter = _SourceResponsesOnly()
            repeated_projection, repeated_journey = _run_attempt(
                Path(temporary).resolve() / "assembly-no-progress",
                _RepeatedAssemblyActionModel(),
                repeated_adapter,
            )
            self.assertEqual(
                (repeated_projection.status, repeated_projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            self.assertEqual(
                repeated_adapter.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            repeated_events = read_evidence(repeated_journey)
            repeated_validation = next(
                event
                for event in repeated_events
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["tool_call_id"] == "repeated-assembly-3"
            )
            # The member has a lookup, so the refusal names it.
            self.assertEqual(
                repeated_validation["payload"]["errors"],
                ["fetch_assembly_report_refsnp_event_unknown"],
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionStarted"
                    and event["payload"]["tool_call_id"] == "repeated-assembly-3"
                    for event in repeated_events
                )
            )

            failed_adapter = _FailingAssemblySourceResponses()
            failed_projection, failed_journey = _run_attempt(
                Path(temporary).resolve() / "assembly-source-failure",
                _AdaptiveInvestigationModel(),
                failed_adapter,
            )
            # The exhausted retry window holds the member for its operational
            # Review instead of failing it.
            self.assertEqual(
                (failed_projection.status, failed_projection.stop_reason),
                ("operational_review_pending", None),
            )
            self.assertEqual(
                failed_adapter.operations,
                [
                    "fetch_refsnp:rs429358",
                    *["fetch_assembly_report:GCF_000001405.40"] * 3,
                ],
            )
            failed_events = read_evidence(failed_journey)
            source_failure = next(
                event
                for event in failed_events
                if event["event_type"] == "InvestigationSourceContactFailed"
                and event["payload"]["operation"] == "fetch_assembly_report"
            )
            execution_start = next(
                event
                for event in failed_events
                if event["event_type"] == "ObservedToolExecutionStarted"
                and event["payload"]["tool_name"] == "fetch_assembly_report"
            )
            execution_failure = next(
                event
                for event in failed_events
                if event["event_type"] == "ObservedToolExecutionFailed"
                and event["payload"]["started_event_id"] == execution_start["event_id"]
            )
            self.assertEqual(
                execution_failure["payload"]["mechanical_error_type"],
                "ReferenceAdapterError",
            )
            self.assertEqual(
                execution_failure["payload"]["stage_failure_event_id"],
                source_failure["event_id"],
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionSucceeded"
                    and event["payload"].get("started_event_id")
                    == execution_start["event_id"]
                    for event in failed_events
                )
            )

            # A malformed adapter return is checked inside the source contact: the
            # contact fails with the unclassified error, the Attempt records its
            # terminal, and the error propagates for the coordinator's hold
            # instead of a retry or a member Review.
            mechanical_adapter = _MalformedAssemblySourceResponse()
            mechanical_root = Path(temporary).resolve() / "assembly-mechanical-failure"
            with self.assertRaises(AttributeError):
                _run_attempt(
                    mechanical_root,
                    _AdaptiveInvestigationModel(),
                    mechanical_adapter,
                )
            self.assertEqual(
                mechanical_adapter.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            (mechanical_file,) = mechanical_root.rglob("journey.jsonl")
            mechanical_events = read_evidence(Journey.open(mechanical_file))
            mechanical_start = next(
                event
                for event in mechanical_events
                if event["event_type"] == "ObservedToolExecutionStarted"
                and event["payload"]["tool_name"] == "fetch_assembly_report"
            )
            mechanical_failure = next(
                event
                for event in mechanical_events
                if event["event_type"] == "ObservedToolExecutionFailed"
                and event["payload"]["started_event_id"] == mechanical_start["event_id"]
            )
            mechanical_source_failure = next(
                event
                for event in mechanical_events
                if event["event_type"] == "InvestigationSourceContactFailed"
                and event["payload"]["operation"] == "fetch_assembly_report"
            )
            self.assertEqual(
                (
                    mechanical_failure["payload"]["mechanical_error_type"],
                    mechanical_source_failure["payload"]["error_type"],
                ),
                ("AttributeError", "AttributeError"),
            )
            self.assertEqual(
                mechanical_failure["payload"]["stage_failure_event_id"],
                mechanical_source_failure["event_id"],
            )
            mechanical_terminal = next(
                event
                for event in mechanical_events
                if event["event_type"] == "InvestigationAttemptFailed"
            )
            self.assertEqual(
                (
                    mechanical_terminal["payload"]["code"],
                    mechanical_terminal["payload"]["source_event_id"],
                ),
                ("source_contact_failed", mechanical_source_failure["event_id"]),
            )
            self.assertFalse(
                any(
                    event["event_type"] == "OperationalFailureRecorded"
                    for event in mechanical_events
                )
            )

            negative_adapter = _NegativeAssemblySourceResponses()
            negative_projection, negative_journey = _run_attempt(
                Path(temporary).resolve() / "assembly-typed-negative",
                _AdaptiveInvestigationModel(),
                negative_adapter,
            )
            self.assertEqual(negative_projection.status, "review_pending")
            # The negative finding is the member's keyed issue.
            self.assertTrue(
                negative_projection.pending_request.question.issue.conflicts
            )
            self.assertEqual(
                negative_adapter.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            negative_events = read_evidence(negative_journey)
            negative_result = next(
                event
                for event in negative_events
                if event["event_type"] == "AssemblyReportToolResultRecorded"
            )
            negative_start = next(
                event
                for event in negative_events
                if event["event_type"] == "ObservedToolExecutionStarted"
                and event["payload"]["tool_name"] == "fetch_assembly_report"
            )
            negative_success = next(
                event
                for event in negative_events
                if event["event_type"] == "ObservedToolExecutionSucceeded"
                and event["payload"]["started_event_id"] == negative_start["event_id"]
            )
            self.assertEqual(
                negative_result["payload"]["result"]["status"], "not_found"
            )
            self.assertEqual(
                negative_success["payload"]["stage_result_event_id"],
                negative_result["event_id"],
            )

            refused_adapter = _SourceResponsesOnly()
            refused_projection, refused_journey = _run_attempt(
                Path(temporary).resolve() / "assembly-pre-contact-refusal",
                _AdaptiveInvestigationModel(first_action_ineligible=True),
                refused_adapter,
            )
            self.assertEqual(refused_projection.status, "review_pending")
            self.assertIsNone(refused_projection.stop_reason)
            self.assertEqual(
                refused_adapter.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            refused_events = read_evidence(refused_journey)
            refused_validation = next(
                event
                for event in refused_events
                if event["event_type"] == "ToolCallValidated"
                and event["payload"]["tool_call_id"] == "investigation-1"
            )
            self.assertEqual(refused_validation["payload"]["status"], "invalid")
            self.assertEqual(
                refused_validation["payload"]["errors"],
                ["fetch_assembly_report_requires_typed_refsnp_result"],
            )
            refused_disposition = next(
                event
                for event in refused_events
                if event["event_type"] == "AgentActionTurnDispositionRecorded"
                and event["payload"]["validation_event_ids"]
                == [refused_validation["event_id"]]
            )
            self.assertEqual(
                refused_disposition["payload"]["disposition"], "singleton_refused"
            )
            self.assertFalse(
                any(
                    event["event_type"] == "ObservedToolExecutionStarted"
                    and event["payload"]["tool_call_id"] == "investigation-1"
                    for event in refused_events
                )
            )
            # Preserve terminal-correlation attacks on the no-progress closure; it
            # stops for work review instead of failing the Attempt.
            stopped_projection, refused_journey = _run_attempt(
                Path(temporary).resolve() / "assembly-terminal-refusal",
                _RepeatedAssemblyActionModel(),
                refused_adapter,
            )
            self.assertEqual(
                (stopped_projection.status, stopped_projection.stop_reason),
                ("agent_work_review_pending", None),
            )
            refused_events = read_evidence(refused_journey)
            attempt_stopped = next(
                event
                for event in refused_events
                if event["event_type"] == "AttemptStopped"
            )
            outer_closure = next(
                event
                for event in refused_events
                if event["event_type"] == "activity_succeeded"
                and event["payload"]["stage"] == "investigation_planning_attempt"
            )
            self.assertEqual(
                outer_closure["payload"]["result_event_id"],
                attempt_stopped["event_id"],
            )
            self.assertFalse(
                any(
                    event["event_type"] == "activity_failed" for event in refused_events
                )
            )
            hostile_attempt_closure = copy.deepcopy(refused_events)
            outer_closure = next(
                event
                for event in hostile_attempt_closure
                if event["event_type"] == "activity_succeeded"
                and event["payload"]["stage"] == "investigation_planning_attempt"
            )
            outer_closure["payload"]["result_event_id"] = outer_closure["event_id"]
            operations_before_reopen = list(refused_adapter.operations)
            with self.assertRaisesRegex(ValueError, "Investigation Attempt closure"):
                project_investigation(hostile_attempt_closure)
            self.assertEqual(refused_adapter.operations, operations_before_reopen)

            hostile_attempt_stage = copy.deepcopy(refused_events)
            for event in hostile_attempt_stage:
                if (
                    event["event_type"] in {"activity_started", "activity_succeeded"}
                    and event["payload"]["stage"] == "investigation_planning_attempt"
                ):
                    event["payload"]["stage"] = "planning_attempt"
            with self.assertRaisesRegex(ValueError, "Investigation Attempt closure"):
                project_investigation(hostile_attempt_stage)
            self.assertEqual(refused_adapter.operations, operations_before_reopen)

            hostile_domain_terminal = copy.deepcopy(refused_events)
            outer_closure_index = next(
                index
                for index, event in enumerate(hostile_domain_terminal)
                if event["event_type"] == "activity_succeeded"
                and event["payload"]["stage"] == "investigation_planning_attempt"
            )
            outer_closure = hostile_domain_terminal[outer_closure_index]
            attempt_terminal = next(
                event
                for event in hostile_domain_terminal
                if event["event_type"] == "AttemptStopped"
            )
            unrelated_attempt_event = copy.deepcopy(attempt_terminal)
            unrelated_attempt_event["event_id"] = str(uuid.uuid4())
            unrelated_attempt_event["event_type"] = "HostileAttemptEvidence"
            unrelated_attempt_event["parent_ids"] = [attempt_terminal["event_id"]]
            outer_closure["payload"]["result_event_id"] = unrelated_attempt_event[
                "event_id"
            ]
            outer_closure["parent_ids"] = [unrelated_attempt_event["event_id"]]
            hostile_domain_terminal.insert(outer_closure_index, unrelated_attempt_event)
            with self.assertRaisesRegex(ValueError, "Investigation Attempt closure"):
                project_investigation(hostile_domain_terminal)
            self.assertEqual(refused_adapter.operations, operations_before_reopen)

    def test_agent_adapts_to_an_ineligible_first_action(self) -> None:
        # The repeated-missing-basis stop is pinned by
        # test_repeated_missing_source_basis_stops_for_review_and_continues_in_a_new_attempt.
        with tempfile.TemporaryDirectory() as temporary:
            model = _AdaptiveInvestigationModel(first_action_ineligible=True)
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve() / "adaptive", model, adapter
            )
            self.assertEqual(projection.status, "review_pending")
            self.assertIsNone(projection.stop_reason)
            self.assertEqual(model.calls, 4)
            self.assertEqual(projection.source_contact_count, 2)
            self.assertEqual(project_investigation(read_evidence(journey)), projection)

    def test_malformed_model_output_stops_without_source_or_vcf_contact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _MalformedOutputModel()
            adapter = _SourceResponsesOnly()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )
            events = read_evidence(journey)

        self.assertEqual(
            (projection.status, projection.stop_reason),
            ("agent_work_review_pending", None),
        )
        self.assertEqual(projection.attempts[0].stop_reason, "failed_turns")
        self.assertEqual(adapter.contact_count, 0)
        # The default's fourth failed turn in a row stops (four output
        # repairs, two before).
        self.assertEqual(model.calls, 4)
        lineage = projection.attempts[0].structured_output_lineages[0]
        self.assertEqual(lineage.max_retries, 4)
        self.assertIsNone(lineage.accepted_output)
        self.assertEqual(
            tuple(item.status for item in lineage.validation_attempts),
            ("invalid",) * 4,
        )
        self.assertEqual(
            tuple(item.output_call_ids for item in lineage.validation_attempts),
            tuple((f"malformed-{n}",) for n in range(1, 5)),
        )
        self.assertEqual(project_investigation(events), projection)
        changed = copy.deepcopy(events)
        call = next(
            event
            for event in changed
            if event["event_type"] == "StructuredOutputCallRecorded"
        )
        call["payload"]["response_event_id"] = next(
            event["event_id"]
            for event in changed
            if event["event_type"] == "InvestigationAttemptBound"
        )
        with self.assertRaisesRegex(
            ValueError, "structured-output call changed its model response"
        ):
            project_investigation(changed)
        changed = copy.deepcopy(events)
        validation = next(
            event for event in changed if event["event_type"] == "AgentOutputValidated"
        )
        validation["payload"]["output"] = {"invalid_json": "changed"}
        with self.assertRaisesRegex(
            ValueError, "structured-output validation changed its exact call"
        ):
            project_investigation(changed)
        changed = copy.deepcopy(events)
        links = [
            event
            for event in changed
            if event["event_type"] == "StructuredOutputValidationLinked"
        ]
        calls = [
            event
            for event in changed
            if event["event_type"] == "StructuredOutputCallRecorded"
        ]
        links[1]["payload"]["response_event_id"] = links[0]["payload"][
            "response_event_id"
        ]
        links[1]["payload"]["output_call_event_ids"] = [calls[0]["event_id"]]
        second_validation = next(
            event
            for event in changed
            if event["event_id"] == links[1]["payload"]["validation_event_id"]
        )
        second_validation["payload"]["output"] = calls[0]["payload"]["arguments"]
        second_validation["parent_ids"] = [calls[0]["event_id"]]
        with self.assertRaisesRegex(
            ValueError, "structured-output call was not consumed exactly once"
        ):
            project_investigation(changed)

    def _begin_with_a_pre_output_crash(self, root):
        """Begin Investigation in a forked child whose first model request crashes."""
        request = {
            "schema_version": "bio-neb.investigation-request/2",
            "asserted_by": "researcher:owner",
            "request": "Investigate rs429358 in sample HG001 on GRCh38.",
            "target": {"rsid": "rs429358"},
        }
        journey = forked_child(
            root,
            parent_file=self.prerequisite(
                "executed", investigation_request=request
            ).journey_file,
        )
        request_file = root / "investigation-request.json"
        request_file.write_text(canonical_json(request), encoding="utf-8")
        adapter = _SourceResponsesOnly()
        result, stdout, stderr = self._pre_output_cli(
            [
                "investigation-begin",
                "--journey-file",
                str(journey.path),
                "--request-file",
                str(request_file),
            ],
            adapter,
        )
        return journey, adapter, result, stdout, stderr

    @staticmethod
    def _pre_output_cli(argv, adapter):
        model = ScriptedModelEndpoint(
            lambda _messages, _info: (_ for _ in ()).throw(
                AssertionError("pre-output model performed contact")
            ),
            adapter=_PreOutputFailureAdapter,
        )
        stdout = StringIO()
        stderr = StringIO()
        with (
            patch.dict(
                os.environ, {"LOCAL_MODEL_KEY": "pre-output-credential-sentinel"}
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = main(
                argv,
                investigation_model_factory=model,
                investigation_source_adapter_factory=lambda: adapter,
            )
        return result, stdout, stderr

    def test_a_pre_output_crash_holds_the_run_and_resume_replans_its_member(
        self,
    ) -> None:
        # Our bug is an invariant that holds the Run, not an ended Run. The crash
        # is reported each time it recurs, never hidden; after the fix `resume`
        # plans its member again.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey, adapter, result, _stdout, stderr = (
                self._begin_with_a_pre_output_crash(root)
            )
            self.assertEqual(result, 2)
            self.assertEqual(
                json.loads(stderr.getvalue())["code"], "coordinator_unhandled_error"
            )
            resume = ["resume", "--journey-file", str(journey.path)]
            observed = observe(ObserveRun(journey_file=str(journey.path)))
            self.assertEqual(observed.status, "run_held")

            # A resume before the fix meets the crash once more and holds again.
            result, _stdout, stderr = self._pre_output_cli(resume, adapter)
            self.assertEqual(result, 2)
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))).status,
                "run_held",
            )
            self.assertEqual(adapter.contact_count, 0)

            # After the fix, resume plans the member afresh.
            from tests.sequential_investigation_support import (
                SequentialAdaptiveModel,
                SequentialSourceResponses,
            )

            fixed_sources = SequentialSourceResponses()
            stdout = StringIO()
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "fixed"}),
                redirect_stdout(stdout),
                redirect_stderr(StringIO()),
            ):
                result = main(
                    resume,
                    investigation_model_factory=SequentialAdaptiveModel().model,
                    investigation_source_adapter_factory=lambda: fixed_sources,
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["status"], "investigation_review_pending"
            )
            attempts = observe(
                ObserveRun(journey_file=str(journey.path))
            ).investigation.attempts
            self.assertEqual(
                [a.collection_member_id for a in attempts],
                [attempts[0].collection_member_id] * len(attempts),
            )
            self.assertEqual(
                [a.outcome for a in attempts[:-1]], ["failed"] * (len(attempts) - 1)
            )
            self.assertEqual(attempts[-1].outcome, "work_completed")

    def test_public_pre_output_failure_reopens_without_contact_or_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journey, adapter, result, stdout, stderr = (
                self._begin_with_a_pre_output_crash(root)
            )

            # The crash holds the Run (R2); it is not an ended Run.
            self.assertEqual(result, 2)
            self.assertEqual(
                json.loads(stderr.getvalue())["code"], "coordinator_unhandled_error"
            )
            events = Journey.open(journey.path).read()
            bound = next(
                event
                for event in events
                if event["event_type"] == "InvestigationAttemptBound"
            )
            failed = next(
                event
                for event in events
                if event["event_type"] == "InvestigationAttemptFailed"
            )
            self.assertEqual(failed["payload"]["code"], "agent_run_failed")
            # The crash came after the member's context was recorded, before any
            # request: that fact is the failure's source.
            (context,) = [
                event
                for event in events
                if event["event_type"] == "InvestigationMemberContextRecorded"
            ]
            self.assertEqual(context["payload"]["bound_event_id"], bound["event_id"])
            self.assertEqual(failed["payload"]["source_event_id"], context["event_id"])
            self.assertRegex(
                failed["payload"]["reason"],
                r"^ValueError@test_investigation\.py:[1-9][0-9]*$",
            )
            self.assertFalse(
                any(
                    event["event_type"]
                    in {
                        "ModelRequestRecorded",
                        "ModelPhysicalContactStarted",
                        "InvestigationSourceContactStarted",
                    }
                    and event.get("payload", {}).get("attempt_id")
                    == bound["payload"]["attempt_id"]
                    for event in events
                )
            )
            self.assertEqual(adapter.contact_count, 0)
            self.assertNotIn("pre-output-secret-sentinel", canonical_json(events))

            hostile = [
                dict(event) for event in read_evidence(Journey.open(journey.path))
            ]
            hostile_failed = next(
                event
                for event in hostile
                if event["event_type"] == "InvestigationAttemptFailed"
            )
            hostile_failed["parent_ids"] = [
                next(
                    event["event_id"]
                    for event in hostile
                    if event["event_type"] == "InvestigationStarted"
                )
            ]
            with self.assertRaisesRegex(
                ValueError, "Investigation failure changed its exact parent"
            ):
                project_investigation(hostile)

            before_reopen = journey.path.read_bytes()
            reopened_stdout = StringIO()
            with redirect_stdout(reopened_stdout), redirect_stderr(StringIO()):
                reopened_result = main(
                    ["observe", "--journey-file", str(journey.path), "--rich"],
                    investigation_model_factory=lambda _loaded, _selected: (
                        _ for _ in ()
                    ).throw(AssertionError("failure reopen contacted model factory")),
                    investigation_source_adapter_factory=lambda: (_ for _ in ()).throw(
                        AssertionError("failure reopen contacted source factory")
                    ),
                )
            self.assertEqual(reopened_result, 0)
            self.assertEqual(journey.path.read_bytes(), before_reopen)
            reopened = json.loads(reopened_stdout.getvalue())
            self.assertEqual(reopened["status"], "run_held")
            self.assertEqual(
                reopened["investigation"]["attempts"][-1]["stop_reason"],
                "agent_run_failed",
            )
            self.assertEqual(
                reopened["investigation"]["attempts"][-1]["failure_reason"],
                failed["payload"]["reason"],
            )
            self.assertIsNone(reopened["investigation"]["authority"])

    def test_source_contact_failure_is_typed_held_and_excludes_error_secret(
        self,
    ) -> None:
        # A transport failure is retried in the shared operation window; once
        # exhausted, the member is held for its operational Review.
        with tempfile.TemporaryDirectory() as temporary:
            model = _AdaptiveInvestigationModel()
            adapter = _FailingSourceResponses()
            projection, journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )
            events = journey.read()

        self.assertEqual(
            (projection.status, projection.stop_reason),
            ("operational_review_pending", None),
        )
        self.assertEqual(adapter.operations, ["fetch_refsnp:rs429358"] * 3)
        self.assertEqual(projection.source_contact_count, 1)
        [failure] = projection.source_failures
        self.assertEqual(
            (failure.operation, failure.error_type),
            ("fetch_refsnp", "ReferenceAdapterError"),
        )
        [operational] = [
            e["payload"]
            for e in events
            if e["event_type"] == "OperationalFailureRecorded"
        ]
        self.assertEqual(
            (operational["failure_class"], operational["contact_attempt"]),
            ("exhausted", 3),
        )
        self.assertTrue(
            any(e["event_type"] == "AgentFailureReviewRequested" for e in events)
        )
        for retained in (
            canonical_json(projection.model_dump(mode="json")),
            canonical_json(events),
        ):
            self.assertNotIn("forbidden-source-error-detail", retained)

    def test_agent_returns_typed_unresolved_without_ineligible_second_contact(
        self,
    ) -> None:
        # Placed only on another assembly: a typed unresolved finding.
        incompatible = json.loads(REFSNP_BODY)
        incompatible["primary_snapshot_data"]["placements_with_allele"][0][
            "placement_annot"
        ]["seq_id_traits_by_assembly"][0]["assembly_accession"] = "GCF_000001405.39"
        with tempfile.TemporaryDirectory() as temporary:
            model = _AdaptiveInvestigationModel()
            adapter = _SourceResponsesOnly(
                refsnp_body=json.dumps(incompatible).encode()
            )
            projection, _journey = _run_attempt(
                Path(temporary).resolve(), model, adapter
            )

        self.assertEqual(projection.status, "review_pending")
        # The typed unresolved finding asks its own issue Review.
        issue = projection.pending_request.question.issue
        self.assertEqual((issue.source, issue.verdict), ("refsnp", "incompatible"))
        self.assertEqual(projection.proposals, ())
        self.assertEqual(model.calls, 2)
        self.assertEqual(adapter.operations, ["fetch_refsnp:rs429358"])

    def test_deterministic_tools_retain_truthful_nonproposal_outcomes(self) -> None:
        # Several ALTs at one locus are candidates; a second placement on the Run's
        # assembly at another locus is the model's to name.
        ambiguous = json.loads(REFSNP_BODY)
        placements = ambiguous["primary_snapshot_data"]["placements_with_allele"]
        elsewhere = copy.deepcopy(placements[0])
        for allele in elsewhere["alleles"]:
            allele["allele"]["spdi"]["position"] = 44908700
        placements.append(elsewhere)
        incompatible = json.loads(REFSNP_BODY)
        incompatible["primary_snapshot_data"]["placements_with_allele"][0][
            "placement_annot"
        ]["seq_id_traits_by_assembly"][0]["assembly_accession"] = "GCF_000001405.39"
        conflicted = json.loads(REFSNP_BODY)
        conflicted["refsnp_id"] = "7412"
        malformed_target = json.loads(REFSNP_BODY)
        malformed_target["primary_snapshot_data"]["placements_with_allele"][0][
            "alleles"
        ][1]["allele"]["spdi"]["seq_id"] = "invalid contig"
        cases = {
            "placement_unnamed": json.dumps(ambiguous).encode(),
            "incompatible": json.dumps(incompatible).encode(),
            "conflicted": json.dumps(conflicted).encode(),
            "malformed": b"not-json",
            "malformed-target": json.dumps(malformed_target).encode(),
        }
        for case, body in cases.items():
            with self.subTest(case=case):
                source = RetrievedSource.create(
                    operation="fetch_refsnp",
                    request_identity="https://example.invalid/refsnp/429358",
                    status_code=200,
                    observed_at="2026-08-26T00:00:00Z",
                    elapsed_time_ms=1,
                    raw_response=body,
                )
                result = normalize_refsnp(
                    rsid="rs429358",
                    assembly_accession="GCF_000001405.40",
                    source=source,
                    raw_response=body,
                )
                self.assertEqual(
                    result.status,
                    "malformed" if case == "malformed-target" else case,
                )
                if result.status == "placement_unnamed":
                    self.assertEqual(len(result.candidates), 2)
                else:
                    self.assertFalse(result.candidates)
                self.assertTrue(result.conflicts)

        wrong_assembly = json.loads(ASSEMBLY_BODY)
        wrong_assembly["reports"][0]["accession"] = "GCF_000001405.39"
        body = json.dumps(wrong_assembly).encode()
        source = RetrievedSource.create(
            operation="fetch_assembly_report",
            request_identity="https://example.invalid/assembly/GCF_000001405.40",
            status_code=200,
            observed_at="2026-08-26T00:00:01Z",
            elapsed_time_ms=1,
            raw_response=body,
        )
        assembly = normalize_assembly_report(
            expected_assembly_accession="GCF_000001405.40",
            expected_assembly_name="GRCh38.p14",
            source=source,
            raw_response=body,
        )
        self.assertEqual(assembly.status, "incompatible")

        malformed_assembly = json.loads(ASSEMBLY_BODY)
        malformed_assembly["reports"][0]["assembly_info"]["assembly_name"] = 38
        malformed_body = json.dumps(malformed_assembly).encode()
        malformed_source = RetrievedSource.create(
            operation="fetch_assembly_report",
            request_identity="https://example.invalid/assembly/GCF_000001405.40",
            status_code=200,
            observed_at="2026-08-26T00:00:01Z",
            elapsed_time_ms=1,
            raw_response=malformed_body,
        )
        malformed_result = normalize_assembly_report(
            expected_assembly_accession="GCF_000001405.40",
            expected_assembly_name="GRCh38.p14",
            source=malformed_source,
            raw_response=malformed_body,
        )
        self.assertEqual(malformed_result.status, "malformed")

    def test_review_approve_stop_and_revise_stale_conflict_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for case in ("approve", "stop", "amend", "clarify"):
                with self.subTest(case=case):
                    model = _AdaptiveInvestigationModel()
                    adapter = _SourceResponsesOnly()
                    projection, journey = _run_attempt(root / case, model, adapter)
                    pending = projection.pending_request
                    assert pending is not None
                    before_stale = len(Journey.open(journey.path).read())
                    with self.assertRaises(InvestigationApplicationRefused) as stale:
                        respond_investigation_review(
                            RespondInvestigationReviewCommand(
                                journey_file=str(journey.path),
                                request_event_id=(
                                    "00000000-0000-4000-8000-000000000999"
                                ),
                                response={
                                    "action": "approve",
                                    "proposal_id": pending.proposal.proposal_id,
                                },
                            )
                        )
                    self.assertEqual(stale.exception.code, "investigation_review_stale")
                    self.assertEqual(
                        len(Journey.open(journey.path).read()), before_stale
                    )
                    response: dict[str, object] = {
                        "action": case,
                        "proposal_id": pending.proposal.proposal_id,
                    }
                    if case in {"amend", "clarify"}:
                        response = {
                            "action": "revise",
                            "proposal_id": pending.proposal.proposal_id,
                            "asserted_by": "researcher:owner",
                            "operations": (
                                {
                                    "operation": case,
                                    "work_item_id": (
                                        pending.proposal.work_items[0].work_item_id
                                    ),
                                    "request": {
                                        "request_text": (
                                            f"{case.title()} and investigate rs429358."
                                        ),
                                        "rsid": "rs429358",
                                    },
                                },
                            ),
                        }
                    if case == "stop":
                        response["reason"] = "Owner stopped before observation."
                    command = RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=pending.request_event_id,
                        response=response,
                    )
                    before_contacts = adapter.contact_count
                    reviewed = respond_investigation_review(command)
                    response_event = next(
                        event
                        for event in reversed(Journey.open(journey.path).read())
                        if event["event_type"] == "InvestigationReviewResponded"
                    )
                    response_payload = response_event["payload"]
                    assert_json_identity(
                        self,
                        response_payload["response_id"],
                        response_payload,
                        "request_event_id response",
                        "investigation-review-response:",
                    )
                    if reviewed.authority is not None:
                        assert_json_identity(
                            self,
                            reviewed.authority.authority_id,
                            reviewed.authority,
                            "proposal_id review_response_id authorized_work_items predecessor_ids",
                            "investigation-authority:",
                        )
                    for revision in reviewed.revisions:
                        assert_json_identity(
                            self,
                            revision.revision_id,
                            revision,
                            "revision_number supersedes_proposal_id response_id members removed_work_item_ids deferred_work_item_ids",
                            "investigation-revision:",
                        )
                        for member in revision.members:
                            assert_json_identity(
                                self,
                                member.member_id,
                                member,
                                "operation previous_work_item_id carried_work_item request",
                                "investigation-revision-member:",
                            )
                    self.assertEqual(adapter.contact_count, before_contacts)
                    expected = {
                        "approve": "approved",
                        "stop": "stopped",
                        "amend": "investigation_work_ready",
                        "clarify": "investigation_work_ready",
                    }[case]
                    self.assertEqual(reviewed.status, expected)
                    self.assertEqual(reviewed.authority is not None, case == "approve")
                    retained_count = len(Journey.open(journey.path).read())
                    replay = respond_investigation_review(command)
                    self.assertEqual(replay, reviewed)
                    self.assertEqual(
                        len(Journey.open(journey.path).read()), retained_count
                    )
                    conflict = RespondInvestigationReviewCommand.model_validate(
                        {
                            "journey_file": str(journey.path),
                            "request_event_id": pending.request_event_id,
                            "response": {
                                "action": "stop",
                                "proposal_id": pending.proposal.proposal_id,
                                "reason": "conflicting replay",
                            },
                        }
                    )
                    with self.assertRaises(InvestigationApplicationRefused) as refused:
                        respond_investigation_review(conflict)
                    self.assertEqual(
                        refused.exception.code, "investigation_review_conflict"
                    )
                    self.assertEqual(
                        len(Journey.open(journey.path).read()), retained_count
                    )

    def test_review_revision_refuses_invalid_membership_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cases = (
                ("duplicate", "investigation_review_duplicate_membership"),
                ("unknown", "investigation_review_unknown_membership"),
                ("missing", "investigation_review_missing_membership"),
                ("stale", "investigation_review_stale"),
            )
            for case, expected_code in cases:
                with self.subTest(case=case):
                    adapter = _SourceResponsesOnly()
                    projection, journey = _run_attempt(
                        root / case, _AdaptiveInvestigationModel(), adapter
                    )
                    pending = projection.pending_request
                    assert pending is not None
                    proposal = pending.proposal
                    member_id = proposal.work_items[0].work_item_id
                    operations: tuple[dict[str, object], ...] = (
                        {"operation": "keep", "work_item_id": member_id},
                    )
                    proposal_id = proposal.proposal_id
                    if case == "duplicate":
                        operations = (
                            *operations,
                            {"operation": "remove", "work_item_id": member_id},
                        )
                    elif case == "unknown":
                        operations = (
                            {
                                "operation": "keep",
                                "work_item_id": "work-item:" + "f" * 64,
                            },
                        )
                    elif case == "missing":
                        operations = (
                            {
                                "operation": "add",
                                "request": {
                                    "request_text": "Investigate added rs7412 scope.",
                                    "rsid": "rs7412",
                                },
                            },
                        )
                    elif case == "stale":
                        proposal_id = "investigation-proposal:" + "f" * 64
                    command = RespondInvestigationReviewCommand.model_validate(
                        {
                            "journey_file": str(journey.path),
                            "request_event_id": pending.request_event_id,
                            "response": {
                                "action": "revise",
                                "proposal_id": proposal_id,
                                "asserted_by": "researcher:owner",
                                "operations": operations,
                            },
                        }
                    )
                    before = journey.path.read_bytes()
                    before_contacts = adapter.contact_count

                    with self.assertRaises(InvestigationApplicationRefused) as refused:
                        respond_investigation_review(command)

                    self.assertEqual(refused.exception.code, expected_code)
                    self.assertEqual(journey.path.read_bytes(), before)
                    self.assertEqual(adapter.contact_count, before_contacts)

    def test_causal_investigation_outage_opens_shared_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "request": "Investigate rs429358 in sample HG001.",
                "target": {"rsid": "rs429358"},
            }
            child = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "investigation-request.json"
            request_file.write_text(canonical_json(request), encoding="utf-8")
            model = _FailingInvestigationModel()
            adapter = _SourceResponsesOnly()
            stdout = StringIO()
            stderr = StringIO()

            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "retry-secret-sentinel"}),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = main(
                    [
                        "investigation-begin",
                        "--journey-file",
                        str(child.path),
                        "--request-file",
                        str(request_file),
                    ],
                    investigation_model_factory=model.model,
                    investigation_source_adapter_factory=lambda: adapter,
                )

            self.assertEqual(result, 0, stderr.getvalue())
            self.assertEqual(
                json.loads(stdout.getvalue())["status"],
                "operational_review_pending",
            )
            self.assertEqual(model.calls, 2)
            self.assertEqual(adapter.contact_count, 0)
            events = read_evidence(Journey.open(child.path))
            # The child also reads its parent's planning contacts; count the
            # Investigation Attempt's own.
            (investigation_attempt,) = [
                event["payload"]["attempt_id"]
                for event in events
                if event["event_type"] == "AttemptStarted"
                and event["payload"]["agent_role"] == "investigation_planner"
            ]
            started_contacts = [
                event
                for event in events
                if event["event_type"] == "ModelPhysicalContactStarted"
                and event["payload"]["attempt_id"] == investigation_attempt
            ]
            self.assertEqual(len(started_contacts), 2)
            self.assertTrue(
                any(
                    event["event_type"] == "ModelAvailabilityRetryScheduled"
                    for event in events
                )
            )
            failed = [
                event for event in events if event["event_type"] == "ModelRequestFailed"
            ]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["payload"]["physical_contact_count"], 2)
            self.assertNotIn("retry-secret-sentinel", canonical_json(events))

    def test_public_begin_reaches_review_when_planner_requires_one_final_output_call(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "request": "Investigate rs429358 in sample HG001.",
                "target": {"rsid": "rs429358"},
            }
            child = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "investigation-request.json"
            request_file.write_text(canonical_json(request), encoding="utf-8")
            model = _LiveDuplicateFinalOutputModel()
            adapter = _SourceResponsesOnly()

            projection = begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(child.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "offline-only"},
                model_factory=model.model,
                source_adapter_factory=lambda: adapter,
            )

            events = read_evidence(Journey.open(child.path))
            output_validation = next(
                event
                for event in events
                if event["event_type"] == "AgentOutputValidated"
            )
            self.assertEqual(
                (
                    projection.status,
                    projection.stop_reason,
                    projection.pending_request is not None,
                    model.final_output_call_counts,
                    output_validation["payload"]["status"],
                    output_validation["payload"]["errors"],
                ),
                ("review_pending", None, True, [1], "valid", []),
            )
            self.assertIsNone(projection.authority)
            self.assertEqual(projection.review_receipts, ())
            self.assertEqual(projection.model_contact_count, 3)
            self.assertEqual(projection.source_contact_count, 2)
            lineage = projection.attempts[0].structured_output_lineages[0]
            self.assertEqual(
                (
                    lineage.tool_name,
                    lineage.max_retries,
                    len(lineage.validation_attempts),
                    lineage.accepted_output,
                ),
                (
                    "compile_proposal",
                    # The default's four failed turns, recorded as the
                    # lineage's most failed turns.
                    4,
                    1,
                    {
                        "refsnp_event_id": next(
                            event["event_id"]
                            for event in events
                            if event["event_type"] == "RefSnpToolResultRecorded"
                        ),
                        "assembly_event_id": next(
                            event["event_id"]
                            for event in events
                            if event["event_type"] == "AssemblyReportToolResultRecorded"
                        ),
                        "outcome": "complete",
                    },
                ),
            )
            validation_attempt = lineage.validation_attempts[0]
            self.assertEqual(validation_attempt.status, "valid")
            self.assertEqual(validation_attempt.errors, ())
            self.assertEqual(
                validation_attempt.validation_event_id, output_validation["event_id"]
            )
            self.assertEqual(
                projection.proposal_output_lineages[0].output_lineages,
                projection.attempts[0].structured_output_lineages,
            )
            self.assertEqual(
                projection.proposal_output_lineages[0].proposal_id,
                projection.proposals[0].proposal_id,
            )
            self.assertEqual(model.calls, 3)
            self.assertEqual(
                adapter.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            continuation = projection.attempts[0].continuation_policy.model_dump(
                mode="json"
            )
            self.assertEqual(continuation, {"failed_turns": 4})
            self.assertNotIn("source_contact_ceiling", continuation)
            self.assertFalse(
                any(
                    event["event_type"] == "ModelAvailabilityRetryScheduled"
                    for event in events
                )
            )
            changed = copy.deepcopy(events)
            changed_validation = next(
                event
                for event in changed
                if event["event_type"] == "AgentOutputValidated"
            )
            changed_validation["parent_ids"] = [
                next(
                    event["event_id"]
                    for event in changed
                    if event["event_type"] == "InvestigationStarted"
                )
            ]
            with self.assertRaisesRegex(
                ValueError, "structured-output validation changed its exact call"
            ):
                project_investigation(changed)
            changed = copy.deepcopy(events)
            for event in changed:
                if event["event_type"] in {
                    "StructuredOutputCallRecorded",
                    "StructuredOutputValidationLinked",
                }:
                    event["payload"]["collection_member_id"] = "rs1:observe_variant"
                if event["event_type"] == "InvestigationProposalRecorded":
                    event["payload"]["output_lineages"][0]["collection_member_id"] = (
                        "rs1:observe_variant"
                    )
            with self.assertRaisesRegex(
                ValueError, "Investigation proposal changed its exact Attempt lineage"
            ):
                project_investigation(changed)

        for invalid_before_duplicate in (False, True):
            with (
                self.subTest(invalid_before_duplicate=invalid_before_duplicate),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                request = {
                    "schema_version": "bio-neb.investigation-request/2",
                    "asserted_by": "researcher:owner",
                    "request": "Investigate rs429358 in sample HG001.",
                    "target": {"rsid": "rs429358"},
                }
                child = forked_child(
                    root,
                    parent_file=self.prerequisite(
                        "executed", investigation_request=request
                    ).journey_file,
                )
                request_file = root / "investigation-request.json"
                request_file.write_text(canonical_json(request), encoding="utf-8")
                model = _LiveDuplicateFinalOutputModel(
                    honor_exactly_one_instruction=False,
                    invalid_before_duplicate=invalid_before_duplicate,
                )

                projection = begin_in_pass(
                    BeginInvestigationCommand(
                        journey_file=str(child.path),
                        request_files=(str(request_file),),
                    ),
                    environ={"LOCAL_MODEL_KEY": "offline-only"},
                    model_factory=model.model,  # noqa: B023
                    source_adapter_factory=_SourceResponsesOnly,
                )

                self.assertEqual(
                    (projection.status, projection.stop_reason),
                    ("agent_work_review_pending", None),
                )
                self.assertEqual(projection.proposals, ())
                self.assertEqual(projection.review_requests, ())
                self.assertIsNone(projection.authority)
                lineage = projection.attempts[0].structured_output_lineages[0]
                self.assertIsNone(lineage.accepted_output)
                # Several outputs in one reply are a failed turn with its
                # correction (every model): the member is corrected until its
                # fourth failed turn in a row, the lineage's most failed turns,
                # stops it for review.
                self.assertEqual(len(lineage.validation_attempts), lineage.max_retries)
                self.assertEqual(
                    projection.model_contact_count, 2 + lineage.max_retries
                )
                self.assertEqual(projection.source_contact_count, 2)
                self.assertEqual(
                    project_investigation(read_evidence(Journey.open(child.path))),
                    projection,
                )
                validation = lineage.validation_attempts[-1]
                self.assertEqual(validation.status, "invalid")
                self.assertEqual(validation.errors, ("multiple_output_calls",))
                self.assertEqual(
                    validation.output_call_ids,
                    ("live-duplicate-final-1", "live-duplicate-final-2"),
                )
                self.assertFalse(
                    any(
                        event["event_type"].startswith("ObservedToolExecution")
                        and event["payload"].get("tool_name") == "compile_proposal"
                        for event in read_evidence(Journey.open(child.path))
                    )
                )
                duplicate_events = read_evidence(Journey.open(child.path))
                changed = copy.deepcopy(duplicate_events)
                duplicate_link = next(
                    event
                    for event in changed
                    if event["event_type"] == "StructuredOutputValidationLinked"
                    and len(event["payload"]["output_call_event_ids"]) == 2
                )
                duplicate_link["payload"]["output_call_event_ids"] = duplicate_link[
                    "payload"
                ]["output_call_event_ids"][:1]
                with self.assertRaisesRegex(
                    ValueError, "structured-output validation changed its full call set"
                ):
                    project_investigation(changed)
                # Several outputs in one reply are one invalid attempt with its
                # correction (every model), so a valid one may follow; this model
                # repeated them until its failed turns stopped it.
                self.assertTrue(
                    next(
                        event["payload"]
                        for event in duplicate_events
                        if event["event_type"] == "AgentOutputValidated"
                        and event["payload"]["errors"] == ["multiple_output_calls"]
                    )["feedback"].startswith("Outputs: you sent `compile_proposal` 2 ")
                )
                corrected = StructuredOutputLineage(
                    collection_member_id=lineage.collection_member_id,
                    tool_name=lineage.tool_name,
                    max_retries=1,
                    validation_attempts=(
                        validation.model_copy(update={"validation_attempt": 1}),
                        validation_attempt.model_copy(update={"validation_attempt": 2}),
                    ),
                    accepted_output=validation_attempt.output,
                )
                self.assertEqual(corrected.accepted_output, validation_attempt.output)

    def test_failed_source_setup_holds_the_run_and_leaves_it_readable(self) -> None:
        # Known product bug (handed to its own ticket): the factory runs after
        # AttemptStarted and before InvestigationAttemptBound, so the held Run
        # keeps an Attempt with no terminal and stops reading.
        from bio_neb.application.advancement import RunAdvancementRefused

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "request": "Investigate rs429358 in this sample.",
                "target": {"rsid": "rs429358"},
            }
            journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed", investigation_request=request
                ).journey_file,
            )
            request_file = root / "investigation-request.json"
            request_file.write_text(canonical_json(request), encoding="utf-8")

            def fail_source_setup() -> _SourceResponsesOnly:
                raise RuntimeError("source adapter setup failed")

            with (
                self.assertRaises(RunAdvancementRefused) as held,
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "cli-secret-sentinel"}),
            ):
                begin_in_pass(
                    BeginInvestigationCommand(
                        journey_file=str(journey.path),
                        request_files=(str(request_file),),
                    ),
                    environ=os.environ,
                    model_factory=(
                        lambda loaded, selected: _AdaptiveInvestigationModel().model(
                            loaded, selected
                        )
                    ),
                    source_adapter_factory=fail_source_setup,
                )
            self.assertEqual(held.exception.code, "coordinator_unhandled_error")
            self.assertIsInstance(held.exception.__cause__, RuntimeError)
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))).status,
                "run_held",
            )

    def test_public_cli_begin_amend_continue_approve_and_hostile_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = {
                "schema_version": "bio-neb.investigation-request/2",
                "asserted_by": "researcher:owner",
                "request": "Investigate rs429358 in this sample.",
                "target": {"rsid": "rs429358"},
            }
            terminal_journey = forked_child(
                root,
                parent_file=self.prerequisite(
                    "executed",
                    investigation_request=request,
                    output_payloads={"VCF": RUN_VCF},
                ).journey_file,
            )
            request_file = root / "investigation-request.json"
            request_file.write_text(canonical_json(request), encoding="utf-8")
            adapter = _SourceResponsesOnly()
            first_model = _AdaptiveInvestigationModel()
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "cli-secret-sentinel"}),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = main(
                    [
                        "investigation-begin",
                        "--journey-file",
                        str(terminal_journey.path),
                        "--request-file",
                        str(request_file),
                    ],
                    investigation_model_factory=lambda _loaded, _selected: (
                        first_model.model(_loaded, _selected)
                    ),
                    investigation_source_adapter_factory=lambda: adapter,
                )
            self.assertEqual(result, 0, stderr.getvalue())
            first_view = json.loads(stdout.getvalue())
            self.assertEqual(first_view["status"], "investigation_review_pending")
            first_request = first_view["review"]["request_event_id"]
            first_proposal = first_view["investigation"]["proposals"][0]

            hostile_policy = copy.deepcopy(
                read_evidence(Journey.open(terminal_journey.path))
            )
            binding = next(
                event["payload"]
                for event in hostile_policy
                if event["event_type"] == "InvestigationAttemptBound"
            )
            binding["continuation_policy"]["failed_turns"] = 8
            binding["continuation_policy_id"] = (
                "continuation-policy:"
                + hashlib.sha256(
                    canonical_json(binding["continuation_policy"]).encode()
                ).hexdigest()
            )
            # The output lineage records the policy's most failed turns, so the
            # replacement is refused there first.
            with self.assertRaisesRegex(ValueError, "exact Attempt lineage"):
                project_investigation(hostile_policy)

            amendment_file = root / "amend.json"
            amendment_file.write_text(
                canonical_json(
                    {
                        "action": "revise",
                        "proposal_id": first_proposal["proposal_id"],
                        "asserted_by": "researcher:owner",
                        "operations": [
                            {
                                "operation": "amend",
                                "work_item_id": first_proposal["work_items"][0][
                                    "work_item_id"
                                ],
                                "request": {
                                    "request_text": (
                                        "Amend and investigate rs429358 again."
                                    ),
                                    "rsid": "rs429358",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            amendment_stderr = StringIO()
            # The amended member's new Attempt sees the earlier results as retained
            # context, not as its own typed results, so it fetches its sources again.
            second_model = _OwnReturnsInvestigationModel()
            with (
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "cli-secret-sentinel"}),
                redirect_stdout(StringIO()),
                redirect_stderr(amendment_stderr),
            ):
                result = main(
                    [
                        "answer",
                        "--journey-file",
                        str(terminal_journey.path),
                        "--request-event-id",
                        first_request,
                        "--response-file",
                        str(amendment_file),
                    ],
                    investigation_model_factory=lambda _loaded, _selected: (
                        second_model.model(_loaded, _selected)
                    ),
                    investigation_source_adapter_factory=lambda: adapter,
                )
            self.assertEqual(result, 0, amendment_stderr.getvalue())
            self.assertEqual(second_model.calls, 3)
            self.assertEqual(adapter.contact_count, 4)

            before = terminal_journey.path.read_bytes()
            reopened_stdout = StringIO()

            def fail_model_reopen(_loaded: object, _selected: object) -> object:
                raise AssertionError("hostile reopen called the model factory")

            def fail_source_reopen() -> object:
                raise AssertionError("hostile reopen called the source factory")

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result = main(
                    [
                        "answer",
                        "--journey-file",
                        str(terminal_journey.path),
                        "--request-event-id",
                        first_request,
                        "--response-file",
                        str(amendment_file),
                    ],
                    investigation_model_factory=fail_model_reopen,
                    investigation_source_adapter_factory=fail_source_reopen,
                )
            self.assertEqual(result, 0)
            self.assertEqual(terminal_journey.path.read_bytes(), before)
            self.assertEqual(adapter.contact_count, 4)

            with (
                redirect_stdout(reopened_stdout),
                redirect_stderr(StringIO()),
            ):
                result = main(
                    [
                        "observe",
                        "--journey-file",
                        str(terminal_journey.path),
                        "--rich",
                    ],
                    investigation_model_factory=fail_model_reopen,
                    investigation_source_adapter_factory=fail_source_reopen,
                )
            self.assertEqual(result, 0)
            self.assertEqual(terminal_journey.path.read_bytes(), before)
            reopened = json.loads(reopened_stdout.getvalue())
            self.assertEqual(reopened["status"], "investigation_review_pending")
            self.assertNotIn("cli-secret-sentinel", reopened_stdout.getvalue())
            amended = project_investigation(
                read_evidence(Journey.open(terminal_journey.path))
            )
            self.assertEqual(len(amended.attempts), 2)
            self.assertEqual(len(amended.review_receipts), 1)
            self.assertEqual(amended.review_receipts[0].request_event_id, first_request)
            self.assertEqual(
                amended.review_receipts[0].response,
                ReviseInvestigationResponse.model_validate_json(
                    amendment_file.read_text(encoding="utf-8")
                ),
            )
            proposal = reopened["investigation"]["proposals"][-1]
            self.assertEqual(proposal["revision_number"], 2)
            self.assertEqual(
                proposal["supersedes_proposal_id"], first_proposal["proposal_id"]
            )
            pending = reopened["investigation"]["pending_request"]
            self.assertNotEqual(pending["request_event_id"], first_request)
            approve_file = root / "approve.json"
            approve_file.write_text(
                canonical_json(
                    {
                        "action": "approve",
                        "proposal_id": proposal["proposal_id"],
                    }
                ),
                encoding="utf-8",
            )
            approve_stderr = StringIO()
            store = RunOutputStore()
            with (
                use_run_advancement_dependencies(
                    environ=TABLE_READING_ENVIRON,
                    investigation_model_factory=TableReaderModel().model,
                    investigation_vcf_source_factory=store,
                ),
                redirect_stdout(StringIO()),
                redirect_stderr(approve_stderr),
            ):
                result = main(
                    [
                        "answer",
                        "--journey-file",
                        str(terminal_journey.path),
                        "--request-event-id",
                        pending["request_event_id"],
                        "--response-file",
                        str(approve_file),
                    ]
                )
            # The approval is durable and continues into observation: the Run
            # supplies its own verified VCF, BAM and tools, one download each.
            self.assertEqual(result, 0, approve_stderr.getvalue())
            self.assertEqual(len(store.gets), 2)
            final = observe(ObserveRun(journey_file=str(terminal_journey.path)))
            self.assertEqual(final.status, "finalization_review_pending")
            self.assertEqual(
                [
                    item.status
                    for item in read_investigation(Journey.open(terminal_journey.path))
                    .collections[-1]
                    .evidence.item_states
                ],
                ["completed"],
            )
            self.assertEqual(
                final.investigation.authority.proposal_id,
                proposal["proposal_id"],
            )
            self.assertEqual(adapter.contact_count, 4)


class RetrievedSourceAuthorityTests(unittest.TestCase):
    """A retrieved source names the authority its operation belongs to (#352)."""

    def _created(self, operation):
        return RetrievedSource.create(
            operation=operation,
            request_identity="https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358",
            status_code=200,
            observed_at="2026-10-15T07:03:26Z",
            elapsed_time_ms=12,
            raw_response=b'{"refsnp_id": "429358"}',
        )

    def test_an_ncbi_source_keeps_the_identity_it_had_before_eva_joined(self):
        source = self._created("fetch_refsnp")

        # Computed at c0550c67, before EVA joined the operations.
        self.assertEqual(
            source.source_id,
            "retrieved-source:"
            "cecfd4115bfea27c37ea85dbfef653a74b7e27d5916191ffe0d1a0ca2352b749",
        )
        self.assertEqual(source.authority, "ncbi")

    def test_each_eva_operation_is_retrieved_from_eva(self):
        for operation in (
            "eva_clustered_variant",
            "eva_submitted_variants",
            "eva_clustered_variants_at",
        ):
            with self.subTest(operation=operation):
                self.assertEqual(self._created(operation).authority, "eva")

    def test_an_operation_under_another_sources_authority_is_refused(self):
        material = self._created("fetch_refsnp").model_dump(
            mode="json", exclude={"source_id"}
        )
        material["authority"] = "eva"
        forged = {
            "source_id": f"retrieved-source:{canonical_json_digest(material)}",
            **material,
        }

        with self.assertRaisesRegex(
            ValidationError, "fetch_refsnp is an ncbi operation"
        ):
            RetrievedSource.model_validate(forged)


if __name__ == "__main__":
    unittest.main()
