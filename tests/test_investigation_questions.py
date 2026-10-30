"""Early request questions through the real Agent and application Review seams."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.application.queries import read_investigation
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.model_endpoint_fakes import corrections as sent_corrections
from tests.sequential_investigation_support import (
    SourceResponsesOnly,
    begin_in_pass,
    run_attempt,
    target_reading_reply,
)


class InvestigationQuestionTests(unittest.TestCase):
    def test_source_reviews_retry_independently_without_reprojecting_an_older_fault(
        self,
    ):
        self._recover_independent_sources("retry")

    def test_stopping_all_source_members_settles_each_and_leaves_no_runnable_work(
        self,
    ):
        self._recover_independent_sources("stop")

    def _recover_independent_sources(self, action):
        from bio_neb.agentic_journey.operational_review import (
            answer_operational_review,
            project_operational_review,
        )
        from bio_neb.failure import ProviderError
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )

        calls = []

        async def respond(messages, info):
            calls.append(messages)
            name, arguments = (
                ("fetch_refsnp", {"rsid": ("rs429358", "rs7412")[len(calls) - 1]})
                if len(calls) <= 2
                else (
                    "request_investigation_information",
                    {
                        "question": "Which allele?",
                        "missing_information": "The allele is unspecified.",
                        "asks_about": "target",
                    },
                )
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, arguments, tool_call_id=f"source-member-{len(calls)}"
                    )
                ]
            )

        class InitiallyRejected(SourceResponsesOnly):
            rejected = True

            def fetch_refsnp(self, rsid):
                if self.rejected:
                    self.operations.append("fetch_refsnp:" + rsid)
                    raise ProviderError("rejected", "Source denied this request.")
                return super().fetch_refsnp(rsid)

        with tempfile.TemporaryDirectory() as directory:
            model = ScriptedModelEndpoint(respond)
            source = InitiallyRejected()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            _, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                source,
                assertion=assertion,
            )
            # One Review for the source's cause: the second member joined it
            # before its call contacted the source.
            requests = project_operational_review(read_evidence(journey)).requests
            self.assertEqual(len(requests), 2)
            self.assertEqual(
                (requests[1].joins_event_id, requests[1].joined_before_contact),
                (requests[0].request_event_id, True),
            )
            self.assertEqual(source.operations, ["fetch_refsnp:rs429358"])
            source.rejected = False
            assignment = investigation_assignment(
                model_factory=model, source_adapter_factory=lambda: source
            )
            for request in reversed(requests):
                answer = {
                    "request_id": request.request.request_id,
                    "action": action,
                    "by": "researcher",
                }
                receipt, fresh = answer_operational_review(
                    journey, request.request_event_id, answer
                )
                self.assertTrue(fresh)
                result = (
                    assignment.run(
                        journey,
                        environ={"LOCAL_MODEL_KEY": "offline"},
                        operational_review_response_event_id=receipt.response_event_id,
                    )
                    if action == "retry"
                    else read_investigation(Journey.open(journey.path)).investigation
                )
                before = journey.path.read_bytes()
                self.assertEqual(
                    answer_operational_review(
                        journey, request.request_event_id, answer
                    ),
                    (receipt, False),
                )
                self.assertEqual(journey.path.read_bytes(), before)
            if action == "stop":
                self.assertEqual(len(calls), 2)
                # Each member's stop settles that member alone: no member is
                # left to run, and the collection is ready to close with both
                # accounted as rejected requests.
                self.assertEqual(result.status, "investigation_work_ready")
                self.assertEqual(result.eligible_member_ids, ())
                self.assertEqual(len(source.operations), 1)
                self.assertFalse(
                    any(
                        r.pending
                        for r in project_operational_review(
                            read_evidence(journey)
                        ).requests
                    )
                )
                return
            self.assertEqual(len(calls), 4)
            # Each item answered on its own re-sends once.
            self.assertEqual(
                source.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_refsnp:rs7412",
                    "fetch_refsnp:rs429358",
                ],
            )
            self.assertEqual(
                {
                    r.question.collection_member_id
                    for r in result.review_requests
                    if r.pending
                },
                {r.member_id for r in requests},
            )
            self.assertIsNone(
                project_operational_review(read_evidence(journey)).pending_request
            )
            events = read_evidence(journey)
            self.assertEqual(
                sum(e["event_type"] == "AgentFailureReviewRequested" for e in events), 2
            )
            bounds = [
                e["payload"]
                for e in events
                if e["event_type"] == "InvestigationAttemptBound"
            ]
            # Each member is its own Attempt: a recovery continues its own
            # member's held Attempt, not the collection's latest one.
            by_attempt = {bound["attempt_id"]: bound for bound in bounds}
            self.assertEqual(len(bounds), 4)
            for bound in bounds[2:]:
                self.assertEqual(
                    by_attempt[bound["continued_attempt_id"]]["collection_member_id"],
                    bound["collection_member_id"],
                )
            retained = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, result
            )
            self.assertEqual(journey.path.read_bytes(), retained)

    def test_a_source_review_answers_each_item_and_a_new_cause_opens_its_own(self):
        """Owner, #476 alignment G4.1, with the orchestrator's answer and the
        owner's amendment under the grant (#477 6483312280): the second item
        joins the open source Review with no source contact; one item is retried
        and one stopped; the retried item's new failure class opens its own
        Review, and the first closes with its last answered item."""
        from bio_neb.agentic_journey.operational_review import (
            answer_operational_review,
            project_operational_review,
        )
        from bio_neb.failure import ProviderError
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )

        calls = []

        async def respond(messages, info):
            calls.append(messages)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_refsnp",
                        {"rsid": ("rs429358", "rs7412")[len(calls) - 1]},
                        tool_call_id=f"source-member-{len(calls)}",
                    )
                ]
            )

        class FailingSource(SourceResponsesOnly):
            failure_class = "rejected"

            def fetch_refsnp(self, rsid):
                self.operations.append("fetch_refsnp:" + rsid)
                raise ProviderError(self.failure_class, "Source failed this request.")

        with tempfile.TemporaryDirectory() as directory:
            model = ScriptedModelEndpoint(respond)
            source = FailingSource()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            _, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                source,
                assertion=assertion,
            )
            first, joined = project_operational_review(read_evidence(journey)).requests
            self.assertEqual(
                (joined.joins_event_id, joined.joined_before_contact),
                (first.request_event_id, True),
            )
            self.assertEqual(source.operations, ["fetch_refsnp:rs429358"])

            def answer(request, action):
                return answer_operational_review(
                    journey,
                    request.request_event_id,
                    {
                        "request_id": request.request.request_id,
                        "action": action,
                        "by": "researcher",
                        **({"reason": "Leave this item."} if action == "stop" else {}),
                    },
                )[0]

            # The first item is retried; its source now fails another way.
            source.failure_class = "semantic"
            receipt = answer(first, "retry")
            investigation_assignment(
                model_factory=model, source_adapter_factory=lambda: source
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                operational_review_response_event_id=receipt.response_event_id,
            )
            contacted = list(source.operations)
            self.assertEqual(contacted[0], "fetch_refsnp:rs429358")
            self.assertNotIn("fetch_refsnp:rs7412", contacted)
            reviews = project_operational_review(read_evidence(journey)).requests
            new = [
                r
                for r in reviews
                if r.request_event_id
                not in {first.request_event_id, joined.request_event_id}
            ]
            # A different failure class is another cause: its own Review.
            self.assertEqual(len(new), 1)
            self.assertIsNone(new[0].joins_event_id)
            self.assertNotEqual(new[0].fault_key, first.fault_key)
            # The joined item is stopped on its own and contacts nothing.
            answer(joined, "stop")
            self.assertEqual(source.operations, contacted)
            final = project_operational_review(read_evidence(journey))
            self.assertEqual(
                {r.request_event_id for r in final.requests if r.pending},
                {new[0].request_event_id},
            )

    def test_an_unreadable_provider_body_is_retained_and_read_as_malformed(self):
        # A success whose body is not one RefSNP record is the source's answer. It
        # does not hold the member for an operational Review with no domain result;
        # it is retained, typed malformed and shown to the model in its source
        # answers.
        from bio_neb.agentic_journey.operational_review import (
            project_operational_review,
        )

        for body, shown in ((b"not-json", ("text", "not-json")), (b"[]", ("json", []))):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                calls = []

                async def respond(messages, info, calls=calls):
                    calls.append(messages)
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "fetch_refsnp",
                                {"rsid": "rs429358"},
                                tool_call_id=f"malformed-source-{len(calls)}",
                            )
                        ]
                    )

                source = SourceResponsesOnly(refsnp_body=body)
                _, journey = run_attempt(
                    Path(directory),
                    SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                    source,
                )
                events = read_evidence(journey)
                self.assertIsNone(project_operational_review(events).pending_request)
                self.assertEqual(source.operations, ["fetch_refsnp:rs429358"])
                [fact] = [
                    e for e in events if e["event_type"] == "RefSnpToolResultRecorded"
                ]
                self.assertEqual(fact["payload"]["result"]["status"], "malformed")
                [answer] = fact["payload"]["source_answers"]
                self.assertEqual((answer["media"], answer["body"]), shown)
                # The next request carried the return to the model.
                self.assertGreater(len(calls), 1)
                self.assertNotIn("RunFailed", [e["event_type"] for e in events])
                retained = journey.path.read_bytes()
                read_investigation(Journey.open(journey.path))
                self.assertEqual(journey.path.read_bytes(), retained)

    def test_source_fault_holds_one_member_while_another_reaches_its_question(self):
        self._source_fault_holds_independent_question()

    def test_source_fault_suppresses_co_emitted_output_before_independent_member(self):
        self._source_fault_holds_independent_question(extra_output=True)

    def _source_fault_holds_independent_question(self, *, extra_output=False):
        from bio_neb.agentic_journey.operational_review import (
            project_operational_review,
        )
        from bio_neb.failure import ProviderError
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )

        calls = []

        async def respond(messages, info):
            calls.append(messages)
            name, args = (
                ("fetch_refsnp", {"rsid": "rs429358"})
                if len(calls) == 1
                else (
                    "request_investigation_information",
                    {
                        "question": "Which allele for this second request?",
                        "missing_information": "The allele was not specified.",
                        "asks_about": "target",
                    },
                )
            )
            parts = [ToolCallPart(name, args, tool_call_id=f"member-{len(calls)}")]
            if len(calls) == 1 and extra_output:
                output = dict.fromkeys(
                    (
                        "rsid",
                        "assembly_accession",
                        "assembly_name",
                        "contig",
                        "one_based_position",
                        "ref",
                        "intended_action",
                    )
                )
                output.update(
                    outcome="unresolved",
                    aliases=[],
                    unresolved_conflicts=["Source not read."],
                )
                parts.append(
                    ToolCallPart(
                        info.output_tools[0].name,
                        output,
                        tool_call_id="unexecuted-output",
                    )
                )
            return ModelResponse(parts=parts)

        class RejectedSource(SourceResponsesOnly):
            def fetch_refsnp(self, rsid):
                self.operations.append("fetch_refsnp:" + rsid)
                raise ProviderError("rejected", "Source denied this request.")

        with tempfile.TemporaryDirectory() as directory:
            source = RejectedSource()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                source,
                assertion=assertion,
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(source.operations, ["fetch_refsnp:rs429358"])
            events = read_evidence(journey)
            operational = project_operational_review(events)
            self.assertEqual(len(operational.requests), 1)
            self.assertEqual(len(projection.review_requests), 1)
            self.assertNotEqual(
                operational.pending_request.member_id,
                projection.pending_request.question.collection_member_id,
            )
            self.assertEqual(projection.eligible_member_ids, ())
            self.assertNotIn("RunFailed", [e["event_type"] for e in events])
            before = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, projection
            )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_a_model_fault_holds_every_member_at_one_review(self):
        """Owner, #476 alignment G4.1, with the orchestrator's answer under the
        owner's grant (#477): the refusal is shown once; the same refusal asks
        the Review, and the other member joins it at its next call, unsent."""
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )
        from bio_neb.failure import ProviderError
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )

        calls = []

        async def respond(messages, info):
            calls.append(messages)
            if len(calls) <= 2:
                raise ModelHTTPError(402, "test", body="rejected")
            name, args = (
                ("fetch_refsnp", {"rsid": "rs429358"})
                if len(calls) == 1
                else (
                    "request_investigation_information",
                    {
                        "question": "Which allele for this second request?",
                        "missing_information": "The allele was not specified.",
                        "asks_about": "target",
                    },
                )
            )
            return ModelResponse(
                parts=[ToolCallPart(name, args, tool_call_id=f"member-{len(calls)}")]
            )

        class RejectedSource(SourceResponsesOnly):
            def fetch_refsnp(self, rsid):
                self.operations.append("fetch_refsnp:" + rsid)
                raise ProviderError("rejected", "Source denied this request.")

        with tempfile.TemporaryDirectory() as directory:
            source = RejectedSource()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                source,
                assertion=assertion,
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(source.operations, [])
            events = read_evidence(journey)
            availability = project_model_availability_review(events)
            lead, joined = availability.requests
            self.assertEqual(
                (joined.joins_event_id, joined.joined_before_sending),
                (lead.request_event_id, True),
            )
            self.assertNotEqual(lead.member_id, joined.member_id)
            self.assertEqual(
                availability.pending_request.request_event_id, lead.request_event_id
            )
            self.assertEqual(len(projection.review_requests), 0)
            self.assertEqual(projection.eligible_member_ids, ())
            self.assertNotIn("RunFailed", [e["event_type"] for e in events])
            before = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, projection
            )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_one_model_review_retries_every_member_from_its_own_frontier(self):
        """Owner, #476 alignment G4.1 (#477 6483312280): the answer-all retry
        releases every member of the Review, each from its own frontier: a member
        that failed resends its pending request, one that joined unsent starts
        fresh."""
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )
        from bio_neb.application.commands import (
            RespondModelAvailabilityReviewCommand,
            respond_model_availability_review,
        )
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )

        calls = []

        answered = []

        async def respond(messages, info):
            calls.append(messages)
            if not answered:
                raise ModelHTTPError(402, "test")
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

        with tempfile.TemporaryDirectory() as directory:
            source = SourceResponsesOnly()
            model = ScriptedModelEndpoint(respond)
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            from bio_neb.investigation.investigation_commands import (
                BeginInvestigationCommand,
            )
            from tests.sequential_investigation_support import forked_child

            root = Path(directory)
            request = {
                "asserted_by": "researcher",
                "requests": [
                    {"request": r.request_text, "target": {"rsid": r.rsid}}
                    for r in assertion.requests
                ],
            }
            journey = forked_child(root, investigation_request=request)
            request_file = root / "request.json"
            request_file.write_text(json.dumps(request))
            begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
                source_adapter_factory=lambda: source,
            )
            requests = project_model_availability_review(
                read_evidence(Journey.open(journey.path))
            ).requests
            lead, joined = requests
            self.assertEqual(joined.joins_event_id, lead.request_event_id)
            assignment = investigation_assignment(
                model_factory=model, source_adapter_factory=lambda: source
            )
            answered.append(True)
            respond_model_availability_review(
                RespondModelAvailabilityReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=lead.request_event_id,
                    response={"action": "retry"},
                    every_item=True,
                ),
                environ={"LOCAL_MODEL_KEY": "secret"},
                attempt=assignment,
            )
            observed = read_investigation(Journey.open(journey.path))
            self.assertEqual(source.operations, [])
            self.assertEqual(
                {
                    r.question.collection_member_id
                    for r in observed.investigation.review_requests
                    if r.pending
                },
                {r.member_id for r in requests},
            )
            events = read_evidence(Journey.open(journey.path))
            # The Fork child's own requests; inherited planning keeps its parent identity.
            logical = [
                e
                for e in events
                if e["event_type"] == "ModelRequestRecorded"
                and e["journey_id"] == journey.journey_id
            ]
            # Each retry of a failed request resends it exactly; a member that
            # joined unsent starts with no inherited request.
            by_id = {e["event_id"]: e for e in logical}
            retries = [
                e for e in logical if e["payload"].get("continuation_request_event_id")
            ]
            self.assertEqual(
                len(retries), sum(not r.joined_before_sending for r in requests)
            )
            for retry in retries:
                self.assertEqual(
                    retry["payload"]["compatible_adapter_evidence"]["request"],
                    by_id[retry["payload"]["continuation_request_event_id"]]["payload"][
                        "compatible_adapter_evidence"
                    ]["request"],
                )
            bounds = [
                e["payload"]
                for e in events
                if e["event_type"] == "InvestigationAttemptBound"
            ]
            # A retry continues its own member's Attempt.
            by_attempt = {bound["attempt_id"]: bound for bound in bounds}
            continued = [b for b in bounds if b["continued_attempt_id"] is not None]
            self.assertTrue(continued)
            for bound in continued:
                self.assertEqual(
                    by_attempt[bound["continued_attempt_id"]]["collection_member_id"],
                    bound["collection_member_id"],
                )
            self.assertEqual(observed.investigation.eligible_member_ids, ())

    def test_one_model_stop_ends_every_member_with_the_reviews_cause(self):
        """Owner, #476 alignment G4.1, with the orchestrator's answer under the
        owner's grant (#477 6483312280): the answer-all stop answers every member
        of the Review with its reason; nothing is sent after it."""
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.application.advancement import (
            AnswerReview,
            ModelAvailabilityReviewAnswer,
            advance,
        )
        from bio_neb.application.queries import ObserveRun, observe
        from bio_neb.investigation.investigation_commands import (
            BeginInvestigationCommand,
        )
        from tests.sequential_investigation_support import forked_child

        calls = []

        async def respond(messages, info):
            # The endpoint refuses every request until the researcher answers.
            calls.append(messages)
            raise ModelHTTPError(402, "test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "asserted_by": "researcher",
                "requests": [
                    {"request": f"Investigate {rsid}", "target": {"rsid": rsid}}
                    for rsid in ("rs429358", "rs7412", "rs1234")
                ],
            }
            journey = forked_child(root, investigation_request=request)
            request_file = root / "request.json"
            request_file.write_text(json.dumps(request))
            model = ScriptedModelEndpoint(respond)
            sources = SourceResponsesOnly()
            begin_in_pass(
                BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                ),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
                source_adapter_factory=lambda: sources,
            )
            initial = observe(ObserveRun(journey_file=str(journey.path)))
            lead, *joined = initial.model_availability_review.requests
            self.assertEqual(len(joined), 2)
            self.assertEqual(
                {request.joins_event_id for request in joined}, {lead.request_event_id}
            )
            answer = AnswerReview(
                journey_file=str(journey.path),
                request_event_id=lead.request_event_id,
                answer=ModelAvailabilityReviewAnswer(
                    kind="model_availability",
                    response={"action": "stop", "reason": "The endpoint is down."},
                    every_item=True,
                ),
            )
            stopped = advance(answer)
            self.assertEqual(
                sum(r.pending for r in stopped.model_availability_review.requests), 0
            )
            stops = [
                e["payload"]["reason"]
                for e in read_evidence(Journey.open(journey.path))
                if e["event_type"] == "ModelAvailabilityStopped"
            ]
            self.assertEqual(stops, ["The endpoint is down."] * 3)
            before = journey.path.read_bytes()
            self.assertEqual(advance(answer), stopped)
            self.assertEqual(journey.path.read_bytes(), before)
            # Each member's refusal was shown once; nothing was sent after a stop.
            sent = len(calls)
            self.assertLessEqual(sent, 6)
            self.assertEqual(advance(answer), stopped)
            self.assertEqual(len(calls), sent)
            self.assertEqual(sources.operations, [])
            self.assertFalse(
                any(
                    e["event_type"] == "RunFailed"
                    for e in read_evidence(Journey.open(journey.path))
                )
            )
            self.assertEqual(
                observe(ObserveRun(journey_file=str(journey.path))), stopped
            )

    def test_early_question_ends_attempt_and_refuses_later_source_call(self):
        requests = []

        async def respond(messages, info):
            requests.append(info)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which allele do you want investigated?",
                            "missing_information": "The requested allele is unspecified.",
                            "asks_about": "target",
                        },
                        tool_call_id="ask-allele",
                    ),
                    ToolCallPart(
                        "fetch_refsnp",
                        {"rsid": "rs429358"},
                        tool_call_id="after-question",
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
                invocation_overrides={"model_requests": 1},
            )
            self.assertEqual(projection.status, "review_pending")
            self.assertEqual(projection.attempts[-1].outcome, "question_pending")
            self.assertEqual(len(requests), 1)
            self.assertEqual(sources.operations, [])
            self.assertEqual(projection.proposals, ())
            question = projection.pending_request.question
            self.assertEqual(question.request.rsid, "rs429358")
            self.assertEqual(question.revision_number, 1)
            self.assertEqual(
                question.question, "Which allele do you want investigated?"
            )
            evidence = read_evidence(Journey.open(journey.path))
            refusal = next(
                e
                for e in evidence
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_call_id"] == "after-question"
            )
            self.assertEqual(refusal["payload"]["status"], "invalid")
            self.assertIn("researcher_question_opened", refusal["payload"]["errors"])
            # The refused call is told why, as planning's are (the member's
            # correction had no reason).
            (feedback,) = [
                e["payload"]["feedback"]
                for e in evidence
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
                and e["payload"]["feedback"]
                and "researcher_question_opened" in e["payload"]["feedback"]
            ]
            self.assertIn(
                '`researcher question` is "request_investigation_information": An '
                "earlier call in this response opened this researcher question",
                feedback,
            )
            before = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, projection
            )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_exact_clarification_resumes_with_retained_source_and_replay_is_inert(self):
        from bio_neb.investigation.investigation import ClarifyInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            InvestigationApplicationRefused,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import source_basis

        calls = []

        async def respond(messages, info):
            # The clarify answer's words are read first.
            if (reply := target_reading_reply(messages, info)) is not None:
                return reply
            calls.append(messages)
            if len(calls) == 1:
                name, args = "fetch_refsnp", {"rsid": "rs429358"}
            elif len(calls) == 2:
                name, args = (
                    "request_investigation_information",
                    {
                        "question": "Which allele should I investigate?",
                        "missing_information": "Please confirm your intended allele.",
                        "asks_about": "target",
                    },
                )
            elif len(calls) == 3:
                name, args = (
                    "fetch_assembly_report",
                    {
                        "refsnp_event_id": source_basis(info)["refsnp_event_id"],
                        "assembly_accession": "GCF_000001405.40",
                    },
                )
            else:
                name, args = (
                    "compile_proposal",
                    {
                        **source_basis(info),
                        "outcome": "complete",
                    },
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, json.dumps(args), tool_call_id=f"call-{len(calls)}"
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            source = SourceResponsesOnly()
            model = ScriptedModelEndpoint(respond)
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                source,
                invocation_overrides={"model_requests": 4},
            )
            question = pending.pending_request.question
            original = [
                e
                for e in read_evidence(Journey.open(journey.path))
                if e["event_type"] == "RefSnpToolResultRecorded"
            ]
            response = ClarifyInvestigationResponse(
                action="clarify",
                collection_member_id=question.collection_member_id,
                revision_number=question.revision_number,
                instruction="Investigate the T to C allele.",
                asserted_by="delegated researcher",
            )
            command = RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=pending.pending_request.request_event_id,
                response=response,
            )
            answered = respond_investigation_review(command)
            self.assertEqual(answered.status, "investigation_work_ready")
            before = journey.path.read_bytes()
            self.assertEqual(respond_investigation_review(command), answered)
            self.assertEqual(journey.path.read_bytes(), before)
            with self.assertRaises(InvestigationApplicationRefused) as conflict:
                respond_investigation_review(
                    command.model_copy(
                        update={
                            "response": response.model_copy(
                                update={"instruction": "A different answer"}
                            )
                        }
                    )
                )
            self.assertEqual(conflict.exception.code, "investigation_review_conflict")
            self.assertEqual(journey.path.read_bytes(), before)
            completed = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
                source_adapter_factory=lambda: source,
            )
            self.assertEqual(completed.status, "review_pending")
            self.assertIsNone(completed.pending_request.question)
            self.assertEqual(
                source.operations,
                ["fetch_refsnp:rs429358", "fetch_assembly_report:GCF_000001405.40"],
            )
            self.assertEqual(
                [
                    e
                    for e in read_evidence(Journey.open(journey.path))
                    if e["event_type"] == "RefSnpToolResultRecorded"
                ],
                original,
            )
            self.assertEqual(
                completed.attempts[-1].effective_invocation_policy.model_requests, 4
            )
            self.assertEqual(len(completed.proposals[-1].work_items), 1)

    def test_pending_member_does_not_hold_sibling_and_answer_preserves_sibling(self):
        from bio_neb.investigation.investigation import (
            ClarifyInvestigationResponse,
            InvestigationRequestItem,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import (
            SequentialSourceResponses,
            source_basis,
        )

        calls = 0

        async def respond(messages, info):
            # The clarify answer's words are read first.
            if (reply := target_reading_reply(messages, info)) is not None:
                return reply
            nonlocal calls
            calls += 1
            if calls == 1:
                name, args = (
                    "request_investigation_information",
                    {
                        "question": "Which allele for rs429358?",
                        "missing_information": "The allele is unspecified.",
                        "asks_about": "target",
                    },
                )
            else:
                rsid, position = (
                    ("rs7412", 44908822) if calls <= 4 else ("rs429358", 44908684)
                )
                phase = (calls - 2) % 3
                if phase == 0:
                    name, args = "fetch_refsnp", {"rsid": rsid}
                elif phase == 1:
                    name, args = (
                        "fetch_assembly_report",
                        {
                            "refsnp_event_id": source_basis(info)["refsnp_event_id"],
                            "assembly_accession": "GCF_000001405.40",
                        },
                    )
                else:
                    name, args = (
                        "compile_proposal",
                        {
                            **source_basis(info),
                            "outcome": "complete",
                        },
                    )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, json.dumps(args), tool_call_id=f"sibling-{calls}"
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = SequentialSourceResponses()
            model = ScriptedModelEndpoint(respond)
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-research-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                sources,
                assertion=assertion,
            )
            # The sibling is its own Attempt in the same batch: it settles while
            # the asking member waits, with no separate continuation.
            self.assertEqual(pending.status, "review_pending")
            first_question = pending.pending_request
            continued = pending
            self.assertEqual(continued.status, "review_pending")
            self.assertEqual(continued.pending_request, first_question)
            self.assertEqual(calls, 4)
            sibling_events = [
                e
                for e in read_evidence(Journey.open(journey.path))
                if e["event_type"]
                in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
            ]
            self.assertEqual(len(sibling_events), 2)
            question = first_question.question
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=first_question.request_event_id,
                    response=ClarifyInvestigationResponse(
                        action="clarify",
                        collection_member_id=question.collection_member_id,
                        revision_number=1,
                        instruction="Investigate T to C",
                        asserted_by="researcher",
                    ),
                )
            )
            completed = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
                source_adapter_factory=lambda: sources,
            )
            self.assertEqual(
                [
                    item.target.one_based_position
                    for item in completed.proposals[-1].work_items
                ],
                [44908684, 44908822],
            )
            self.assertEqual(
                sources.operations,
                [
                    "fetch_refsnp:rs7412",
                    "fetch_assembly_report:GCF_000001405.40",
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            self.assertEqual(
                [
                    e
                    for e in read_evidence(Journey.open(journey.path))
                    if e["event_id"] in {event["event_id"] for event in sibling_events}
                ],
                sibling_events,
            )

    def test_rejection_retains_request_accounting_without_source_or_work(self):
        from bio_neb.investigation.investigation import RejectInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            RespondInvestigationReviewCommand,
            respond_investigation_review,
        )

        async def respond(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which allele?",
                            "missing_information": "The request leaves the allele unspecified.",
                            "asks_about": "target",
                        },
                        tool_call_id="reject-question",
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            pending, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            request = pending.pending_request
            result = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=request.request_event_id,
                    response=RejectInvestigationResponse(
                        action="reject",
                        collection_member_id=request.question.collection_member_id,
                        revision_number=1,
                        asserted_by="researcher",
                        reason="Do not pursue this request.",
                    ),
                )
            )
            self.assertEqual(sources.operations, [])
            self.assertEqual(len(result.attempts), 1)
            self.assertFalse(
                next(
                    r
                    for r in result.review_requests
                    if r.request_event_id == request.request_event_id
                ).pending
            )
            proposal = result.proposals[-1]
            self.assertEqual(proposal.work_items, ())
            self.assertEqual(proposal.unresolved_request_member_ids, ())
            member = proposal.request_members[0]
            self.assertIsNone(member.refsnp)
            self.assertEqual(member.request.rsid, "rs429358")
            self.assertEqual(member.rejection.reason, "Do not pursue this request.")
            self.assertEqual(
                member.rejection.request_event_id, request.request_event_id
            )
            self.assertEqual(result.fact_slots[0].verified_value, None)
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, result
            )

    def test_native_answers_address_independent_questions_and_retain_rejection(self):
        import contextlib
        import io

        from bio_neb.application.advancement import (
            BeginInvestigation,
            ObserveRun,
            advance,
            observe,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.reviews import pending_reviews
        from bio_neb.cli import main
        from tests.sequential_investigation_support import (
            SequentialSourceResponses,
            forked_child,
            source_basis,
        )

        calls = []

        async def respond(messages, info):
            # The clarify answer's words are read first.
            if (reply := target_reading_reply(messages, info)) is not None:
                return reply
            calls.append(messages)
            number = len(calls)
            if number <= 2:
                name, args = (
                    "request_investigation_information",
                    {
                        "question": "Which allele should I investigate?",
                        "missing_information": "The requested allele is unspecified.",
                        "asks_about": "target",
                    },
                )
            elif number == 3:
                name, args = "fetch_refsnp", {"rsid": "rs7412"}
            elif number == 4:
                name, args = (
                    "fetch_assembly_report",
                    {
                        "refsnp_event_id": source_basis(info)["refsnp_event_id"],
                        "assembly_accession": "GCF_000001405.40",
                    },
                )
            else:
                name, args = (
                    "compile_proposal",
                    {
                        **source_basis(info),
                        "outcome": "complete",
                    },
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, json.dumps(args), tool_call_id=f"public-{number}"
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "asserted_by": "researcher",
                "requests": [
                    {"request": f"Investigate {rsid}", "target": {"rsid": rsid}}
                    for rsid in ("rs429358", "rs7412")
                ],
            }
            journey = forked_child(root, investigation_request=request)
            request_file = root / "request.json"
            request_file.write_text(json.dumps(request))
            sources = SequentialSourceResponses()
            model = ScriptedModelEndpoint(respond)
            with use_run_advancement_dependencies(
                environ={"LOCAL_MODEL_KEY": "offline"},
                investigation_model_factory=model,
                investigation_source_adapter_factory=lambda: sources,
            ):
                pending = advance(
                    BeginInvestigation(
                        journey_file=str(journey.path),
                        request_files=(str(request_file),),
                    )
                )
                questions = pending_reviews(pending)
                self.assertEqual(len(questions), 2)
                self.assertEqual(sources.operations, [])
                # Members ask concurrently: pick each question by its
                # member's request, not by recording order.
                second, first = sorted(
                    questions,
                    key=lambda r: r.request.question.request.rsid != "rs7412",
                )
                response_file = root / "response.json"
                answer = {
                    "action": "clarify",
                    "collection_member_id": second.request.question.collection_member_id,
                    "revision_number": 1,
                    "instruction": "Investigate T to C",
                    "asserted_by": "delegated researcher",
                }
                response_file.write_text(json.dumps(answer))
                argv = [
                    "answer",
                    "--journey-file",
                    str(journey.path),
                    "--request-event-id",
                    second.request_event_id,
                    "--response-file",
                    str(response_file),
                ]
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(main(argv), 0)
                remaining = observe(ObserveRun(journey_file=str(journey.path)))
                self.assertEqual(
                    [r.request_event_id for r in pending_reviews(remaining)],
                    [first.request_event_id],
                )
                self.assertEqual(len(calls), 5)
                before = journey.path.read_bytes()
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(main(argv), 0)
                self.assertEqual(journey.path.read_bytes(), before)
                self.assertEqual(len(calls), 5)
                for change in (
                    {"collection_member_id": "another-member"},
                    {"revision_number": 2},
                ):
                    response_file.write_text(
                        json.dumps(
                            {
                                **answer,
                                "collection_member_id": first.request.question.collection_member_id,
                                **change,
                            }
                        )
                    )
                    refused = io.StringIO()
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(refused),
                    ):
                        self.assertEqual(
                            main(
                                [
                                    *argv[:3],
                                    "--request-event-id",
                                    first.request_event_id,
                                    "--response-file",
                                    str(response_file),
                                ]
                            ),
                            2,
                        )
                    self.assertIn("investigation_review_stale", refused.getvalue())
                    self.assertEqual(journey.path.read_bytes(), before)
                    self.assertEqual(len(calls), 5)
                response_file.write_text(
                    json.dumps(
                        {
                            "action": "reject",
                            "collection_member_id": first.request.question.collection_member_id,
                            "revision_number": 1,
                            "reason": "Do not pursue rs429358",
                            "asserted_by": "delegated researcher",
                        }
                    )
                )
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        main(
                            [
                                *argv[:3],
                                "--request-event-id",
                                first.request_event_id,
                                "--response-file",
                                str(response_file),
                            ]
                        ),
                        0,
                    )
                final = observe(ObserveRun(journey_file=str(journey.path)))
                self.assertEqual(
                    final.investigation.proposals[-1]
                    .request_members[0]
                    .rejection.reason,
                    "Do not pursue rs429358",
                )
                self.assertEqual(
                    [
                        item.target.one_based_position
                        for item in final.investigation.proposals[-1].work_items
                    ],
                    [44908822],
                )
                self.assertEqual(
                    sources.operations,
                    ["fetch_refsnp:rs7412", "fetch_assembly_report:GCF_000001405.40"],
                )
                self.assertEqual(len(calls), 5)

                from bio_neb.application.advancement import (
                    AnswerReview,
                    InvestigationReviewAnswer,
                )
                from bio_neb.investigation.investigation import (
                    StopInvestigationResponse,
                )

                request = final.current_review.request
                ended = advance(
                    AnswerReview(
                        journey_file=str(journey.path),
                        request_event_id=request.request_event_id,
                        answer=InvestigationReviewAnswer(
                            kind="investigation",
                            response=StopInvestigationResponse(
                                action="stop",
                                proposal_id=request.proposal.proposal_id,
                                reason="End the remaining scientific work.",
                            ),
                        ),
                    )
                )
                self.assertEqual(ended.status, "finalization_review_pending")
                basis = ended.finalization.pending_request.basis
                self.assertEqual(basis.outcome, "investigation_stopped")
                retained = json.dumps(
                    [entry.model_dump(mode="json") for entry in basis.accounting]
                )
                for exact in (
                    "Do not pursue rs429358",
                    "delegated researcher",
                    "Which allele should I investigate?",
                    final.investigation.proposals[-1].work_items[0].work_item_id,
                ):
                    self.assertIn(exact, retained)
                self.assertEqual(basis.admissions, ())
                self.assertEqual(len(calls), 5)
                self.assertEqual(
                    observe(ObserveRun(journey_file=str(journey.path))), ended
                )

    def test_source_before_question_in_same_response_survives_reopen(self):
        async def respond(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_refsnp",
                        {"rsid": "rs429358"},
                        tool_call_id="before-question",
                    ),
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which sample?",
                            "missing_information": "Sample scope is missing.",
                            "asks_about": "target",
                        },
                        tool_call_id="question",
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            self.assertEqual(projection.status, "review_pending")
            self.assertEqual(sources.operations, ["fetch_refsnp:rs429358"])
            self.assertEqual(len(projection.attempts[-1].source_result_event_ids), 1)
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, projection
            )

    def test_open_question_does_not_override_sibling_effort_review(self):
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )

        calls = 0

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            name, args = (
                (
                    "request_investigation_information",
                    {
                        "question": "Which sample?",
                        "missing_information": "Sample scope is missing.",
                        "asks_about": "target",
                    },
                )
                if calls != 2
                else ("fetch_refsnp", {"rsid": "rs7412"})
            )
            return ModelResponse(
                parts=[ToolCallPart(name, args, tool_call_id=f"hold-{calls}")]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            model = ScriptedModelEndpoint(respond)
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-research-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            # Both members run in the first batch, each in its own Attempt: one
            # asks, the other stops on its own spent allowance.
            held, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                sources,
                assertion=assertion,
                invocation_overrides={"model_requests": 1},
            )
            self.assertEqual(held.status, "effort_review_pending")
            self.assertEqual(held.eligible_member_ids, ())
            self.assertEqual(len([r for r in held.review_requests if r.pending]), 1)
            self.assertEqual(calls, 2)
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, held
            )

            from bio_neb.agentic_journey.effort import (
                ExtendAgentEffortResponse,
                project_effort_review,
                respond_effort_review,
            )
            from bio_neb.investigation.investigation import RejectInvestigationResponse
            from bio_neb.investigation.investigation_assignment import (
                investigation_assignment,
            )
            from bio_neb.investigation.investigation_commands import (
                RespondInvestigationReviewCommand,
                respond_investigation_review,
            )

            request = held.pending_request
            rejected = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=request.request_event_id,
                    response=RejectInvestigationResponse(
                        action="reject",
                        collection_member_id=request.question.collection_member_id,
                        revision_number=1,
                        reason="Do not pursue A",
                        asserted_by="researcher",
                    ),
                )
            )
            self.assertEqual(rejected.status, "effort_review_pending")
            self.assertEqual(rejected.proposals, ())
            current = Journey.open(journey.path)
            effort_request = project_effort_review(
                read_evidence(current)
            ).pending_request
            receipt = respond_effort_review(
                current,
                effort_request.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=1),
            )
            resumed = investigation_assignment(
                model_factory=model, source_adapter_factory=lambda: sources
            ).run(
                current,
                environ={"LOCAL_MODEL_KEY": "offline"},
                effort_review_response_event_id=receipt.response_event_id,
            )
            pending_members = [
                r.question.request.rsid for r in resumed.review_requests if r.pending
            ]
            self.assertEqual(pending_members, ["rs7412"])
            self.assertEqual(calls, 3)
            self.assertEqual(sources.operations, ["fetch_refsnp:rs7412"])
            self.assertEqual(
                resumed.attempts[-1].effective_invocation_policy.model_requests, 2
            )

    def test_revised_request_rejection_preserves_carried_work(self):
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            KeepInvestigationMember,
            RejectInvestigationResponse,
            RetryInvestigationMember,
            ReviseInvestigationResponse,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        async def ask(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which sample?",
                            "missing_information": "Sample scope is missing.",
                            "asks_about": "target",
                        },
                        tool_call_id="revision-question",
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            source = SequentialSourceResponses()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-research-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            original, journey = run_attempt(
                Path(directory), SequentialAdaptiveModel(), source, assertion=assertion
            )
            proposal = original.proposals[-1]
            carried, retry = proposal.work_items
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=original.pending_request.request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=proposal.proposal_id,
                        asserted_by="researcher",
                        operations=(
                            KeepInvestigationMember(
                                operation="keep", work_item_id=carried.work_item_id
                            ),
                            RetryInvestigationMember(
                                operation="retry", work_item_id=retry.work_item_id
                            ),
                        ),
                    ),
                )
            )
            pending = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=ScriptedModelEndpoint(ask),
                source_adapter_factory=lambda: source,
            )
            question = pending.pending_request.question
            result = respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.pending_request.request_event_id,
                    response=RejectInvestigationResponse(
                        action="reject",
                        collection_member_id=question.collection_member_id,
                        revision_number=2,
                        reason="Do not pursue retry",
                        asserted_by="researcher",
                    ),
                )
            )
            self.assertEqual(result.proposals[-1].work_items, (carried,))
            self.assertEqual(
                result.proposals[-1].request_members[-1].rejection.reason,
                "Do not pursue retry",
            )
            self.assertEqual(
                result.proposals[-1].supersedes_proposal_id, proposal.proposal_id
            )
            self.assertEqual(
                source.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                    "fetch_refsnp:rs7412",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, result
            )

    def test_reopen_refuses_retargeted_question_even_with_updated_result_digest(self):
        from copy import deepcopy

        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            model_visible,
            tool_model_view,
        )
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
            investigation_output_member_id,
            project_investigation,
        )
        from bio_neb.journey import canonical_json_digest

        async def ask(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "request_investigation_information",
                        {
                            "question": "Which allele?",
                            "missing_information": "Allele intent missing",
                            "asks_about": "target",
                        },
                        tool_call_id="exact-question",
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-research-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            sources = SourceResponsesOnly()
            _, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(ask)),
                sources,
                assertion=assertion,
            )
            original = read_evidence(journey)
            for mutation in ("member", "revision", "arguments"):
                with self.subTest(mutation=mutation):
                    hostile = deepcopy(original)
                    event = next(
                        e
                        for e in hostile
                        if e["event_type"] == "InvestigationReviewRequested"
                    )
                    question = event["payload"]["question"]
                    if mutation == "member":
                        question["collection_member_id"] = (
                            investigation_output_member_id(assertion.requests[1], 1)
                        )
                        question["request"] = assertion.requests[1].model_dump(
                            mode="json"
                        )
                    elif mutation == "revision":
                        question["revision_number"] = 2
                    else:
                        question["question"] = "A changed question"
                    receipt = next(
                        e
                        for e in hostile
                        if e["event_type"] == "ObservedToolExecutionSucceeded"
                    )
                    # The receipt's digest is recomputed over the changed fact's
                    # model view, so only the question's own check can refuse
                    # it.
                    shown, _guarded = model_visible(
                        tool_model_view(
                            model_view_of("request_investigation_information"), event
                        )
                    )
                    receipt["payload"]["model_visible_result_digest"] = (
                        canonical_json_digest(shown)
                    )
                    with self.assertRaisesRegex(
                        ValueError, "Investigation question changed"
                    ):
                        project_investigation(hostile)
            self.assertEqual(sources.operations, [])

    def test_agent_work_continue_preserves_completed_sibling_and_sources(self):
        from bio_neb.agentic_journey.agent_work_review import (
            ContinueAgentWorkResponse,
            project_agent_work_review,
            respond_agent_work_review,
        )
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        scientific = SequentialAdaptiveModel()
        calls = 0

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls > 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "compile_proposal",
                            {"outcome": "complete"},
                            tool_call_id=f"bad-output-{calls}",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SequentialSourceResponses()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-research-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            stopped, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
                assertion=assertion,
            )
            self.assertEqual(stopped.status, "agent_work_review_pending")
            completed_events = [
                e
                for e in read_evidence(journey)
                if e["event_type"]
                in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
            ]
            self.assertEqual(len(completed_events), 2)
            request = project_agent_work_review(read_evidence(journey)).pending_request
            answer = respond_agent_work_review(
                journey,
                request.request_event_id,
                ContinueAgentWorkResponse(action="continue"),
            )
            resumed = investigation_assignment(
                model_factory=scientific.model,
                source_adapter_factory=lambda: sources,
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                work_review_response_event_id=answer.response_event_id,
            )
            self.assertEqual(resumed.status, "review_pending")
            self.assertEqual(
                [
                    item.target.one_based_position
                    for item in resumed.proposals[-1].work_items
                ],
                [44908684, 44908822],
            )
            self.assertEqual(
                sources.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                    "fetch_refsnp:rs7412",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            self.assertEqual(
                [
                    e
                    for e in read_evidence(journey)
                    if e["event_type"]
                    in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
                ][:2],
                completed_events,
            )
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, resumed
            )

    def test_member_agent_work_reject_settles_that_member_and_the_collection_closes(
        self,
    ):
        """A member's Agent work stop rejected by the researcher settles that member
        alone: the Investigation continues and the collection proposal accounts it
        beside its completed sibling (#389 examination finding 9)."""
        self._member_settled("reject")

    def test_a_dismissed_member_leaves_the_pool_and_the_collection_closes(self):
        """Dismissing a member's stop (#447: "I want to search rs9999, it does not
        exist") settles that member as its reject does; the rest go on."""
        self._member_settled("dismiss")

    def _member_settled(self, action):
        from bio_neb.agentic_journey.agent_work_review import (
            DismissAgentWorkResponse,
            RejectAgentWorkResponse,
            project_agent_work_review,
            respond_agent_work_review,
        )
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
            project_investigation,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            continue_investigation,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        scientific = SequentialAdaptiveModel()
        calls = 0

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls > 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "compile_proposal",
                            {"outcome": "complete"},
                            tool_call_id=f"bad-output-{calls}",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SequentialSourceResponses()
            assertion = UserAssertion.create_collection(
                asserted_by="researcher",
                predecessor_ids=("offline-research-request",),
                requests=tuple(
                    InvestigationRequestItem(
                        request_text=f"Investigate {rsid}", rsid=rsid
                    )
                    for rsid in ("rs429358", "rs7412")
                ),
            )
            stopped, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
                assertion=assertion,
            )
            self.assertEqual(stopped.status, "agent_work_review_pending")
            operations_at_stop = list(sources.operations)
            request = project_agent_work_review(read_evidence(journey)).pending_request
            effects = {item.action: item.effect for item in request.legal_actions}
            self.assertIn("settles this member as not resolved", effects["reject"])
            self.assertIn("other members continue", effects[action])
            response = {
                "reject": RejectAgentWorkResponse,
                "dismiss": DismissAgentWorkResponse,
            }
            answer = respond_agent_work_review(
                journey,
                request.request_event_id,
                response[action](
                    action=action,
                    asserted_by="researcher",
                    reason="rs7412 cannot be resolved here; go on without it.",
                ),
            )
            settled = project_investigation(read_evidence(journey))
            self.assertEqual(settled.status, "investigation_work_ready")
            self.assertEqual(settled.eligible_member_ids, ())

            closed = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=scientific.model,
                source_adapter_factory=lambda: sources,
            )
            self.assertEqual(closed.status, "review_pending")
            proposal = closed.proposals[-1]
            self.assertEqual(
                [item.target.one_based_position for item in proposal.work_items],
                [44908684],
            )
            rejected = next(
                member
                for member in proposal.request_members
                if member.request.rsid == "rs7412"
            )
            self.assertEqual(
                rejected.rejection.request_event_id, request.request_event_id
            )
            self.assertEqual(
                rejected.rejection.response_event_id, answer.response_event_id
            )
            self.assertEqual(rejected.rejection.asserted_by, "researcher")
            self.assertEqual(
                rejected.rejection.reason,
                "rs7412 cannot be resolved here; go on without it.",
            )
            # No member was planned again: closing contacted no source.
            self.assertEqual(sources.operations, operations_at_stop)

    def test_a_member_stop_of_each_kind_settles_that_member_and_the_collection_closes(
        self,
    ):
        """Member scope applies to every stop kind (owner B2; #446 Q3 A): a stop
        answered on one member's effort, operational or availability Review settles
        that member as not resolved, accounted like a rejected request, while its
        sibling's proposal goes on. The rejection names who stopped it only where
        the answer does (owner, 2026-10-25: the operational stop's `by`)."""
        from pydantic_ai.exceptions import ModelHTTPError

        from bio_neb.agentic_journey.effort import (
            StopAgentEffortResponse,
            project_effort_review,
            respond_effort_review,
        )
        from bio_neb.agentic_journey.model_availability import (
            StopModelAvailabilityResponse,
            project_model_availability_review,
        )
        from bio_neb.agentic_journey.operational_review import (
            answer_operational_review,
            project_operational_review,
        )
        from bio_neb.application.commands import (
            RespondModelAvailabilityReviewCommand,
            respond_model_availability_review,
        )
        from bio_neb.failure import ProviderError
        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
            project_investigation,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            continue_investigation,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
            _current_member_messages,
        )

        class Source(SequentialSourceResponses):
            def fetch_refsnp(self, rsid):
                if rsid == "rs7412":
                    self.operations.append("fetch_refsnp:" + rsid)
                    raise ProviderError("rejected", "Source denied this request.")
                return super().fetch_refsnp(rsid)

        def stop_effort(journey):
            request = project_effort_review(read_evidence(journey)).pending_request
            return respond_effort_review(
                journey,
                request.request_event_id,
                StopAgentEffortResponse(action="stop", reason="It only searched."),
            ).response_event_id, request.request_event_id

        def stop_operational(journey):
            request = project_operational_review(read_evidence(journey)).pending_request
            receipt, _ = answer_operational_review(
                journey,
                request.request_event_id,
                {
                    "request_id": request.request.request_id,
                    "action": "stop",
                    "by": "researcher:owner",
                    "reason": "The source denies rs7412.",
                },
            )
            return receipt.response_event_id, request.request_event_id

        def stop_availability(journey):
            from unittest import mock

            request = project_model_availability_review(
                read_evidence(journey)
            ).pending_request
            # The command's Run read needs a registered Run; this Journey holds
            # only the Investigation, so that read is left out here.
            with mock.patch("bio_neb.application.queries.review_journey"):
                receipt = respond_model_availability_review(
                    RespondModelAvailabilityReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=request.request_event_id,
                        response=StopModelAvailabilityResponse(
                            action="stop", reason="The endpoint refuses rs7412."
                        ),
                    ),
                    environ={"LOCAL_MODEL_KEY": "offline"},
                )
            return receipt.response_event_id, request.request_event_id

        cases = (
            ("effort", "effort_review_pending", stop_effort, "It only searched.", None),
            (
                "operational",
                "operational_review_pending",
                stop_operational,
                "The source denies rs7412.",
                "researcher:owner",
            ),
            (
                # The collection reports a member's held model contact with its
                # held operations.
                "availability",
                "operational_review_pending",
                stop_availability,
                "The endpoint refuses rs7412.",
                None,
            ),
        )
        for kind, held, stop, reason, asserted_by in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                scientific = SequentialAdaptiveModel()
                searches = 0

                async def respond(messages, info, kind=kind, scientific=scientific):
                    nonlocal searches
                    request, *_ = _current_member_messages(messages)
                    if request["rsid"] != "rs7412":
                        return await scientific.respond(messages, info)
                    if kind == "availability":
                        raise ModelHTTPError(402, "test", body="rejected")
                    if kind == "operational":
                        call = ("fetch_refsnp", {"rsid": "rs7412"})
                    else:
                        searches += 1
                        call = ("search_journey", {"text": f"rs7412 {searches}"})
                    return ModelResponse(
                        parts=[ToolCallPart(*call, tool_call_id=f"{kind}-{searches}")]
                    )

                sources = Source()
                assertion = UserAssertion.create_collection(
                    asserted_by="researcher",
                    predecessor_ids=("offline-research-request",),
                    requests=tuple(
                        InvestigationRequestItem(
                            request_text=f"Investigate {rsid}", rsid=rsid
                        )
                        for rsid in ("rs429358", "rs7412")
                    ),
                )
                stopped, journey = run_attempt(
                    Path(directory),
                    SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                    sources,
                    assertion=assertion,
                    invocation_overrides={"model_requests": 4},
                )
                self.assertEqual(stopped.status, held)
                answer_id, request_id = stop(journey)
                # The command records through its own handle of the Journey.
                journey = Journey.open(journey.path)
                settled = project_investigation(read_evidence(journey))
                self.assertEqual(settled.status, "investigation_work_ready")
                self.assertEqual(settled.eligible_member_ids, ())

                closed = continue_investigation(
                    ContinueInvestigationCommand(journey_file=str(journey.path)),
                    environ={"LOCAL_MODEL_KEY": "offline"},
                    model_factory=scientific.model,
                    source_adapter_factory=lambda sources=sources: sources,
                )
                self.assertEqual(closed.status, "review_pending")
                proposal = closed.proposals[-1]
                self.assertEqual(
                    [item.target.one_based_position for item in proposal.work_items],
                    [44908684],
                )
                (rejected,) = [
                    member.rejection
                    for member in proposal.request_members
                    if member.request.rsid == "rs7412"
                ]
                self.assertEqual(
                    (
                        rejected.request_event_id,
                        rejected.response_event_id,
                        rejected.reason,
                        rejected.asserted_by,
                    ),
                    (request_id, answer_id, reason, asserted_by),
                )
                dumped = rejected.model_dump(mode="json")
                self.assertEqual("asserted_by" in dumped, asserted_by is not None)

    def test_a_question_of_no_researcher_kind_is_refused_and_the_member_proposes(
        self,
    ):
        """A member question names what it asks in asks_about (#472 Q4 a, ADR 0049: no
        word list judges its text): the sample's own result is no kind the researcher
        answers, so a question naming it is refused by the Tool's schema before it
        becomes a Review, and the member then proposes from its sources (#389: 24 of 25
        questions asked for it). Replaces B35-B37's word-list tests (retired with
        `asks_for_sample_result`)."""
        self._assert_refused_then_proposes(
            "What does HG001 show at rs429358?", "The sample's genotype there."
        )

    def _assert_refused_then_proposes(self, question, missing_information):
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        scientific = SequentialAdaptiveModel()
        replies = []
        corrections = []

        async def respond(messages, info):
            replies.append(info)
            if len(replies) == 2:
                corrections.extend(sent_corrections(messages[-1]))
            if len(replies) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "request_investigation_information",
                            {
                                "question": question,
                                "missing_information": missing_information,
                                "asks_about": "sample_result",
                            },
                            tool_call_id="genotype-question",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SequentialSourceResponses()
            projection, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            evidence = read_evidence(Journey.open(journey.path))
            refusal = next(
                e
                for e in evidence
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_call_id"] == "genotype-question"
            )
            self.assertEqual(refusal["payload"]["status"], "invalid")
            (error,) = refusal["payload"]["errors"]
            self.assertTrue(
                error.startswith("tool_arguments_invalid:asks_about:"), error
            )
            # The member's next request carries the refusal's correction.
            (correction,) = corrections
            self.assertIn("asks_about", correction)
            self.assertEqual(
                [r for r in projection.review_requests if r.question is not None], []
            )
            self.assertEqual(projection.status, "review_pending")
            self.assertEqual(
                [
                    item.target.one_based_position
                    for item in projection.proposals[-1].work_items
                ],
                [44908684],
            )
            self.assertEqual(
                sources.operations,
                ["fetch_refsnp:rs429358", "fetch_assembly_report:GCF_000001405.40"],
            )

    def test_clarified_member_sees_its_question_beside_the_answer_and_a_next_step(
        self,
    ):
        """After a clarify, the member's next Attempt reads its own question beside
        the researcher's answer and the product's next step, and then proposes
        (#389: wording B proceeded 11 times of 14)."""
        from bio_neb.investigation.investigation import ClarifyInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import (
            SequentialAdaptiveModel,
            SequentialSourceResponses,
        )

        scientific = SequentialAdaptiveModel()
        first_prompts = []

        async def respond(messages, info):
            if len(messages) == 1:
                first_prompts.append(
                    json.loads(
                        next(
                            part.content
                            for part in messages[0].parts
                            if part.part_kind == "user-prompt"
                        )
                    )
                )
            if len(first_prompts) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "request_investigation_information",
                            {
                                "question": "Which allele?",
                                "missing_information": "The request does not specify an allele.",
                                "asks_about": "target",
                            },
                            tool_call_id="intent-question",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SequentialSourceResponses()
            model = ScriptedModelEndpoint(respond)
            pending, journey = run_attempt(
                Path(directory), SimpleNamespace(model=model), sources
            )
            question = pending.pending_request.question
            self.assertNotIn("next_step", first_prompts[0])
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=pending.pending_request.request_event_id,
                    response=ClarifyInvestigationResponse(
                        action="clarify",
                        collection_member_id=question.collection_member_id,
                        revision_number=1,
                        instruction="Any allele dbSNP lists for it.",
                        asserted_by="researcher:owner",
                    ),
                )
            )
            completed = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
                source_adapter_factory=lambda: sources,
            )
            self.assertEqual(len(first_prompts), 2)
            (answer,) = first_prompts[1]["researcher_answers"]
            self.assertEqual(answer["question"], "Which allele?")
            self.assertEqual(answer["instruction"], "Any allele dbSNP lists for it.")
            next_step = first_prompts[1]["next_step"]
            self.assertIn("fetch_refsnp", next_step)
            self.assertIn("GCF_000001405.40", next_step)
            self.assertIn("compile_proposal", next_step)
            self.assertIn("after the researcher approves", next_step)
            self.assertEqual(completed.status, "review_pending")
            self.assertEqual(len(completed.proposals[-1].work_items), 1)


class QuestionKindTests(unittest.TestCase):
    """What a member question asks is typed by its asker (#472 Q4 a, ADR 0049: "a word
    list ... that extracts or judges a scientific value" is a rule a surface may not set).
    Replaces B36-B37 (`asks_for_sample_result` over #389's recorded questions, retired)."""

    def test_the_kinds_are_what_the_researcher_can_supply(self):
        from typing import get_args

        from pydantic import ValidationError

        from bio_neb.investigation.investigation import (
            InvestigationQuestion,
            InvestigationRequestItem,
            QuestionKind,
        )

        self.assertEqual(get_args(QuestionKind), ("target", "version", "intent"))
        fields = {
            "assertion_id": "assertion",
            "collection_member_id": "member",
            "revision_number": 1,
            "request": InvestigationRequestItem(
                request_text="rs429358", rsid="rs429358"
            ),
            "question": "What does HG001 show at rs429358?",
            "missing_information": "The genotype there.",
        }
        self.assertEqual(
            InvestigationQuestion(**fields, asks_about="target").asks_about, "target"
        )
        with self.assertRaises(ValidationError) as refused:
            InvestigationQuestion(**fields, asks_about="sample_result")
        # The schema refuses the kind itself, as no researcher-supplied kind.
        self.assertEqual(
            [(error["loc"], error["type"]) for error in refused.exception.errors()],
            [(("asks_about",), "literal_error")],
        )

    def test_admission_reads_no_word_of_a_question(self):
        from bio_neb.investigation.investigation import InvestigationRequestItem
        from bio_neb.investigation.investigation_agent import (
            investigation_action_errors,
        )

        for question in (
            "Which allele?",
            "Is rs429358 present in HG001?",
            "What is the genotype at rs429358?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    investigation_action_errors(
                        SimpleNamespace(
                            request=InvestigationRequestItem(
                                request_text="rs429358", rsid="rs429358"
                            ),
                            raw_answers=(),
                            researcher_answers=(),
                            target_reads=[],
                            started_event_id="started",
                            source_events=[],
                            question_event=None,
                            validation_event_id=None,
                        ),
                        SimpleNamespace(
                            tool_name="request_investigation_information",
                            arguments={
                                "question": question,
                                "missing_information": "x",
                                "asks_about": "intent",
                            },
                        ),
                    ),
                    (),
                )
