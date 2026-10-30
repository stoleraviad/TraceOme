"""One Investigation item's history (owner, #476 alignment G2.3 A; #479 Q2 A, Q-a).

The item's earlier work has one definition: the work builder, the repeat guard and
the decision admission read the same ancestry, and one rule says what a target change
keeps (target-free reads) and drops (decisions, gene-change placements and picks).
A continued item searches the sequence names its earlier Attempt read (a3 C10) and
its earlier decision still applies.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, ToolCallPart

from bio_neb.agentic_journey.action_policy import ProposedAgentAction
from bio_neb.agentic_journey.effort import (
    ExtendAgentEffortResponse,
    project_effort_review,
    respond_effort_review,
)
from bio_neb.agentic_journey.repeated_calls import repeated_call_of
from bio_neb.investigation.investigation import (
    AddInvestigationMember,
    AmendInvestigationMember,
    ClarifyInvestigationResponse,
    KeepInvestigationMember,
    ReviseInvestigationResponse,
    UserAssertion,
)
from bio_neb.investigation.investigation_agent import investigation_action_errors
from bio_neb.investigation.investigation_assignment import investigation_assignment
from bio_neb.investigation.investigation_commands import (
    ContinueInvestigationCommand,
    RespondInvestigationReviewCommand,
    continue_investigation,
    respond_investigation_review,
)
from bio_neb.investigation.investigation_request import (
    InvestigationRequestItem,
    StatedPosition,
)
from bio_neb.investigation.item_history import item_call_history, item_history
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.refsnp_retained_bodies import RS429358, RS2228570, RS4149056
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    input_and_assertion,
    latest_assignment,
    run_attempt,
)

MEMBER = "member-a"
ACCESSION = "GCF_000001405.40"


def _event(event_id, event_type, **payload):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "parent_ids": [],
        "payload": payload,
    }


def _bound(attempt_id, revision=1, continued=None):
    return _event(
        f"bound-{attempt_id}",
        "InvestigationAttemptBound",
        attempt_id=attempt_id,
        collection_member_id=MEMBER,
        revision_number=revision,
        **({"continued_attempt_id": continued} if continued else {}),
    )


def _item_events():
    """An item's first Attempt: a lookup, the sequence names it read, a decision;
    then a researcher's amend (a new target), the item's continued Attempt, and a
    later revision's first Attempt of the same item."""
    return [
        _event("start-a1", "AttemptStarted", attempt_id="a1"),
        _bound("a1"),
        _event(
            "contact-a1",
            "InvestigationSourceContactStarted",
            attempt_id="a1",
            operation="ncbi_sequence_names",
            request_key=ACCESSION,
        ),
        _event(
            "names-a1",
            "InvestigationSourceRetrieved",
            attempt_id="a1",
            source={"operation": "ncbi_sequence_names", "status_code": 200},
        ),
        _event(
            "lookup-a1",
            "RefSnpToolResultRecorded",
            attempt_id="a1",
            collection_member_id=MEMBER,
            result={"status": "sequence_unnamed"},
        ),
        _event(
            "decided-a1",
            "InvestigationLookupDecided",
            attempt_id="a1",
            collection_member_id=MEMBER,
            refsnp_event_id="lookup-a1",
            decision={"kind": "sequence"},
        ),
        _event(
            "amend",
            "InvestigationReviewResponded",
            response={
                "action": "amend",
                "collection_member_id": MEMBER,
                "revision_number": 1,
                "target": {"rsid": "rs2"},
            },
        ),
        _event("start-a2", "AttemptStarted", attempt_id="a2"),
        _bound("a2", continued="a1"),
        _event(
            "decided-a2",
            "InvestigationLookupDecided",
            attempt_id="a2",
            collection_member_id=MEMBER,
            refsnp_event_id="lookup-a1",
            decision={"kind": "version"},
        ),
        _event("start-a3", "AttemptStarted", attempt_id="a3"),
        _bound("a3", revision=2),
    ]


class TargetChangeRuleTests(unittest.TestCase):
    """The one stated rule for what a target change keeps (owner #479 Q2 A)."""

    def history(self, events, attempt_id, revision):
        return item_history(
            events,
            attempt_id=attempt_id,
            member_id=MEMBER,
            revision_number=revision,
            read_artifact=lambda event: f"bytes of {event['event_id']}".encode(),
        )

    def test_an_answer_that_changes_the_target_keeps_reads_and_drops_decisions(self):
        events = _item_events()[:9]
        history = self.history(events, "a2", 1)

        # Kept: the lookup as evidence and the sequence names it read (C10).
        self.assertEqual([e["event_id"] for e in history.source_results], ["lookup-a1"])
        self.assertEqual(
            history.sequence_reports, {ACCESSION: ("names-a1", b"bytes of names-a1")}
        )
        # Dropped: the old target's decision no longer applies.
        self.assertEqual(history.decisions, [])
        # The replaced target's last lookup stays shown, marked replaced by the amend.
        self.assertEqual(history.replaced_result["event_id"], "lookup-a1")
        self.assertEqual(history.target_event_id, "amend")
        self.assertEqual(history.replaced_result_ids, frozenset({"lookup-a1"}))

    def test_a_clarify_that_types_no_target_changes_nothing(self):
        events = copy.deepcopy(_item_events()[:9])
        response = next(e for e in events if e["event_id"] == "amend")["payload"][
            "response"
        ]
        response["action"] = "clarify"
        del response["target"]
        history = self.history(events, "a2", 1)

        self.assertEqual([e["event_id"] for e in history.decisions], ["decided-a1"])
        self.assertIsNone(history.replaced_result)

    def test_a_decision_after_the_change_applies(self):
        history = self.history(_item_events()[:10], "a2", 1)

        self.assertEqual([e["event_id"] for e in history.decisions], ["decided-a2"])

    def test_a_new_revision_keeps_target_free_reads_and_drops_decisions(self):
        # Q-a yes: the item a revision's record says it continues keeps its
        # target-free reads, though its amended request has another member id.
        from bio_neb.investigation.investigation import (
            investigation_output_member_id,
            request_member_id,
        )

        first = InvestigationRequestItem(
            request_text="Does HG001 carry rs1?", rsid="rs1"
        )
        amended = InvestigationRequestItem(request_text="I meant rs2.", rsid="rs2")
        old_member = investigation_output_member_id(first, 0)
        new_member = investigation_output_member_id(amended, 0)
        events = [
            _event("start-a1", "AttemptStarted", attempt_id="a1"),
            _event(
                "bound-a1",
                "InvestigationAttemptBound",
                attempt_id="a1",
                collection_member_id=old_member,
                revision_number=1,
                assertion_id="assertion-1",
                work_requests=[first.model_dump(mode="json")],
            ),
            *_item_events()[2:4],
            _event(
                "lookup-a1",
                "RefSnpToolResultRecorded",
                attempt_id="a1",
                collection_member_id=old_member,
                result={"status": "sequence_unnamed"},
            ),
            _event(
                "decided-a1",
                "InvestigationLookupDecided",
                attempt_id="a1",
                collection_member_id=old_member,
                refsnp_event_id="lookup-a1",
                decision={"kind": "sequence"},
            ),
            _event(
                "proposal",
                "InvestigationProposalRecorded",
                proposal={
                    "proposal_id": "proposal-1",
                    "revision_number": 1,
                    "request_members": [
                        {
                            "request_member_id": request_member_id(
                                "assertion-1", first, 0
                            ),
                            "work_item_ids": ["work-item:w"],
                        }
                    ],
                },
            ),
            _event(
                "revision",
                "InvestigationCollectionRevisionRecorded",
                revision={
                    "revision_number": 2,
                    "supersedes_proposal_id": "proposal-1",
                    "members": [
                        {
                            "operation": "amend",
                            "previous_work_item_id": "work-item:w",
                            "request": amended.model_dump(mode="json"),
                        }
                    ],
                },
            ),
            _event("start-a3", "AttemptStarted", attempt_id="a3"),
            _event(
                "bound-a3",
                "InvestigationAttemptBound",
                attempt_id="a3",
                collection_member_id=new_member,
                revision_number=2,
                assertion_id="assertion-2",
                work_requests=[amended.model_dump(mode="json")],
            ),
        ]

        history = item_history(
            events,
            attempt_id="a3",
            member_id=new_member,
            revision_number=2,
            read_artifact=lambda event: b"names",
        )

        self.assertEqual([e["event_id"] for e in history.source_results], ["lookup-a1"])
        self.assertIn(ACCESSION, history.sequence_reports)
        self.assertEqual(history.decisions, [])
        self.assertEqual(history.target_event_id, "bound-a3")
        self.assertEqual(history.replaced_result["event_id"], "lookup-a1")
        # A request the revision adds continues no item.
        added = copy.deepcopy(events)
        next(e for e in added if e["event_id"] == "revision")["payload"]["revision"][
            "members"
        ][0] = {"operation": "add", "request": amended.model_dump(mode="json")}
        self.assertEqual(
            item_history(
                added,
                attempt_id="a3",
                member_id=new_member,
                revision_number=2,
                read_artifact=lambda event: b"names",
            ).source_results,
            [],
        )

    def test_a_decision_citing_a_replaced_result_is_refused(self):
        # A dropped old-target fact is not offered again: no decision cites it.
        lookup = {
            "event_id": "lookup-a1",
            "event_type": "RefSnpToolResultRecorded",
            "payload": {"result": {"status": "sequence_unnamed"}},
        }
        deps = SimpleNamespace(
            request=InvestigationRequestItem(
                request_text="The paper reports chr19:1.",
                position=StatedPosition(contig="chr19", position=1),
            ),
            researcher_answers=(),
            raw_answers=(),
            target_reads=[],
            started_event_id="started",
            source_events=[lookup],
            replaced_result_ids=frozenset({"lookup-a1"}),
        )

        errors = investigation_action_errors(
            deps,
            ProposedAgentAction(
                tool_name="name_sequence",
                tool_call_id="call",
                arguments={
                    "refsnp_event_id": "lookup-a1",
                    "assembly_accession": ACCESSION,
                    "record": "NC_000019.10",
                    "quote": "chr19",
                },
            ),
        )

        self.assertEqual(errors, ("decision_result_replaced",))


class ItemRepeatGuardTests(unittest.TestCase):
    """The repeat guard reads the item's history, the work builder's (#479 G2.3)."""

    def events(self, between=()):
        call = {"tool_name": "name_sequence", "tool_call_id": "c1", "arguments": {}}
        return [
            _bound("a1"),
            _event("v1", "ToolCallValidated", attempt_id="a1", **call),
            _event(
                "s1",
                "ObservedToolExecutionStarted",
                attempt_id="a1",
                validation_event_id="v1",
            ),
            _event("r1", "InvestigationLookupDecided", attempt_id="a1"),
            _event(
                "ok1",
                "ObservedToolExecutionSucceeded",
                attempt_id="a1",
                started_event_id="s1",
                stage_result_event_id="r1",
            ),
            *between,
            _event("start-a2", "AttemptStarted", attempt_id="a2"),
            _bound("a2", continued="a1"),
            _event(
                "context-a2",
                "InvestigationMemberContextRecorded",
                attempt_id="a2",
                collection_member_id=MEMBER,
            ),
        ]

    def repeat(self, events, **history):
        return repeated_call_of(
            events,
            attempt_id="a2",
            action=ProposedAgentAction(
                tool_name="name_sequence", tool_call_id="c2", arguments={}
            ),
            **history,
        )

    def test_the_earlier_attempts_unchanged_call_is_a_repeat(self):
        events = self.events()

        # The Attempt's own history sees no earlier call; the item's does.
        self.assertIsNone(self.repeat(events))
        self.assertEqual(self.repeat(events, history=item_call_history), "v1")

    def test_a_researcher_answer_to_this_items_work_is_a_change(self):
        # An answer to the item's question, or to a Review its own work stopped at
        # (a continue may instruct the work to fetch again).
        own_request = _event("own-request", "AgentWorkReviewRequested", attempt_id="a1")
        for between in (
            (
                _event(
                    "answer",
                    "InvestigationReviewResponded",
                    response={"action": "clarify", "collection_member_id": MEMBER},
                ),
            ),
            (
                own_request,
                _event(
                    "answer",
                    "AgentWorkReviewResponded",
                    request_event_id="own-request",
                    response={"action": "continue"},
                ),
            ),
        ):
            with self.subTest(answer=between[-1]["event_type"]):
                self.assertIsNone(
                    self.repeat(self.events(between), history=item_call_history)
                )

    def test_a_lead_reviews_answer_changes_nothing_of_its_joined_item(self):
        # Review r2 ST1: a keyed Review's joined item is answered by its own
        # request; only that answer is this item's change.
        lead = _event(
            "lead", "ModelAvailabilityReviewRequested", attempt_id="sibling-attempt"
        )
        joined = _event(
            "joined",
            "ModelAvailabilityReviewRequested",
            attempt_id="a1",
            joins_event_id="lead",
        )
        for answered, expected in (("lead", "v1"), ("joined", None)):
            with self.subTest(answered=answered):
                answer = _event(
                    "answer",
                    "ModelAvailabilityReviewResponded",
                    request_event_id=answered,
                    response={"action": "retry"},
                )
                self.assertEqual(
                    self.repeat(
                        self.events((lead, joined, answer)), history=item_call_history
                    ),
                    expected,
                )

    def test_a_siblings_review_answer_changes_nothing_of_this_item(self):
        # Review r1 ST1/SP3: an answer to another item's Review is no change here.
        for kind in (
            "AgentWorkReviewResponded",
            "AgentEffortReviewResponded",
            "AgentFailureReviewResponded",
            "ModelAvailabilityReviewResponded",
        ):
            with self.subTest(answer=kind):
                events = self.events(
                    (
                        _event(
                            "sibling-request",
                            kind.replace("Responded", "Requested"),
                            attempt_id="sibling-attempt",
                        ),
                        _event(
                            "sibling-answer",
                            kind,
                            request_event_id="sibling-request",
                            response={"action": "continue"},
                        ),
                    )
                )
                self.assertEqual(self.repeat(events, history=item_call_history), "v1")


def _latest_cite(messages) -> str:
    returns = [
        part["content"]
        for message in ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        for part in message["parts"]
        if part["part_kind"] == "tool-return"
        and part["tool_name"] == "locate_variant"
        and isinstance(part["content"], dict)
        and "cite" in part["content"]
    ]
    return returns[-1]["cite"]["refsnp_event_id"]


class ContinuedItemTests(unittest.TestCase):
    """a3 C10 through the Agent: after its Attempt stops and is continued, the item
    searches the sequence names its earlier Attempt read, and its earlier decision
    still applies, with no new GET of those names."""

    def test_a_continued_item_searches_its_names_and_keeps_its_decision(self):
        calls = [
            ("locate_variant", {"source": "ncbi_dbsnp"}),
            (
                "name_sequence",
                lambda messages: {
                    "refsnp_event_id": _latest_cite(messages),
                    "assembly_accession": ACCESSION,
                    "record": "NC_000019.10",
                    "quote": "chr19",
                },
            ),
            # The continued Attempt: the search its earlier Attempt's names admit.
            (
                "search_sequence_names",
                {"assembly_accession": ACCESSION, "text": "19"},
            ),
            ("locate_variant", {"source": "ncbi_dbsnp"}),
        ]
        sent = []

        async def respond(messages, info):
            sent.append(info)
            name, arguments = calls[len(sent) - 1]
            if callable(arguments):
                arguments = arguments(messages)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, json.dumps(arguments), tool_call_id=f"c-{len(sent)}"
                    )
                ]
            )

        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="paper:doi:10.1000/example",
            requests=(
                InvestigationRequestItem(
                    request_text="The paper reports chr19:44908684.",
                    position=StatedPosition(
                        contig="chr19", position=44908684, ref="T", alt="C"
                    ),
                ),
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        sources = SourceResponsesOnly(
            refsnp_bodies={"rs429358": RS429358},
            reference_bases={"NC_000019.10:44908683:1": "T"},
            listed_rsids={"NC_000019.10:44908683:T:C": (429358,)},
        )
        model = SimpleNamespace(model=ScriptedModelEndpoint(respond))
        with tempfile.TemporaryDirectory() as directory:
            stopped, journey = run_attempt(
                Path(directory),
                model,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
                invocation_overrides={"model_requests": 2},
            )
            self.assertEqual(stopped.status, "effort_review_pending")
            names_before = sources.operations.count(f"ncbi_sequence_names:{ACCESSION}")
            pending = project_effort_review(read_evidence(journey)).pending_request
            receipt = respond_effort_review(
                journey,
                pending.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=2),
            )
            investigation_assignment(
                model_factory=model.model, source_adapter_factory=lambda: sources
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                effort_review_response_event_id=receipt.response_event_id,
            )
            events = read_evidence(journey)

        validations = {
            e["payload"]["tool_call_id"]: e["payload"]
            for e in events
            if e["event_type"] == "ToolCallValidated"
        }
        # The search is admitted on the names the earlier Attempt read (C10).
        self.assertEqual(validations["c-3"]["errors"], [])
        (search,) = [e for e in events if e["event_type"] == "SequenceNamesRead"]
        self.assertTrue(search["payload"]["result"]["rows"])
        # The earlier Attempt's decision applies: the later lookup reads it.
        (decided,) = [
            e for e in events if e["event_type"] == "InvestigationLookupDecided"
        ]
        later = [
            e
            for e in events
            if e["event_type"] == "RefSnpToolResultRecorded"
            and e["payload"]["tool_call_id"] == "c-4"
        ]
        self.assertEqual(
            later[0]["payload"]["decision_event_ids"], [decided["event_id"]]
        )
        self.assertNotEqual(
            later[0]["payload"]["attempt_id"], decided["payload"]["attempt_id"]
        )
        # The search read retained bytes: no GET of those names for it.
        self.assertEqual(
            sources.operations[: len(sources.operations)].count(
                f"ncbi_sequence_names:{ACCESSION}"
            )
            - names_before,
            1,
        )


class RevisionLineageTests(unittest.TestCase):
    """A collection revision's amended item continues the item its record names
    (review r1 SP2): its target-free reads stay and its replaced lookup is shown,
    though the amended request has another member id."""

    def test_a_collection_amend_keeps_the_replaced_items_reads(self):
        from bio_neb.investigation.investigation import (
            AmendInvestigationMember,
            ReviseInvestigationResponse,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from bio_neb.journey import Journey
        from tests.refsnp_retained_bodies import RS4149056
        from tests.sequential_investigation_support import SequentialAdaptiveModel

        sources = SourceResponsesOnly(
            refsnp_bodies={"rs429358": RS429358, "rs4149056": RS4149056}
        )
        model = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            original, journey = run_attempt(Path(directory), model, sources)
            old = [
                e
                for e in read_evidence(journey)
                if e["event_type"] == "RefSnpToolResultRecorded"
            ][-1]
            proposal = original.pending_request.proposal
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=original.pending_request.request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=proposal.proposal_id,
                        asserted_by="researcher:owner",
                        operations=(
                            AmendInvestigationMember(
                                operation="amend",
                                work_item_id=proposal.work_items[0].work_item_id,
                                request=InvestigationRequestItem(
                                    request_text="Actually investigate rs4149056 "
                                    "in HG001.",
                                    rsid="rs4149056",
                                ),
                            ),
                        ),
                    ),
                )
            )
            continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: sources,
            )
            events = read_evidence(Journey.open(journey.path))

        contexts = [
            e for e in events if e["event_type"] == "InvestigationMemberContextRecorded"
        ]
        self.assertEqual(len(contexts), 2)
        new = contexts[-1]["payload"]
        self.assertNotEqual(
            new["collection_member_id"], contexts[0]["payload"]["collection_member_id"]
        )
        self.assertEqual(new["replaced_result_event_id"], old["event_id"])
        history = item_history(
            events,
            attempt_id=new["attempt_id"],
            member_id=new["collection_member_id"],
            revision_number=2,
            read_artifact=lambda _event: b"",
        )
        # The old item's lookup is evidence still; no decision of its target applies.
        self.assertIn(old["event_id"], [e["event_id"] for e in history.source_results])
        self.assertEqual(history.decisions, [])


class RetargetAndKeepTests(unittest.TestCase):
    """Review r2 (SP4, SP2): a target read from a clarify changes the live item before
    its next lookup, while a reading that states none keeps its decisions; and an
    item's history crosses a revision that only kept it."""

    def _clarified(self, reading):
        count = 0
        words = "Use chr1:100 T>C instead." if "site" in reading else "Check again."

        async def respond(messages, info):
            nonlocal count
            count += 1
            if count == 1:
                name, args = "locate_variant", {"source": "ncbi_dbsnp"}
            elif count == 2:
                name, args = (
                    "name_sequence",
                    {
                        "refsnp_event_id": _latest_cite(messages),
                        "assembly_accession": "GCF_000001405.40",
                        "record": "NC_000019.10",
                        "quote": "chr19",
                    },
                )
            elif count == 3:
                name, args = (
                    "request_investigation_information",
                    {
                        "question": "Which target should I check?",
                        "missing_information": "Please clarify the intended site.",
                        "asks_about": "target",
                    },
                )
            elif count == 4:
                answer = latest_assignment(messages)["researcher_answers"][-1]
                name, args = (
                    "record_target_reading",
                    {"source_event_id": answer["response_event_id"], **reading},
                )
            elif count == 5:
                name, args = "locate_variant", {"source": "ncbi_dbsnp"}
            else:
                name, args = (
                    "request_investigation_information",
                    {
                        "question": "Stop this bounded probe?",
                        "missing_information": "Probe done.",
                        "asks_about": "intent",
                    },
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(name, json.dumps(args), tool_call_id=f"probe-{count}")
                ]
            )

        model = SimpleNamespace(model=ScriptedModelEndpoint(respond))
        source = SourceResponsesOnly(refsnp_bodies={"rs429358": RS429358})
        inp, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            predecessor_ids=(inp.input_id,),
            requests=(
                InvestigationRequestItem(
                    request_text="The paper reports chr19:44908684 T>C.",
                    position=StatedPosition(
                        contig="chr19", position=44908684, ref="T", alt="C"
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            first, journey = run_attempt(
                Path(directory),
                model,
                source,
                investigation_input=inp,
                assertion=assertion,
            )
            question = first.pending_request.question
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=first.pending_request.request_event_id,
                    response=ClarifyInvestigationResponse(
                        action="clarify",
                        collection_member_id=question.collection_member_id,
                        revision_number=question.revision_number,
                        instruction=words,
                        asserted_by="researcher:owner",
                    ),
                )
            )
            continued = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model.model,
                source_adapter_factory=lambda: source,
            )
            events = read_evidence(Journey.open(journey.path))
        self.assertTrue(
            [e for e in events if e["event_type"] == "InvestigationTargetRead"]
        )
        (decided,) = [
            e for e in events if e["event_type"] == "InvestigationLookupDecided"
        ]
        lookups = [
            e
            for e in events
            if e["event_type"] == "RefSnpToolResultRecorded"
            and e["payload"]["tool_call_id"] == "probe-5"
        ]
        self.assertTrue(
            lookups, f"the lookup after the reading records ({continued.status})"
        )
        return decided, lookups[0]

    def test_a_target_typed_after_clarify_drops_prior_decisions_before_lookup(self):
        _decided, lookup = self._clarified(
            {
                "quote": "chr1:100 T>C",
                "site": {"contig": "chr1", "position": 100, "ref": "T", "alt": "C"},
            }
        )
        self.assertEqual(lookup["payload"].get("decision_event_ids", []), [])

    def test_a_reading_that_states_no_target_keeps_prior_decisions(self):
        decided, lookup = self._clarified(
            {"reason": "The answer asks to check again and names no other site."}
        )
        self.assertEqual(lookup["payload"]["decision_event_ids"], [decided["event_id"]])

    def test_item_history_crosses_a_revision_that_kept_its_work(self):
        sources = SourceResponsesOnly(
            refsnp_bodies={
                "rs429358": RS429358,
                "rs4149056": RS4149056,
                "rs2228570": RS2228570,
            }
        )
        model = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            first, journey = run_attempt(Path(directory), model, sources)
            old = [
                e
                for e in read_evidence(journey)
                if e["event_type"] == "RefSnpToolResultRecorded"
            ][-1]

            def revise(review, operations):
                respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=review.request_event_id,
                        response=ReviseInvestigationResponse(
                            action="revise",
                            proposal_id=review.proposal.proposal_id,
                            asserted_by="researcher:owner",
                            operations=tuple(operations),
                        ),
                    )
                )
                return continue_investigation(
                    ContinueInvestigationCommand(journey_file=str(journey.path)),
                    environ={"LOCAL_MODEL_KEY": "offline"},
                    model_factory=model.model,
                    source_adapter_factory=lambda: sources,
                )

            old_work = first.pending_request.proposal.work_items[0]
            second = revise(
                first.pending_request,
                [
                    KeepInvestigationMember(
                        operation="keep", work_item_id=old_work.work_item_id
                    ),
                    AddInvestigationMember(
                        operation="add",
                        request=InvestigationRequestItem(
                            request_text="Also investigate rs4149056 in HG001.",
                            rsid="rs4149056",
                        ),
                    ),
                ],
            )
            carried = next(
                w
                for w in second.pending_request.proposal.work_items
                if "rs429358" in w.target.aliases
            )
            operations = [
                AmendInvestigationMember(
                    operation="amend",
                    work_item_id=w.work_item_id,
                    request=InvestigationRequestItem(
                        request_text="Actually investigate rs2228570 in HG001.",
                        rsid="rs2228570",
                    ),
                )
                if w.work_item_id == carried.work_item_id
                else KeepInvestigationMember(
                    operation="keep", work_item_id=w.work_item_id
                )
                for w in second.pending_request.proposal.work_items
            ]
            revise(second.pending_request, operations)
            events = read_evidence(Journey.open(journey.path))
        contexts = [
            e for e in events if e["event_type"] == "InvestigationMemberContextRecorded"
        ]
        self.assertEqual(len(contexts), 3)
        self.assertEqual(
            contexts[-1]["payload"].get("replaced_result_event_id"), old["event_id"]
        )


if __name__ == "__main__":
    unittest.main()
