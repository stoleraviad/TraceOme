"""Scientific assignment uses shared current context and correlated history."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from bio_neb.application.advancement import RunAdvancementRefused
from bio_neb.investigation.investigation import UserAssertion
from bio_neb.recording import read_evidence
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.model_lifetime_support import OwnedModelEndpoint
from tests.refsnp_retained_bodies import (
    RS429358,
    RS4149056,
    RS4149056_DOCUMENT,
    RS60037639,
    RS61228756,
    RS754204844,
    RS756427959,
    RS878871521,
    refsnp_body,
)
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    begin_in_pass,
    input_and_assertion,
    run_attempt,
    source_basis,
)


class InvestigationAssignmentTests(unittest.TestCase):
    def test_an_identical_fetch_is_stopped_and_its_observation_survives_effort(
        self,
    ):
        from bio_neb.agentic_journey.effort import (
            ExtendAgentEffortResponse,
            project_effort_review,
            respond_effort_review,
        )
        from bio_neb.application.queries import read_investigation
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )
        from bio_neb.journey import Journey

        calls = 0
        selected_ref = None

        async def respond(messages, info):
            nonlocal calls, selected_ref
            calls += 1
            if calls <= 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            {"rsid": "rs429358"},
                            tool_call_id=f"identical-{calls}",
                        )
                    ]
                )
            basis = source_basis(info)
            if calls == 3:
                selected_ref = basis["refsnp_event_id"]
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_assembly_report",
                            {
                                "refsnp_event_id": basis["refsnp_event_id"],
                                "assembly_accession": "GCF_000001405.40",
                            },
                            tool_call_id="after-extension",
                        )
                    ]
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "compile_proposal",
                        json.dumps(
                            {
                                **basis,
                                "refsnp_event_id": selected_ref,
                                "outcome": "complete",
                            }
                        ),
                        tool_call_id="selected-after-extension",
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            model = ScriptedModelEndpoint(respond)
            stopped, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                sources,
                invocation_overrides={"model_requests": 2},
            )
            self.assertEqual(stopped.status, "effort_review_pending")
            observations = [
                event
                for event in read_evidence(journey)
                if event["event_type"] == "RefSnpToolResultRecorded"
            ]
            # The identical refetch, with nothing changed since, is stopped before
            # its source contact (G-B).
            self.assertEqual(
                [event["payload"]["tool_call_id"] for event in observations],
                ["identical-1"],
            )
            self.assertEqual(
                [
                    (e["payload"]["tool_call_id"], e["payload"]["errors"])
                    for e in read_evidence(journey)
                    if e["event_type"] == "ToolCallValidated"
                ],
                [("identical-1", []), ("identical-2", ["repeated_call"])],
            )
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, stopped
            )
            request = project_effort_review(read_evidence(journey)).pending_request
            answer = respond_effort_review(
                journey,
                request.request_event_id,
                ExtendAgentEffortResponse(action="extend", additional_requests=2),
            )
            resumed = investigation_assignment(
                model_factory=model,
                source_adapter_factory=lambda: sources,
            ).run(
                journey,
                environ={"LOCAL_MODEL_KEY": "offline"},
                effort_review_response_event_id=answer.response_event_id,
            )
            self.assertEqual(resumed.status, "review_pending")
            self.assertEqual(
                sources.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            self.assertEqual(
                resumed.proposals[-1].work_items[0].target.one_based_position, 44908684
            )
            retained = [
                event
                for event in read_evidence(journey)
                if event["event_type"] == "RefSnpToolResultRecorded"
            ]
            self.assertEqual(retained, observations)
            before = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, resumed
            )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_a_members_output_citing_another_members_basis_is_refused(
        self,
    ):
        from copy import deepcopy

        from bio_neb.investigation.investigation import (
            InvestigationRequestItem,
            UserAssertion,
            project_investigation,
        )
        from bio_neb.journey import Journey
        from tests.sequential_investigation_support import (
            SequentialSourceResponses,
            input_and_assertion,
        )

        class Sources(SequentialSourceResponses):
            # rs61228756's retained record is merged into rs429358, whose
            # retained record lists the merge (no unrelated lookup).
            def fetch_refsnp(self, rsid):
                if rsid in {"rs61228756", "rs429358"}:
                    self.refsnp_body = {
                        "rs61228756": RS61228756,
                        "rs429358": RS429358,
                    }[rsid]
                    return SourceResponsesOnly.fetch_refsnp(self, rsid)
                return super().fetch_refsnp(rsid)

        scientific = SequentialAdaptiveModel()
        calls = 0
        first_basis = None
        borrowed = False

        async def respond(messages, info):
            nonlocal calls, first_basis, borrowed
            calls += 1
            if calls <= 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            {"rsid": "rs61228756" if calls == 1 else "rs429358"},
                            tool_call_id=f"recover-{calls}",
                        )
                    ]
                )
            response = await scientific.respond(messages, info)
            part = response.parts[0]
            if part.tool_name == "compile_proposal":
                arguments = part.args_as_dict()
                # The first member's output cites its own lookup; the second
                # borrows it once (the draft names no rsID).
                if first_basis is None:
                    first_basis = arguments["refsnp_event_id"]
                elif not borrowed:
                    borrowed = True
                    arguments["refsnp_event_id"] = first_basis
                    part.args = json.dumps(arguments)
            return response

        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=tuple(
                InvestigationRequestItem(request_text=f"Investigate {rsid}.", rsid=rsid)
                for rsid in ("rs61228756", "rs7412")
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        with tempfile.TemporaryDirectory() as directory:
            sources = Sources()
            model = ScriptedModelEndpoint(respond)
            # Each member is its own Attempt with its own allowance: the second
            # member no longer starts on the first one's spent requests, so
            # neither stops at six (Errors 18935528).
            resumed, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                sources,
                assertion=assertion,
                invocation_overrides={"model_requests": 6},
            )
            observations = [
                e
                for e in read_evidence(journey)
                if e["event_type"]
                in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
            ]
            self.assertEqual(resumed.status, "review_pending")
            self.assertEqual(len(sources.operations), 5)
            self.assertEqual(
                [len(a.model_request_event_ids) for a in resumed.attempts], [4, 4]
            )
            self.assertEqual(len({a.collection_member_id for a in resumed.attempts}), 2)
            self.assertEqual(
                [
                    item.target.one_based_position
                    for item in resumed.proposals[-1].work_items
                ],
                [44908684, 44908822],
            )
            invalid = [
                e["payload"]
                for e in read_evidence(journey)
                if e["event_type"] == "AgentOutputValidated"
                and e["payload"]["status"] == "invalid"
            ]
            self.assertEqual(len(invalid), 1)
            self.assertIn("refsnp_observation_not_owned", str(invalid[0]))
            # It names the member's own lookup results to cite.
            member = invalid[0]["errors"]
            self.assertEqual(member[0], "refsnp_observation_not_owned")
            self.assertTrue(member[1].startswith("refsnp_event_id: expected"), member)
            hostile = deepcopy(read_evidence(journey))
            first = next(
                e for e in hostile if e["event_id"] == observations[0]["event_id"]
            )
            first["payload"]["collection_member_id"] = observations[-1]["payload"][
                "collection_member_id"
            ]
            with self.assertRaisesRegex(ValueError, "member, query or call identity"):
                project_investigation(hostile)
            frozen = journey.path.read_bytes()
            self.assertEqual(
                project_investigation(read_evidence(Journey.open(journey.path))),
                resumed,
            )
            self.assertEqual(journey.path.read_bytes(), frozen)

    def test_selects_earlier_compatible_observations_after_duplicate_and_negative_results(
        self,
    ):
        from dataclasses import replace

        from bio_neb.application.queries import read_investigation
        from bio_neb.journey import Journey
        from tests.sequential_investigation_support import ASSEMBLY_BODY, source_basis

        class Sources(SourceResponsesOnly):
            def fetch_assembly_report(self, accession):
                response = super().fetch_assembly_report(accession)
                return replace(
                    response,
                    body=ASSEMBLY_BODY.replace(b"GCF_000001405.40", accession.encode()),
                )

        calls = 0
        selected = {}

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                name, args = "fetch_refsnp", {"rsid": "rs429358"}
            elif calls in (2, 3):
                selected["refsnp_event_id"] = source_basis(info)["refsnp_event_id"]
                name, args = (
                    "fetch_assembly_report",
                    {
                        "refsnp_event_id": selected["refsnp_event_id"],
                        "assembly_accession": "GCF_000001405.39"
                        if calls == 2
                        else "GCF_000001405.40",
                    },
                )
            elif calls == 4:
                selected["assembly_event_id"] = source_basis(info)["assembly_event_id"]
                name, args = "fetch_refsnp", {"rsid": "rs429358"}
            elif calls == 5:
                # A later negative: the duplicate observation checked against
                # another assembly (a lookup of another rsID is refused).
                selected["duplicate"] = source_basis(info)["refsnp_event_id"]
                name, args = (
                    "fetch_assembly_report",
                    {
                        "refsnp_event_id": selected["duplicate"],
                        "assembly_accession": "GCF_000001405.39",
                    },
                )
            else:
                name = "compile_proposal"
                args = {
                    "refsnp_event_id": selected["duplicate"]
                    if calls == 6
                    else selected["refsnp_event_id"],
                    "assembly_event_id": selected["assembly_event_id"],
                    "outcome": "complete",
                }
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, json.dumps(args), tool_call_id=f"explore-{calls}"
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = Sources()
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            self.assertEqual(result.status, "review_pending")
            self.assertEqual(
                sources.operations,
                [
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.39",
                    "fetch_assembly_report:GCF_000001405.40",
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.39",
                ],
            )
            events = read_evidence(journey)
            observations = [
                e
                for e in events
                if e["event_type"]
                in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
            ]
            self.assertEqual(
                [e["payload"]["result"]["status"] for e in observations],
                ["resolved", "incompatible", "compatible", "resolved", "incompatible"],
            )
            self.assertEqual(
                observations[0]["payload"]["result"]["result_id"],
                observations[3]["payload"]["result"]["result_id"],
            )
            self.assertNotEqual(
                observations[0]["event_id"], observations[3]["event_id"]
            )
            validations = [
                e["payload"]
                for e in events
                if e["event_type"] == "AgentOutputValidated"
            ]
            self.assertEqual(
                [item["status"] for item in validations], ["invalid", "valid"]
            )
            self.assertIn("assembly_observation_basis_mismatch", str(validations[0]))
            # It says which report checks the cited lookup: none does here, so it
            # names the call that checks it.
            self.assertEqual(
                validations[0]["errors"][1],
                "assembly_event_id: no compatible assembly report checked "
                f"{observations[3]['event_id']}; call fetch_assembly_report("
                f'refsnp_event_id="{observations[3]["event_id"]}", '
                'assembly_accession="GCF_000001405.40")',
            )
            chosen = (
                result.proposal_output_lineages[-1].output_lineages[0].accepted_output
            )
            self.assertEqual(
                (chosen["refsnp_event_id"], chosen["assembly_event_id"]),
                (observations[0]["event_id"], observations[2]["event_id"]),
            )
            verified = [
                value
                for value in result.requirement_values
                if value.source == "tool_result"
            ]
            self.assertEqual(
                verified[0].lineage_event_ids, (observations[0]["event_id"],)
            )
            self.assertEqual(
                result.proposals[-1].work_items[0].target.one_based_position, 44908684
            )
            before = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, result
            )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_changed_query_recovers_and_keeps_earlier_negative_observation(self):
        from bio_neb.application.queries import read_investigation
        from bio_neb.journey import Journey

        # The lawful changed query is dbSNP's merge: rs61228756 is merged into
        # rs429358 (a lookup of an unrelated rsID is refused).
        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs61228756 in sample HG001.",
            rsid="rs61228756",
            predecessor_ids=(investigation_input.input_id,),
        )
        scientific = SequentialAdaptiveModel()
        calls = 0

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls <= 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            {"rsid": "rs61228756" if calls == 1 else "rs429358"},
                            tool_call_id=f"adapt-{calls}",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(
                refsnp_bodies={"rs61228756": RS61228756, "rs429358": RS429358}
            )
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            self.assertEqual(
                result.status,
                "review_pending",
                [
                    (e["event_type"], e["payload"])
                    for e in read_evidence(journey)
                    if "Failed" in e["event_type"]
                ],
            )
            self.assertEqual(
                sources.operations,
                [
                    "fetch_refsnp:rs61228756",
                    "fetch_refsnp:rs429358",
                    "fetch_assembly_report:GCF_000001405.40",
                ],
            )
            observations = [
                e
                for e in read_evidence(journey)
                if e["event_type"] == "RefSnpToolResultRecorded"
            ]
            self.assertEqual(
                [e["payload"]["result"]["status"] for e in observations],
                ["merged", "resolved"],
            )
            self.assertEqual(
                result.proposals[-1].work_items[0].target.one_based_position, 44908684
            )
            before = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, result
            )
            self.assertEqual(journey.path.read_bytes(), before)
            from copy import deepcopy

            from bio_neb.investigation.investigation import project_investigation

            for field, value in (
                ("query", "rs123"),
                ("tool_call_id", "another-call"),
                ("collection_member_id", "another-member"),
            ):
                with self.subTest(changed=field):
                    changed = deepcopy(read_evidence(journey))
                    source = next(
                        e
                        for e in changed
                        if e["event_id"] == observations[0]["event_id"]
                    )
                    source["payload"][field] = value
                    with self.assertRaisesRegex(
                        ValueError, "member, query or call identity"
                    ):
                        project_investigation(changed)

    def test_domain_tool_receives_only_shared_call_context(self):
        from functools import wraps
        from unittest.mock import patch

        from bio_neb.agentic_journey.invocation import ToolContext
        from bio_neb.investigation.investigation_agent import INVESTIGATION_TOOLS

        fetch = INVESTIGATION_TOOLS["fetch_refsnp"]
        contexts = []

        @wraps(fetch)
        def checked(context, *args, **kwargs):
            contexts.append(context)
            return fetch(context, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(INVESTIGATION_TOOLS, {"fetch_refsnp": checked}),
        ):
            result, _journey = run_attempt(
                Path(directory), SequentialAdaptiveModel(), SourceResponsesOnly()
            )
        self.assertEqual(result.status, "review_pending")
        self.assertEqual(len(contexts), 1)
        self.assertIs(type(contexts[0]), ToolContext)
        self.assertEqual(set(vars(contexts[0])), {"deps", "tool_call_id"})

    def test_domain_output_repair_preserves_sources_and_reopens_corrected_proposal(
        self,
    ):
        from pydantic_ai.messages import RetryPromptPart

        from bio_neb.application.queries import read_investigation
        from bio_neb.journey import Journey

        scientific = SequentialAdaptiveModel()
        refused = False
        feedback = []

        async def respond(messages, info):
            nonlocal refused
            feedback.extend(
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart) and isinstance(part.content, str)
            )
            response = await scientific.respond(messages, info)
            call = response.parts[0]
            if call.tool_name == "compile_proposal" and not refused:
                arguments = json.loads(call.args)
                arguments["one_based_position"] = 2
                call.args = json.dumps(arguments)
                refused = True
            return response

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            self.assertEqual(result.status, "review_pending")
            self.assertEqual(len(sources.operations), 2)
            events = read_evidence(journey)
            outputs = [
                e["payload"]
                for e in events
                if e["event_type"] == "AgentOutputValidated"
            ]
            self.assertEqual([item["status"] for item in outputs], ["invalid", "valid"])
            # A re-typed locus field is refused by its field, never
            # overwritten; the accepted output names no locus.
            self.assertEqual(
                outputs[0]["errors"], ["one_based_position:extra_forbidden"]
            )
            self.assertIn('"type": "extra_forbidden"', str(feedback))
            self.assertNotIn("one_based_position", outputs[-1]["output"])
            before = journey.path.read_bytes()
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation.status,
                "review_pending",
            )
            self.assertEqual(journey.path.read_bytes(), before)

    def test_public_begin_validates_without_acquisition_and_releases_its_one_client(
        self,
    ):
        from bio_neb.application.advancement import (
            ResumeRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.queries import read_investigation
        from bio_neb.investigation.investigation_commands import (
            BeginInvestigationCommand,
        )
        from bio_neb.journey import Journey
        from tests.sequential_investigation_support import forked_child

        for cleanup in (None, RuntimeError("controlled cleanup failure")):
            with (
                self.subTest(cleanup=cleanup is not None),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                request = {
                    "schema_version": "bio-neb.investigation-request/2",
                    "asserted_by": "researcher:owner",
                    "request": "Investigate rs429358 in sample HG001.",
                    "target": {"rsid": "rs429358"},
                }
                journey = forked_child(root, investigation_request=request)
                request_file = root / "request.json"
                request_file.write_text(json.dumps(request))
                command = BeginInvestigationCommand(
                    journey_file=str(journey.path),
                    request_files=(str(request_file),),
                )
                scientific = SequentialAdaptiveModel()
                sources = SourceResponsesOnly()
                models = []

                def factory(
                    loaded,
                    selected,
                    scientific=scientific,
                    cleanup=cleanup,
                    models=models,
                ):
                    model = OwnedModelEndpoint(scientific.respond, close_error=cleanup)
                    models.append(model)
                    return model(loaded, selected)

                # The Fork child's inherited registration binds the runtime; its
                # unavailable credential is retained before the start is written
                # and holds the Run, with no model or source contact.
                with self.assertRaises(RunAdvancementRefused) as held:
                    begin_in_pass(
                        command,
                        environ={},
                        model_factory=factory,
                        source_adapter_factory=lambda sources=sources: sources,
                    )
                self.assertEqual(
                    (
                        held.exception.status,
                        held.exception.failure.failure_class,
                        held.exception.code,
                    ),
                    ("run_held", "invariant", "model_credential_unavailable"),
                )
                types = [e["event_type"] for e in Journey.open(journey.path).read()]
                self.assertEqual(types.count("RuntimeConfigurationFailed"), 1)
                self.assertNotIn("InvestigationStarted", types)
                self.assertEqual(models, [])
                self.assertEqual(sources.operations, [])
                # The fix is the credential; resume lifts the hold and begins.
                resumed = use_run_advancement_dependencies(
                    environ={"LOCAL_MODEL_KEY": "offline"},
                    investigation_model_factory=factory,
                    investigation_source_adapter_factory=lambda sources=sources: (
                        sources
                    ),
                )
                if cleanup is None:
                    with resumed:
                        advance(ResumeRun(journey_file=str(journey.path)))
                else:
                    # The untyped cleanup fault holds the Run, keeping its cause.
                    with resumed, self.assertRaises(RunAdvancementRefused) as raised:
                        advance(ResumeRun(journey_file=str(journey.path)))
                    self.assertEqual(
                        raised.exception.code, "coordinator_unhandled_error"
                    )
                    self.assertIs(raised.exception.__cause__, cleanup)
                before = journey.path.read_bytes()
                reopened = read_investigation(Journey.open(journey.path))

                self.assertEqual(reopened.investigation.status, "review_pending")
                self.assertEqual(journey.path.read_bytes(), before)
                self.assertEqual(len(models), 1)
                self.assertEqual(
                    (models[0].transport.requests, models[0].transport.close_started),
                    (3, 1),
                )
                self.assertTrue(models[0].transport.closed)
                self.assertEqual(len(sources.operations), 2)
                self.assertFalse(
                    any(
                        e["event_type"] == "InvestigationAttemptFailed"
                        for e in journey.read()
                    )
                )

    def test_investigation_receives_context_and_reads_same_run_history(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        scientific = SequentialAdaptiveModel()
        calls = []

        async def respond(messages, info):
            tools = {tool.name for tool in info.function_tools}
            assert "search_journey" in tools
            calls.append(messages)
            if len(calls) == 1:
                text = str(messages)
                assert "history_grants_effect_authority" in text
                assert "Resolve the exact scientific request" in text
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "search_journey",
                            {"kinds": ["InvestigationStarted"]},
                            tool_call_id="inspect-scientific-history",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        model = SimpleNamespace(model=ScriptedModelEndpoint(respond))
        sources = SourceResponsesOnly()
        result, journey = run_attempt(tmp_path, model, sources)
        assert result.status == "review_pending"
        assert len(sources.operations) == 2
        events = read_evidence(journey)
        receipt = next(
            event for event in events if event["event_type"] == "JourneySearchRead"
        )
        assert len(receipt["payload"]["result"]["events"]) == 1
        assert (
            receipt["payload"]["result"]["events"][0]["event_type"]
            == "InvestigationStarted"
        )
        assert (
            len(
                [
                    event
                    for event in events
                    if event["event_type"] == "ObservedToolExecutionSucceeded"
                ]
            )
            == 3
        )

    def test_reopen_refuses_orphan_cross_call_and_stale_tool_receipts(self):
        import copy

        from bio_neb.investigation.investigation import project_investigation

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            result, journey = run_attempt(
                Path(directory), SequentialAdaptiveModel(), sources
            )
            self.assertEqual(result.status, "review_pending")
            events = read_evidence(journey)
            successes = [
                e for e in events if e["event_type"] == "ObservedToolExecutionSucceeded"
            ]
            self.assertEqual(len(successes), 2)
            first, second = successes
            changes = {
                "orphan": (
                    "started_event_id",
                    "00000000-0000-0000-0000-000000000000",
                    "terminal without its exact start",
                ),
                "cross_call": (
                    "stage_result_event_id",
                    first["payload"]["stage_result_event_id"],
                    "source Tool terminal changed its stage result",
                ),
                "stale_attempt": (
                    "attempt_id",
                    "00000000-0000-0000-0000-000000000000",
                    "source Tool terminal changed its unique call",
                ),
                "wrong_return": (
                    "model_visible_result_digest",
                    "0" * 64,
                    "canonical result digest",
                ),
            }
            before = journey.path.read_bytes()
            for case, (field, value, reason) in changes.items():
                with self.subTest(case=case):
                    hostile = copy.deepcopy(events)
                    changed = next(
                        e for e in hostile if e["event_id"] == second["event_id"]
                    )
                    changed["payload"][field] = value
                    with self.assertRaisesRegex(ValueError, reason):
                        project_investigation(hostile)
            self.assertEqual(journey.path.read_bytes(), before)
            self.assertEqual(len(sources.operations), 2)

    def test_investigation_source_tools_keep_agent_chosen_arguments_visible(self):
        scientific = SequentialAdaptiveModel()
        offered = []

        async def respond(messages, info):
            offered.append({tool.name: tool for tool in info.function_tools})
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            model = SimpleNamespace(model=ScriptedModelEndpoint(respond))
            result, journey = run_attempt(Path(directory), model, SourceResponsesOnly())
            # Server-bound values are recorded with the request, never sent.
            recorded = [
                {
                    tool["name"]: tool
                    for tool in e["payload"]["compatible_adapter_evidence"]["request"][
                        "function_tools"
                    ]
                }
                for e in read_evidence(journey)
                if e["event_type"] == "ModelRequestRecorded"
            ]

        self.assertEqual(result.status, "review_pending")
        self.assertEqual(len(offered), 3)
        self.assertEqual(len(recorded), 3)
        for tools, definitions in zip(offered, recorded, strict=True):
            self.assertEqual(
                set(tools),
                {
                    # Both rsID lookup Tools on every Run.
                    "fetch_refsnp",
                    "fetch_eva_variant",
                    # A stated position's lookup is offered on every Run.
                    "locate_variant",
                    # The searchable view of a lookup's sequence names.
                    "search_sequence_names",
                    # Each decision a lookup asks for, its own Tool.
                    "name_sequence",
                    "name_version",
                    "name_placement",
                    "quote_listed_none",
                    # NCBI's genome reports, to name a stated version.
                    "search_assemblies",
                    "fetch_assembly_report",
                    "search_journey",
                    "request_investigation_information",
                    # The model types the target a member's words state.
                    "record_target_reading",
                    # A typed gene change is placed, then a candidate picked.
                    "place_gene_change",
                    "pick_gene_change_candidate",
                    # A member's paper, read with the reading Tools; they wait for a
                    # member no paper proposed, as here.
                    "outline_paper",
                    "grep_paper",
                    "read_paper",
                    "filter_paper_rows",
                },
            )
            for name, argument in (
                ("fetch_refsnp", "rsid"),
                ("fetch_assembly_report", "assembly_accession"),
            ):
                self.assertIn(
                    argument, tools[name].parameters_json_schema["properties"]
                )
                self.assertEqual(definitions[name]["metadata"]["bound_arguments"], {})
                self.assertIn(
                    "pattern",
                    tools[name].parameters_json_schema["properties"][argument],
                )

    def test_investigation_requests_carry_the_runs_one_tool_choice(self):
        # Investigation ends through its compile_proposal output Tool, yet nothing
        # forces the call: every request records and sends the Run's one choice,
        # auto.
        with tempfile.TemporaryDirectory() as directory:
            endpoint = ScriptedModelEndpoint(SequentialAdaptiveModel().respond)
            model = SimpleNamespace(model=endpoint)
            result, journey = run_attempt(Path(directory), model, SourceResponsesOnly())
            requests = [
                e["payload"]["compatible_adapter_evidence"]["request"]
                for e in read_evidence(journey)
                if e["event_type"] == "ModelRequestRecorded"
            ]

        self.assertEqual(result.status, "review_pending")
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(endpoint.bodies), 3)
        for request, body in zip(requests, endpoint.bodies, strict=True):
            self.assertEqual(request["model_settings"]["tool_choice"], "auto")
            self.assertEqual(
                [tool["name"] for tool in request["output_tools"]],
                ["compile_proposal"],
            )
            self.assertIs(request["allow_text_output"], False)
            self.assertEqual(body["tool_choice"], "auto")

    def test_a_stage_may_name_its_output_tool_as_the_tool_choice(self):
        # The output Tool is one the stage offers: a configured choice naming
        # it is recorded with each request and sent by name. A named choice
        # forces a call, so it needs the model's recorded grammar support.
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            ModelRegistry,
            model_registry,
            use_model_registry,
        )

        registry = model_registry()
        supported = {
            model: replace(entry, grammar_evidence="test")
            for model, entry in registry.entries.items()
        }
        with (
            use_model_registry(
                ModelRegistry(
                    supported,
                    supported[registry.default.model],
                    registry.default_reference,
                )
            ),
            tempfile.TemporaryDirectory() as directory,
        ):
            endpoint = ScriptedModelEndpoint(SequentialAdaptiveModel().respond)
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=endpoint),
                SourceResponsesOnly(),
                agent_invocation={
                    "scopes": {
                        "stage": {"source_reading": {"tool_choice": "compile_proposal"}}
                    }
                },
            )
            requests = [
                e["payload"]["compatible_adapter_evidence"]["request"]
                for e in read_evidence(journey)
                if e["event_type"] == "ModelRequestRecorded"
            ]

        self.assertEqual(result.status, "review_pending")
        self.assertTrue(requests)
        self.assertEqual(len(requests), len(endpoint.bodies))
        for request, body in zip(requests, endpoint.bodies, strict=True):
            self.assertEqual(
                request["model_settings"]["tool_choice"], "compile_proposal"
            )
            self.assertEqual(
                body["tool_choice"],
                {"type": "function", "function": {"name": "compile_proposal"}},
            )

    def test_the_output_tool_reaches_the_model_through_its_schema_projection(self):
        # The selected model's endpoint policy projects every Tool definition it is
        # shown, the output Tool included; a stand-in projection proves it.
        from dataclasses import replace
        from unittest.mock import patch

        from bio_neb.agentic_journey.model_settings import ModelEndpointPolicy

        def project(_policy, definition):
            return replace(definition, description=definition.description + " [P]")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(ModelEndpointPolicy, "project_tool_schema", project),
        ):
            model = SimpleNamespace(
                model=ScriptedModelEndpoint(SequentialAdaptiveModel().respond)
            )
            _result, journey = run_attempt(
                Path(directory), model, SourceResponsesOnly()
            )
            requests = [
                e["payload"]["compatible_adapter_evidence"]["request"]
                for e in read_evidence(journey)
                if e["event_type"] == "ModelRequestRecorded"
            ]
        self.assertTrue(requests)
        for request in requests:
            shown = [*request["function_tools"], *request["output_tools"]]
            self.assertIn("compile_proposal", [tool["name"] for tool in shown])
            self.assertEqual(
                [t["name"] for t in shown if not t["description"].endswith(" [P]")],
                [],
            )

    def test_investigation_agent_retrieves_a_referenced_exchange_and_continues(self):
        import json

        from pydantic_ai.messages import ToolReturnPart

        from bio_neb.application.queries import read_investigation
        from bio_neb.journey import Journey, canonical_json

        scientific = SequentialAdaptiveModel()
        calls = []
        references = []
        recovered = []

        async def respond(messages, info):
            tools = {tool.name: tool for tool in info.function_tools}
            calls.append(messages)
            if len(calls) == 1:
                self.assertIn(
                    "payload_slice",
                    tools["search_journey"].parameters_json_schema["properties"],
                )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "search_journey",
                            {"kinds": ["ModelResponseRecorded"]},
                            tool_call_id="find-retained-exchange",
                        )
                    ]
                )
            returned = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart)
                and part.tool_name == "search_journey"
            ]
            if len(calls) == 2:
                reference = returned[-1]["result"]["events"][0]["payload_reference"]
                references.append(reference)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "search_journey",
                            {"payload_slice": {"event_id": reference["event_id"]}},
                            tool_call_id="read-retained-exchange",
                        )
                    ]
                )
            if len(calls) == 3:
                chunk = returned[-1]["result"]["payload_slice"]
                self.assertIsNone(chunk["next_offset"])
                recovered.append(chunk["text"])
                self.assertEqual(json.loads(chunk["text"])["request_number"], 1)
                self.assertIn("find-retained-exchange", chunk["text"])
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            model = SimpleNamespace(model=ScriptedModelEndpoint(respond))
            result, journey = run_attempt(Path(directory), model, sources)
            self.assertEqual(result.status, "review_pending")
            self.assertEqual(len(sources.operations), 2)
            events = read_evidence(journey)
            source = next(
                event
                for event in events
                if event["event_id"] == references[0]["event_id"]
            )
            self.assertEqual(recovered, [canonical_json(source["payload"])])
            self.assertEqual(
                read_investigation(Journey.open(journey.path)),
                read_investigation(journey),
            )
            changed = journey.read()
            receipt = next(
                event
                for event in reversed(changed)
                if event["event_type"] == "JourneySearchRead"
            )
            receipt["payload"]["result"]["payload_slice"]["text"] = "forged exchange"
            journey.path.write_text(
                "".join(canonical_json(event) + "\n" for event in changed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "search receipt changed its exact Run prefix"
            ):
                read_investigation(Journey.open(journey.path))

    def test_ordered_premature_calls_exhaust_with_explanations_and_reopen(self):
        from pydantic_ai.messages import ToolReturnPart

        from bio_neb.agentic_journey.agent_work_review import project_agent_work_review
        from bio_neb.application.queries import read_investigation
        from bio_neb.journey import Journey

        corrections = []

        async def respond(messages, info):
            corrections.extend(
                str(part.content)
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_assembly_report",
                        {
                            "assembly_accession": "GCF_000001405.40",
                            "refsnp_event_id": "missing",
                        },
                        tool_call_id="early-a",
                    ),
                    ToolCallPart(
                        "fetch_assembly_report",
                        {
                            "assembly_accession": "GCF_000001405.40",
                            "refsnp_event_id": "missing",
                        },
                        tool_call_id="early-b",
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            self.assertEqual(
                read_investigation(Journey.open(journey.path)),
                read_investigation(journey),
            )
            self.assertEqual(
                (result.status, result.stop_reason), ("agent_work_review_pending", None)
            )
            self.assertEqual(sources.operations, [])
            events = read_evidence(journey)
            dispositions = [
                e
                for e in events
                if e["event_type"] == "AgentActionTurnDispositionRecorded"
            ]
            # Each reply's second call copies its first: it collapses, and its
            # first call is refused. Nothing of the reply worked, so each is a
            # failed turn, and the fourth in a row stops the Attempt.
            self.assertEqual(
                [e["payload"]["correction_count"] for e in dispositions],
                [1, 2, 3, 4],
            )
            self.assertEqual(
                {e["payload"]["disposition"] for e in dispositions},
                {"multi_call_refused"},
            )
            routed = [e for e in events if e["event_type"] == "AgentTurnRouted"]
            self.assertEqual(
                [e["payload"]["faults"] for e in routed],
                [[{"key": "copied_calls", "count": count}] for count in range(1, 5)],
            )
            self.assertEqual(
                {tuple(e["payload"]["copies"].items()) for e in routed},
                {(("early-b", "early-a"),)},
            )
            started = next(e for e in events if e["event_type"] == "AttemptStarted")
            exhaustion = started["payload"]["effective_invocation_policy"]["exhaustion"]
            self.assertNotIn("repeated_multi_action", exhaustion)
            self.assertEqual(exhaustion["failed_turns"], "review")
            review = project_agent_work_review(events).pending_request
            self.assertEqual(
                (review.cause, review.source_event_id),
                ("failed_turns", dispositions[-1]["event_id"]),
            )
            self.assertTrue(
                any(
                    e["payload"].get("disposition") == "multi_call_refused"
                    for e in events
                )
            )
            self.assertFalse(
                any(e["event_type"] == "ObservedToolExecutionStarted" for e in events)
            )
        text = str(corrections)
        self.assertIn("fetch_assembly_report", text)
        # The slot is the lookup the call cites, with the value sent (it
        # blamed `assembly_accession` before).
        self.assertIn('`refsnp_event_id` is "missing"', text)
        self.assertIn("fetch_refsnp first", text)

    def test_a_followed_merge_preserves_request_and_exposes_the_verified_rsid(self):
        # Replaces the agent-chosen rsID case: an unrelated lookup is refused
        # before contact, so a verified rsID differs from the requested one only
        # through dbSNP's documented merge.
        from bio_neb.application.queries import read_investigation
        from bio_neb.journey import Journey

        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs61228756 in sample HG001.",
            rsid="rs61228756",
            predecessor_ids=(investigation_input.input_id,),
        )
        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(
                refsnp_bodies={"rs61228756": RS61228756, "rs429358": RS429358}
            )
            result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            self.assertEqual(result.status, "review_pending", str(result))
            self.assertEqual(
                sources.operations[:2],
                ["fetch_refsnp:rs61228756", "fetch_refsnp:rs429358"],
            )
            member = result.pending_request.proposal.request_members[0]
            self.assertEqual(member.request.rsid, "rs61228756")
            self.assertEqual(
                (member.refsnp.requested_rsid, member.refsnp.verified_rsid),
                ("rs61228756", "rs429358"),
            )
            self.assertIn("rs61228756", member.refsnp.merged_rsids)
            slot = result.pending_request.fact_slots[0]
            self.assertEqual(
                (slot.requested_value, slot.verified_value, slot.mismatch),
                ("rs61228756", "rs429358", True),
            )
            self.assertEqual(slot.verification_source, "NCBI RefSNP")
            self.assertEqual(result.requirement_values[-1].status, "verified")
            self.assertEqual(result.requirement_values[-1].source, "tool_result")
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation.fact_slots,
                result.fact_slots,
            )

    def test_negative_source_result_and_combined_findings_reach_next_request(self):
        import json

        from pydantic_ai.messages import ToolReturnPart

        from bio_neb.agentic_journey.tool_execution import negative_domain_result

        scientific = SequentialAdaptiveModel()
        views = []
        returns = []

        async def respond(messages, info):
            # The request's view is the JSON line its instructions end with.
            views.append(
                json.loads(info.instructions.rpartition("\n")[2])["stage_findings"]
            )
            # A Tool result reaches the model as its sent JSON.
            returns.extend(
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.tool_name == "fetch_refsnp"
            )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(refsnp_body=b'{"refsnp_id":"429358"}')
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            self.assertEqual(result.status, "review_pending")
            self.assertEqual(sources.operations, ["fetch_refsnp:rs429358"])
            event = next(
                e
                for e in read_evidence(journey)
                if e["event_type"] == "RefSnpToolResultRecorded"
            )
            body = b'{"refsnp_id":"429358"}'
            # The model reads the result's status, the source's statement, what
            # dbSNP answered and one next step; the Tool ran (OQ1).
            (view,) = returns
            self.assertEqual(view["status"], "source_answered")
            self.assertEqual(view["cite"], {"refsnp_event_id": event["event_id"]})
            self.assertEqual(
                view["source_answers"],
                [
                    {
                        "operation": "fetch_refsnp",
                        "request": (
                            "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358"
                        ),
                        "status_code": 200,
                        "shown": "whole",
                        "body": json.loads(body),
                    }
                ],
            )
            self.assertNotIn("source_answers", view["statement"])
            self.assertIn("source_answers", view["next"])
            self.assertTrue(
                any(
                    e["event_type"] == "ObservedToolExecutionSucceeded"
                    and e["payload"]["stage_result_event_id"] == event["event_id"]
                    for e in read_evidence(journey)
                )
            )
            self.assertTrue(negative_domain_result("investigation_planner", event))
        self.assertEqual(views[0]["findings"], [])
        self.assertEqual(views[1]["findings"][0]["status"], "source_answered")
        self.assertEqual(views[1]["output"], "verified")

    def test_source_tool_returns_are_their_rows_model_view_beside_the_event_id(self):
        # RED-28: the model can cite the exact recorded result it was given, and
        # reads beside it what each source answered. Each return is its row's
        # model view, built from the fact without hashes, timings or internal ids,
        # and the fact's id rides in the metadata.
        from pydantic_ai.messages import ToolReturnPart

        from bio_neb.agentic_journey.tool_execution import (
            model_view_of,
            tool_model_view,
        )

        scientific = SequentialAdaptiveModel()
        returns = {}

        async def respond(messages, info):
            for message in messages:
                for part in message.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name in {
                        "fetch_refsnp",
                        "fetch_assembly_report",
                    }:
                        returns[part.tool_name] = (part.content, part.metadata)
            return await scientific.respond(messages, info)

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            self.assertEqual(result.status, "review_pending")
            events = read_evidence(journey)
            recorded = {
                event["event_type"]: event
                for event in events
                if event["event_type"]
                in {"RefSnpToolResultRecorded", "AssemblyReportToolResultRecorded"}
            }
            kept = {
                part["tool_name"]: part["metadata"]
                for event in events
                if event["event_type"] == "ModelRequestRecorded"
                for message in event["payload"]["compatible_adapter_evidence"][
                    "request"
                ]["messages"]
                for part in message["parts"]
                if part["part_kind"] == "tool-return"
            }
        for tool, event_type, cite, body in (
            (
                "fetch_refsnp",
                "RefSnpToolResultRecorded",
                "refsnp_event_id",
                sources.refsnp_body,
            ),
            (
                "fetch_assembly_report",
                "AssemblyReportToolResultRecorded",
                "assembly_event_id",
                sources.assembly_body,
            ),
        ):
            with self.subTest(tool=tool):
                view, sent_metadata = returns[tool]
                event = recorded[event_type]
                self.assertEqual(view, tool_model_view(model_view_of(tool), event))
                # The fact's id is recorded beside the view and never sent.
                self.assertIsNone(sent_metadata)
                self.assertEqual(kept[tool], {"event_id": event["event_id"]})
                self.assertEqual(view["cite"], {cite: event["event_id"]})
                self.assertEqual(view["status"], event["payload"]["result"]["status"])
                self.assertTrue(view["next"])
                [answer] = view["source_answers"]
                self.assertEqual(
                    answer,
                    {
                        "operation": tool,
                        "request": event["payload"]["source_answers"][0][
                            "request_identity"
                        ],
                        "status_code": 200,
                        "shown": "whole",
                        "body": json.loads(body),
                    },
                )
                self.assertFalse(
                    set(keys(view))
                    & {
                        "sha256",
                        "byte_count",
                        "raw_response_sha256",
                        "observed_at",
                        "elapsed_time_ms",
                        "result_id",
                        "source_id",
                        "assessment_id",
                        "evidence_id",
                        "attempt_id",
                    }
                )

    def test_invalid_lookup_is_refused_before_ordered_source_calls(self):
        scientific = SequentialAdaptiveModel()
        calls = 0

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            {"rsid": "../../7412"},
                            tool_call_id="bad-shape",
                        )
                    ]
                )
            if calls == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp",
                            {"rsid": "rs429358"},
                            tool_call_id="rsid-valid",
                        ),
                    ]
                )
            if calls == 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_assembly_report",
                            {
                                "assembly_accession": "GCF_000001405.40",
                                "refsnp_event_id": source_basis(info)[
                                    "refsnp_event_id"
                                ],
                            },
                            tool_call_id="assembly-valid",
                        ),
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            self.assertEqual(result.status, "review_pending")
            self.assertEqual(
                sources.operations,
                ["fetch_refsnp:rs429358", "fetch_assembly_report:GCF_000001405.40"],
            )
            events = read_evidence(journey)
            invalid = next(
                e
                for e in events
                if e["event_type"] == "ToolCallValidated"
                and e["payload"]["tool_call_id"] == "bad-shape"
            )
            self.assertEqual(invalid["payload"]["status"], "invalid")
            self.assertIn(
                "tool_arguments_invalid:rsid", invalid["payload"]["errors"][0]
            )
            starts = [
                e for e in events if e["event_type"] == "ObservedToolExecutionStarted"
            ]
            self.assertEqual(
                [e["payload"]["tool_call_id"] for e in starts],
                ["rsid-valid", "assembly-valid"],
            )
            self.assertEqual(result.model_contact_count, 4)

    def test_several_source_alts_do_not_imply_an_rsid_mismatch(self):
        # Was the allele-ambiguity case: several ALTs are now one candidate Work
        # item each; the requested rsID still verifies without a mismatch.
        import copy
        import json

        from tests.sequential_investigation_support import REFSNP_BODY

        body = json.loads(REFSNP_BODY)
        alleles = body["primary_snapshot_data"]["placements_with_allele"][0]["alleles"]
        alternate = copy.deepcopy(alleles[1])
        alternate["allele"]["spdi"]["inserted_sequence"] = "A"
        alleles.append(alternate)
        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(refsnp_body=json.dumps(body).encode())
            result, _ = run_attempt(Path(directory), SequentialAdaptiveModel(), sources)
        self.assertEqual(result.status, "review_pending")
        proposal = result.pending_request.proposal
        self.assertEqual([item.target.alt for item in proposal.work_items], ["A", "C"])
        member = proposal.request_members[0]
        self.assertEqual(member.refsnp.status, "resolved")
        self.assertEqual(
            member.work_item_ids,
            tuple(item.work_item_id for item in proposal.work_items),
        )
        self.assertFalse(result.fact_slots[0].mismatch)
        self.assertEqual(result.fact_slots[0].verified_value, "rs429358")
        self.assertEqual(
            sources.operations,
            ["fetch_refsnp:rs429358", "fetch_assembly_report:GCF_000001405.40"],
        )

    def test_agent_chosen_assembly_is_checked_against_candidate(self):
        import json

        from tests.sequential_investigation_support import ASSEMBLY_BODY

        scientific = SequentialAdaptiveModel()
        body = json.loads(ASSEMBLY_BODY)
        body["reports"][0]["accession"] = "GCF_000001405.39"

        async def respond(messages, info):
            response = await scientific.respond(messages, info)
            part = response.parts[0]
            if part.tool_name == "fetch_assembly_report":
                part.args = json.dumps(
                    {**part.args_as_dict(), "assembly_accession": "GCF_000001405.39"}
                )
            return response

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(assembly_body=json.dumps(body).encode())
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            (assembly,) = [
                e
                for e in journey.read()
                if e["event_type"] == "AssemblyReportToolResultRecorded"
            ]
        self.assertEqual(result.status, "review_pending")
        self.assertEqual(
            sources.operations[-1], "fetch_assembly_report:GCF_000001405.39"
        )
        # The unresolved finding asks its own issue Review: no proposal.
        self.assertEqual(result.proposals, ())
        issue = result.pending_request.question.issue
        self.assertEqual(
            (issue.source, issue.verdict), ("assembly_report", "incompatible")
        )
        self.assertEqual(issue.evidence_event_ids[-1], assembly["event_id"])
        self.assertEqual(assembly["payload"]["result"]["status"], "incompatible")
        self.assertEqual(
            assembly["payload"]["result"]["expected_assembly_accession"],
            "GCF_000001405.40",
        )


class RequestBoundLookupTests(unittest.TestCase):
    """Every RefSNP lookup is bound to the member's request or a documented merge."""

    def test_a_lookup_of_another_rsid_is_refused_before_contact(self):
        # Request rs429358, query rs7412.
        scientific = SequentialAdaptiveModel()
        seen: list[str] = []

        async def respond(messages, info):
            seen.append(ModelMessagesTypeAdapter.dump_json(messages).decode())
            if len(seen) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "fetch_refsnp", {"rsid": "rs7412"}, tool_call_id="other"
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly()
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
            )
            events = read_evidence(journey)

        self.assertEqual(result.status, "review_pending")
        self.assertNotIn("fetch_refsnp:rs7412", sources.operations)
        self.assertEqual(sources.operations[0], "fetch_refsnp:rs429358")
        refused = next(
            event
            for event in events
            if event["event_type"] == "ToolCallValidated"
            and event["payload"]["tool_call_id"] == "other"
        )
        self.assertEqual(refused["payload"]["status"], "invalid")
        self.assertEqual(
            refused["payload"]["errors"], ["fetch_refsnp_query_not_requested"]
        )
        self.assertIn(
            "Look up only an rsID this member's researcher stated",
            seen[1],
        )

    def test_a_merge_target_is_looked_up_once_a_retained_result_names_it(self):
        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs60037639 in sample HG001.",
            rsid="rs60037639",
            predecessor_ids=(investigation_input.input_id,),
        )
        calls = 0

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            lookup = {1: "rs4149056", 2: "rs60037639", 3: "rs4149056"}.get(calls)
            if lookup is not None:
                call = ToolCallPart(
                    "fetch_refsnp", {"rsid": lookup}, tool_call_id=f"lookup-{calls}"
                )
            else:
                # Close the member on the merged record, which needs no
                # candidate shape: its conflicts are the unresolved output's.
                merged = next(
                    row
                    for row in json.loads(info.instructions.rpartition("\n")[2])[
                        "stage_findings"
                    ]["findings"]
                    if row["status"] == "merged"
                )
                call = ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "refsnp_event_id": merged["source_event_id"],
                        "outcome": "unresolved",
                    },
                    tool_call_id="close",
                )
            return ModelResponse(parts=[call])

        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(
                refsnp_bodies={
                    "rs60037639": RS60037639,
                    "rs4149056": RS4149056,
                }
            )
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(
            sources.operations, ["fetch_refsnp:rs60037639", "fetch_refsnp:rs4149056"]
        )
        validations = {
            event["payload"]["tool_call_id"]: event["payload"]
            for event in events
            if event["event_type"] == "ToolCallValidated"
        }
        self.assertEqual(
            validations["lookup-1"]["errors"], ["fetch_refsnp_query_not_requested"]
        )
        self.assertEqual(validations["lookup-3"]["status"], "valid")
        self.assertEqual(result.status, "review_pending")


def _merged_from_request_body() -> bytes:
    """rs4149056's retained record without the merge entry for rs60037639.

    Derived, synthetic inconsistency: rs60037639's record names rs4149056 as its
    current rsID, but this version of rs4149056's record does not list
    rs60037639 among the rsIDs merged into it.
    """
    document = copy.deepcopy(RS4149056_DOCUMENT)
    document["dbsnp1_merges"] = [
        merge
        for merge in document["dbsnp1_merges"]
        if merge["merged_rsid"] != "60037639"
    ]
    return refsnp_body(document)


def _merged_request():
    investigation_input, _ = input_and_assertion()
    return investigation_input, UserAssertion.create(
        asserted_by="researcher:owner",
        request_text="Investigate rs60037639 in sample HG001.",
        rsid="rs60037639",
        predecessor_ids=(investigation_input.input_id,),
    )


class RetrievedTargetIntentTests(unittest.TestCase):
    """Before the proposal Review, each retrieved target is checked against its request."""

    def test_a_documented_merge_is_accepted_and_recorded_as_a_rename(self):
        investigation_input, assertion = _merged_request()
        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(
                refsnp_bodies={"rs60037639": RS60037639, "rs4149056": RS4149056}
            )
            result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(
            sources.operations[:2],
            ["fetch_refsnp:rs60037639", "fetch_refsnp:rs4149056"],
        )
        self.assertFalse(
            [
                event
                for event in events
                if event["event_type"] == "AgentOutputValidated"
                and event["payload"]["status"] == "invalid"
            ]
        )
        self.assertIsNone(result.pending_request.question)
        member = result.pending_request.proposal.request_members[0]
        self.assertEqual(
            (member.request.rsid, member.refsnp.requested_rsid),
            ("rs60037639", "rs60037639"),
        )
        self.assertEqual(member.refsnp.verified_rsid, "rs4149056")
        self.assertIn("rs60037639", member.refsnp.merged_rsids)

    def test_an_undocumented_difference_is_corrected_then_asked(self):
        investigation_input, assertion = _merged_request()
        scientific = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(
                refsnp_bodies={
                    "rs60037639": RS60037639,
                    "rs4149056": _merged_from_request_body(),
                }
            )
            result, journey = run_attempt(
                Path(directory),
                scientific,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)
            from bio_neb.application.queries import read_investigation
            from bio_neb.journey import Journey

            reopened = read_investigation(Journey.open(journey.path)).investigation

        refused = [
            event
            for event in events
            if event["event_type"] == "AgentOutputValidated"
            and event["payload"]["status"] == "invalid"
        ]
        reason = "requested rs60037639, retrieved rs4149056"
        # The refusal names the question Tool as its next action.
        ask = (
            "if the request means another target, ask the researcher which one "
            "with request_investigation_information"
        )
        # The default's fourth failed turn in a row stops (four repairs, two
        # before): the fourth refusal asks.
        self.assertEqual(
            [event["payload"]["errors"] for event in refused],
            # The difference is typed after the code.
            [["retrieved_target_not_requested", "rsid", reason, ask]] * 4,
        )
        # The failure module wrote the correction the model read next.
        self.assertIn(reason, refused[0]["payload"]["feedback"])
        self.assertIn(reason, scientific.visible_messages[-1])
        self.assertEqual(result.proposals, ())
        self.assertEqual(result.attempts[-1].outcome, "question_pending")
        question = result.pending_request.question
        self.assertEqual(question.request.rsid, "rs60037639")
        self.assertEqual(question.refused_output_event_id, refused[-1]["event_id"])
        self.assertEqual(
            question.question,
            # Either source's record: the question names neither.
            "The source returned a different rsID than this request names "
            f"({reason}), and no merge record of the source links them. Clarify "
            "with the rsID you mean, or reject this request.",
        )
        self.assertEqual(result.pending_request.legal_actions, ("clarify", "reject"))
        # The one item-question shape: it cites the lookup the refused output
        # rested on and copies that result's statement.
        cited = refused[-1]["payload"]["output"]["refsnp_event_id"]
        (lookup,) = [e for e in events if e["event_id"] == cited]
        self.assertEqual(question.cited_event_id, cited)
        self.assertEqual(
            question.statement, lookup["payload"]["result"]["assessment"]["rationale"]
        )
        self.assertEqual(reopened, result)

    def test_a_clarify_answer_naming_the_retrieved_rsid_admits_it(self):
        from bio_neb.investigation.investigation import ClarifyInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )

        investigation_input, assertion = _merged_request()
        scientific = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(
                refsnp_bodies={
                    "rs60037639": RS60037639,
                    "rs4149056": _merged_from_request_body(),
                }
            )
            asked, journey = run_attempt(
                Path(directory),
                scientific,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            question = asked.pending_request.question
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=asked.pending_request.request_event_id,
                    response=ClarifyInvestigationResponse(
                        action="clarify",
                        collection_member_id=question.collection_member_id,
                        revision_number=1,
                        instruction="Yes, rs4149056 is the variant I mean.",
                        asserted_by="researcher:owner",
                    ),
                )
            )
            completed = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=scientific.model,
                source_adapter_factory=lambda: sources,
            )

        self.assertEqual(completed.status, "review_pending")
        proposal = completed.pending_request.proposal
        member = proposal.request_members[0]
        self.assertEqual(
            (member.request.rsid, member.refsnp.verified_rsid),
            ("rs60037639", "rs4149056"),
        )
        self.assertTrue(proposal.work_items)
        self.assertEqual(
            {item.target.contig for item in proposal.work_items}, {"NC_000012.12"}
        )


class PerAltCandidateTests(unittest.TestCase):
    """Several source-backed ALTs become candidate Work items; approval selects."""

    def test_each_source_alt_is_a_candidate_and_approval_selects_one(self):
        from bio_neb.application.queries import read_investigation
        from bio_neb.investigation.investigation import ApproveInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            RespondInvestigationReviewCommand,
            respond_investigation_review,
        )
        from bio_neb.journey import Journey

        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs4149056 in sample HG001.",
            rsid="rs4149056",
            predecessor_ids=(investigation_input.input_id,),
        )
        with tempfile.TemporaryDirectory() as directory:
            sources = SourceResponsesOnly(refsnp_bodies={"rs4149056": RS4149056})
            result, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            proposal = result.pending_request.proposal
            # The retained record lists T>A and T>C.
            self.assertEqual(
                [
                    (
                        item.target.contig,
                        item.target.one_based_position,
                        item.target.ref,
                        item.target.alt,
                    )
                    for item in proposal.work_items
                ],
                [
                    ("NC_000012.12", 21178615, "T", "A"),
                    ("NC_000012.12", 21178615, "T", "C"),
                ],
            )
            member = proposal.request_members[0]
            ids = tuple(item.work_item_id for item in proposal.work_items)
            self.assertEqual(member.work_item_ids, ids)
            self.assertEqual(proposal.unresolved_request_member_ids, ())

            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=result.pending_request.request_event_id,
                    response=ApproveInvestigationResponse(
                        action="approve",
                        proposal_id=proposal.proposal_id,
                        work_item_ids=(ids[1],),
                    ),
                )
            )
            approved = read_investigation(Journey.open(journey.path)).investigation

        self.assertEqual(
            tuple(
                item.work_item_id for item in approved.authority.authorized_work_items
            ),
            (ids[1],),
        )
        self.assertEqual(approved.pending_request.remaining_work_item_ids, (ids[0],))
        self.assertEqual(approved.pending_work_item_ids, (ids[0],))

    def test_indel_alleles_become_work_items_in_the_vcf_comparable_form(self):
        # The RefSNP result keeps dbSNP's reference-expanded SPDI form; each Work
        # item's target is the trimmed, left-aligned allele (its first changed
        # base, an empty side for an insertion or deletion). The VCF forms:
        # rs756427959 is 14397 CTGT>C; rs754204844 is 175160809 TC>T;
        # rs878871521's first ALT is MT 302 A>AC. The derived rs4149056
        # (synthetic: T>A replaced by T>TT) keeps its SNV T>C as is.
        mixed = copy.deepcopy(RS4149056_DOCUMENT)
        placement = mixed["primary_snapshot_data"]["placements_with_allele"][0]
        placement["alleles"][1]["allele"]["spdi"]["inserted_sequence"] = "TT"
        cases = {
            "rs756427959": (RS756427959, [("NC_000001.11", 14398, "TGT", "")]),
            "rs754204844": (RS754204844, [("NC_000001.11", 175160810, "C", "")]),
            "rs878871521": (
                RS878871521,
                [("NC_012920.1", 303, "", "C" * count) for count in (1, 2, 3, 11, 14)],
            ),
            "rs4149056": (
                refsnp_body(mixed),
                [
                    ("NC_000012.12", 21178615, "T", "C"),
                    ("NC_000012.12", 21178615, "", "T"),
                ],
            ),
        }
        for rsid, (body, targets) in cases.items():
            with self.subTest(rsid=rsid):
                investigation_input, _ = input_and_assertion()
                assertion = UserAssertion.create(
                    asserted_by="researcher:owner",
                    request_text=f"Investigate {rsid} in sample HG001.",
                    rsid=rsid,
                    predecessor_ids=(investigation_input.input_id,),
                )
                with tempfile.TemporaryDirectory() as directory:
                    result, _journey = run_attempt(
                        Path(directory),
                        SequentialAdaptiveModel(),
                        SourceResponsesOnly(refsnp_bodies={rsid: body}),
                        investigation_input=investigation_input,
                        assertion=assertion,
                    )

                proposal = result.pending_request.proposal
                self.assertEqual(
                    sorted(
                        (
                            item.target.contig,
                            item.target.one_based_position,
                            item.target.ref,
                            item.target.alt,
                        )
                        for item in proposal.work_items
                    ),
                    sorted(targets),
                )
                self.assertEqual(
                    {item.target.aliases for item in proposal.work_items}, {(rsid,)}
                )

    def test_a_retried_indel_item_reselects_its_own_allele(self):
        # The internal retry phrase names the Work item's empty allele as "-"
        # (rs878871521's insertion "->C" among five), and the retried member
        # keeps exactly that candidate.
        from bio_neb.investigation.investigation import (
            KeepInvestigationMember,
            RetryInvestigationMember,
            ReviseInvestigationResponse,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )

        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create(
            asserted_by="researcher:owner",
            request_text="Investigate rs878871521 in sample HG001.",
            rsid="rs878871521",
            predecessor_ids=(investigation_input.input_id,),
        )
        scientific = SequentialAdaptiveModel()
        sources = SourceResponsesOnly(refsnp_bodies={"rs878871521": RS878871521})
        with tempfile.TemporaryDirectory() as directory:
            original, journey = run_attempt(
                Path(directory),
                scientific,
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            proposal = original.pending_request.proposal
            [retried] = [item for item in proposal.work_items if item.target.alt == "C"]
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=original.pending_request.request_event_id,
                    response=ReviseInvestigationResponse(
                        action="revise",
                        proposal_id=proposal.proposal_id,
                        asserted_by="researcher:owner",
                        operations=tuple(
                            RetryInvestigationMember(
                                operation="retry", work_item_id=item.work_item_id
                            )
                            if item is retried
                            else KeepInvestigationMember(
                                operation="keep", work_item_id=item.work_item_id
                            )
                            for item in proposal.work_items
                        ),
                    ),
                )
            )
            continued = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=scientific.model,
                source_adapter_factory=lambda: sources,
            )

        self.assertEqual(continued.status, "review_pending")
        revised = continued.proposals[-1]
        member = revised.request_members[-1]
        self.assertEqual(
            member.request.request_text,
            "Retry observe_variant for rs878871521 in sample HG001; allele "
            "NC_012920.1:303 ->C.",
        )
        self.assertEqual(
            member.request.position.identity, ("NC_012920.1", 303, "", "C")
        )
        # The same allele from a fresh source pair: a new target, the same variant.
        self.assertEqual(
            [
                (
                    item.target.contig,
                    item.target.one_based_position,
                    item.target.ref,
                    item.target.alt,
                    item.previous_work_item_id,
                )
                for item in revised.work_items
                if item.target.alt == "C"
            ],
            [("NC_012920.1", 303, "", "C", retried.work_item_id)],
        )


class IntentQuestionRoutingTests(unittest.TestCase):
    """Only the failed turn that stops decides the route (review r1, #447)."""

    def test_a_later_different_output_fault_keeps_its_own_review(self):
        # Review r1 S1: the stop's own refusal decides the route. An intent
        # refusal followed by a schema refusal is the shared Agent-work Review.
        investigation_input, assertion = _merged_request()
        scientific = SequentialAdaptiveModel()
        outputs = 0

        async def respond(messages, info):
            nonlocal outputs
            response = await scientific.respond(messages, info)
            call = response.parts[0]
            if call.tool_name == "compile_proposal":
                outputs += 1
                # The output whose failed turn is the default's fourth in a row
                # is a schema fault.
                if outputs == 4:
                    arguments = call.args_as_dict()
                    del arguments["outcome"]
                    call.args = json.dumps(arguments)
            return response

        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                SourceResponsesOnly(
                    refsnp_bodies={
                        "rs60037639": RS60037639,
                        "rs4149056": _merged_from_request_body(),
                    }
                ),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(
            [
                event["payload"]["errors"][:1]
                for event in events
                if event["event_type"] == "AgentOutputValidated"
            ],
            [["retrieved_target_not_requested"]] * 3 + [["outcome:missing"]],
        )
        self.assertEqual(result.status, "agent_work_review_pending")
        self.assertFalse(
            [
                event
                for event in events
                if event["event_type"] == "InvestigationReviewRequested"
                and event["payload"].get("question") is not None
            ]
        )

    def test_text_endings_after_one_intent_refusal_keep_their_own_review(self):
        # Review r2: one intent refusal is the first failed turn; the text endings
        # that follow are the next ones and stop the Attempt, and that stop is
        # theirs.
        from pydantic_ai.messages import TextPart

        investigation_input, assertion = _merged_request()
        scientific = SequentialAdaptiveModel()
        refused = False

        async def respond(messages, info):
            nonlocal refused
            if refused:
                return ModelResponse(parts=[TextPart("Still considering.")])
            response = await scientific.respond(messages, info)
            refused = response.parts[0].tool_name == "compile_proposal"
            return response

        with tempfile.TemporaryDirectory() as directory:
            result, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                SourceResponsesOnly(
                    refsnp_bodies={
                        "rs60037639": RS60037639,
                        "rs4149056": _merged_from_request_body(),
                    }
                ),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        self.assertEqual(
            [
                event["payload"]["errors"][:1]
                for event in events
                if event["event_type"] == "AgentOutputValidated"
            ],
            [["retrieved_target_not_requested"]],
        )
        self.assertEqual(result.status, "agent_work_review_pending")
        self.assertEqual(result.attempts[-1].stop_reason, "failed_turns")
        self.assertFalse(
            [
                event
                for event in events
                if event["event_type"] == "InvestigationReviewRequested"
                and event["payload"].get("question") is not None
            ]
        )

    def test_the_question_must_cite_the_refusal_that_spent_the_allowance(self):
        from bio_neb.investigation.investigation import project_investigation

        investigation_input, assertion = _merged_request()
        with tempfile.TemporaryDirectory() as directory:
            _asked, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                SourceResponsesOnly(
                    refsnp_bodies={
                        "rs60037639": RS60037639,
                        "rs4149056": _merged_from_request_body(),
                    }
                ),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)

        first_refusal = next(
            event["event_id"]
            for event in events
            if event["event_type"] == "AgentOutputValidated"
        )
        stale = copy.deepcopy(events)
        question = next(
            event
            for event in stale
            if event["event_type"] == "InvestigationReviewRequested"
        )
        question["payload"]["question"]["refused_output_event_id"] = first_refusal
        with self.assertRaisesRegex(ValueError, "changed its refused output"):
            project_investigation(stale)

    def test_an_answered_question_restarts_only_its_own_members_repair(self):
        # Review r1 S2: member A has a refused output and is held for model
        # availability when member B asks; answering B leaves A's lineage.

        from bio_neb.investigation.investigation import (
            ClarifyInvestigationResponse,
            InvestigationRequestItem,
        )
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from tests.sequential_investigation_support import (
            SequentialSourceResponses,
            _current_member_messages,
        )

        scientific = SequentialAdaptiveModel()
        state = {"a_refused": False, "a_held": False, "b_asked": False}

        async def respond(messages, info):
            request = _current_member_messages(messages)[0]
            if request["rsid"] == "rs429358" and state["a_refused"]:
                # After its refused output the member only writes text, so its
                # failed turns hold it for the Agent work Review: a member-scoped
                # hold, where an open model endpoint Review would hold every
                # member.
                state["a_held"] = True
                return ModelResponse(parts=[TextPart("Still thinking.")])
            if request["rsid"] == "rs7412" and not state["b_asked"]:
                state["b_asked"] = True
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "request_investigation_information",
                            {
                                "question": "Which allele?",
                                "missing_information": "The allele is unclear.",
                                "asks_about": "target",
                            },
                            tool_call_id="b-question",
                        )
                    ]
                )
            response = await scientific.respond(messages, info)
            call = response.parts[0]
            if (
                request["rsid"] == "rs429358"
                and call.tool_name == "compile_proposal"
                and not state["a_refused"]
            ):
                state["a_refused"] = True
                arguments = call.args_as_dict()
                del arguments["outcome"]
                call.args = json.dumps(arguments)
            return response

        investigation_input, _ = input_and_assertion()
        assertion = UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=tuple(
                InvestigationRequestItem(request_text=f"Investigate {rsid}.", rsid=rsid)
                for rsid in ("rs429358", "rs7412")
            ),
            predecessor_ids=(investigation_input.input_id,),
        )
        model = ScriptedModelEndpoint(respond)
        with tempfile.TemporaryDirectory() as directory:
            sources = SequentialSourceResponses()
            asked, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
            )
            question = asked.pending_request.question
            respond_investigation_review(
                RespondInvestigationReviewCommand(
                    journey_file=str(journey.path),
                    request_event_id=asked.pending_request.request_event_id,
                    response=ClarifyInvestigationResponse(
                        action="clarify",
                        collection_member_id=question.collection_member_id,
                        revision_number=1,
                        instruction="The rs7412 target is correct.",
                        asserted_by="researcher:owner",
                    ),
                )
            )
            answered = continue_investigation(
                ContinueInvestigationCommand(journey_file=str(journey.path)),
                environ={"LOCAL_MODEL_KEY": "offline"},
                model_factory=model,
                source_adapter_factory=lambda: sources,
            )

        # Each member is its own Attempt: the held member's refused output stays
        # in its own Attempt while the asking member's answer restarts only the
        # asker.
        (held_member,) = [
            lineage
            for attempt in asked.attempts
            for lineage in attempt.structured_output_lineages
        ]
        self.assertEqual(
            [item.status for item in held_member.validation_attempts], ["invalid"]
        )
        kept = {
            lineage.collection_member_id: lineage
            for lineage in answered.collection_lineages
        }
        self.assertEqual(
            kept[held_member.collection_member_id].validation_attempts,
            held_member.validation_attempts,
        )
        self.assertIsNotNone(kept[question.collection_member_id].accepted_output)


def _member_collection(count: int):
    """A collection of ``count`` resolvable rsIDs and sources that answer each."""
    from bio_neb.investigation.investigation import InvestigationRequestItem
    from tests.sequential_investigation_support import SequentialSourceResponses

    rsids = tuple(f"rs{1000 + index}" for index in range(count))

    class Sources(SequentialSourceResponses):
        positions = {
            **SequentialSourceResponses.positions,
            **{rsid: 15000 + index for index, rsid in enumerate(rsids)},
        }

    investigation_input, _ = input_and_assertion()
    assertion = UserAssertion.create_collection(
        asserted_by="researcher:owner",
        requests=tuple(
            InvestigationRequestItem(
                request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
            )
            for rsid in rsids
        ),
        predecessor_ids=(investigation_input.input_id,),
    )
    return investigation_input, assertion, Sources


class _Overlapping:
    """The adaptive planner, holding each request until ``expected`` overlap."""

    def __init__(self, expected: int, respond=None) -> None:
        self.scientific = SequentialAdaptiveModel()
        self.respond_with = respond or self.scientific.respond
        self.expected, self.in_flight, self.peak = expected, 0, 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        import asyncio

        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            for _ in range(100):
                if self.peak >= self.expected:
                    break
                await asyncio.sleep(0.01)
            return await self.respond_with(messages, info)
        finally:
            self.in_flight -= 1


class IndependentMemberTests(unittest.TestCase):
    """Each collection member is its own Attempt, run together and collected (#105)."""

    def test_members_run_concurrently_under_the_bound_and_close_one_proposal(self):
        from bio_neb.application.queries import read_investigation
        from bio_neb.journey import Journey

        investigation_input, assertion, Sources = _member_collection(6)
        model = _Overlapping(expected=3)
        with tempfile.TemporaryDirectory() as directory:
            sources = Sources()
            planned, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
                investigation_concurrency={"planner_members": 3, "table_chunks": 1},
            )
            events = read_evidence(journey)
            reopened = read_investigation(Journey.open(journey.path)).investigation

        self.assertEqual(model.peak, 3)
        self.assertEqual(planned.status, "review_pending")
        # One Attempt per member, each bound to it, with its own start scope.
        members = [attempt.collection_member_id for attempt in planned.attempts]
        self.assertEqual(len(members), 6)
        self.assertEqual(len(set(members)), 6)
        starts = [e for e in events if e["event_type"] == "AttemptStarted"]
        self.assertEqual(
            sorted(e["payload"]["member_id"] for e in starts), sorted(members)
        )
        # Members interleave, yet one proposal closes over all six, in order.
        (proposal,) = planned.proposals
        self.assertEqual(
            [item.target.one_based_position for item in proposal.work_items],
            [15000 + index for index in range(6)],
        )
        recorded = [
            e for e in events if e["event_type"] == "InvestigationProposalRecorded"
        ]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(reopened, planned)

    def test_a_collection_past_one_attempts_allowance_has_no_member_cap(self):
        investigation_input, assertion, Sources = _member_collection(70)
        model = SequentialAdaptiveModel()
        with tempfile.TemporaryDirectory() as directory:
            planned, journey = run_attempt(
                Path(directory),
                model,
                Sources(),
                investigation_input=investigation_input,
                assertion=assertion,
                investigation_concurrency={"planner_members": 8, "table_chunks": 1},
            )
            requests = [
                e
                for e in read_evidence(journey)
                if e["event_type"] == "ModelRequestRecorded"
            ]

        # 70 members, three requests each: past one Attempt's 200-request
        # allowance, with no member cap and no effort stop.
        self.assertEqual(len(requests), 210)
        self.assertEqual(planned.status, "review_pending")
        self.assertEqual(len(planned.proposals[-1].work_items), 70)
        self.assertTrue(
            all(
                len(attempt.model_request_event_ids) == 3
                for attempt in planned.attempts
            )
        )

    def test_a_held_member_never_spends_the_next_members_allowance(self):
        # Errors 18935528: after one member's operational Review, the next
        # member's start met the shared allowance and the effort stop lost its
        # transcript. Each member now has its own Attempt and allowance.
        from pydantic_ai.exceptions import ModelHTTPError

        from tests.sequential_investigation_support import _current_member_messages

        investigation_input, assertion, Sources = _member_collection(3)
        scientific = SequentialAdaptiveModel()

        async def respond(messages, info):
            request = _current_member_messages(messages)[0]
            if request["rsid"] == "rs1000":
                raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})
            return await scientific.respond(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            planned, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                Sources(),
                investigation_input=investigation_input,
                assertion=assertion,
                invocation_overrides={"model_requests": 1},
            )
            events = read_evidence(journey)

        from bio_neb.agentic_journey.effort import project_effort_review
        from bio_neb.agentic_journey.model_availability import (
            project_model_availability_review,
        )

        self.assertEqual(len(planned.attempts), 3)
        # The first member holds for its availability Review. While it is open
        # each sibling joins it at its next call and spends nothing of its own
        # allowance; every Attempt has its terminal.
        first, *joined = project_model_availability_review(events).requests
        self.assertEqual(first.attempt_id, planned.attempts[0].attempt_id)
        self.assertEqual(
            sorted(r.attempt_id for r in joined),
            sorted(a.attempt_id for a in planned.attempts[1:]),
        )
        self.assertTrue(all(r.joined_before_sending for r in joined))
        self.assertEqual(project_effort_review(events).requests, ())
        self.assertNotIn(
            "coordinator_unhandled_error", json.dumps([e["payload"] for e in events])
        )

    def test_eight_settle_while_two_ask_and_each_answer_continues_only_its_member(
        self,
    ):
        from bio_neb.application.queries import read_investigation
        from bio_neb.investigation.investigation import ClarifyInvestigationResponse
        from bio_neb.investigation.investigation_commands import (
            ContinueInvestigationCommand,
            RespondInvestigationReviewCommand,
            continue_investigation,
            respond_investigation_review,
        )
        from bio_neb.journey import Journey
        from tests.sequential_investigation_support import _current_member_messages

        investigation_input, assertion, Sources = _member_collection(10)
        asking = {"rs1002", "rs1006"}
        asked: set[str] = set()
        scientific = SequentialAdaptiveModel()

        async def respond(messages, info):
            rsid = _current_member_messages(messages)[0]["rsid"]
            if rsid in asking and rsid not in asked:
                asked.add(rsid)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "request_investigation_information",
                            {
                                "question": f"Which allele of {rsid}?",
                                "missing_information": "The allele is unclear.",
                                "asks_about": "target",
                            },
                            tool_call_id=f"ask-{rsid}",
                        )
                    ]
                )
            return await scientific.respond(messages, info)

        model = ScriptedModelEndpoint(respond)
        with tempfile.TemporaryDirectory() as directory:
            sources = Sources()
            held, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model),
                sources,
                investigation_input=investigation_input,
                assertion=assertion,
                investigation_concurrency={"planner_members": 4, "table_chunks": 1},
            )
            reopened = read_investigation(Journey.open(journey.path)).investigation
            questions = sorted(
                (r for r in held.review_requests if r.pending),
                key=lambda r: r.question.request.rsid,
            )

            # Eight members settle beside two separate, exact questions; nothing
            # is proposed yet and the eight results are already visible.
            self.assertEqual(held.status, "review_pending")
            self.assertEqual(held.proposals, ())
            self.assertEqual(
                [q.question.request.rsid for q in questions], sorted(asking)
            )
            self.assertEqual(
                len({q.question.collection_member_id for q in questions}), 2
            )
            settled = [
                lineage
                for lineage in held.collection_lineages
                if lineage.accepted_output is not None
            ]
            self.assertEqual(len(settled), 8)
            self.assertEqual(len(sources.operations), 16)
            self.assertEqual(reopened, held)

            def answer(question):
                respond_investigation_review(
                    RespondInvestigationReviewCommand(
                        journey_file=str(journey.path),
                        request_event_id=question.request_event_id,
                        response=ClarifyInvestigationResponse(
                            action="clarify",
                            collection_member_id=question.question.collection_member_id,
                            revision_number=1,
                            instruction="The listed allele is correct.",
                            asserted_by="researcher:owner",
                        ),
                    )
                )
                return continue_investigation(
                    ContinueInvestigationCommand(journey_file=str(journey.path)),
                    environ={"LOCAL_MODEL_KEY": "offline"},
                    model_factory=model,
                    source_adapter_factory=lambda: sources,
                )

            attempts_before = len(held.attempts)
            first = answer(questions[0])
            # Only the answered member is planned again; its siblings' work and
            # sources are not repeated, and the other question still waits.
            self.assertEqual(len(first.attempts), attempts_before + 1)
            self.assertEqual(
                first.attempts[-1].collection_member_id,
                questions[0].question.collection_member_id,
            )
            self.assertEqual(len(sources.operations), 18)
            self.assertEqual(first.proposals, ())
            self.assertEqual(
                [r.request_event_id for r in first.review_requests if r.pending],
                [questions[1].request_event_id],
            )
            second = answer(questions[1])
            self.assertEqual(len(sources.operations), 20)
            (proposal,) = second.proposals
            self.assertEqual(len(proposal.work_items), 10)
            self.assertEqual(second.status, "review_pending")
            self.assertEqual(
                read_investigation(Journey.open(journey.path)).investigation, second
            )

    def test_interleaved_members_refuse_cross_member_orphaned_and_stale_results(
        self,
    ):
        from bio_neb.investigation.investigation import project_investigation

        investigation_input, assertion, Sources = _member_collection(4)
        model = _Overlapping(expected=4)
        with tempfile.TemporaryDirectory() as directory:
            planned, journey = run_attempt(
                Path(directory),
                SimpleNamespace(model=model.model),
                Sources(),
                investigation_input=investigation_input,
                assertion=assertion,
                investigation_concurrency={"planner_members": 4, "table_chunks": 1},
            )
            events = read_evidence(journey)

        self.assertEqual(model.peak, 4)
        self.assertEqual(project_investigation(events), planned)
        attempts = [a.attempt_id for a in planned.attempts]
        # The members' facts interleave in the Journey.
        owners = [
            e["payload"]["attempt_id"]
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
        ]
        self.assertNotEqual(owners, sorted(owners, key=attempts.index))

        def refused(change, reason):
            hostile = copy.deepcopy(events)
            change(hostile)
            with self.assertRaisesRegex(ValueError, reason):
                project_investigation(hostile)

        def result_of(hostile, attempt_id):
            return next(
                e
                for e in hostile
                if e["event_type"] == "RefSnpToolResultRecorded"
                and e["payload"]["attempt_id"] == attempt_id
            )

        # A sibling's source result claimed by another member's Attempt.
        refused(
            lambda h: result_of(h, attempts[0])["payload"].update(
                attempt_id=attempts[1]
            ),
            "source Tool terminal changed its stage result",
        )
        # A result whose Tool start is gone (orphaned).
        refused(
            lambda h: h.remove(
                next(
                    e
                    for e in h
                    if e["event_type"] == "ObservedToolExecutionStarted"
                    and e["payload"]["attempt_id"] == attempts[2]
                )
            ),
            "source scope lacks its exact unique Tool execution",
        )
        # A member Attempt bound to a revision the collection never had.
        refused(
            lambda h: next(
                e
                for e in h
                if e["event_type"] == "InvestigationAttemptBound"
                and e["payload"]["attempt_id"] == attempts[3]
            )["payload"].update(revision_number=2),
            "Investigation Attempt changed its revision basis",
        )

    def test_a_members_recorded_scope_must_match_its_own_work(self):
        # Standards r1 S1: the start and binding scopes are checked against the
        # member's actual requests, model requests, results and outputs.
        from bio_neb.investigation.investigation import project_investigation

        investigation_input, assertion, Sources = _member_collection(2)
        with tempfile.TemporaryDirectory() as directory:
            planned, journey = run_attempt(
                Path(directory),
                SequentialAdaptiveModel(),
                Sources(),
                investigation_input=investigation_input,
                assertion=assertion,
            )
            events = read_evidence(journey)
        first, second = (a.attempt_id for a in planned.attempts)
        sibling = planned.attempts[1].collection_member_id

        def scoped(event_type, attempt_id, field, value):
            hostile = copy.deepcopy(events)
            next(
                e
                for e in hostile
                if e["event_type"] == event_type
                and e["payload"]["attempt_id"] == attempt_id
            )["payload"][field] = value
            return hostile

        for name, hostile in (
            (
                "start names the sibling",
                scoped("AttemptStarted", first, "member_id", sibling),
            ),
            (
                "binding names the sibling",
                scoped(
                    "InvestigationAttemptBound", first, "collection_member_id", sibling
                ),
            ),
            (
                "binding names an invented member",
                scoped(
                    "InvestigationAttemptBound",
                    second,
                    "collection_member_id",
                    "request-member:" + "0" * 64,
                ),
            ),
        ):
            with (
                self.subTest(name),
                self.assertRaisesRegex(
                    ValueError, "Investigation Attempt changed its member scope"
                ),
            ):
                project_investigation(hostile)


# The member's job as a probe tested it (`JOB_OBJECTIVE` and `ASK_INTENT_DESCRIPTION`).
# Nemotron acts on its user message, so a member reads its job there, not the researcher's
# question.
MEMBER_JOB = (
    "Propose this member's variant locus for the researcher's review. Look up the requested "
    "rsID with the lookup Tool of the source that suits the Run's organism, check its "
    "assembly report, then return compile_proposal. You "
    "do not answer whether the sample carries the variant: the Run reads the sample's VCF "
    "itself after the researcher approves your proposal, so never try to read the genotype "
    "and never ask the researcher for it."
)
# A member that states a position starts at locate_variant, not the rsID lookup.
POSITION_MEMBER_JOB = (
    "Propose this member's variant locus for the researcher's review. Locate the stated "
    "position with locate_variant, check its assembly report, then return compile_proposal. "
    "You do not answer whether the sample carries the variant: the Run reads the sample's "
    "VCF itself after the researcher approves your proposal, so never try to read the "
    "genotype and never ask the researcher for it."
)
# The ask's examples: a plain ask, and one citing the lookup that placed the target
# elsewhere.
ASK_EXAMPLES = (
    'Example: request_investigation_information(question="Which allele?", '
    'missing_information="The request does not specify an allele.", asks_about="target")'
    '\nExample: request_investigation_information(question="<which target the '
    'researcher means>", missing_information="<what the source\'s answer leaves '
    'open>", asks_about="target", refsnp_event_id="<the cite of the lookup that placed '
    'it elsewhere>")'
)
INTENT_ONLY_ASK = (
    "Ask the researcher about their intent only: which variant, allele, position or assembly "
    "they mean, when the request and retained answers cannot tell you, naming in asks_about "
    "what you ask (target, version or intent). When a lookup result prompts the question "
    "(its source places the target on another organism or assembly), cite it in "
    "refsnp_event_id: code copies its statement. Never ask for "
    "evidence, a genotype, the sample's VCF or approval of your proposal: the Run reads the "
    "VCF after the researcher approves the proposal. Answers remain User assertions, never "
    "scientific evidence or approval.\n\n" + ASK_EXAMPLES
)
NEMOTRON = "nvidia/Nemotron-3-Ultra-550b-a55b"
QWEN = "Qwen/Qwen3.5-397B-A17B"


class MemberJobTests(unittest.TestCase):
    """What a member's first request shows its model: Nemotron's states the member's
    job and an intent-only ask Tool (#401); Qwen's bytes stay as they were (D5)."""

    def _first_request(self, model_name, **attempt):
        with tempfile.TemporaryDirectory() as directory:
            model = SimpleNamespace(
                model=ScriptedModelEndpoint(SequentialAdaptiveModel().respond)
            )
            _, journey = run_attempt(
                Path(directory),
                model,
                attempt.pop("sources", SourceResponsesOnly()),
                model_name=model_name,
                **attempt,
            )
            request = next(
                e["payload"]["compatible_adapter_evidence"]["request"]
                for e in read_evidence(journey)
                if e["event_type"] == "ModelRequestRecorded"
            )
        # The member message comes first; Nemotron's injected notices follow it.
        prompt = next(
            part.content
            for message in ModelMessagesTypeAdapter.validate_python(request["messages"])
            for part in message.parts
            if part.part_kind == "user-prompt"
        )
        descriptions = {
            tool["name"]: tool["description"] for tool in request["function_tools"]
        }
        return json.loads(prompt), descriptions

    def test_a_nemotron_member_reads_its_job_and_an_intent_only_ask_tool(self):
        member, descriptions = self._first_request(NEMOTRON)

        self.assertEqual(member["assignment"]["objective"], MEMBER_JOB)
        self.assertEqual(
            descriptions["request_investigation_information"], INTENT_ONLY_ASK
        )

    def test_a_nemotron_member_stating_a_position_reads_the_locate_variant_job(self):
        from bio_neb.investigation.investigation_request import (
            InvestigationRequestItem,
            StatedPosition,
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

        member, _ = self._first_request(
            NEMOTRON, investigation_input=investigation_input, assertion=assertion
        )

        self.assertEqual(member["assignment"]["objective"], POSITION_MEMBER_JOB)

    def test_a_qwen_member_keeps_its_recorded_objective_and_ask_tool(self):
        member, descriptions = self._first_request(QWEN)

        self.assertEqual(
            member["assignment"]["objective"],
            "Resolve the exact scientific request from admitted source evidence",
        )
        self.assertEqual(
            descriptions["request_investigation_information"],
            "Ask only for missing researcher intent for this exact request and "
            "revision, naming in asks_about what you ask (target, version or intent), "
            "and in refsnp_event_id the lookup result that prompts it, if one does (its "
            "source places the target on another organism or assembly); answers remain "
            "User assertions, never scientific evidence or approval.\n\n"
            + ASK_EXAMPLES,
        )


class CheckedBeforeAppendTests(unittest.TestCase):
    def test_a_lookup_result_its_own_reopen_check_refuses_is_never_written(self):
        """#476 (owner, 2026-10-28, C8: errors defined out of existence, the
        save-time check the last guard; ADR 0032: no unchecked append): a lookup
        result whose own reopen check fails is not appended, and the hold names the
        fact type, the check and the call. Bounded writer faults (Spec r1 SP3's
        and r2 SP4's probes): the real writer records the result under a foreign
        member, or naming a source no retained event has."""
        from bio_neb.agentic_journey.lifecycle import Activity

        record = Activity.record
        faults = {
            "foreign member": (
                {"collection_member_id": "foreign"},
                "source observation changed its member, query or call identity",
            ),
            # Spec r2 SP4: a source no retained event has.
            "unknown source": (
                {"source_event_id": "00000000-0000-0000-0000-000000000099"},
                "typed Tool result changed its exact source lineage",
            ),
        }
        for label, (update, reason) in faults.items():
            with self.subTest(label):
                self._refused(record, update, reason)

    def _refused(self, record, update, reason):
        from unittest.mock import patch

        from bio_neb.agentic_journey.lifecycle import Activity
        from bio_neb.failure import Invariant, RecordedFactRefused
        from bio_neb.journey import Journey

        def writer_fault(activity, fact):
            if fact.event_type == "RefSnpToolResultRecorded":
                fact = fact.model_copy(update=update)
            return record(activity, fact)

        async def respond(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_refsnp", {"rsid": "rs429358"}, tool_call_id="lookup"
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(Activity, "record", writer_fault),
                self.assertRaises(RecordedFactRefused) as refused,
            ):
                run_attempt(
                    Path(directory),
                    SimpleNamespace(model=ScriptedModelEndpoint(respond)),
                    SourceResponsesOnly(),
                    invocation_overrides={"model_requests": 2},
                )
            (journey_file,) = Path(directory).glob("*/journey.jsonl")
            events = read_evidence(Journey.open(journey_file))
        self.assertIsInstance(refused.exception, Invariant)
        self.assertEqual(refused.exception.invariant_id, "recorded_fact_refused")
        self.assertEqual(
            {
                key: refused.exception.evidence[key]
                for key in ("event_type", "check", "reason", "tool_call_id")
            },
            {
                "event_type": "RefSnpToolResultRecorded",
                "check": "source_observation_reopens",
                "reason": reason,
                "tool_call_id": "lookup",
            },
        )
        self.assertNotIn("RefSnpToolResultRecorded", {e["event_type"] for e in events})
