"""Cost guard: one prerequisite build re-reads the Journey history a bounded number of times.

The ceilings pin full-history work in one default ``approved_input`` build.
Before #196 the same build measured 3,751 ``Journey.read``, 1,620 ``evidence_revision``,
1,810 ``read_passes`` and 6 ``_load_events`` calls (#194), and every revision call
re-read the whole history. #196 pinned 402 ``Journey.read`` and 190 ``read_passes``;
it still deep-copied the whole planning timeline 626 times, once per offered Tool.
A ceiling may only shrink, and equals the measured count.
"""

import dataclasses
import json
import shutil
import sys
import tempfile
import unittest
import uuid
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

import bio_neb.agentic_journey.assignment_context as assignment_context
import bio_neb.application.planning_read as planning_read
import bio_neb.application.queries as queries
import bio_neb.investigation.investigation as investigation
import bio_neb.recording as recording
from bio_neb.agentic_journey.assignment_context import search_journey_events
from bio_neb.application.queries import ObserveRun, observe
from bio_neb.investigation.investigation_commands import BeginInvestigationCommand
from bio_neb.journey import Journey, ReadOnlyDict, ReadOnlyList, canonical_json
from bio_neb.recording import read_evidence
from tests.item_state_runs import (
    FINALIZATION_PENDING,
    INVESTIGATION_REVIEW_PENDING,
    item_state_dependencies,
    native_answer,
)
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.observation_input_fakes import RunOutputStore
from tests.parabricks_operation_fakes import approved_input
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SourceResponsesOnly,
    approved_collection_observation,
    begin_in_pass,
    collection_request,
)
from tests.terminal_journey_fakes import forked_child

# Measured on a fixed source build. The revision is answered from held state, so its calls
# read no history; the six loads are independent opens outside any interval. Planning
# readers share the owner's read-only events, writes fold families from held state and
# closed Pass journals are validated once per owner. The coordinator asks the preparation
# question without a Tool call, and the plan records no output requirement. The Attempt
# runner's hold check and request count and the loop's admission session read the owner's
# shared view; a Tool result that binds no artifact reads no history for its sources. In
# evidence_revision: record_answered_targets is one more definition each model request
# prepares, and each preparation reads the stage (two revisions) on this build's 14
# requests (the answer check itself reads none); the recipe selection's offered organism
# records (its membership enum) and its quote check, the reference choice's membership
# check (a row of the page it cites) and its quote check (the researcher's words), and the
# supplied choice's statement enum and the sample's and the supplied accession's quote
# checks read the planning evidence. No supplied accession is read from an Execution
# Review's words. read_passes measured 163 (one shared evidence view per checked Tool
# fact). Each nonzero ceiling is twice its measure: they catch a runaway; the zero rows
# stay structural.
_CEILINGS = {
    "Journey.read": 112,
    "Journey.read within evidence_revision": 0,
    "Journey.project within evidence_revision": 0,
    "evidence_revision": 3_200,
    "read_passes": 326,
    "read_passes within evidence_revision": 0,
    "Journey._load_events": 12,
    "planning timeline copy": 0,
}

# One observe of a forked child after its approved Investigation review. It
# once interpreted one snapshot twice and deep-copied 292 shared events. The
# last copy was the page of a search receipt checked again; each receipt is now
# checked once per owner.
_FORKED_CHILD_OBSERVE_CEILINGS = {
    "repeated snapshot interpretation": 0,
    "shared event deepcopy": 0,
}


def _counted(counts, name, original, *, inside=None):
    def call(*args, **kwargs):
        counts[name] += 1
        if inside is not None and counts[inside]:
            counts[f"{name} within {inside.removeprefix('active ')}"] += 1
        return original(*args, **kwargs)

    return call


def _timeline_copies(counts, original):
    # A whole timeline list copied by a planning reader (once per offered
    # Tool), or any event copied while the owner builds its context.
    def call(value, *args, **kwargs):
        if isinstance(value, list) or (
            counts["active owned planning context"]
            and isinstance(value, Mapping)
            and "event_id" in value
        ):
            counts["planning timeline copy"] += 1
        return original(value, *args, **kwargs)

    return call


def _owned_context_scope(counts, original):
    def call(*args, **kwargs):
        counts["active owned planning context"] += 1
        try:
            return original(*args, **kwargs)
        finally:
            counts["active owned planning context"] -= 1

    return call


def _revision_scope(counts, original):
    def call(*args, **kwargs):
        counts["evidence_revision"] += 1
        counts["active evidence_revision"] += 1
        try:
            return original(*args, **kwargs)
        finally:
            counts["active evidence_revision"] -= 1

    return call


class PrerequisiteBuildCostTests(unittest.TestCase):
    def test_default_approved_input_build_bounds_full_history_loads(self):
        counts = Counter()
        with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
            stack.enter_context(
                patch.object(
                    Journey,
                    "read",
                    autospec=True,
                    side_effect=_counted(
                        counts,
                        "Journey.read",
                        Journey.read,
                        inside="active evidence_revision",
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    Journey,
                    "project",
                    autospec=True,
                    side_effect=_counted(
                        counts,
                        "Journey.project",
                        Journey.project,
                        inside="active evidence_revision",
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    Journey,
                    "_load_events",
                    side_effect=_counted(
                        counts, "Journey._load_events", Journey._load_events
                    ),
                )
            )
            evidence_revision = _revision_scope(counts, recording.evidence_revision)
            read_passes = _counted(
                counts,
                "read_passes",
                recording.read_passes,
                inside="active evidence_revision",
            )
            stack.enter_context(
                patch.object(recording, "evidence_revision", evidence_revision)
            )
            stack.enter_context(patch.object(recording, "read_passes", read_passes))
            stack.enter_context(
                patch.object(
                    planning_read,
                    "read_only_json",
                    _timeline_copies(counts, planning_read.read_only_json),
                )
            )
            stack.enter_context(
                patch.object(
                    planning_read,
                    "_read_owned_planning_context",
                    _owned_context_scope(
                        counts, planning_read._read_owned_planning_context
                    ),
                )
            )
            stack.enter_context(patch.object(queries, "read_passes", read_passes))

            approved_input(Path(temporary) / "case")

        for name, ceiling in _CEILINGS.items():
            with self.subTest(name=name):
                # A ceiling at twice its measure, not a pin.
                self.assertLessEqual(counts[name], ceiling, dict(counts))


def _mutable_values(value, path="read", seen=None):
    """Paths of mutable containers or unfrozen models reachable from ``value``."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, BaseModel):
        found = [] if value.model_config.get("frozen") else [path]
        return found + [
            item
            for name in type(value).model_fields
            for item in _mutable_values(getattr(value, name), f"{path}.{name}", seen)
        ]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        found = [] if value.__dataclass_params__.frozen else [path]
        return found + [
            item
            for field in dataclasses.fields(value)
            for item in _mutable_values(
                getattr(value, field.name), f"{path}.{field.name}", seen
            )
        ]
    if isinstance(value, (ReadOnlyDict, ReadOnlyList, tuple, frozenset)):
        items = value.values() if isinstance(value, ReadOnlyDict) else value
        return [item for child in items for item in _mutable_values(child, path, seen)]
    if isinstance(value, (dict, list, set, bytearray)):
        return [path]
    return []


def _event_copies(counts, original):
    # A shared Journey event deep-copied whole; nested copies are not counted.
    def call(self, memo):
        if "event_id" in self:
            counts["shared event deepcopy"] += 1
        return original(self, memo)

    return call


class ForkedChildReadCostTests(unittest.TestCase):
    """A Fork child's reads share its checked planning and interpret each snapshot once.

    Since #211 a child reads its inherited parent prefix and own facts as one
    timeline (ADR 0031), so every copy or repeated interpretation of that
    timeline is paid per read. #278 measured the 8 fork-consumer modules at
    about twice their pre-#211 seconds and pinned these counts.
    """

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        _, journey, _, _, _ = approved_collection_observation(
            Path(directory.name).resolve(), cardinality=1, forked=True
        )
        cls.journey_file = journey.path

    def test_planning_reads_share_one_read_that_refuses_mutation(self):
        journey = Journey.open(self.journey_file)
        owned = planning_read.read_planning(journey)
        interpreted = queries.read_planning(journey)

        # Identity, not equality: a per-call copy is the cost being removed.
        self.assertTrue(planning_read.read_planning(journey) is owned, "copied")
        self.assertTrue(queries.read_planning(journey) is interpreted, "copied")
        self.assertEqual(owned, interpreted)
        self.assertEqual(_mutable_values(owned), [])
        replay = planning_read.read_planning_events(
            planning_read.planning_evidence(journey)
        )
        self.assertEqual(replay, owned)
        self.assertEqual(_mutable_values(replay), [])
        schema = owned.hitl.requests[0].legal_actions[0].response_schema
        fact = owned.execution.facts[0]
        refused = (
            lambda: schema.__setitem__("type", "array"),
            lambda: fact.result.pop(next(iter(fact.result))),
            lambda: fact.tool_input.update(changed=True),
        )
        for mutate in refused:
            with self.subTest(mutate=mutate), self.assertRaises(TypeError) as raised:
                mutate()
            self.assertIn("read-only", str(raised.exception))

    def test_search_copies_only_the_returned_page(self):
        journey = Journey.open(self.journey_file)
        events = journey.events_view()
        counts = Counter()
        with patch.object(
            ReadOnlyDict,
            "__deepcopy__",
            _event_copies(counts, ReadOnlyDict.__deepcopy__),
        ):
            page = search_journey_events(events, journey.journey_id, limit=3)

        self.assertGreater(len(events), 100)
        self.assertEqual(len(page["events"]), 3)
        self.assertEqual(counts["shared event deepcopy"], 3)
        # The page is the caller's own copy; the shared evidence is untouched.
        page["events"][0]["payload"] = {}
        self.assertNotEqual(journey.events_view()[-3]["payload"], {})

    def test_one_child_observe_interprets_each_snapshot_once(self):
        counts = Counter()
        snapshots = set()
        original = queries._InterpretationContext.__init__

        def interpretation(self, planning_context, **kwargs):
            if id(planning_context) in snapshots:
                counts["repeated snapshot interpretation"] += 1
            snapshots.add(id(planning_context))
            original(self, planning_context, **kwargs)

        with (
            patch.object(queries._InterpretationContext, "__init__", interpretation),
            patch.object(
                ReadOnlyDict,
                "__deepcopy__",
                _event_copies(counts, ReadOnlyDict.__deepcopy__),
            ),
        ):
            observation = observe(ObserveRun(journey_file=str(self.journey_file)))

        self.assertEqual(observation.run_provenance, "causal_child_or_fork")
        for name, ceiling in _FORKED_CHILD_OBSERVE_CEILINGS.items():
            with self.subTest(name=name):
                self.assertEqual(
                    counts[name], ceiling, f"pin the {name} ceiling to {counts[name]}"
                )


def _searching_planner():
    """The scripted planner, opening every member's Attempt with one history search."""
    scientific = SequentialAdaptiveModel()

    async def respond(messages, info):
        offered = {tool.name for tool in info.function_tools}
        searched = any(
            isinstance(part, ToolReturnPart) and part.tool_name == "search_journey"
            for message in messages
            for part in message.parts
        )
        if "fetch_refsnp" in offered and not searched:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_journey",
                        {"kinds": ["InvestigationStarted"]},
                        tool_call_id=f"search-{len(messages)}",
                    )
                ]
            )
        return await scientific.respond(messages, info)

    return ScriptedModelEndpoint(respond)


def _counted_everywhere(stack, original, record):
    """Count ``original`` at every module that bound it, eagerly or lazily."""

    def call(*args, **kwargs):
        record(*args, **kwargs)
        return original(*args, **kwargs)

    for module in list(sys.modules.values()):
        if getattr(module, original.__name__, None) is original:
            stack.enter_context(patch.object(module, original.__name__, call))


class InvestigationRebuildCostTests(unittest.TestCase):
    """One rebuild folds each Investigation prefix once (ADR 0032, #400).

    #389's 8.8 h Run spent 86 % of its time re-projecting the whole
    Investigation once per Attempt and re-running every search receipt in each
    of those projections.
    """

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        document = collection_request(3)
        journey = forked_child(root, investigation_request=document)
        request_file = root / "investigation-request.json"
        request_file.write_text(canonical_json(document), encoding="utf-8")
        pending = begin_in_pass(
            BeginInvestigationCommand(
                journey_file=str(journey.path), request_files=(str(request_file),)
            ),
            environ={"LOCAL_MODEL_KEY": "offline-test-value"},
            model_factory=_searching_planner(),
            source_adapter_factory=SourceResponsesOnly,
        )
        assert pending.status == "review_pending", pending.status
        journey = Journey.open(journey.path)
        cls.journey_file = journey.path
        cls.receipt_ids = [
            event["event_id"]
            for event in read_evidence(journey)
            if event["event_type"] == "JourneySearchRead"
        ]

    def _observe_counting(self):
        projected = Counter()
        checked = Counter()
        with ExitStack() as stack:
            _counted_everywhere(
                stack,
                investigation.project_investigation,
                lambda events, **_: projected.update(
                    [tuple(event["event_id"] for event in events)]
                ),
            )
            _counted_everywhere(
                stack,
                assignment_context.validate_search_receipt,
                lambda prefix, fact, arguments: checked.update([fact["event_id"]]),
            )
            observation = observe(ObserveRun(journey_file=str(self.journey_file)))
        return observation, projected, checked

    def test_one_observe_projects_each_investigation_prefix_once(self):
        observation, projected, _ = self._observe_counting()

        self.assertEqual(observation.investigation.status, "review_pending")
        self.assertEqual(len(observation.investigation.attempts), 3)
        repeated = {
            len(prefix): count for prefix, count in projected.items() if count > 1
        }
        self.assertEqual(repeated, {}, "prefix length -> projections")

    def test_one_observe_checks_each_search_receipt_once(self):
        _, _, checked = self._observe_counting()

        # Each member's search, beside the planning search the Fork inherits.
        self.assertEqual(len(self.receipt_ids), 4)
        self.assertEqual(checked, Counter(self.receipt_ids))

    def test_cached_reads_equal_the_full_fold_at_every_prefix(self):
        """At every prefix, as one owner grows its reads, the cached projection and
        Attempt terminals equal a detached full fold that checks every receipt,
        refusals included; the owner checks each receipt once."""
        from bio_neb.agentic_journey.attempt_journey import attempt_events
        from bio_neb.application.planning_read import (
            PrefixSnapshot,
            prefix_snapshot,
        )
        from bio_neb.investigation.investigation_assignment import (
            investigation_assignment,
        )

        terminal = investigation_assignment().reopen.domain_bound.terminal

        def outcome(read, *arguments):
            try:
                return ("read", read(*arguments))
            except ValueError as error:
                return ("refused", str(error))

        def reads(snapshot, prefix, starts):
            projected = outcome(
                snapshot.derived, investigation.project_snapshot_investigation
            )
            terminals = []
            for started in starts:
                selected = outcome(attempt_events, prefix, started)
                if started in prefix and selected[0] == "read":
                    terminals.append(outcome(terminal, snapshot, selected[1]))
            return projected, terminals

        # The snapshot holds its owner weakly; the reader keeps it alive.
        owner = Journey.open(self.journey_file)
        owned = prefix_snapshot(owner)
        starts = [
            event
            for event in owned.timeline
            if event["event_type"] == "AttemptStarted"
            and event["payload"]["agent_role"] == "investigation_planner"
        ]
        self.assertEqual(len(starts), 3)
        checked = Counter()
        cached = []
        with ExitStack() as stack:
            _counted_everywhere(
                stack,
                assignment_context.validate_search_receipt,
                lambda prefix, fact, arguments: checked.update([fact["event_id"]]),
            )
            for index, event_id in enumerate(owned.prefix_event_ids):
                prefix = owned.timeline[: index + 1]
                snapshot = outcome(owned.through, event_id)
                cached.append(
                    snapshot
                    if snapshot[0] == "refused"
                    else reads(snapshot[1], prefix, starts)
                )

        compared = Counter()
        for index, expected in enumerate(cached):
            prefix = owned.timeline[: index + 1]
            detached = outcome(PrefixSnapshot, prefix)
            full = (
                detached
                if detached[0] == "refused"
                else reads(detached[1], prefix, starts)
            )
            with self.subTest(prefix=index + 1):
                self.assertEqual(expected, full)
            if detached[0] == "read":
                compared[full[0][0]] += 1
                compared["terminals"] += len(full[1])
        # Both branches ran: prefixes that read, prefixes that refuse alike (an
        # Attempt mid-flight), and each Attempt's terminal from its end onward.
        self.assertGreater(compared["read"], 0, compared)
        self.assertGreater(compared["refused"], 0, compared)
        self.assertGreaterEqual(compared["terminals"], len(starts), compared)
        self.assertEqual(checked, Counter(self.receipt_ids))

    def test_pass_metadata_at_a_prefix_end_changes_no_investigation_fold(self):
        """At every prefix ending in Pass metadata or a Fork origin, the full fold
        equals the fold of the prefix before them, refusals included (#437 fix 3)."""
        events = read_evidence(Journey.open(self.journey_file))

        def fold(prefix):
            try:
                return ("read", investigation.project_investigation(prefix))
            except ValueError as error:
                return ("refused", str(error))

        start = next(e for e in events if e["event_type"] == "InvestigationStarted")
        compared = Counter()
        for index in range(1, len(events) + 1):
            prefix = events[:index]
            domain = len(_interpreted(prefix))
            if domain in {0, index}:
                continue
            # The same prefix with a second start before its metadata refuses.
            hostile = [
                *prefix[:domain],
                {**start, "event_id": str(uuid.uuid4())},
                *prefix[domain:],
            ]
            for name, candidate, before in (
                ("retained", prefix, domain),
                ("hostile", hostile, domain + 1),
            ):
                with self.subTest(prefix=index, case=name):
                    expected = fold(candidate)
                    self.assertEqual(fold(candidate[:before]), expected)
                compared[expected[0]] += 1
        self.assertGreater(compared["read"], 0, compared)
        self.assertGreater(compared["refused"], 0, compared)

    def test_an_independent_open_rechecks_a_receipt_its_process_already_checked(self):
        owner = Journey.open(self.journey_file)
        queries.read_investigation(owner)
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / self.journey_file.parent.name
            shutil.copytree(self.journey_file.parent, copy)
            changed = owner.read()
            receipt = next(
                event
                for event in reversed(changed)
                if event["event_type"] == "JourneySearchRead"
            )
            receipt["payload"]["result"]["events"] = []
            (copy / self.journey_file.name).write_text(
                "".join(
                    canonical_json(event) + "\n"
                    for event in changed
                    if event["journey_id"] == owner.journey_id
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "search receipt changed its exact Run prefix"
            ):
                queries.read_investigation(Journey.open(copy / self.journey_file.name))
        # The original owner still reads its own checked receipts.
        self.assertEqual(
            queries.read_investigation(owner).investigation.status, "review_pending"
        )


class MemberAdmissionCostTests(unittest.TestCase):
    """Admitting N members extends the owner's held pieces, not N whole folds (#437).

    #105's comparison: planning 40 members took 40 s and 80 members 182 s with
    instant fakes, each admission re-reading the whole Journey.
    """

    def test_admitting_members_parses_each_investigation_fact_once(self):
        from bio_neb.application.advancement import (
            BeginInvestigation,
            advance,
            use_run_advancement_dependencies,
        )

        parsed = Counter()
        original = investigation.validate_investigation_fact
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = collection_request(3)
            journey = forked_child(root, investigation_request=document)
            request_file = root / "investigation-request.json"
            request_file.write_text(canonical_json(document), encoding="utf-8")

            def counted(event):
                fact = original(event)
                if fact is not None:
                    parsed[event["event_id"]] += 1
                return fact

            with (
                use_run_advancement_dependencies(
                    environ={"LOCAL_MODEL_KEY": "offline-test-value"},
                    investigation_model_factory=SequentialAdaptiveModel().model,
                    investigation_source_adapter_factory=SourceResponsesOnly,
                ),
                patch.object(investigation, "validate_investigation_fact", counted),
            ):
                pending = advance(
                    BeginInvestigation(
                        journey_file=str(journey.path),
                        request_files=(str(request_file),),
                    )
                )
            starts = [
                e
                for e in read_evidence(Journey.open(journey.path))
                if e["event_type"] == "AttemptStarted"
                and e["payload"]["agent_role"] == "investigation_planner"
            ]

        self.assertEqual(pending.status, "investigation_review_pending")
        self.assertEqual(len(starts), 3)
        self.assertGreater(len(parsed), 3)
        self.assertEqual(
            {event: count for event, count in parsed.items() if count > 1}, {}
        )


def _interpreted(events):
    """A prefix's identity for its Investigation: Pass metadata and a Fork origin
    at its end change no Investigation fact, so they do not make a new prefix."""
    ids = [event["event_id"] for event in events]
    types = [event["event_type"] for event in events]
    while types and types[-1] in {"PassStarted", "PassClosed", "RunForked"}:
        ids.pop()
        types.pop()
    return tuple(ids)


class CommandReadOnceCostTests(RunPrerequisites, unittest.TestCase):
    """One command pays each Investigation prefix's interpretation once (#437).

    Each append gives the owner a new revision. Before #437 every revision's read
    folded the head, each collection prefix and each member's origin prefix
    again: an offline Finalization answer on #418's Journey read the whole
    Investigation 11 times.
    """

    def test_a_finalization_answer_projects_each_investigation_prefix_once(self):
        """Prefixes that differ only by trailing Pass metadata count as one."""
        baseline = self.prerequisite(FINALIZATION_PENDING)
        request_id, response = baseline.value
        projected = Counter()
        with item_state_dependencies(RunOutputStore()), ExitStack() as stack:
            _counted_everywhere(
                stack,
                investigation.project_investigation,
                lambda events, **_: projected.update([_interpreted(events)]),
            )
            code, error = native_answer(baseline.journey_file, request_id, response)
        observation = observe(ObserveRun(journey_file=str(baseline.journey_file)))

        self.assertEqual(code, 0, error)
        self.assertEqual(observation.status, "grounded_report_published")
        repeated = {
            len(prefix): count for prefix, count in projected.items() if count > 1
        }
        self.assertGreater(len(projected), 1)
        self.assertEqual(repeated, {}, "prefix length -> projections")

    def test_a_finalization_answer_checks_each_basis_report_and_table_piece_once(self):
        """The Finalization basis, each publication and each table chunk and window
        are fixed by their admitted events: one command checks each once (#437)."""
        import bio_neb.investigation.vcf_table_reading as table_reading
        import bio_neb.reporting.finalization as finalization
        import bio_neb.reporting.grounded_report as grounded_report

        baseline = self.prerequisite(FINALIZATION_PENDING)
        request_id, response = baseline.value
        checked = Counter()
        with item_state_dependencies(RunOutputStore()), ExitStack() as stack:
            _counted_everywhere(
                stack,
                finalization.compile_finalization_basis,
                lambda candidate: checked.update(
                    [("basis", candidate.source_head_event["event_id"])]
                ),
            )
            _counted_everywhere(
                stack,
                grounded_report._project_report_publication,
                lambda events, **kwargs: checked.update(
                    [("publication", kwargs["publication_event_id"])]
                ),
            )
            _counted_everywhere(
                stack,
                table_reading.table_rows,
                lambda events, stage: checked.update([("chunk rows", len(events))]),
            )
            _counted_everywhere(
                stack,
                table_reading._validate_window,
                lambda timeline, window: checked.update(
                    [("window", window.collection_event_id, window.work_item_id)]
                ),
            )
            code, error = native_answer(baseline.journey_file, request_id, response)

        self.assertEqual(code, 0, error)
        kinds = Counter(key[0] for key in checked)
        self.assertGreater(kinds["basis"], 0, kinds)
        self.assertGreater(kinds["publication"], 0, kinds)
        self.assertGreater(kinds["chunk rows"], 0, kinds)
        self.assertGreater(kinds["window"], 0, kinds)
        self.assertEqual({key: n for key, n in checked.items() if n > 1}, {})

    def test_a_member_contact_count_folds_no_prefix_before_its_evidence(self):
        """Each member's inherited contact counts are its collection's, checked
        once, not a whole projection of the prefix before its evidence (#437 fix 4)."""
        journey_file = self.prerequisite(FINALIZATION_PENDING).journey_file
        events = read_evidence(Journey.open(journey_file))
        projected = []
        with ExitStack() as stack:
            _counted_everywhere(
                stack,
                investigation.project_investigation,
                lambda events, **_: projected.append(events[-1]["event_type"]),
            )
            observation = observe(ObserveRun(journey_file=str(journey_file)))
        recorded = queries.read_investigation(Journey.open(journey_file))

        members = [
            e for e in events if e["event_type"] == "InvestigationEvidenceRecorded"
        ]
        self.assertEqual(len(members), 2)
        self.assertEqual(observation.status, "finalization_review_pending")
        self.assertNotIn("InvestigationWorkItemStarted", projected)
        # Each member's recorded frontier equals a full fold before its evidence.
        for member in recorded.recorded_members:
            origin = investigation.project_investigation(
                events[: events.index(member.event)]
            )
            with self.subTest(member=member.event["event_id"]):
                self.assertEqual(
                    (member.model_contact_count, member.source_contact_count),
                    (origin.model_contact_count, origin.source_contact_count),
                )
        self.assertEqual(len(recorded.recorded_members), 2)
        # A prefix ending in observation facts folds as the prefix before them,
        # so VCF accounting reads one projection per batch.
        observation = {
            "InvestigationWorkItemStarted",
            "InvestigationEvidenceRecorded",
            "InvestigationCollectionEvidenceRecorded",
        }
        compared = 0
        for index, event in enumerate(events):
            if event["event_type"] not in observation:
                continue
            anchor = max(
                position
                for position in range(index)
                if events[position]["event_type"] not in observation
            )
            with self.subTest(prefix=index + 1):
                self.assertEqual(
                    investigation.project_investigation(events[: index + 1]),
                    investigation.project_investigation(events[: anchor + 1]),
                )
            compared += 1
        self.assertGreaterEqual(compared, 5)

    def test_a_command_checks_each_attempt_continuation_once(self):
        """An Attempt's continuation is fixed by the prefix up to its start, so a
        command's reads check each one once, not once per reader and revision
        (#437). The writer's per-append family admission still refolds the
        families without a step (``docs/design/journey-path.md``, per write)."""
        import bio_neb.agentic_journey.continuation as continuation
        from bio_neb.journey_contracts import SemanticFamily

        admitting = Counter()
        admit = SemanticFamily.admit

        def counted_admit(family, events):
            admitting["active"] += 1
            try:
                return admit(family, events)
            finally:
                admitting["active"] -= 1

        baseline = self.prerequisite(INVESTIGATION_REVIEW_PENDING)
        request_id, response = baseline.value
        checked = Counter()
        with item_state_dependencies(RunOutputStore()), ExitStack() as stack:
            stack.enter_context(patch.object(SemanticFamily, "admit", counted_admit))
            _counted_everywhere(
                stack,
                continuation._check_attempt_continuation,
                lambda events, started: None
                if admitting["active"]
                else checked.update([started["event_id"]]),
            )
            code, error = native_answer(baseline.journey_file, request_id, response)

        self.assertEqual(code, 0, error)
        self.assertGreater(len(checked), 1)
        self.assertEqual(
            {start: count for start, count in checked.items() if count > 1}, {}
        )

    def test_a_command_reads_each_attempt_windows_turns_once(self):
        """An Attempt window's turns and the requests after its replies are fixed
        by its own events and the prefix before them, so a command reads each
        window once, not once per reader and revision (owner, #444 Q1). The
        writer's per-append family admission still refolds without a step."""
        import bio_neb.agentic_journey.turn_reopen as turn_reopen
        from bio_neb.journey_contracts import SemanticFamily

        admitting = Counter()
        admit = SemanticFamily.admit

        def counted_admit(family, events):
            admitting["active"] += 1
            try:
                return admit(family, events)
            finally:
                admitting["active"] -= 1

        def window(selected, **_kwargs):
            if not admitting["active"]:
                start = next(e for e in selected if e["event_type"] == "AttemptStarted")
                read.update([(start["event_id"], selected[-1]["event_id"])])

        baseline = self.prerequisite(INVESTIGATION_REVIEW_PENDING)
        request_id, response = baseline.value
        read = Counter()
        with item_state_dependencies(RunOutputStore()), ExitStack() as stack:
            stack.enter_context(patch.object(SemanticFamily, "admit", counted_admit))
            _counted_everywhere(stack, turn_reopen.validate_action_turns, window)
            code, error = native_answer(baseline.journey_file, request_id, response)

        self.assertEqual(code, 0, error)
        self.assertGreater(len(read), 1)
        self.assertEqual({key: count for key, count in read.items() if count > 1}, {})

    def test_pieces_an_owner_holds_across_appends_equal_a_fresh_fold(self):
        """Every prefix and Investigation an owner held while its command appended
        equals what a fresh open derives for that prefix (#437 write-extends)."""
        from bio_neb.application.advancement import advance, review_response_from_json

        baseline = self.prerequisite(FINALIZATION_PENDING)
        request_id, response = baseline.value
        with item_state_dependencies(RunOutputStore()):
            owner = Journey.open(baseline.journey_file)
            before = len(read_evidence(owner))
            view = observe(ObserveRun(journey_file=str(owner.path)), journey=owner)
            published = advance(
                review_response_from_json(view, request_id, json.dumps(response)),
                journey=owner,
            )
        fresh = read_evidence(Journey.open(baseline.journey_file))

        self.assertEqual(published.status, "grounded_report_published")
        self.assertGreater(len(fresh), before)
        import bio_neb.agentic_journey.continuation as continuation
        import bio_neb.agentic_journey.operational_review as operational_review
        import bio_neb.investigation.item_evidence as item_evidence
        import bio_neb.investigation.vcf_table_reading as table_reading
        import bio_neb.reporting.finalization as finalization
        import bio_neb.reporting.grounded_report as grounded_report
        from bio_neb.application.planning_read import PrefixSnapshot

        parsers = {
            parse.__name__: parse
            for parse in (
                investigation.validate_investigation_fact,
                finalization.validate_finalization_request_fact,
                finalization.validate_finalization_review_fact,
            )
        }
        parsers["investigation_fact"] = investigation.validate_investigation_fact
        ids = [event["event_id"] for event in fresh]
        held = Counter()
        for key, value in owner._derived.items():
            if key[0] == "prefix_snapshot":
                with self.subTest(prefix=key[1]):
                    self.assertEqual(
                        value.prefix_event_ids, tuple(ids[: ids.index(key[1]) + 1])
                    )
                held["prefix_snapshot"] += 1
            elif key[0] == "investigation_projection":
                with self.subTest(projection=key[1]):
                    self.assertEqual(
                        value,
                        investigation.project_investigation(
                            fresh[: ids.index(key[1]) + 1]
                        ),
                    )
                held["investigation_projection"] += 1
            elif key[0] in parsers:
                with self.subTest(fact=key):
                    self.assertEqual(value, parsers[key[0]](fresh[ids.index(key[1])]))
                held["fact"] += 1
            elif key[0] == "finalization_basis":
                prefix = PrefixSnapshot(fresh[: ids.index(key[1]) + 1])
                with self.subTest(basis=key[1]):
                    self.assertEqual(
                        value,
                        finalization.compile_finalization_basis(
                            prefix.investigation.candidate
                        ),
                    )
                held["finalization_basis"] += 1
            elif key[0] == "report_publication":
                prefix = fresh[: ids.index(key[1]) + 1]
                with self.subTest(publication=key[1]):
                    self.assertEqual(
                        value,
                        grounded_report.project_grounded_report(prefix),
                    )
                held["report_publication"] += 1
            elif key[0] == "attempt_continuation":
                with self.subTest(continuation=key[1]):
                    self.assertEqual(
                        value,
                        continuation.validate_attempt_continuation(
                            fresh, fresh[ids.index(key[1])]
                        ),
                    )
                held["attempt_continuation"] += 1
            elif key[0] == "operational_review_before":
                with self.subTest(operational=key[1]):
                    self.assertEqual(
                        value,
                        operational_review.project_operational_review(
                            fresh[: ids.index(key[1])]
                        ),
                    )
                held["operational_review_before"] += 1
            elif key[0] == "table_rows":
                index = ids.index(key[1])
                chunk = item_evidence.validate_item_evidence_fact(fresh[index])
                stage = table_reading._historical_stage(
                    fresh[:index],
                    chunk,
                    {
                        proposal.proposal_id: proposal
                        for proposal in investigation.project_investigation(
                            fresh
                        ).proposals
                    },
                )
                with self.subTest(chunk=key[1]):
                    self.assertEqual(
                        value, table_reading.table_rows(fresh[:index], stage)
                    )
                held["table_rows"] += 1
            elif key[0] == "member_context":
                # A member's context, rendered once from its fact's pointers.
                from bio_neb.agentic_journey.role_instructions import (
                    role_instructions,
                )
                from bio_neb.investigation.member_context import rendered_text

                fact = investigation.validate_investigation_fact(
                    fresh[ids.index(key[1])]
                )
                started = next(
                    e
                    for e in fresh
                    if e["event_type"] == "AttemptStarted"
                    and e["payload"]["attempt_id"] == fact.attempt_id
                )
                role, version, form = started["payload"]["instruction_identity"].split(
                    ":"
                )
                with self.subTest(member_context=key[1]):
                    self.assertEqual(
                        value,
                        rendered_text(
                            fact,
                            {e["event_id"]: e for e in fresh},
                            role_instructions(role, version=version, form=form),
                            investigation.validate_investigation_fact,
                        ),
                    )
                held["member_context"] += 1
            elif key[0] == "attempt_transcripts":
                # An Attempt window's turns and the requests after its replies,
                # read once: a fresh read of the same window passes.
                from bio_neb.agentic_journey.turn_reopen import validate_action_turns

                prefix = fresh[: ids.index(key[2]) + 1]
                starts = [e for e in fresh if e["event_type"] == "AttemptStarted"]
                # The window as the owner held it: its events through key[2].
                (whole,) = [
                    window
                    for _begin, _end, window in queries._attempt_windows(fresh, starts)
                    if key[1] in {e["event_id"] for e in window}
                ]
                last = [e["event_id"] for e in whole].index(key[2])
                selected = whole[: last + 1]
                started = next(e for e in selected if e["event_id"] == key[1])
                with self.subTest(transcripts=key[1:]):
                    self.assertIsNone(value)
                    self.assertIsNone(
                        validate_action_turns(
                            selected,
                            role=started["payload"]["agent_role"],
                            assignment=queries._recorded_assignment(
                                prefix, started["payload"]["attempt_id"]
                            ),
                            timeline=prefix,
                        )()
                    )
                held["attempt_transcripts"] += 1
            elif key[0] == "table_window":
                window = item_evidence.validate_item_evidence_fact(
                    fresh[ids.index(key[1])]
                )
                with self.subTest(window=key[1]):
                    # The fresh check passes too: it raises when it refuses.
                    self.assertIsNone(table_reading._validate_window(fresh, window))
                    self.assertEqual(
                        key[2],
                        table_reading.run_table_settings(fresh).window_flank_bases,
                    )
                held["table_window"] += 1
            elif key[0] == "item_evidence_fold":
                _assert_held_fold_equals_fresh(self, fresh, key[1], value)
                held["item_evidence_fold"] += 1
        self.assertGreater(held["prefix_snapshot"], 0, held)
        self.assertGreater(held["investigation_projection"], 1, held)
        self.assertGreater(held["fact"], 1, held)
        self.assertEqual(held["finalization_basis"], 1, held)
        self.assertEqual(held["report_publication"], 1, held)
        for kind in (
            "attempt_continuation",
            "operational_review_before",
            "table_rows",
            "table_window",
            "member_context",
            "attempt_transcripts",
            "item_evidence_fold",
        ):
            self.assertGreater(held[kind], 0, held)


def _assert_held_fold_equals_fresh(test, fresh, fold_event_id, value) -> None:
    """A fold's held check over the prefix before it (#414, #437 read (b)): a
    fresh fold of that prefix gives the same items."""
    import bio_neb.investigation.item_evidence as item_evidence
    from bio_neb.investigation.investigation_observation import (
        validate_investigation_collection_evidence_fact,
    )

    ids = [event["event_id"] for event in fresh]
    index = ids.index(fold_event_id)
    fact = item_evidence.validate_item_evidence_fact(fresh[index])
    (collection,) = [
        validate_investigation_collection_evidence_fact(event)
        for event in fresh
        if event["event_id"] == fact.collection_event_id
    ]
    with test.subTest(fold=fold_event_id):
        test.assertIsNone(value)
        test.assertEqual(
            fact.items,
            item_evidence.fold_item_evidence(
                fact.collection_event_id, collection, fresh[:index]
            ),
        )


if __name__ == "__main__":
    unittest.main()
