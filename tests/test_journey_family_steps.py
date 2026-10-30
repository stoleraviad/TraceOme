"""Writer step folds equal full folds, and the writer equals a fresh open (#197).

One real build supplies the history: the default ``approved_input`` planning Run,
then the execution operation up to its CPU preparation review. It holds planning
facts, closed processing Passes and execution path facts written by the writer.
"""

import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

from bio_neb.journey import Journey
from bio_neb.journey_contracts import semantic_families
from bio_neb.parabricks.operation import execute_parabricks
from tests.parabricks_operation_fakes import approved_input, input_storage_provider


def _outcome(call, *args):
    try:
        return "admitted", call(*args)
    except ValueError as error:
        return "refused", type(error), str(error)


def _fresh_views(path: Path) -> dict:
    """Checked views of an independent open of the current disk bytes."""
    events, retained = Journey._load_events(path)
    fresh = Journey._admitted(path, events[0].journey_id, events, retained)
    names = {
        family.name
        for family in semantic_families(frozenset(e.event_type for e in events))
    }
    return {name: fresh.checked_view(name) for name in names}


class FamilyStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.mismatches = []
        cls.appends = 0
        original = Journey._append

        def append(journey, candidate):
            original(journey, candidate)
            cls.appends += 1
            fresh = _fresh_views(journey.path)
            writer = {name: journey.checked_view(name) for name in fresh}
            if writer != fresh:
                cls.mismatches.append(
                    (
                        candidate.event.event_type,
                        sorted(n for n in fresh if fresh[n] != writer[n]),
                    )
                )

        with patch.object(Journey, "_append", append):
            approved = approved_input(Path(cls._directory.name) / "case")
            execute_parabricks(approved, infrastructure_factory=input_storage_provider)
        cls.events = Journey.open(Path(approved.journey_file)).read()

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def stepped_families(self, events):
        families = [
            family
            for family in semantic_families(frozenset(e["event_type"] for e in events))
            if family.step is not None
        ]
        self.assertEqual(
            {family.name for family in families}, {"execution", "processing_pass"}
        )
        return families

    def assert_steps_equal_folds(self, events):
        for family in self.stepped_families(events):
            state = family.admit([])
            for index, event in enumerate(events):
                folded = _outcome(family.admit, events[: index + 1])
                stepped = _outcome(family.admit_next, state, event, events[:index])
                self.assertEqual(
                    stepped, folded, f"{family.name} {index} {event['event_type']}"
                )
                if folded[0] == "refused":
                    break
                state = stepped[1]

    def test_build_covers_every_stepped_family_with_its_facts(self):
        types = [e["event_type"] for e in self.events]
        self.assertGreaterEqual(types.count("PassClosed"), 2)
        self.assertGreaterEqual(types.count("ExecutionPathRecorded"), 3)

    def test_stepwise_folding_equals_full_fold_at_every_prefix(self):
        self.assert_steps_equal_folds(self.events)

    def test_step_refuses_exactly_where_the_full_fold_refuses(self):
        # Dropping or swapping a family fact breaks the history at a known
        # point; the step must refuse there with the fold's own reason.
        positions = [
            index
            for index, event in enumerate(self.events)
            if event["event_type"]
            in {"PassStarted", "PassClosed", "ExecutionPathRecorded"}
        ]
        variants = [
            self.events[:index] + self.events[index + 1 :] for index in positions
        ] + [
            [
                *self.events[:first],
                self.events[second],
                *self.events[first + 1 : second],
                self.events[first],
                *self.events[second + 1 :],
            ]
            for first, second in pairwise(positions)
        ]
        refused = 0
        for variant in variants:
            self.assert_steps_equal_folds(variant)
            refused += any(
                _outcome(family.admit, variant)[0] == "refused"
                for family in self.stepped_families(variant)
            )
        # Dropping a trailing fact (the last closure or review) can leave a
        # lawful shorter history, so not every variant refuses; most must, or
        # the equal-refusal check above would be vacuous.
        self.assertGreater(refused, len(variants) // 2)

    def test_writer_checked_views_equal_a_fresh_open_after_every_append(self):
        self.assertGreater(self.appends, len(self.events) // 2)
        self.assertEqual(self.mismatches, [])


if __name__ == "__main__":
    unittest.main()
