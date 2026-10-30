"""One Fork creates an idle child Run from a closed point of a retained parent."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bio_neb.application.fork import ForkRefused, ForkRun, fork_run
from bio_neb.cli import main
from bio_neb.journey import Journey, JourneyValidationError
from bio_neb.recording import processing_pass, read_passes


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


def _pass(journey: Journey, label: str, *, close: str | None = "waiting") -> None:
    """One coordinator interval holding one fact with an artifact."""
    with journey.processing(), processing_pass(journey) as interval:
        journey.record_with_artifact(
            "NoteRetained",
            {"label": label},
            f"bytes of {label}".encode(),
            "text/plain",
            parent_ids=[journey.head_event_id],
        )
        if close is not None:
            interval.close(close)


def _parent(root: Path, *passes: str, last: str | None = "waiting") -> Journey:
    journey = Journey.create(root)
    journey.record("NoteRegistered", {"label": "registered"})
    for label in passes[:-1]:
        _pass(journey, label)
    _pass(journey, passes[-1], close=last)
    return Journey.open(journey.path)


class RunForkTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def fork(self, parent: Journey, name: str = "child", **point):
        return fork_run(
            ForkRun(
                parent_journey_file=str(parent.path),
                child_run_root=str(self.root / name),
                **point,
            )
        )

    def test_fork_at_a_closed_head_creates_an_idle_child_over_the_exact_prefix(self):
        parent = _parent(self.root / "parent", "first")
        before = _tree(self.root / "parent")

        receipt = self.fork(parent)

        self.assertEqual(_tree(self.root / "parent"), before)
        child = Journey.open(Path(receipt.child_journey_file))
        events = child.read()
        inherited = parent.read()
        self.assertEqual(receipt.status, "fork_created")
        self.assertNotEqual(child.journey_id, parent.journey_id)
        self.assertEqual(events[: len(inherited)], inherited)
        origin = events[len(inherited)]
        self.assertEqual(
            (origin["event_type"], origin["journey_id"], origin["parent_ids"]),
            ("RunForked", child.journey_id, [parent.head_event_id]),
        )
        self.assertEqual(len(events), len(inherited) + 1)
        self.assertEqual(receipt.origin.parent, parent.identity())
        self.assertEqual(receipt.origin.point_event_id, parent.head_event_id)
        self.assertEqual(receipt.origin.point_disposition, "waiting")
        self.assertEqual(receipt.inherited_event_count, len(inherited))
        # The child's file holds only its origin, whose artifact is the parent prefix.
        self.assertEqual(
            len(Path(receipt.child_journey_file).read_bytes().splitlines()), 1
        )
        self.assertEqual(
            child.read_artifact(origin["artifact_references"][0]),
            parent.path.read_bytes(),
        )
        note = next(e for e in events if e["event_type"] == "NoteRetained")
        self.assertEqual(
            child.read_artifact(note["artifact_references"][0]), b"bytes of first"
        )

    def test_an_open_pass_head_is_refused_and_no_child_root_appears(self):
        parent = _parent(self.root / "parent", "first", last=None)
        before = _tree(self.root / "parent")

        with self.assertRaises(ForkRefused) as refused:
            self.fork(parent)

        self.assertEqual(refused.exception.code, "fork_point_not_closed")
        self.assertEqual(refused.exception.detail, "NoteRetained")
        self.assertEqual(_tree(self.root / "parent"), before)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["parent"])

    def test_an_earlier_closed_point_excludes_later_parent_history(self):
        parent = _parent(self.root / "parent", "first", "second", last="interrupted")
        events = parent.read()
        point = next(e for e in events if e["event_type"] == "PassClosed")

        receipt = self.fork(parent, point_event_id=point["event_id"])

        child = Journey.open(Path(receipt.child_journey_file)).read()
        cut = events.index(point) + 1
        self.assertEqual(child[:cut], events[:cut])
        self.assertEqual([e["event_type"] for e in child[cut:]], ["RunForked"])
        self.assertNotIn("second", [e["payload"].get("label") for e in child])
        self.assertEqual(receipt.origin.parent.event_count, cut)

    def test_an_unknown_point_is_refused(self):
        parent = _parent(self.root / "parent", "first")
        with self.assertRaises(ForkRefused) as refused:
            self.fork(parent, point_event_id="00000000-0000-4000-8000-000000000000")
        self.assertEqual(refused.exception.code, "fork_point_unknown")
        self.assertFalse((self.root / "child").exists())

    def test_the_same_request_returns_the_same_child_and_another_point_conflicts(self):
        parent = _parent(self.root / "parent", "first", "second", last="completed")
        created = self.fork(parent)
        tree = _tree(self.root / "child")

        replayed = self.fork(parent)
        first = next(e for e in parent.read() if e["event_type"] == "PassClosed")
        with self.assertRaises(ForkRefused) as conflict:
            self.fork(parent, point_event_id=first["event_id"])
        (self.root / "occupied").mkdir()
        (self.root / "occupied" / "note").write_text("not a Run")
        with self.assertRaises(ForkRefused) as occupied:
            self.fork(parent, "occupied")

        self.assertEqual(replayed.status, "fork_replayed")
        self.assertEqual(
            replayed.model_copy(update={"status": "fork_created"}), created
        )
        self.assertEqual(_tree(self.root / "child"), tree)
        self.assertEqual(conflict.exception.code, "fork_child_conflict")
        self.assertEqual(occupied.exception.code, "fork_child_root_not_fresh")
        # A deliberate new request at another root creates another child.
        other = self.fork(parent, "another")
        self.assertNotEqual(other.child_journey_id, created.child_journey_id)

    def test_the_child_records_its_own_passes_after_the_inherited_prefix(self):
        parent = _parent(self.root / "parent", "first")
        receipt = self.fork(parent)
        child = Journey.open(Path(receipt.child_journey_file))

        _pass(child, "child work", close="completed")

        reopened = Journey.open(child.path)
        passes = read_passes(reopened)
        # The inherited closed Pass keeps its parent Run identity; the child's
        # own Pass journal is bound to the child.
        self.assertEqual(
            [
                (view.disposition, view.journal_complete, view.contacts[-1].run_id)
                for view in passes
            ],
            [
                ("waiting", True, parent.journey_id),
                ("completed", True, child.journey_id),
            ],
        )
        own = [e for e in reopened.read() if e["journey_id"] == child.journey_id]
        self.assertEqual(
            [e["event_type"] for e in own],
            ["RunForked", "PassStarted", "NoteRetained", "PassClosed"],
        )
        self.assertEqual(
            parent.path.read_bytes(), Journey.open(parent.path).retained_prefix
        )

    def test_a_fork_of_a_child_reads_the_whole_lineage(self):
        parent = _parent(self.root / "parent", "first")
        child = Journey.open(Path(self.fork(parent).child_journey_file))
        _pass(child, "child work", close="completed")
        child = Journey.open(child.path)

        grandchild = Journey.open(
            Path(self.fork(child, "grandchild").child_journey_file)
        )

        events = grandchild.read()
        self.assertEqual(events[: child.revision], child.read())
        self.assertEqual([e["event_type"] for e in events].count("RunForked"), 2)
        self.assertEqual(len({e["journey_id"] for e in events}), 3)

    def test_an_altered_inherited_prefix_refuses_to_open(self):
        parent = _parent(self.root / "parent", "first")
        receipt = self.fork(parent)
        child = Journey.open(Path(receipt.child_journey_file))
        origin = child.read()[receipt.inherited_event_count]
        prefix = child.artifact_path(origin["artifact_references"][0])
        prefix.chmod(0o600)
        prefix.write_bytes(prefix.read_bytes().replace(b"first", b"forst"))

        with self.assertRaisesRegex(
            JourneyValidationError, "Fork parent prefix digest does not match"
        ):
            Journey.open(child.path)

    def test_an_unreadable_parent_is_refused_before_any_child_root(self):
        parent = self.root / "parent" / "journey.jsonl"
        parent.parent.mkdir()
        parent.write_text('{"not": "a journey"}\n', encoding="utf-8")

        with self.assertRaises(ForkRefused) as refused:
            fork_run(
                ForkRun(
                    parent_journey_file=str(parent),
                    child_run_root=str(self.root / "child"),
                )
            )

        self.assertEqual(refused.exception.code, "parent_journey_unreadable")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["parent"])

    def test_a_parent_changed_during_creation_publishes_no_child(self):
        parent = _parent(self.root / "parent", "first")
        create_fork = Journey.create_fork

        def append_to_parent(*args, **kwargs):
            child = create_fork(*args, **kwargs)
            # Another writer extends the parent while the child is staged.
            _pass(Journey.open(parent.path), "late", close="waiting")
            return child

        with (
            patch.object(Journey, "create_fork", side_effect=append_to_parent),
            self.assertRaises(ForkRefused) as refused,
        ):
            self.fork(parent)

        self.assertEqual(refused.exception.code, "fork_parent_changed")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["parent"])

    def test_a_child_at_the_before_approval_point_waits_at_the_same_review(self):
        # The Run waiting at its Execution Review ends with a closed Pass (ADR
        # 0031's first child point); the child observes as the parent did.
        from bio_neb.application.queries import ObserveRun, observe
        from tests.planning_assignment_fakes import (
            execution_configuration,
            prepare_configured_run,
        )

        source = self.root / "source"
        source.mkdir()
        registration, review, _ = prepare_configured_run(
            source, configuration=execution_configuration(source)
        )
        parent = Path(registration.journey_file)
        before = parent.read_bytes()

        receipt = fork_run(
            ForkRun(
                parent_journey_file=str(parent),
                child_run_root=str(self.root / "child"),
            )
        )
        child = observe(ObserveRun(journey_file=receipt.child_journey_file))

        self.assertEqual(receipt.origin.point_disposition, "waiting")
        self.assertEqual(child.status, review.status)
        self.assertEqual(child.current_review.kind, "execution")
        self.assertEqual(
            child.current_review.request_event_id,
            review.current_review.request_event_id,
        )
        self.assertEqual(child.run_provenance, "causal_child_or_fork")
        self.assertIsNone(review.run_provenance)
        # The observation names the reading Run; inherited facts keep the parent's.
        self.assertEqual(child.journey_id, receipt.child_journey_id)
        self.assertEqual(review.journey_id, receipt.origin.parent.journey_id)
        self.assertEqual(parent.read_bytes(), before)

    def cli(self, *argv: str) -> tuple[int, str, str]:
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(list(argv))
        return code, output.getvalue(), errors.getvalue()

    def test_cli_fork_prints_the_receipt_and_a_typed_refusal(self):
        closed = _parent(self.root / "closed", "first")
        open_ = _parent(self.root / "open", "first", last=None)

        code, output, errors = self.cli(
            "fork",
            "--parent-journey-file",
            str(closed.path),
            "--child-run-root",
            str(self.root / "child"),
        )
        refused, _, refusal = self.cli(
            "fork",
            "--parent-journey-file",
            str(open_.path),
            "--child-run-root",
            str(self.root / "refused"),
        )

        self.assertEqual((code, errors), (0, ""))
        receipt = json.loads(output)
        self.assertEqual(receipt["status"], "fork_created")
        self.assertEqual(receipt["origin"]["parent"], closed.identity().model_dump())
        self.assertTrue(Path(receipt["child_journey_file"]).is_file())
        self.assertEqual(refused, 2)
        self.assertEqual(
            json.loads(refusal),
            {
                "schema_version": "bio-neb.agentic-cli/1",
                "frontier": "refused",
                "code": "fork_point_not_closed",
                "disposition": {"kind": "caller_correction_and_reissue"},
                "fields": ["fork", "NoteRetained"],
            },
        )
        self.assertFalse((self.root / "refused").exists())


if __name__ == "__main__":
    unittest.main()
