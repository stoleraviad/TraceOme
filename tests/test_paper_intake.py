"""A Run paper's intake: retained once, read once, recorded, or refused into its Review (#420)."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from bio_neb.failure import ReviewAnswer
from bio_neb.journey import Journey
from bio_neb.paper_intake.documents import normalize_text
from bio_neb.paper_intake.intake import (
    answer_paper_review,
    register_paper_files,
    take_supplied_papers,
)
from bio_neb.paper_intake.journey import paper_id
from tests.paper_files_support import pdf_bytes, zipped


class SuppliedPaperTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root))
        self.journey = Journey.create(self.root / "run")
        self.started = self.journey.record("JourneyStarted", {}, [])

    def supply(self, *paths: Path) -> None:
        register_paper_files(
            self.journey, [str(p) for p in paths], self.started["event_id"]
        )

    def papers(self):
        return self.journey.checked_view("run_papers")

    def answer(self, request, action, **more):
        return answer_paper_review(
            self.journey,
            request.request_event_id,
            ReviewAnswer(
                request_id=request.request.request_id,
                action=action,
                by="researcher",
                **more,
            ),
        )

    def test_a_supplied_pdf_is_retained_read_and_recorded_with_its_derived_id(self):
        pdf = pdf_bytes(["APOE rs429358 at 19:44908684"])
        path = self.root / "paper.pdf"
        path.write_bytes(pdf)
        self.supply(path)

        (event,) = take_supplied_papers(self.journey)

        (paper,) = self.papers().papers
        self.assertEqual(event["event_type"], "PaperRecorded")
        self.assertEqual((paper.origin, paper.paper_file), ("file", str(path)))
        self.assertEqual(
            paper.paper_id,
            "paper-"
            + hashlib.sha256(hashlib.sha256(pdf).hexdigest().encode()).hexdigest(),
        )
        (document,) = paper.documents
        self.assertEqual(document.document_id, "main")
        events = {e["event_id"]: e for e in self.journey.events_view()}
        read = events[document.read_event_id]
        retained = events[read["payload"]["retained_event_id"]]
        self.assertEqual(
            self.journey.read_artifact(retained["artifact_references"][0]), pdf
        )
        self.assertEqual(
            self.journey.read_artifact(read["artifact_references"][0]).decode(),
            normalize_text("## Page 1\nAPOE rs429358 at 19:44908684"),
        )
        self.assertEqual(self.papers().files_due, ())

    def test_a_zip_s_unreadable_member_is_listed_beside_the_documents_it_gives(self):
        path = self.root / "tables.zip"
        path.write_bytes(zipped({"S1.tsv": b"rs4680\tCOMT\n", "S2.bin": b"\x00"}))
        self.supply(path)

        take_supplied_papers(self.journey)

        (paper,) = self.papers().papers
        self.assertEqual([d.document_id for d in paper.documents], ["supp:S1.tsv"])
        self.assertEqual(
            [(u.name, u.reason) for u in paper.unread],
            [("S2.bin", "no reader for .bin")],
        )

    def test_a_missing_file_asks_its_review_and_an_amended_path_is_read(self):
        path = self.root / "paper.txt"
        self.supply(path)

        take_supplied_papers(self.journey)

        (request,) = self.papers().requests
        self.assertEqual(request.refusal.code, "paper_file_missing")
        self.assertEqual(
            request.request.legal_actions, ("retry", "amend", "continue", "abort")
        )
        self.assertEqual(request.request.amendable, ("paper_file",))
        self.assertEqual(take_supplied_papers(self.journey), ())  # nothing is due

        found = self.root / "found.txt"
        found.write_text("rs429358 is reported.")
        self.answer(request, "amend", field="paper_file", value=str(found))
        take_supplied_papers(self.journey)

        (paper,) = self.papers().papers
        self.assertEqual(paper.paper_file, str(found))
        self.assertFalse(self.papers().review_pending)

    def test_an_unreadable_file_is_refused_and_continue_drops_it(self):
        path = self.root / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        self.supply(path)

        take_supplied_papers(self.journey)
        (request,) = self.papers().requests
        self.assertEqual(request.refusal.code, "paper_file_unreadable")
        self.answer(request, "continue")

        (intake,) = self.papers().intakes
        self.assertEqual(intake.state, "dropped")
        self.assertEqual(self.papers().papers, ())
        self.assertEqual(take_supplied_papers(self.journey), ())

    def test_retry_reads_the_same_path_again_and_abort_ends_the_intake(self):
        path = self.root / "late.txt"
        self.supply(path)
        take_supplied_papers(self.journey)
        (request,) = self.papers().requests

        path.write_text("rs4680 COMT")
        self.answer(request, "retry")
        take_supplied_papers(self.journey)
        self.assertEqual(len(self.papers().papers), 1)

        other = self.root / "never.txt"
        self.supply(other)
        take_supplied_papers(self.journey)
        pending = [r for r in self.papers().requests if r.pending]
        self.answer(pending[0], "abort")
        self.assertTrue(self.papers().aborted)

    def test_an_amendment_names_a_declared_field_and_an_absolute_path(self):
        """Owner #420 P1: a supplied file's Review amends only ``paper_file``, an
        absolute path; a refused answer writes nothing and the Review stays open."""
        path = self.root / "paper.txt"
        self.supply(path)
        take_supplied_papers(self.journey)
        (request,) = self.papers().requests
        written = len(self.journey.read())

        for field, value, reason in (
            (
                "paper_reference",
                "doi:10.1/x",
                "a paper amendment names one declared field and a text",
            ),
            ("paper_file", "rel.txt", "a paper file path must be absolute"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, reason):
                    self.answer(request, "amend", field=field, value=value)
        self.assertEqual(len(self.journey.read()), written)
        self.assertTrue(self.papers().review_pending)

    def test_a_recorded_paper_that_names_other_files_than_it_kept_is_refused(self):
        """#420 ST1: the paper's identity is bound to the files its intake retained;
        a forged digest list with its matching id is not admitted."""
        from bio_neb.paper_intake.journey import (
            PaperDocumentRead,
            PaperDocumentRetained,
            PaperRecorded,
            RecordedDocument,
            RecordedFile,
        )

        path = self.root / "paper.txt"
        self.supply(path)
        (intake,) = self.papers().intakes
        content, text = b"rs4680 COMT", "rs4680 COMT"
        digest = hashlib.sha256(content).hexdigest()
        text_digest = hashlib.sha256(text.encode()).hexdigest()

        def write(fact, parent, artifact=None):
            payload = fact.model_dump(mode="json", exclude={"event_type"})
            if artifact is None:
                return self.journey.record(fact.event_type, payload, [parent])
            return self.journey.record_with_artifact(
                fact.event_type, payload, artifact, "text/plain", [parent]
            )

        retained = write(
            PaperDocumentRetained(
                intake_id=intake.intake_id,
                name="paper.txt",
                role="supplied",
                media_type="text/plain",
                location=str(path),
                sha256=digest,
                byte_count=len(content),
            ),
            intake.source_event_id,
            content,
        )
        read = write(
            PaperDocumentRead(
                intake_id=intake.intake_id,
                retained_event_id=retained["event_id"],
                document_id="main",
                name="paper.txt",
                text_sha256=text_digest,
                characters=len(text),
            ),
            retained["event_id"],
            text.encode(),
        )
        forged_digest = "0" * 64

        def recorded(files):
            return PaperRecorded(
                intake_id=intake.intake_id,
                paper_id=paper_id([f.sha256 for f in files]),
                origin="file",
                paper_file=str(path),
                files=files,
                documents=(
                    RecordedDocument(
                        document_id="main",
                        name="paper.txt",
                        sha256=digest,
                        text_sha256=text_digest,
                        read_event_id=read["event_id"],
                    ),
                ),
            )

        written = len(self.journey.read())
        for files in (
            (
                RecordedFile(
                    retained_event_id=retained["event_id"], sha256=forged_digest
                ),
            ),
            (),
        ):
            with self.subTest(files=files):
                with self.assertRaisesRegex(
                    ValueError, "other files than its intake kept|at least 1"
                ):
                    write(recorded(files), read["event_id"])
        self.assertEqual(len(self.journey.read()), written)
        # The true inventory is admitted.
        write(
            recorded(
                (RecordedFile(retained_event_id=retained["event_id"], sha256=digest),)
            ),
            read["event_id"],
        )
        self.assertEqual(len(self.papers().papers), 1)

    def test_a_recorded_paper_reads_no_document_from_a_refused_attempt(self):
        """#420 ST1 (r2): after a refusal and a retry, the paper's documents must be
        read from the files this recording kept, not from the refused attempt's."""
        from bio_neb.paper_intake.intake import refuse_paper
        from bio_neb.paper_intake.journey import (
            PaperDocumentRead,
            PaperDocumentRetained,
            PaperRecorded,
            RecordedDocument,
            RecordedFile,
        )

        path = self.root / "paper.txt"
        self.supply(path)
        (intake,) = self.papers().intakes
        head = [intake.source_event_id]

        def write(fact, artifact=None):
            payload = fact.model_dump(mode="json", exclude={"event_type"})
            event = (
                self.journey.record(fact.event_type, payload, head)
                if artifact is None
                else self.journey.record_with_artifact(
                    fact.event_type, payload, artifact, "text/plain", head
                )
            )
            head[:] = [event["event_id"]]
            return event

        def attempt(content: bytes):
            digest = hashlib.sha256(content).hexdigest()
            retained = write(
                PaperDocumentRetained(
                    intake_id=intake.intake_id,
                    name="paper.txt",
                    role="supplied",
                    media_type="text/plain",
                    location=str(path),
                    sha256=digest,
                    byte_count=len(content),
                ),
                content,
            )
            read = write(
                PaperDocumentRead(
                    intake_id=intake.intake_id,
                    retained_event_id=retained["event_id"],
                    document_id="main",
                    name="paper.txt",
                    text_sha256=digest,
                    characters=len(content),
                ),
                content,
            )
            return retained, read, digest

        _, old_read, old_digest = attempt(b"the refused attempt's text")
        refuse_paper(
            write,
            intake_id=intake.intake_id,
            origin="file",
            code="paper_file_unreadable",
            detail="refused for the test",
            paper_file=str(path),
        )
        (request,) = self.papers().requests
        answered = self.answer(request, "retry")
        head[:] = [answered["event_id"]]
        new_retained, _, new_digest = attempt(b"the current attempt's text")
        stale = PaperRecorded(
            intake_id=intake.intake_id,
            paper_id=paper_id([new_digest]),
            origin="file",
            paper_file=str(path),
            files=(
                RecordedFile(
                    retained_event_id=new_retained["event_id"], sha256=new_digest
                ),
            ),
            documents=(
                RecordedDocument(
                    document_id="main",
                    name="paper.txt",
                    sha256=old_digest,
                    text_sha256=old_digest,
                    read_event_id=old_read["event_id"],
                ),
            ),
        )
        written = len(self.journey.read())

        with self.assertRaisesRegex(ValueError, "names a document it did not read"):
            write(stale)
        self.assertEqual(len(self.journey.read()), written)

    def test_a_whitespace_only_file_is_unreadable_and_asks_its_review(self):
        """#420 SP5: a file whose text is only whitespace holds no text."""
        path = self.root / "blank.txt"
        path.write_bytes(b" \n\t\n")
        self.supply(path)

        take_supplied_papers(self.journey)

        (request,) = self.papers().requests
        self.assertEqual(request.refusal.code, "paper_file_unreadable")
        self.assertEqual(self.papers().papers, ())

    def test_a_malformed_docx_is_unreadable_and_asks_its_review(self):
        """#420 SP4: a supported format whose parser fails is an intake refusal, not
        an application fault."""
        path = self.root / "paper.docx"
        path.write_bytes(zipped({"word/document.xml": b"<w:document><unclosed"}))
        self.supply(path)

        take_supplied_papers(self.journey)

        (request,) = self.papers().requests
        self.assertEqual(request.refusal.code, "paper_file_unreadable")

    def test_a_reopened_journey_checks_each_document_against_its_bytes(self):
        path = self.root / "paper.txt"
        path.write_text("rs4680 COMT")
        self.supply(path)
        take_supplied_papers(self.journey)

        reopened = Journey.open(self.journey.path)

        self.assertEqual(
            reopened.checked_view("run_papers").papers, self.papers().papers
        )


class PaperIdTests(unittest.TestCase):
    def test_paper_id_is_the_digest_of_the_sorted_file_digests_one_per_line(self):
        b, a = "b" * 64, "a" * 64
        expected = "paper-" + hashlib.sha256(f"{a}\n{b}".encode()).hexdigest()

        self.assertEqual(paper_id([b, a]), expected)
        self.assertNotEqual(
            paper_id([b, a]),
            "paper-" + hashlib.sha256(f"{b}\n{a}".encode()).hexdigest(),
        )
        self.assertNotEqual(
            paper_id([b, a]), "paper-" + hashlib.sha256(f"{a}{b}".encode()).hexdigest()
        )


class SuppliedPaperRunTests(unittest.TestCase):
    """A Run given a paper file reads it after the FASTQs' inspection, before planning."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root))

    def begin(self, paper: Path):
        from bio_neb.application.advancement import ResumeRun, advance
        from tests.application_fakes import _registered

        receipt, paths = _registered(self.root, paper_files=(paper,))
        return receipt, paths, advance(ResumeRun(journey_file=receipt.journey_file))

    def answer(self, result, action, composition=None, **more):
        from bio_neb.application.advancement import (
            AnswerReview,
            PaperReviewAnswer,
            advance,
            use_run_advancement_composition,
        )

        request = result.current_review.request
        intent = AnswerReview(
            journey_file=result.journey_file,
            request_event_id=request.request_event_id,
            answer=PaperReviewAnswer(
                response=ReviewAnswer(
                    request_id=request.request.request_id,
                    action=action,
                    by="researcher",
                    **more,
                )
            ),
        )
        if composition is None:
            return advance(intent)
        with use_run_advancement_composition(composition):
            return advance(intent)

    def test_an_unreadable_paper_asks_its_review_before_any_model_contact(self):
        paper = self.root / "scan.pdf"
        paper.write_bytes(b"%PDF-1.4")

        _, _, result = self.begin(paper)

        self.assertEqual(result.status, "paper_review_pending")
        self.assertEqual(result.current_review.kind, "paper")
        self.assertEqual(
            [a.action for a in result.current_review.legal_actions],
            ["retry", "amend", "continue", "abort"],
        )
        self.assertEqual(result.inspection.result.status, "succeeded")
        self.assertEqual(len(result.attempts), 0)

    def test_an_amended_paper_is_recorded_before_planning_s_first_attempt(self):
        from tests.planning_assignment_fakes import PlanningComposition

        missing = self.root / "missing.pdf"
        receipt, paths, pending = self.begin(missing)
        found = self.root / "found.txt"
        found.write_text("rs1234 is reported in this paper.")

        result = self.answer(
            pending,
            "amend",
            PlanningComposition(receipt, paths),
            field="paper_file",
            value=str(found),
        )

        self.assertEqual(result.status, "planning_hitl_pending")
        (paper,) = result.run_papers.papers
        self.assertEqual(paper.paper_file, str(found))
        types = [
            e["event_type"] for e in Journey.open(Path(receipt.journey_file)).read()
        ]
        self.assertLess(
            types.index("InputMetadataInspected"), types.index("PaperRecorded")
        )
        self.assertLess(types.index("PaperRecorded"), types.index("AttemptStarted"))
        # A supplied paper is no citation: the plan declares no cited_paper, and the
        # Run never enters the paper stage.
        events = Journey.open(Path(receipt.journey_file)).read()
        stages = [
            e["payload"]["stage"]
            for e in events
            if e["event_type"] == "ModelRequestRecorded"
        ]
        self.assertIn("reference", stages)
        self.assertNotIn("paper", stages)

    def test_continue_goes_on_to_planning_without_the_paper(self):
        from tests.planning_assignment_fakes import PlanningComposition

        receipt, paths, pending = self.begin(self.root / "missing.pdf")

        result = self.answer(pending, "continue", PlanningComposition(receipt, paths))

        self.assertEqual(result.status, "planning_hitl_pending")
        self.assertEqual(result.run_papers.papers, ())
        self.assertEqual(result.run_papers.intakes[0].state, "dropped")

    def test_abort_ends_the_run_as_the_researcher_s_rejection(self):
        receipt, _, pending = self.begin(self.root / "missing.pdf")

        result = self.answer(pending, "abort")

        events = Journey.open(Path(receipt.journey_file)).read()
        (failed,) = [e for e in events if e["event_type"] == "RunFailed"]
        self.assertEqual(failed["payload"]["code"], "rejected_by_user")
        self.assertEqual(
            failed["payload"]["source_event_id"],
            next(e for e in events if e["event_type"] == "PaperReviewAnswered")[
                "event_id"
            ],
        )
        self.assertNotIn(
            result.status, {"paper_review_pending", "paper_intake_pending"}
        )
        self.assertEqual(len(result.attempts), 0)


if __name__ == "__main__":
    unittest.main()
