"""The planning model's Galaxy prebuilt-index Tools (#464; ADR 0049: code copies, the
model decides, code checks membership).

Once the genomic FASTA is verified, the model searches one of Galaxy's repositories
(`search_prebuilt_indexes`), reads a listed candidate's contigs (`read_prebuilt_index`)
and records the Run's index (`record_prebuilt_index_choice`), or none for the CPU build.
The genomic FASTA's work completes only on that choice, and an Execution Review revise
reopens it (owner #464 I2). These tests drive the Tools through a real planning
assignment whose only doubles are the model and the sources (Galaxy's datacache through
the product adapter's opener)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bio_neb.agentic_journey.action_policy import ProposedAgentAction
from bio_neb.application.advancement import (
    AnswerReview,
    ExecutionReviewAnswer,
    PlanningReviewAnswer,
    ResumeRun,
    advance,
    use_run_advancement_composition,
)
from bio_neb.application.planning_read import PrefixSnapshot
from bio_neb.journey import Journey
from bio_neb.parabricks.execution_proposal import ReviseExecutionResponse
from bio_neb.parabricks.planning_hitl import SelectPlanningHitlResponse
from bio_neb.recording import read_evidence
from bio_neb.reference_discovery.reference_tools import reference_tool_errors
from tests.planning_assignment_fakes import configured_planning_run
from tests.prebuilt_index_support import (
    GALAXY_ACCESSION,
    GALAXY_FASTA,
    GALAXY_INDEX,
    GalaxyOpener,
    offline_galaxy_files,
    write_bundle,
)
from tests.reference_fakes import OFFLINE_ASSEMBLY_REPORT, ReferenceAdapter


def _to_execution_review(receipt, composition, answer=None):
    """Advance the Run (or answer its Review), answering each planning question with
    its first candidate, to its Execution Review."""
    review = advance(answer or ResumeRun(journey_file=receipt.journey_file))
    while (
        review.current_review is not None and review.current_review.kind == "planning"
    ):
        question = review.current_review.request
        review = advance(
            AnswerReview(
                journey_file=receipt.journey_file,
                request_event_id=question.request_event_id,
                answer=PlanningReviewAnswer(
                    kind="planning",
                    response=SelectPlanningHitlResponse(
                        action="select",
                        candidate_id=question.candidates[0].candidate_id,
                    ),
                ),
            )
        )
    if review.current_review is None or review.current_review.kind != "execution":
        raise AssertionError((review.status, review.current_review))
    return review


class _PlannedRun(unittest.TestCase):
    """One planning Run to its Execution Review over the served Galaxy files."""

    listed = True
    # The model checks Galaxy before it describes the genomic FASTA.
    index_first = False
    report: str | None = None
    # The Run's organism (its name, NCBI taxid, and the accession discovery lists) and
    # BRC's name of it; by default the planning fakes' human.
    organism: tuple[str, int, str] | None = None
    galaxy_name = "Human"

    @property
    def accession(self) -> str:
        return GALAXY_ACCESSION if self.organism is None else self.organism[2]

    def galaxy_files(self, root: Path) -> dict[str, bytes]:
        return offline_galaxy_files(root, accession=self.accession)

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.opener = GalaxyOpener(
            files=self.galaxy_files(self.root / "galaxy"),
            listed=self.listed,
            accession=self.accession,
            name=self.galaxy_name,
        )
        self.receipt, self.composition, _ = configured_planning_run(
            self.root / "run",
            galaxy=self.opener,
            **({} if self.organism is None else {"organism": self.organism[0]}),
        )
        if self.organism is not None:
            name, taxid, accession = self.organism
            self.composition.reference_adapter = ReferenceAdapter(
                scientific_name=name, ncbi_tax_id=taxid, accession=accession
            )
        # NCBI's report is answered and retained at the description, as NCBI's FTP
        # answers it.
        self.composition.companion_adapter.answers = True
        self.composition.index_first = self.index_first
        if self.report is not None:
            self.composition.companion_adapter.report = self.report
        with use_run_advancement_composition(self.composition):
            self.review = _to_execution_review(self.receipt, self.composition)
        self.events = read_evidence(Journey.open(Path(self.receipt.journey_file)))

    def of(self, event_type: str) -> list[dict]:
        return [e for e in self.events if e["event_type"] == event_type]

    def through(self, event_id: str) -> list[dict]:
        index = next(i for i, e in enumerate(self.events) if e["event_id"] == event_id)
        return self.events[: index + 1]

    def errors(self, tool: str, **arguments) -> tuple[str, ...]:
        """The domain refusals of a call on the Run's whole planning prefix."""
        events = self.events
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: events))
        return reference_tool_errors(deps, ProposedAgentAction(tool, "call", arguments))


class ExcludedSequencesTests(_PlannedRun):
    """Owner, #464 ("primey for the ref we are running"): the report's alt loci are
    outside the Run's reference; the Review's package names them for the CPU build
    and input storage's measure to drop."""

    report = (
        OFFLINE_ASSEMBLY_REPORT
        + "alt1\talt-scaffold\t1\tChromosome\tKI9.1\t=\tNT_9.1\tALT_REF_LOCI_1\t8\tna\n"
    )

    def test_the_package_names_the_reports_alt_loci_to_drop(self):
        reference = self.review.current_review.request.proposal.execution_package
        reference = reference.reference
        self.assertEqual(reference.excluded_sequences, ("NT_9.1",))
        self.assertEqual(
            [(c.name, c.length) for c in reference.reference_sequences],
            [("chr-offline", 16)],
        )


class ChosenIndexTests(_PlannedRun):
    def test_the_model_searches_reads_and_chooses_and_the_review_binds_it(self):
        [listed] = self.of("PrebuiltIndexesListed")
        [read] = self.of("PrebuiltIndexRead")
        [chosen] = self.of("PrebuiltIndexChosen")
        rows = listed["payload"]["result"]["rows"]
        self.assertEqual(
            [(row["value"], row["candidate_id"]) for row in rows],
            [(GALAXY_ACCESSION, f"galaxy:brc:{GALAXY_ACCESSION}")],
        )
        contigs = read["payload"]["result"]["contigs"]
        self.assertEqual(
            [(p["requested_name"], p["status"]) for p in contigs["requested"]],
            [("chr-offline", "paired")],
        )
        self.assertEqual(chosen["payload"]["read_event_id"], read["event_id"])

        reference = self.review.current_review.request.proposal.execution_package
        reference = reference.reference
        self.assertEqual(
            reference.prebuilt_index.candidate.candidate_id,
            f"galaxy:brc:{GALAXY_ACCESSION}",
        )
        self.assertEqual(
            [(c.name, c.length) for c in reference.reference_sequences],
            [("chr-offline", 16)],
        )
        # The Review binds what the choice bound, and the choice is in its evidence.
        self.assertEqual(
            reference.prebuilt_index.model_dump(mode="json"),
            chosen["payload"]["candidate"],
        )
        closure = {
            item.source_event_id
            for item in self.review.current_review.request.proposal.evidence_closure
        }
        self.assertIn(chosen["event_id"], closure)

    def test_the_read_pairs_against_the_description_s_retained_report(self):
        [read] = self.of("PrebuiltIndexRead")
        cited = read["payload"]["result"]["assembly_report_event_id"]
        [retrieved] = [e for e in self.events if e["event_id"] == cited]
        self.assertEqual(retrieved["event_type"], "ReferenceSourceRetrieved")
        self.assertTrue(
            retrieved["payload"]["request_identity"].endswith("_assembly_report.txt")
        )
        # NCBI's report is read once, at the description; the read GETs the .ann only.
        self.assertEqual(len(self.composition.companion_adapter.reports), 1)
        self.assertEqual(
            [a["request_identity"] for a in read["payload"]["source_answers"]],
            [f"{GALAXY_INDEX}.ann"],
        )

    def test_the_retained_report_is_read_through_the_journey_contract(self):
        """A Journey without its artifact reader is a broken port, not an absent
        report: it never authorizes a fresh NCBI GET (standards r1 ST4)."""
        from bio_neb.reference_discovery.reference_tools import _retained_report

        events = self.events
        grounding = PrefixSnapshot(events).companions.prebuilt_index.grounding
        partial = SimpleNamespace(journey=SimpleNamespace(read=lambda: events))
        with self.assertRaises(AttributeError):
            _retained_report(partial, grounding.assembly_accession)
        journey = Journey.open(Path(self.receipt.journey_file))
        event_id, body = _retained_report(
            SimpleNamespace(journey=journey), grounding.assembly_accession
        )
        [read] = self.of("PrebuiltIndexRead")
        self.assertEqual(
            event_id, read["payload"]["result"]["assembly_report_event_id"]
        )
        self.assertEqual(body, OFFLINE_ASSEMBLY_REPORT.encode())

    def test_nothing_large_is_downloaded_before_the_review(self):
        gets = [url for method, url in self.opener.sent if method == "GET"]
        self.assertNotIn(GALAXY_FASTA, gets)
        self.assertNotIn(f"{GALAXY_INDEX}.bwt", gets)
        [read] = self.of("PrebuiltIndexRead")
        heads = [url for method, url in self.opener.sent if method == "HEAD"]
        self.assertEqual(
            sorted(heads),
            sorted(f["url"] for f in read["payload"]["result"]["files"]),
        )

    def test_the_fasta_work_completes_only_on_the_choice(self):
        [chosen] = self.of("PrebuiltIndexChosen")
        before = PrefixSnapshot(self.through(chosen["event_id"])[:-1]).companions
        after = PrefixSnapshot(self.through(chosen["event_id"])).companions
        self.assertEqual(before.status, "reference_facts_progress")
        self.assertIsNone(before.prebuilt_index.choice)
        self.assertEqual(after.status, "reference_facts")
        self.assertEqual(
            after.prebuilt_index.choice.source_event_id, chosen["event_id"]
        )


class IndexBeforeFastaTests(_PlannedRun):
    """Owner, #459 (2026-10-30: "the idea for this entire process is to happen in the
    beginning of the run before we have the fast"): the index is the verified
    assembly's, so the model searches, reads and chooses before any companion is
    described; the read pairs against NCBI's report of the accession, fetched by the
    read itself, and the Review binds the choice as before."""

    index_first = True

    def test_the_index_is_chosen_before_the_fasta_is_described(self):
        [chosen] = self.of("PrebuiltIndexChosen")
        [read] = self.of("PrebuiltIndexRead")
        companions = self.of("CompanionObservationRecorded")
        position = {e["event_id"]: i for i, e in enumerate(self.events)}
        self.assertTrue(companions)
        self.assertLess(
            position[chosen["event_id"]],
            min(position[e["event_id"]] for e in companions),
        )
        self.assertEqual(
            chosen["payload"]["grounding"]["assembly_accession"], GALAXY_ACCESSION
        )
        # No description had retained the report: the read fetched it by accession.
        self.assertIsNone(read["payload"]["result"]["assembly_report_event_id"])
        self.assertIn(
            f"accession:{GALAXY_ACCESSION}", self.composition.companion_adapter.reports
        )
        reference = self.review.current_review.request.proposal.execution_package
        self.assertEqual(
            reference.reference.prebuilt_index.model_dump(mode="json"),
            chosen["payload"]["candidate"],
        )


class _SuggestedAssemblyRun(unittest.TestCase):
    """The researcher asks for GRCh37 (`GRCH37`), which NCBI lists as replaced by
    GRCh38.p14 (`GRCH38`); Galaxy BRC has an index of GRCh38 only, and the prompt
    states rs429358's allele chr19:45411941 T>C with no version (#459 D3/D4). The
    planning model suggests GRCh38; ``answer`` is the candidate the researcher picks."""

    answer: str
    # The researcher's words: by default the prompt's position; a scenario may also
    # name the requested accession itself.
    intent = "Also check chr19:45411941 T>C."

    def setUp(self) -> None:
        from tests.position_move_support import GRCH37, GRCH38, grch37_move_sources

        self.requested, self.newer = GRCH37, GRCH38
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        self.opener = GalaxyOpener(files=offline_galaxy_files(root / "galaxy"))
        self.receipt, self.composition, _ = configured_planning_run(
            root / "run",
            galaxy=self.opener,
            scientific_intent=self.intent,
            scientific_rsid=None,
            scientific_site={
                "contig": "chr19",
                "position": 45411941,
                "ref": "T",
                "alt": "C",
            },
        )
        self.composition.reference_adapter = ReferenceAdapter(
            accession=GRCH37, newer=GRCH38
        )
        self.composition.companion_adapter.answers = True
        self.composition.index_first = True
        self.composition.variant_sources = grch37_move_sources()
        with use_run_advancement_composition(self.composition):
            self.question = self._to_question()
            self.review = _to_execution_review(
                self.receipt,
                self.composition,
                AnswerReview(
                    journey_file=self.receipt.journey_file,
                    request_event_id=self.question.request_event_id,
                    answer=PlanningReviewAnswer(
                        kind="planning",
                        response=SelectPlanningHitlResponse(
                            action="select", candidate_id=f"assembly:{self.answer}"
                        ),
                    ),
                ),
            )
        self.events = read_evidence(Journey.open(Path(self.receipt.journey_file)))

    def _to_question(self):
        review = advance(ResumeRun(journey_file=self.receipt.journey_file))
        if review.current_review is None or review.current_review.kind != "planning":
            raise AssertionError((review.status, review.current_review))
        return review.current_review.request

    def of(self, event_type: str) -> list[dict]:
        return [e for e in self.events if e["event_type"] == event_type]


class UseTheNewerAssemblyTests(_SuggestedAssemblyRun):
    """Owner, #459 (2026-10-30): "we will do the entire run with this new reference
    and with the new index"; the known SNV is moved by NCBI's equivalence and checked
    on GRCh38's own bases before the researcher answers."""

    answer = "GCF_000001405.40"

    def test_the_question_offers_the_newer_assembly_with_the_target_moved(self):
        self.assertEqual(
            [c.candidate_id for c in self.question.candidates],
            [f"assembly:{self.newer}", f"assembly:{self.requested}"],
        )
        [suggested] = self.of("AssemblyWithIndexSuggested")
        [target] = suggested["payload"]["result"]["targets"]
        self.assertEqual(
            (target["outcome"], target["placed"]),
            (
                "moved",
                {
                    "contig": "NC_000019.10",
                    "position": 44908684,
                    "ref": "T",
                    "alt": "C",
                },
            ),
        )
        self.assertEqual(target["stated_assembly"]["name"], self.requested)
        self.assertIn("NC_000019.10:44908684", self.question.question)

    def test_each_get_of_the_check_is_retained_once(self):
        """#458 (each GET its own retained record) and live a2 F1: the known targets'
        check retains every GET once, as its source answered it."""
        [suggested] = self.of("AssemblyWithIndexSuggested")
        identities = [
            a["request_identity"] for a in suggested["payload"]["source_answers"]
        ]
        self.assertTrue(identities)
        self.assertEqual(len(identities), len(set(identities)))

    def test_the_coordinator_asks_with_no_model_turn(self):
        """Owner, #459 D4: the suggestion Review is "opened with no model turn": the
        Attempt ends at the suggestion and the coordinator records the question."""
        [suggested] = self.of("AssemblyWithIndexSuggested")
        [asked] = [
            e
            for e in self.of("PlanningHitlRequested")
            if suggested["event_id"] in e["payload"]["evidence_event_ids"]
        ]
        self.assertEqual(asked["payload"]["owner"], "assembly_suggestion")
        between = self.events[self.events.index(suggested) : self.events.index(asked)]
        self.assertNotIn("ModelRequestRecorded", [e["event_type"] for e in between])
        [checkpoint] = [
            e for e in between if e["event_type"] == "SliceCheckpointReached"
        ]
        self.assertEqual(
            checkpoint["payload"]["assembly_suggestion_event_id"], suggested["event_id"]
        )

    def test_the_researcher_s_index_binds_the_run_s_choice(self):
        """Owner, #459 D4 ("use Y for the whole Run"): while the answer stands, the
        index choice on Y is the chosen index; the CPU build is refused."""
        [chosen] = self.of("PrebuiltIndexChosen")
        events = self.events[: self.events.index(chosen)]
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: events))
        self.assertEqual(
            reference_tool_errors(
                deps,
                ProposedAgentAction(
                    "record_prebuilt_index_choice",
                    "call",
                    {"index": "none", "reason": "Build it on CPU."},
                ),
            ),
            ("choice_not_the_researchers_index",),
        )

    def test_an_execution_revise_reopens_the_researcher_s_index(self):
        """#464 I2 and review r2 SP3-R2: the researcher's later Execution revise
        reopens the index choice; the Run stays on the chosen assembly, and the CPU
        build of it is admitted."""
        from bio_neb.application.planning_read import current_planning
        from bio_neb.parabricks.planning_commands import (
            RespondExecutionHitlCommand,
            respond_execution_hitl,
        )

        request = self.review.current_review.request
        respond_execution_hitl(
            RespondExecutionHitlCommand(
                journey_file=self.receipt.journey_file,
                request_event_id=request.request_event_id,
                response=ReviseExecutionResponse(
                    action="revise",
                    proposal_id=request.proposal.proposal_id,
                    instruction="Keep this reference but build its index on CPU.",
                    revalidation_scope="execution_planning",
                ),
            )
        )
        journey = Journey.open(Path(self.receipt.journey_file))
        planning = current_planning(journey)
        self.assertIsNotNone(planning.companions.prebuilt_index.reopened_by_event_id)
        self.assertEqual(planning.reference.requested_assembly, self.requested)
        self.assertEqual(
            reference_tool_errors(
                SimpleNamespace(journey=journey),
                ProposedAgentAction(
                    "record_prebuilt_index_choice",
                    "call",
                    {"index": "none", "reason": "The researcher revised it to CPU."},
                ),
            ),
            (),
        )

    def test_the_whole_run_is_on_the_newer_assembly_with_its_galaxy_index(self):
        proposal = self.review.current_review.request.proposal
        reference = proposal.execution_package
        self.assertEqual(reference.reference.assembly_accession, self.newer)
        self.assertEqual(
            reference.reference.prebuilt_index.candidate.candidate_id,
            f"galaxy:brc:{self.newer}",
        )
        # The Execution Review states the assembly the researcher asked for.
        self.assertEqual(reference.reference.requested_assembly, self.requested)
        self.assertEqual(proposal.reference_basis.requested_assembly, self.requested)


class MovedTargetTests(_SuggestedAssemblyRun):
    """Owner, #459 D3 (2026-10-30): the SNV moved before Parabricks keeps its move as
    its placement, stated on the requested assembly; the Investigation reads it on
    GRCh38 and does not move it again (one lookup, #462 R2)."""

    answer = "GCF_000001405.40"

    def test_the_moved_target_reaches_the_investigation_placed(self):
        from bio_neb.investigation.investigation_request import (
            PlacedPosition,
            current_investigation_request,
        )

        planned = current_investigation_request(
            Journey.open(Path(self.receipt.journey_file))
        )
        [item] = [i for i in planned.request.items if i.position is not None]
        self.assertEqual(item.stated_assembly.name, self.requested)
        self.assertEqual(item.placement.step, "database_equivalence")
        self.assertEqual(
            item.placed,
            PlacedPosition(contig="NC_000019.10", position=44908684, ref="T", alt="C"),
        )

    def test_the_investigation_reads_the_placed_position_without_moving_it(self):
        from bio_neb.investigation.investigation_request import (
            current_investigation_request,
            effective_location,
        )
        from bio_neb.investigation.investigation_sources import (
            look_up_stated_position,
        )
        from tests.position_move_support import grch37_move_sources

        planned = current_investigation_request(
            Journey.open(Path(self.receipt.journey_file))
        )
        [item] = [i for i in planned.request.items if i.position is not None]
        location = effective_location(item, ())
        sources = grch37_move_sources()

        def fetch(operation, key):
            from bio_neb.investigation.investigation import RetrievedSource
            from bio_neb.providers.eva_investigation import fetch_source

            contact = fetch_source(sources, operation, key)
            return (
                RetrievedSource.create(
                    operation=operation,
                    request_identity=contact.request_identity,
                    status_code=contact.status_code,
                    observed_at=contact.observed_at,
                    elapsed_time_ms=contact.elapsed_time_ms,
                    raw_response=contact.body,
                ),
                contact.body,
            )

        result = look_up_stated_position(
            position=location.stated,
            placed=location.placed,
            stated_assembly=location.stated_assembly,
            rsid=None,
            requested_rsid=None,
            assembly_accession=self.newer,
            fetch=fetch,
            literal=True,
            source="ncbi_dbsnp",
        )
        self.assertEqual(result.status, "resolved")
        self.assertNotIn(
            "ncbi_spdi_equivalents",
            {operation.split(":")[0] for operation in sources.operations},
        )


class SwitchedMethodTests(_SuggestedAssemblyRun):
    """Owner, #459 D4 (2026-10-30): a Run the researcher switched to the suggested
    assembly, approved and executed, states the assembly they asked for in its method
    and its report's reference line, through the actual Execution and method seams."""

    answer = "GCF_000001405.40"

    def test_the_executed_method_states_the_requested_assembly(self):
        from bio_neb.parabricks.plan import approved_execution_input
        from bio_neb.parabricks.planning_commands import (
            RespondExecutionHitlCommand,
            respond_execution_hitl,
        )
        from bio_neb.reporting.finalization import run_method
        from bio_neb.reporting.readable_report import _method_lines
        from tests.execution_approval import approve_execution
        from tests.h100_stage_fakes import h100_provider
        from tests.parabricks_operation_fakes import (
            execute_and_read_path,
            input_storage_provider,
        )
        from tests.reference_fakes import SOURCE_GZIP

        request = self.review.current_review.request
        respond_execution_hitl(
            RespondExecutionHitlCommand(
                journey_file=self.receipt.journey_file,
                request_event_id=request.request_event_id,
                response=approve_execution(request.proposal),
            )
        )
        approved = approved_execution_input(
            Journey.open(Path(self.receipt.journey_file))
        )

        def factory(plan, journal):
            provider = input_storage_provider(plan, journal)
            provider.sources.update(self.opener.bodies)
            provider.sources[plan.reference.source_url] = SOURCE_GZIP
            return h100_provider(provider, plan, approved)

        state = execute_and_read_path(approved, infrastructure_factory=factory)
        self.assertEqual(state.disposition, "succeeded")
        method = run_method(Journey.open(Path(self.receipt.journey_file)).read())
        self.assertEqual(
            (method.reference_accession, method.requested_reference_accession),
            (self.newer, self.requested),
        )
        self.assertIn(
            f"(the researcher asked for {self.requested} and chose this assembly, "
            "whose Galaxy index fits it, for the whole Run; each target's placement "
            "on it is in its own row)",
            _method_lines(SimpleNamespace(method=method))[1],
        )


class NamedOlderAccessionTests(_SuggestedAssemblyRun):
    """Owner, #459 after live a1 (2026-10-30): "a user will say an old assembly, and we
    will test if the run will sort in Galaxy a new assembly"; "we need to see that the
    original old reference is real and fitting … if index exists, it will supersede
    the decision". The researcher names GRCh37's accession; it is recorded and verified
    first, Galaxy has no index of it, the model lists the organism's assemblies during
    the index work with GRCh37 still verified, and the suggestion of GRCh38 reaches the
    Execution Review."""

    answer = "GCF_000001405.40"
    intent = "Use the reference GCF_000001405.25. Also check chr19:45411941 T>C."

    def test_the_named_reference_is_verified_before_the_newer_one_is_found(self):
        [supplied, *rest] = self.of("ReferenceChoiceRecorded")
        self.assertEqual(
            (
                supplied["payload"]["assembly_accession"],
                supplied["payload"]["discovery_event_id"],
            ),
            (self.requested, None),
        )
        position = {e["event_id"]: i for i, e in enumerate(self.events)}
        [first_description, *_] = self.of("ReferenceDescriptionRecorded")
        pages = self.of("ReferenceAssembliesRecorded")
        [listing, *_] = self.of("PrebuiltIndexesListed")
        page = next(
            p for p in pages if position[p["event_id"]] > position[listing["event_id"]]
        )
        self.assertLess(
            position[first_description["event_id"]], position[page["event_id"]]
        )
        # The page read during the index work keeps the requested assembly verified.
        from bio_neb.application.planning_read import PrefixSnapshot

        through = self.events[: position[page["event_id"]] + 1]
        reference = PrefixSnapshot(through).reference
        self.assertEqual(
            reference.verified_reference.assembly_accession, self.requested
        )

    def test_the_execution_review_shows_the_newer_assembly_and_the_requested_one(self):
        proposal = self.review.current_review.request.proposal
        reference = proposal.execution_package.reference
        self.assertEqual(
            (reference.assembly_accession, reference.requested_assembly),
            (self.newer, self.requested),
        )
        self.assertEqual(
            reference.prebuilt_index.candidate.candidate_id, f"galaxy:brc:{self.newer}"
        )


class BuildTheRequestedAssemblyTests(_SuggestedAssemblyRun):
    """Owner, #459: "If none exist, we continue with the CPU"; the researcher may also
    keep the requested assembly, whose index the CPU job builds."""

    answer = "GCF_000001405.25"

    def test_the_run_stays_on_the_requested_assembly_and_builds_on_cpu(self):
        reference = self.review.current_review.request.proposal.execution_package
        self.assertEqual(reference.reference.assembly_accession, self.requested)
        self.assertIsNone(reference.reference.prebuilt_index)
        # No switch: the package's identity carries no requested assembly.
        self.assertNotIn(
            "requested_assembly", reference.model_dump(mode="json")["reference"]
        )
        [chosen] = self.of("PrebuiltIndexChosen")
        self.assertIsNone(chosen["payload"]["read_event_id"])


class NewerAssemblyReadTests(_SuggestedAssemblyRun):
    """Owner, #459 (2026-10-30): with no Galaxy index of the requested X, the model
    finds X's newer version Y and reads Y's index against Y's own NCBI report, never
    against X's ("newer references will be always different")."""

    answer = "GCF_000001405.40"

    def before_the_answer(self):
        """The Run's events through the question, as the model's calls met them."""
        [responded] = [
            e
            for e in self.of("PlanningHitlResponded")
            if e["payload"]["request_event_id"] == self.question.request_event_id
        ]
        return self.events[: self.events.index(responded)]

    def errors(self, tool, **arguments):
        events = self.before_the_answer()
        deps = SimpleNamespace(journey=SimpleNamespace(read=lambda: events))
        return reference_tool_errors(deps, ProposedAgentAction(tool, "call", arguments))

    def test_the_newer_version_s_index_is_read_against_its_own_report(self):
        before = self.before_the_answer()
        listings = [e for e in before if e["event_type"] == "PrebuiltIndexesListed"]
        self.assertEqual(
            [e["payload"]["tool_input"]["search_text"] for e in listings],
            [self.requested, self.newer],
        )
        [read] = [e for e in before if e["event_type"] == "PrebuiltIndexRead"]
        self.assertEqual(
            read["payload"]["tool_input"]["assembly_accession"], self.newer
        )
        self.assertEqual(
            read["payload"]["tool_input"]["grounding"]["assembly_accession"],
            self.requested,
        )
        reports = self.composition.companion_adapter.reports
        self.assertEqual(reports[0], f"accession:{self.newer}")
        self.assertNotIn(f"accession:{self.requested}", reports)

    def test_another_assembly_s_index_is_never_the_run_s_own_choice(self):
        """Owner, #459 D4: Y becomes the Run's reference only on the researcher's
        answer; the model cannot bind Y's index to a Run on X."""
        [read] = [
            e
            for e in self.before_the_answer()
            if e["event_type"] == "PrebuiltIndexRead"
        ]
        self.assertEqual(
            self.errors(
                "record_prebuilt_index_choice",
                index="chosen",
                read_event_id=read["event_id"],
                reason="Y's index fits.",
            ),
            ("index_of_another_assembly",),
        )

    def test_an_assembly_this_run_never_listed_cannot_be_paired(self):
        """ADR 0049 membership: the other assembly is a row of this Run's discovery."""
        [listing] = [
            e
            for e in self.before_the_answer()
            if e["event_type"] == "PrebuiltIndexesListed"
            and e["payload"]["tool_input"]["search_text"] == self.newer
        ]
        self.assertEqual(
            self.errors(
                "read_prebuilt_index",
                listing_event_id=listing["event_id"],
                value=self.newer,
                assembly_accession="GCF_000009999.1",
            ),
            ("index_assembly_not_discovered",),
        )


class CaenorhabditisTests(_PlannedRun):
    """A C. elegans request chooses its Galaxy BRC index through the three Tools:
    the model searches BRC for the verified accession, reads the listed row's
    contigs and chooses that read; the Execution Review binds it (#464's observable
    result, on the offline Galaxy fixture)."""

    organism = ("Caenorhabditis elegans", 6239, "GCF_000002985.6")
    galaxy_name = "C. elegans (WBcel235)"

    def test_the_request_chooses_its_brc_index(self):
        [listed] = self.of("PrebuiltIndexesListed")
        [read] = self.of("PrebuiltIndexRead")
        [chosen] = self.of("PrebuiltIndexChosen")
        self.assertEqual(
            (
                listed["payload"]["tool_input"]["repository"],
                listed["payload"]["tool_input"]["search_text"],
            ),
            ("brc", "GCF_000002985.6"),
        )
        self.assertEqual(
            [
                (row["value"], row["name"], row["candidate_id"])
                for row in listed["payload"]["result"]["rows"]
            ],
            [
                (
                    "GCF_000002985.6",
                    "C. elegans (WBcel235)",
                    "galaxy:brc:GCF_000002985.6",
                )
            ],
        )
        self.assertEqual(
            read["payload"]["tool_input"]["listing_event_id"], listed["event_id"]
        )
        self.assertTrue(
            all(
                pair["status"] == "paired"
                for pair in read["payload"]["result"]["contigs"]["requested"]
            )
        )
        self.assertEqual(chosen["payload"]["read_event_id"], read["event_id"])
        self.assertEqual(
            chosen["payload"]["grounding"]["assembly_accession"], "GCF_000002985.6"
        )
        reference = self.review.current_review.request.proposal.execution_package
        reference = reference.reference
        self.assertEqual(reference.assembly_accession, "GCF_000002985.6")
        self.assertEqual(
            reference.prebuilt_index.candidate.candidate_id,
            "galaxy:brc:GCF_000002985.6",
        )
        self.assertEqual(
            reference.prebuilt_index.model_dump(mode="json"),
            chosen["payload"]["candidate"],
        )


class MembershipRefusalTests(_PlannedRun):
    """A read names a candidate row of a search page of this Run's genomic FASTA, and
    a choice cites a read of it (ADR 0049: code checks membership; owner #464)."""

    def test_a_read_citing_no_search_page_of_the_run_is_refused(self):
        [read] = self.of("PrebuiltIndexRead")
        for listing in (read["event_id"], "ffffffff-ffff-ffff-ffff-ffffffffffff"):
            with self.subTest(listing=listing):
                self.assertEqual(
                    self.errors(
                        "read_prebuilt_index",
                        listing_event_id=listing,
                        value=GALAXY_ACCESSION,
                    ),
                    ("index_listing_unknown",),
                )

    def test_a_value_the_page_does_not_list_is_refused(self):
        [listed] = self.of("PrebuiltIndexesListed")
        self.assertEqual(
            self.errors(
                "read_prebuilt_index",
                listing_event_id=listed["event_id"],
                value="GCF_000001405.39",
            ),
            ("index_row_not_on_listing",),
        )
        self.assertEqual(
            self.errors(
                "read_prebuilt_index",
                listing_event_id=listed["event_id"],
                value=GALAXY_ACCESSION,
            ),
            (),
        )

    def test_a_choice_citing_no_read_of_the_run_is_refused(self):
        [listed] = self.of("PrebuiltIndexesListed")
        self.assertEqual(
            self.errors(
                "record_prebuilt_index_choice",
                index="chosen",
                read_event_id=listed["event_id"],
                reason="It is listed.",
            ),
            ("index_read_unknown",),
        )

    def test_a_choice_s_answer_and_its_read_agree(self):
        """A chosen index cites its read; no index cites none (standards r1 ST1)."""
        [read] = self.of("PrebuiltIndexRead")
        for index, read_event_id, refused in (
            ("chosen", None, ("prebuilt_choice_read_missing",)),
            ("none", read["event_id"], ("prebuilt_choice_read_unexpected",)),
            ("chosen", read["event_id"], ()),
            ("none", None, ()),
        ):
            with self.subTest(index=index, read=read_event_id):
                self.assertEqual(
                    self.errors(
                        "record_prebuilt_index_choice",
                        index=index,
                        read_event_id=read_event_id,
                        reason="Stated.",
                    ),
                    refused,
                )

    def test_a_choice_states_a_reason(self):
        """The reason takes the planning choices' one shape (`ChoiceReason`, as a
        reference choice's: ADR 0040's Tool arguments)."""
        [read] = self.of("PrebuiltIndexRead")
        for reason in ("", " padded ", "x" * 2001):
            with self.subTest(reason=reason[:8]):
                self.assertEqual(
                    self.errors(
                        "record_prebuilt_index_choice",
                        index="chosen",
                        read_event_id=read["event_id"],
                        reason=reason,
                    ),
                    ("prebuilt_choice_reason_invalid",),
                )


class MissingSequenceTests(_PlannedRun):
    """An index that lacks a requested sequence is refused at the choice: the
    Run's reference is every requested sequence at its length (owner #464)."""

    def galaxy_files(self, root):
        files = offline_galaxy_files(root)
        (root / "other").mkdir()
        other = write_bundle(
            root / "other" / f"{GALAXY_ACCESSION}.fa",
            (("chr-other", "", "ACGTACGTACGTACGT"),),
        )
        files[f"{GALAXY_INDEX}.ann"] = Path(f"{other}.ann").read_bytes()
        return files

    def test_a_read_that_lacks_a_requested_sequence_cannot_be_chosen(self):
        [read] = self.of("PrebuiltIndexRead")
        contigs = read["payload"]["result"]["contigs"]
        self.assertEqual(
            [(p["requested_name"], p["status"]) for p in contigs["requested"]],
            [("chr-offline", "missing")],
        )
        self.assertEqual(
            [(e["name"], e["ncbi_role"]) for e in contigs["extras"]],
            [("chr-other", None)],
        )
        self.assertEqual(
            self.errors(
                "record_prebuilt_index_choice",
                index="chosen",
                read_event_id=read["event_id"],
                reason="Closest.",
            ),
            ("index_lacks_requested_sequences",),
        )

    def test_none_records_no_index_and_the_run_plans_its_cpu_build(self):
        [chosen] = self.of("PrebuiltIndexChosen")
        # The model's answer names no index and cites no read (its null is an absent
        # value); the Run reopened it to reach this Review.
        [call] = [
            e["payload"]
            for e in self.events
            if e["event_type"] == "ToolCallValidated"
            and e["payload"]["tool_name"] == "record_prebuilt_index_choice"
        ]
        self.assertEqual(call["arguments"]["index"], "none")
        self.assertNotIn("read_event_id", call["arguments"])
        self.assertIsNone(chosen["payload"]["read_event_id"])
        self.assertIsNone(chosen["payload"]["candidate"])
        package = self.review.current_review.request.proposal.execution_package
        self.assertIsNone(package.reference.prebuilt_index)


class UnlistedTests(_PlannedRun):
    """A search that lists no candidate: the model chooses none, the CPU build."""

    listed = False

    def test_an_empty_page_is_galaxy_s_answer_and_none_is_chosen(self):
        [listed] = self.of("PrebuiltIndexesListed")
        result = listed["payload"]["result"]
        self.assertEqual(
            (result["status"], result["rows"], result["total_rows"]),
            ("observed", [], 0),
        )
        self.assertEqual(self.of("PrebuiltIndexRead"), [])
        [chosen] = self.of("PrebuiltIndexChosen")
        self.assertIsNone(chosen["payload"]["candidate"])


class RevisionTests(_PlannedRun):
    """The researcher amends the index choice at the Execution Review's revise: their
    words go back to the planning model, which chooses again (owner #464 I2)."""

    def test_a_revise_reopens_the_choice_and_the_new_proposal_binds_the_new_one(self):
        original = self.review.execution_hitl.pending_request
        with use_run_advancement_composition(self.composition):
            revised = _to_execution_review(
                self.receipt,
                self.composition,
                AnswerReview(
                    journey_file=self.receipt.journey_file,
                    request_event_id=original.request_event_id,
                    answer=ExecutionReviewAnswer(
                        kind="execution",
                        response=ReviseExecutionResponse(
                            action="revise",
                            proposal_id=original.proposal.proposal_id,
                            instruction="Check the Galaxy index again.",
                            revalidation_scope="execution_planning",
                        ),
                    ),
                ),
            )
        events = read_evidence(Journey.open(Path(self.receipt.journey_file)))
        [revision] = [
            e
            for e in events
            if e["event_type"] == "ExecutionHitlResponded"
            and e["payload"]["response"]["action"] == "revise"
        ]
        reopened = PrefixSnapshot(
            events[: events.index(revision) + 1]
        ).companions.prebuilt_index
        self.assertIsNone(reopened.choice)
        self.assertEqual(reopened.reopened_by_event_id, revision["event_id"])
        first, second = [e for e in events if e["event_type"] == "PrebuiltIndexChosen"]
        self.assertLess(events.index(first), events.index(revision))
        self.assertLess(events.index(revision), events.index(second))
        proposal = revised.execution_hitl.pending_request.proposal
        self.assertIn(
            second["event_id"],
            {item.source_event_id for item in proposal.evidence_closure},
        )
        self.assertEqual(
            proposal.execution_package.reference.prebuilt_index.model_dump(mode="json"),
            second["payload"]["candidate"],
        )


class ReopenedReferenceTests(unittest.TestCase):
    """Reopen checks every event an admitted call cites against the earlier Journey;
    no value of a singular reference is exempt (standards r1 ST1, ADR 0038)."""

    def admit(self, tool, arguments):
        from bio_neb.agentic_journey import planning_assignment
        from bio_neb.agentic_journey.turn_reopen import _validate_admitted_action

        _validate_admitted_action(
            ProposedAgentAction(tool, "call", arguments),
            role="planning_agent",
            reopen=planning_assignment.planning_assignment().reopen,
            exposed_tool_names={tool},
            prior_event_ids=set(),
            admitted_action_fingerprints=set(),
            bound_arguments={},
        )

    def test_an_unrelated_singular_reference_spelled_none_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cites an event not retained"):
            self.admit(
                "read_prebuilt_index",
                {"listing_event_id": "none", "value": GALAXY_ACCESSION},
            )

    def test_the_no_index_choice_cites_no_event(self):
        self.admit(
            "record_prebuilt_index_choice",
            {"index": "none", "read_event_id": None, "reason": "None fits."},
        )


class AnsweredStatusTests(unittest.TestCase):
    """Galaxy's own answer about a request (a client status with its body) reaches
    the model as the Tool's result, with no verdict on it (#458, ADR 0049)."""

    def test_a_catalogue_s_404_is_galaxy_s_answer_the_model_reads(self):
        import io
        import urllib.error
        import uuid
        from email.message import Message

        from bio_neb.agentic_journey.source_answers import collected_answers
        from bio_neb.agentic_journey.source_contacts import call_source
        from bio_neb.providers.galaxy_datacache import GalaxyDatacache
        from bio_neb.reference_discovery.prebuilt_index_tool import (
            PrebuiltIndexGrounding,
            SearchPrebuiltIndexesInput,
            search_catalogues,
        )
        from bio_neb.reference_discovery.reference_tools import _operational_result

        def not_found(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 404, "status", Message(), io.BytesIO(b"no such file")
            )

        tool_input = SearchPrebuiltIndexesInput(
            grounding=PrebuiltIndexGrounding(
                reference_event_id=str(uuid.uuid4()),
                assembly_accession=GALAXY_ACCESSION,
            ),
            repository="managed",
        )
        with collected_answers("search_prebuilt_indexes") as got:
            result = search_catalogues(
                tool_input,
                galaxy=GalaxyDatacache(opener=not_found),
                contact=lambda surface, call: call_source(surface, call),
            )

        self.assertEqual(
            (result.status, result.failure.code), ("failed", "source_answered")
        )
        self.assertEqual(
            [(a.status_code, a.body) for a in got], [(404, b"no such file")]
        )
        self.assertIsNone(
            _operational_result({"payload": {"result": result.model_dump(mode="json")}})
        )


if __name__ == "__main__":
    unittest.main()
