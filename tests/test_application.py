import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    RegistrationError,
    register_run,
)
from bio_neb.application.queries import (
    ReviewJourneyQuery,
    project_journey_events,
    review_journey,
)
from bio_neb.input_inspection.commands import (
    InspectRegisteredInputsCommand,
    inspect_registered_inputs,
)
from bio_neb.input_inspection.local_files import LocalFilesAdapter
from bio_neb.journey import Journey
from bio_neb.locations import LOCAL_STATE_OVERRIDE


def runtime_configuration(**extra):
    value = {
        "contract_version": "1.0",
        "heartbeat_interval_seconds": 30,
        "model_roster": {
            "qwen": {
                "provider": "nebius",
                "endpoint": "https://api.example.test",
                "model": "Qwen/Qwen3.5-397B-A17B",
                "credential_env": "NEBIUS_TOKEN",
            }
        },
        "agent_invocation": {"model_reference": "qwen"},
    }
    value.update(extra)
    return value


def paired_inputs(root: Path, roles=("read_1", "read_2")):
    """One supported forward/reverse FASTQ pair, written under ``root``."""
    inputs = []
    for index, role in enumerate(roles):
        path = root / f"sample-{index}.fastq.gz"
        path.write_bytes((f"input-{index}").encode())
        inputs.append(RegisterFastqInput(role=role, registered_path=str(path)))
    return tuple(inputs)


def register_fixture(root: Path, roles=("read_1", "read_2")):
    run_root = root / "runs"
    prompt = root / "prompt.txt"
    config = root / "runtime.json"
    prompt.write_bytes(b"exact scientific intent\r\nwith trailing newline\r\n")
    config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
    inputs = paired_inputs(root, roles)
    receipt = register_run(
        RegisterRunCommand(
            run_root=str(run_root),
            prompt_file=str(prompt),
            runtime_config_file=str(config),
            inputs=inputs,
        ),
        repository_root=Path.cwd(),
    )
    return receipt, prompt, config, inputs


def inspect_registered(receipt):
    after = inspect_registered_inputs(
        InspectRegisteredInputsCommand(journey_file=receipt.journey_file),
        adapter=LocalFilesAdapter(),
    )
    event = Journey.open(Path(receipt.journey_file)).read()[-1]
    return after.inspection.result, event


def inspect_missing_registered_inputs(root: Path):
    receipt, _, _, inputs = register_fixture(root)
    for item in inputs:
        Path(item.registered_path).unlink()
    failed = inspect_registered_inputs(
        InspectRegisteredInputsCommand(journey_file=receipt.journey_file)
    )
    return receipt, failed


class ApplicationTests(unittest.TestCase):
    def test_registration_infers_repository_boundary_for_internal_and_external_roots(
        self,
    ):
        repository = Path(__file__).resolve().parents[1]
        local_state = repository / ".bio-neb-local"
        # Pin this checkout's own local state, so the boundary holds even when the
        # checkout itself sits under another checkout's local state.
        self.enterContext(
            patch.dict(os.environ, {LOCAL_STATE_OVERRIDE: str(local_state)})
        )
        with tempfile.TemporaryDirectory(dir=repository.parent) as directory:
            external = Path(directory).resolve()
            prompt = external / "prompt.txt"
            config = external / "runtime.json"
            fastq = external / "reads.fastq"
            fastq2 = external / "reads-2.fastq"
            prompt.write_text("inspect these reads", encoding="utf-8")
            config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
            fastq.write_bytes(b"reads")
            fastq2.write_bytes(b"mates")
            command = RegisterRunCommand(
                run_root=str(external / "run"),
                prompt_file=str(prompt),
                runtime_config_file=str(config),
                inputs=(
                    RegisterFastqInput(role="read1", registered_path=str(fastq)),
                    RegisterFastqInput(role="read2", registered_path=str(fastq2)),
                ),
            )
            internal = repository / "registration-guard-must-not-create"
            self.addCleanup(shutil.rmtree, internal, ignore_errors=True)
            with self.assertRaises(RegistrationError) as raised:
                register_run(command.model_copy(update={"run_root": str(internal)}))
            self.assertEqual(raised.exception.code, "run_root_invalid")
            self.assertFalse(internal.exists())

            # The gitignored local-state directory is the one sanctioned run-root
            # location inside a checkout.
            if not local_state.exists():
                local_state.mkdir()
                self.addCleanup(local_state.rmdir)
            with tempfile.TemporaryDirectory(dir=local_state) as sanctioned:
                sanctioned_root = Path(sanctioned) / "run"
                sanctioned_receipt = register_run(
                    command.model_copy(update={"run_root": str(sanctioned_root)})
                )
                self.assertTrue(Path(sanctioned_receipt.journey_file).exists())

            receipt = register_run(command)
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )
            self.assertEqual(review.status, "inputs_registered")
            self.assertEqual(
                review.registered_input_set.run_root, str(external / "run")
            )

    def test_registration_rejects_invalid_heartbeat_before_journey_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt, config, fastq = (
                root / "prompt",
                root / "runtime.json",
                root / "reads",
            )
            prompt.write_text("prompt", encoding="utf-8")
            fastq.write_bytes(b"reads")
            for index, heartbeat in enumerate(
                (float("inf"), float("nan"), 0, -1, True, "30")
            ):
                with self.subTest(heartbeat=heartbeat):
                    config.write_text(
                        json.dumps(
                            runtime_configuration(heartbeat_interval_seconds=heartbeat)
                        ),
                        encoding="utf-8",
                    )
                    run_root = root / f"runs-{index}"
                    with self.assertRaises(RegistrationError) as refused:
                        register_run(
                            RegisterRunCommand(
                                run_root=str(run_root),
                                prompt_file=str(prompt),
                                runtime_config_file=str(config),
                                inputs=(
                                    RegisterFastqInput(
                                        role="READ", registered_path=str(fastq)
                                    ),
                                ),
                            ),
                            repository_root=Path.cwd(),
                        )
                    self.assertEqual(
                        refused.exception.code, "runtime_selections_invalid"
                    )
                    self.assertFalse(run_root.exists())

    def test_registration_retains_exact_prompt_paths_and_choices_for_a_supported_pair(
        self,
    ):
        for roles in (("read_1", "read_2"), ("forward_read", "reverse_read")):
            with (
                self.subTest(roles=roles),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                receipt, prompt, config, inputs = register_fixture(root, roles)
                review = review_journey(
                    ReviewJourneyQuery(journey_file=receipt.journey_file)
                )

                self.assertEqual(review.status, "inputs_registered")
                self.assertEqual(
                    review.request.prompt,
                    "exact scientific intent\r\nwith trailing newline\r\n",
                )
                self.assertEqual(review.request.prompt_file, str(prompt))
                self.assertEqual(
                    review.registered_input_set.run_root, str(root / "runs")
                )
                self.assertEqual(
                    review.registered_input_set.runtime_config_file, str(config)
                )
                self.assertEqual(
                    [
                        item.registered_path
                        for item in review.registered_input_set.inputs
                    ],
                    [item.registered_path for item in inputs],
                )
                self.assertEqual(
                    review.registered_input_set.runtime_selections.model_roster[
                        "qwen"
                    ].model,
                    "Qwen/Qwen3.5-397B-A17B",
                )
                self.assertEqual(
                    [item.event_type for item in review.lineage],
                    [
                        "JourneyStarted",
                        "ResearchRequestRegistered",
                        "RunInputsRegistered",
                    ],
                )
                self.assertEqual(
                    review.lineage[2].parent_event_ids,
                    (review.request.source_event_id,),
                )

    def test_lexical_paths_are_retained_but_resolved_external_root_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            # A temporary checkout with separate local state keeps the refused
            # in-checkout root, and any regression's residue, out of this checkout.
            repository = root / "repository"
            repository.mkdir()
            self.enterContext(
                patch.dict(
                    os.environ, {LOCAL_STATE_OVERRIDE: str(root / "local-state")}
                )
            )
            (root / "spare").mkdir()
            prompt = str(root / "spare" / ".." / "prompt.txt")
            config = str(root / "spare" / ".." / "runtime.json")
            input_path = str(root / "spare" / ".." / "sample.fastq")
            mate_path = str(root / "spare" / ".." / "sample-2.fastq")
            Path(prompt).write_bytes(b"prompt")
            Path(config).write_text(
                json.dumps(runtime_configuration()), encoding="utf-8"
            )
            Path(input_path).write_bytes(b"reads")
            Path(mate_path).write_bytes(b"mates")
            command = RegisterRunCommand(
                run_root=str(root / "spare" / ".." / "runs"),
                prompt_file=prompt,
                runtime_config_file=config,
                inputs=(
                    RegisterFastqInput(role="read_1", registered_path=input_path),
                    RegisterFastqInput(role="read_2", registered_path=mate_path),
                ),
            )

            receipt = register_run(command, repository_root=repository)
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )

            self.assertEqual(review.request.prompt_file, prompt)
            self.assertEqual(review.registered_input_set.run_root, command.run_root)
            self.assertEqual(
                review.registered_input_set.inputs[0].registered_path, input_path
            )
            with self.assertRaises(RegistrationError) as raised:
                register_run(
                    command.model_copy(update={"run_root": str(repository / "runs")}),
                    repository_root=repository,
                )
            self.assertEqual(raised.exception.code, "run_root_invalid")

    def test_registration_translates_invalid_runtime_selections_before_journey_creation(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt, config, fastq = (
                root / "prompt",
                root / "runtime.json",
                root / "reads",
            )
            prompt.write_bytes(b"prompt")
            fastq.write_bytes(b"reads")
            config.write_text(
                json.dumps(runtime_configuration(api_key="secret-value")),
                encoding="utf-8",
            )
            run_root = root / "runs"

            with self.assertRaises(RegistrationError) as raised:
                register_run(
                    RegisterRunCommand(
                        run_root=str(run_root),
                        prompt_file=str(prompt),
                        runtime_config_file=str(config),
                        inputs=(
                            RegisterFastqInput(role="READ", registered_path=str(fastq)),
                        ),
                    ),
                    repository_root=Path.cwd(),
                )

            self.assertEqual(raised.exception.code, "runtime_selections_invalid")
            self.assertFalse(run_root.exists())

    def test_registration_retains_the_researcher_snp_request_and_reopens_it_unchanged(
        self,
    ):
        from bio_neb.investigation.investigation_request import (
            InvestigationRequest,
            InvestigationRequestItem,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt, config, fastq = (
                root / "prompt",
                root / "runtime.json",
                root / "reads",
            )
            prompt.write_bytes(b"prompt")
            fastq.write_bytes(b"reads")
            config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
            request_file = root / "snp-request.json"
            request_bytes = (
                b'{"asserted_by": "researcher:owner", "requests": ['
                b'{"request": "Investigate rs429358 in HG001.",'
                b' "target": {"rsid": "rs429358"}},'
                b'{"request": "Investigate rs7412 in HG001.",'
                b' "target": {"rsid": "rs7412"}}]}\n'
            )
            request_file.write_bytes(request_bytes)

            receipt = register_run(
                RegisterRunCommand(
                    run_root=str(root / "runs"),
                    prompt_file=str(prompt),
                    runtime_config_file=str(config),
                    request_files=(str(request_file),),
                    inputs=paired_inputs(root),
                ),
                repository_root=Path.cwd(),
            )
            request_file.write_bytes(b"later local edits are not the captured request")

            reopened = Journey.open(Path(receipt.journey_file))
            (captured,) = reopened.checked_view("investigation_request")
            events = reopened.read()
            registration = next(
                e for e in events if e["event_type"] == "RunInputsRegistered"
            )
            fact = events[-1]
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )

        self.assertEqual(
            captured.request,
            InvestigationRequest(
                asserted_by="researcher:owner",
                items=(
                    InvestigationRequestItem(
                        request_text="Investigate rs429358 in HG001.",
                        rsid="rs429358",
                    ),
                    InvestigationRequestItem(
                        request_text="Investigate rs7412 in HG001.", rsid="rs7412"
                    ),
                ),
            ),
        )
        self.assertEqual(captured.request_file, str(request_file))
        self.assertEqual(reopened.read_artifact(captured.artifact), request_bytes)
        self.assertEqual(fact["event_type"], "InvestigationRequestRegistered")
        self.assertEqual(fact["event_id"], captured.source_event_id)
        self.assertEqual(fact["parent_ids"], [registration["event_id"]])
        self.assertEqual(review.status, "inputs_registered")
        self.assertEqual(review.investigation_request, captured)

    def test_several_request_files_are_each_kept_in_order_with_their_asserters(self):
        # A researcher's list beside a paper's /3; each file keeps its bytes,
        # asserter and order, and each target its own asserter.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt = root / "prompt.txt"
            prompt.write_text("Investigate HG001.", encoding="utf-8")
            config = root / "runtime.json"
            config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
            documents = {
                "list.json": {
                    "schema_version": "bio-neb.investigation-request/3",
                    "asserted_by": "researcher",
                    "requests": [
                        {"request": "rs429358", "target": {"rsid": "rs429358"}},
                        {"request": "rs7412", "target": {"rsid": "rs7412"}},
                    ],
                },
                "paper.json": {
                    "schema_version": "bio-neb.investigation-request/3",
                    "asserted_by": "paper:doi:10.1000/x",
                    "requests": [
                        {"request": "rs7412 again", "target": {"rsid": "rs7412"}},
                        {"request": "rs4149056", "target": {"rsid": "rs4149056"}},
                    ],
                },
            }
            files = []
            for name, document in documents.items():
                path = root / name
                path.write_text(json.dumps(document), encoding="utf-8")
                files.append(path)

            receipt = register_run(
                RegisterRunCommand(
                    run_root=str(root / "runs"),
                    prompt_file=str(prompt),
                    runtime_config_file=str(config),
                    request_files=tuple(str(path) for path in files),
                    inputs=paired_inputs(root),
                ),
                repository_root=Path.cwd(),
            )
            reopened = Journey.open(Path(receipt.journey_file))
            captured = reopened.checked_view("investigation_request")
            registration = next(
                e for e in reopened.read() if e["event_type"] == "RunInputsRegistered"
            )
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )

            self.assertEqual(
                [(c.request_file, c.request.asserted_by) for c in captured],
                [(str(files[0]), "researcher"), (str(files[1]), "paper:doi:10.1000/x")],
            )
            self.assertEqual(
                [reopened.read_artifact(c.artifact) for c in captured],
                [path.read_bytes() for path in files],
            )
            facts = [
                e
                for e in reopened.read()
                if e["event_type"] == "InvestigationRequestRegistered"
            ]
            self.assertEqual(
                [f["parent_ids"] for f in facts], [[registration["event_id"]]] * 2
            )
        # The files' targets in order, each once; the paper's own keeps its asserter.
        request = review.investigation_request.request
        self.assertEqual(
            [(i.rsid, i.asserted_by) for i in request.items],
            [
                ("rs429358", None),
                ("rs7412", None),
                ("rs4149056", "paper:doi:10.1000/x"),
            ],
        )
        self.assertEqual(request.asserted_by, "researcher")

    def test_distinct_alleles_of_one_rsid_stay_distinct_targets(self):
        # A shared rsID is no shared target; an exact repeat is one.
        def item(alt, rsid="rs123"):
            return {
                "request": f"{rsid} at 12:21178615 T>{alt}",
                "target": {
                    "rsid": rsid,
                    "position": {
                        "contig": "12",
                        "position": 21178615,
                        "ref": "T",
                        "alt": alt,
                    },
                },
            }

        documents = {
            "one.json": {
                "schema_version": "bio-neb.investigation-request/3",
                "asserted_by": "paper:doi:10.1000/a",
                "requests": [item("A"), item("C"), item("A")],
            },
            "two.json": {
                "schema_version": "bio-neb.investigation-request/3",
                "asserted_by": "researcher",
                "requests": [item("G"), item("C")],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt = root / "prompt.txt"
            prompt.write_text("Investigate HG001.", encoding="utf-8")
            config = root / "runtime.json"
            config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
            files = []
            for name, document in documents.items():
                path = root / name
                path.write_text(json.dumps(document), encoding="utf-8")
                files.append(str(path))
            receipt = register_run(
                RegisterRunCommand(
                    run_root=str(root / "runs"),
                    prompt_file=str(prompt),
                    runtime_config_file=str(config),
                    request_files=tuple(files),
                    inputs=paired_inputs(root),
                ),
                repository_root=Path.cwd(),
            )
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )

        self.assertEqual(
            [
                (i.position.alt, i.asserted_by)
                for i in review.investigation_request.request.items
            ],
            [("A", None), ("C", None), ("G", "researcher")],
        )

    def test_registration_without_a_request_captures_none(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _, _, _ = register_fixture(Path(directory).resolve())
            reopened = Journey.open(Path(receipt.journey_file))
            review = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            )

        self.assertEqual(reopened.checked_view("investigation_request"), ())
        self.assertIsNone(review.investigation_request)

    def test_invalid_snp_request_refuses_registration_before_journey_creation(self):
        cases = {
            "not_json": (
                b"rs429358",
                "investigation_request_invalid",
                ("request_file",),
            ),
            "rsid_not_in_request": (
                b'{"asserted_by": "researcher:owner",'
                b' "request": "Investigate APOE.", "target": {"rsid": "rs429358"}}',
                "investigation_request_missing_adjacent_rsid",
                ("request_file.requests[0].request",),
            ),
            "missing_file": (None, "investigation_request_invalid", ("request_file",)),
        }
        for name, (content, code, fields) in cases.items():
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                prompt, config, fastq = (
                    root / "prompt",
                    root / "runtime.json",
                    root / "reads",
                )
                prompt.write_bytes(b"prompt")
                fastq.write_bytes(b"reads")
                config.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
                request_file = root / "snp-request.json"
                if content is not None:
                    request_file.write_bytes(content)
                run_root = root / "runs"

                with self.assertRaises(RegistrationError) as raised:
                    register_run(
                        RegisterRunCommand(
                            run_root=str(run_root),
                            prompt_file=str(prompt),
                            runtime_config_file=str(config),
                            request_files=(str(request_file),),
                            inputs=(
                                RegisterFastqInput(
                                    role="READ", registered_path=str(fastq)
                                ),
                            ),
                        ),
                        repository_root=Path.cwd(),
                    )

                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.fields, fields)
                self.assertFalse(run_root.exists())

    def test_inspection_evidence_and_lineage_reopen_without_live_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, prompt, config, inputs = register_fixture(root)
            result, event = inspect_registered(receipt)
            original = review_journey(
                ReviewJourneyQuery(journey_file=receipt.journey_file)
            ).model_dump(mode="json")
            retained = json.dumps(Journey.open(Path(receipt.journey_file)).read())

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(len(result.observation.inputs), 2)
            self.assertEqual(original["status"], "inspection_succeeded")
            self.assertEqual(
                original["inspection"]["source_event_id"], event["event_id"]
            )
            self.assertEqual(
                original["inspection"]["parent_event_ids"],
                [original["registered_input_set"]["source_event_id"]],
            )
            self.assertNotIn("secret-value", retained)
            self.assertNotIn("NEBIUS_TOKEN=", retained)

            prompt.unlink()
            config.unlink()
            for item in inputs:
                Path(item.registered_path).unlink()
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path.cwd() / "src")
            # The provider and adapter modules the read path imports, by exact name.
            # A stored Run with Execution contacts and an Investigation also imports
            # the last five. Every other provider or adapter module is a live
            # dependency.
            secret_free = sorted(
                [
                    *(
                        f"bio_neb.providers.{name}"
                        for name in (
                            "job_observations",
                            "nebius_context",
                            "object_locations",
                            "runtime",
                            "nebius_common",
                            "ports",
                            "ncbi_investigation",
                            "ncbi_datasets",
                            # The source transport family and its registered
                            # operations, which those two now import.
                            "source_transport",
                            "source_operations",
                            # Three later secret-free modules the read path imports:
                            # the copied placement records the Investigation's facts
                            # hold, the assembly-report reader its lookups use, and
                            # the public EVA adapter whose fetch seam the Tool
                            # inventory imports, the same kind as ncbi_investigation.
                            # None holds a credential or opens a socket on import.
                            "variant_placements",
                            "ncbi_assembly_report",
                            "eva_investigation",
                            # The public paper adapter whose search and download the
                            # planning paper Tools import; no credential, no socket
                            # on import.
                            "paper_sources",
                        )
                    ),
                    "bio_neb.agentic_journey.adapters",
                ]
            )
            script = (
                f"SECRET_FREE = {secret_free!r}; "
                "import builtins,json,socket,sys; from bio_neb.journey import Journey; "
                "original=builtins.__import__; "
                "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(RuntimeError('live dependency import forbidden')) if name.startswith(('bio_neb.providers','bio_neb.agentic_journey.adapters')) and name not in SECRET_FREE else original(name,*a,**k); "
                "socket.socket=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network forbidden')); "
                "from bio_neb.application.queries import ReviewJourneyQuery,review_journey; "
                "print(json.dumps(review_journey(ReviewJourneyQuery(journey_file=sys.argv[1])).model_dump(mode='json'),sort_keys=True))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script, receipt.journey_file],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(json.loads(completed.stdout), original)

    def test_application_records_exact_inspection_failure_and_registration_parent(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, failed = inspect_missing_registered_inputs(root)

            self.assertEqual(failed.status, "input_review_pending")
            self.assertEqual(
                [failure.code for failure in failed.inspection.result.failures],
                ["input_missing", "input_missing"],
            )
            self.assertEqual(
                failed.inspection.parent_event_ids,
                (failed.registered_input_set.source_event_id,),
            )

    def test_query_rejects_inspection_facts_not_derived_from_retained_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = inspect_missing_registered_inputs(root)
            forged = Journey.open(Path(receipt.journey_file)).read()
            next(e for e in forged if e["event_type"] == "InputMetadataInspected")[
                "payload"
            ]["result"]["failures"][0]["code"] = "input_empty"
            with self.assertRaisesRegex(
                ValueError, "inspection facts do not derive from selected observation"
            ):
                project_journey_events(forged, journey_file=receipt.journey_file)

    def test_query_rejects_inspection_outside_registration_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = inspect_missing_registered_inputs(root)
            unrelated = Journey.open(Path(receipt.journey_file)).read()
            next(e for e in unrelated if e["event_type"] == "InputMetadataInspected")[
                "parent_ids"
            ] = [
                next(
                    event["event_id"]
                    for event in unrelated
                    if event["event_type"] == "ResearchRequestRegistered"
                )
            ]
            with self.assertRaisesRegex(ValueError, "outside selection lineage"):
                project_journey_events(unrelated, journey_file=receipt.journey_file)


class RetiredJourneyFactTests(unittest.TestCase):
    """#172 retired these facts with no compatibility reader (owner clean cut)."""

    RETIRED = (
        "ExecutionAuthorityGranted",
        "AgentActionReviewRequested",
        "AgentActionReviewResponded",
    )

    def test_review_refuses_a_journey_carrying_a_retired_fact(self):
        for event_type in self.RETIRED:
            with (
                self.subTest(event_type=event_type),
                tempfile.TemporaryDirectory() as directory,
            ):
                receipt, _, _, _ = register_fixture(Path(directory).resolve())
                journey = Journey.open(Path(receipt.journey_file))
                journey.record(event_type, {}, [journey.read()[-1]["event_id"]])

                with self.assertRaisesRegex(
                    ValueError, f"retired_journey_fact:{event_type}"
                ):
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    )


if __name__ == "__main__":
    unittest.main()
