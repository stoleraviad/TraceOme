"""Connected proof of retained Review and Run interaction helpers through the CLI."""

import asyncio
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx

import tests.run_interaction_cli_support as cli_support
from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration
from tests.proposal_information_fakes import runtime_configuration
from tests.run_prerequisites import RunPrerequisites
from tests.sequential_investigation_support import (
    SequentialAdaptiveModel,
    SequentialSourceResponses,
    _MalformedOutputModel,
)
from tests.terminal_journey_fakes import published_continuation


def helper_module():
    path = Path(cli_support.__file__)
    spec = importlib.util.spec_from_file_location("cli_proof_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewCliProofTests(RunPrerequisites, unittest.TestCase):
    explicit_composition = True

    def test_cli_child_begin_answer_and_duplicate_preserve_fresh_review(self):
        self.run_scenario(failed=False)

    def test_cli_zero_exit_begin_held_for_review_is_frozen_without_success_receipt(
        self,
    ):
        self.run_scenario(failed=True)

    def run_scenario(self, *, failed):
        helper = helper_module()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve()
            # The read-only executed parent is built once per class; each scenario
            # forks its own child from it.
            parent = self.prerequisite(
                "executed", investigation_request=helper.REQUEST
            ).journey_file
            packet = published_continuation(root, parent_file=parent)
            runtime = root / "runtime.json"
            runtime.write_text(
                RuntimeConfiguration.model_validate(
                    runtime_configuration()
                ).model_dump_json(),
                encoding="utf-8",
            )
            for name, value in {
                "PARENT": parent,
                "PACKET": packet,
                "PARENT_SHA": helper.identity(parent)["sha256"],
                "PACKET_SHA": helper.identity(packet)["sha256"],
            }.items():
                stack.enter_context(patch.object(helper, name, value))
            stack.enter_context(
                patch.dict(os.environ, {"LOCAL_MODEL_KEY": "offline-test"})
            )
            models, sources = [], []

            class OfflineModel:
                contacts = 0

                def __new__(cls, _loaded, _selected):
                    model = (
                        _MalformedOutputModel() if failed else SequentialAdaptiveModel()
                    )
                    models.append(model)
                    return model.model(_loaded, _selected)

            def offline_sources():
                source = SequentialSourceResponses()
                sources.append(source)
                return source

            stack.enter_context(
                patch.object(helper, "ContactCeilingModel", OfflineModel)
            )
            stack.enter_context(patch.object(helper, "sources", offline_sources))

            def phase(name, *arguments, directory=None):
                receipt = root / (directory or name) / "receipt.json"
                with redirect_stdout(io.StringIO()):
                    helper.main(
                        [
                            name,
                            "--inputs",
                            str(root / "inputs"),
                            "--receipt",
                            str(receipt),
                            *arguments,
                        ],
                        explicit_composition=self.explicit_composition,
                    )
                return json.loads(receipt.read_bytes())

            phase("preflight", "--runtime", str(runtime))
            donor_bytes = parent.read_bytes(), packet.read_bytes()
            child = phase(
                "child",
                "--child-root",
                str(root / "new-child"),
            )
            self.assertEqual(child["cli_calls"][0]["argv"][0], "fork")
            journey = next((root / "new-child").glob("*/journey.jsonl"))
            args = (
                "--child-root",
                str(root / "new-child"),
                "--journey-file",
                str(journey),
            )
            if failed:
                with self.assertRaisesRegex(ValueError, "fresh initial Review"):
                    phase("begin", *args)
                self.assertFalse((root / "begin/receipt.json").exists())
                self.assertTrue((root / "begin/receipt.started.json").exists())
                cli = json.loads((root / "begin/cli-1.json").read_bytes())
                self.assertEqual(cli["argv"][0], "investigation-begin")
                self.assertEqual(cli["exit_code"], 0)
                # Malformed output spends its repair allowance and stops for the
                # Agent work Review, not a failed Investigation.
                self.assertEqual(cli["output"]["status"], "agent_work_review_pending")
                before = journey.read_bytes()
                with self.assertRaises(FileExistsError):
                    phase("begin", *args)
                self.assertEqual(journey.read_bytes(), before)
                self.assertEqual(len(models), 1)
            else:
                begin = phase("begin", *args)
                request_id = begin["investigation"]["pending_request"][
                    "request_event_id"
                ]
                frozen = journey.read_bytes()
                with self.assertRaisesRegex(ValueError, "fresh unapproved"):
                    phase(
                        "revise",
                        *args,
                        "--request-event-id",
                        "wrong-request",
                        directory="stale",
                    )
                self.assertEqual(journey.read_bytes(), frozen)
                self.assertEqual(len(models), 1)
                revised = phase("revise", *args, "--request-event-id", request_id)
                self.assertEqual(
                    [call["argv"][0] for call in revised["cli_calls"]],
                    ["observe", "answer", "observe", "answer", "observe", "observe"],
                )
                self.assertNotEqual(
                    revised["observation"]["current_review"]["request_event_id"],
                    request_id,
                )
                self.assertTrue(revised["duplicate_contact_free"])
                self.assertEqual(revised["contact_accounting"]["attempts"], 2)
                self.assertEqual(len(models), 2)
                self.assertEqual([source.contact_count for source in sources], [2, 2])
                self.assertFalse(revised["observation"]["investigation"]["authorities"])
            self.assertEqual((parent.read_bytes(), packet.read_bytes()), donor_bytes)


class RunInteractionCliProofTests(ReviewCliProofTests):
    explicit_composition = False

    def test_wrong_or_missing_interaction_schema_cannot_skip_comparison(self):
        helper = helper_module()
        for document in (None, {}, {"schema_version": "old"}):
            with (
                self.subTest(document=document),
                self.assertRaisesRegex(
                    ValueError, "expected current Run interaction schema"
                ),
            ):
                helper.verify_interaction(document, Path("/unopened/journey.jsonl"))

    def test_unexpected_failure_retains_physical_contact_accounting(self):
        from types import SimpleNamespace

        helper = helper_module()
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "failed.json"
            args = SimpleNamespace(phase="revise", receipt=receipt)
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with helper.retained_contact_accounting(args):
                    helper.ContactCeilingModel.contacts = 3
                    helper.SourceCeilingTransport.sends = [
                        "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358"
                    ]
                    raise RuntimeError("test failure")
            accounting = json.loads(receipt.with_suffix(".contacts.json").read_bytes())
            self.assertEqual(accounting["outcome"], "failed")
            self.assertEqual(accounting["phase_http_model_send_attempts"], 3)
            self.assertEqual(accounting["phase_http_source_send_attempts"], 1)
            self.assertFalse(receipt.exists())


class ContactCeilingProofTests(unittest.TestCase):
    """The proof helper's contact ceilings, independent of the composition variants
    above, so discovery runs each once (#229 COST-8)."""

    def test_model_send_ceiling_refuses_ninth_request_before_transport(self):
        helper = helper_module()
        model = object.__new__(helper.ContactCeilingModel)
        model.expected_url = "https://example.test/v1/chat/completions"
        request = httpx.Request("POST", model.expected_url)
        for _ in range(8):
            asyncio.run(model.count_send(request))
        with self.assertRaisesRegex(RuntimeError, "ceiling exhausted"):
            asyncio.run(model.count_send(request))
        self.assertEqual(helper.ContactCeilingModel.contacts, 8)
        with self.assertRaisesRegex(RuntimeError, "unselected model"):
            asyncio.run(model.count_send(httpx.Request("GET", model.expected_url)))

    def test_source_ceiling_refuses_extra_or_unselected_url_before_http(self):
        helper = helper_module()
        transport = helper.SourceCeilingTransport()
        allowed = [
            "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358",
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_000001405.40/dataset_report",
        ]

        def response(*_args, **_kwargs):
            body = io.BytesIO(b"{}")
            body.status = 200
            body.headers = {}
            return body

        with patch("urllib.request.OpenerDirector.open", side_effect=response) as http:
            with self.assertRaisesRegex(RuntimeError, "ceiling exceeded"):
                transport.get("https://api.ncbi.nlm.nih.gov/unselected", max_bytes=100)
            for url in allowed:
                self.assertEqual(transport.get(url, max_bytes=100).status_code, 200)
            with self.assertRaisesRegex(RuntimeError, "ceiling exceeded"):
                transport.get(allowed[0], max_bytes=100)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(helper.SourceCeilingTransport.sends, allowed)


if __name__ == "__main__":
    unittest.main()
