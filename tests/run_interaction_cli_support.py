"""Maintained CLI Review and Run interaction scenario support.

Extracted from frozen 272/273 proof helpers by ticket 282. Both explicit Review
composition and default Run continuation remain covered without loading archives.
Donor paths and identities are supplied by each temporary offline test scenario.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path

from bio_neb import cli as runner
from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
from bio_neb.agentic_journey.role_instructions import role_instructions
from bio_neb.agentic_journey.runtime_config import read_runtime_config
from bio_neb.application.advancement import (
    use_run_advancement_composition,
)
from bio_neb.application.queries import ObserveRun, observe
from bio_neb.application.run_interaction import run_interaction
from bio_neb.investigation.investigation_commands import (
    ContinueInvestigationCommand,
    continue_investigation,
)
from bio_neb.journey import canonical_json
from bio_neb.providers.ncbi_datasets import UrllibNcbiTransport
from bio_neb.providers.ncbi_investigation import (
    NcbiInvestigationSourceAdapter,
)
from bio_neb.providers.nebius_model_endpoint import (
    NebiusModelEndpointAdapter,
)

PARENT = None
PACKET = None
PARENT_SHA = None
PACKET_SHA = None


# The child's request; its executed parent registers it, so the parent's plan
# names it and the child begins from it.
REQUEST = {
    "schema_version": "bio-neb.investigation-request/2",
    "asserted_by": "researcher:owner",
    "requests": [
        {
            "request": "Investigate rs429358 in sample HG001.",
            "target": {"rsid": "rs429358"},
            "normalized_question_action": "observe_variant",
        }
    ],
}


def identity(path):
    body = path.read_bytes()
    return {
        "path": str(path),
        "byte_count": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(canonical_json(value))


def donors():
    parent, packet = identity(PARENT), identity(PACKET)
    if parent["sha256"] != PARENT_SHA or packet["sha256"] != PACKET_SHA:
        raise ValueError("frozen donor identity changed")
    return {"parent": parent, "packet": packet}


class ContactCeilingModel(NebiusModelEndpointAdapter):
    """Experiment-only ceiling; the ordinary Adapter still owns wire policy."""

    contacts = 0

    def __init__(self, loaded, selected):
        super().__init__(loaded, selected)
        self.expected_url = selected.endpoint.rstrip("/") + "/chat/completions"
        self.client._client.event_hooks["request"].append(self.count_send)

    async def count_send(self, request):
        if request.method != "POST" or str(request.url) != self.expected_url:
            raise RuntimeError("unselected model HTTP operation")
        if type(self).contacts >= 8:
            raise RuntimeError("selected eight-model-contact phase ceiling exhausted")
        type(self).contacts += 1


class SourceCeilingTransport(UrllibNcbiTransport):
    sends = []

    def get(self, url, *, max_bytes):
        allowed = {
            "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358",
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_000001405.40/dataset_report",
        }
        if url not in allowed or len(type(self).sends) >= 2:
            raise RuntimeError("selected source operation/phase ceiling exceeded")
        type(self).sends.append(url)
        return super().get(url, max_bytes=max_bytes)


def sources():
    return NcbiInvestigationSourceAdapter(transport=SourceCeilingTransport())


class RevisionComposition:
    def continue_investigation(self, observation):
        return continue_investigation(
            ContinueInvestigationCommand(journey_file=observation.journey_file),
            environ=os.environ,
            model_factory=ContactCeilingModel,
            source_adapter_factory=sources,
        )


def checked_inputs(directory):
    receipt = json.loads((directory / "inputs.json").read_bytes())
    for name in ("runtime", "request"):
        item = receipt[name]
        if identity(Path(item["path"])) != item:
            raise ValueError("selected input changed")
    if receipt["donors"] != donors():
        raise ValueError("selected donor changed")
    return receipt


def verify_interaction(document, journey_file):
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "bio-neb.run-interaction/1"
    ):
        raise ValueError("expected current Run interaction schema")
    before = identity(journey_file)
    contacts = ContactCeilingModel.contacts, tuple(SourceCeilingTransport.sends)
    direct = run_interaction(observe(ObserveRun(journey_file=str(journey_file))))
    if document != direct.model_dump(mode="json"):
        raise ValueError("CLI and direct Run interaction meaning differ")
    if before != identity(journey_file) or contacts != (
        ContactCeilingModel.contacts,
        tuple(SourceCeilingTransport.sends),
    ):
        raise ValueError("observation changed Journey or contact counts")


@contextmanager
def retained_contact_accounting(args):
    outcome = "failed"
    try:
        yield
        outcome = "succeeded"
    finally:
        save(
            args.receipt.with_suffix(".contacts.json"),
            {
                "phase": args.phase,
                "outcome": outcome,
                "phase_http_model_send_attempts": ContactCeilingModel.contacts,
                "phase_http_source_send_attempts": len(SourceCeilingTransport.sends),
                "phase_source_urls": SourceCeilingTransport.sends,
                "provider_creates": 0,
                "vcf_reads": 0,
                "limit": "Selected HTTP send attempts, including failed sends; not proof of server receipt.",
            },
        )


def _arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "child", "begin", "revise"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--child-root", type=Path)
    parser.add_argument("--journey-file", type=Path)
    parser.add_argument("--request-event-id")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "revise" and not args.request_event_id:
        parser.error("revise requires --request-event-id")
    return args


def main(argv=None, *, explicit_composition=False):
    args = _arguments(argv)
    cli_calls = []

    def cli(arguments, **dependencies):
        output = io.StringIO()
        if explicit_composition and arguments[0] == "answer":
            dependencies = {}
        with redirect_stdout(output):
            exit_code = runner.main(arguments, **dependencies)
        document = json.loads(output.getvalue()) if output.getvalue().strip() else None
        call = {"argv": arguments, "exit_code": exit_code, "output": document}
        save(args.receipt.parent / f"cli-{len(cli_calls) + 1}.json", call)
        cli_calls.append(call)
        print(canonical_json(call), flush=True)
        if exit_code != 0:
            raise ValueError("public runner CLI failed; phase remains frozen")
        if not explicit_composition and arguments[0] in {
            "investigation-begin",
            "answer",
        }:
            verify_interaction(document, args.journey_file)
        return document

    def observation():
        return cli(["observe", "--journey-file", str(args.journey_file), "--rich"])

    with nullcontext() if explicit_composition else retained_contact_accounting(args):
        if args.phase == "preflight":
            if args.inputs.exists():
                raise ValueError("fresh inputs directory required")
            runtime = read_runtime_config(args.runtime)
            policy = runtime.agent_invocation_policy(
                "investigation_planner", AgentInvocationOverrides()
            )
            save(args.inputs / "runtime.json", runtime.model_dump(mode="json"))
            save(args.inputs / "request.json", REQUEST)
            instructions = role_instructions("investigation_planner")
            receipt = {
                "donors": donors(),
                "runtime": identity(args.inputs / "runtime.json"),
                "request": identity(args.inputs / "request.json"),
                "effective_policy": policy.model_dump(mode="json"),
                "instruction_sha256": hashlib.sha256(instructions.content).hexdigest(),
                "limits": {
                    "attempts": 2,
                    "model_contacts_per_phase": 8,
                    "source_contacts_total": 4,
                    "provider_creates": 0,
                    "consequential_execution": 0,
                    "vcf_reads": 0,
                },
            }
            save(args.inputs / "inputs.json", receipt)
        else:
            inputs = checked_inputs(args.inputs)
            save(
                args.receipt.with_suffix(".started.json"),
                {"phase": args.phase, "inputs": identity(args.inputs / "inputs.json")},
            )
            if args.phase == "child":
                receipt = cli(
                    [
                        "fork",
                        "--parent-journey-file",
                        str(PARENT),
                        "--child-run-root",
                        str(args.child_root),
                    ]
                )
            elif args.phase == "begin":
                cli(
                    [
                        "investigation-begin",
                        "--journey-file",
                        str(args.journey_file),
                        "--request-file",
                        inputs["request"]["path"],
                    ],
                    investigation_model_factory=ContactCeilingModel,
                    investigation_source_adapter_factory=sources,
                )
                view = observation()
                result = view["investigation"]
                if (
                    result["pending_request"] is None
                    or len(result["pending_request"]["proposal"]["work_items"]) != 1
                    or view["current_review"] is None
                    or view["current_review"]["kind"] != "investigation"
                    or result["authorities"]
                ):
                    raise ValueError("one-member fresh initial Review not reached")
                receipt = {
                    "investigation": result,
                    "http_model_send_attempts": ContactCeilingModel.contacts,
                }
            else:
                view = observation()
                request = view["investigation"]["pending_request"]
                if (
                    request is None
                    or request["request_event_id"] != args.request_event_id
                    or len(request["proposal"]["work_items"]) != 1
                    or view["investigation"]["authorities"]
                ):
                    raise ValueError("fresh unapproved one-member Review required")
                item = request["proposal"]["work_items"][0]
                if "rs429358" not in item["target"]["aliases"]:
                    raise ValueError("selected scientific target changed")
                response = {
                    "action": "revise",
                    "proposal_id": request["proposal"]["proposal_id"],
                    "asserted_by": "researcher:owner",
                    "operations": [
                        {
                            "operation": "amend",
                            "work_item_id": item["work_item_id"],
                            "request": {
                                "rsid": "rs429358",
                                "request_text": "Recheck rs429358 in sample HG001 using current RefSNP and assembly evidence.",
                            },
                        }
                    ],
                }
                response_file = args.receipt.parent / "review-response.json"
                save(response_file, response)
                command = [
                    "answer",
                    "--journey-file",
                    str(args.journey_file),
                    "--request-event-id",
                    args.request_event_id,
                    "--response-file",
                    str(response_file),
                ]
                with (
                    use_run_advancement_composition(RevisionComposition())
                    if explicit_composition
                    else nullcontext()
                ):
                    cli(
                        command,
                        investigation_model_factory=ContactCeilingModel,
                        investigation_source_adapter_factory=sources,
                    )
                    result = observation()
                    frozen = identity(args.journey_file)
                    contacts = ContactCeilingModel.contacts
                    source_sends = tuple(SourceCeilingTransport.sends)
                    cli(
                        command,
                        investigation_model_factory=ContactCeilingModel,
                        investigation_source_adapter_factory=sources,
                    )
                    replay = observation()
                if (
                    result != replay
                    or identity(args.journey_file) != frozen
                    or contacts != ContactCeilingModel.contacts
                    or source_sends != tuple(SourceCeilingTransport.sends)
                ):
                    raise ValueError("duplicate Review repeated effects")
                if (
                    result["current_review"] is None
                    or result["current_review"]["kind"] != "investigation"
                ):
                    raise ValueError("fresh Investigation Review not reached")
                if (
                    result["current_review"]["request_event_id"]
                    == request["request_event_id"]
                    or result["investigation"]["authorities"]
                ):
                    raise ValueError("revision did not retain fresh approval boundary")
                receipt = {
                    "observation": result,
                    "http_model_send_attempts": contacts,
                    "duplicate_contact_free": True,
                    "response": response,
                    "journey": frozen,
                }
            checked_inputs(args.inputs)
            if args.phase in ("begin", "revise"):
                view = observation()
                expected_attempts = 1 if args.phase == "begin" else 2
                investigation = view["investigation"]
                if len(investigation["attempts"]) != expected_attempts:
                    raise ValueError(
                        "Attempt accounting changed the expected continuation"
                    )
                if (
                    investigation["authorities"]
                    or view["investigation_evidence"] is not None
                    or view["grounded_report"]["report"] is not None
                ):
                    raise ValueError(
                        "unselected scientific authority or consequential observation"
                    )
                receipt["contact_accounting"] = {
                    "attempts": len(investigation["attempts"]),
                    "journey_model_contact_count": investigation["model_contact_count"],
                    "journey_source_contact_count": investigation[
                        "source_contact_count"
                    ],
                    "phase_http_model_send_attempts": ContactCeilingModel.contacts,
                    "phase_http_source_send_attempts": len(
                        SourceCeilingTransport.sends
                    ),
                    "phase_source_urls": SourceCeilingTransport.sends,
                    "provider_creates": 0,
                    "vcf_reads": 0,
                    "limit": "HTTP send attempts and retained Journey contacts; not proof of server receipt. Only selected model POST and two read-only NCBI GET URLs are callable in this helper.",
                }
        receipt = {**receipt, "cli_calls": cli_calls}
        save(args.receipt, receipt)
        print(canonical_json({"phase": args.phase, "receipt": identity(args.receipt)}))
