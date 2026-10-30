"""Reusable proposal-information models and builders for application tests."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai.messages import ModelResponse, ToolCallPart

from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    register_run,
)
from tests.model_endpoint_fakes import ScriptedModelEndpoint


def runtime_configuration(*, model: str = "Qwen/Qwen3.5-397B-A17B"):
    return {
        "contract_version": "1.0",
        "heartbeat_interval_seconds": 1,
        # The fixture model sits under "qwen"; the build's default is "nemotron".
        "agent_invocation": {"model_reference": "qwen"},
        "model_roster": {
            "qwen": {
                "provider": "nebius",
                "endpoint": "https://api.tokenfactory.us-central1.nebius.com",
                "model": model,
                "credential_env": "LOCAL_MODEL_KEY",
            },
            "nemotron": {
                "provider": "nebius",
                "endpoint": "https://api.tokenfactory.us-central1.nebius.com/v1",
                "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
                "credential_env": "LOCAL_MODEL_KEY",
            },
        },
        # The observation tools of the locked environment.
        "investigation_tools": {
            "bcftools": {"package": "pysam", "package_version": "0.24.1"},
            "samtools": {"package": "pysam", "package_version": "0.24.1"},
        },
    }


class InspectionModel:
    def __init__(self, input_set_id):
        self.input_set_id = input_set_id
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "inspect_input_metadata",
                        {"input_set_id": self.input_set_id},
                        tool_call_id="inspect-call",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "checkpoint": "registered_inputs_inspected",
                        "input_set_id": self.input_set_id,
                        "inspected": True,
                    },
                    tool_call_id="inspection-output",
                )
            ]
        )


def proposal_draft(review, *, path_by_role):
    requirements = [
        {
            "kind": "research_goal",
            "subject_key": "requested_variant",
            "domain": "scientific",
            "meaning": "The exact variant named in the research request.",
            "knowledge": {"state": "asserted", "quote": "rs1234"},
            "acquisition_route": "use_supplied_assertion",
            "validation": {"kind": "none"},
        },
    ]
    for role in path_by_role:
        requirements.append(
            {
                "kind": "registered_input",
                "subject_key": "registered_" + role.lower().replace("-", "_"),
                "domain": "operational",
                "meaning": "The exact registered FASTQ input path.",
                "knowledge": {
                    "state": "observed",
                    "input_role": role,
                    "fact": "registered_path",
                },
                "acquisition_route": "validate_with_tool",
                "validation": {"kind": "none"},
            }
        )
    for kind, key in (
        ("sample_identity", "sample_identity"),
        ("taxon_identity", "taxon_identity"),
        ("reference_assembly_identity", "reference_assembly"),
        ("companion_asset", "reference_companions"),
        ("analysis_recipe", "analysis_recipe"),
        ("resource_profile", "resource_profile"),
    ):
        asserted_value = {"sample_identity": "sample-001"}.get(kind)
        if asserted_value is not None:
            knowledge = {
                "state": "asserted",
                "value": asserted_value,
                "quote": asserted_value,
            }
            acquisition_route = "use_supplied_assertion"
            validation = {
                "kind": "tool_evidence",
                "requirement": f"Obtain exact evidence for {kind}.",
            }
        else:
            knowledge = {
                "state": "to_find",
                "evidence_event_ids": [review.request.source_event_id],
            }
            acquisition_route = "validate_with_tool"
            validation = {
                "kind": "tool_evidence",
                "requirement": f"Obtain exact evidence for {kind}.",
            }
        requirements.append(
            {
                "kind": kind,
                "subject_key": key,
                "domain": "scientific" if kind != "resource_profile" else "operational",
                "meaning": f"Information required for {kind}.",
                "knowledge": knowledge,
                "acquisition_route": acquisition_route,
                "validation": validation,
            }
        )
    requirements.append(
        {
            "kind": "scientific_investigation",
            "subject_key": "requested_variant_rs1234",
            "domain": "scientific",
            "meaning": "Whether the sample carries rs1234.",
            "knowledge": {"state": "asserted", "value": "rs1234", "quote": "rs1234"},
            "acquisition_route": "use_supplied_assertion",
            "validation": {
                "kind": "tool_evidence",
                "requirement": "Observe the variant after execution.",
            },
            "investigation_purpose": "known_variant",
        }
    )
    return {"requirements": requirements}


def registered_run(
    root: Path,
    *,
    prompt_text: str = (
        "Analyze sample sample-001 for the asserted SNP rs1234 and produce a VCF."
    ),
    runtime_config: dict | None = None,
):
    prompt = root / "prompt.txt"
    config = root / "runtime.json"
    paths = {"read_1": root / "r1.fastq", "read_2": root / "r2.fastq"}
    prompt.write_text(prompt_text, encoding="utf-8")
    config.write_text(
        json.dumps(runtime_config or runtime_configuration()), encoding="utf-8"
    )
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"reads-{index}".encode())
    receipt = register_run(
        RegisterRunCommand(
            run_root=str(root / "runs"),
            prompt_file=str(prompt),
            runtime_config_file=str(config),
            inputs=tuple(
                RegisterFastqInput(role=role, registered_path=str(path))
                for role, path in paths.items()
            ),
        ),
        repository_root=Path.cwd(),
    )
    return receipt, paths
