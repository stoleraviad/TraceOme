import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from pydantic_ai.tools import ToolDefinition

from bio_neb.agentic_journey.model_settings import NEMOTRON_SUPER_VL_ENDPOINT_POLICY
from bio_neb.agentic_journey.planning_assignment import (
    planning_assignment,
)
from bio_neb.agentic_journey.runtime_config import (
    RuntimeConfigError,
    read_runtime_config,
    resolve_runtime_config,
)
from bio_neb.application.commands import (
    RegisterFastqInput,
    RegisterRunCommand,
    register_run,
)
from bio_neb.application.queries import (
    ENDED_RUN_STATUSES,
    ReviewJourneyQuery,
    review_journey,
)
from bio_neb.failure import CorrectionWording, Invariant
from bio_neb.journey import Journey
from bio_neb.recording import read_evidence
from tests.input_correction_fakes import prepare_input_correction
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.planning_assignment_fakes import parabricks_account


def _attempt_configuration():
    return {
        "contract_version": "1.0",
        "heartbeat_interval_seconds": 1,
        "agent_invocation": {"model_reference": "qwen"},
        "model_roster": {
            "qwen": {
                "provider": "nebius",
                "endpoint": "http://127.0.0.1:18089/v1",
                "model": "Qwen/Qwen3.5-397B-A17B",
                "credential_env": "LOCAL_MODEL_KEY",
            }
        },
    }


def _registered_attempt_fixture(root: Path, *, model: str | None = None):
    prompt = root / "prompt.txt"
    config = root / "runtime.json"
    fastqs = (root / "r1.fastq", root / "r2.fastq")
    prompt.write_text("inspect my registered inputs", encoding="utf-8")
    configuration = _attempt_configuration()
    if model is not None:
        configuration["model_roster"]["qwen"]["model"] = model
    config.write_text(json.dumps(configuration), encoding="utf-8")
    for index, fastq in enumerate(fastqs):
        fastq.write_bytes(f"reads-{index}".encode())
    receipt = register_run(
        RegisterRunCommand(
            run_root=str(root / "runs"),
            prompt_file=str(prompt),
            runtime_config_file=str(config),
            inputs=tuple(
                RegisterFastqInput(role=f"read_{index + 1}", registered_path=str(path))
                for index, path in enumerate(fastqs)
            ),
        ),
        repository_root=Path.cwd(),
    )
    return receipt, config


def _assert_configuration_failure_held(test, receipt, held, *, code):
    """The failure is retained once, ends nothing, and the hold names it (#216)."""
    events = read_evidence(Journey.open(Path(receipt.journey_file)))
    review = review_journey(ReviewJourneyQuery(journey_file=receipt.journey_file))
    *before, leaf = [
        e for e in events if e["event_type"] not in {"PassStarted", "PassClosed"}
    ]
    test.assertEqual(leaf["event_type"], "RuntimeConfigurationFailed")
    # Retained at the head its planning dispatch started from.
    test.assertEqual(leaf["parent_ids"], [before[-1]["event_id"]])
    test.assertEqual(leaf["payload"]["run_id"], events[0]["journey_id"])
    test.assertEqual(leaf["payload"]["code"], code)
    test.assertNotIn("RunFailed", [e["event_type"] for e in events])
    test.assertEqual(
        (held.invariant_id, held.evidence),
        (
            code,
            {
                "configuration_failure_event_id": leaf["event_id"],
                "violations": leaf["payload"]["violations"],
            },
        ),
    )
    # Outside a coordinator Pass nothing is held yet; the Run has not ended.
    test.assertNotIn(review.status, ENDED_RUN_STATUSES)
    test.assertIsNone(review.failure_code)


def _resolve_through_registration(path, environ, *, agent_role):
    """The attempt runner's path: registration records the file's selections, and
    each Attempt resolves the role and its credential from them alone."""

    registered = read_runtime_config(path).runtime_selections()
    return resolve_runtime_config(registered, environ, agent_role=agent_role)


class RuntimeConfigTests(unittest.TestCase):
    def test_tool_choice_is_refused_from_runtime_configuration(self):
        # The forced Tool choice is derived from a role's shape, never read from a
        # runtime file.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            configuration = _attempt_configuration()
            configuration["agent_invocation"] = {
                "model_reference": "qwen",
                "model_settings": {"tool_choice": "required"},
            }
            path.write_text(json.dumps(configuration), encoding="utf-8")

            with self.assertRaises(RuntimeConfigError) as refused:
                read_runtime_config(path)

        self.assertEqual(refused.exception.code, "runtime_config_invalid")
        self.assertIn(
            "agent_invocation.model_settings.tool_choice", refused.exception.violations
        )

    def test_retained_request_settings_carry_only_the_recorded_tool_choice(self):
        """Owner, #476 alignment G1.2 B: the Run records one Tool choice, so a
        retained request reopens only with that choice; the forced default is
        gone."""
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfiguration,
            resolve_runtime_config,
            validate_retained_model_request,
        )

        policy = resolve_runtime_config(
            RuntimeConfiguration.model_validate(_attempt_configuration()),
            {"LOCAL_MODEL_KEY": "offline"},
            agent_role="planning_agent",
        ).invocation_policy

        def request(model_settings):
            return {
                "contact_policy": policy.model_contact.model_dump(mode="json"),
                "stage": "reference",
                "compatible_adapter_evidence": {
                    "request": {"model_settings": model_settings}
                },
            }

        settings = policy.stage_model_settings("reference").request_settings()
        validate_retained_model_request(
            request({**settings, "tool_choice": "auto"}), policy
        )
        for refused in (None, "required", "none", ["inspect_input_metadata"]):
            with (
                self.subTest(tool_choice=refused),
                self.assertRaisesRegex(ValueError, "retained invocation settings"),
            ):
                validate_retained_model_request(
                    request(
                        settings
                        if refused is None
                        else {**settings, "tool_choice": refused}
                    ),
                    policy,
                )

    def test_model_effort_has_explicit_default_kind_and_invocation_limits(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

        value = _attempt_configuration()
        public = RuntimeConfiguration.model_validate(value)
        for role in ("planning_agent", "investigation_planner"):
            policy = public.agent_invocation_policy(role, AgentInvocationOverrides())
            self.assertEqual(policy.model_dump()["model_requests"], 200)
        value["agent_invocation"] = {
            "model_reference": "qwen",
            "model_requests": 30,
            "scopes": {"role": {"investigation_planner": {"model_requests": 12}}},
        }
        public = RuntimeConfiguration.model_validate(value)
        self.assertEqual(
            public.agent_invocation_policy(
                "investigation_planner", AgentInvocationOverrides()
            ).model_requests,
            12,
        )
        self.assertEqual(
            public.agent_invocation_policy(
                "investigation_planner", AgentInvocationOverrides(model_requests=2)
            ).model_requests,
            2,
        )
        with self.assertRaises(ValidationError):
            AgentInvocationOverrides(model_requests=0)

    def test_each_role_and_stage_may_have_its_own_failed_turns(self):
        # One failure rule for every role, on the one override path:
        # `agent_invocation.failed_turns` for every role, its role scope for one,
        # a stage scope for its own requests.
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfiguration,
            RuntimeSelections,
        )

        value = _attempt_configuration()
        value["agent_invocation"] = {**value["agent_invocation"], "failed_turns": 2}
        plain = RuntimeConfiguration.model_validate(value)
        self.assertEqual(
            plain.runtime_selections().model_dump(mode="json")["agent_invocation"][
                "scopes"
            ]["role"],
            {},
        )
        # A Tool-driven role takes the rule too (no role is excluded).
        value["agent_invocation"]["scopes"] = {
            "role": {
                "report_explainer": {"failed_turns": 5},
                "planning_agent": {"failed_turns": 3},
            }
        }
        public = RuntimeConfiguration.model_validate(value)

        def continuation(config, role, overrides=None):
            return config.agent_invocation_policy(
                role, overrides or AgentInvocationOverrides()
            ).continuation

        self.assertEqual(continuation(public, "report_explainer").failed_turns, 5)
        self.assertEqual(continuation(public, "planning_agent").failed_turns, 3)
        self.assertEqual(continuation(public, "investigation_planner").failed_turns, 2)
        self.assertEqual(continuation(public, "table_reader").failed_turns, 2)
        # A work Review's grant replaces them all for the next Attempt.
        self.assertEqual(
            continuation(
                public, "report_explainer", AgentInvocationOverrides(failed_turns=9)
            ).model_dump(mode="json"),
            {"failed_turns": 9},
        )
        selections = public.runtime_selections()
        dumped = selections.model_dump(mode="json")
        self.assertEqual(dumped["agent_invocation"]["failed_turns"], 2)
        self.assertEqual(
            dumped["agent_invocation"]["scopes"]["role"],
            {
                "report_explainer": {"failed_turns": 5},
                "planning_agent": {"failed_turns": 3},
            },
        )
        self.assertEqual(RuntimeSelections.model_validate(dumped), selections)
        # A stage scope tunes its own requests' number; others keep the role's.
        from bio_neb.agentic_journey.role_declaration import role as declared_role

        stage = declared_role("planning_agent").stages[0]
        value["agent_invocation"]["scopes"]["stage"] = {stage: {"failed_turns": 7}}
        staged = continuation(
            RuntimeConfiguration.model_validate(value), "planning_agent"
        )
        self.assertEqual(staged.stage_failed_turns, {stage: 7})
        self.assertEqual((staged.limit(stage), staged.limit(None)), (7, 3))
        self.assertEqual(staged.most_failed_turns, 7)
        # Zero failed turns is no rule.
        value["agent_invocation"]["scopes"] = {
            "role": {"planning_agent": {"failed_turns": 0}}
        }
        with self.assertRaises(ValidationError):
            RuntimeConfiguration.model_validate(value)

    def test_model_rules_and_explicit_settings_survive_registration_roundtrip(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfiguration,
            RuntimeSelections,
        )

        raw = _attempt_configuration()
        raw["agent_invocation"] = {
            "model_reference": "qwen",
            "model_settings": {
                "temperature": 0.6,
                "max_tokens": None,
                "parallel_tool_calls": True,
            },
            # The role scope is the narrower layer: its explicit values, an
            # explicit omission included, win over the global ones.
            "scopes": {
                "role": {
                    "planning_agent": {
                        "model_settings": {
                            "temperature": None,
                            "top_k": 1,
                            "max_tokens": 32768,
                        }
                    },
                    "investigation_planner": {
                        "model_settings": {"parallel_tool_calls": False}
                    },
                }
            },
        }
        original = RuntimeConfiguration.model_validate(raw)
        retained = RuntimeConfiguration.model_validate_json(original.model_dump_json())
        strict_config = RuntimeConfiguration.model_validate_json(
            original.model_dump_json(), strict=True
        )
        strict_selections = RuntimeSelections.model_validate_json(
            original.runtime_selections().model_dump_json(), strict=True
        )
        for config in (original, retained, strict_config, strict_selections):
            policy = config.agent_invocation_policy(
                "planning_agent", AgentInvocationOverrides()
            )
            self.assertEqual(
                policy.model_settings.request_settings(),
                {
                    "parallel_tool_calls": True,
                    "max_tokens": 32768,
                    "top_k": 1,
                },
            )
            self.assertEqual(
                policy.settings_selection.baseline.request_settings(),
                {
                    "parallel_tool_calls": False,
                    "max_tokens": 245760,
                },
            )
            self.assertEqual(
                policy.settings_selection.context_budget.input_reserve, 16384
            )
            disabled = config.agent_invocation_policy(
                "investigation_planner", AgentInvocationOverrides()
            )
            self.assertEqual(
                disabled.model_settings.request_settings(),
                {"parallel_tool_calls": False, "temperature": 0.6},
            )
        self.assertEqual(original.runtime_selections(), retained.runtime_selections())
        self.assertEqual(original.runtime_selections(), strict_selections)
        self.assertIsInstance(
            strict_selections.model_roster["qwen"].settings_selection.accepted_settings,
            tuple,
        )
        hostile = original.runtime_selections().model_dump(mode="json")
        hostile["model_roster"]["qwen"]["settings_selection"][
            "accepted_settings"
        ].append("extra_body")
        with self.assertRaises(ValidationError) as raised:
            RuntimeSelections.model_validate_json(json.dumps(hostile), strict=True)
        self.assertEqual(raised.exception.errors()[0]["type"], "literal_error")

    def test_selections_recorded_before_the_nemotron_profile_still_read(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfiguration,
            retained_runtime_selections,
        )

        # Copied from a recorded Run's RunInputsRegistered
        # `payload.runtime_selections`. Its baseline has no chat-template keys;
        # that Run is kept readable.
        recorded = {
            "agent_continuation": {"planning_agent": {"output_repair": 0}},
            "agent_invocation": {
                "exhaustion": {
                    "model_availability": "review",
                    "singleton_action": "review",
                },
                "loop_retry": 0,
                "model_reference": "qwen",
                "model_requests": 200,
                "model_settings": {"parallel_tool_calls": True},
                "repeated_call": 3,
                "same_call_correction": 5,
                "scopes": {"role": {}, "stage": {}, "tool": {}},
                "tool_retry": 0,
                "truncation_retry": 1,
            },
            "failure": {
                "scopes": {
                    "stage": {
                        stage: {
                            "operation": {
                                "retry_attempts": 1,
                                "transient": "ask",
                            }
                        }
                        for stage in (
                            "admit_candidate",
                            "cpu_preparation",
                            "h100",
                            "input_storage",
                        )
                    }
                }
            },
            "heartbeat_interval_seconds": 30.0,
            "investigation_continuation": {"output_repair": 1},
            "model_roster": {
                "qwen": {
                    "credential_env": "NEBIUS_API_KEY",
                    "endpoint": "https://api.tokenfactory.us-central1.nebius.com/v1",
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "provider": "nebius",
                    "settings_selection": {
                        "accepted_settings": [
                            "parallel_tool_calls",
                            "max_tokens",
                            "temperature",
                            "top_p",
                            "top_k",
                            "seed",
                        ],
                        "baseline": {
                            "max_tokens": 245760,
                            "parallel_tool_calls": False,
                            "seed": None,
                            "temperature": None,
                            "top_k": None,
                            "top_p": None,
                        },
                        "context_budget": {
                            "context_window": 262144,
                            "input_reserve": 16384,
                            "serialized_body_safety_margin": 8192,
                            "source": "preserved_product_configuration",
                        },
                        "model": "Qwen/Qwen3.5-397B-A17B",
                    },
                }
            },
        }
        # The record's correction allowances are retired (no compatibility for
        # old Runs), so the record as written is refused; without them its
        # profile still reads.
        with self.assertRaisesRegex(ValueError, "Extra inputs are not permitted"):
            retained_runtime_selections(json.loads(json.dumps(recorded)))
        recorded = json.loads(json.dumps(recorded))
        del recorded["agent_continuation"]
        del recorded["investigation_continuation"]
        for name in ("same_call_correction", "repeated_call", "truncation_retry"):
            del recorded["agent_invocation"][name]
        retained = retained_runtime_selections(recorded)
        self.assertEqual(
            retained.agent_invocation_policy(
                "planning_agent", AgentInvocationOverrides()
            ).model_settings.request_settings(),
            {"parallel_tool_calls": True, "max_tokens": 245760},
        )
        # A key recorded before the profile is still required.
        incomplete = json.loads(json.dumps(recorded))
        del incomplete["model_roster"]["qwen"]["settings_selection"]["baseline"]["seed"]
        with self.assertRaisesRegex(
            ValueError, "retained runtime selections must be complete"
        ):
            retained_runtime_selections(incomplete)
        # A Nemotron Run records the chat-template keys its profile sets.
        value = _attempt_configuration()
        value["model_roster"]["qwen"]["model"] = "nvidia/Nemotron-3-Ultra-550b-a55b"
        dumped = (
            RuntimeConfiguration.model_validate(value)
            .runtime_selections()
            .model_dump(mode="json")
        )
        baseline = dumped["model_roster"]["qwen"]["settings_selection"]["baseline"]
        self.assertEqual(
            (baseline["reasoning"], baseline["force_nonempty_content"]),
            ("medium", True),
        )
        self.assertEqual(
            retained_runtime_selections(dumped).model_dump(mode="json"), dumped
        )

    def test_investigation_tools_section_is_recorded_only_when_the_runtime_names_it(
        self,
    ):
        # An additive optional section names the observation tools, so Runs
        # registered before it still read back exactly. A fresh Run naming none
        # records the locked package.
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfiguration,
            retained_runtime_selections,
        )

        tools = {
            "bcftools": {"package": "pysam", "package_version": "0.24.1"},
            "samtools": {"package": "pysam", "package_version": "0.24.1"},
        }
        named = RuntimeConfiguration.model_validate(
            {**_attempt_configuration(), "investigation_tools": tools}
        ).runtime_selections()
        recorded = named.model_dump(mode="json")
        self.assertEqual(recorded["investigation_tools"], tools)
        self.assertEqual(
            retained_runtime_selections(recorded).model_dump(mode="json"), recorded
        )
        unnamed = (
            RuntimeConfiguration.model_validate(_attempt_configuration())
            .runtime_selections()
            .model_dump(mode="json")
        )
        self.assertEqual(unnamed["investigation_tools"], tools)
        earlier = {
            key: value for key, value in unnamed.items() if key != "investigation_tools"
        }
        self.assertEqual(
            retained_runtime_selections(earlier).model_dump(mode="json"), earlier
        )
        for refused in (
            {**tools, "bcftools": {"package": "bioconda", "package_version": "1.24"}},
            {**tools, "tabix": {"package": "pysam", "package_version": "0.24.1"}},
        ):
            with self.subTest(refused=refused), self.assertRaises(ValidationError):
                RuntimeConfiguration.model_validate(
                    {**_attempt_configuration(), "investigation_tools": refused}
                )
        # A fresh file naming one tool keeps the other's build value; a
        # recorded section is complete or refused.
        with self.assertRaises(ValidationError) as incomplete:
            retained_runtime_selections(
                {**recorded, "investigation_tools": {"bcftools": tools["bcftools"]}}
            )
        self.assertEqual(
            [(e["loc"], e["type"]) for e in incomplete.exception.errors()],
            [(("investigation_tools", "samtools"), "missing")],
        )

    def test_retained_model_rules_cannot_redefine_a_fresh_baseline(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfiguration,
            RuntimeSelections,
        )

        raw = (
            RuntimeConfiguration.model_validate(_attempt_configuration())
            .runtime_selections()
            .model_dump(mode="json")
        )
        alternate = json.loads(json.dumps(raw["model_roster"]["qwen"]))
        alternate["model"] = "nvidia/Nemotron-3-Ultra-550b-a55b"
        rules = alternate["settings_selection"]
        rules["model"] = alternate["model"]
        rules["accepted_settings"] = ["parallel_tool_calls", "max_tokens"]
        rules["context_budget"]["context_window"] = 65536
        rules["baseline"]["max_tokens"] = 49152
        raw["model_roster"]["alternate"] = alternate
        with self.assertRaisesRegex(ValidationError, "model-owned baseline"):
            RuntimeConfiguration.model_validate({"contract_version": "1.0", **raw})
        # A previously recorded model budget remains reconstructible independently
        # of the defaults installed now, and still validates its own request values.
        raw["agent_invocation"]["scopes"]["role"] = {
            "planning_agent": {"model_reference": "alternate"}
        }
        config = RuntimeSelections.model_validate(raw)
        policy = config.agent_invocation_policy(
            "planning_agent", AgentInvocationOverrides()
        )
        self.assertEqual(policy.model_settings.max_tokens, 49152)
        for name, value in (("max_tokens", 49153), ("temperature", 0.6)):
            refused = json.loads(json.dumps(raw))
            refused["agent_invocation"]["scopes"]["role"]["planning_agent"][
                "model_settings"
            ] = {name: value}
            with (
                self.subTest(setting=name),
                self.assertRaises(ValidationError) as raised,
            ):
                RuntimeSelections.model_validate(refused)
            self.assertEqual(
                [
                    (".".join(map(str, item["loc"])), item["type"])
                    for item in raised.exception.errors()
                ],
                [
                    (
                        "agent_invocation.scopes.role.planning_agent."
                        f"model_settings.{name}",
                        "model_setting_refused",
                    )
                ],
            )

    def test_each_model_has_one_endpoint_policy_with_todays_qwen_values(self):
        # The Model endpoint policy owns a model's request defaults, accepted settings, Tool schema
        # projection and correction wording. Qwen carries today's values, and the one correction text;
        # Nemotron's adapter is its family profile. An unlisted model has no adapter.
        from bio_neb.agentic_journey.model_settings import model_endpoint_policy

        registered_qwen = {
            "model": "Qwen/Qwen3.5-397B-A17B",
            "accepted_settings": [
                "parallel_tool_calls",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "seed",
            ],
            "baseline": {
                "max_tokens": 245760,
                "parallel_tool_calls": False,
                "seed": None,
                "temperature": None,
                "top_k": None,
                "top_p": None,
            },
            "context_budget": {
                "context_window": 262144,
                "input_reserve": 16384,
                "serialized_body_safety_margin": 8192,
                "source": "preserved_product_configuration",
            },
        }
        registered_nemotron = {
            **registered_qwen,
            "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
            "accepted_settings": [
                *registered_qwen["accepted_settings"],
                "reasoning",
                "force_nonempty_content",
            ],
            # R1 "everything on", R1 correction 1.0/0.95, S2 decision. Qwen's
            # 245,760 reply allowance, kept apart from the window.
            "baseline": {
                **registered_qwen["baseline"],
                "temperature": 1.0,
                "top_p": 0.95,
                "reasoning": "medium",
                "force_nonempty_content": True,
            },
            # Nebius reports Nemotron 3 Ultra's context length as 1,048,576.
            "context_budget": {
                **registered_qwen["context_budget"],
                "context_window": 1048576,
            },
            # Every registered injection pattern and quirk trigger.
            "injection_patterns": [
                "action_commit",
                "plan_review",
                "literal_guard",
                "result_suffix",
                "repeat_call",
                "progress_notice",
                "hint_as_user",
                "text_form_tool_call",
                "text_form_tool_call_cut_off",
                "think_tags",
                "call_reasoning_missing",
                "argument_name",
                "whitespace_runaway",
                "refused_call_context",
                "empty_arguments",
                "placeholder_argument",
            ],
        }
        # Nemotron's Tool projection adds its call-reasoning argument (`test_nebius_model_endpoint.CallReasoningProjectionTests`) and its
        # correction wording is its own (`test_action_admission_feedback.ModelCorrectionWordingTests`).
        for model, registered, todays_qwen_adapter in (
            ("Qwen/Qwen3.5-397B-A17B", registered_qwen, True),
            ("nvidia/Nemotron-3-Ultra-550b-a55b", registered_nemotron, False),
        ):
            with self.subTest(model=model):
                policy = model_endpoint_policy(model)
                self.assertEqual(
                    policy.settings_selection.model_dump(mode="json"), registered
                )
                if not todays_qwen_adapter:
                    continue
                # A Tool with no canonical-text rule is shown as offered.
                definition = ToolDefinition(
                    name="probe",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"rsid": {"type": "string", "pattern": "^rs"}},
                    },
                )
                self.assertIs(policy.project_tool_schema(definition), definition)
                # Qwen reads the one correction text.
                self.assertEqual(policy.correction_wording, CorrectionWording())

    def test_a_nemotron_size_entry_states_only_its_differences_from_the_family(self):
        from bio_neb.agentic_journey.model_settings import (
            NEMOTRON_FAMILY,
            model_endpoint_policy,
        )

        ultra = model_endpoint_policy("nvidia/Nemotron-3-Ultra-550b-a55b")
        # Ultra is the first entry and overrides no setting; every size names its
        # own instruction form.
        self.assertEqual(ultra.instruction_form, "nemotron_3_ultra")
        served = {
            "endpoint": ultra.endpoint,
            "instruction_form": ultra.instruction_form,
        }
        self.assertEqual(
            ultra, NEMOTRON_FAMILY.model(ultra.settings_selection.model, **served)
        )
        smaller = NEMOTRON_FAMILY.model(
            "nvidia/smaller-size", **served, reasoning="off"
        )
        self.assertEqual(smaller.settings_selection.model, "nvidia/smaller-size")
        self.assertEqual(
            smaller.settings_selection.baseline.model_dump(),
            {**ultra.settings_selection.baseline.model_dump(), "reasoning": "off"},
        )
        for same in ("accepted_settings", "context_budget"):
            self.assertEqual(
                getattr(smaller.settings_selection, same),
                getattr(ultra.settings_selection, same),
            )
        self.assertEqual(smaller.wire_settings, ultra.wire_settings)
        # A size entry is a typed setting value or nothing.
        for overrides, refusal in (
            ({"reasoning": "low"}, (("reasoning",), "literal_error")),
            ({"thinking": True}, (("thinking",), "extra_forbidden")),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaises(ValidationError) as refused,
            ):
                NEMOTRON_FAMILY.model("nvidia/smaller-size", **served, **overrides)
            self.assertEqual(
                [(error["loc"], error["type"]) for error in refused.exception.errors()],
                [refusal],
            )

    def test_the_global_model_reference_gives_every_role_the_nemotron_profile(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

        value = _attempt_configuration()
        value["model_roster"]["nemotron"] = {
            **value["model_roster"]["qwen"],
            "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
        }
        value["agent_invocation"] = {"model_reference": "nemotron"}
        config = RuntimeConfiguration.model_validate(value)
        for role in ("planning_agent", "investigation_planner"):
            with self.subTest(role=role):
                policy = config.agent_invocation_policy(
                    role, AgentInvocationOverrides()
                )
                self.assertEqual(policy.model_reference, "nemotron")
                self.assertEqual(
                    policy.model_settings.request_settings(),
                    {
                        "parallel_tool_calls": False,
                        "max_tokens": 245760,
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "reasoning": "medium",
                        "force_nonempty_content": True,
                    },
                )
                self.assertEqual(
                    set(policy.settings_sources.values()), {"model_default"}
                )

    def test_a_qwen_config_setting_a_nemotron_only_setting_is_refused_by_its_key(self):
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

        for name, setting in (
            ("reasoning", "medium"),
            ("force_nonempty_content", True),
        ):
            for scope, path in (
                ({"model_settings": {name: setting}}, ()),
                (
                    {
                        "scopes": {
                            "role": {
                                "planning_agent": {"model_settings": {name: setting}}
                            }
                        }
                    },
                    ("scopes", "role", "planning_agent"),
                ),
            ):
                with self.subTest(name=name, path=path):
                    value = _attempt_configuration()
                    value["model_roster"]["qwen"]["model"] = "Qwen/Qwen3.5-397B-A17B"
                    value["agent_invocation"] = {**value["agent_invocation"], **scope}
                    with self.assertRaises(ValidationError) as refused:
                        RuntimeConfiguration.model_validate(value)
                    self.assertEqual(
                        [
                            (error["loc"], error["type"])
                            for error in refused.exception.errors()
                        ],
                        [
                            (
                                ("agent_invocation", *path, "model_settings", name),
                                # Unaccepted and without a wire row; the
                                # translation-table check reports last.
                                "model_setting_untranslatable",
                            )
                        ],
                    )

    def test_model_baseline_reaches_every_role_and_explicit_omission_wins(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

        config = RuntimeConfiguration.model_validate(_attempt_configuration())
        for role in (
            "planning_agent",
            "investigation_planner",
        ):
            with self.subTest(role=role):
                baseline = config.agent_invocation_policy(
                    role, AgentInvocationOverrides()
                )
                self.assertEqual(
                    baseline.model_settings.request_settings(),
                    {
                        "parallel_tool_calls": False,
                        "max_tokens": 245760,
                    },
                )
                value = _attempt_configuration()
                value["agent_invocation"] = {
                    "model_reference": "qwen",
                    "scopes": {
                        "role": {role: {"model_settings": {"max_tokens": None}}}
                    },
                }
                omitted = RuntimeConfiguration.model_validate(
                    value
                ).agent_invocation_policy(role, AgentInvocationOverrides())
                self.assertEqual(
                    omitted.model_settings.request_settings(),
                    {
                        "parallel_tool_calls": False,
                    },
                )

    def test_an_explicit_null_template_control_overrides_nemotron_at_every_scope(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfiguration,
            retained_runtime_selections,
        )

        omitted = {"reasoning": None, "force_nonempty_content": None}
        for scope, invocation in (
            ("global", {"model_settings": omitted}),
            (
                "role",
                {"scopes": {"role": {"planning_agent": {"model_settings": omitted}}}},
            ),
            (
                "stage",
                {"scopes": {"stage": {"reference": {"model_settings": omitted}}}},
            ),
        ):
            value = _attempt_configuration()
            value["model_roster"]["qwen"]["model"] = "nvidia/Nemotron-3-Ultra-550b-a55b"
            value["agent_invocation"] = {**value["agent_invocation"], **invocation}
            config = RuntimeConfiguration.model_validate(value)
            # The retained registration reads back the same explicit omission.
            retained = retained_runtime_selections(
                config.runtime_selections().model_dump(mode="json")
            )
            for source, selections in (("fresh", config), ("retained", retained)):
                with self.subTest(scope=scope, source=source):
                    policy = selections.agent_invocation_policy(
                        "planning_agent", AgentInvocationOverrides()
                    )
                    resolved = (
                        policy.stage_settings["reference"]
                        if scope == "stage"
                        else policy
                    )
                    self.assertEqual(
                        policy.stage_model_settings("reference").request_settings(),
                        {
                            "parallel_tool_calls": False,
                            "max_tokens": 245760,
                            "temperature": 1.0,
                            "top_p": 0.95,
                        },
                    )
                    self.assertEqual(
                        {name: resolved.settings_sources.get(name) for name in omitted},
                        {name: scope for name in omitted},
                    )

    def test_unknown_roster_reference_and_unlisted_model_are_refused_by_their_paths(
        self,
    ):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

        # A model the registry does not list has no adapter: no fallback stands
        # in for it, and the provider is never asked to discover it.
        value = _attempt_configuration()
        value["model_roster"]["qwen"]["model"] = "temporarily-delisted/model"
        with self.assertRaises(ValidationError) as unlisted:
            RuntimeConfiguration.model_validate(value)
        self.assertEqual(
            [(e["loc"], e["type"]) for e in unlisted.exception.errors()],
            [(("model_roster", "qwen", "model"), "model_unlisted")],
        )
        value = _attempt_configuration()
        for configured, location in (
            ({"model_reference": "missing"}, "agent_invocation.model_reference"),
            (
                {
                    "scopes": {
                        "role": {
                            "investigation_planner": {"model_reference": "missing"}
                        }
                    }
                },
                "agent_invocation.scopes.role.investigation_planner.model_reference",
            ),
        ):
            value["agent_invocation"] = {
                **_attempt_configuration()["agent_invocation"],
                **configured,
            }
            with self.subTest(location=location):
                with self.assertRaises(ValidationError) as raised:
                    RuntimeConfiguration.model_validate(value)
                self.assertEqual(
                    [".".join(map(str, e["loc"])) for e in raised.exception.errors()],
                    [location],
                )
        # A model is chosen only by configuration; no command carries one.
        for name, value in (("model_reference", "qwen"), ("model_settings", {})):
            with self.subTest(command_field=name):
                with self.assertRaises(ValidationError) as raised:
                    AgentInvocationOverrides.model_validate({name: value})
                self.assertEqual(
                    [(e["loc"], e["type"]) for e in raised.exception.errors()],
                    [((name,), "extra_forbidden")],
                )

    def test_delisted_model_503_keeps_availability_lineage_and_selected_identity(self):
        from pydantic_ai.exceptions import ModelHTTPError

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _registered_attempt_fixture(Path(directory).resolve())

            def factory(loaded, selected):
                self.assertEqual(loaded.invocation_policy.model_reference, "qwen")
                self.assertEqual(selected.model, "Qwen/Qwen3.5-397B-A17B")

                async def unavailable(messages, info):
                    raise ModelHTTPError(503, selected.model, "temporarily unavailable")

                return ScriptedModelEndpoint(unavailable)(loaded, selected)

            journey = Journey.open(Path(receipt.journey_file))
            # No real waits: the window's delays are its cap times a zero jitter.
            with patch(
                "bio_neb.agentic_journey.model_observation.random.random",
                return_value=0,
            ):
                result = planning_assignment(model_factory=factory).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            events = read_evidence(journey)
        self.assertEqual(result.status, "model_availability_review_pending")
        self.assertFalse(
            any(e["event_type"] == "RuntimeConfigurationFailed" for e in events)
        )
        failures = [
            e["payload"]
            for e in events
            if e["event_type"] == "ModelPhysicalContactFailed"
        ]
        self.assertEqual(len(failures), 2)
        self.assertTrue(
            all(
                p["http_status_code"] == 503
                and p["failure_reason"] == "provider_failure"
                for p in failures
            )
        )
        started = next(
            e["payload"] for e in events if e["event_type"] == "AttemptStarted"
        )
        self.assertEqual(
            started["effective_invocation_policy"]["model_reference"], "qwen"
        )
        request = next(
            e["payload"] for e in events if e["event_type"] == "ModelRequestRecorded"
        )
        self.assertEqual(
            request["nebius_model_endpoint"]["model"], "Qwen/Qwen3.5-397B-A17B"
        )

    def test_investigation_preflight_resolves_policy_and_credential_without_a_client(
        self,
    ):
        from bio_neb.agentic_journey.agent_roles import RuntimeBinding
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration
        from bio_neb.investigation.investigation_assignment import (
            resolve_investigation_configuration,
        )

        value = _attempt_configuration()
        value["model_roster"]["nemotron"] = {
            **value["model_roster"]["qwen"],
            "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
            "credential_env": "OTHER_KEY",
        }
        value["agent_invocation"] = {
            "scopes": {
                "role": {"investigation_planner": {"model_reference": "nemotron"}}
            }
        }
        public = RuntimeConfiguration.model_validate(value)
        prepared = resolve_investigation_configuration(
            RuntimeBinding(public.runtime_selections(), "input-set"),
            {"OTHER_KEY": "selected-secret"},
            overrides=AgentInvocationOverrides(),
        )
        self.assertEqual(prepared.loaded.credential, "selected-secret")
        self.assertEqual(
            prepared.model_selection.model, "nvidia/Nemotron-3-Ultra-550b-a55b"
        )
        self.assertEqual(prepared.effective_policy.model_reference, "nemotron")

    def test_unresolvable_investigation_configuration_raises_its_configuration_error(
        self,
    ):
        # The one binding is the Run's registration (a Fork child inherits it), so
        # an unresolvable configuration raises its own error before any write.
        from bio_neb.agentic_journey.agent_roles import RuntimeBinding
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from bio_neb.agentic_journey.runtime_config import (
            RuntimeConfigError,
            RuntimeConfiguration,
        )
        from bio_neb.investigation.investigation_assignment import (
            resolve_investigation_configuration,
        )

        public = RuntimeConfiguration.model_validate(_attempt_configuration())
        binding = RuntimeBinding(public.runtime_selections(), "input-set")
        # The selected credential is absent from the environment.
        with self.assertRaises(RuntimeConfigError) as ordinary:
            resolve_investigation_configuration(
                binding, {}, overrides=AgentInvocationOverrides()
            )
        self.assertEqual(ordinary.exception.code, "model_credential_unavailable")

    def test_role_model_reference_is_a_configuration_only_invocation_selection(self):
        from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
        from bio_neb.agentic_journey.model_settings import model_endpoint_policy
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration

        value = {
            "contract_version": "1.0",
            "agent_invocation": {"model_reference": "qwen"},
            "heartbeat_interval_seconds": 1,
            "model_roster": {
                name: {
                    "provider": "nebius",
                    "endpoint": "https://api.tokenfactory.us-central1.nebius.com/v1",
                    "model": model,
                    "credential_env": "MODEL_KEY",
                }
                for name, model in (
                    ("qwen", "Qwen/Qwen3.5-397B-A17B"),
                    ("nemotron", "nvidia/Nemotron-3-Ultra-550b-a55b"),
                )
            },
        }
        original = RuntimeConfiguration.model_validate(value)
        value["agent_invocation"] = {
            "model_reference": "qwen",
            "scopes": {
                "role": {"investigation_planner": {"model_reference": "nemotron"}}
            },
        }
        edited = RuntimeConfiguration.model_validate(value)
        roles = (
            "planning_agent",
            "investigation_planner",
        )
        for role in roles:
            with self.subTest(role=role):
                before = original.agent_invocation_policy(
                    role, AgentInvocationOverrides()
                )
                after = edited.agent_invocation_policy(role, AgentInvocationOverrides())
                self.assertEqual(before.model_reference, "qwen")
                self.assertEqual(
                    after.model_reference,
                    "nemotron" if role == "investigation_planner" else "qwen",
                )
                # Each role's settings are its selected model's own defaults.
                for policy in (before, after):
                    self.assertEqual(
                        policy.model_settings,
                        model_endpoint_policy(
                            edited.model_for_policy(policy).model
                        ).settings_selection.baseline,
                    )
                self.assertEqual(before.continuation, after.continuation)
        selected = edited.agent_invocation_policy(
            "investigation_planner", AgentInvocationOverrides()
        )
        self.assertEqual(
            edited.model_for_policy(selected).model,
            "nvidia/Nemotron-3-Ultra-550b-a55b",
        )

    def test_verification_defaults_and_explicit_invocation_are_retained_independently(
        self,
    ):
        from bio_neb.application.verification_policy import (
            ArtifactVerificationOverrides,
            retained_artifact_verification_policy,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            value = _attempt_configuration()
            value["artifact_verification"] = {
                "mode": "independent_full_read",
                "reason": "explicit runtime audit",
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            registered = read_runtime_config(path)
            ordinary = registered.artifact_verification_policy(
                ArtifactVerificationOverrides()
            )
            amended = registered.artifact_verification_policy(
                ArtifactVerificationOverrides(mode="reuse_creation_proof")
            )
            self.assertEqual(ordinary.effective.mode, "independent_full_read")
            self.assertEqual(amended.effective.mode, "reuse_creation_proof")
            self.assertEqual(
                amended.overrides.model_dump(mode="json"),
                {"mode": "reuse_creation_proof", "reason": None},
            )
            self.assertEqual(ordinary.overrides.model_dump(mode="json"), {})
            self.assertNotEqual(ordinary.policy_id, amended.policy_id)
            retained = amended.model_dump(mode="json")
            value["artifact_verification"] = {"mode": "reuse_creation_proof"}
            path.write_text(json.dumps(value), encoding="utf-8")
            current = read_runtime_config(path)
            self.assertEqual(
                retained_artifact_verification_policy(
                    retained, defaults=registered.artifact_verification
                ),
                amended,
            )
            with self.assertRaises(ValueError):
                retained_artifact_verification_policy(
                    retained, defaults=current.artifact_verification
                )

    def test_verification_policy_refuses_incomplete_or_weakened_retained_settings(self):
        from bio_neb.application.verification_policy import (
            ArtifactVerificationDefaults,
            ArtifactVerificationOverrides,
            resolve_artifact_verification_policy,
            retained_artifact_verification_policy,
        )

        for settings in (
            {"mode": None},
            {"mode": "skip"},
            {"sensitive_use": "reuse_creation_proof"},
            {"destination_admission": "reuse_creation_proof"},
        ):
            with self.subTest(overrides=settings), self.assertRaises(ValidationError):
                ArtifactVerificationOverrides.model_validate(settings)
        for settings in (
            {"mode": None},
            {"sensitive_use": "reuse_creation_proof"},
            {"destination_admission": "reuse_creation_proof"},
        ):
            with self.subTest(defaults=settings), self.assertRaises(ValidationError):
                ArtifactVerificationDefaults.model_validate(settings)

        policy = resolve_artifact_verification_policy(
            ArtifactVerificationDefaults(), ArtifactVerificationOverrides()
        )
        self.assertEqual(policy.effective.mode, "reuse_creation_proof")
        for part in ("defaults", "effective"):
            for field in ("mode", "reason"):
                retained = policy.model_dump(mode="json")
                retained[part].pop(field)
                with (
                    self.subTest(missing=f"{part}.{field}"),
                    self.assertRaises(ValueError),
                ):
                    retained_artifact_verification_policy(retained)
        for field in ("defaults", "overrides", "effective", "policy_id"):
            retained = policy.model_dump(mode="json")
            retained.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                retained_artifact_verification_policy(retained)

        changes = (
            (
                "identity",
                lambda value: value.update(
                    policy_id="artifact-verification-policy:wrong"
                ),
            ),
            (
                "effective",
                lambda value: value["effective"].update(mode="independent_full_read"),
            ),
            (
                "overrides",
                lambda value: value["overrides"].update(mode="independent_full_read"),
            ),
            (
                "null override",
                lambda value: value["overrides"].update(mode=None),
            ),
        )
        for name, change in changes:
            retained = policy.model_dump(mode="json")
            change(retained)
            with self.subTest(tamper=name), self.assertRaises(ValueError):
                retained_artifact_verification_policy(retained)

    def test_hostile_factory_identity_is_refused_before_physical_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt, _ = _registered_attempt_fixture(root)
            contact_count = 0

            async def contact(messages, info):
                nonlocal contact_count
                del messages, info
                contact_count += 1
                raise AssertionError("hostile model must not be contacted")

            endpoint = ScriptedModelEndpoint(contact)

            def hostile(loaded, selected):
                # The real Adapter, advertising an endpoint other than the selected one.
                adapter = endpoint(loaded, selected)
                adapter.nebius_model_endpoint = {
                    "provider": "nebius",
                    "model": "substituted/model",
                    "contacted_url": "http://127.0.0.1:18089/v1",
                }
                return adapter

            with self.assertRaises(Invariant) as held:
                planning_assignment(model_factory=hostile).run(
                    prepare_input_correction(Journey.open(Path(receipt.journey_file))),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            _assert_configuration_failure_held(
                self, receipt, held.exception, code="runtime_config_changed"
            )
            events = read_evidence(Journey.open(Path(receipt.journey_file)))

        self.assertEqual(contact_count, 0)
        self.assertFalse(
            any(
                event["event_type"]
                in {"ModelRequestRecorded", "ModelPhysicalContactStarted"}
                for event in events
            )
        )

    def test_a_registered_selection_resolves_from_registration_never_its_file(self):
        # Registration records the resolved selections; every later Attempt
        # resolves from them alone, so an edited or removed file changes
        # nothing.
        endpoint = "https://api.tokenfactory.us-central1.nebius.com/v1"
        for model in ("Qwen/Qwen3.5-397B-A17B", "nvidia/Nemotron-3-Ultra-550b-a55b"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.json"
                value = {
                    "contract_version": "1.0",
                    "agent_invocation": {"model_reference": "qwen"},
                    "model_roster": {"qwen": {"endpoint": endpoint, "model": model}},
                }
                path.write_text(json.dumps(value), encoding="utf-8")
                registered = read_runtime_config(path).runtime_selections()
                recorded = registered.model_dump_json()
                value["agent_invocation"]["model_settings"] = {
                    "parallel_tool_calls": True
                }
                value["model_roster"]["qwen"]["model"] = "drifted/model"
                path.write_text(json.dumps(value), encoding="utf-8")
                for state in ("edited", "removed"):
                    if state == "removed":
                        path.unlink()
                    loaded = resolve_runtime_config(
                        registered,
                        {"NEBIUS_API_KEY": "secret"},
                        required_endpoint=endpoint,
                        agent_role="planning_agent",
                    )
                    self.assertEqual(
                        (
                            loaded.selections.model_roster["qwen"].model,
                            loaded.model.model,
                            loaded.invocation_policy.model_settings.parallel_tool_calls,
                        ),
                        (model, model, False),
                    )
                self.assertEqual(registered.model_dump_json(), recorded)

    def test_valid_config_resolves_credential_without_placing_it_in_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(
                json.dumps(
                    {
                        "contract_version": "1.0",
                        "agent_invocation": {"model_reference": "qwen"},
                        "heartbeat_interval_seconds": 30,
                        "model_roster": {
                            "qwen": {
                                "provider": "nebius",
                                "endpoint": "https://api.example.test",
                                "model": "Qwen/Qwen3.5-397B-A17B",
                                "credential_env": "NEBIUS_TOKEN",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded = _resolve_through_registration(
                path,
                {"NEBIUS_TOKEN": "secret-value"},
                agent_role="planning_agent",
            )

            self.assertEqual(loaded.credential, "secret-value")
            self.assertNotIn(
                "secret-value", json.dumps(loaded.selections.model_dump(mode="json"))
            )

    def test_public_contract_is_credential_independent_and_shared_with_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            value = {
                "contract_version": "1.0",
                "agent_invocation": {"model_reference": "qwen"},
                "heartbeat_interval_seconds": 30,
                "model_roster": {
                    "qwen": {
                        "provider": "nebius",
                        "endpoint": "https://api.example.test",
                        "model": "Qwen/Qwen3.5-397B-A17B",
                        "credential_env": "ONLY_THIS_TOKEN",
                    }
                },
            }
            path.write_text(json.dumps(value), encoding="utf-8")

            public_config = read_runtime_config(path)
            loaded = _resolve_through_registration(
                path,
                {"ONLY_THIS_TOKEN": "secret"},
                agent_role="planning_agent",
            )

            self.assertEqual(public_config.runtime_selections(), loaded.selections)
            with self.assertRaises(ValidationError):
                public_config.heartbeat_interval_seconds = 10
            with self.assertRaises(ValidationError):
                public_config.model_roster["qwen"].model = "changed/model"
            self.assertEqual(
                public_config.runtime_selections().model_dump(mode="json"),
                {
                    "model_roster": {
                        "qwen": {
                            **value["model_roster"]["qwen"],
                            "endpoint": "https://api.example.test/v1",
                            "settings_selection": {
                                "model": "Qwen/Qwen3.5-397B-A17B",
                                "accepted_settings": [
                                    "parallel_tool_calls",
                                    "max_tokens",
                                    "temperature",
                                    "top_p",
                                    "top_k",
                                    "seed",
                                ],
                                "baseline": {
                                    "parallel_tool_calls": False,
                                    "max_tokens": 245760,
                                    "temperature": None,
                                    "top_p": None,
                                    "top_k": None,
                                    "seed": None,
                                },
                                "context_budget": {
                                    "context_window": 262144,
                                    "input_reserve": 16384,
                                    "serialized_body_safety_margin": 8192,
                                    "source": "preserved_product_configuration",
                                },
                            },
                        },
                        # The build's default entry stays in the roster.
                        "nemotron": {
                            "provider": "nebius",
                            "endpoint": "https://api.tokenfactory.us-north1.nebius.com/v1",
                            "model": "nvidia/Nemotron-3.5-Super-VL",
                            "credential_env": "NEBIUS_API_KEY",
                            "settings_selection": NEMOTRON_SUPER_VL_ENDPOINT_POLICY.settings_selection.model_dump(
                                mode="json"
                            ),
                        },
                    },
                    "heartbeat_interval_seconds": 30.0,
                    "agent_invocation": {
                        "model_reference": "qwen",
                        "model_requests": 200,
                        "tool_choice": "auto",
                        "scopes": {"role": {}, "stage": {}, "tool": {}},
                        "loop_retry": 0,
                        "tool_retry": 0,
                        "exhaustion": {
                            "singleton_action": "review",
                            "model_availability": "review",
                        },
                        # An omitted failure rule records no key: the role
                        # default's four failed turns apply.
                        "model_settings": {},
                    },
                    # The failure module's section records omissions as omitted;
                    # model contact moved here.
                    "failure": {},
                    # The build's locked tools and verification (Q3).
                    "investigation_tools": {
                        tool: {"package": "pysam", "package_version": "0.24.1"}
                        for tool in ("bcftools", "samtools")
                    },
                    "artifact_verification": {
                        "mode": "reuse_creation_proof",
                        "reason": None,
                    },
                },
            )

            with self.assertRaises(RuntimeConfigError) as raised:
                _resolve_through_registration(
                    path,
                    {"UNRELATED_TOKEN": "secret"},
                    agent_role="planning_agent",
                )
            self.assertEqual(raised.exception.code, "model_credential_unavailable")
            self.assertEqual(
                raised.exception.violations,
                ("model_roster.qwen.credential_env",),
            )

    def test_reading_and_loading_reject_the_same_public_selections(self):
        valid = {
            "contract_version": "1.0",
            "agent_invocation": {"model_reference": "qwen"},
            "heartbeat_interval_seconds": 30,
            "model_roster": {
                "qwen": {
                    "provider": "nebius",
                    "endpoint": "https://api.example.test",
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "credential_env": "MODEL_KEY",
                }
            },
        }
        cases = {
            "zero failed turns": {
                **valid,
                "agent_invocation": {"failed_turns": 0},
            },
            "coerced failed turns": {
                **valid,
                "agent_invocation": {"failed_turns": "2"},
            },
            "retired repeated-call cap": {
                **valid,
                "agent_invocation": {"repeated_call": 3},
            },
            "unsupported adapter": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "provider": "unsupported",
                    }
                },
            },
            "blank model": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "model": " ",
                    }
                },
            },
            "invalid credential name": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "credential_env": "KEY=secret",
                    }
                },
            },
            "invalid endpoint port": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "endpoint": "https://api.example.test:bad",
                    }
                },
            },
            "missing endpoint hostname": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "endpoint": "https://:80",
                    }
                },
            },
            "endpoint query": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "endpoint": "https://api.example.test?token=secret",
                    }
                },
            },
            "endpoint fragment": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "endpoint": "https://api.example.test#secret",
                    }
                },
            },
            "endpoint trailing v1 slash": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "endpoint": "https://api.example.test/v1/",
                    }
                },
            },
            "endpoint extra path": {
                **valid,
                "model_roster": {
                    "qwen": {
                        **valid["model_roster"]["qwen"],
                        "endpoint": "https://api.example.test/chat/completions",
                    }
                },
            },
        }
        for name, value in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.json"
                path.write_text(json.dumps(value), encoding="utf-8")

                with self.assertRaises(RuntimeConfigError) as read_error:
                    read_runtime_config(path)
                with self.assertRaises(RuntimeConfigError) as load_error:
                    _resolve_through_registration(
                        path,
                        {"MODEL_KEY": "secret"},
                        agent_role="planning_agent",
                    )

                self.assertEqual(read_error.exception.code, "runtime_config_invalid")
                self.assertEqual(
                    read_error.exception.violations,
                    load_error.exception.violations,
                )
                # Each case changes one field; its path is the one violation.
                endpoint = "model_roster.qwen.endpoint"
                expected_path = {
                    "zero failed turns": "agent_invocation.failed_turns",
                    "coerced failed turns": "agent_invocation.failed_turns",
                    "retired repeated-call cap": "agent_invocation.repeated_call",
                    "unsupported adapter": "model_roster.qwen.provider",
                    "blank model": "model_roster.qwen.model",
                    "invalid credential name": "model_roster.qwen.credential_env",
                    "invalid endpoint port": endpoint,
                    "missing endpoint hostname": endpoint,
                    "endpoint query": endpoint,
                    "endpoint fragment": endpoint,
                    "endpoint trailing v1 slash": endpoint,
                    "endpoint extra path": endpoint,
                }[name]
                self.assertEqual(read_error.exception.violations, (expected_path,))

    def test_failure_section_registers_exactly_and_is_refused_by_path(self):
        # The failure module's own section, recorded at registration as
        # written.
        from bio_neb.application.commands import RegistrationError
        from bio_neb.failure import FailurePolicy, ModelContactPolicy
        from tests.planning_assignment_fakes import execution_configuration

        failure = {
            "operation": {"backoff_seconds": 2},
            "model_contact": {"max_wait_seconds": 90},
            # The development Run's total, the runtime's only limit.
            "limits": {"run_contacts": 60},
            "scopes": {
                "stage": {
                    "h100": {
                        "operation": {
                            "retry_attempts": 1,
                            "transient": "ask",
                        }
                    },
                    "vcf_observation": {"operation": {"retry_attempts": 2}},
                },
                "tool": {"fetch_refsnp": {"operation": {"transient": "ask"}}},
                "role": {"planning_agent": {"model_contact": {"contacts": 2}}},
            },
        }

        def register(root: Path, runtime: dict):
            (root / "prompt.txt").write_text("inspect my inputs", encoding="utf-8")
            (root / "r1.fastq").write_bytes(b"reads")
            (root / "r2.fastq").write_bytes(b"mates")
            (root / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
            return register_run(
                RegisterRunCommand(
                    run_root=str(root / "runs"),
                    prompt_file=str(root / "prompt.txt"),
                    runtime_config_file=str(root / "runtime.json"),
                    inputs=(
                        RegisterFastqInput(
                            role="read_1", registered_path=str(root / "r1.fastq")
                        ),
                        RegisterFastqInput(
                            role="read_2", registered_path=str(root / "r2.fastq")
                        ),
                    ),
                ),
                repository_root=Path.cwd(),
            )

        with tempfile.TemporaryDirectory() as directory:
            receipt = register(
                Path(directory), {**_attempt_configuration(), "failure": failure}
            )
            registration = next(
                e
                for e in Journey.open(Path(receipt.journey_file)).read()
                if e["event_type"] == "RunInputsRegistered"
            )
            self.assertEqual(
                registration["payload"]["runtime_selections"]["failure"], failure
            )

        with tempfile.TemporaryDirectory() as directory:
            parabricks = execution_configuration(Path(directory)).model_dump(
                mode="json"
            )
        base = _attempt_configuration()
        stage = {"operation": {"retry_attempts": 1}}
        refused = {
            # The old locations have no compatibility reading (clean cut).
            "parabricks.failure_policy": {
                **base,
                "parabricks": {
                    **parabricks,
                    "failure_policy": FailurePolicy().model_dump(mode="json"),
                },
            },
            "agent_invocation.model_contact": {
                **base,
                "agent_invocation": {
                    "model_contact": ModelContactPolicy().model_dump(mode="json")
                },
            },
            # An unknown surface, including an Agent stage, names its dict key.
            "failure.scopes.stage.parabricks_h100.[key]": {
                **base,
                "failure": {"scopes": {"stage": {"parabricks_h100": stage}}},
            },
            "failure.scopes.stage.input_correction.[key]": {
                **base,
                "failure": {"scopes": {"stage": {"input_correction": stage}}},
            },
            "failure.scopes.tool.request_planning_hitl.[key]": {
                **base,
                "failure": {"scopes": {"tool": {"request_planning_hitl": stage}}},
            },
            "failure.scopes.stage.h100.operation.retry_budget": {
                **base,
                "failure": {
                    "scopes": {"stage": {"h100": {"operation": {"retry_budget": 2}}}}
                },
            },
            # Model contact is per role (U8); no site knows its role's Tool (U7).
            "failure.scopes.stage.h100.model_contact": {
                **base,
                "failure": {
                    "scopes": {"stage": {"h100": {"model_contact": {"contacts": 2}}}}
                },
            },
            "failure.scopes.role.planning_agent.operation": {
                **base,
                "failure": {"scopes": {"role": {"planning_agent": {"operation": {}}}}},
            },
            "failure.operation": {**base, "failure": {"operation": None}},
            # A Tool's limits are code; the runtime names only the Run total.
            "failure.limits.contacts": {
                **base,
                "failure": {"limits": {"contacts": 5}},
            },
            "failure.scopes.tool.fetch_refsnp.limits": {
                **base,
                "failure": {
                    "scopes": {"tool": {"fetch_refsnp": {"limits": {"contacts": 5}}}}
                },
            },
        }
        for path, runtime in refused.items():
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(RegistrationError) as raised:
                    register(root, runtime)
                self.assertEqual(raised.exception.code, "runtime_selections_invalid")
                self.assertEqual(raised.exception.fields, (path,))
                self.assertFalse((root / "runs").exists())

    def test_retired_correction_fields_are_refused_as_runtime_config_invalid(self):
        valid = {
            "contract_version": "1.0",
            "agent_invocation": {"model_reference": "qwen"},
            "heartbeat_interval_seconds": 30,
            "model_roster": {
                "qwen": {
                    "provider": "nebius",
                    "endpoint": "https://api.example.test",
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "credential_env": "MODEL_KEY",
                }
            },
        }
        # The retired per-role action correction, the retired per-role
        # output-repair sections and the four per-kind allowances have no
        # compatibility reading; a Tool scope has no failure rule.
        retired = {
            f"retired {name}": {
                **valid,
                "agent_invocation": {"model_reference": "qwen", name: 2},
            }
            for name in (
                "same_call_correction",
                "repeated_call",
                "truncation_retry",
                "output_repair",
            )
        }
        cases = {
            **retired,
            "role output repair": {
                **valid,
                "agent_invocation": {
                    "model_reference": "qwen",
                    "scopes": {"role": {"report_explainer": {"output_repair": 2}}},
                },
            },
            "tool failed turns": {
                **valid,
                "agent_invocation": {
                    "model_reference": "qwen",
                    "scopes": {"tool": {"search_journey": {"failed_turns": 2}}},
                },
            },
            "planning action correction": {
                **valid,
                "agent_continuation": {
                    "planning_agent": {"output_repair": 0, "action_correction": 3}
                },
            },
            "investigation action correction": {
                **valid,
                "investigation_continuation": {
                    "output_repair": 1,
                    "action_correction": 1,
                },
            },
            "role action correction": {
                **valid,
                "agent_invocation": {
                    "model_reference": "qwen",
                    "scopes": {"role": {"planning_agent": {"action_correction": 3}}},
                },
            },
        }
        for name, value in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.json"
                path.write_text(json.dumps(value), encoding="utf-8")

                with self.assertRaises(RuntimeConfigError) as raised:
                    _resolve_through_registration(
                        path,
                        {"MODEL_KEY": "secret"},
                        agent_role="planning_agent",
                    )
                self.assertEqual(raised.exception.code, "runtime_config_invalid")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            for role in ("planning_agent", "investigation_planner"):
                with self.subTest(role=role):
                    continuation = _resolve_through_registration(
                        path, {"MODEL_KEY": "secret"}, agent_role=role
                    ).invocation_policy.continuation
                    from bio_neb.agentic_journey.continuation_policy import (
                        ContinuationPolicy,
                    )

                    # One failure rule: the role default's four failed turns.
                    self.assertEqual(continuation, ContinuationPolicy(failed_turns=4))

    def test_extra_fields_fail_before_credential_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(
                json.dumps(
                    {
                        "contract_version": "1.0",
                        "agent_invocation": {"model_reference": "qwen"},
                        "heartbeat_interval_seconds": 30,
                        "model_roster": {
                            "qwen": {
                                "provider": "nebius",
                                "endpoint": "https://api.example.test",
                                "model": "Qwen/Qwen3.5-397B-A17B",
                                "credential_env": "MISSING_TOKEN",
                                "api_key": "must-not-be-accepted",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeConfigError) as raised:
                _resolve_through_registration(path, {}, agent_role="planning_agent")

            self.assertEqual(raised.exception.code, "runtime_config_invalid")
            self.assertIn("model_roster.qwen.api_key", raised.exception.violations)

    def test_endpoint_userinfo_is_rejected_as_secret_bearing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(
                json.dumps(
                    {
                        "contract_version": "1.0",
                        "agent_invocation": {"model_reference": "qwen"},
                        "heartbeat_interval_seconds": 30,
                        "model_roster": {
                            "qwen": {
                                "provider": "nebius",
                                "endpoint": "https://user:password@api.example.test",
                                "model": "Qwen/Qwen3.5-397B-A17B",
                                "credential_env": "NEBIUS_TOKEN",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeConfigError) as raised:
                _resolve_through_registration(
                    path,
                    {"NEBIUS_TOKEN": "secret"},
                    agent_role="planning_agent",
                )

            self.assertEqual(raised.exception.code, "runtime_config_invalid")
            self.assertIn("model_roster.qwen.endpoint", raised.exception.violations)

    def test_nonfinite_or_nonpositive_heartbeat_is_rejected(self):
        for heartbeat in (float("inf"), float("nan"), 0, -1, True, "30"):
            with (
                self.subTest(heartbeat=heartbeat),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "runtime.json"
                value = {
                    "contract_version": "1.0",
                    "agent_invocation": {"model_reference": "qwen"},
                    "heartbeat_interval_seconds": heartbeat,
                    "model_roster": {
                        "qwen": {
                            "provider": "nebius",
                            "endpoint": "https://api.example.test/v1",
                            "model": "Qwen/Qwen3.5-397B-A17B",
                            "credential_env": "MODEL_KEY",
                        }
                    },
                }
                path.write_text(json.dumps(value), encoding="utf-8")

                with self.assertRaises(RuntimeConfigError) as raised:
                    _resolve_through_registration(
                        path,
                        {"MODEL_KEY": "dummy"},
                        agent_role="planning_agent",
                    )

                self.assertIn("heartbeat_interval_seconds", raised.exception.violations)

    def test_attempt_runtime_preflight_refuses_invalid_changed_or_unavailable_config(
        self,
    ):
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            QWEN_ENDPOINT_POLICY,
            model_registry,
            use_model_registry,
        )

        # A model listed when the Run registered and delisted since is refused
        # when its Attempt resolves, by its path.
        delisted = replace(
            QWEN_ENDPOINT_POLICY,
            settings_selection=QWEN_ENDPOINT_POLICY.settings_selection.model_copy(
                update={"model": "delisted/model"}
            ),
        )
        cases = (
            (
                "registered model no longer listed",
                "delisted/model",
                {"LOCAL_MODEL_KEY": "registered-secret"},
                "runtime_config_invalid",
                "registered-secret",
            ),
            (
                "credential unavailable",
                "Qwen/Qwen3.5-397B-A17B",
                {"UNRELATED_MODEL_KEY": "unrelated-secret"},
                "model_credential_unavailable",
                "unrelated-secret",
            ),
        )
        for name, model, environ, code, forbidden in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                with use_model_registry(model_registry().replacing(delisted)):
                    receipt, _config = _registered_attempt_fixture(root, model=model)
                constructed = False

                def forbidden_factory(loaded, _selected):
                    nonlocal constructed
                    del loaded
                    constructed = True
                    raise AssertionError("model must not be constructed")

                with self.assertRaises(Invariant) as held:
                    planning_assignment(model_factory=forbidden_factory).run(
                        prepare_input_correction(
                            Journey.open(Path(receipt.journey_file))
                        ),
                        environ=environ,
                    )

                self.assertFalse(constructed)
                self.assertIsNone(
                    review_journey(
                        ReviewJourneyQuery(journey_file=receipt.journey_file)
                    ).attempt
                )
                _assert_configuration_failure_held(
                    self, receipt, held.exception, code=code
                )
                events = read_evidence(Journey.open(Path(receipt.journey_file)))
                self.assertFalse(
                    any(
                        event["event_type"] == "ModelPhysicalContactStarted"
                        for event in events
                    )
                )
                if forbidden is not None:
                    self.assertNotIn(forbidden, json.dumps(events))


if __name__ == "__main__":
    unittest.main()


class RetiredWorkPolicyKeyTests(unittest.TestCase):
    """#105's bounded runtime-file tolerance for the retired operational route."""

    def test_a_runtime_file_naming_the_retired_route_reads_and_journey_facts_refuse(
        self,
    ):
        from bio_neb.agentic_journey.continuation_policy import (
            InvestigationWorkPolicy,
        )
        from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration
        from tests.proposal_information_fakes import runtime_configuration

        retired = {
            "item_disposition": "retain_classified_outcome",
            "other_work": "continue_eligible",
            "operational_failure": "stop_shared_or_unknown",
            "unresolved_identity": "block_finalization",
            "completion": "allow_partial_with_limitations",
        }
        # A registered file from an older release reads as today's
        # policy.
        read = RuntimeConfiguration.model_validate(
            {**runtime_configuration(), "investigation_work": retired}
        )
        self.assertEqual(
            read.investigation_work,
            InvestigationWorkPolicy(completion="allow_partial_with_limitations"),
        )
        # Any other value of the retired key still refuses.
        with self.assertRaises(ValidationError):
            RuntimeConfiguration.model_validate(
                {
                    **runtime_configuration(),
                    "investigation_work": {**retired, "operational_failure": "other"},
                }
            )
        # A Journey fact that recorded it keeps refusing.
        with self.assertRaises(ValidationError):
            InvestigationWorkPolicy.model_validate(retired)


class BuildDefaultsTests(unittest.TestCase):
    """A Run configured from nothing, or from a partial file, takes the build's
    typed defaults; only the values a file names change (#440)."""

    SUPER_VL = "nvidia/Nemotron-3.5-Super-VL"
    QWEN = "Qwen/Qwen3.5-397B-A17B"

    def _read(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return read_runtime_config(path)

    def test_a_configuration_from_nothing_is_the_registry_default_entry(self):
        configuration = read_runtime_config(None)
        self.assertEqual(
            configuration.model_roster["nemotron"].model_dump(
                mode="json", exclude={"settings_selection"}
            ),
            {
                "provider": "nebius",
                "endpoint": "https://api.tokenfactory.us-north1.nebius.com/v1",
                "model": self.SUPER_VL,
                "credential_env": "NEBIUS_API_KEY",
            },
        )
        self.assertEqual(set(configuration.model_roster), {"nemotron"})
        invocation = configuration.agent_invocation
        self.assertEqual(invocation.model_reference, "nemotron")
        # The entry states no failure rule: the role default's applies.
        self.assertIsNone(invocation.failed_turns)
        # The one Tool choice is recorded once, globally.
        self.assertEqual(invocation.tool_choice, "auto")
        self.assertEqual(invocation.scopes.role, {})
        self.assertEqual(configuration.heartbeat_interval_seconds, 30.0)
        self.assertIsNone(configuration.parabricks)
        self.assertEqual(
            configuration.investigation_tools.model_dump(mode="json"),
            {
                tool: {"package": "pysam", "package_version": "0.24.1"}
                for tool in ("bcftools", "samtools")
            },
        )

    def test_a_partial_file_overrides_only_what_it_names(self):
        everything = read_runtime_config(None).model_dump(mode="json")
        partial = self._read(
            {"contract_version": "1.0", "heartbeat_interval_seconds": 5.0}
        )
        self.assertEqual(
            partial.model_dump(mode="json"),
            {**everything, "heartbeat_interval_seconds": 5.0},
        )

    def test_a_sparse_object_override_keeps_the_values_it_does_not_name(self):
        everything = read_runtime_config(None).model_dump(mode="json")
        roster = self._read(
            {
                "contract_version": "1.0",
                "model_roster": {"nemotron": {"credential_env": "OTHER_MODEL_KEY"}},
            }
        )
        self.assertEqual(
            roster.model_dump(mode="json")["model_roster"],
            {
                "nemotron": {
                    **everything["model_roster"]["nemotron"],
                    "credential_env": "OTHER_MODEL_KEY",
                }
            },
        )
        named = {"package": "pysam", "package_version": "0.24.0"}
        tools = self._read(
            {"contract_version": "1.0", "investigation_tools": {"bcftools": named}}
        )
        self.assertEqual(
            tools.model_dump(mode="json")["investigation_tools"],
            {**everything["investigation_tools"], "bcftools": named},
        )
        # A roster key the build does not have names its own model.
        with self.assertRaises(RuntimeConfigError) as refused:
            self._read(
                {
                    "contract_version": "1.0",
                    "model_roster": {"other": {"credential_env": "OTHER_MODEL_KEY"}},
                }
            )
        self.assertIn("model_roster.other.model", refused.exception.violations)

    def test_a_file_selecting_qwen_names_only_the_model_and_takes_its_entry(self):
        configuration = self._read(
            {
                "contract_version": "1.0",
                "agent_invocation": {"model_reference": "qwen"},
                "model_roster": {"qwen": {"model": self.QWEN}},
            }
        )
        # The roster merges by key into the default roster.
        self.assertEqual(set(configuration.model_roster), {"nemotron", "qwen"})
        qwen = configuration.model_roster["qwen"]
        self.assertEqual(
            (qwen.endpoint, qwen.credential_env),
            ("https://api.tokenfactory.us-central1.nebius.com/v1", "NEBIUS_API_KEY"),
        )
        invocation = configuration.agent_invocation
        self.assertIsNone(invocation.failed_turns)
        # Qwen names no Tool choice, so its roles keep the default rule.
        self.assertEqual(
            {role: entry.tool_choice for role, entry in invocation.scopes.role.items()},
            {},
        )

    def test_an_unlisted_model_is_refused_at_configuration_by_its_path(self):
        with self.assertRaises(RuntimeConfigError) as raised:
            self._read(
                {
                    "contract_version": "1.0",
                    "model_roster": {
                        "nemotron": {
                            "provider": "nebius",
                            "endpoint": "http://127.0.0.1:18089/v1",
                            "model": "unlisted/model",
                            "credential_env": "LOCAL_MODEL_KEY",
                        }
                    },
                }
            )
        self.assertEqual(
            (raised.exception.code, raised.exception.violations),
            ("runtime_config_invalid", ("model_roster.nemotron.model",)),
        )

    def test_the_registry_marks_one_default_entry_and_no_entry_states_failed_turns(
        self,
    ):
        from dataclasses import replace

        from bio_neb.agentic_journey.model_settings import (
            BUILD_MODEL_REGISTRY,
            ModelUnlisted,
            model_endpoint_policy,
        )

        registry = BUILD_MODEL_REGISTRY
        default = registry.default
        self.assertEqual(
            (
                default.model,
                default.endpoint,
                default.tool_choice,
                registry.default_reference,
            ),
            (
                self.SUPER_VL,
                "https://api.tokenfactory.us-north1.nebius.com/v1",
                None,
                "nemotron",
            ),
        )
        # One failure rule: no entry states its own number, so a fresh Run takes
        # none from its model.
        self.assertEqual(
            {model: entry.failed_turns for model, entry in registry.entries.items()},
            {
                model: None
                for model in (
                    self.SUPER_VL,
                    self.QWEN,
                    "nvidia/Nemotron-3-Ultra-550b-a55b",
                )
            },
        )
        # Every entry takes the one default Tool choice.
        self.assertEqual(default.run_defaults(), {"tool_choice": "auto"})
        self.assertEqual(
            registry.entries[self.QWEN].run_defaults(), {"tool_choice": "auto"}
        )
        # An entry that states one makes it a Run default.
        self.assertEqual(
            replace(default, failed_turns=6).run_defaults(),
            {"failed_turns": 6, "tool_choice": "auto"},
        )
        with self.assertRaises(ModelUnlisted):
            model_endpoint_policy("unlisted/model")

    def test_the_default_investigation_tools_are_the_locked_package(self):
        import re

        pin = re.search(
            r'"pysam==([^"]+)"',
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
                encoding="utf-8"
            ),
        )
        tools = read_runtime_config(None).investigation_tools
        self.assertEqual(
            {tools.bcftools.package_version, tools.samtools.package_version},
            {pin.group(1)},
        )


class BoundModelPolicyTests(unittest.TestCase):
    """A Run begun with no runtime file registers the build's resolved values;
    each Attempt resolves from them alone and binds its model policy once (#440)."""

    SUPER_VL = "nvidia/Nemotron-3.5-Super-VL"

    def _command(self, root: Path, **values):
        prompt = root / "prompt.txt"
        fastqs = (root / "r1.fastq", root / "r2.fastq")
        prompt.write_text("inspect my registered inputs", encoding="utf-8")
        for index, fastq in enumerate(fastqs):
            fastq.write_bytes(f"reads-{index}".encode())
        return RegisterRunCommand(
            run_root=str(root / "runs"),
            prompt_file=str(prompt),
            inputs=tuple(
                RegisterFastqInput(role=f"read_{index + 1}", registered_path=str(path))
                for index, path in enumerate(fastqs)
            ),
            **values,
        )

    def test_a_run_begun_with_no_file_registers_the_default_entry_as_resolved_values(
        self,
    ):
        from bio_neb.agentic_journey.agent_roles import RuntimeBinding
        from bio_neb.agentic_journey.attempt_runner import resolve_attempt_configuration
        from bio_neb.agentic_journey.continuation_policy import (
            AgentInvocationOverrides,
        )
        from bio_neb.agentic_journey.model_settings import BUILD_MODEL_REGISTRY
        from bio_neb.application.input_journey import RunInputsRegistered

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = register_run(self._command(root), repository_root=Path.cwd())
            events = read_evidence(Journey.open(Path(receipt.journey_file)))
        registered = next(
            e["payload"] for e in events if e["event_type"] == "RunInputsRegistered"
        )
        self.assertNotIn("runtime_config_file", registered)
        selections = registered["runtime_selections"]
        self.assertEqual(
            {
                key: value
                for key, value in selections["model_roster"]["nemotron"].items()
                if key != "settings_selection"
            },
            {
                "provider": "nebius",
                "endpoint": "https://api.tokenfactory.us-north1.nebius.com/v1",
                "model": self.SUPER_VL,
                "credential_env": "NEBIUS_API_KEY",
            },
        )
        invocation = selections["agent_invocation"]
        self.assertEqual(invocation["model_reference"], "nemotron")
        # The entry states no failure rule, so the registration records none.
        self.assertNotIn("failed_turns", invocation)
        # The one Tool choice is registered once, globally.
        self.assertEqual(invocation["tool_choice"], "auto")
        self.assertEqual(invocation["scopes"]["role"], {})
        # A later Attempt resolves from the registration alone: there is no file.
        fact = RunInputsRegistered.model_validate(
            {"event_type": "RunInputsRegistered", **registered}
        )
        configuration = resolve_attempt_configuration(
            RuntimeBinding(fact.runtime_selections, fact.input_set_id),
            {"NEBIUS_API_KEY": "offline"},
            agent_role="planning_agent",
            overrides=AgentInvocationOverrides(),
        )
        self.assertIs(configuration.model, BUILD_MODEL_REGISTRY.default)
        self.assertEqual(
            configuration.effective_policy.continuation.model_dump(mode="json"),
            {"failed_turns": 4},
        )

    def test_begin_refuses_a_missing_credential_by_name_before_any_journey(self):
        from bio_neb.application.advancement import (
            BeginRun,
            advance,
            use_run_advancement_dependencies,
        )
        from bio_neb.application.commands import RegistrationError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = self._command(root)
            with (
                use_run_advancement_dependencies(environ={"OTHER_KEY": "x"}),
                self.assertRaises(RegistrationError) as refused,
            ):
                advance(BeginRun(**command.model_dump()))
            self.assertFalse((root / "runs").exists())
        self.assertEqual(
            (refused.exception.code, refused.exception.fields),
            ("model_credential_unavailable", ("model_roster.nemotron.credential_env",)),
        )

    def test_the_bound_policy_reaches_the_roles_assembly_and_adapter(self):
        from dataclasses import replace

        from pydantic_ai.messages import ModelResponse, TextPart

        from bio_neb.agentic_journey.model_settings import (
            QWEN_ENDPOINT_POLICY,
            model_registry,
            use_model_registry,
        )
        from bio_neb.failure import CorrectionWording

        # A fake entry for the Run's model, passed through the public registry
        # seam: the Attempt binds it once, and every reader takes it from there.
        fake = replace(
            QWEN_ENDPOINT_POLICY,
            instruction_form="nemotron_3_5_super",
            correction_wording=CorrectionWording(
                text_ending="FAKE ENTRY: call one of {offered}."
            ),
        )
        adapters = []
        received = []

        async def text_only(messages, info):
            received.append(json.dumps([str(message) for message in messages]))
            return ModelResponse(parts=[TextPart("no call")])

        def factory(loaded, selected):
            self.assertIs(loaded.model, fake)
            adapter = ScriptedModelEndpoint(text_only)(loaded, selected)
            adapters.append(adapter)
            return adapter

        with tempfile.TemporaryDirectory() as directory:
            receipt, config = _registered_attempt_fixture(Path(directory).resolve())
            config.unlink()
            journey = Journey.open(Path(receipt.journey_file))
            with use_model_registry(model_registry().replacing(fake)):
                planning_assignment(model_factory=factory).run(
                    prepare_input_correction(journey),
                    environ={"LOCAL_MODEL_KEY": "secret"},
                )
            events = read_evidence(journey)
        self.assertEqual([adapter.endpoint_policy for adapter in adapters], [fake])
        started = next(
            e["payload"] for e in events if e["event_type"] == "AttemptStarted"
        )
        self.assertEqual(
            started["instruction_identity"], "planning_agent:v22:nemotron_3_5_super"
        )
        self.assertGreater(len(received), 1)
        self.assertIn("FAKE ENTRY: call one of", received[1])


class ParabricksDefaultsTests(unittest.TestCase):
    """A Run with no Parabricks section registers the build's defaults, the
    account file's values and its own per-Run leaves (#454)."""

    RUN_ID = "454-live-20261024T112855Z-ac17e721-a1"

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.account_file = self.root / "parabricks-account.json"
        self.account = parabricks_account(self.root)
        self.account_file.write_text(json.dumps(self.account), encoding="utf-8")

    def _command(self, run_id: str = RUN_ID, **values) -> RegisterRunCommand:
        prompt = self.root / "prompt.txt"
        fastqs = (self.root / "r1.fastq", self.root / "r2.fastq")
        prompt.write_text("inspect my registered inputs", encoding="utf-8")
        for index, fastq in enumerate(fastqs):
            fastq.write_bytes(f"reads-{index}".encode())
        return RegisterRunCommand(
            run_root=str(self.root / "live-runs" / run_id),
            prompt_file=str(prompt),
            inputs=tuple(
                RegisterFastqInput(role=f"read_{index + 1}", registered_path=str(path))
                for index, path in enumerate(fastqs)
            ),
            **values,
        )

    def _registered(self, command, **values) -> tuple[list, dict | None]:
        receipt = register_run(command, repository_root=Path.cwd(), **values)
        events = read_evidence(Journey.open(Path(receipt.journey_file)))
        configured = [
            event
            for event in events
            if event["event_type"] == "ParabricksConfigurationRegistered"
        ]
        self.assertLessEqual(len(configured), 1)
        return events, configured[0] if configured else None

    def _runtime_file(self, document: dict) -> str:
        path = self.root / f"runtime-{len(list(self.root.glob('runtime-*')))}.json"
        path.write_text(json.dumps({"contract_version": "1.0", **document}))
        return str(path)

    def test_no_section_registers_the_defaults_the_account_and_the_runs_own_leaves(
        self,
    ):
        events, configured = self._registered(
            self._command(), parabricks_account=self.account_file
        )

        run_root = self.root / "live-runs" / self.RUN_ID
        token = "454-live-20261024t112855z-ac17e721-a1"
        registration = next(
            e for e in events if e["event_type"] == "RunInputsRegistered"
        )
        self.assertEqual(configured["parent_ids"], [registration["event_id"]])
        self.assertNotIn("runtime_config_file", registration["payload"])
        account = self.account["launch"]
        self.assertEqual(
            configured["payload"]["configuration"],
            {
                "launch": {
                    # Per Run, from the Run root.
                    "object_scope": token,
                    "object_storage": {
                        **account["object_storage"],
                        "bucket": f"bio-neb-{token}",
                        "mount_path": "/mnt/object-storage",
                    },
                    "work_directory": str(run_root / "runs" / "execution" / "work"),
                    # Per account.
                    "nebius": account["nebius"],
                    "context": account["context"],
                    "cleanup_owner": account["cleanup_owner"],
                    # The build's.
                    "local_reference_source": None,
                    "heartbeat_interval_seconds": 5,
                    "bounded_get_bytes": 8 * 1024 * 1024,
                    "verification_mode": "reuse_creation_proof",
                    "verification_overrides": {},
                },
                "observation_interval_seconds": 5,
                "runtime_selection": self.account["runtime_selection"],
            },
        )

    def test_a_files_section_is_registered_unchanged_in_the_same_shape(self):
        _, defaulted = self._registered(
            self._command(), parabricks_account=self.account_file
        )
        section = defaulted["payload"]["configuration"]

        # The same section from a file, for another Run, with the account present:
        # the file's section wins whole and is not refilled.
        _, supplied = self._registered(
            self._command(
                "454-live-20261024T120000Z-ac17e721-a2",
                runtime_config_file=self._runtime_file({"parabricks": section}),
            ),
            parabricks_account=self.account_file,
        )

        self.assertEqual(supplied["payload"], defaulted["payload"])

    def test_a_file_without_a_section_takes_the_default_section(self):
        _, configured = self._registered(
            self._command(
                runtime_config_file=self._runtime_file(
                    {"heartbeat_interval_seconds": 15}
                )
            ),
            parabricks_account=self.account_file,
        )

        self.assertEqual(
            configured["payload"]["configuration"]["launch"]["object_scope"],
            "454-live-20261024t112855z-ac17e721-a1",
        )

    def test_a_missing_or_invalid_account_registers_no_section(self):
        from bio_neb.parabricks.preparation_review import information_gap

        extra = {**self.account, "launch": {**self.account["launch"], "x": 1}}
        missing = {
            **self.account,
            "launch": {
                key: value
                for key, value in self.account["launch"].items()
                if key != "cleanup_owner"
            },
        }
        per_run = {
            **self.account,
            "launch": {**self.account["launch"], "object_scope": "other/run"},
        }
        mismatched = json.loads(json.dumps(self.account))
        mismatched["launch"]["context"]["topology"]["project_id"] = "project-other"
        for name, content in (
            ("absent", None),
            ("not json", "{"),
            ("not an object", "[]"),
            ("an extra key", json.dumps(extra)),
            ("a missing key", json.dumps(missing)),
            ("a per-Run key", json.dumps(per_run)),
            ("an invalid composition", json.dumps(mismatched)),
        ):
            with self.subTest(name):
                account_file = self.root / f"account-{name}.json"
                if content is not None:
                    account_file.write_text(content, encoding="utf-8")
                events, configured = self._registered(
                    self._command(f"454-live-20261024T112855Z-ac17e721-{name[:3]}"),
                    parabricks_account=account_file,
                )
                self.assertIsNone(configured)
                self.assertEqual(
                    information_gap(events),
                    (
                        "execution_configuration_required",
                        ("launch", "runtime_selection"),
                    ),
                )

    def test_an_account_whose_runtime_is_not_this_build_s_worker_is_refused(self):
        """#464 live a2 F4: the lane's account file named the i300b image (request /6,
        worker b3323f4e) while the build sends /7 to worker 0447bb17; the old worker
        refused the paid H100 job's request. An account whose runtime image is not
        this build's worker (request schema or worker sha256) is refused before
        `begin` registers it, and readiness blocks on it."""
        from bio_neb.content import canonical_json_digest
        from bio_neb.parabricks.configuration import (
            ParabricksAccountRefused,
            default_configuration,
        )
        from bio_neb.parabricks.execution_runtime import ATTESTED_WORKER_SHA256

        old_schema = "bio-neb.parabricks-worker-request/6"
        old_worker = "b" * 64
        for name, (old, new) in {
            "request schema": (None, old_schema),
            "worker sha256": (ATTESTED_WORKER_SHA256, old_worker),
        }.items():
            with self.subTest(name):
                text = json.dumps(self.account)
                if old is None:
                    text = text.replace("bio-neb.parabricks-worker-request/7", new)
                else:
                    text = text.replace(old, new)
                self.assertNotEqual(text, json.dumps(self.account))
                # A consistent older account, as a2's was: its snapshot id re-minted.
                document = json.loads(text)
                capability = document["runtime_selection"]["capability"]
                stale_id = capability["snapshot_id"]
                text = text.replace(
                    stale_id,
                    canonical_json_digest(
                        {k: v for k, v in capability.items() if k != "snapshot_id"}
                    ),
                )
                account_file = self.root / f"account-{name[:7]}.json"
                account_file.write_text(text, encoding="utf-8")
                with self.assertRaises(ParabricksAccountRefused) as refused:
                    default_configuration(account_file, str(self.root / "run"))
                self.assertEqual(refused.exception.code, "runtime_not_this_build")

    def test_an_account_naming_a_prepared_candidate_is_refused(self):
        """Owner decision 2026-10-29 (#464): a Run's index is the Execution
        Review's bound prebuilt index or one CPU job, so an account file that
        still names #107's prepared candidate is refused by that field, and the
        Run registers no section."""
        from bio_neb.parabricks.configuration import (
            ParabricksAccountRefused,
            default_configuration,
        )
        from bio_neb.parabricks.preparation_review import information_gap

        candidate = {
            "endpoint_url": "https://storage.eu-north1.example.test",
            "region": "eu-north1",
            "bucket": "prepared",
            "manifest_object_key": "prepared/manifest.json",
            "manifest_byte_count": 32,
            "manifest_sha256": "a" * 64,
            "inventory_prefix": "prepared/",
        }
        for value in (candidate, None):
            with self.subTest(value=value):
                account = json.loads(json.dumps(self.account))
                account["launch"]["remote_prepared_candidate"] = value
                account_file = self.root / "account-with-candidate.json"
                account_file.write_text(json.dumps(account), encoding="utf-8")
                with self.assertRaises(ParabricksAccountRefused) as refused:
                    default_configuration(account_file, str(self.root / "run"))
                self.assertEqual(
                    (refused.exception.code, refused.exception.fields),
                    (
                        "parabricks_account_invalid",
                        ("launch.remote_prepared_candidate",),
                    ),
                )
                events, configured = self._registered(
                    self._command(
                        "454-live-20261024T112855Z-ac17e721-"
                        + ("can" if value else "nul")
                    ),
                    parabricks_account=account_file,
                )
                self.assertIsNone(configured)
                self.assertEqual(
                    information_gap(events),
                    (
                        "execution_configuration_required",
                        ("launch", "runtime_selection"),
                    ),
                )

    def test_no_account_path_registers_no_section(self):
        _, configured = self._registered(self._command())

        self.assertIsNone(configured)

    def test_per_run_leaves_are_lawful_for_any_run_root_name(self):
        from bio_neb.parabricks.configuration import (
            ParabricksConfiguration,
            default_configuration,
        )

        long_id = "454-" + "a-very-long-purpose" * 4 + "-20261024T112855Z-ac17e721-a1"
        buckets = set()
        for name in (self.RUN_ID, "My Run_2", long_id, long_id + "0"):
            with self.subTest(name):
                configuration = default_configuration(
                    self.account_file, str(self.root / name)
                )
                self.assertIsInstance(configuration, ParabricksConfiguration)
                bucket = configuration.launch.object_storage.bucket
                self.assertLessEqual(len(bucket), 63)
                self.assertTrue(bucket.startswith("bio-neb-"))
                buckets.add(bucket)
        self.assertEqual(len(buckets), 4)
        self.assertEqual(
            default_configuration(
                self.account_file, str(self.root / "My Run_2")
            ).launch.object_scope,
            "my-run-2",
        )

    def test_the_product_begin_reads_the_operator_homes_account_file(self):
        from bio_neb.application import advancement

        class Registered(Exception):
            pass

        received = {}

        def register(command, **values):
            received.update(values)
            raise Registered

        home = self.root / "home"
        with (
            patch.dict("os.environ", {"BIO_NEB_CONTROLLER_HOME": str(home)}),
            patch.object(advancement, "register_run", register),
            advancement.use_run_advancement_dependencies(
                environ={"NEBIUS_API_KEY": "offline"}
            ),
            self.assertRaises(Registered),
        ):
            advancement.advance(advancement.BeginRun(**self._command().model_dump()))

        self.assertEqual(
            received["parabricks_account"], home / "parabricks-account.json"
        )
