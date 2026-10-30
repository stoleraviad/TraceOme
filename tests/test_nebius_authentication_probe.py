from __future__ import annotations

import unittest
from unittest.mock import patch

from bio_neb.providers.nebius_authentication import (
    ADAPTER_ENTRY_AUTHENTICATION_SCHEMA,
    NebiusAuthenticationProbe,
)
from bio_neb.providers.nebius_common import (
    AUTHENTICATION_EXIT_CODES,
    AmbiguousNebiusCreate,
    NebiusAuthenticationRequired,
    NebiusCliError,
    NebiusCliFailureEvidence,
    NebiusTenantNotActive,
    nebius_common_arguments,
)
from tests.nebius_serverless_job_contact_fakes import _target

_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_WHOAMI = {
    "user_profile": {
        "tenants": [
            {
                "tenant_id": "tenant-exact",
                "tenant_user_account_state": "ACTIVE",
            },
        ],
    },
}


def _service_account_whoami(*, active=True, parent_id="project-exact") -> dict:
    return {
        "service_account_profile": {
            "info": {
                "metadata": {"id": "serviceaccount-exact", "parent_id": parent_id},
                "spec": {},
                "status": {"active": active},
            }
        }
    }


def _evidence(child_exit_code: int, classification: str) -> NebiusCliFailureEvidence:
    return NebiusCliFailureEvidence(
        operation="iam.whoami",
        child_exit_code=child_exit_code,
        stdout_byte_count=0,
        stdout_sha256=_EMPTY_SHA256,
        stderr_byte_count=96,
        stderr_sha256="c" * 64,
        stderr_classification=classification,
    )


class _RecordingCli:
    """The exact seam `NebiusCliRunner.json` presents to an Adapter."""

    def __init__(self, error: BaseException | None = None, payload=None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._error = error
        self._payload = _WHOAMI if payload is None else payload

    def json(self, *arguments: str) -> dict:
        self.calls.append(arguments)
        if self._error is not None:
            raise self._error
        return self._payload


def _probe(cli: _RecordingCli) -> tuple[NebiusAuthenticationProbe, list[dict]]:
    emitted: list[dict] = []
    probe = NebiusAuthenticationProbe(
        cli_runner=cli,
        target=_target(),
        emit=emitted.append,
    )
    return probe, emitted


class NebiusAuthenticationProbeTests(unittest.TestCase):
    def test_entry_probe_is_one_read_only_call_with_a_stated_auth_timeout(self) -> None:
        cli = _RecordingCli()
        probe, emitted = _probe(cli)

        probe()

        common = nebius_common_arguments(_target())
        self.assertEqual(cli.calls, [("iam", "whoami", *common)])
        self.assertEqual(
            cli.calls[0][2:],
            (
                "--profile",
                "accepted-profile",
                "--retries",
                "1",
                "--auth-timeout",
                "15m0s",
                "--no-check-update",
                "--no-progress",
                "--format",
                "json",
            ),
        )
        self.assertEqual(emitted[0]["auth_timeout"], "15m0s")
        self.assertEqual(emitted[-1]["outcome"], "authenticated")
        self.assertEqual(emitted[-1]["child_exit_code"], 0)

    def test_one_whoami_serves_the_entry_probe_and_every_later_stage(self) -> None:
        """Ticket 144: the Run's `iam whoami` happens exactly once."""

        cli = _RecordingCli()
        probe, emitted = _probe(cli)

        probe()
        probe()
        probe.prove(_target())

        self.assertEqual(len(cli.calls), 1)
        self.assertEqual(len(emitted), 2)

    def test_a_different_selection_is_never_covered_by_an_earlier_proof(self) -> None:
        cli = _RecordingCli()
        probe, _ = _probe(cli)
        other = _target().model_copy(update={"profile": "other-profile"})

        probe()
        probe.prove(other)

        self.assertEqual(len(cli.calls), 2)
        self.assertEqual(cli.calls[1][3], "other-profile")

    def test_authentication_scope_and_failures_do_not_leak_between_invocations(self):
        from dataclasses import replace

        from bio_neb.providers.nebius_common import NebiusAccessScope

        scope = NebiusAccessScope(
            "selected-profile", "selected-tenant", "selected-project"
        )
        operations = []
        ready = False

        class Runner:
            def json(inner, *args):
                operations.append(args)
                if args[:2] == ("iam", "whoami"):
                    return {
                        "user_profile": {
                            "user_account_state": "ACTIVE",
                            "tenants": [
                                {
                                    "tenant_id": tenant,
                                    "tenant_user_account_state": "ACTIVE",
                                }
                                for tenant in ("selected-tenant", "other-tenant")
                            ],
                        }
                    }
                return {
                    "metadata": {
                        "id": args[4],
                        "parent_id": scope.tenant_id if ready else "wrong-tenant",
                    }
                }

        runner = Runner()
        probe = NebiusAuthenticationProbe(cli_runner=runner, emit=lambda _: None)
        with self.assertRaisesRegex(
            NebiusCliError, "provider_authentication_unavailable"
        ):
            probe.authenticate(scope)
        ready = True
        self.assertEqual(probe.authenticate(scope), "user_profile")
        probe.authenticate(scope)
        self.assertEqual(
            len(operations), 4
        )  # Failed project proof did not cache identity.
        for field in ("profile", "tenant_id", "project_id"):
            scope = replace(scope, **{field: "other-" + field})
            if field == "tenant_id":
                scope = replace(scope, tenant_id="other-tenant")
            probe.authenticate(scope)
        self.assertEqual(len(operations), 10)
        NebiusAuthenticationProbe(cli_runner=runner, emit=lambda _: None).authenticate(
            scope
        )
        self.assertEqual(
            len(operations), 12
        )  # A new invocation proves the same scope anew.

    def test_every_record_declares_the_contact_and_spends_no_create_ceiling(
        self,
    ) -> None:
        cli = _RecordingCli()
        probe, emitted = _probe(cli)

        probe()

        self.assertEqual(
            [record["kind"] for record in emitted],
            ["authentication_probe_started", "authentication_probe_finished"],
        )
        for record in emitted:
            with self.subTest(kind=record["kind"]):
                self.assertEqual(
                    record["schema_version"], ADAPTER_ENTRY_AUTHENTICATION_SCHEMA
                )
                self.assertEqual(record["operation"], "iam.whoami")
                self.assertEqual(record["provider_contact_count"], 1)
                self.assertEqual(record["provider_mutation_count"], 0)
                self.assertEqual(record["create_ceiling_consumed"], 0)
        self.assertIs(emitted[0]["human_authentication_may_be_required"], True)

    def test_the_started_record_precedes_the_call_that_may_stall(self) -> None:
        order: list[str] = []

        class _Stalling(_RecordingCli):
            def json(self, *arguments: str) -> dict:
                order.append("contact")
                return super().json(*arguments)

        probe = NebiusAuthenticationProbe(
            cli_runner=_Stalling(),
            target=_target(),
            emit=lambda record: order.append(str(record["kind"])),
        )

        probe()

        self.assertEqual(
            order,
            [
                "authentication_probe_started",
                "contact",
                "authentication_probe_finished",
            ],
        )

    def test_documented_authentication_exit_codes_halt_with_their_own_name(
        self,
    ) -> None:
        self.assertEqual(AUTHENTICATION_EXIT_CODES, frozenset({7, 47}))
        for child_exit_code in sorted(AUTHENTICATION_EXIT_CODES):
            with self.subTest(child_exit_code=child_exit_code):
                cli = _RecordingCli(
                    NebiusCliError(
                        "provider_command_failed",
                        failure_evidence=_evidence(
                            child_exit_code, "interactive_authentication_required"
                        ),
                    )
                )
                probe, emitted = _probe(cli)

                with self.assertRaises(NebiusAuthenticationRequired) as raised:
                    probe()

                self.assertEqual(
                    raised.exception.code,
                    "provider_authentication_required",
                )
                self.assertNotIsInstance(raised.exception, AmbiguousNebiusCreate)
                self.assertEqual(
                    raised.exception.failure_evidence.child_exit_code, child_exit_code
                )
                self.assertEqual(
                    raised.exception.journey_reason(),
                    f"provider_authentication_required;exit={child_exit_code}",
                )
                self.assertEqual(emitted[-1]["outcome"], "authentication_required")
                self.assertEqual(emitted[-1]["child_exit_code"], child_exit_code)

    def test_other_nonzero_exits_stay_the_general_contact_failure(self) -> None:
        for child_exit_code, classification in (
            (13, "not_found"),
            (8, "authorization_denied"),
            (40, "unknown"),
        ):
            with self.subTest(child_exit_code=child_exit_code):
                original = NebiusCliError(
                    "provider_command_failed",
                    failure_evidence=_evidence(child_exit_code, classification),
                )
                probe, emitted = _probe(_RecordingCli(original))

                with self.assertRaises(NebiusCliError) as raised:
                    probe()

                self.assertIs(raised.exception, original)
                self.assertNotIsInstance(raised.exception, NebiusAuthenticationRequired)
                self.assertEqual(emitted[-1]["outcome"], "contact_failed")
                self.assertEqual(emitted[-1]["child_exit_code"], child_exit_code)

    def test_a_boundary_failure_without_evidence_is_never_read_as_authentication(
        self,
    ) -> None:
        original = NebiusCliError("provider_command_unavailable")
        probe, emitted = _probe(_RecordingCli(original))

        with self.assertRaises(NebiusCliError) as raised:
            probe()

        self.assertIs(raised.exception, original)
        self.assertEqual(emitted[-1]["outcome"], "contact_failed")
        self.assertIsNone(emitted[-1]["child_exit_code"])

    def test_an_inactive_tenant_is_not_the_halt_that_asks_for_a_browser(self) -> None:
        for payload in (
            {"user_profile": {"tenants": []}},
            {
                "user_profile": {
                    "tenants": [
                        {
                            "tenant_id": "tenant-exact",
                            "tenant_user_account_state": "SUSPENDED",
                        }
                    ]
                }
            },
            {"user_profile": {"tenants": [{"tenant_id": "tenant-other"}]}},
            {},
        ):
            with self.subTest(payload=payload):
                probe, emitted = _probe(_RecordingCli(payload=payload))

                with self.assertRaises(NebiusTenantNotActive) as raised:
                    probe()

                self.assertEqual(raised.exception.code, "provider_tenant_not_active")
                self.assertNotIsInstance(raised.exception, NebiusAuthenticationRequired)
                self.assertEqual(emitted[-1]["outcome"], "tenant_not_active")

    def test_an_active_service_account_in_the_selected_project_is_authenticated(
        self,
    ) -> None:
        """Issue 35: a key-based service-account profile has no tenants list."""

        cli = _RecordingCli(payload=_service_account_whoami())
        probe, emitted = _probe(cli)

        probe()
        probe.prove(_target())

        self.assertEqual(len(cli.calls), 1)
        self.assertEqual(emitted[-1]["outcome"], "authenticated")

    def test_a_service_account_that_proves_nothing_about_the_project_is_refused(
        self,
    ) -> None:
        for payload in (
            _service_account_whoami(active=False),
            _service_account_whoami(parent_id="project-other"),
            _service_account_whoami(active="true"),
            {"service_account_profile": {}},
        ):
            with self.subTest(payload=payload):
                probe, emitted = _probe(_RecordingCli(payload=payload))

                with self.assertRaises(NebiusTenantNotActive):
                    probe()

                self.assertEqual(emitted[-1]["outcome"], "tenant_not_active")

    def test_a_refused_proof_is_never_memoized(self) -> None:
        cli = _RecordingCli(payload={"user_profile": {"tenants": []}})
        probe, _ = _probe(cli)

        for _ in range(2):
            with self.assertRaises(NebiusTenantNotActive):
                probe()

        self.assertEqual(len(cli.calls), 2)

    def test_no_probe_record_carries_stdout_stderr_or_any_payload_value(self) -> None:
        for error in (
            None,
            NebiusCliError(
                "provider_command_failed",
                failure_evidence=_evidence(7, "authentication_failed"),
            ),
        ):
            with self.subTest(failed=error is not None):
                probe, emitted = _probe(_RecordingCli(error))
                if error is None:
                    probe()
                else:
                    # Exit 7 is a documented authentication exit.
                    with self.assertRaises(NebiusCliError) as raised:
                        probe()
                    self.assertEqual(
                        raised.exception.code, "provider_authentication_required"
                    )
                self.assertTrue(emitted)

                published = set()
                for record in emitted:
                    published.update(record)
                self.assertEqual(
                    published & {"stdout", "stderr", "stdout_sha256", "user_profile"},
                    set(),
                )
                for record in emitted:
                    for value in record.values():
                        if isinstance(value, str):
                            self.assertNotIn("tenant-exact", value)
                            self.assertNotIn("user_profile", value)

    def test_an_unbound_probe_refuses_the_entry_precondition_before_contact(
        self,
    ) -> None:
        cli = _RecordingCli()
        probe = NebiusAuthenticationProbe(cli_runner=cli, emit=lambda _: None)

        with self.assertRaises(NebiusCliError) as raised:
            probe()

        self.assertEqual(
            raised.exception.code,
            "provider_authentication_target_missing",
        )
        self.assertEqual(cli.calls, [])

    def test_an_argv_without_the_stated_bound_refuses_before_any_contact(self) -> None:
        cli = _RecordingCli()
        probe, emitted = _probe(cli)

        with patch(
            "bio_neb.providers.nebius_authentication.nebius_common_arguments",
            lambda target: ("--profile", target.profile),
        ):
            with self.assertRaises(NebiusCliError) as raised:
                probe()

        self.assertEqual(raised.exception.code, "provider_auth_timeout_unstated")
        self.assertEqual(cli.calls, [])
        self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()
