from __future__ import annotations

# Split ownership keeps the established symbols explicit for review.
# ruff: noqa: F401
import copy
import hashlib
import unittest
from dataclasses import replace

from bio_neb.journey import canonical_json
from bio_neb.observation import AdapterHeartbeat
from bio_neb.parabricks.execution_observations import ProviderTerminalContact
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    ProviderStateObserved,
)
from bio_neb.providers.nebius_common import (
    AmbiguousNebiusCreate,
    NebiusCliError,
    NebiusCliInstallIdentity,
    NebiusCliResponse,
)
from bio_neb.providers.nebius_serverless_job_contact import (
    NebiusJobContactError,
    NebiusServerlessJobContact,
    ProviderReadyJobRequest,
)
from bio_neb.providers.runtime import NebiusCliTarget
from tests.nebius_serverless_job_contact_fakes import (
    _ENVIRONMENT,
    _SECRET_ENVIRONMENT,
    _h100_request,
    _NativeCli,
    _request,
    _target,
    observe_contact,
)


def _full_window(*markers: str) -> str:
    """A window at the ``--tail 1000`` bound: the markers, then plain tool output."""

    lines = [*markers]
    lines.extend(
        f"[t0] plain tool output line {index}" for index in range(len(lines), 1000)
    )
    return "\n".join(lines)


class NebiusServerlessJobObservationTests(unittest.TestCase):
    def test_poll_keeps_terminal_state_when_optional_log_is_unusable(self):
        for output in ("\n".join(["line"] * 1001), "contact-access", "BIO_NEB_BROKEN"):
            with self.subTest(output=output[:30]):
                cli = _NativeCli(
                    initial_state="COMPLETED",
                    states=("COMPLETED",),
                    log_outputs=(output,),
                )
                contact, created = self._create_contact(cli)
                state, _, log = contact.poll(created.provider_job_id, window=1000)
                self.assertEqual(state.provider_state, "COMPLETED")
                if output != "BIO_NEB_BROKEN":
                    self.assertIsNone(log)
                self.assertEqual(
                    sum(call[:3] == ("ai", "job", "create") for call in cli.calls), 1
                )

    def _create_contact(self, cli, request=None):
        contact = NebiusServerlessJobContact(cli_runner=cli)
        created = contact.create(
            request or _request(),
            environment_bindings=_ENVIRONMENT,
            secret_environment_bindings=_SECRET_ENVIRONMENT,
            reattest_before_create=lambda: None,
            record_create_invoked=lambda: None,
        )
        return contact, created

    def _assert_later_get_validation_matrix(self, validation_group):
        cases = (
            ("missing status", lambda item: item.pop("status"), "payload_invalid"),
            (
                "job id",
                lambda item: item["metadata"].update(id="other"),
                "envelope_drift",
            ),
            (
                "job name",
                lambda item: item["metadata"].update(name="other"),
                "envelope_drift",
            ),
            (
                "project parent",
                lambda item: item["metadata"].update(parent_id="other"),
                "envelope_drift",
            ),
            (
                "image",
                lambda item: item["spec"].update(image="other"),
                "envelope_drift",
            ),
            (
                "arguments",
                lambda item: item["spec"].update(args="other"),
                "envelope_drift",
            ),
            (
                "platform",
                lambda item: item["spec"].update(platform="other"),
                "envelope_drift",
            ),
            (
                "preset",
                lambda item: item["spec"].update(preset="other"),
                "envelope_drift",
            ),
            (
                "disk",
                lambda item: item["spec"]["disk"].update(size_bytes="1"),
                "envelope_drift",
            ),
            (
                "shared memory",
                lambda item: item["spec"].update(shm_size_bytes="1"),
                "envelope_drift",
            ),
            (
                "subnet",
                lambda item: item["spec"].update(subnet_id="other"),
                "envelope_drift",
            ),
            (
                "timeout",
                lambda item: item["spec"].update(timeout="1s"),
                "envelope_drift",
            ),
            (
                "restart",
                lambda item: item["spec"].update(restart_policy="always"),
                "envelope_drift",
            ),
            (
                "explicit container command",
                lambda item: item["spec"].update(container_command=""),
                "envelope_drift",
            ),
            (
                "injected files",
                lambda item: item["spec"].update(injected_files=[]),
                "envelope_drift",
            ),
            (
                "environment",
                lambda item: item["spec"].update(environment_variables=[]),
                "envelope_drift",
            ),
            (
                "unknown state",
                lambda item: item["status"].update(state="UNKNOWN"),
                "payload_invalid",
            ),
            (
                "creation timestamp drift",
                lambda item: item["metadata"].update(
                    created_at="2026-08-14T10:00:03+00:00"
                ),
                "payload_invalid",
            ),
            (
                "missing started timestamp",
                lambda item: item["status"].pop("started_at"),
                "payload_invalid",
            ),
        )
        # An in-flight step back (STARTING -> PROVISIONING) is no longer refused:
        # the provider orders no in-flight states. A change after a terminal
        # still is, tested in test_nebius_serverless_job_create.
        group_fields = {
            "identity": {"job id", "job name", "project parent"},
            "resource_envelope": {
                "image",
                "arguments",
                "platform",
                "preset",
                "disk",
                "shared memory",
                "subnet",
                "timeout",
                "restart",
                "explicit container command",
                "injected files",
                "environment",
            },
            "state_timestamps": {
                "missing status",
                "creation timestamp drift",
                "missing started timestamp",
            },
        }
        selected_fields = group_fields.get(validation_group, set())
        for field, mutate, category in cases:
            if field not in selected_fields:
                continue
            with self.subTest(field=field, category=category):
                cli = _NativeCli(states=("RUNNING",), get_mutators=(mutate,))
                contact, created = self._create_contact(cli)

                with self.assertRaises(NebiusJobContactError) as caught:
                    list(
                        observe_contact(
                            contact, created, heartbeat_interval_seconds=1.0
                        )
                    )

                self.assertEqual(caught.exception.category, category)

        if validation_group != "details":
            return
        for details in ("contact-access", "line one\nline two"):
            with self.subTest(later_details=details):

                def mutate(item, details=details):
                    item["status"].update(state_details=details)

                # state_details is a terminal-failure explanation, so the safety scan
                # is exercised exactly where the provider actually populates it.
                cli = _NativeCli(states=("FAILED",), get_mutators=(mutate,))
                contact, created = self._create_contact(cli)
                with self.assertRaises(NebiusJobContactError) as caught:
                    list(
                        observe_contact(
                            contact, created, heartbeat_interval_seconds=1.0
                        )
                    )
                self.assertEqual(caught.exception.category, "payload_invalid")

    def test_later_get_revalidates_job_identity(self):
        self._assert_later_get_validation_matrix("identity")

    def test_later_get_revalidates_resource_envelope(self):
        self._assert_later_get_validation_matrix("resource_envelope")

    def test_later_get_revalidates_state_and_timestamps(self):
        self._assert_later_get_validation_matrix("state_timestamps")

    def test_later_get_rejects_unsafe_details(self):
        self._assert_later_get_validation_matrix("details")

    def test_invalid_json_during_the_name_lookup_is_bounded(self):
        lookup_cli = _NativeCli(lookup_error="provider_json_invalid")
        with self.assertRaises(NebiusJobContactError) as lookup_caught:
            NebiusServerlessJobContact(cli_runner=lookup_cli).create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=lambda: None,
            )
        self.assertEqual(lookup_caught.exception.category, "lookup_failed")
        self.assertEqual(lookup_caught.exception.reason, "provider_json_invalid")
        self.assertFalse(lookup_cli.created)

    def test_invalid_json_during_later_get_is_bounded(self):
        get_cli = _NativeCli(get_error="provider_json_invalid")
        contact, created = self._create_contact(get_cli)
        with self.assertRaises(NebiusJobContactError) as get_caught:
            list(observe_contact(contact, created, heartbeat_interval_seconds=1.0))
        self.assertEqual(get_caught.exception.category, "contact_failed")
        self.assertEqual(get_caught.exception.reason, "provider_json_invalid")

    def test_state_details_are_read_only_as_a_terminal_failure_explanation(self):
        """Ticket 140 section 2, from two live runs that agree.

        Ticket 146 saw state_details populated only in the terminal FAILED envelope and
        absent in all 41 PROVISIONING samples.  Ticket 146a saw it absent through an
        entire successful run, including the terminal COMPLETED.  It is a failure
        explanation, not a progress channel, so it is not read anywhere else.
        """

        for state in ("STARTING", "RUNNING", "COMPLETED"):
            with self.subTest(nonfailure_state=state):
                cli = _NativeCli(
                    initial_state=state,
                    create_mutator=lambda item: item["status"].update(
                        state_details={"code": "OK", "message": "progress"}
                    ),
                )
                _, created = self._create_contact(cli)
                self.assertEqual(created.initial_state.state_details, ())

    def test_instances_are_recorded_as_diagnostic_context_and_never_classified(self):
        """Ticket 140 section 2: instances[] is context, never an outcome.

        It has disagreed with the job in both directions -- Ticket 146 saw an instance
        FAILED under a job-level PROVISIONING, Ticket 146a saw an instance RUNNING under
        a terminal COMPLETED -- and its `state` key is optional in both runs.  So it is
        retained verbatim and never read with the job-level vocabulary.
        """

        cases = (
            [{"state": "FAILED", "compute_instance_state": "STOPPED"}],
            [{"compute_instance_id": "computeinstance-x"}],
        )
        for instances in cases:
            with self.subTest(instances=instances):
                cli = _NativeCli(
                    initial_state="PROVISIONING",
                    create_mutator=lambda item, instances=instances: item[
                        "status"
                    ].update(instances=instances),
                )
                _, created = self._create_contact(cli)

                self.assertEqual(created.initial_state.provider_state, "PROVISIONING")
                self.assertEqual(
                    created.initial_state.instances,
                    tuple(canonical_json(entry) for entry in instances),
                )

    def test_a_settled_create_operation_never_makes_a_running_job_succeed(self):
        """Ticket 140's most consequential rule, proved at the contact boundary.

        Ticket 146 section 9.4 told this ticket to poll the `Create job` operation record
        as the outcome channel.  Ticket 146a proved that wrong: the record settles about
        16 s after the container starts and is byte-identical for the rest of the run, so
        `status: {}` there means *create succeeded and the container started*, never
        *the job finished*.  This contact therefore never asks for it -- the outcome is
        status.state plus state_details and nothing else -- and a job still RUNNING must
        not terminate observation however settled that operation looks.
        """

        class SettledOperationCli(_NativeCli):
            def json(self, *arguments):
                if arguments[:3] == ("ai", "job", "operation"):
                    raise AssertionError(
                        "the Create job operation is create diagnosis, not the outcome"
                    )
                return super().json(*arguments)

        cli = SettledOperationCli(
            states=("RUNNING", "RUNNING", "COMPLETED"), log_outputs=("", "", "")
        )
        contact, created = self._create_contact(cli)
        observations = list(
            observe_contact(contact, created, heartbeat_interval_seconds=1.0)
        )

        terminal = [
            item for item in observations if isinstance(item, ProviderTerminalContact)
        ]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].provider_state, "COMPLETED")
        self.assertFalse(
            any(call[:3] == ("ai", "job", "operation") for call in cli.calls)
        )

    def test_safe_details_accept_supported_shapes(self):
        safe = (
            "capacity allocated",
            {"code": "OK", "message": "capacity allocated"},
            ["capacity allocated", "worker starting"],
        )
        for details in safe:
            with self.subTest(details=details):
                cli = _NativeCli(
                    initial_state="FAILED",
                    create_mutator=lambda item, details=details: item["status"].update(
                        state_details=details
                    ),
                )
                _, created = self._create_contact(cli)
                self.assertTrue(created.initial_state.state_details)
                if isinstance(details, dict):
                    self.assertEqual(
                        created.initial_state.state_details,
                        (canonical_json(details),),
                    )
                    self.assertIn(
                        '"message":"capacity allocated"',
                        created.initial_state.state_details[0],
                    )

    def test_safe_details_reject_unsafe_strings_and_shapes(self):
        unsafe = (
            " contact-access",
            "line one\nline two",
            "secret-exact",
            "version-exact",
            "secret-exact@version-exact",
            {"message": "line one\nline two"},
            {"secret_id": "secret-exact"},
            ["safe", "bad\0detail"],
            [{"not": "a string sequence"}],
            [{"code": "not-a-string-sequence"}],
        )
        for details in unsafe:
            with self.subTest(details=details):
                cli = _NativeCli(
                    initial_state="FAILED",
                    create_mutator=lambda item, details=details: item["status"].update(
                        state_details=details
                    ),
                )
                contact = NebiusServerlessJobContact(cli_runner=cli)
                with self.assertRaises(NebiusJobContactError) as caught:
                    contact.create(
                        _request(),
                        environment_bindings=_ENVIRONMENT,
                        secret_environment_bindings=_SECRET_ENVIRONMENT,
                        reattest_before_create=lambda: None,
                        record_create_invoked=lambda: None,
                    )
                self.assertEqual(caught.exception.category, "payload_invalid")

    def test_state_changes_heartbeats_and_provider_terminal_are_ordered(self):
        cli = _NativeCli(
            states=("STARTING", "RUNNING", "RUNNING", "COMPLETED"),
            log_outputs=("", "", ""),
        )
        contact, created = self._create_contact(cli)

        observations = list(
            observe_contact(contact, created, heartbeat_interval_seconds=1.0)
        )

        self.assertEqual(
            [
                item.provider_state
                for item in observations
                if isinstance(item, ProviderStateObserved)
            ],
            ["STARTING", "RUNNING", "COMPLETED"],
        )
        self.assertEqual(
            sum(isinstance(item, AdapterHeartbeat) for item in observations), 2
        )
        self.assertIsInstance(observations[-1], ProviderTerminalContact)

    def test_log_unavailable_repeat_and_overlap_are_retained(self):
        first = '[t1] BIO_NEB_TEST {"n":1}'
        second = '[t2] BIO_NEB_TEST {"n":2}'
        third = '[t3] BIO_NEB_TEST {"n":3}'
        accepted = (
            (None, ""),
            (first, first),
            ("\n".join((first, second)), "\n".join((second, third))),
        )
        for windows in accepted:
            with self.subTest(windows=windows):
                cli = _NativeCli(
                    initial_state="RUNNING",
                    states=("COMPLETED",),
                    log_outputs=windows,
                )
                contact, created = self._create_contact(cli)
                observations = list(
                    observe_contact(contact, created, heartbeat_interval_seconds=1.0)
                )
                logs = [
                    item
                    for item in observations
                    if isinstance(item, BoundedProviderLogObserved)
                ]
                self.assertEqual(len(logs), 2)

    def test_short_marker_less_window_is_a_partial_read_the_next_window_continues(
        self,
    ):
        # The live a6 H100 read at 07:38:56 returned one line of a ~102-line log
        # while the job was RUNNING, and the job later completed. A marker-less
        # window shorter than the tail bound is a transient partial read:
        # retained as its own observation, never judged as aged out.
        first = '[t1] BIO_NEB_TEST {"n":1}'
        second = '[t2] BIO_NEB_TEST {"n":2}'
        for partial in ("", "[t9] one plain line of tool output"):
            with self.subTest(partial=partial):
                cli = _NativeCli(
                    initial_state="RUNNING",
                    states=("RUNNING", "COMPLETED"),
                    log_outputs=(first, partial, "\n".join((first, second))),
                )
                contact, created = self._create_contact(cli)

                observations = list(
                    observe_contact(contact, created, heartbeat_interval_seconds=1.0)
                )

                logs = [
                    item
                    for item in observations
                    if isinstance(item, BoundedProviderLogObserved)
                ]
                self.assertEqual(
                    [item.line_count for item in logs],
                    [1, len(partial.splitlines()), 2],
                )
                self.assertEqual(observations[-1].marker_lines, (first, second))

    def test_consecutive_partial_windows_recover_at_the_ceiling_and_fail_past_it(
        self,
    ):
        first = '[t1] BIO_NEB_TEST {"n":1}'
        second = '[t2] BIO_NEB_TEST {"n":2}'
        ceiling = 3
        recovered = _NativeCli(
            initial_state="RUNNING",
            states=("RUNNING",) * ceiling + ("COMPLETED",),
            log_outputs=(first, *[""] * ceiling, "\n".join((first, second))),
        )
        contact, created = self._create_contact(recovered)
        observations = list(
            observe_contact(contact, created, heartbeat_interval_seconds=1.0)
        )
        self.assertEqual(
            sum(isinstance(item, BoundedProviderLogObserved) for item in observations),
            ceiling + 2,
        )
        self.assertEqual(observations[-1].marker_lines, (first, second))

        exhausted = _NativeCli(
            initial_state="RUNNING",
            states=("RUNNING",) * (ceiling + 1) + ("COMPLETED",),
            log_outputs=(first, *[""] * (ceiling + 1), "\n".join((first, second))),
        )
        contact, created = self._create_contact(exhausted)
        with self.assertRaises(NebiusJobContactError) as caught:
            list(observe_contact(contact, created, heartbeat_interval_seconds=1.0))
        self.assertEqual(caught.exception.category, "log_invalid")
        self.assertEqual(
            caught.exception.reason, "marker_window_partial_ceiling_exceeded"
        )
        window = caught.exception.log_window
        self.assertEqual(
            (
                window.line_count,
                window.byte_count,
                window.retained_marker_count,
                window.consecutive_partial_windows,
                window.partial_window_ceiling,
                window.window_text,
            ),
            (0, 0, 1, ceiling + 1, ceiling, ""),
        )

    def test_short_non_overlapping_marker_window_is_a_partial_read(self):
        first = '[t1] BIO_NEB_TEST {"n":1}'
        second = '[t2] BIO_NEB_TEST {"n":2}'
        third = '[t3] BIO_NEB_TEST {"n":3}'
        cli = _NativeCli(
            initial_state="RUNNING",
            states=("RUNNING", "COMPLETED"),
            log_outputs=(first, third, "\n".join((first, second, third))),
        )
        contact, created = self._create_contact(cli)

        observations = list(
            observe_contact(contact, created, heartbeat_interval_seconds=1.0)
        )

        self.assertEqual(observations[-1].marker_lines, (first, second, third))

    def test_log_aged_out_disappeared_or_conflicting_windows_fail_closed(self):
        first = '[t1] BIO_NEB_TEST {"n":1}'
        second = '[t2] BIO_NEB_TEST {"n":2}'
        rejected = (
            (first, _full_window(second), "marker_window_aged_out"),
            (first, _full_window(), "marker_window_aged_out"),
            (
                first,
                '[t1] BIO_NEB_TEST {"n":99}',
                "marker_window_conflict",
            ),
        )
        for prior, later, reason in rejected:
            with self.subTest(reason=reason, later=later[:32]):
                cli = _NativeCli(
                    initial_state="RUNNING",
                    states=("COMPLETED",),
                    log_outputs=(prior, later),
                )
                contact, created = self._create_contact(cli)
                with self.assertRaises(NebiusJobContactError) as caught:
                    list(
                        observe_contact(
                            contact, created, heartbeat_interval_seconds=1.0
                        )
                    )
                self.assertEqual(caught.exception.category, "log_invalid")
                self.assertEqual(caught.exception.reason, reason)
                window = caught.exception.log_window
                self.assertEqual(
                    (
                        window.line_count,
                        window.byte_count,
                        window.sha256,
                        window.retained_marker_count,
                        window.consecutive_partial_windows,
                        window.partial_window_ceiling,
                    ),
                    (
                        len(later.splitlines()),
                        len(later.encode()),
                        hashlib.sha256(later.encode()).hexdigest(),
                        1,
                        0,
                        3,
                    ),
                )
                self.assertEqual(window.window_text, later[:65536])
                self.assertEqual(window.window_text_truncated, len(later) > 65536)

    def test_marker_grammar_failure_retains_the_offending_window(self):
        first = '[t1] BIO_NEB_TEST {"n":1}'
        later = "\n".join((first, "BIO_NEB_TEST without a provider timestamp"))
        cli = _NativeCli(
            initial_state="RUNNING", states=("COMPLETED",), log_outputs=(first, later)
        )
        contact, created = self._create_contact(cli)
        with self.assertRaises(NebiusJobContactError) as caught:
            list(observe_contact(contact, created, heartbeat_interval_seconds=1.0))
        self.assertEqual(caught.exception.category, "log_invalid")
        self.assertTrue(
            caught.exception.reason.startswith("ValueError@execution_observations.py:")
        )
        window = caught.exception.log_window
        self.assertEqual(
            (window.line_count, window.retained_marker_count, window.window_text),
            (2, 1, later),
        )

    def test_cpu_and_h100_requests_preserve_exact_resource_differences(self):
        calls = []
        for request in (_request(), _h100_request()):
            cli = _NativeCli()
            self._create_contact(cli, request=request)
            calls.append(
                next(call for call in cli.calls if call[:3] == ("ai", "job", "create"))
            )

        cpu, h100 = calls
        self.assertEqual(cpu[cpu.index("--platform") + 1], "cpu-d3")
        self.assertEqual(cpu[cpu.index("--preset") + 1], "4vcpu-16gb")
        self.assertEqual(cpu[cpu.index("--disk-size") + 1], "100Gi")
        self.assertNotIn("--shm-size", cpu)
        self.assertEqual(h100[h100.index("--platform") + 1], "gpu-h100-sxm")
        self.assertEqual(h100[h100.index("--preset") + 1], "1gpu-16vcpu-200gb")
        self.assertEqual(h100[h100.index("--disk-size") + 1], "250Gi")
        self.assertEqual(h100[h100.index("--shm-size") + 1], "16Gi")

    def test_terminal_provider_failure_has_no_later_log_contact(self):
        cli = _NativeCli(states=("FAILED",))
        contact = NebiusServerlessJobContact(cli_runner=cli)
        created = contact.create(
            _request(),
            environment_bindings=_ENVIRONMENT,
            secret_environment_bindings=_SECRET_ENVIRONMENT,
            reattest_before_create=lambda: None,
            record_create_invoked=lambda: None,
        )

        with self.assertRaises(NebiusJobContactError) as caught:
            list(observe_contact(contact, created, heartbeat_interval_seconds=1.0))

        self.assertEqual(caught.exception.category, "provider_failed")
        self.assertFalse(any(call[:3] == ("ai", "job", "logs") for call in cli.calls))


if __name__ == "__main__":
    unittest.main()
