from __future__ import annotations

# Split ownership keeps the established symbols explicit for review.
# ruff: noqa: F401
import copy
import hashlib
import unittest
from dataclasses import replace

from bio_neb.observation import AdapterHeartbeat
from bio_neb.parabricks.execution_observations import ProviderTerminalContact
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    ProviderStateObserved,
)
from bio_neb.providers.nebius_authentication import (
    NebiusAuthenticationProbe,
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
    _not_found,
    _request,
    _target,
    observe_contact,
)


class NebiusServerlessJobContactTests(unittest.TestCase):
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

    def test_preflight_owns_topology_without_version_subprocess(self):
        cli = _NativeCli()
        NebiusServerlessJobContact(cli_runner=cli).preflight(_target())

        self.assertFalse(any(call == ("version",) for call in cli.calls))
        self.assertTrue(
            all(
                call[-10:]
                == (
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
                )
                for call in cli.calls
            )
        )

    def test_readiness_is_one_bounded_chain_per_run_not_one_per_stage(self):
        """Ticket 144: the capsule says one readiness chain before any mutation."""

        cli = _NativeCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)

        contact.preflight(_target())
        first = list(cli.calls)
        contact.preflight(_target())

        self.assertEqual(
            [call[:2] for call in first],
            [("iam", "whoami"), ("iam", "project"), ("vpc", "subnet")],
        )
        self.assertEqual(cli.calls, first)

    def test_a_different_selection_still_gets_its_own_readiness_chain(self):
        cli = _NativeCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)

        other = _target().model_copy(update={"tenant_id": "tenant-other"})
        contact.preflight(_target())
        with self.assertRaises(NebiusJobContactError):
            contact.preflight(other)

        self.assertEqual(len(cli.calls), 4)

    def test_an_entry_proof_leaves_no_whoami_for_the_stages(self):
        cli = _NativeCli()
        probe = NebiusAuthenticationProbe(
            cli_runner=cli, target=_target(), emit=lambda _: None
        )
        probe()
        contact = NebiusServerlessJobContact(cli_runner=cli, authentication=probe)

        contact.preflight(_target())

        whoami = [call[:2] for call in cli.calls].count(("iam", "whoami"))
        self.assertEqual(whoami, 1)

    def test_preflight_requires_exact_callable_native_identity(self):
        class MissingPreflightCli(_NativeCli):
            preflight = None

        mismatched = NebiusCliInstallIdentity(
            executable="/opt/nebius/bin/nebius",
            resolved_executable="/opt/nebius/bin/nebius",
            version="0.12.263",
            byte_count=1,
            sha256="b" * 64,
        )
        cases = (
            (MissingPreflightCli(), "cli_identity_preflight_missing"),
            (
                _NativeCli(preflight_identity=mismatched),
                "cli_identity_mismatch",
            ),
            (
                _NativeCli(preflight_failure=RuntimeError("identity unavailable")),
                "cli_identity_preflight_failed",
            ),
        )
        for cli, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(NebiusJobContactError) as caught:
                    NebiusServerlessJobContact(cli_runner=cli).preflight(_target())
                self.assertEqual(caught.exception.category, "runtime_invalid")
                self.assertEqual(caught.exception.reason, reason)
                self.assertEqual(cli.calls, [])

    def test_preflight_rejects_inactive_tenant_and_parent_or_subnet_drift(self):
        def inactive(payload):
            payload["user_profile"]["tenants"][0]["tenant_user_account_state"] = (
                "INACTIVE"
            )

        def project_parent(payload):
            payload["metadata"]["parent_id"] = "other"

        def subnet_state(payload):
            payload["status"]["state"] = "CREATING"

        cases = (
            (("iam", "whoami"), inactive, "authentication"),
            (("iam", "project"), project_parent, "runtime_invalid"),
            (("vpc", "subnet"), subnet_state, "runtime_invalid"),
        )
        for selected, mutate, category in cases:
            with self.subTest(selected=selected):

                class DriftCli(_NativeCli):
                    def json(
                        self,
                        *arguments,
                        selected=selected,
                        mutate=mutate,
                    ):
                        payload = super().json(*arguments)
                        if arguments[:2] == selected:
                            mutate(payload)
                        return payload

                with self.assertRaises(NebiusJobContactError) as caught:
                    NebiusServerlessJobContact(cli_runner=DriftCli()).preflight(
                        _target()
                    )
                self.assertEqual(caught.exception.category, category)

    def test_lookup_answering_for_another_name_fails_before_callbacks(self):
        """A name lookup that answers about a different job proves nothing.

        Ticket 140 replaced an `ai job list` scan with `get-by-name`, so there is no
        longer a list to miscount -- but the lookup must still refuse an envelope whose
        metadata.name is not the exact name asked about, before any callback runs.
        """

        class OtherNameCli(_NativeCli):
            def json(self, *arguments):
                if arguments[:3] == ("ai", "job", "get-by-name"):
                    self.calls.append(arguments)
                    return {"metadata": {"id": "aijob-other", "name": "job-other"}}
                return super().json(*arguments)

        cli = OtherNameCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)
        callbacks = []
        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: callbacks.append("reattest"),
                record_create_invoked=lambda: callbacks.append("record"),
            )
        self.assertEqual(caught.exception.category, "payload_invalid")
        self.assertEqual(callbacks, [])

    def test_lookup_failure_cannot_prove_precreate_absence(self):
        """Only a NotFound answer proves absence; any other failure refuses.

        `get-by-name`'s rpc NotFound is the only absence proof the CLI offers, measured
        in Tickets 146 and 146a.  A lookup that fails for any other reason has not proved
        the name is free, so create must not run.  This case carries no failure evidence
        at all, which is the weakest possible claim and still must not pass.
        """

        cli = _NativeCli(lookup_error="provider_command_failed")
        contact = NebiusServerlessJobContact(cli_runner=cli)
        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=lambda: None,
            )
        self.assertEqual(caught.exception.category, "lookup_failed")
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_extra_call_exit_code_still_proves_absence_with_a_not_found_error(self):
        """Exit 53 is the same NotFound, reported through the CLI's extra-call rule.

        Ticket 145: docs.nebius.com/cli/exit-codes adds 40 to the target code when the
        error comes from an extra call the CLI makes internally, and Ticket 138 measured
        both halves -- 13 from `get-by-name` and 53 from `ai job get` for the identical
        condition.  Ticket 140 keyed on the bare 13, so a `get-by-name` resolving through
        a secondary lookup would have refused instead of proving the name free.  That
        direction fails closed rather than duplicating a job, which is why it is a
        correctness gap and not a safety one -- but it is still wrong.
        """

        cli = _NativeCli(absence_error=_not_found(child_exit_code=53))
        contact, created = self._create_contact(cli)

        self.assertEqual(created.create.disposition, "created")
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in cli.calls), 1
        )

    def test_not_found_exit_code_without_a_not_found_error_is_not_absence(self):
        """The number alone never proves absence; the provider error must agree.

        Ticket 145 requires both halves.  A NotFound-shaped exit code carrying some other
        provider error has not said the name is free, and the fake supplies exactly that
        near miss because no live run can be asked to produce one on demand.
        """

        cli = _NativeCli(absence_error=_not_found(stderr_classification="unavailable"))
        contact = NebiusServerlessJobContact(cli_runner=cli)
        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=lambda: None,
            )

        self.assertEqual(caught.exception.category, "lookup_failed")
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_authentication_failure_is_never_read_as_absence(self):
        """Exit 7 is any authentication error, and must never authorize a create.

        This is the one direction in Ticket 145's exit-code contract that fails *open*:
        reading an expired session as "the job does not exist" would prove absence
        falsely and create a second real H100 job under a name that already has one.
        Ticket 140 reached the right outcome by falling through to `lookup_failed`, but
        nothing pinned it, so a later widening of the accepted set had nothing to break.
        """

        cli = _NativeCli(
            absence_error=_not_found(
                child_exit_code=7, stderr_classification="authentication_failed"
            )
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

        self.assertEqual(caught.exception.category, "lookup_failed")
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_preexisting_name_stops_before_guards_callback_and_create(self):
        cli = _NativeCli(preexisting=True)
        contact = NebiusServerlessJobContact(cli_runner=cli)
        ordered = []

        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: ordered.append("reattest"),
                record_create_invoked=lambda: ordered.append("callback"),
            )

        self.assertEqual(caught.exception.category, "name_not_absent")
        self.assertEqual(caught.exception.reason, "exact_name_preexisting")
        self.assertEqual(ordered, [])
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_application_callback_exceptions_propagate_unchanged_before_create(self):
        for owner in ("reattest", "record"):
            with self.subTest(owner=owner):
                cli = _NativeCli()
                contact = NebiusServerlessJobContact(cli_runner=cli)
                failure = RuntimeError(owner)
                ordered = []

                def reattest():
                    ordered.append("reattest")  # noqa: B023 - Callback invoked within this subtest.
                    if owner == "reattest":  # noqa: B023 - Callback invoked within this subtest.
                        raise failure  # noqa: B023 - Callback invoked within this subtest.

                def record():
                    ordered.append("record")  # noqa: B023 - Callback invoked within this subtest.
                    if owner == "record":  # noqa: B023 - Callback invoked within this subtest.
                        raise failure  # noqa: B023 - Callback invoked within this subtest.

                with self.assertRaises(RuntimeError) as caught:
                    contact.create(
                        _request(),
                        environment_bindings=_ENVIRONMENT,
                        secret_environment_bindings=_SECRET_ENVIRONMENT,
                        reattest_before_create=reattest,
                        record_create_invoked=record,
                    )

                self.assertIs(caught.exception, failure)
                self.assertEqual(
                    ordered,
                    ["reattest"] if owner == "reattest" else ["reattest", "record"],
                )
                self.assertFalse(
                    any(call[:3] == ("ai", "job", "create") for call in cli.calls)
                )

    def test_create_requires_callable_application_callback(self):
        cli = _NativeCli()
        with self.assertRaises(TypeError):
            NebiusServerlessJobContact(cli_runner=cli).create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=None,  # type: ignore[arg-type]
            )
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_callback_reentrancy_cannot_create_twice(self):
        cli = _NativeCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)
        callbacks = []

        def reentrant_record():
            callbacks.append("outer")
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: callbacks.append("inner-reattest"),
                record_create_invoked=lambda: callbacks.append("inner-record"),
            )

        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: callbacks.append("reattest"),
                record_create_invoked=reentrant_record,
            )

        self.assertEqual(caught.exception.category, "contact_failed")
        self.assertEqual(caught.exception.reason, "create_operation_in_progress")
        self.assertEqual(callbacks, ["reattest", "outer"])
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_repeated_request_cannot_create_twice(self):
        cli = _NativeCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)
        callbacks = []
        contact.create(
            _request(),
            environment_bindings=_ENVIRONMENT,
            secret_environment_bindings=_SECRET_ENVIRONMENT,
            reattest_before_create=lambda: None,
            record_create_invoked=lambda: callbacks.append("record"),
        )
        with self.assertRaises(NebiusJobContactError):
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=lambda: callbacks.append("record"),
            )
        self.assertEqual(callbacks, ["record"])
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in cli.calls), 1
        )

    def test_different_name_reentry_fails_before_any_inner_contact(self):
        cli = _NativeCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)
        callbacks = []

        def reenter_different_name():
            callbacks.append("outer-record")
            contact.create(
                replace(_request(), provider_job_name="job-other"),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: callbacks.append("inner-reattest"),
                record_create_invoked=lambda: callbacks.append("inner-record"),
            )

        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: callbacks.append("outer-reattest"),
                record_create_invoked=reenter_different_name,
            )

        self.assertEqual(caught.exception.category, "contact_failed")
        self.assertEqual(caught.exception.reason, "create_operation_in_progress")
        self.assertEqual(callbacks, ["outer-reattest", "outer-record"])
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "get-by-name") for call in cli.calls), 1
        )
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_secret_environment_projection_is_derived_from_private_bindings(self):
        cli = _NativeCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)
        request = replace(_request(), version_bound_secret_environment_names=())

        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                request,
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=lambda: None,
            )

        self.assertEqual(caught.exception.category, "payload_invalid")
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))

    def test_ambiguous_create_takes_identity_by_name_without_recreating(self):
        """The name lookup is the identity path, not a recovery from a lost response.

        Ticket 140 section 4: every job id this project has ever held came through this
        lookup, and under Fork A the create response is 33 bytes of prose that is never
        parsed.  So an ambiguous create changes only the recorded disposition -- it takes
        the same identity path a clean create takes, and never creates a second job.
        """

        cli = _NativeCli(ambiguous=True, states=("STARTING", "COMPLETED"))
        contact = NebiusServerlessJobContact(cli_runner=cli)
        ordered = []
        created = contact.create(
            _request(),
            environment_bindings=_ENVIRONMENT,
            secret_environment_bindings=_SECRET_ENVIRONMENT,
            reattest_before_create=lambda: ordered.append("reattest"),
            record_create_invoked=lambda: ordered.append("callback"),
        )

        self.assertEqual(ordered, ["reattest", "callback"])
        self.assertEqual(
            created.create.disposition, "created_with_ambiguous_create_response"
        )
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in cli.calls), 1
        )

    def test_clean_create_records_its_verbatim_response_and_takes_identity_by_name(
        self,
    ):
        """Create's stdout is retained as Run proof and never read for identity.

        Under --async the response is one `Job ID:` line and `create` ignores --format
        entirely -- measured identically in Tickets 143, 146 and 146a.  Ticket 140
        section 3 keeps those bytes verbatim with their content address, while identity
        comes from the name lookup instead.
        """

        cli = _NativeCli()
        contact, created = self._create_contact(cli)

        self.assertEqual(created.create.disposition, "created")
        self.assertEqual(created.create_response, "Job ID: aijob-exact\n")
        self.assertEqual(
            created.create.response_byte_count,
            len(created.create_response.encode("utf-8")),
        )
        self.assertEqual(
            created.create.response_sha256,
            hashlib.sha256(created.create_response.encode("utf-8")).hexdigest(),
        )
        self.assertTrue(
            any(call[:3] == ("ai", "job", "get-by-name") for call in cli.calls)
        )

    def test_create_argv_requests_no_deadline_and_submits_asynchronously(self):
        """Ticket 140 decision D1 and section 5, proved at the argv boundary.

        --timeout is passed nowhere: the APP authors no deadline, and the server injects
        its own spec.timeout, which the envelope records.  --async is present because
        submission is submit-then-observe and the job does not need this process alive.
        """

        cli = _NativeCli()
        contact, created = self._create_contact(cli)
        create_call = next(
            call for call in cli.calls if call[:3] == ("ai", "job", "create")
        )

        self.assertNotIn("--timeout", create_call)
        self.assertIn("--async", create_call)
        self.assertEqual(created.envelope.timeout, "86400s")

    def test_ambiguous_create_without_a_named_job_refuses(self):
        """An ambiguous create whose name still does not exist is not a success.

        There is no match count to miscount any more: `get-by-name` answers with one
        job or with exit 13.  If create was ambiguous and the name is still absent
        afterwards, nothing was created and no identity may be manufactured.
        """

        class NeverCreatedCli(_NativeCli):
            def _create(self, arguments):
                super()._create(arguments)
                self.created = False
                raise AmbiguousNebiusCreate()

        cli = NeverCreatedCli()
        contact = NebiusServerlessJobContact(cli_runner=cli)
        with self.assertRaises(NebiusJobContactError) as caught:
            contact.create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=lambda: None,
            )

        self.assertEqual(caught.exception.category, "lookup_failed")
        self.assertEqual(caught.exception.reason, "created_name_absent_after_create")
        self.assertEqual(caught.exception.name_match_count, 0)
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in cli.calls), 1
        )

    def test_lookup_identity_must_be_a_canonical_provider_job_id(self):
        """The name lookup is the identity source, so its id is validated exactly."""

        for invalid in ("", " ", "aijob-", "aijob-EXACT"):
            with self.subTest(identity=repr(invalid)):
                cli = _NativeCli(match_id=invalid)
                with self.assertRaises(NebiusJobContactError) as caught:
                    NebiusServerlessJobContact(cli_runner=cli).create(
                        _request(),
                        environment_bindings=_ENVIRONMENT,
                        secret_environment_bindings=_SECRET_ENVIRONMENT,
                        reattest_before_create=lambda: None,
                        record_create_invoked=lambda: None,
                    )
                self.assertEqual(caught.exception.category, "envelope_drift")

    def test_unrecognised_state_is_recorded_and_never_fatal(self):
        """Ticket 137's crash site, and the generalized fix for it.

        `_decode_job` used to raise TypeError("provider state is unknown") for any state
        outside a twelve-name allowlist whose six invented names did not include
        PROVISIONING -- the state every real job starts in.  Classification is now by
        terminality alone, so an unrecognised state is carried through with the
        provider's literal string and polled again.  IMAGE_PULLING is the concrete case:
        a real documented state this project has still never observed, which must not be
        able to crash a Run the first time it appears.
        """

        for state in ("PROVISIONING", "IMAGE_PULLING", "CANCELLING", "SOMETHING_NEW"):
            with self.subTest(state=state):
                cli = _NativeCli(initial_state=state)
                _, created = self._create_contact(cli)
                self.assertEqual(created.initial_state.provider_state, state)

    def test_an_in_flight_state_may_step_back_and_the_poll_waits_through_it(self):
        """#387 celegans-a2: 28 polls PROVISIONING -> STARTING -> IMAGE_PULLING, then
        a state ranked lower failed the Run as "provider state regressed" on a job
        that went on to complete. In-flight states have no order (owner, #439;
        #443): each step back is polled through and reported as a change."""

        for initial, states in (
            ("PROVISIONING", ("STARTING", "IMAGE_PULLING", "STARTING", "RUNNING")),
            ("PROVISIONING", ("IMAGE_PULLING", "PROVISIONING", "RUNNING")),
        ):
            with self.subTest(states=states):
                cli = _NativeCli(initial_state=initial, states=(*states, "COMPLETED"))
                contact, created = self._create_contact(cli)
                polled = [
                    contact.poll_state(created.provider_job_id)
                    for _ in (*states, "COMPLETED")
                ]
                self.assertEqual(
                    [(state.provider_state, changed) for state, changed in polled],
                    [(state, True) for state in (*states, "COMPLETED")],
                )

    def test_a_state_after_a_terminal_still_fails_naming_both_states(self):
        cli = _NativeCli(initial_state="RUNNING", states=("COMPLETED", "RUNNING"))
        contact, created = self._create_contact(cli)
        contact.poll_state(created.provider_job_id)
        with self.assertRaises(NebiusJobContactError) as caught:
            contact.poll_state(created.provider_job_id)
        self.assertEqual(caught.exception.category, "payload_invalid")
        self.assertIn("COMPLETED -> RUNNING", str(caught.exception))

    def test_terminal_failure_without_timestamps_is_accepted(self):
        """Ticket 146's real terminal envelope, which the old contract rejected.

        That job died before its container started, so `ai job get` returned
        {state, state_details} with neither started_at nor finished_at.  The contract
        used to demand both on any terminal state and would have refused a truthful
        provider answer.
        """

        def strip(item):
            item["status"] = {
                "state": "FAILED",
                "state_details": {
                    "code": "TimeoutExceeded",
                    "message": "Job timed out before startup completed.",
                },
            }

        cli = _NativeCli(initial_state="FAILED", create_mutator=strip)
        _, created = self._create_contact(cli)

        self.assertEqual(created.initial_state.provider_state, "FAILED")
        self.assertIsNone(created.initial_state.started_at)
        self.assertIsNone(created.initial_state.finished_at)
        self.assertIn("TimeoutExceeded", created.initial_state.state_details[0])

    def _assert_initial_job_validation_matrix(self, validation_group):
        cases = (
            ("missing spec", lambda item: item.pop("spec"), "payload_invalid"),
            (
                "invalid job id",
                lambda item: item["metadata"].update(id="other"),
                "envelope_drift",
            ),
            *(
                (
                    f"non-canonical job id {invalid!r}",
                    lambda item, invalid=invalid: item["metadata"].update(id=invalid),
                    "envelope_drift",
                )
                for invalid in (" aijob-exact", "aijob-exact\n", "aijob-EXACT")
            ),
            (
                # Caught earlier than envelope drift now: the name lookup itself refuses
                # an envelope that answers about a different job, before decode runs.
                "job name",
                lambda item: item["metadata"].update(name="other"),
                "payload_invalid",
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
                # The APP authors no deadline, so a differing
                # spec.timeout is not drift -- it is the server's answer, recorded as
                # observed.  What is still refused is a provider that states none at all.
                "timeout",
                lambda item: item["spec"].pop("timeout"),
                "payload_invalid",
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
                "timezone-naive creation",
                lambda item: item["metadata"].update(created_at="2026-08-14T10:00:00"),
                "payload_invalid",
            ),
            (
                "terminal timestamp order",
                lambda item: item["status"].update(
                    state="COMPLETED",
                    finished_at="2026-08-14T09:59:59+00:00",
                ),
                "payload_invalid",
            ),
        )
        group_fields = {
            "identity": {
                "invalid job id",
                "non-canonical job id ' aijob-exact'",
                "non-canonical job id 'aijob-exact\\n'",
                "non-canonical job id 'aijob-EXACT'",
                "job name",
                "project parent",
            },
            "resource_envelope": {
                "missing spec",
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
            "timestamps": {
                "timezone-naive creation",
                "terminal timestamp order",
            },
        }
        selected_fields = group_fields[validation_group]
        for field, mutate, category in cases:
            if field not in selected_fields:
                continue
            with self.subTest(field=field, category=category):
                cli = _NativeCli(create_mutator=mutate)
                contact = NebiusServerlessJobContact(cli_runner=cli)

                with self.assertRaises(NebiusJobContactError) as caught:
                    contact.create(
                        _request(),
                        environment_bindings=_ENVIRONMENT,
                        secret_environment_bindings=_SECRET_ENVIRONMENT,
                        reattest_before_create=lambda: None,
                        record_create_invoked=lambda: None,
                    )

                self.assertEqual(caught.exception.category, category)

    def test_initial_job_identity_is_validated(self):
        self._assert_initial_job_validation_matrix("identity")

    def test_initial_job_resource_envelope_is_validated(self):
        self._assert_initial_job_validation_matrix("resource_envelope")

    def test_initial_provider_timestamps_are_validated(self):
        self._assert_initial_job_validation_matrix("timestamps")

    def test_provider_may_omit_only_an_empty_environment_projection(self):
        request = replace(
            _request(),
            environment_names=(),
            version_bound_secret_environment_names=(),
        )
        cli = _NativeCli(
            create_mutator=lambda item: item["spec"].pop("environment_variables"),
            get_mutators=(lambda item: item["spec"].pop("environment_variables"),),
        )
        contact = NebiusServerlessJobContact(cli_runner=cli)

        created = contact.create(
            request,
            environment_bindings=(),
            secret_environment_bindings=(),
            reattest_before_create=lambda: None,
            record_create_invoked=lambda: None,
        )
        observations = tuple(
            observe_contact(contact, created, heartbeat_interval_seconds=1.0)
        )

        self.assertIsInstance(observations[-1], ProviderTerminalContact)
        self.assertEqual(created.envelope.environment_names, ())

        nonempty_cli = _NativeCli(
            create_mutator=lambda item: item["spec"].pop("environment_variables")
        )
        with self.assertRaises(NebiusJobContactError) as caught:
            NebiusServerlessJobContact(cli_runner=nonempty_cli).create(
                _request(),
                environment_bindings=_ENVIRONMENT,
                secret_environment_bindings=_SECRET_ENVIRONMENT,
                reattest_before_create=lambda: None,
                record_create_invoked=lambda: None,
            )
        self.assertEqual(caught.exception.category, "envelope_drift")


if __name__ == "__main__":
    unittest.main()
