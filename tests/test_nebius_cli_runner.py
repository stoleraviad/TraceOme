from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

from bio_neb.agentic_journey.failure_cause import cause_identity
from bio_neb.providers.nebius_common import (
    PROVIDER_STDERR_CLOSE,
    PROVIDER_STDERR_OPEN,
    AmbiguousNebiusCreate,
    NebiusCliError,
    NebiusCliFailureEvidence,
    NebiusCliInstallIdentity,
    NebiusCliRunner,
)

_SECRET_KEY = "secret-key-sentinel"
# A live NotFound answer had exactly this stderr shape, and the runner lifts
# exactly two identifiers out of it.  `desc =` and the request id stay with the
# free text, which stays on the operator record.
_NOT_FOUND_STDERR = (
    "Error: rpc error: code = NotFound desc = not found "
    "request = bc33135d-cd17-4ed8-b47c-ea91eee160e4\n"
    "\n"
    "This issue may be linked to API request calls. The following information may be "
    "helpful while debugging the issue:\n"
    "  Trace ID: ba6fd61a3f05c84c639836764059c44e\n"
)
# The reason the split exists at all: the CLI's authentication prompt arrives on the
# same stream carrying a live OAuth URL.
_AUTH_PROMPT_STDERR = (
    "Switch to your browser to complete the authentication process.\n"
    "open browser: https://auth.example.invalid/oauth2/authorize"
    "?code_challenge=Zm9vYmFyc2VudGluZWxjaGFsbGVuZ2U"
    f"&state=state-sentinel-value&token={_SECRET_KEY}\n"
)


class NebiusCliRunnerTests(unittest.TestCase):
    _EXECUTABLE = "/opt/nebius/bin/nebius"
    _SHA256 = "a" * 64

    def _identity(self, **changes) -> NebiusCliInstallIdentity:
        values = {
            "executable": self._EXECUTABLE,
            "resolved_executable": self._EXECUTABLE,
            "version": "0.12.263",
            "byte_count": 83_259_554,
            "sha256": self._SHA256,
        }
        values.update(changes)
        return NebiusCliInstallIdentity(**values)

    def _inspector(self, executable):
        self.assertEqual(executable, self._EXECUTABLE)
        return SimpleNamespace(
            resolved_executable=self._EXECUTABLE,
            byte_count=83_259_554,
            sha256=self._SHA256,
        )

    def test_pinned_runner_emits_exact_shell_free_argv_after_one_preflight(self):
        calls = []
        inspections = []

        def inspector(executable):
            inspections.append(executable)
            return self._inspector(executable)

        def process_runner(argv, **kwargs):
            calls.append((argv, kwargs))
            stdout = "0.12.263\n" if argv[-1] == "version" else '{"items": []}'
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        runner = NebiusCliRunner(
            expected_identity=self._identity(),
            process_runner=process_runner,
            executable_inspector=inspector,
        )

        self.assertEqual(runner.json("ai", "job", "list"), {"items": []})
        self.assertEqual(runner.json("profile", "list"), {"items": []})
        self.assertEqual(
            [call[0] for call in calls],
            [
                (self._EXECUTABLE, "version"),
                (self._EXECUTABLE, "ai", "job", "list"),
                (self._EXECUTABLE, "profile", "list"),
            ],
        )
        for argv, kwargs in calls:
            self.assertEqual(argv[0], self._EXECUTABLE)
            self.assertNotIn("-d", argv)
            self.assertNotIn("Ubuntu-24.04", argv)
            self.assertIs(kwargs["shell"], False)
            self.assertIs(kwargs["check"], False)
            self.assertIs(kwargs["capture_output"], True)
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "strict")
        self.assertEqual(
            runner.preflight().to_receipt(),
            {
                "schema_version": "bio-neb.nebius-cli-install-identity/1",
                "executable": self._EXECUTABLE,
                "resolved_executable": self._EXECUTABLE,
                "version": "0.12.263",
                "byte_count": 83_259_554,
                "sha256": self._SHA256,
            },
        )
        self.assertEqual(inspections, [self._EXECUTABLE])

    def test_executable_identity_drift_stops_before_provider_command(self):
        for drift in ("resolved_executable", "byte_count", "sha256"):
            calls = []
            observed = {
                "resolved_executable": self._EXECUTABLE,
                "byte_count": 83_259_554,
                "sha256": self._SHA256,
            }
            observed[drift] = {
                "resolved_executable": "/opt/nebius/bin/drifted",
                "byte_count": 1,
                "sha256": "b" * 64,
            }[drift]
            runner = NebiusCliRunner(
                expected_identity=self._identity(),
                process_runner=lambda argv, calls=calls, **kwargs: calls.append(argv),
                executable_inspector=lambda executable, observed=observed: (
                    SimpleNamespace(**observed)
                ),
            )
            with (
                self.subTest(drift=drift),
                self.assertRaises(NebiusCliError) as caught,
            ):
                runner("iam", "whoami")
            self.assertEqual(caught.exception.code, "provider_cli_identity_mismatch")
            self.assertEqual(calls, [])

    def test_cli_version_drift_stops_before_provider_command(self):
        calls = []

        def wrong_version(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "0.12.236\n", "")

        runner = NebiusCliRunner(
            expected_identity=self._identity(),
            process_runner=wrong_version,
            executable_inspector=self._inspector,
        )
        with self.assertRaises(NebiusCliError) as caught:
            runner("iam", "whoami")
        self.assertEqual(caught.exception.code, "provider_cli_version_mismatch")
        self.assertEqual(calls, [(self._EXECUTABLE, "version")])

    def test_bucket_create_authentication_and_ordinary_rejection_are_distinct(self):
        for stderr, expected in (
            (f"unauthenticated secret={_SECRET_KEY}", "authentication"),
            (f"ordinary rejection secret={_SECRET_KEY}", "unknown"),
        ):

            def process_runner(argv, stderr=stderr, **kwargs):
                if argv[-1] == "version":
                    return subprocess.CompletedProcess(argv, 0, "0.12.263\n", "")
                return subprocess.CompletedProcess(argv, 1, "", stderr)

            runner = NebiusCliRunner(
                expected_identity=self._identity(),
                process_runner=process_runner,
                executable_inspector=self._inspector,
            )
            with (
                self.subTest(expected=expected),
                self.assertRaises(NebiusCliError) as caught,
            ):
                runner("storage", "bucket", "create", "--name", "exact")
            self.assertEqual(caught.exception.provider_reason, expected)
            self.assertNotIn(_SECRET_KEY, repr(caught.exception))

    def test_read_only_failures_retain_bounded_operation_evidence(self):
        cases = (
            (("profile", "list"), "profile.list"),
            (
                ("iam", "tenant", "list", "--profile", "research"),
                "iam.tenant.list",
            ),
            (
                (
                    "iam",
                    "project",
                    "list",
                    "--profile",
                    "research",
                    "--parent-id",
                    "tenant-exact",
                ),
                "iam.project.list",
            ),
            (
                (
                    "compute",
                    "gpu-cluster",
                    "list",
                    "--profile",
                    "research",
                ),
                "compute.gpu-cluster.list",
            ),
        )
        for arguments, operation in cases:
            stderr = (
                "Switch to your browser to complete the authentication process. "
                f'open browser: exec: "cmd.exe": executable file not found; {_SECRET_KEY}'
            )

            def process_runner(argv, **kwargs):
                if argv[-1] == "version":
                    return subprocess.CompletedProcess(argv, 0, "0.12.263\n", "")
                return subprocess.CompletedProcess(argv, 7, "", stderr)  # noqa: B023 - Invoked within this subtest before the next iteration.

            runner = NebiusCliRunner(
                expected_identity=self._identity(),
                process_runner=process_runner,
                executable_inspector=self._inspector,
            )
            with (
                self.subTest(operation=operation),
                self.assertRaises(NebiusCliError) as caught,
            ):
                runner(*arguments)
            evidence = caught.exception.failure_evidence
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence.operation, operation)
            self.assertEqual(evidence.child_exit_code, 7)
            self.assertEqual(evidence.stdout_byte_count, 0)
            self.assertEqual(evidence.stdout_sha256, hashlib.sha256(b"").hexdigest())
            self.assertEqual(evidence.stderr_byte_count, len(stderr.encode("utf-8")))
            self.assertEqual(
                evidence.stderr_sha256,
                hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                evidence.stderr_classification,
                "interactive_authentication_required",
            )
            self.assertNotIn(_SECRET_KEY, repr(caught.exception))
            self.assertNotIn(_SECRET_KEY, repr(evidence))

    def test_json_from_stdin_preserves_exact_native_argv_and_in_memory_input(self):
        calls = []

        def process_runner(argv, **kwargs):
            calls.append((argv, kwargs))
            stdout = (
                "0.12.263\n"
                if argv[-1] == "version"
                else '{"ServerURL":"registry.example"}'
            )
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        runner = NebiusCliRunner(
            expected_identity=self._identity(),
            process_runner=process_runner,
            executable_inspector=self._inspector,
        )

        self.assertEqual(
            runner.json_from_stdin(
                "registry.example\n", "registry", "docker-credential", "get"
            ),
            {"ServerURL": "registry.example"},
        )
        argv, kwargs = calls[-1]
        self.assertEqual(
            argv,
            (
                self._EXECUTABLE,
                "registry",
                "docker-credential",
                "get",
            ),
        )
        self.assertEqual(kwargs["input"], "registry.example\n")
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(argv[0], self._EXECUTABLE)

    def test_create_timeout_is_ambiguous(self):
        invocation = 0

        def timeout_after_version(argv, **kwargs):
            nonlocal invocation
            invocation += 1
            if invocation == 1:
                return subprocess.CompletedProcess(argv, 0, "0.12.263\n", "")
            raise subprocess.TimeoutExpired(argv, 1)

        runner = NebiusCliRunner(
            expected_identity=self._identity(),
            process_runner=timeout_after_version,
            executable_inspector=self._inspector,
        )
        with self.assertRaises(AmbiguousNebiusCreate):
            runner("storage", "bucket", "create", "--name", "exact")

    def test_invalid_utf8_is_bounded_provider_output_failure(self):
        def invalid_utf8(argv, **kwargs):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")

        runner = NebiusCliRunner(
            expected_identity=self._identity(),
            process_runner=invalid_utf8,
            executable_inspector=self._inspector,
        )
        with self.assertRaises(NebiusCliError) as caught:
            runner("iam", "whoami")
        self.assertEqual(caught.exception.code, "provider_output_invalid_utf8")

    def test_real_fake_executable_proves_direct_process_and_identity_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory, "nebius")
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "print('0.12.263' if sys.argv[1:] == ['version'] "
                "else '{\"items\": []}')\n",
                encoding="utf-8",
            )
            executable.chmod(0o700)
            content = executable.read_bytes()
            identity = NebiusCliInstallIdentity(
                executable=executable.as_posix(),
                resolved_executable=executable.resolve().as_posix(),
                version="0.12.263",
                byte_count=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            runner = NebiusCliRunner(expected_identity=identity)

            self.assertEqual(runner.preflight(), identity)
            self.assertEqual(runner.json("profile", "list"), {"items": []})

    def _failed_contact(
        self, *, exit_code: int, stderr: str, arguments: tuple[str, ...]
    ) -> tuple[NebiusCliError, str]:
        def process_runner(argv, **kwargs):
            if argv[-1] == "version":
                return subprocess.CompletedProcess(argv, 0, "0.12.263\n", "")
            return subprocess.CompletedProcess(argv, exit_code, "", stderr)

        runner = NebiusCliRunner(
            expected_identity=self._identity(),
            process_runner=process_runner,
            executable_inspector=self._inspector,
        )
        operator_record = io.StringIO()
        with redirect_stderr(operator_record), self.assertRaises(NebiusCliError) as c:
            runner(*arguments)
        return c.exception, operator_record.getvalue()

    def test_provider_diagnosis_crosses_the_boundary_as_identifiers(self):
        error, _ = self._failed_contact(
            exit_code=13,
            stderr=_NOT_FOUND_STDERR,
            arguments=("ai", "job", "get-by-name", "--name", "exact"),
        )
        evidence = error.failure_evidence
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.child_exit_code, 13)
        self.assertEqual(evidence.rpc_code, "NotFound")
        self.assertEqual(evidence.trace_id, "ba6fd61a3f05c84c639836764059c44e")
        self.assertEqual(evidence.stderr_classification, "not_found")
        self.assertEqual(
            error.journey_reason(),
            "provider_command_failed;exit=13;rpc=NotFound"
            ";trace=ba6fd61a3f05c84c639836764059c44e",
        )
        self.assertIn("rpc_code:NotFound", evidence.public_fields())
        self.assertIn(
            "trace_id:ba6fd61a3f05c84c639836764059c44e", evidence.public_fields()
        )
        # The identifiers cross and the provider's free text does not.  `desc =` and
        # the request id are deliberately left behind with it.
        journey_bytes = " ".join((error.journey_reason(), *evidence.public_fields()))
        for retained_nowhere in (
            "desc",
            "not found",
            "bc33135d-cd17-4ed8-b47c-ea91eee160e4",
            "This issue may be linked",
        ):
            self.assertNotIn(retained_nowhere, journey_bytes)

    def test_raw_provider_stderr_reaches_only_the_operator_record(self):
        error, operator_record = self._failed_contact(
            exit_code=7,
            stderr=_AUTH_PROMPT_STDERR,
            arguments=("iam", "tenant", "list", "--profile", "research"),
        )
        evidence = error.failure_evidence
        self.assertIsNotNone(evidence)
        # The operator record keeps every byte, delimited and bound to its own digest.
        self.assertIn(PROVIDER_STDERR_OPEN, operator_record)
        self.assertIn(PROVIDER_STDERR_CLOSE, operator_record)
        self.assertIn(f"stderr_sha256:{evidence.stderr_sha256}", operator_record)
        self.assertIn(_AUTH_PROMPT_STDERR, operator_record)
        # The Journey side keeps the exit code and nothing the prompt was carrying.
        self.assertEqual(evidence.child_exit_code, 7)
        self.assertIsNone(evidence.rpc_code)
        self.assertIsNone(evidence.trace_id)
        self.assertEqual(error.journey_reason(), "provider_command_failed;exit=7")
        self.assertEqual(
            evidence.stderr_classification, "interactive_authentication_required"
        )
        journey_bytes = " ".join(
            (
                error.journey_reason(),
                *evidence.public_fields(),
                repr(error),
                repr(evidence),
                repr(error.__dict__),
            )
        )
        for oauth_material in (
            "code_challenge",
            "Zm9vYmFyc2VudGluZWxjaGFsbGVuZ2U",
            "state-sentinel-value",
            "https://auth.example.invalid",
            _SECRET_KEY,
        ):
            self.assertIn(oauth_material, operator_record)
            self.assertNotIn(oauth_material, journey_bytes)

    def test_one_not_found_answer_is_retained_raw_at_both_of_its_exit_codes(self):
        # docs.nebius.com/cli/exit-codes: an error raised by an extra call the CLI makes
        # internally reports the target code plus 40, so 13 and 53 are one condition.
        # This boundary retains the literal number and classifies the answer; nothing
        # here keys behaviour on the bare code, which is what lets an absence proof
        # require both halves.
        retained = []
        for exit_code in (13, 53):
            with self.subTest(exit_code=exit_code):
                error, _ = self._failed_contact(
                    exit_code=exit_code,
                    stderr=_NOT_FOUND_STDERR,
                    arguments=("ai", "job", "get", "--id", "aijob-exact"),
                )
                evidence = error.failure_evidence
                self.assertEqual(evidence.child_exit_code, exit_code)
                retained.append(
                    (
                        evidence.rpc_code,
                        evidence.stderr_classification,
                        evidence.trace_id,
                    )
                )
        self.assertEqual(retained[0], retained[1])
        self.assertEqual(retained[0][:2], ("NotFound", "not_found"))

    def test_bounded_provider_identifiers_refuse_uncanonical_material(self):
        base = {
            "operation": "ai.job.get",
            "child_exit_code": 13,
            "stdout_byte_count": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_byte_count": 1,
            "stderr_sha256": hashlib.sha256(b"x").hexdigest(),
            "stderr_classification": "not_found",
        }
        for field, value in (
            ("rpc_code", "Not Found"),
            ("rpc_code", f"NotFound {_SECRET_KEY}"),
            ("rpc_code", ""),
            ("trace_id", "BA6FD61A3F05C84C639836764059C44E"),
            ("trace_id", "ba6fd61a3f05c84c639836764059c44"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    NebiusCliFailureEvidence(**base, **{field: value})


def _raise_deep_type_error() -> None:
    raise TypeError(f"provider payload rejected: {_SECRET_KEY}")


_DEEP_TYPE_ERROR_LINE = _raise_deep_type_error.__code__.co_firstlineno + 1


def _raise_through_one_wrapper() -> None:
    _raise_deep_type_error()


class FailureCauseTests(unittest.TestCase):
    """Ticket 139: one shared helper names a refusal's cause as code identity."""

    @staticmethod
    def _caught(raiser) -> BaseException:
        try:
            raiser()
        except Exception as error:
            return error
        raise AssertionError("the raiser did not raise")

    def test_cause_names_the_raise_site_rather_than_the_catch_site(self):
        cause = cause_identity(self._caught(_raise_through_one_wrapper))
        self.assertEqual(
            cause, f"TypeError@test_nebius_cli_runner.py:{_DEEP_TYPE_ERROR_LINE}"
        )

    def test_cause_separates_defects_a_bare_class_name_collapses(self):
        # Three defects, one class name, and once one indistinguishable
        # `reason: TypeError`: each must stay distinguishable on the record.
        def first() -> None:
            raise TypeError("provider job omitted spec")

        def second() -> None:
            raise TypeError("provider job status is not an object")

        def third() -> None:
            raise TypeError("provider omitted its own spec timeout")

        causes = [
            cause_identity(self._caught(raiser)) for raiser in (first, second, third)
        ]
        self.assertEqual(
            [cause.partition("@")[0] for cause in causes],
            ["TypeError", "TypeError", "TypeError"],
        )
        self.assertEqual(len(set(causes)), 3)

    def test_cause_carries_code_identity_and_never_the_exception_message(self):
        cause = cause_identity(self._caught(_raise_deep_type_error))
        self.assertNotIn(_SECRET_KEY, cause)
        self.assertNotIn("provider payload rejected", cause)
        class_name, separator, site = cause.partition("@")
        module_name, colon, line = site.rpartition(":")
        self.assertEqual((separator, colon), ("@", ":"))
        self.assertEqual(class_name, "TypeError")
        self.assertEqual(module_name, "test_nebius_cli_runner.py")
        self.assertTrue(line.isdigit())

    def test_cause_falls_back_to_the_class_name_without_a_traceback(self):
        self.assertEqual(cause_identity(ValueError("never raised")), "ValueError")


if __name__ == "__main__":
    unittest.main()
