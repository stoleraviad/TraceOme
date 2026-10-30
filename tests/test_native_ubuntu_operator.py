from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from bio_neb.devtools import native_ubuntu_operator as operator

SHA = "a" * 40


class NativeUbuntuOperatorTests(unittest.TestCase):
    def test_all_capsule_operations_refuse_noncanonical_run_root(self):
        arbitrary = "/tmp/arbitrary-capsule"
        operations = (
            lambda: operator.status(arbitrary),
            lambda: operator.attach(arbitrary),
            lambda: operator.interrupt(arbitrary),
            lambda: operator.inspect(arbitrary),
        )
        with (
            patch.object(operator, "RUN_ROOTS", Path("/canonical/runs")),
            patch.object(
                operator,
                "_canonical_existing_path",
                side_effect=lambda value, _field: Path(value),
            ),
        ):
            for operation in operations:
                with (
                    self.subTest(operation=operation),
                    self.assertRaisesRegex(
                        operator.OperatorRefused, "canonical_run_root_required"
                    ),
                ):
                    operation()

    def test_operator_paths_requires_basename_run_id_agreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = Path(temporary) / "runs"
            root = roots / "literal-run-id"
            record = root / "operator"
            record.mkdir(parents=True)
            (record / "command.json").write_text(
                json.dumps(
                    {
                        "run_id": "different-run-id",
                        "run_root": str(root),
                        "operator_directory": str(record),
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(operator, "RUN_ROOTS", roots),
                patch.object(
                    operator,
                    "_canonical_existing_path",
                    side_effect=lambda value, _field: Path(value),
                ),
                self.assertRaisesRegex(
                    operator.OperatorRefused, "operator_run_identity_mismatch"
                ),
            ):
                operator._operator_paths(str(root))

    def test_help_exposes_exactly_six_public_operations(self):
        parser = operator._parser()
        choices = parser._subparsers._group_actions[0].choices
        self.assertEqual(
            set(choices),
            {"provision", "start", "status", "attach", "interrupt", "inspect"},
        )
        help_text = parser.format_help()
        self.assertNotIn("_run-child", help_text)

    def test_internal_child_dispatch_accepts_only_exact_private_argv(self):
        with patch.object(operator, "_run_child", return_value=7) as run_child:
            self.assertEqual(
                operator.main(["_run-child", "--run-root", "/native/run"]), 7
            )
        run_child.assert_called_once_with("/native/run")

        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(
                operator.main(
                    ["_run-child", "--run-root", "/native/run", "unexpected"]
                ),
                2,
            )
        self.assertEqual(
            json.loads(error.getvalue())["code"], "internal_child_argv_invalid"
        )

    def test_allowlisted_environment_preserves_supplied_path_and_drops_unlisted(self):
        supplied_path = "/home/operator/.local/bin:/usr/local/bin:/usr/bin:/bin"
        selected = operator.allowlisted_environment(
            {
                "HOME": "/home/operator",
                "PATH": supplied_path,
                "WSL_INTEROP": "/run/WSL/123_interop",
                "TERM": "screen-256color",
                "MODEL_API_KEY": "secret",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "TMUX": "/tmp/old,1,0",
                "TMUX_PANE": "%9",
            }
        )

        self.assertEqual(
            selected,
            {
                "HOME": "/home/operator",
                "PATH": supplied_path,
                "TERM": "screen-256color",
            },
        )

    def test_allowlisted_environment_uses_minimal_path_only_when_unsupplied(self):
        selected = operator.allowlisted_environment(
            {
                "HOME": "/home/bhalu",
                "MODEL_API_KEY": "secret",
                "TMUX": "/tmp/old,1,0",
                "TMUX_PANE": "%9",
            }
        )

        self.assertEqual(
            selected,
            {
                "HOME": "/home/bhalu",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
        )

    def test_cli_preserves_every_command_argument_without_shell_parsing(self):
        captured = {}

        def fake_start(**kwargs):
            captured.update(kwargs)
            return {"status": "started"}

        argv = [
            "start",
            "--run-root",
            "/home/bhalu/bio-neb-live-runs/operator-argv-proof",
            "--run-id",
            "operator-argv-proof",
            "--checkout",
            f"/home/bhalu/bio-neb-controller/checkouts/{SHA}",
            "--expected-sha",
            SHA,
            "--",
            "python",
            "-c",
            "print('a b; $HOME')",
            "literal * ? [x]",
            "--flag=value with spaces",
        ]
        output = io.StringIO()
        with (
            patch.object(operator, "start", side_effect=fake_start),
            redirect_stdout(output),
        ):
            self.assertEqual(operator.main(argv), 0)

        self.assertEqual(
            captured["argv"],
            [
                "python",
                "-c",
                "print('a b; $HOME')",
                "literal * ? [x]",
                "--flag=value with spaces",
            ],
        )
        self.assertEqual(json.loads(output.getvalue()), {"status": "started"})

    def test_credential_command_requires_exact_uv_env_file_argv(self):
        path = Path(operator.CONTROLLER_ENV)
        accepted = ["uv", "run", "--env-file", str(path), "pytest", "-q"]
        operator._validate_credential_argv(accepted, path)

        must_use_uv = "credential_command_must_use_uv_env_file"
        refused = (
            (["bash", "-lc", "uv run pytest"], must_use_uv),
            (["uv", "run", "pytest"], must_use_uv),
            (
                ["uv", "run", "--env-file", "/tmp/other.env", "pytest"],
                "credential_command_env_file_mismatch",
            ),
            (["uv", "--env-file", str(path), "run", "pytest"], must_use_uv),
            (
                [
                    "uv",
                    "run",
                    "--env-file",
                    str(path),
                    "--env-file",
                    str(path),
                    "pytest",
                ],
                "credential_command_must_use_one_env_file",
            ),
            (["uv", "run", f"--env-file={path}", "pytest"], must_use_uv),
        )
        for argv, code in refused:
            with (
                self.subTest(argv=argv),
                self.assertRaises(operator.OperatorRefused) as raised,
            ):
                operator._validate_credential_argv(argv, path)
            self.assertEqual(raised.exception.code, code)

    def test_stream_guard_refuses_secret_split_across_chunks_before_publication(self):
        guard = operator._StreamGuard((b"top-secret",))
        first = guard.feed(b"safe top-se")
        self.assertEqual(first, b"safe ")
        with self.assertRaises(operator._SecretOutputRefused):
            guard.feed(b"cret trailing")

    def test_stream_guard_releases_safe_prompt_without_newline_or_later_output(self):
        guard = operator._StreamGuard((b"synthetic-credential-longer-than-the-prompt",))
        self.assertEqual(
            guard.feed(b"Review [approve/stop]: "), b"Review [approve/stop]: "
        )

    def test_stream_guard_excludes_secret_prefix_at_every_split(self):
        patterns = (b"synthetic-secret", b"synthetic-second", b"ababab", b"!")
        for secret in patterns:
            for split in range(1, len(secret)):
                with self.subTest(secret=secret, split=split):
                    guard = operator._StreamGuard(patterns)
                    self.assertEqual(guard.feed(b"safe " + secret[:split]), b"safe ")
                    with self.assertRaises(operator._SecretOutputRefused):
                        guard.feed(secret[split:] + b" after")
        guard = operator._StreamGuard(patterns)
        self.assertEqual(guard.feed(b"safe sy"), b"safe ")
        self.assertEqual(guard.feed(b"ndicate: "), b"syndicate: ")
        with self.assertRaises(operator._SecretOutputRefused):
            guard.feed(b"!")

    def test_stream_guard_retains_ambiguous_prefix_until_disambiguation_or_eof(self):
        guard = operator._StreamGuard((b"ababab", b"synthetic-secret"))
        for byte in b"synthetic-secre":
            self.assertEqual(guard.feed(bytes([byte])), b"")
            self.assertEqual(guard.feed(b""), b"")
        with self.assertRaises(operator._SecretOutputRefused):
            guard.feed(b"t")
        self.assertEqual(guard.feed(b"aba"), b"")
        self.assertEqual(guard.feed(b"X"), b"abaX")
        self.assertEqual(guard.feed(b"synthetic"), b"")
        self.assertEqual(guard.feed(b"", final=True), b"synthetic")
        self.assertEqual(operator._StreamGuard(()).feed(b"prompt: "), b"prompt: ")

    def test_tee_publishes_native_review_prompt_before_the_child_can_answer(self):
        prompt = b"Enter one exact legal Review response JSON: "
        targets = (io.BytesIO(), io.BytesIO())
        blocks = iter((prompt, b"accepted\n", b""))

        def read(_size):
            block = next(blocks)
            if block == b"accepted\n":
                for target in targets:
                    self.assertEqual(target.getvalue(), prompt)
            return block

        errors = []
        refused = operator.threading.Event()
        operator._tee(
            SimpleNamespace(read=read),
            targets,
            operator._StreamGuard((b"synthetic-secret-" * 8,)),
            errors,
            refused,
        )
        self.assertEqual(errors, [])
        self.assertFalse(refused.is_set())
        for target in targets:
            self.assertEqual(target.getvalue(), prompt + b"accepted\n")

    def test_start_refuses_unsafe_socket_bytes_and_names_before_launch_or_reservation(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkout = base / SHA
            checkout.mkdir()
            root = base / "run"
            with (
                patch.object(operator, "RUN_ROOTS", base),
                patch.object(operator, "CHECKOUTS_ROOT", base),
                patch.object(operator, "_TMUX_SOCKET_ROOT", base / ("é" * 25)),
                patch.object(operator, "_run") as run,
            ):
                with self.assertRaisesRegex(
                    operator.OperatorRefused, "tmux_socket_path_too_long"
                ):
                    operator.start(
                        run_root=str(root),
                        run_id="run",
                        checkout=str(checkout),
                        expected_sha=SHA,
                        argv=["/bin/true"],
                    )
                for name in ("../escape", "bad;name", "a" * 129, "é"):
                    with (
                        self.subTest(name=name),
                        self.assertRaisesRegex(
                            operator.OperatorRefused, "run_id_invalid"
                        ),
                    ):
                        operator.start(
                            run_root=str(root),
                            run_id=name,
                            checkout=str(checkout),
                            expected_sha=SHA,
                            argv=["/bin/true"],
                        )
                run.assert_not_called()
                self.assertFalse(root.exists())

    def test_start_preserves_run_identity_with_usable_isolated_tmux_locators(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkout = base / "checkouts" / SHA
            checkout.mkdir(parents=True)
            socket_root = base / "sockets"
            socket_root.mkdir()
            commands = []
            for roots, run_id in (
                (base / "runs", "same-run-id"),
                (base / ("retained-root-" * 12) / "runs", "same-run-id"),
                (base / "dotted" / "runs", "release.1"),
            ):
                roots.mkdir(parents=True)
                root = roots / run_id
                sessions = []

                def launch(argv, *, root=root, sessions=sessions, **_kwargs):
                    if "new-session" in argv:
                        # Native tmux normalizes periods; the returned locator must
                        # still name that session, not the unmodified Run ID.
                        sessions.append(argv[argv.index("-s") + 1].replace(".", "_"))
                        (root / "operator" / "started.json").write_text("{}")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")

                with (
                    patch.object(operator, "RUN_ROOTS", roots),
                    patch.object(operator, "CHECKOUTS_ROOT", checkout.parent),
                    patch.object(
                        operator, "_TMUX_SOCKET_ROOT", socket_root, create=True
                    ),
                    patch.object(operator, "_require_ext4"),
                    patch.object(operator, "_checkout_identity"),
                    patch.object(
                        operator,
                        "_operator_identity",
                        return_value={
                            "ubuntu": {},
                            "helper": {},
                            "tools": {"tmux": {"path": "/usr/bin/tmux"}},
                        },
                    ),
                    patch.object(operator, "_run", side_effect=launch),
                ):
                    result = operator.start(
                        run_root=str(root),
                        run_id=root.name,
                        checkout=str(checkout),
                        expected_sha=SHA,
                        argv=["/bin/true"],
                        environ={},
                    )
                command = result["command"]
                socket = Path(command["tmux_socket"])
                self.assertLessEqual(len(os.fsencode(socket)), 107)
                self.assertFalse(socket.is_relative_to(root))
                self.assertEqual(socket.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(command["run_root"], str(root))
                self.assertEqual(command["run_id"], run_id)
                self.assertEqual(sessions, [command["tmux_session"]])
                commands.append(command)
            self.assertNotEqual(commands[0]["tmux_socket"], commands[1]["tmux_socket"])

    def test_credential_literal_parser_refuses_quoted_escaped_and_interpolated_values(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "controller.env"
            for document in (
                b'MODEL_KEY="quoted"\n',
                b"MODEL_KEY=escaped\\nvalue\n",
                b"MODEL_KEY=value with space\n",
                b"MODEL_KEY=value#comment\n",
                b"MODEL_KEY=${OTHER}\n",
            ):
                path.write_bytes(document)
                with (
                    self.subTest(document=document),
                    self.assertRaisesRegex(
                        operator.OperatorRefused,
                        "credential_environment_literal_refused",
                    ),
                ):
                    operator._parse_credential_literals(path)

    def test_proc_stat_parser_handles_spaces_and_parentheses_in_comm(self):
        suffix = ["S", "1", "456", *(["0"] * 16), "999"]
        parsed = operator._parse_proc_stat(
            f"123 (worker name (nested)) {' '.join(suffix)}\n"
        )
        self.assertEqual(parsed, {"pid": 123, "pgid": 456, "start_ticks": 999})

    def test_accessible_malformed_proc_stat_fails_closed_for_residue(self):
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "iterdir", return_value=iter((Path("/proc/123"),))),
            patch.object(Path, "read_text", return_value="123 malformed"),
            self.assertRaisesRegex(
                operator.OperatorRefused, "process_group_inventory_malformed"
            ),
        ):
            operator._process_group_members(123)

    def test_provision_refuses_existing_checkout_before_git_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "candidate.bundle"
            bundle.write_bytes(b"bundle")
            cache = root / "repo.git"
            checkouts = root / "checkouts"
            runs = root / "runs"
            checkouts.mkdir()
            runs.mkdir()
            (checkouts / SHA).mkdir()
            with (
                patch.object(
                    operator,
                    "_native_path",
                    side_effect=lambda value, _field: Path(value),
                ),
                patch.object(operator, "_require_no_symlink_components"),
                patch.object(operator, "_mkdir_private_tail"),
                patch.object(operator, "_require_ext4"),
                patch.object(operator, "_run") as run,
                self.assertRaisesRegex(
                    operator.OperatorRefused, "candidate_checkout_collision"
                ),
            ):
                operator._provision_candidate_at(
                    bundle=str(bundle),
                    expected_sha=SHA,
                    bare_cache=cache,
                    checkouts_root=checkouts,
                    run_roots=runs,
                )
        run.assert_not_called()

    def test_provision_refuses_bundle_without_expected_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "candidate.bundle"
            bundle.write_bytes(b"bundle")
            cache = root / "repo.git"
            checkouts = root / "checkouts"
            runs = root / "runs"
            cache.parent.mkdir(exist_ok=True)
            checkouts.mkdir()
            runs.mkdir()

            def fake_run(argv, **_kwargs):
                stdout = (
                    f"{'b' * 40} refs/heads/candidate\n" if "list-heads" in argv else ""
                )
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

            with (
                patch.object(
                    operator,
                    "_native_path",
                    side_effect=lambda value, _field: Path(value),
                ),
                patch.object(operator, "_require_no_symlink_components"),
                patch.object(operator, "_mkdir_private_tail"),
                patch.object(operator, "_require_ext4"),
                patch.object(
                    operator, "_referencing_operator_records", return_value=[]
                ),
                patch.object(operator, "_run", side_effect=fake_run),
                self.assertRaisesRegex(
                    operator.OperatorRefused, "candidate_bundle_sha_missing"
                ),
            ):
                operator._provision_candidate_at(
                    bundle=str(bundle),
                    expected_sha=SHA,
                    bare_cache=cache,
                    checkouts_root=checkouts,
                    run_roots=runs,
                )

    def _provision_with_run_roots(
        self, runs: Path, named_run_roots: tuple[str, ...] = ()
    ) -> dict:
        root = runs.parent
        bundle = root / "candidate.bundle"
        bundle.write_bytes(b"bundle")
        checkouts = root / "checkouts"
        checkouts.mkdir()
        # Every provisioning effect, in order; a refusal test asserts it stays empty.
        self.provision_effects: list[object] = []

        def fake_run(argv, **_kwargs):
            self.provision_effects.append(tuple(argv))
            stdout = f"{SHA} refs/heads/candidate\n" if "list-heads" in argv else ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        with (
            patch.object(
                operator,
                "_native_path",
                side_effect=lambda value, _field: Path(value),
            ),
            patch.object(operator, "_require_no_symlink_components"),
            patch.object(
                operator,
                "_mkdir_private_tail",
                side_effect=lambda path, _field: self.provision_effects.append(
                    ("mkdir", path)
                ),
            ),
            patch.object(operator, "_require_ext4"),
            patch.object(operator, "_run", side_effect=fake_run),
            patch.object(operator, "_git_output", return_value=SHA),
            patch.object(operator, "_checkout_identity"),
        ):
            return operator._provision_candidate_at(
                bundle=str(bundle),
                expected_sha=SHA,
                bare_cache=root / "repo.git",
                checkouts_root=checkouts,
                run_roots=runs,
                named_run_roots=named_run_roots,
            )

    def _run_root(self, runs: Path, name: str, command: str) -> Path:
        run_root = runs / name
        (run_root / "operator").mkdir(parents=True)
        (run_root / "operator" / "command.json").write_text(command, encoding="utf-8")
        return run_root

    def test_provision_skips_an_unreadable_old_run_root_and_names_it(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        other = json.dumps({"checkout": "/elsewhere"})
        self._run_root(runs, "readable-old", other)
        unreadable = self._run_root(runs, "164-vcf-refusal-r1", other)
        unreadable.chmod(0o600)
        self.addCleanup(unreadable.chmod, 0o700)

        report = self._provision_with_run_roots(runs)

        self.assertEqual(report["status"], "provisioned")
        self.assertEqual(report["skipped_unreadable_run_roots"], [str(unreadable)])

    def test_provision_skips_an_unreadable_root_even_if_it_names_the_candidate(self):
        # An unreadable record cannot be checked, so the root is skipped and named
        # whatever it references.
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        candidate = json.dumps({"checkout": str(runs.parent / "checkouts" / SHA)})
        hidden = self._run_root(runs, "hidden-reference", candidate)
        hidden.chmod(0o600)
        self.addCleanup(hidden.chmod, 0o700)

        report = self._provision_with_run_roots(runs)

        self.assertEqual(report["status"], "provisioned")
        self.assertEqual(report["skipped_unreadable_run_roots"], [str(hidden)])

    def test_provision_refuses_a_named_run_root_it_cannot_read(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        named = self._run_root(runs, "107-live-a1", json.dumps({"checkout": "/x"}))
        named.chmod(0o600)
        self.addCleanup(named.chmod, 0o700)

        with self.assertRaises(operator.OperatorRefused) as refused:
            self._provision_with_run_roots(runs, named_run_roots=("107-live-a1",))

        self.assertEqual(refused.exception.code, "named_run_root_unreadable")
        self.assertEqual(refused.exception.fields, ("107-live-a1",))
        self.assertEqual(self.provision_effects, [])

    def test_provision_refuses_a_named_run_root_that_is_a_symlink(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        candidate = json.dumps({"checkout": str(runs.parent / "checkouts" / SHA)})
        target = self._run_root(runs.parent / "elsewhere", "real", candidate)
        runs.mkdir()
        (runs / "107-live-a1").symlink_to(target, target_is_directory=True)

        with self.assertRaises(operator.OperatorRefused) as refused:
            self._provision_with_run_roots(runs, named_run_roots=("107-live-a1",))

        self.assertEqual(refused.exception.code, "named_run_root_invalid")
        self.assertEqual(refused.exception.fields, ("107-live-a1",))
        self.assertEqual(self.provision_effects, [])

    def test_provision_refuses_a_named_run_root_that_is_not_a_directory(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        runs.mkdir()
        (runs / "107-live-a1").write_text("not a run root", encoding="utf-8")

        with self.assertRaises(operator.OperatorRefused) as refused:
            self._provision_with_run_roots(runs, named_run_roots=("107-live-a1",))

        self.assertEqual(refused.exception.code, "named_run_root_invalid")
        self.assertEqual(self.provision_effects, [])

    def test_provision_refuses_a_named_run_root_that_names_the_candidate(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        candidate = json.dumps({"checkout": str(runs.parent / "checkouts" / SHA)})
        self._run_root(runs, "107-live-a1", candidate)

        with self.assertRaises(operator.OperatorRefused) as refused:
            self._provision_with_run_roots(runs, named_run_roots=("107-live-a1",))

        self.assertEqual(refused.exception.code, "candidate_checkout_referenced")
        self.assertEqual(self.provision_effects, [])

    def test_provision_accepts_a_named_run_root_that_does_not_exist_yet(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        runs.mkdir()

        report = self._provision_with_run_roots(runs, named_run_roots=("107-live-a2",))

        self.assertEqual(report["status"], "provisioned")
        self.assertEqual(report["named_run_roots"], ["107-live-a2"])

    def test_provision_refuses_a_named_run_root_that_is_not_a_run_id(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        runs.mkdir()

        with self.assertRaises(operator.OperatorRefused) as refused:
            self._provision_with_run_roots(runs, named_run_roots=("../escape",))

        self.assertEqual(refused.exception.code, "run_id_invalid")
        self.assertEqual(self.provision_effects, [])

    def test_provision_cli_passes_each_named_run_root(self):
        with (
            patch.object(
                operator, "provision_candidate", return_value={"status": "ok"}
            ) as provision,
            redirect_stdout(io.StringIO()),
        ):
            code = operator.main(
                [
                    "provision",
                    "--bundle",
                    "/b.bundle",
                    "--expected-sha",
                    SHA,
                    "--named-run-root",
                    "107-live-a1",
                    "--named-run-root",
                    "107-live-a2",
                ]
            )

        self.assertEqual(code, 0)
        provision.assert_called_once_with(
            bundle="/b.bundle",
            expected_sha=SHA,
            named_run_roots=("107-live-a1", "107-live-a2"),
        )

    def test_provision_still_refuses_a_readable_corrupt_operator_record(self):
        runs = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runs"
        self._run_root(runs, "corrupt", "{not json")

        with self.assertRaises(operator.OperatorRefused) as refused:
            self._provision_with_run_roots(runs)

        self.assertEqual(refused.exception.code, "operator_record_invalid")

    def test_empty_terminal_objects_never_satisfy_cleanup_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary)
            self.assertFalse(
                operator._local_terminal_record_valid(
                    {},
                    schema="bio-neb.native-operator-finished/1",
                    operator=record,
                    candidate_sha=SHA,
                )
            )

    def test_exact_cleanup_target_refuses_bulk_and_noncanonical_targets(self):
        root = "/home/bhalu/bio-neb-controller/checkouts"
        exact = f"{root}/{SHA}"
        self.assertEqual(
            str(
                operator.require_exact_cleanup_target(
                    target=exact, checkouts_root=root, expected_sha=SHA
                )
            ).replace("\\", "/"),
            exact,
        )
        exact_required = "cleanup_exact_target_required"
        for target, code in (
            (f"{root}/*", "bulk_cleanup_refused"),
            (root, exact_required),
            (f"{root}/{'b' * 40}", exact_required),
            (f"{root}/{SHA}/.venv", exact_required),
        ):
            with (
                self.subTest(target=target),
                self.assertRaises(operator.OperatorRefused) as raised,
            ):
                operator.require_exact_cleanup_target(
                    target=target, checkouts_root=root, expected_sha=SHA
                )
            self.assertEqual(raised.exception.code, code)

    def test_cleanup_is_not_a_public_operation(self):
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as refused:
            operator.main(["cleanup", "--target", f"/checkouts/{SHA}"])
        self.assertEqual(refused.exception.code, 2)
        self.assertIn("invalid choice: 'cleanup'", error.getvalue())

    def test_checkout_identity_requires_exact_detached_clean_locked_tree(self):
        checkout = Path("/native/checkouts") / SHA
        clean_results = iter((SHA, ""))
        with (
            patch.object(
                operator,
                "_git_output",
                side_effect=lambda *_args, **_kwargs: next(clean_results),
            ),
            patch.object(
                operator,
                "_run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr=""),
            ),
            patch.object(Path, "is_file", return_value=True),
        ):
            operator._checkout_identity(checkout, SHA)

        dirty_results = iter((SHA, " M tracked.py"))
        with (
            patch.object(
                operator,
                "_git_output",
                side_effect=lambda *_args, **_kwargs: next(dirty_results),
            ),
            patch.object(
                operator,
                "_run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr=""),
            ),
            patch.object(Path, "is_file", return_value=True),
            self.assertRaisesRegex(
                operator.OperatorRefused, "candidate_checkout_dirty"
            ),
        ):
            operator._checkout_identity(checkout, SHA)

    def test_checkout_identity_ignores_uv_editable_install_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            source_ignore = Path(__file__).resolve().parents[1] / ".gitignore"
            (checkout / ".gitignore").write_text(
                source_ignore.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (checkout / "uv.lock").write_text("", encoding="utf-8")
            subprocess.run(
                ["git", "init", str(checkout)], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", str(checkout), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "add", ".gitignore", "uv.lock"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "commit", "-m", "fixture"],
                check=True,
                capture_output=True,
            )
            expected_sha = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "-C", str(checkout), "checkout", "--detach", expected_sha],
                check=True,
                capture_output=True,
            )
            metadata = checkout / "src" / "bio_neb_clean_agentic_journey.egg-info"
            metadata.mkdir(parents=True)
            for name in (
                "PKG-INFO",
                "SOURCES.txt",
                "dependency_links.txt",
                "requires.txt",
                "top_level.txt",
            ):
                (metadata / name).write_text("generated\n", encoding="utf-8")

            operator._checkout_identity(checkout, expected_sha)

    def test_status_separates_viewer_local_and_application_facts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "operator"
            record.mkdir()
            (record / "command.json").write_text(
                json.dumps(
                    {
                        "checkout": f"/checkouts/{SHA}",
                        "candidate_sha": SHA,
                        "tmux_socket": str(record / "tmux.sock"),
                        "tmux_session": "bio-neb-status-proof",
                    }
                ),
                encoding="utf-8",
            )
            (record / "started.json").write_text(
                json.dumps({"child_pid": 5, "child_pgid": 5, "child_start_ticks": 9}),
                encoding="utf-8",
            )
            (record / "finished.json").write_text("{}", encoding="utf-8")
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(operator, "_process_identity_matches", return_value=True),
                patch.object(
                    operator,
                    "_tmux_facts",
                    return_value={
                        "server_present": False,
                        "session_present": False,
                        "viewer_attached": False,
                        "pane_dead": None,
                        "pane_pid": None,
                        "pane_current_command": None,
                    },
                ),
            ):
                result = operator.status("/ignored")

        self.assertEqual(
            result["local_command"]["lifecycle"], "abnormal_local_termination"
        )
        self.assertFalse(result["viewer"]["attached"])
        self.assertFalse(result["viewer"]["server_present"])
        self.assertFalse(result["local_command"]["finished_valid"])
        self.assertFalse(result["application_remote_evidence"]["interpreted"])

    def test_status_treats_dead_pane_as_abnormal_even_with_matching_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "operator"
            record.mkdir()
            (record / "command.json").write_text(
                json.dumps(
                    {
                        "checkout": f"/checkouts/{SHA}",
                        "candidate_sha": SHA,
                        "tmux_socket": "/tmp/tmux.sock",
                        "tmux_session": "bio-neb-dead-pane-proof",
                    }
                ),
                encoding="utf-8",
            )
            (record / "started.json").write_text(
                json.dumps({"child_pid": 5, "child_pgid": 5, "child_start_ticks": 9}),
                encoding="utf-8",
            )
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(operator, "_process_identity_matches", return_value=True),
                patch.object(
                    operator,
                    "_tmux_facts",
                    return_value={
                        "server_present": True,
                        "session_present": True,
                        "viewer_attached": False,
                        "pane_dead": True,
                        "pane_pid": 5,
                        "pane_current_command": "python",
                    },
                ),
            ):
                result = operator.status("/ignored")

        self.assertEqual(
            result["local_command"]["lifecycle"], "abnormal_local_termination"
        )

    def test_inspect_refuses_secret_bearing_pane_capture_before_persisting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "operator"
            record.mkdir()
            environment = root / "controller.env"
            environment.write_bytes(b"MODEL_KEY=literal-secret\n")
            command = {
                "credential_environment": {"path": str(environment)},
                "tmux_socket": "/tmp/tmux.sock",
                "tmux_session": "bio-neb-pane-capture-proof",
                "tools": {"tmux": {"path": "/usr/bin/tmux"}},
            }
            (record / "command.json").write_text(json.dumps(command), encoding="utf-8")
            observed = {"viewer": {"session_present": True}}
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(operator, "status", return_value=observed),
                patch.object(
                    operator,
                    "_run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="pane contains literal-secret\n",
                        stderr="",
                    ),
                ),
                self.assertRaisesRegex(
                    operator.OperatorRefused, "pane_capture_credential_value_refused"
                ),
            ):
                operator.inspect("/ignored")

            self.assertFalse((record / "pane-capture.txt").exists())

    def test_inspect_reports_cleanup_only_when_a_preservation_boundary_is_named(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "operator"
            record.mkdir()
            command = {"checkout": str(root / SHA), "candidate_sha": SHA}
            (record / "command.json").write_text(json.dumps(command), encoding="utf-8")
            reported = {"eligible": False, "reasons": ["probe"]}
            cases = (
                ({}, None),
                ({"preserved_sha": "b" * 40}, (None, "b" * 40)),
                ({"preserved_checkout": "/kept"}, ("/kept", None)),
                (
                    {"preserved_checkout": "/kept", "preserved_sha": "b" * 40},
                    ("/kept", "b" * 40),
                ),
            )
            for arguments, boundary in cases:
                with (
                    self.subTest(arguments=arguments),
                    patch.object(
                        operator, "_operator_paths", return_value=(root, record)
                    ),
                    patch.object(
                        operator,
                        "status",
                        return_value={"viewer": {"session_present": False}},
                    ),
                    patch.object(
                        operator, "cleanup_eligibility", return_value=reported
                    ) as eligibility,
                ):
                    result = operator.inspect("/ignored", **arguments)

                if boundary is None:
                    self.assertIsNone(result["cleanup"])
                    eligibility.assert_not_called()
                else:
                    self.assertEqual(result["cleanup"], reported)
                    eligibility.assert_called_once_with(
                        target=str(root / SHA),
                        expected_sha=SHA,
                        preserved_checkout=boundary[0],
                        preserved_sha=boundary[1],
                    )

    def test_inspect_refuses_the_retired_windows_repository_flag(self):
        with redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as exited:
                operator._parser().parse_args(
                    ["inspect", "--run-root", "/run", "--windows-repository", "/repo"]
                )
        self.assertEqual(exited.exception.code, 2)
        self.assertIn("unrecognized arguments: --windows-repository", error.getvalue())

    def test_interrupt_refuses_stale_pid_before_record_or_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "operator"
            record.mkdir()
            for name, value in (
                (
                    "command.json",
                    {"tmux_socket": "/tmp/socket", "tmux_session": "session"},
                ),
                (
                    "started.json",
                    {"child_pid": 123, "child_pgid": 123, "child_start_ticks": 99},
                ),
            ):
                (record / name).write_text(json.dumps(value), encoding="utf-8")
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(operator, "_process_identity_matches", return_value=False),
                patch.object(os, "killpg", create=True) as killpg,
                self.assertRaisesRegex(
                    operator.OperatorRefused, "interrupt_stale_process_identity"
                ),
            ):
                operator.interrupt("/ignored")

        killpg.assert_not_called()
        self.assertFalse((record / "interruption.json").exists())

    def test_interrupt_records_sigint_without_provider_or_tmux_kill_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "operator"
            record.mkdir()
            command = {"tmux_socket": "/tmp/socket", "tmux_session": "session"}
            started = {"child_pid": 123, "child_pgid": 123, "child_start_ticks": 99}
            (record / "command.json").write_text(json.dumps(command), encoding="utf-8")
            (record / "started.json").write_text(json.dumps(started), encoding="utf-8")
            published = {}

            def fake_publish(path, value, **_kwargs):
                published[path.name] = value

            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(operator, "_process_identity_matches", return_value=True),
                patch.object(operator, "_atomic_publish", side_effect=fake_publish),
                patch.object(os, "killpg", create=True) as killpg,
            ):
                result = operator.interrupt("/ignored")

        killpg.assert_called_once_with(123, operator.signal.SIGINT)
        self.assertEqual(published["interruption.json"], result)
        self.assertFalse(result["provider_cancellation_inferred"])
        self.assertFalse(result["tmux_server_killed"])

    def test_cleanup_eligibility_reports_only_and_never_deletes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / SHA
            candidate.mkdir()
            cache = root / "cache.git"
            cache.mkdir()
            runs = root / "runs"
            runs.mkdir()
            with (
                patch.object(
                    operator,
                    "require_exact_cleanup_target",
                    side_effect=lambda **kwargs: Path(kwargs["target"]),
                ),
                patch.object(
                    operator,
                    "_native_path",
                    side_effect=lambda value, _field: Path(value),
                ),
                patch.object(operator, "_checkout_identity"),
                patch.object(
                    operator, "_commit_reachable", return_value=True
                ) as reachable,
                patch.object(
                    operator, "_commit_ancestor", return_value=True
                ) as ancestor,
                patch.object(
                    operator,
                    "_referencing_operator_records",
                    return_value=[root / "terminal-run"],
                ),
                patch.object(
                    operator,
                    "_read_json",
                    side_effect=lambda path: (
                        {
                            "schema_version": "bio-neb.native-operator-command/1",
                            "candidate_sha": SHA,
                            "checkout": str(candidate),
                        }
                        if path.name == "command.json"
                        else {"started": True}
                        if path.name == "started.json"
                        else {"terminal": True}
                        if path.name == "finished.json"
                        else None
                    ),
                ),
                patch.object(operator, "_receipt_pointer_valid", return_value=True),
                patch.object(operator, "_started_record_valid", return_value=True),
                patch.object(
                    operator,
                    "_local_terminal_record_valid",
                    side_effect=lambda _record, **kwargs: kwargs["schema"].endswith(
                        "finished/1"
                    ),
                ),
                patch.object(
                    operator, "_tmux_facts", return_value={"session_present": False}
                ),
                patch.object(operator, "_processes_referencing", return_value=[]),
                patch.object(
                    Path, "unlink", side_effect=AssertionError("must not delete")
                ),
            ):
                result = operator._cleanup_eligibility_at(
                    target=str(candidate),
                    expected_sha=SHA,
                    preserved_checkout=str(root / ("b" * 40)),
                    preserved_sha="b" * 40,
                    bare_cache=str(cache),
                    checkouts_root=str(root),
                    run_roots=str(runs),
                )

        self.assertTrue(result["eligible"])
        self.assertFalse(result["deletion_performed"])
        self.assertEqual(result["deletion_authority"], "none")
        # The native bare cache is the only commit boundary consulted.
        self.assertEqual(reachable.call_args_list, [call(cache, SHA)])
        self.assertEqual(ancestor.call_args_list, [call(cache, SHA, "b" * 40)])

    def test_cleanup_eligibility_fails_closed_without_boundary_or_run_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / SHA
            candidate.mkdir()
            cache = root / "cache.git"
            cache.mkdir()
            runs = root / "runs"
            runs.mkdir()
            with (
                patch.object(
                    operator, "require_exact_cleanup_target", return_value=candidate
                ),
                patch.object(
                    operator,
                    "_native_path",
                    side_effect=lambda value, _field: Path(value),
                ),
                patch.object(operator, "_checkout_identity"),
                patch.object(
                    operator, "_commit_reachable", return_value=True
                ) as reachable,
                patch.object(
                    operator, "_referencing_operator_records", return_value=[]
                ),
                patch.object(operator, "_processes_referencing", return_value=[]),
            ):
                result = operator._cleanup_eligibility_at(
                    target=str(candidate),
                    expected_sha=SHA,
                    bare_cache=str(cache),
                    checkouts_root=str(root),
                    run_roots=str(runs),
                )

        self.assertFalse(result["eligible"])
        self.assertIn("cleanup_preservation_boundary_unknown", result["reasons"])
        self.assertIn("receipt_bound_attempt_reference_missing", result["reasons"])
        self.assertEqual(reachable.call_args_list, [call(cache, SHA)])

    def test_cleanup_eligibility_explicitly_preserves_current_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / SHA
            candidate.mkdir()
            with (
                patch.object(
                    operator, "require_exact_cleanup_target", return_value=candidate
                ),
                patch.object(
                    operator,
                    "_native_path",
                    side_effect=lambda value, _field: Path(value),
                ),
                patch.object(operator, "_checkout_identity"),
                patch.object(operator, "_commit_reachable", return_value=True),
                patch.object(operator, "_commit_ancestor", return_value=True),
                patch.object(
                    operator, "_referencing_operator_records", return_value=[]
                ),
                patch.object(operator, "_processes_referencing", return_value=[]),
            ):
                result = operator._cleanup_eligibility_at(
                    target=str(candidate),
                    expected_sha=SHA,
                    preserved_checkout=str(candidate),
                    preserved_sha=SHA,
                    bare_cache=str(root),
                    checkouts_root=str(root),
                    run_roots=str(root),
                )

        self.assertFalse(result["eligible"])
        self.assertIn("current_checkout_preserved", result["reasons"])

    def test_operator_identity_accepts_the_devcontainer_release_only(self):
        tool_identity = {
            "path": "/tool",
            "version": "tool 1",
            "byte_count": 1,
            "sha256": "f" * 64,
        }
        cases = {
            "debian-13": (
                'ID=debian\nVERSION_ID="13"\nPRETTY_NAME="Debian 13"\n',
                True,
            ),
            "debian-12": (
                'ID=debian\nVERSION_ID="12"\nPRETTY_NAME="Debian 12"\n',
                False,
            ),
            "ubuntu-22.04": ('ID=ubuntu\nVERSION_ID="22.04"\n', False),
        }
        for name, (release, accepted) in cases.items():
            with (
                self.subTest(name=name),
                patch.object(Path, "read_text", return_value=release),
                patch.object(
                    operator,
                    "_run",
                    return_value=SimpleNamespace(
                        stdout="Linux 6.6 x86_64\n", stderr=""
                    ),
                ),
                patch.object(
                    operator.shutil,
                    "which",
                    side_effect=lambda name: f"/usr/bin/{name}",
                ),
                patch.object(operator, "_version_identity", return_value=tool_identity),
            ):
                if accepted:
                    result = operator._operator_identity()
                    self.assertEqual(result["ubuntu"]["id"], "debian")
                    self.assertEqual(result["ubuntu"]["version_id"], "13")
                else:
                    with self.assertRaises(operator.OperatorRefused) as refused:
                        operator._operator_identity()
                    self.assertEqual(
                        refused.exception.code, "accepted_os_release_required"
                    )

    def test_operator_identity_records_ubuntu_helper_and_exact_tools(self):
        tool_identity = {
            "path": "/tool",
            "version": "tool 1",
            "byte_count": 1,
            "sha256": "f" * 64,
        }
        requested_tool_identities = []

        def fake_version_identity(executable, *version_arguments):
            requested_tool_identities.append((executable, version_arguments))
            return tool_identity

        with (
            patch.object(
                Path,
                "read_text",
                return_value='ID=ubuntu\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04"\n',
            ),
            patch.object(
                operator,
                "_run",
                return_value=SimpleNamespace(
                    stdout="Linux 6.8 x86_64 GNU/Linux\n", stderr=""
                ),
            ),
            patch.object(
                operator.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"
            ),
            patch.object(
                operator, "_version_identity", side_effect=fake_version_identity
            ),
        ):
            result = operator._operator_identity()

        self.assertEqual(result["ubuntu"]["version_id"], "24.04")
        self.assertEqual(set(result["tools"]), {"python", "uv", "git", "tmux"})
        self.assertEqual(
            result["helper"]["module"],
            "bio_neb.devtools.native_ubuntu_operator",
        )
        self.assertIn(
            (str(operator.UV_EXECUTABLE), ("--version",)), requested_tool_identities
        )
        self.assertRegex(result["helper"]["sha256"], r"^[0-9a-f]{64}$")

    def test_started_record_binds_pane_tty_and_distinct_live_streams(self):
        command = {
            "argv": ["python", "-c", "print('ok')"],
            "candidate_sha": SHA,
            "checkout": f"/checkouts/{SHA}",
            "cwd": f"/checkouts/{SHA}",
            "tmux_socket": "/runs/a/operator/tmux.sock",
            "tmux_session": "bio-neb-a",
        }
        started = operator._started_record(
            command=command,
            operator_directory=Path("/runs/a/operator"),
            identity={"pid": 10, "pgid": 10, "start_ticks": 20},
            pane_tty={
                "path": "/dev/pts/4",
                "device": 1,
                "inode": 2,
                "inherited_by_child": True,
            },
        )

        self.assertTrue(started["child_stdin"]["inherited_by_child"])
        self.assertEqual(started["child_stdin"]["path"], "/dev/pts/4")
        self.assertNotEqual(
            started["child_stdout"]["path"], started["child_stderr"]["path"]
        )
        self.assertIn("pane_stdout", started["child_stdout"]["transport"])
        self.assertIn("pane_stderr", started["child_stderr"]["transport"])


if __name__ == "__main__":
    unittest.main()
