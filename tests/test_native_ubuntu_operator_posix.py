from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bio_neb.devtools import native_ubuntu_operator as operator

SHA = "a" * 40


def _write_child_capsule(
    root: Path, argv: list[str], *, credential: Path | None = None
) -> Path:
    record = root / "operator"
    record.mkdir()
    (record / "stdout.log").write_bytes(b"")
    (record / "stderr.log").write_bytes(b"")
    command = {
        "schema_version": "bio-neb.native-operator-command/1",
        "candidate_sha": SHA,
        "checkout": "/native/checkout",
        "cwd": str(root),
        "argv": argv,
        "tmux_socket": str(record / "tmux.sock"),
        "tmux_session": "bio-neb-test",
        "credential_environment": None
        if credential is None
        else {"path": str(credential)},
    }
    (record / "command.json").write_text(json.dumps(command), encoding="utf-8")
    (record / "launch-ready.json").write_text("{}", encoding="utf-8")
    return record


class NativeUbuntuOperatorPosixTests(unittest.TestCase):
    def test_first_provision_safely_creates_absent_private_cache_and_checkout_roots(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "candidate.bundle"
            bundle.write_bytes(b"bundle")
            cache = root / "controller" / "repo.git"
            checkouts = root / "checkouts"
            runs = root / "runs"
            runs.mkdir()

            def fake_run(argv, **_kwargs):
                items = list(argv)
                if items[:3] == ["git", "init", "--bare"]:
                    Path(items[-1]).mkdir(parents=True, exist_ok=True)
                if "list-heads" in items:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=f"{SHA} refs/heads/candidate\n",
                        stderr="",
                    )
                if "rev-parse" in items:
                    return SimpleNamespace(returncode=0, stdout=f"{SHA}\n", stderr="")
                if "worktree" in items and "add" in items:
                    target = Path(items[-2])
                    target.mkdir()
                    (target / "uv.lock").write_text("", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch.object(operator, "_require_ext4"),
                patch.object(operator, "_checkout_identity"),
                patch.object(operator, "_run", side_effect=fake_run),
            ):
                receipt = operator._provision_candidate_at(
                    bundle=str(bundle),
                    expected_sha=SHA,
                    bare_cache=cache,
                    checkouts_root=checkouts,
                    run_roots=runs,
                )
            self.assertTrue(cache.is_dir())
            self.assertTrue(checkouts.is_dir())
            self.assertEqual(receipt["candidate_sha"], SHA)

    def test_canonical_path_refuses_symlink_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(
                operator.OperatorRefused, "symlink_path_refused"
            ):
                operator._require_no_symlink_components(linked, "checkout")

    def test_receipt_pointer_binds_candidate_run_terminal_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            record = run_root / "operator"
            record.mkdir()
            terminal = record / "finished.json"
            evidence = run_root / "fake-evidence.json"
            terminal.write_text('{"terminal":true}\n', encoding="utf-8")
            evidence.write_text('{"fake":true}\n', encoding="utf-8")
            receipt = run_root / "receipt.json"
            document = {
                "schema_version": "bio-neb.native-operator-receipt/1",
                "candidate_sha": SHA,
                "run_root": str(run_root),
                "local_terminal_path": str(terminal),
                "local_terminal_sha256": operator._sha256(terminal),
                "evidence_path": str(evidence),
                "evidence_sha256": operator._sha256(evidence),
            }
            receipt.write_text(json.dumps(document), encoding="utf-8")
            pointer = {
                "schema_version": "bio-neb.native-operator-receipt-pointer/1",
                "candidate_sha": SHA,
                "receipt_path": str(receipt),
                "receipt_sha256": operator._sha256(receipt),
            }
            (record / "receipt-pointer.json").write_text(
                json.dumps(pointer), encoding="utf-8"
            )
            self.assertTrue(operator._receipt_pointer_valid(record, SHA))

            document["candidate_sha"] = "b" * 40
            receipt.write_text(json.dumps(document), encoding="utf-8")
            pointer["receipt_sha256"] = operator._sha256(receipt)
            (record / "receipt-pointer.json").write_text(
                json.dumps(pointer), encoding="utf-8"
            )
            self.assertFalse(operator._receipt_pointer_valid(record, SHA))

    def test_child_nonzero_exit_finalizes_distinct_streams(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _write_child_capsule(
                root,
                [
                    sys.executable,
                    "-c",
                    "import sys; print('stdout-fact'); print('stderr-fact', file=sys.stderr); sys.exit(7)",
                ],
            )
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(
                    operator,
                    "_pane_tty_identity",
                    return_value={
                        "path": "/dev/pts/test",
                        "device": 1,
                        "inode": 2,
                        "inherited_by_child": True,
                    },
                ),
                patch.object(operator, "_set_foreground_process_group"),
                patch.object(
                    operator.sys, "stdout", SimpleNamespace(buffer=io.BytesIO())
                ),
                patch.object(
                    operator.sys, "stderr", SimpleNamespace(buffer=io.BytesIO())
                ),
            ):
                self.assertEqual(operator._run_child(str(root)), 7)

            finished = json.loads(
                (record / "finished.json").read_text(encoding="utf-8")
            )
            self.assertEqual(finished["exit_code"], 7)
            self.assertIsNone(finished["signal"])
            self.assertEqual((record / "stdout.log").read_text(), "stdout-fact\n")
            self.assertEqual((record / "stderr.log").read_text(), "stderr-fact\n")

    def test_child_signal_exit_finalizes_signal_fact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _write_child_capsule(
                root,
                [
                    sys.executable,
                    "-c",
                    "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
                ],
            )
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(
                    operator,
                    "_pane_tty_identity",
                    return_value={"path": "/dev/pts/test", "inherited_by_child": True},
                ),
                patch.object(operator, "_set_foreground_process_group"),
                patch.object(
                    operator.sys, "stdout", SimpleNamespace(buffer=io.BytesIO())
                ),
                patch.object(
                    operator.sys, "stderr", SimpleNamespace(buffer=io.BytesIO())
                ),
            ):
                self.assertEqual(operator._run_child(str(root)), 143)
            finished = json.loads(
                (record / "finished.json").read_text(encoding="utf-8")
            )
            self.assertEqual(finished["signal"], operator.signal.SIGTERM)
            self.assertIsNone(finished["exit_code"])

    def test_child_secret_output_freezes_before_secret_reaches_streams(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = root / "controller.env"
            environment.write_bytes(b"MODEL_KEY=split-secret\n")
            record = _write_child_capsule(
                root,
                [
                    sys.executable,
                    "-c",
                    "import os; os.write(1, b'before split-'); os.write(1, b'secret after')",
                ],
                credential=environment,
            )
            visible = io.BytesIO()
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(
                    operator,
                    "_pane_tty_identity",
                    return_value={"path": "/dev/pts/test", "inherited_by_child": True},
                ),
                patch.object(operator, "_set_foreground_process_group"),
                patch.object(operator.sys, "stdout", SimpleNamespace(buffer=visible)),
                patch.object(
                    operator.sys, "stderr", SimpleNamespace(buffer=io.BytesIO())
                ),
            ):
                self.assertEqual(operator._run_child(str(root)), 125)

            retained = (record / "stdout.log").read_bytes()
            self.assertNotIn(b"split-secret", retained)
            self.assertNotIn(b"split-secret", visible.getvalue())
            self.assertFalse((record / "finished.json").exists())
            frozen = json.loads((record / "frozen.json").read_text(encoding="utf-8"))
            self.assertEqual(frozen["reason"], "credential_value_in_child_output")

    def test_child_descendant_residue_freezes_instead_of_finishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _write_child_capsule(root, [sys.executable, "-c", "pass"])
            residue = [{"pid": 999, "pgid": 500, "start_ticks": 22}]
            with (
                patch.object(operator, "_operator_paths", return_value=(root, record)),
                patch.object(
                    operator,
                    "_pane_tty_identity",
                    return_value={"path": "/dev/pts/test", "inherited_by_child": True},
                ),
                patch.object(operator, "_set_foreground_process_group"),
                patch.object(operator, "_process_group_members", return_value=residue),
                patch.object(
                    operator.sys, "stdout", SimpleNamespace(buffer=io.BytesIO())
                ),
                patch.object(
                    operator.sys, "stderr", SimpleNamespace(buffer=io.BytesIO())
                ),
            ):
                self.assertEqual(operator._run_child(str(root)), 125)

            self.assertFalse((record / "finished.json").exists())
            frozen = json.loads((record / "frozen.json").read_text(encoding="utf-8"))
            self.assertEqual(frozen["reason"], "owned_process_group_residue")
            self.assertEqual(frozen["residue"], residue)

    def test_credential_validation_returns_names_and_never_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            parent.chmod(0o700)
            environment = parent / "controller.env"
            environment.write_text(
                "MODEL_KEY=top-secret\nS3_KEY=also-secret\n", encoding="utf-8"
            )
            environment.chmod(0o600)
            with patch.object(operator, "CONTROLLER_ENV", environment):
                result = operator.validate_credential_environment(
                    str(environment), ("MODEL_KEY", "S3_KEY")
                )

        rendered = json.dumps(result, sort_keys=True)
        self.assertEqual(result["present_names"], ["MODEL_KEY", "S3_KEY"])
        self.assertNotIn("top-secret", rendered)
        self.assertNotIn("also-secret", rendered)

    def test_capsule_accepts_exactly_the_controller_and_model_credential_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            parent.chmod(0o700)
            files = {
                name: parent / name
                for name in ("controller.env", "model.env", "other.env")
            }
            for path in files.values():
                path.write_text("MODEL_KEY=top-secret\n", encoding="utf-8")
                path.chmod(0o600)
            with (
                patch.object(operator, "CONTROLLER_ENV", files["controller.env"]),
                patch.object(operator, "MODEL_ENV", files["model.env"]),
            ):
                for name in ("controller.env", "model.env"):
                    with self.subTest(accepted=name):
                        result = operator.validate_credential_environment(
                            str(files[name]), ("MODEL_KEY",)
                        )
                        self.assertEqual(result["path"], str(files[name]))
                        self.assertEqual(result["present_names"], ["MODEL_KEY"])
                with self.assertRaisesRegex(
                    operator.OperatorRefused, "credential_environment_path_refused"
                ):
                    operator.validate_credential_environment(
                        str(files["other.env"]), ("MODEL_KEY",)
                    )
                # The model file carries the same ownership and mode checks.
                files["model.env"].chmod(0o644)
                with self.assertRaisesRegex(
                    operator.OperatorRefused, "credential_environment_file_mode_refused"
                ):
                    operator.validate_credential_environment(
                        str(files["model.env"]), ("MODEL_KEY",)
                    )


if __name__ == "__main__":
    unittest.main()
