from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from bio_neb.devtools.configured_secret_scan import main


class ConfiguredSecretScanPublicTests(unittest.TestCase):
    def test_public_cli_writes_zero_match_receipt_for_arbitrary_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence_root = root / "ticket-183-evidence"
            evidence_root.mkdir()
            (evidence_root / "artifact.json").write_text(
                '{"status":"safe"}', encoding="utf-8"
            )
            receipt = root / "receipts" / "configured-secret-scan.json"
            receipt.parent.mkdir()
            secret = "test-only-secret-that-must-not-be-retained"
            environment = dict(os.environ)
            environment["TEST_SCAN_TOKEN"] = secret

            completed = subprocess.run(
                (
                    sys.executable,
                    "-m",
                    "bio_neb.devtools.configured_secret_scan",
                    "--root",
                    str(evidence_root),
                    "--receipt",
                    str(receipt),
                    "--credential-name",
                    "TEST_SCAN_TOKEN",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = json.loads(receipt.read_text(encoding="utf-8"))
            material = {
                "schema_version": "bio-neb.configured-secret-scan/1",
                "scan_kind": "value_aware_contact_free",
                "scanned_root": str(evidence_root),
                "credential_names": ["TEST_SCAN_TOKEN"],
                "scanned_file_count": 1,
                "scanned_files": document["scanned_files"],
                "matching_credential_names": [],
                "matching_transform_names": [],
                "matching_value_count": 0,
                "zero_secret_result": True,
                "contacted_external_surface": False,
                "secret_values_retained": False,
            }
            self.assertEqual(
                document["scan_id"],
                "configured-secret-scan:"
                + hashlib.sha256(
                    json.dumps(
                        material,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(
                document["schema_version"], "bio-neb.configured-secret-scan/1"
            )
            self.assertEqual(document["credential_names"], ["TEST_SCAN_TOKEN"])
            self.assertEqual(document["scanned_file_count"], 1)
            self.assertEqual(document["matching_credential_names"], [])
            self.assertEqual(document["matching_transform_names"], [])
            self.assertEqual(document["matching_value_count"], 0)
            self.assertTrue(document["zero_secret_result"])
            self.assertFalse(document["contacted_external_surface"])
            self.assertFalse(document["secret_values_retained"])
            self.assertNotIn(secret, receipt.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(completed.stdout), document)

    def test_public_cli_refuses_raw_json_url_and_base64_secret_matches(self) -> None:
        secret = 'test-only / "encoded" secret'
        encoded_values = {
            "raw": secret,
            "json_escaped": json.dumps(secret, ensure_ascii=True)[1:-1],
            "url_encoded": urllib.parse.quote(secret, safe=""),
            "url_form_encoded": urllib.parse.quote_plus(secret, safe=""),
            "base64": base64.b64encode(secret.encode("utf-8")).decode("ascii"),
        }
        for expected_transform, encoded_value in encoded_values.items():
            with self.subTest(expected_transform=expected_transform):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    evidence_root = root / "evidence"
                    evidence_root.mkdir()
                    (evidence_root / "artifact.txt").write_text(
                        encoded_value, encoding="utf-8"
                    )
                    receipt = root / "configured-secret-scan.json"
                    environment = dict(os.environ)
                    environment["TEST_SCAN_TOKEN"] = secret

                    completed = subprocess.run(
                        (
                            sys.executable,
                            "-m",
                            "bio_neb.devtools.configured_secret_scan",
                            "--root",
                            str(evidence_root),
                            "--receipt",
                            str(receipt),
                            "--credential-name",
                            "TEST_SCAN_TOKEN",
                        ),
                        check=False,
                        capture_output=True,
                        text=True,
                        env=environment,
                    )

                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(
                        completed.stderr,
                        "configured_secret_scan_refused:configured secret value detected\n",
                    )
                    document = json.loads(receipt.read_text(encoding="utf-8"))
                    self.assertEqual(
                        document["matching_credential_names"], ["TEST_SCAN_TOKEN"]
                    )
                    self.assertIn(
                        expected_transform, document["matching_transform_names"]
                    )
                    self.assertEqual(document["matching_value_count"], 1)
                    self.assertFalse(document["zero_secret_result"])
                    self.assertNotIn(secret, receipt.read_text(encoding="utf-8"))

    def test_public_cli_refuses_configured_secret_representations_in_paths(
        self,
    ) -> None:
        secret = "test-only secret path"
        encoded_secret = urllib.parse.quote_plus(secret, safe="")
        for placement in ("root", "file"):
            with self.subTest(placement=placement):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    evidence_root = (
                        root / f"evidence-{secret}"
                        if placement == "root"
                        else root / "evidence"
                    )
                    evidence_root.mkdir()
                    artifact_name = (
                        f"artifact-{encoded_secret}.txt"
                        if placement == "file"
                        else "artifact.txt"
                    )
                    (evidence_root / artifact_name).write_text("safe", encoding="utf-8")
                    receipt = root / f"{placement}-receipt.json"
                    environment = dict(os.environ)
                    environment["TEST_SCAN_TOKEN"] = secret

                    completed = subprocess.run(
                        (
                            sys.executable,
                            "-m",
                            "bio_neb.devtools.configured_secret_scan",
                            "--root",
                            str(evidence_root),
                            "--receipt",
                            str(receipt),
                            "--credential-name",
                            "TEST_SCAN_TOKEN",
                        ),
                        check=False,
                        capture_output=True,
                        text=True,
                        env=environment,
                    )

                    self.assertEqual(completed.returncode, 2)
                    self.assertFalse(receipt.exists())
                    self.assertIn(
                        "configured secret value detected in evidence path",
                        completed.stderr,
                    )
                    self.assertNotIn(secret, completed.stderr)
                    self.assertNotIn(encoded_secret, completed.stderr)

    def test_public_cli_refuses_file_swapped_to_symlink_before_open(self) -> None:
        secret = "test-only-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence_root = root / "evidence"
            evidence_root.mkdir()
            artifact = evidence_root / "artifact.txt"
            artifact.write_text("safe", encoding="utf-8")
            outside = root / "outside.txt"
            outside.write_text(secret, encoding="utf-8")
            receipt = root / "receipt.json"
            original_is_file = Path.is_file

            def swap_after_file_check(path: Path) -> bool:
                result = original_is_file(path)
                if path == artifact and result and not path.is_symlink():
                    path.unlink()
                    path.symlink_to(outside)
                return result

            stderr = StringIO()
            with (
                patch.dict(os.environ, {"TEST_SCAN_TOKEN": secret}, clear=False),
                patch.object(
                    Path,
                    "is_file",
                    autospec=True,
                    side_effect=swap_after_file_check,
                ),
                redirect_stderr(stderr),
            ):
                result = main(
                    (
                        "--root",
                        str(evidence_root),
                        "--receipt",
                        str(receipt),
                        "--credential-name",
                        "TEST_SCAN_TOKEN",
                    )
                )

            self.assertEqual(result, 2)
            self.assertFalse(receipt.exists())
            self.assertIn("refuses symlinks or unsafe files", stderr.getvalue())

    def test_public_cli_refuses_unsafe_files_and_receipt_collisions(self) -> None:
        secret = "test-only-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            environment = {"TEST_SCAN_TOKEN": secret}

            def invoke(evidence_root: Path, receipt: Path) -> tuple[int, str]:
                stderr = StringIO()
                with (
                    patch.dict(os.environ, environment, clear=False),
                    redirect_stderr(stderr),
                ):
                    result = main(
                        (
                            "--root",
                            str(evidence_root),
                            "--receipt",
                            str(receipt),
                            "--credential-name",
                            "TEST_SCAN_TOKEN",
                        )
                    )
                return result, stderr.getvalue()

            symlink_root = root / "symlink"
            symlink_root.mkdir()
            target = root / "target.txt"
            target.write_text("safe", encoding="utf-8")
            (symlink_root / "linked.txt").symlink_to(target)
            result, stderr = invoke(symlink_root, root / "symlink-receipt.json")
            self.assertEqual(result, 2)
            self.assertIn("evidence-root scan refuses symlinks", stderr)

            auth_root = root / "auth"
            auth_root.mkdir()
            (auth_root / "artifact.txt").write_text(
                "Authorization: Bearer redacted", encoding="utf-8"
            )
            result, stderr = invoke(auth_root, root / "auth-receipt.json")
            self.assertEqual(result, 2)
            self.assertIn("authentication header", stderr)

            unstable_root = root / "unstable"
            unstable_root.mkdir()
            unstable_file = unstable_root / "artifact.txt"
            unstable_file.write_text("safe", encoding="utf-8")
            original_read = os.read
            changed = False

            def unstable_read(descriptor: int, byte_count: int) -> bytes:
                nonlocal changed
                content = original_read(descriptor, byte_count)
                if not changed:
                    unstable_file.write_bytes(b"safe-changed")
                    changed = True
                return content

            with patch.object(os, "read", side_effect=unstable_read):
                result, stderr = invoke(unstable_root, root / "unstable-receipt.json")
            self.assertEqual(result, 2)
            self.assertIn("evidence file changed while scanning", stderr)

            collision_root = root / "collision"
            collision_root.mkdir()
            (collision_root / "artifact.txt").write_text("safe", encoding="utf-8")
            collision_receipt = root / "existing-receipt.json"
            collision_receipt.write_text("retain", encoding="utf-8")
            result, stderr = invoke(collision_root, collision_receipt)
            self.assertEqual(result, 2)
            self.assertIn("unused exact absolute path", stderr)
            self.assertEqual(collision_receipt.read_text(encoding="utf-8"), "retain")


if __name__ == "__main__":
    unittest.main()
