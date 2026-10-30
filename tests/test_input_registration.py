import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from bio_neb.application.commands import (
    RegisterFastqInput,
)
from bio_neb.cli import main
from tests.proposal_information_fakes import runtime_configuration


class InputRegistrationTests(unittest.TestCase):
    def test_registration_preserves_canonical_read_role_forms(self):
        registered_path = str((Path.cwd() / "read.fastq").resolve())
        for role in ("read_1", "read_2", "forward_read", "reverse_read"):
            with self.subTest(role=role):
                self.assertEqual(
                    RegisterFastqInput(role=role, registered_path=registered_path).role,
                    role,
                )

    def test_public_registration_refuses_alias_collision_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prompt = root / "prompt.txt"
            runtime = root / "runtime.json"
            read1 = root / "r1.fastq"
            read2 = root / "r2.fastq"
            prompt.write_text("Analyze the registered pair.", encoding="utf-8")
            runtime.write_text(json.dumps(runtime_configuration()), encoding="utf-8")
            read1.write_bytes(b"reads-1")
            read2.write_bytes(b"reads-2")
            for index, roles in enumerate(
                (("read1", "read_1"), ("read2", "read_2"), ("read1", "read1"))
            ):
                with self.subTest(roles=roles):
                    run_root = root / f"must-not-exist-{index}"
                    errors = io.StringIO()
                    with redirect_stderr(errors):
                        exit_code = main(
                            [
                                "begin",
                                "--run-root",
                                str(run_root),
                                "--prompt-file",
                                str(prompt),
                                "--runtime-config",
                                str(runtime),
                                "--fastq",
                                f"{roles[0]}={read1}",
                                "--fastq",
                                f"{roles[1]}={read2}",
                            ]
                        )

                    self.assertEqual(exit_code, 2)
                    self.assertEqual(
                        json.loads(errors.getvalue())["code"], "command_invalid"
                    )
                    self.assertFalse(run_root.exists())
