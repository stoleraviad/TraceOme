import tempfile
import unittest
from pathlib import Path

from bio_neb.persistence import retain_immutable, write_once


class PersistenceTests(unittest.TestCase):
    def test_append_durably_preserves_prefix_and_refuses_short_write(self):
        from unittest.mock import MagicMock, patch

        from bio_neb.persistence import append_durable

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal"
            path.write_bytes(b"first\n")
            append_durable(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\nsecond\n")
            stream = MagicMock()
            stream.write.return_value = 2
            with (
                patch.object(Path, "open") as opened,
                patch("bio_neb.persistence.os.fsync"),
            ):
                opened.return_value.__enter__.return_value = stream
                with self.assertRaisesRegex(OSError, "short journal write"):
                    append_durable(path, b"third\n")
                stream.flush.assert_not_called()

    def test_immutable_content_reuse_checks_bytes_and_never_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts" / "identity.json"
            retain_immutable(path, b'{"verified":true}')
            retain_immutable(path, b'{"verified":true}')
            with self.assertRaisesRegex(ValueError, "immutable content changed"):
                retain_immutable(path, b'{"verified":false}')
            self.assertEqual(path.read_bytes(), b'{"verified":true}')

    def test_write_once_preserves_exact_bytes_and_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "record"
            content = b'\x00{"text":"caf\xc3\xa9"}\n'
            self.assertIsNone(write_once(target, content))
            with self.assertRaises(FileExistsError):
                write_once(target, b"replacement")
            self.assertEqual(target.read_bytes(), content)
            with self.assertRaises(FileNotFoundError):
                write_once(root / "absent" / "record", content)
            self.assertFalse((root / "absent").exists())
