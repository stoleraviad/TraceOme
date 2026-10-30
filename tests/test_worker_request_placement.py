import errno
import os
import stat
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bio_neb.parabricks.worker_request_placement as placement_module
from bio_neb.parabricks.worker_request import (
    RuntimeSmokeParameters,
    compile_worker_request,
)
from bio_neb.parabricks.worker_request_placement import (
    NativeWorkerRequestPlacement,
    PlacedWorkerRequest,
    WorkerRequestPlacementError,
)


class NativeWorkerRequestPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.placement = NativeWorkerRequestPlacement()
        self.compiled = compile_worker_request(
            RuntimeSmokeParameters(request_id="placement-smoke")
        )

    def assert_placement_failed(
        self, refused: WorkerRequestPlacementError, cause: str | int
    ) -> None:
        # Placement maps every local failure to one reason; the chained cause
        # (its message, or its errno for an OS refusal) names which one fired.
        self.assertEqual(refused.reason, "worker_request_placement_failed")
        actual = refused.__cause__
        if isinstance(cause, int):
            self.assertEqual(getattr(actual, "errno", None), cause)
        else:
            self.assertEqual(str(actual), cause)

    def test_new_request_is_durably_placed_with_private_parents_and_stable_0600_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            source = Path(root) / "one" / "two" / "request.json"
            real_fsync = os.fsync
            fsynced: list[int] = []

            def observed_fsync(descriptor: int) -> None:
                fsynced.append(descriptor)
                real_fsync(descriptor)

            with patch.object(placement_module.os, "fsync", observed_fsync):
                placed = self.placement.place(
                    self.compiled,
                    source_path=source.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )

            self.assertIs(placed.compiled, self.compiled)
            self.assertEqual(source.read_bytes(), self.compiled.request_bytes)
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(source.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(source.parent.parent.stat().st_mode), 0o700)
            self.assertEqual(len(fsynced), 4)

    def test_successful_placement_claims_exactly_one_source_container_request_route(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            source = Path(root) / "request.json"
            placed = self.placement.place(
                self.compiled,
                source_path=source.as_posix(),
                container_path="/opt/bio-neb/request.json",
            )
            self.assertEqual(source.read_bytes(), self.compiled.request_bytes)
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o600)
            self.assertEqual(
                placed.injection_provider_values,
                (f"{source.as_posix()}:/opt/bio-neb/request.json",),
            )
            self.assertEqual(
                placed.normalized_arguments, ("/opt/bio-neb/request.json",)
            )
            self.assertEqual(placed.injected_paths, placed.normalized_arguments)

    def test_same_byte_reuse_preserves_identity_metadata_and_performs_no_fsync(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            source = Path(root) / "request.json"
            source.write_bytes(self.compiled.request_bytes)
            source.chmod(0o600)
            before = source.stat()
            with patch.object(
                placement_module.os,
                "fsync",
                side_effect=AssertionError("reuse must not fsync"),
            ):
                placed = self.placement.place(
                    self.compiled,
                    source_path=source.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )
            after = source.stat()
            self.assertEqual(before.st_ino, after.st_ino)
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
            self.assertEqual(before.st_ctime_ns, after.st_ctime_ns)
            self.assertIs(placed.compiled, self.compiled)

    def test_refuses_wrong_mode_or_nonregular_existing_targets_without_repair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            base = Path(root)
            wrong_mode = base / "wide.json"
            wrong_mode.write_bytes(self.compiled.request_bytes)
            wrong_mode.chmod(0o640)
            directory = base / "directory.json"
            directory.mkdir()
            target = base / "target.json"
            target.write_bytes(self.compiled.request_bytes)
            target.chmod(0o600)
            symlink = base / "symlink.json"
            symlink.symlink_to(target)
            dangling = base / "dangling.json"
            dangling.symlink_to(base / "absent.json")
            fifo = base / "fifo.json"
            os.mkfifo(fifo, 0o600)

            drifted = "request descriptor and source path identity drifted"
            for candidate, cause in (
                (wrong_mode, "request file does not have effective mode 0600"),
                (directory, drifted),
                (symlink, errno.ELOOP),
                (dangling, errno.ELOOP),
                (fifo, drifted),
            ):
                with self.subTest(candidate=candidate.name):
                    before = candidate.lstat()
                    with self.assertRaises(WorkerRequestPlacementError) as raised:
                        self.placement.place(
                            self.compiled,
                            source_path=candidate.as_posix(),
                            container_path="/opt/bio-neb/request.json",
                        )
                    self.assert_placement_failed(raised.exception, cause)
                    after = candidate.lstat()
                    self.assertEqual(
                        (before.st_dev, before.st_ino), (after.st_dev, after.st_ino)
                    )
            self.assertEqual(stat.S_IMODE(wrong_mode.stat().st_mode), 0o640)

    def test_refuses_existing_byte_drift_with_bytes_and_metadata_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            changed = bytearray(self.compiled.request_bytes)
            changed[-1] ^= 1
            cases = {
                "different-size": (
                    b"different",
                    "request file size differs from compiled bytes",
                ),
                "same-length": (
                    bytes(changed),
                    "request file changed during stable readback",
                ),
            }
            for name, (altered, cause) in cases.items():
                with self.subTest(name=name):
                    source = Path(root) / f"{name}.json"
                    source.write_bytes(altered)
                    source.chmod(0o600)
                    before = source.stat()

                    with self.assertRaises(WorkerRequestPlacementError) as raised:
                        self.placement.place(
                            self.compiled,
                            source_path=source.as_posix(),
                            container_path="/opt/bio-neb/request.json",
                        )
                    self.assert_placement_failed(raised.exception, cause)

                    after = source.stat()
                    self.assertEqual(source.read_bytes(), altered)
                    self.assertEqual(
                        (
                            after.st_dev,
                            after.st_ino,
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                            stat.S_IMODE(after.st_mode),
                        ),
                        (
                            before.st_dev,
                            before.st_ino,
                            before.st_size,
                            before.st_mtime_ns,
                            before.st_ctime_ns,
                            stat.S_IMODE(before.st_mode),
                        ),
                    )

    def test_refuses_parent_file_and_symlink_collisions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            base = Path(root)
            real = base / "real"
            real.mkdir()
            link = base / "linked"
            link.symlink_to(real, target_is_directory=True)
            collision = base / "collision"
            collision.write_text("file")
            for source in (
                link / "request.json",
                collision / "request.json",
            ):
                with self.subTest(source=source):
                    with self.assertRaises(WorkerRequestPlacementError) as raised:
                        self.placement.place(
                            self.compiled,
                            source_path=source.as_posix(),
                            container_path="/opt/bio-neb/request.json",
                        )
                    self.assert_placement_failed(raised.exception, errno.ENOTDIR)

    def test_refuses_noncanonical_source_or_container_paths_before_mutation(
        self,
    ) -> None:
        invalid = (
            "relative/request.json",
            "/",
            "/tmp/./request.json",
            "/tmp/../request.json",
            "/tmp//request.json",
            "/tmp/request.json/",
            "/mnt/c:/request.json",
            "/tmp/request file.json",
            "/tmp\\request.json",
            "/tmp/$(touch-x).json",
            "/tmp/request;echo.json",
        )
        for value in invalid:
            with self.subTest(field="source_path", value=value):
                with self.assertRaisesRegex(
                    WorkerRequestPlacementError, "worker_request_path_invalid"
                ):
                    self.placement.place(
                        self.compiled,
                        source_path=value,
                        container_path="/opt/bio-neb/request.json",
                    )

        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            for index, value in enumerate(invalid):
                source = Path(root) / f"case-{index}" / "request.json"
                with (
                    self.subTest(field="container_path", value=value),
                    self.assertRaisesRegex(
                        WorkerRequestPlacementError, "worker_request_path_invalid"
                    ),
                ):
                    self.placement.place(
                        self.compiled,
                        source_path=source.as_posix(),
                        container_path=value,
                    )
                self.assertFalse(source.parent.exists())

    def test_refuses_forged_or_tampered_compiled_request_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            source = Path(root) / "new" / "request.json"
            with self.assertRaisesRegex(
                WorkerRequestPlacementError, "compiled_worker_request_required"
            ):
                self.placement.place(  # type: ignore[arg-type]
                    object(),
                    source_path=source.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )
            self.assertFalse(source.parent.exists())

            tampered = compile_worker_request(
                RuntimeSmokeParameters(request_id="tampered-smoke")
            )
            object.__setattr__(tampered, "request_bytes", b"{}")
            with self.assertRaises(WorkerRequestPlacementError) as raised:
                self.placement.place(
                    tampered,
                    source_path=source.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )
            self.assert_placement_failed(
                raised.exception, "worker_request_identity_invalid"
            )
            self.assertFalse(source.parent.exists())

    def test_partial_writes_continue_until_exact_request_is_placed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            partial = Path(root) / "partial.json"
            real_write = os.write

            def half_write(descriptor: int, body: memoryview) -> int:
                size = max(1, len(body) // 2)
                return real_write(descriptor, body[:size])

            with patch.object(placement_module.os, "write", half_write):
                self.placement.place(
                    self.compiled,
                    source_path=partial.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )
            self.assertEqual(partial.read_bytes(), self.compiled.request_bytes)

    def test_zero_progress_write_fails_before_claim_and_retains_empty_residue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            stalled = Path(root) / "stalled.json"
            placed = None
            with (
                patch.object(placement_module.os, "write", return_value=0),
                self.assertRaises(WorkerRequestPlacementError) as raised,
            ):
                placed = self.placement.place(
                    self.compiled,
                    source_path=stalled.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )
            self.assert_placement_failed(
                raised.exception, "request write made no progress"
            )
            self.assertIsNone(placed)
            self.assertTrue(stalled.exists())
            self.assertEqual(stalled.read_bytes(), b"")

    def test_refuses_stable_descriptor_or_path_metadata_drift(self) -> None:
        fields = {
            "st_dev": 1,
            "st_ino": 1,
            "st_size": 1,
            "st_mtime_ns": 1,
            "st_ctime_ns": 1,
            "st_mode": 0o040,
        }
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            source = Path(root) / "request.json"
            source.write_bytes(self.compiled.request_bytes)
            source.chmod(0o600)
            parent = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                observed = source.stat()
                for field, delta in fields.items():
                    values = {
                        name: getattr(observed, name)
                        for name in (
                            "st_dev",
                            "st_ino",
                            "st_size",
                            "st_mtime_ns",
                            "st_ctime_ns",
                            "st_mode",
                        )
                    }
                    values[field] += delta
                    changed = SimpleNamespace(**values)
                    with (
                        self.subTest(field=field),
                        patch.object(
                            placement_module.os,
                            "fstat",
                            side_effect=(observed, observed, changed),
                        ),
                        self.assertRaises(OSError),
                    ):
                        self.placement._place_at(
                            parent, source.name, self.compiled.request_bytes
                        )

                changed_path = SimpleNamespace(
                    **{
                        name: (
                            getattr(observed, name) + 1
                            if name == "st_ino"
                            else getattr(observed, name)
                        )
                        for name in (
                            "st_dev",
                            "st_ino",
                            "st_size",
                            "st_mtime_ns",
                            "st_ctime_ns",
                            "st_mode",
                        )
                    }
                )
                with (
                    patch.object(
                        placement_module,
                        "_path_stat",
                        side_effect=(observed, changed_path),
                    ),
                    self.assertRaises(OSError),
                ):
                    self.placement._place_at(
                        parent, source.name, self.compiled.request_bytes
                    )
            finally:
                os.close(parent)

    def test_fsync_and_readback_failures_preserve_created_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            base = Path(root)
            fsync_failed = base / "fsync-failed.json"
            with (
                patch.object(
                    placement_module.os, "fsync", side_effect=OSError("fsync")
                ),
                self.assertRaises(WorkerRequestPlacementError) as raised,
            ):
                self.placement.place(
                    self.compiled,
                    source_path=fsync_failed.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )
            self.assert_placement_failed(raised.exception, "fsync")
            self.assertTrue(fsync_failed.exists())

            readback_failed = base / "readback-failed.json"
            with (
                patch.object(placement_module.os, "read", side_effect=OSError("read")),
                self.assertRaises(WorkerRequestPlacementError) as raised,
            ):
                self.placement.place(
                    self.compiled,
                    source_path=readback_failed.as_posix(),
                    container_path="/opt/bio-neb/request.json",
                )
            self.assert_placement_failed(raised.exception, "read")
            self.assertTrue(readback_failed.exists())

    def test_fetched_provider_route_reconciles_exactly_one_request_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            placed = self.placement.place(
                self.compiled,
                source_path=(Path(root) / "request.json").as_posix(),
                container_path="/opt/bio-neb/request.json",
            )
            self.assertEqual(
                self.placement.reconcile_fetched_route(
                    placed,
                    normalized_arguments=("/opt/bio-neb/request.json",),
                    injected_paths=("/opt/bio-neb/request.json",),
                ),
                self.compiled.identity,
            )
            for arguments, injected in (
                ((), placed.injected_paths),
                ((*placed.normalized_arguments, "--extra"), placed.injected_paths),
                (("/opt/bio-neb/changed.json",), placed.injected_paths),
                (placed.normalized_arguments, ()),
                (
                    placed.normalized_arguments,
                    (*placed.injected_paths, placed.container_path),
                ),
                (placed.normalized_arguments, ("/opt/bio-neb/changed.json",)),
            ):
                with self.assertRaisesRegex(
                    WorkerRequestPlacementError, "provider_request_route_drift"
                ):
                    self.placement.reconcile_fetched_route(
                        placed,
                        normalized_arguments=arguments,
                        injected_paths=injected,
                    )

    def test_observed_worker_request_reconciles_exact_schema_size_and_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="bio-neb-placement-") as root:
            placed = self.placement.place(
                self.compiled,
                source_path=(Path(root) / "request.json").as_posix(),
                container_path="/opt/bio-neb/request.json",
            )
            identity = placed.request_identity
            self.assertEqual(
                self.placement.reconcile_worker_identity(
                    placed,
                    request_schema=identity.schema_version,
                    byte_count=identity.byte_count,
                    request_sha256=identity.sha256,
                ),
                identity,
            )
            for schema, byte_count, request_sha256 in (
                (
                    "unexpected-worker-request-schema",
                    identity.byte_count,
                    identity.sha256,
                ),
                (identity.schema_version, identity.byte_count + 1, identity.sha256),
                (identity.schema_version, identity.byte_count, "0" * 64),
            ):
                with self.assertRaisesRegex(
                    WorkerRequestPlacementError, "worker_request_identity_drift"
                ):
                    self.placement.reconcile_worker_identity(
                        placed,
                        request_schema=schema,
                        byte_count=byte_count,
                        request_sha256=request_sha256,
                    )

    def test_placed_value_cannot_be_forged(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(PlacedWorkerRequest)),
            ("compiled", "source_path", "container_path"),
        )
        with self.assertRaises(TypeError):
            PlacedWorkerRequest()


if __name__ == "__main__":
    unittest.main()
