import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from bio_neb.investigation.scientific_continuation import (
    ScientificContinuationPacket,
)
from bio_neb.investigation.vcf_observation import (
    AttestedPlainVcf,
    BcftoolsRuntimeBinding,
    CompletedProcess,
    DeterministicVcfObservationTool,
    LocalFileObservation,
    ObserveVariantInput,
    PreparedVcfObservationReceipt,
    VcfMaterializationBinding,
)
from tests.content_identity_vectors import assert_json_identity
from tests.terminal_journey_fakes import published_continuation


def _current_continuation() -> ScientificContinuationPacket:
    """Use a current synthetic execution packet without provider contact."""
    with tempfile.TemporaryDirectory() as directory:
        return ScientificContinuationPacket.model_validate_json(
            published_continuation(Path(directory).resolve()).read_bytes(), strict=True
        )


def setUpModule() -> None:
    # Build the execution packet at test setup, not import: a setup refusal then
    # stays a reported error and discovery still sees every method.
    global _VALIDATED_CONTINUATION, ASSEMBLY, SAMPLE_ID, ORIGINAL_SHA, ORIGINAL_SIZE
    _VALIDATED_CONTINUATION = _current_continuation()
    ASSEMBLY = _VALIDATED_CONTINUATION.reference.assembly_accession
    SAMPLE_ID = _VALIDATED_CONTINUATION.sample.sample_id
    ORIGINAL_SHA = _VALIDATED_CONTINUATION.vcf.sha256
    ORIGINAL_SIZE = _VALIDATED_CONTINUATION.vcf.byte_count


ORIGINAL = "/runs/t122/artifact/variants.vcf"
ACQUISITION_RECEIPT = "/runs/t122/ticket122-vcf-acquisition-receipt.json"
RUNTIME = "/runs/t121/runtime/bin/bcftools"
PROBE_RECEIPT = "/runs/t121/ticket121-expanded-vcf-probe-receipt.json"
DERIVATION = "/runs/t124/derived"
BGZF = f"{DERIVATION}/variants.vcf.gz"


def regions_path(text: str) -> str:
    """Where a query's regions file is written: beside the derived VCF, by its text's
    digest (#467)."""
    return f"{DERIVATION}/regions-{hashlib.sha256(text.encode()).hexdigest()[:16]}.tsv"


CSI = f"{BGZF}.csi"
RUNTIME_SHA = "3" * 64
BGZF_BYTES = b"bgzf derived bytes"
BGZF_SHA = hashlib.sha256(BGZF_BYTES).hexdigest()
CSI_SHA = "5" * 64
ACQUISITION_RECEIPT_SHA = "6" * 64
PROBE_RECEIPT_SHA = "7" * 64


def file_state(
    path,
    *,
    sha256,
    size=100,
    writable=False,
    executable=False,
    inode="10",
    modified=1000,
):
    return LocalFileObservation(
        path=path,
        exists=True,
        is_regular=True,
        is_symlink=False,
        is_writable=writable,
        is_executable=executable,
        byte_count=size,
        sha256=sha256,
        device="7",
        inode=inode,
        modified_at_ns=modified,
    )


def missing(path):
    return LocalFileObservation(
        path=path,
        exists=False,
        is_regular=False,
        is_symlink=False,
        is_writable=False,
        is_executable=False,
        byte_count=None,
        sha256=None,
        device=None,
        inode=None,
        modified_at_ns=None,
    )


def query_row(*, alts="G", genotype="1/1", pos=783006, ref="A", quality=".\t."):
    return (
        f"chr1\t{pos}\t{ref}\t{alts}\t{quality}\t{SAMPLE_ID}\t{genotype}\t.\t.\t.\n"
    ).encode()


def vcf_header(*contig_lines, samples=None):
    lines = [
        "##fileformat=VCFv4.2",
        *(
            contig_lines
            or (
                "##contig=<ID=chr1,length=248956422>",
                "##contig=<ID=chr2,length=242193529>",
            )
        ),
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples or (SAMPLE_ID,)),
    ]
    return ("\n".join(lines) + "\n").encode()


def continuation() -> ScientificContinuationPacket:
    return _VALIDATED_CONTINUATION.model_copy(deep=True)


def binding(**changes):
    material = {
        "attested_vcf": AttestedPlainVcf(
            local_path=ORIGINAL,
            byte_count=ORIGINAL_SIZE,
            sha256=ORIGINAL_SHA,
            acquisition_receipt_path=ACQUISITION_RECEIPT,
            acquisition_receipt_sha256=ACQUISITION_RECEIPT_SHA,
        ),
        "runtime": BcftoolsRuntimeBinding(
            executable_path=RUNTIME,
            executable_sha256=RUNTIME_SHA,
            bcftools_version="1.24",
            htslib_version="1.24",
            probe_receipt_path=PROBE_RECEIPT,
            probe_receipt_sha256=PROBE_RECEIPT_SHA,
        ),
        "derivation_root": DERIVATION,
        "continuation_sample_id": SAMPLE_ID,
        "expected_vcf_sample_names": (SAMPLE_ID,),
    }
    material.update(changes)
    return VcfMaterializationBinding(**material)


class FakeAdapter:
    """Script the pysam dispatcher's recorded behaviour (#107 comment 6186513990).

    `--version` and `query --list-samples` print nothing, and a named `--output`
    goes to stdout instead of the file.
    """

    def __init__(self):
        self.files = {
            ORIGINAL: file_state(ORIGINAL, sha256=ORIGINAL_SHA, size=ORIGINAL_SIZE),
            ACQUISITION_RECEIPT: file_state(
                ACQUISITION_RECEIPT,
                sha256=ACQUISITION_RECEIPT_SHA,
                size=500,
                inode="14",
            ),
            RUNTIME: file_state(
                RUNTIME, sha256=RUNTIME_SHA, executable=True, inode="11"
            ),
            PROBE_RECEIPT: file_state(
                PROBE_RECEIPT,
                sha256=PROBE_RECEIPT_SHA,
                size=700,
                inode="15",
            ),
        }
        self.directories = set()
        self.calls = []
        self.inspected = []
        self.query_result = CompletedProcess(0, b"", b"")
        self.fail_purpose = None
        self.header = vcf_header()

    def inspect_file(self, path):
        self.inspected.append(path)
        return copy.deepcopy(self.files.get(path, missing(path)))

    def path_exists(self, path):
        return path in self.directories or path in self.files

    def create_fresh_directory(self, path):
        if self.path_exists(path):
            raise FileExistsError(path)
        self.directories.add(path)

    def write_fresh_file(self, path, data):
        # A query's regions file: kept as written, never inspected.
        if self.path_exists(path):
            raise FileExistsError(path)
        self.files[path] = data

    def run(self, argv, *, timeout_seconds, stdout_path=None):
        self.calls.append((argv, timeout_seconds, stdout_path))
        if (
            "--output" in argv
            or argv[1:] in (("--version",),)
            or "--list-samples" in argv
        ):
            purpose = "dispatcher_quirk"
            payload = BGZF_BYTES if "--output" in argv else b""
        elif argv[1:3] == ("view", "--no-version") and "--header-only" in argv:
            purpose = "header_inventory"
            payload = self.header
        elif argv[1] == "view":
            purpose = "bgzf_derivation"
            payload = BGZF_BYTES
        elif argv[1:3] == ("index", "--csi"):
            purpose = "csi_derivation"
            self.files[CSI] = file_state(CSI, sha256=CSI_SHA, size=25, inode="13")
            payload = b""
        else:
            purpose = "query"
            if stdout_path is not None:
                raise AssertionError("query output is captured")
            return self.query_result
        if purpose == self.fail_purpose:
            return CompletedProcess(9, b"", b"bounded failure")
        if stdout_path is not None:
            if self.path_exists(stdout_path):
                raise FileExistsError(stdout_path)
            self.files[stdout_path] = file_state(
                stdout_path,
                sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
                inode="12",
            )
            payload = b""
        return CompletedProcess(0, payload, b"")


def prepared(adapter=None):
    selected = adapter or FakeAdapter()
    receipt = DeterministicVcfObservationTool().prepare(
        continuation(), binding(), selected
    )
    if not isinstance(receipt, PreparedVcfObservationReceipt):
        raise AssertionError(receipt)
    return selected, receipt


class LocalBcftoolsAdapterTests(unittest.TestCase):
    def test_a_regions_file_whose_write_fails_is_never_published(self):
        """A regions file is at its path only once complete (#467 Standards r1 ST1,
        ADR 0029: incomplete creation produces no usable result)."""
        from unittest import mock

        from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory).resolve() / "regions-0123456789abcdef.tsv")
            adapter = LocalBcftoolsAdapter()
            with (
                mock.patch("os.fsync", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                adapter.write_fresh_file(path, b"chr1\t100\t100\n")

            self.assertFalse(adapter.path_exists(path))
            self.assertEqual(os.listdir(directory), [])
            adapter.write_fresh_file(path, b"chr1\t100\t100\n")
            self.assertEqual(Path(path).read_bytes(), b"chr1\t100\t100\n")

    def test_mutable_admission_never_reuses_same_timestamp_content(self):
        from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "mutable.vcf.gz"
            adapter = LocalBcftoolsAdapter()
            for _ in range(1000):
                path.write_bytes(b"before")
                adapter.inspect_file(str(path))
                before = path.stat()
                path.write_bytes(b"after!")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                self.assertEqual(
                    adapter.inspect_file(str(path)).sha256,
                    "f778dacc09a326990d63568063853fdffb39d070ee6049054e23b1a666d1bf63",
                )

    def test_owned_file_changes_and_unowned_content_require_fresh_reads(self):
        from bio_neb.investigation.local_bcftools import LocalBcftoolsAdapter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "mutable.vcf.gz"
            path.write_bytes(b"before")
            adapter = LocalBcftoolsAdapter()
            reads = []
            fdopen = os.fdopen

            def measured_open(*args, **kwargs):
                stream = fdopen(*args, **kwargs)
                read = stream.read

                def measured_read(size=-1):
                    content = read(size)
                    reads.append(content)
                    return content

                stream.read = measured_read
                return stream

            with patch("os.fdopen", side_effect=measured_open):
                first = adapter.inspect_file(str(path))
                original = path.stat()
                path.write_bytes(b"after!")
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
                changed = adapter.inspect_file(str(path))
                self.assertNotEqual(changed.sha256, first.sha256)
                self.assertEqual(changed.sha256, hashlib.sha256(b"after!").hexdigest())
                self.assertEqual(changed.modified_at_ns, first.modified_at_ns)
                self.assertEqual(adapter.inspect_file(str(path)), changed)
                self.assertEqual(
                    LocalBcftoolsAdapter().inspect_file(str(path)), changed
                )
                path.unlink()
                self.assertFalse(adapter.inspect_file(str(path)).exists)
                path.write_bytes(b"new!!!")
                self.assertEqual(
                    adapter.inspect_file(str(path)).sha256,
                    hashlib.sha256(b"new!!!").hexdigest(),
                )
                path.chmod(0o500)
                metadata = adapter.inspect_file(str(path))
                self.assertFalse(metadata.is_writable)
                self.assertTrue(metadata.is_executable)
                path.unlink()
                path.symlink_to(Path(directory) / "elsewhere")
                with self.assertRaisesRegex(ValueError, "non-symlink"):
                    adapter.inspect_file(str(path))
            self.assertEqual(b"".join(reads), b"beforeafter!after!after!new!!!new!!!")

    def test_owned_files_reuse_full_reads_across_operations(self):
        from bio_neb.application.local_file_observation import (
            ObservationFiles,
        )
        from bio_neb.investigation.local_bcftools import (
            LocalBcftoolsAdapter,
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = [
                Path(directory) / name
                for name in (
                    "variants.vcf",
                    "variants.vcf.gz",
                    "variants.vcf.gz.csi",
                    "bcftools",
                )
            ]
            for path in paths:
                path.write_bytes(b"owned input bytes")
            files = ObservationFiles()
            self.addCleanup(files.close)
            adapter = LocalBcftoolsAdapter(files=files)
            read_bytes = []
            original_fdopen = os.fdopen

            def measured_open(*args, **kwargs):
                stream = original_fdopen(*args, **kwargs)
                original_read = stream.read

                def read(size=-1):
                    content = original_read(size)
                    read_bytes.append(len(content))
                    return content

                stream.read = read
                return stream

            with patch("os.fdopen", side_effect=measured_open):
                first = [adapter.inspect_file(str(path)) for path in paths]
                for _ in range(3):
                    self.assertEqual(
                        [adapter.inspect_file(str(path)) for path in paths], first
                    )
            self.assertEqual(sum(read_bytes), 4 * len(b"owned input bytes"))

    def test_file_observation_hashes_one_no_follow_descriptor(self):
        from bio_neb.investigation.local_bcftools import (
            LocalBcftoolsAdapter,
        )

        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory).resolve() / "variants.vcf"
            content = b"##fileformat=VCFv4.2\n"
            selected.write_bytes(content)
            with patch(
                "bio_neb.investigation.local_bcftools.os.open",
                wraps=os.open,
            ) as opened:
                observed = LocalBcftoolsAdapter().inspect_file(str(selected))
            self.assertEqual(observed.byte_count, len(content))
            self.assertEqual(observed.sha256, hashlib.sha256(content).hexdigest())
            flags = opened.call_args.args[1]
            if hasattr(os, "O_NOFOLLOW"):
                self.assertTrue(flags & os.O_NOFOLLOW)

    def test_file_observation_refuses_identity_change_during_full_read(self):
        from bio_neb.investigation.local_bcftools import (
            LocalBcftoolsAdapter,
        )

        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory).resolve() / "variants.vcf"
            selected.write_bytes(b"##fileformat=VCFv4.2\n")
            stable = os.stat(selected, follow_symlinks=False)
            changed = SimpleNamespace(
                st_dev=stable.st_dev,
                st_ino=stable.st_ino,
                st_mode=stable.st_mode,
                st_size=stable.st_size + 1,
                st_mtime_ns=stable.st_mtime_ns + 1,
                st_ctime_ns=stable.st_ctime_ns,
            )
            with (
                patch(
                    "bio_neb.investigation.local_bcftools.os.fstat",
                    side_effect=(stable, changed),
                ),
                self.assertRaisesRegex(ValueError, "changed while hashing"),
            ):
                LocalBcftoolsAdapter().inspect_file(str(selected))

    def test_file_observation_refuses_path_replacement_after_full_read(self):
        from bio_neb.investigation.local_bcftools import (
            LocalBcftoolsAdapter,
        )

        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory).resolve() / "variants.vcf"
            selected.write_bytes(b"##fileformat=VCFv4.2\n")
            stable = os.stat(selected, follow_symlinks=False)
            replaced = SimpleNamespace(
                st_dev=stable.st_dev,
                st_ino=stable.st_ino + 1,
                st_mode=stable.st_mode,
                st_size=stable.st_size,
                st_mtime_ns=stable.st_mtime_ns,
                st_ctime_ns=stable.st_ctime_ns,
            )
            with (
                patch(
                    "bio_neb.investigation.local_bcftools.os.lstat",
                    side_effect=(stable, replaced),
                ),
                self.assertRaisesRegex(ValueError, "path changed while hashing"),
            ):
                LocalBcftoolsAdapter().inspect_file(str(selected))


class VcfPreparationTests(unittest.TestCase):
    def test_preparation_binds_investigation_authority_and_compiles_only_fixed_shell_free_argv(
        self,
    ) -> None:
        selected = FakeAdapter()
        authority_id = "investigation-authority:" + "8" * 64
        authority = SimpleNamespace(
            continuation_id=authority_id,
            journey_id="11111111-1111-4111-8111-111111111111",
            reference=SimpleNamespace(assembly_accession=ASSEMBLY),
            sample=SimpleNamespace(sample_id=SAMPLE_ID),
            vcf=continuation().vcf,
        )

        result = DeterministicVcfObservationTool().prepare(
            authority,
            binding(),
            selected,
            authority_kind="investigation_authority",
        )

        self.assertIsInstance(result, PreparedVcfObservationReceipt)
        assert isinstance(result, PreparedVcfObservationReceipt)
        self.assertEqual(result.authority_kind, "investigation_authority")
        self.assertEqual(result.continuation_id, authority_id)
        self.assertEqual(
            [call[0] for call in selected.calls],
            [
                (RUNTIME, "view", "--no-version", "--output-type", "z", ORIGINAL),
                (RUNTIME, "index", "--csi", BGZF),
                (RUNTIME, "view", "--no-version", "--header-only", BGZF),
            ],
        )

    def test_preparation_with_pysam_dispatcher_behaviour_reads_the_header_once_without_rehash(
        self,
    ):
        adapter = FakeAdapter()
        self.assertIsInstance(continuation(), ScientificContinuationPacket)

        result = DeterministicVcfObservationTool().prepare(
            continuation(), binding(), adapter
        )

        self.assertIsInstance(result, PreparedVcfObservationReceipt)
        assert isinstance(result, PreparedVcfObservationReceipt)
        assert_json_identity(
            self,
            result.receipt_id,
            result,
            "status schema_version authority_kind continuation_id journey_id assembly "
            "continuation_sample_id vcf_descriptor acquisition_receipt_sha256 acquisition_receipt_file "
            "runtime probe_receipt_file derived_bgzf derived_csi vcf_sample_names vcf_contigs invocations",
            "vcf-observation-preparation:",
        )
        self.assertEqual(result.schema_version, "bio-neb.vcf-observation-preparation/2")
        # The dispatcher's quirks are never relied on: no --version, no
        # --list-samples and no named --output; the derived BGZF is the stdout of
        # one view, written by the adapter to a fresh path.
        self.assertEqual(
            [(call[0], call[2]) for call in adapter.calls],
            [
                (
                    (RUNTIME, "view", "--no-version", "--output-type", "z", ORIGINAL),
                    BGZF,
                ),
                ((RUNTIME, "index", "--csi", BGZF), None),
                ((RUNTIME, "view", "--no-version", "--header-only", BGZF), None),
            ],
        )
        self.assertEqual(
            tuple(item.purpose for item in result.invocations),
            ("bgzf_derivation", "csi_derivation", "header_inventory"),
        )
        bgzf_invocation = result.invocations[0]
        self.assertEqual(
            (bgzf_invocation.stdout_byte_count, bgzf_invocation.stdout_sha256),
            (len(BGZF_BYTES), BGZF_SHA),
        )
        self.assertEqual(result.derived_bgzf.sha256, BGZF_SHA)
        self.assertEqual(result.vcf_sample_names, (SAMPLE_ID,))
        self.assertEqual(
            [(item.name, item.length) for item in result.vcf_contigs],
            [("chr1", 248956422), ("chr2", 242193529)],
        )
        self.assertEqual(result.skipped_contigs, ())
        self.assertIsNone(result.original)
        # Trusted reuse (0029): the attested VCF and the executable are never
        # inspected; each derived file is observed once, at creation.
        self.assertNotIn(ORIGINAL, adapter.inspected)
        self.assertNotIn(RUNTIME, adapter.inspected)
        self.assertEqual(adapter.inspected.count(BGZF), 1)
        self.assertEqual(
            result.acquisition_receipt_file.sha256,
            ACQUISITION_RECEIPT_SHA,
        )
        self.assertEqual(result.probe_receipt_file.sha256, PROBE_RECEIPT_SHA)
        self.assertEqual(result.vcf_descriptor, continuation().vcf)
        replayed = PreparedVcfObservationReceipt.model_validate(
            result.model_dump(mode="json")
        )
        self.assertEqual(replayed, result)
        repeated = DeterministicVcfObservationTool().prepare(
            continuation(), binding(), FakeAdapter()
        )
        self.assertEqual(repeated, result)
        for field, value in (
            (("derived_bgzf", "sha256"), "f" * 64),
            (("invocations", 0, "stdout_sha256"), "e" * 64),
            (("invocations", 2, "argv", 3), "--list-samples"),
        ):
            hostile = result.model_dump(mode="json")
            target = hostile
            for key in field[:-1]:
                target = target[key]
            target[field[-1]] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                PreparedVcfObservationReceipt.model_validate(hostile)
        with self.assertRaises(ValidationError):
            result.receipt_id = "vcf-observation-preparation:" + "0" * 64

    def test_preparation_refuses_every_authority_and_derivation_failure(self):
        cases = {}

        acquisition_receipt = FakeAdapter()
        acquisition_receipt.files.pop(ACQUISITION_RECEIPT)
        cases["acquisition receipt missing"] = (
            acquisition_receipt,
            binding(),
            "binding_invalid",
        )

        probe_receipt = FakeAdapter()
        probe_receipt.files[PROBE_RECEIPT] = file_state(
            PROBE_RECEIPT, sha256="b" * 64, size=700, inode="15"
        )
        cases["probe receipt drift"] = (
            probe_receipt,
            binding(),
            "runtime_identity_mismatch",
        )

        existing = FakeAdapter()
        existing.directories.add(DERIVATION)
        cases["existing root"] = (
            existing,
            binding(),
            "derivation_target_exists",
        )

        for purpose in ("bgzf_derivation", "csi_derivation", "header_inventory"):
            failed = FakeAdapter()
            failed.fail_purpose = purpose
            cases[f"{purpose} failure"] = (failed, binding(), "tool_failure")

        wrong_sample = FakeAdapter()
        wrong_sample.header = vcf_header(samples=("unbound-sample",))
        cases["sample mismatch"] = (
            wrong_sample,
            binding(),
            "derived_output_invalid",
        )

        no_usable_contig = FakeAdapter()
        no_usable_contig.header = vcf_header("##contig=<ID=chr1>")
        cases["no usable contig"] = (
            no_usable_contig,
            binding(),
            "derived_output_invalid",
        )

        for name, (adapter, selected_binding, code) in cases.items():
            with self.subTest(name=name):
                result = DeterministicVcfObservationTool().prepare(
                    continuation(), selected_binding, adapter
                )
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.code, code)

    def test_an_explicit_full_read_records_the_attested_vcf_once(self):
        # O2: the attested VCF is read only when an independent full read is
        # selected; a different or missing original then refuses preparation.
        adapter = FakeAdapter()
        result = DeterministicVcfObservationTool().prepare(
            continuation(), binding(), adapter, read_original=True
        )
        self.assertIsInstance(result, PreparedVcfObservationReceipt)
        assert isinstance(result, PreparedVcfObservationReceipt)
        self.assertEqual(result.original, adapter.files[ORIGINAL])
        self.assertEqual(adapter.inspected.count(ORIGINAL), 1)
        self.assertEqual(
            PreparedVcfObservationReceipt.model_validate(
                result.model_dump(mode="json")
            ),
            result,
        )

        mismatched = FakeAdapter()
        mismatched.files[ORIGINAL] = file_state(
            ORIGINAL, sha256="8" * 64, size=ORIGINAL_SIZE
        )
        unavailable = FakeAdapter()
        unavailable.files.pop(ORIGINAL)
        for adapter, code in (
            (mismatched, "original_identity_mismatch"),
            (unavailable, "input_unavailable"),
        ):
            with self.subTest(code=code):
                refused = DeterministicVcfObservationTool().prepare(
                    continuation(), binding(), adapter, read_original=True
                )
                self.assertEqual((refused.status, refused.code), ("failed", code))
                self.assertEqual(adapter.calls, [])

    def test_lookup_limits_are_named_defaults_that_a_caller_may_change(self):
        # The limits are default variables, not hard-wired values.
        from bio_neb.investigation.observation_defaults import VcfLookupDefaults

        defaults = VcfLookupDefaults()
        self.assertEqual(
            (
                defaults.preparation_timeout_seconds,
                defaults.query_timeout_seconds,
                defaults.query_output_byte_limit,
            ),
            (3_600, 60, 1_048_576),
        )
        chosen = VcfLookupDefaults(
            preparation_timeout_seconds=7,
            query_timeout_seconds=3,
            query_output_byte_limit=10,
        )
        adapter = FakeAdapter()
        tool = DeterministicVcfObservationTool(chosen)
        receipt = tool.prepare(continuation(), binding(), adapter)
        adapter.query_result = CompletedProcess(
            0, query_row(pos=783006, ref="A", alts="G"), b""
        )
        result = tool.observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 783006,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )

        self.assertEqual([call[1] for call in adapter.calls], [7, 7, 7, 3])
        self.assertEqual(
            (result.outcome, result.violations), ("refused", ("output_too_large",))
        )

    def test_binding_is_strict_and_never_accepts_command_authority(self):
        raw = binding().model_dump(mode="json")
        raw["argv"] = ["sh", "-c", "arbitrary"]
        with self.assertRaises(ValidationError):
            VcfMaterializationBinding.model_validate(raw)
        with self.assertRaises(ValidationError):
            VcfMaterializationBinding.model_validate(
                {**binding().model_dump(mode="json"), "derivation_root": "relative"}
            )


class RealVcfShapeTests(unittest.TestCase):
    """The VCF shapes real callers write (#339 step 3 cases, #340 rows 64 and 66)."""

    def observe(
        self, rows, *, alt="C", ref="T", contig="chr1", header=None, position=100
    ):
        adapter = FakeAdapter()
        if header is not None:
            adapter.header = header
        adapter, receipt = prepared(adapter)
        self.adapter = adapter
        adapter.query_result = CompletedProcess(0, b"".join(rows), b"")
        calls = len(adapter.calls)
        result = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": contig,
                "one_based_position": position,
                "ref": ref,
                "alt": alt,
            },
            continuation(),
            receipt,
            adapter,
        )
        self.assertEqual(
            type(result).model_validate(result.model_dump(mode="json")), result
        )
        return result, receipt, adapter.calls[calls:]

    def test_a_header_contig_without_calls_reads_no_record(self):
        # I1: the inventory comes from the header, not from contigs with records.
        result, _, calls = self.observe(
            [],
            contig="chrY",
            header=vcf_header(
                "##contig=<ID=chr1,length=248956422>",
                "##contig=<ID=chrY,length=57227415>",
            ),
        )

        self.assertEqual(result.outcome, "no_record")
        self.assertEqual(len(calls), 1)

    def test_an_unusable_contig_row_is_skipped_and_recorded_while_other_targets_run(
        self,
    ):
        # A row without length no longer fails the whole preparation. An HLA name
        # is usable: regions reach bcftools as a file, so a name with `*` and `:`
        # is one field (ADR 0049).
        header = vcf_header(
            "##contig=<ID=chr1,length=248956422>",
            "##contig=<ID=HLA-A*01:01:01:01,length=3503>",
            "##contig=<ID=chrUn_nolength>",
        )
        observed, receipt, _ = self.observe(
            [query_row(pos=100, ref="T", alts="C", genotype="0/1")], header=header
        )
        skipped, _, calls = self.observe([], contig="chrUn_nolength", header=header)
        hla, _, _ = self.observe(
            [], contig="HLA-A*01:01:01:01", position=10, header=header
        )

        self.assertEqual(
            [(item.contig_id, item.reason) for item in receipt.skipped_contigs],
            [("chrUn_nolength", "missing_length")],
        )
        self.assertEqual(
            [item.name for item in receipt.vcf_contigs], ["chr1", "HLA-A*01:01:01:01"]
        )
        self.assertEqual(hla.outcome, "no_record")
        self.assertEqual(
            hla.query_invocation.regions_file.text, "HLA-A*01:01:01:01\t10\t10\n"
        )
        self.assertEqual(observed.outcome, "exact_allele_observed")
        self.assertEqual(
            (skipped.code, skipped.violations),
            ("target_unresolved", ("contig_unusable",)),
        )
        self.assertEqual(calls, [])

    def test_an_upstream_record_spanning_the_target_is_a_typed_overlap(self):
        # E1: a deletion at 98 spans 100; today's parser refused it as output_invalid.
        deletion = query_row(pos=98, ref="CAT", alts="C", genotype="0/1")

        result, _, _ = self.observe([deletion])

        self.assertEqual(result.outcome, "locus_overlapped")
        self.assertEqual(
            [(item.one_based_position, item.ref, item.alts) for item in result.records],
            [(98, "CAT", ("C",))],
        )

    def test_an_upstream_record_that_does_not_reach_the_target_is_refused(self):
        # Only a REF that spans the target makes an overlap.
        disjoint, _, _ = self.observe(
            [query_row(pos=1, ref="A", alts="G", genotype="0/1")]
        )
        self.assertEqual(
            (disjoint.code, disjoint.violations), ("tool_failure", ("output_invalid",))
        )
        spanning, _, _ = self.observe(
            [query_row(pos=98, ref="CAT", alts="C", genotype="0/1")]
        )
        self.assertEqual(spanning.outcome, "locus_overlapped")
        retained = spanning.model_dump(mode="json")
        retained["records"][0].update(one_based_position=97, ref="CA", alts=["C"])
        with self.assertRaisesRegex(ValueError, "changed requested locus"):
            type(spanning).model_validate(retained)

    def test_a_symbolic_upstream_deletion_overlaps_and_never_matches(self):
        # A <DEL> states its span in INFO/END, not in its REF.
        alone, _, _ = self.observe(
            [query_row(pos=90, ref="A", alts="<DEL>", genotype="0/1")]
        )
        beside, _, _ = self.observe(
            [
                query_row(pos=90, ref="A", alts="<DEL>", genotype="0/1"),
                query_row(pos=100, ref="T", alts="C", genotype="0/1"),
            ]
        )

        self.assertEqual(alone.outcome, "locus_overlapped")
        self.assertEqual(alone.records[0].alts, ("<DEL>",))
        self.assertEqual(beside.outcome, "exact_allele_observed")
        self.assertEqual([item.alts for item in beside.other_records], [("<DEL>",)])

    def test_an_exact_record_beside_an_upstream_overlap_still_matches(self):
        # E2: the exact record answers; the overlapping deletion stays beside it.
        result, _, _ = self.observe(
            [
                query_row(pos=98, ref="CAT", alts="C", genotype="0/1"),
                query_row(pos=100, ref="T", alts="C", genotype="0/1"),
            ]
        )

        self.assertEqual(result.outcome, "exact_allele_observed")
        self.assertEqual(result.record.one_based_position, 100)
        self.assertEqual(
            [item.one_based_position for item in result.other_records], [98]
        )

    def test_multiallelic_padded_records_are_split_and_trimmed_before_matching(self):
        # F1: TG>T,CG with GT 1/2 carries T>C as its second ALT.
        result, _, _ = self.observe(
            [query_row(pos=100, ref="TG", alts="T,CG", genotype="1/2")]
        )
        upstream, _, _ = self.observe(
            [query_row(pos=99, ref="AT", alts="AC", genotype="0/1")]
        )

        self.assertEqual(result.outcome, "exact_allele_observed")
        self.assertEqual(result.genotype.target_alt_index, 2)
        self.assertEqual(result.record.alts, ("T", "CG"))
        self.assertEqual(upstream.outcome, "exact_allele_observed")
        self.assertEqual(upstream.record.one_based_position, 99)

    def test_filter_qual_gq_dp_and_ad_are_recorded_without_a_gate(self):
        # G1: a LowQual call is still the exact observation, with its quality kept.
        row = (f"chr1\t100\tT\tC\tLowQual\t3\t{SAMPLE_ID}\t0/1\t15\t5\t2,3\n").encode()

        result, _, _ = self.observe([row])

        self.assertEqual(result.outcome, "exact_allele_observed")
        self.assertEqual(
            (result.record.filters, result.record.qual), (("LowQual",), "3")
        )
        call = result.record.genotypes[0]
        self.assertEqual((call.gq, call.dp, call.ad), (15, 5, (2, 3)))
        missing, _, _ = self.observe([query_row(pos=100, ref="T", alts="C")])
        call = missing.record.genotypes[0]
        self.assertEqual(
            (missing.record.filters, missing.record.qual, call.gq, call.dp, call.ad),
            ((), None, None, None, None),
        )

    def test_a_fully_called_sibling_alt_reads_target_not_carried(self):
        # The call at T is 0/1 for C, so a T>A target is not carried.
        result, _, _ = self.observe(
            [query_row(pos=100, ref="T", alts="C", genotype="0/1")], alt="A"
        )
        half_called, _, _ = self.observe(
            [query_row(pos=100, ref="T", alts="C", genotype="./1")], alt="A"
        )
        other_reference, _, _ = self.observe(
            [query_row(pos=100, ref="G", alts="C", genotype="0/1")], alt="A"
        )

        self.assertEqual(result.outcome, "target_allele_not_carried")
        self.assertIsNone(result.genotype.target_alt_index)
        self.assertEqual(result.genotype.allele_indexes, (0, 1))
        self.assertEqual(half_called.outcome, "locus_present_identity_mismatch")
        self.assertEqual(other_reference.outcome, "locus_present_identity_mismatch")

    def test_an_alt_of_unknown_sequence_leaves_absence_unestablished(self):
        # An N-bearing or symbolic sibling could be the target's own ALT, so
        # the fully called record cannot say it is not carried.
        for alts in ("C,N", "C,<INS>"):
            with self.subTest(alts=alts):
                result, _, _ = self.observe(
                    [query_row(pos=100, ref="T", alts=alts, genotype="1/2")], alt="A"
                )
                self.assertEqual(result.outcome, "locus_present_identity_mismatch")

    def test_a_carried_alt_of_unknown_sequence_leaves_the_listed_target_unresolved(
        self,
    ):
        # The record lists the target ALT, but the call carries a sibling of
        # unknown sequence that could be the target's own allele.
        for row, ref, alt in (
            (query_row(pos=99, ref="A", alts="AC,<INS>", genotype="0/2"), "", "C"),
            (query_row(pos=100, ref="T", alts="C,N", genotype="0/2"), "T", "C"),
        ):
            with self.subTest(alts=row):
                result, _, _ = self.observe([row], ref=ref, alt=alt)
                self.assertEqual(result.outcome, "target_genotype_unresolved")
                self.assertEqual(result.genotype.state, "target_alt_not_carried")
                retained = result.model_dump(mode="json")
                del retained["outcome"], retained["claim_scope"]
                from bio_neb.investigation.vcf_observation import (
                    TargetAlleleNotCarried,
                )

                with self.assertRaisesRegex(ValueError, "not-carried result"):
                    TargetAlleleNotCarried.model_validate(retained)
        known, _, _ = self.observe(
            [query_row(pos=100, ref="T", alts="C,G", genotype="0/2")], alt="C"
        )
        self.assertEqual(known.outcome, "target_allele_not_carried")
        retained = known.model_dump(mode="json")
        del retained["outcome"], retained["claim_scope"]
        from bio_neb.investigation.vcf_observation import TargetGenotypeUnresolved

        with self.assertRaisesRegex(ValueError, "unresolved result"):
            TargetGenotypeUnresolved.model_validate(retained)

    def test_a_spanning_deletion_star_allele_never_matches_but_its_call_answers(self):
        # A retained record (rs74621188): T>*,G 1/2 under an upstream 11-base
        # deletion.
        rows = [
            query_row(pos=95, ref="ACGTATGCATGC", alts="A", genotype="0/1"),
            query_row(pos=100, ref="T", alts="*,G", genotype="1/2"),
        ]

        carried, _, _ = self.observe(rows, alt="G")
        sibling, _, _ = self.observe(rows, alt="A")

        self.assertEqual(carried.outcome, "exact_allele_observed")
        self.assertEqual(carried.genotype.target_alt_index, 2)
        self.assertEqual(
            [item.one_based_position for item in carried.other_records], [95]
        )
        self.assertEqual(sibling.outcome, "target_allele_not_carried")
        self.assertIsNone(sibling.genotype.target_alt_index)

    def test_an_anchored_left_aligned_indel_answers_its_trimmed_target(self):
        # The retained live VCF's shapes: a Work item holds the trimmed allele;
        # the VCF record carries its anchor base in REF.
        def region(calls):
            # The one query's regions file, by the path its argv names.
            [(argv, _, _)] = calls
            contig, start, end = self.adapter.files[argv[4]].decode().split()
            return f"{contig}:{start}-{end}"

        # rs756427959: target 14398 TGT>- ; the VCF writes 14397 CTGT>C 0/1.
        deletion, _, calls = self.observe(
            [query_row(pos=99, ref="CTGT", alts="C", genotype="0/1")],
            ref="TGT",
            alt="",
        )
        self.assertEqual(deletion.outcome, "exact_allele_observed")
        self.assertEqual(
            (deletion.record.one_based_position, deletion.record.ref),
            (99, "CTGT"),
        )
        self.assertEqual(deletion.genotype.target_alt_index, 1)
        self.assertEqual(region(calls), "chr1:100-102")

        # rs878871521: targets 303 ->C, ->CC and ->CCC; the VCF writes MT
        # 302 A>AC,ACC 1/2, so the third is fully called without it.
        insertion = [query_row(pos=99, ref="A", alts="AC,ACC", genotype="1/2")]
        for inserted, outcome, index in (
            ("C", "exact_allele_observed", 1),
            ("CC", "exact_allele_observed", 2),
            ("CCC", "target_allele_not_carried", None),
        ):
            with self.subTest(inserted=inserted):
                result, _, calls = self.observe(insertion, ref="", alt=inserted)
                self.assertEqual(result.outcome, outcome)
                self.assertEqual(result.genotype.target_alt_index, index)
                self.assertEqual(region(calls), "chr1:99-99")

        # rs754204844: target 175160810 C>- ; the VCF writes 175160809 TC>T 0/1
        # beside the spanning-deletion record 175160810 C>*,T 1/2.
        beside_star, _, _ = self.observe(
            [
                query_row(pos=99, ref="TC", alts="T", genotype="0/1"),
                query_row(pos=100, ref="C", alts="*,T", genotype="1/2"),
            ],
            ref="C",
            alt="",
        )
        self.assertEqual(beside_star.outcome, "exact_allele_observed")
        self.assertEqual(beside_star.record.one_based_position, 99)
        self.assertEqual(
            [item.alts for item in beside_star.other_records], [("*", "T")]
        )

    def test_the_lookup_input_admits_every_allele_a_target_admits(self):
        # The VCF input takes VariantTarget's alleles (N, and one empty side); an
        # N-bearing target is looked up and never matches.
        for ref, alt in (("TGT", ""), ("", "C"), ("N", "A"), ("TC", "GA")):
            with self.subTest(ref=ref, alt=alt):
                target = ObserveVariantInput(
                    assembly=ASSEMBLY,
                    contig="chr1",
                    one_based_position=100,
                    ref=ref,
                    alt=alt,
                )
                self.assertEqual((target.ref, target.alt), (ref, alt))
        for ref, alt, field, reason in (
            ("", "", (), "REF and ALT must differ"),
            ("T", "T", (), "REF and ALT must differ"),
            ("t", "C", ("ref",), "allele must contain uppercase A/C/G/T/N only"),
            ("T", "*", ("alt",), "allele must contain uppercase A/C/G/T/N only"),
            ("T", "<DEL>", ("alt",), "allele must contain uppercase A/C/G/T/N only"),
        ):
            with self.subTest(ref=ref, alt=alt):
                with self.assertRaises(ValidationError) as refused:
                    ObserveVariantInput(
                        assembly=ASSEMBLY,
                        contig="chr1",
                        one_based_position=100,
                        ref=ref,
                        alt=alt,
                    )
                [error] = refused.exception.errors()
                self.assertEqual(error["loc"], field)
                self.assertIn(reason, error["msg"])
        unknown_base, _, _ = self.observe(
            [query_row(pos=100, ref="N", alts="A", genotype="0/1")], ref="N", alt="A"
        )
        self.assertEqual(unknown_base.outcome, "locus_present_identity_mismatch")

    def test_a_record_inside_a_deletion_is_at_the_locus_not_upstream(self):
        # The query spans a deletion's REF, so a record that starts inside it
        # (chr1:101 C>T for the target chr1:100 TC>-) is at the locus: an identity
        # mismatch, never an upstream overlap.
        inside, _, calls = self.observe(
            [query_row(pos=101, ref="C", alts="T", genotype="0/1")], ref="TC", alt=""
        )
        self.assertEqual(inside.outcome, "locus_present_identity_mismatch")
        self.assertEqual(inside.query_invocation.regions_file.text, "chr1\t100\t101\n")
        upstream, _, _ = self.observe(
            [query_row(pos=98, ref="CATCG", alts="C", genotype="0/1")],
            ref="TC",
            alt="",
        )
        self.assertEqual(upstream.outcome, "locus_overlapped")
        retained = inside.model_dump(mode="json")
        retained["outcome"] = "locus_overlapped"
        retained["claim_scope"] = upstream.claim_scope
        with self.assertRaisesRegex(ValueError, "overlap result contains a record"):
            type(upstream).model_validate(retained)

    def test_an_insertion_is_not_carried_only_beside_a_sibling_insertion(self):
        # A record at an insertion's anchor is at its locus, but only one that
        # lists another insertion at the same point (and no symbolic ALT) can say
        # a fully called sample lacks the target's.
        for alts, ref in (
            ("<INS>", "A"),
            ("G", "A"),
            ("A", "N"),
            ("AC,<INS>", "A"),
            ("AC,AN", "A"),
            ("AC,*", "A"),
        ):
            with self.subTest(alts=alts, ref=ref):
                result, _, _ = self.observe(
                    [query_row(pos=99, ref=ref, alts=alts, genotype="0/1")],
                    ref="",
                    alt="CCC",
                )
                self.assertEqual(result.outcome, "locus_present_identity_mismatch")
        sibling, _, _ = self.observe(
            [query_row(pos=99, ref="A", alts="AC,ACC", genotype="1/2")],
            ref="",
            alt="CCC",
        )
        self.assertEqual(sibling.outcome, "target_allele_not_carried")
        retained = sibling.model_dump(mode="json")
        retained["record"]["alts"] = ["<INS>"]
        retained["record"]["genotypes"][0]["genotype"] = "0/1"
        retained["genotype"].update(raw_genotype="0/1", allele_indexes=[0, 1])
        with self.assertRaisesRegex(ValueError, "not-carried result contradicts"):
            type(sibling).model_validate(retained)

    def test_an_insertion_after_the_last_base_is_looked_up_at_its_anchor(self):
        # A contig of 100 bases; the VCF anchors the insertion after base 100 on
        # that base (100 A>AC), and the target is chr1:101 ->C.
        header = vcf_header("##contig=<ID=chr1,length=100>")
        result, _, calls = self.observe(
            [query_row(pos=100, ref="A", alts="AC", genotype="0/1")],
            ref="",
            alt="C",
            position=101,
            header=header,
        )
        self.assertEqual(result.outcome, "exact_allele_observed")
        self.assertEqual(result.query_invocation.regions_file.text, "chr1\t100\t100\n")
        beyond, _, calls = self.observe(
            [], ref="", alt="C", position=102, header=header
        )
        self.assertEqual(
            (beyond.code, beyond.violations),
            ("target_unresolved", ("one_based_position",)),
        )
        self.assertEqual(calls, [])


class VcfWindowQueryTests(unittest.TestCase):
    """An item's window read (ADR 0047) on the one VCF reader."""

    def query(self, rows, *, flank_bases, position=100, header=None, ref="T", alt="C"):
        from bio_neb.investigation.vcf_observation import VcfWindowQuery

        adapter = FakeAdapter()
        if header is not None:
            adapter.header = header
        adapter, receipt = prepared(adapter)
        adapter.query_result = CompletedProcess(0, b"".join(rows), b"")
        inspected, calls = len(adapter.inspected), len(adapter.calls)
        result = DeterministicVcfObservationTool().query_window(
            ObserveVariantInput(
                assembly=ASSEMBLY,
                contig="chr1",
                one_based_position=position,
                ref=ref,
                alt=alt,
            ),
            VcfWindowQuery(flank_bases=flank_bases),
            continuation(),
            receipt,
            adapter,
            authority_kind="scientific_continuation",
        )
        self.assertEqual(len(adapter.inspected), inspected)
        self.assertEqual(
            type(result).model_validate(result.model_dump(mode="json")), result
        )
        return result, adapter.calls[calls:]

    def test_window_query_reads_upstream_and_downstream_records_with_the_lookup_grammar(
        self,
    ):
        rows = [
            query_row(pos=40, ref="CATG", alts="C", genotype="0/1"),
            query_row(pos=100, ref="T", alts="C", genotype="1/1"),
            query_row(pos=150, ref="G", alts="A", genotype="0/1", quality="PASS\t70"),
        ]

        result, calls = self.query(rows, flank_bases=60)

        self.assertEqual(result.outcome, "window_observed")
        self.assertEqual((result.window_start, result.window_end), (40, 160))
        self.assertEqual(
            [item.one_based_position for item in result.records], [40, 100, 150]
        )
        self.assertEqual(result.records[2].filters, ("PASS",))
        self.assertEqual(
            calls[0][0],
            (
                RUNTIME,
                "query",
                "--allow-undef-tags",
                "--regions-file",
                regions_path("chr1\t40\t160\n"),
                "--format",
                r"%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%QUAL"
                r"[\t%SAMPLE\t%GT\t%GQ\t%DP\t%AD]\n",
                BGZF,
            ),
        )
        clamped, calls = self.query([], flank_bases=500, position=100)
        self.assertEqual((clamped.window_start, clamped.records), (1, ()))
        self.assertEqual(clamped.query_invocation.regions_file.text, "chr1\t1\t600\n")

    def test_a_terminal_insertion_window_ends_at_the_contig(self):
        # The window of chr1:101 ->C on a 100-base contig is read around its
        # anchor base, 100.
        header = vcf_header("##contig=<ID=chr1,length=100>")
        result, calls = self.query(
            [query_row(pos=100, ref="A", alts="AC", genotype="0/1")],
            flank_bases=10,
            position=101,
            header=header,
            ref="",
            alt="C",
        )
        self.assertEqual(result.outcome, "window_observed")
        self.assertEqual((result.window_start, result.window_end), (90, 100))
        self.assertEqual(result.query_invocation.regions_file.text, "chr1\t90\t100\n")
        zero_flank, calls = self.query(
            [query_row(pos=100, ref="A", alts="AC", genotype="0/1")],
            flank_bases=0,
            position=101,
            header=header,
            ref="",
            alt="C",
        )
        self.assertEqual((zero_flank.window_start, zero_flank.window_end), (100, 100))
        beyond, calls = self.query(
            [], flank_bases=10, position=102, header=header, ref="", alt="C"
        )
        self.assertEqual(
            (beyond.code, beyond.violations), ("target_unresolved", ("contig",))
        )
        self.assertEqual(calls, [])
        deletion, calls = self.query([], flank_bases=10, ref="TGT", alt="")
        self.assertEqual((deletion.window_start, deletion.window_end), (90, 112))

    def test_a_window_fault_with_no_process_result_is_a_named_tool_failure(self):
        # A missing executable or an exhausted timeout leaves no process result;
        # the refusal names it instead of claiming an invocation.
        from subprocess import TimeoutExpired

        from bio_neb.failure import (
            FailureOverrides,
            FailureScopes,
            FailureTuning,
            OperationEntry,
            run_failure_tuning,
        )
        from bio_neb.investigation.vcf_observation import VcfWindowQuery

        class Faulty(FakeAdapter):
            fault: Exception = OSError()

            def run(self, argv, *, timeout_seconds, stdout_path=None):
                if argv[1] == "query":
                    self.calls.append((argv, timeout_seconds, stdout_path))
                    raise self.fault
                return super().run(
                    argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
                )

        tuning = FailureTuning(
            scopes=FailureScopes(
                stage={
                    "vcf_observation": OperationEntry(
                        operation=FailureOverrides(retry_attempts=2, backoff_seconds=0)
                    )
                }
            )
        )
        for fault, runs in (
            (FileNotFoundError("bcftools"), 1),
            (TimeoutExpired(("bcftools",), 60), 2),
        ):
            with self.subTest(fault=type(fault).__name__):
                adapter, receipt = prepared(Faulty())
                adapter.fault = fault
                before = len(adapter.calls)
                with run_failure_tuning(tuning):
                    refused = DeterministicVcfObservationTool().query_window(
                        ObserveVariantInput(
                            assembly=ASSEMBLY,
                            contig="chr1",
                            one_based_position=100,
                            ref="T",
                            alt="C",
                        ),
                        VcfWindowQuery(flank_bases=10),
                        continuation(),
                        receipt,
                        adapter,
                        authority_kind="scientific_continuation",
                    )
                self.assertEqual(
                    (refused.code, refused.violations, refused.query_invocation),
                    ("tool_failure", (type(fault).__name__,), None),
                )
                self.assertEqual(len(adapter.calls) - before, runs)
                self.assertEqual(
                    type(refused).model_validate(refused.model_dump(mode="json")),
                    refused,
                )
                hostile = refused.model_dump(mode="json")
                hostile["violations"] = []
                with self.assertRaisesRegex(ValueError, "query evidence"):
                    type(refused).model_validate(hostile)

    def test_a_window_row_before_the_window_is_refused(self):
        rows = [query_row(pos=10, ref="A", alts="G", genotype="0/1")]
        refused, _ = self.query(rows, flank_bases=20)
        self.assertEqual(
            (refused.code, refused.violations), ("tool_failure", ("output_invalid",))
        )

    def test_window_query_refusals_are_typed_and_claim_nothing_biological(self):
        too_wide, calls = self.query([], flank_bases=1_001)
        self.assertEqual(
            (too_wide.code, too_wide.violations), ("invalid_window", ("flank_bases",))
        )
        self.assertEqual(calls, [])
        unusable, calls = self.query(
            [],
            flank_bases=0,
            header=vcf_header("##contig=<ID=chr1>", "##contig=<ID=chr2,length=9>"),
        )
        self.assertEqual(unusable.outcome, "refused")
        self.assertEqual(calls, [])
        for name, process, violation in (
            ("exit", CompletedProcess(1, b"", b"bounded"), "exit_1"),
            ("malformed", CompletedProcess(0, b"wrong\trow\n", b""), "output_invalid"),
            (
                "downstream of the window",
                CompletedProcess(0, query_row(pos=161, ref="T", alts="C"), b""),
                "output_invalid",
            ),
            (
                "sample",
                CompletedProcess(
                    0, b"chr1\t100\tT\tC\t.\t.\tother\t0/1\t.\t.\t.\n", b""
                ),
                "sample_identity",
            ),
        ):
            with self.subTest(name=name):
                adapter = FakeAdapter()
                adapter, receipt = prepared(adapter)
                adapter.query_result = process
                from bio_neb.investigation.vcf_observation import VcfWindowQuery

                refused = DeterministicVcfObservationTool().query_window(
                    ObserveVariantInput(
                        assembly=ASSEMBLY,
                        contig="chr1",
                        one_based_position=100,
                        ref="T",
                        alt="C",
                    ),
                    VcfWindowQuery(flank_bases=60),
                    continuation(),
                    receipt,
                    adapter,
                    authority_kind="scientific_continuation",
                )
                self.assertEqual(
                    (refused.code, refused.violations), ("tool_failure", (violation,))
                )
                self.assertIsNotNone(refused.query_invocation)
                self.assertIn("no coverage", refused.claim_scope.lower())


class VcfObservationTests(unittest.TestCase):
    def test_exact_allele_retains_exact_record_and_genotype(self):
        adapter, receipt = prepared()
        adapter.query_result = CompletedProcess(0, query_row(), b"")

        result = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 783006,
                "ref": "A",
                "alt": "G",
                "display_label": "rs-display-only",
            },
            continuation(),
            receipt,
            adapter,
        )

        self.assertEqual(result.outcome, "exact_allele_observed")
        self.assertEqual(result.record.alts, ("G",))
        self.assertEqual(result.genotype.state, "target_alt_carried")
        self.assertEqual(result.genotype.target_alt_index, 1)
        self.assertEqual(result.query_invocation.purpose, "variant_query")
        self.assertEqual(result.query_invocation.return_code, 0)
        self.assertEqual(
            [(item.sample_name, item.genotype) for item in result.record.genotypes],
            [(SAMPLE_ID, "1/1")],
        )
        query = adapter.calls[-1][0]
        self.assertEqual(
            query,
            (
                RUNTIME,
                "query",
                "--allow-undef-tags",
                "--regions-file",
                regions_path("chr1\t783006\t783006\n"),
                "--format",
                r"%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%QUAL"
                r"[\t%SAMPLE\t%GT\t%GQ\t%DP\t%AD]\n",
                BGZF,
            ),
        )
        self.assertNotIn("rs-display-only", query)
        self.assertEqual(result.query_invocation.argv, query)

    def test_genotype_semantics_follow_the_target_alt_index_without_overstatement(self):
        cases = (
            (
                "biallelic reference",
                "G",
                "G",
                "0/0",
                "target_allele_not_carried",
                "target_alt_not_carried",
                1,
                False,
                None,
            ),
            (
                "later ALT carried",
                "T",
                "G,T",
                "0/2",
                "exact_allele_observed",
                "target_alt_carried",
                2,
                False,
                None,
            ),
            (
                "different ALT carried",
                "G",
                "G,T",
                "0/2",
                "target_allele_not_carried",
                "target_alt_not_carried",
                1,
                False,
                None,
            ),
            (
                "unphased no call",
                "G",
                "G",
                "./.",
                "target_genotype_unresolved",
                "no_call",
                1,
                False,
                "missing_allele",
            ),
            (
                "partial no call",
                "G",
                "G",
                "1/.",
                "target_genotype_unresolved",
                "no_call",
                1,
                False,
                "missing_allele",
            ),
            (
                "phased no call",
                "G",
                "G",
                ".|.",
                "target_genotype_unresolved",
                "no_call",
                1,
                True,
                "missing_allele",
            ),
            (
                "out of range",
                "G",
                "G,T",
                "3/3",
                "target_genotype_unresolved",
                "invalid",
                1,
                False,
                "out_of_range",
            ),
            (
                "malformed",
                "G",
                "G",
                "x/1",
                "target_genotype_unresolved",
                "invalid",
                1,
                None,
                "malformed",
            ),
            (
                "phased target",
                "G",
                "G",
                "0|1",
                "exact_allele_observed",
                "target_alt_carried",
                1,
                True,
                None,
            ),
            # A one-allele call states no phase, and a call whose alleles carry
            # both separators states none as a whole (VCF 4.3).
            (
                "haploid target",
                "G",
                "G",
                "1",
                "exact_allele_observed",
                "target_alt_carried",
                1,
                None,
                None,
            ),
            (
                "haploid reference",
                "G",
                "G",
                "0",
                "target_allele_not_carried",
                "target_alt_not_carried",
                1,
                None,
                None,
            ),
            (
                "haploid no call",
                "G",
                "G",
                ".",
                "target_genotype_unresolved",
                "no_call",
                1,
                None,
                "missing_allele",
            ),
            (
                "mixed-phase polyploid target",
                "G",
                "G",
                "0/1|1",
                "exact_allele_observed",
                "target_alt_carried",
                1,
                None,
                None,
            ),
        )
        for (
            name,
            target_alt,
            record_alts,
            genotype,
            expected_outcome,
            expected_state,
            expected_index,
            expected_phased,
            expected_reason,
        ) in cases:
            with self.subTest(name=name):
                adapter, receipt = prepared()
                adapter.query_result = CompletedProcess(
                    0,
                    query_row(alts=record_alts, genotype=genotype),
                    b"",
                )
                result = DeterministicVcfObservationTool().observe(
                    {
                        "assembly": ASSEMBLY,
                        "contig": "chr1",
                        "one_based_position": 783006,
                        "ref": "A",
                        "alt": target_alt,
                    },
                    continuation(),
                    receipt,
                    adapter,
                )
                self.assertEqual(result.outcome, expected_outcome)
                self.assertEqual(result.genotype.state, expected_state)
                self.assertEqual(result.genotype.target_alt_index, expected_index)
                self.assertEqual(result.genotype.phased, expected_phased)
                self.assertEqual(result.genotype.reason, expected_reason)
                self.assertEqual(
                    type(result).model_validate(result.model_dump(mode="json")),
                    result,
                )
                hostile = result.model_dump(mode="json")
                hostile["genotype"]["raw_genotype"] = "9/9"
                with self.assertRaises(ValidationError):
                    type(result).model_validate(hostile)
                # A phase the call does not state is refused on reopen.
                stamped = result.model_dump(mode="json")
                stamped["genotype"]["phased"] = not expected_phased
                with self.assertRaises(ValidationError):
                    type(result).model_validate(stamped)

    def test_result_matrix_distinguishes_mismatch_no_record_and_ambiguity(self):
        cases = {
            "mismatch": (
                query_row(ref="C", alts="G,T", genotype="0/1"),
                "locus_present_identity_mismatch",
            ),
            "no record": (b"", "no_record"),
            "ambiguous": (
                query_row(genotype="0/1") + query_row(alts="G,T", genotype="1/2"),
                "ambiguous_records",
            ),
        }
        for name, (payload, expected) in cases.items():
            with self.subTest(name=name):
                adapter, receipt = prepared()
                adapter.query_result = CompletedProcess(0, payload, b"")
                result = DeterministicVcfObservationTool().observe(
                    {
                        "assembly": ASSEMBLY,
                        "contig": "chr1",
                        "one_based_position": 783006,
                        "ref": "A",
                        "alt": "G",
                    },
                    continuation(),
                    receipt,
                    adapter,
                )
                self.assertEqual(result.outcome, expected)
                retained = result.model_dump_json().lower()
                if expected == "no_record":
                    self.assertIn("reference genotype is not established", retained)
                else:
                    self.assertNotIn("reference genotype is established", retained)

    def test_invalid_and_unresolved_targets_refuse_before_process_contact(self):
        invalid_targets = (
            {
                # A name a regions file cannot carry as one field.
                "assembly": ASSEMBLY,
                "contig": "chr1\t1",
                "one_based_position": 1,
                "ref": "A",
                "alt": "G",
            },
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 0,
                "ref": "A",
                "alt": "G",
            },
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 1,
                "ref": "A",
                "alt": "*",
            },
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 1,
                "ref": "A",
                "alt": "G",
                "argv": ["sh", "-c", "arbitrary"],
            },
        )
        for target in invalid_targets:
            adapter, receipt = prepared()
            before = len(adapter.calls)
            result = DeterministicVcfObservationTool().observe(
                target, continuation(), receipt, adapter
            )
            self.assertEqual(result.code, "invalid_target")
            self.assertEqual(len(adapter.calls), before)

        adapter, receipt = prepared()
        before = len(adapter.calls)
        unresolved = DeterministicVcfObservationTool().observe(
            {
                "assembly": "GCF_other",
                "contig": "chr1",
                "one_based_position": 1,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )
        self.assertEqual(unresolved.code, "target_unresolved")
        self.assertEqual(len(adapter.calls), before)

        unresolved_contig = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr99",
                "one_based_position": 1,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )
        self.assertEqual(unresolved_contig.code, "target_unresolved")
        self.assertEqual(unresolved_contig.violations, ("contig",))
        self.assertEqual(len(adapter.calls), before)

        past_contig_end = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 248956423,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )
        self.assertEqual(past_contig_end.code, "target_unresolved")
        self.assertEqual(past_contig_end.violations, ("one_based_position",))
        self.assertEqual(len(adapter.calls), before)

    def test_input_and_tool_failures_are_closed_and_claim_nothing_biological(self):
        target = ObserveVariantInput(
            assembly=ASSEMBLY,
            contig="chr1",
            one_based_position=783006,
            ref="A",
            alt="G",
        )
        cases = {}

        nonzero, receipt = prepared()
        nonzero.query_result = CompletedProcess(1, b"", b"bounded")
        cases["nonzero"] = (nonzero, receipt, "tool_failure")

        malformed, receipt = prepared()
        malformed.query_result = CompletedProcess(0, b"wrong\trow\n", b"")
        cases["malformed output"] = (malformed, receipt, "tool_failure")

        wrong_sample, receipt = prepared()
        wrong_sample.query_result = CompletedProcess(
            0, b"chr1\t783006\tA\tG\t.\t.\tunbound\t0/1\t.\t.\t.\n", b""
        )
        cases["sample drift"] = (wrong_sample, receipt, "tool_failure")

        for name, (adapter, receipt, code) in cases.items():
            with self.subTest(name=name):
                inspected = len(adapter.inspected)
                result = DeterministicVcfObservationTool().observe(
                    target, continuation(), receipt, adapter
                )
                self.assertEqual(result.outcome, "refused")
                self.assertEqual(result.code, code)
                retained = result.model_dump_json().lower()
                self.assertIn("no biological absence", retained)
                self.assertNotIn("diagnosis established", retained)
                self.assertIsNotNone(result.query_invocation)
                self.assertEqual(result.query_invocation.purpose, "variant_query")
                # Trusted reuse (0029): a query inspects no prepared file again.
                self.assertEqual(len(adapter.inspected), inspected)

    def test_a_query_timeout_is_retried_by_the_vcf_observation_stage_policy(self):
        from subprocess import TimeoutExpired

        from bio_neb.failure import (
            FailureOverrides,
            FailureScopes,
            FailureTuning,
            OperationEntry,
            run_failure_tuning,
        )

        class Interrupted(FakeAdapter):
            faults: list[Exception] = []

            def run(self, argv, *, timeout_seconds, stdout_path=None):
                if argv[1] == "query" and self.faults:
                    self.calls.append((argv, timeout_seconds, stdout_path))
                    raise self.faults.pop(0)
                return super().run(
                    argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
                )

        target = ObserveVariantInput(
            assembly=ASSEMBLY,
            contig="chr1",
            one_based_position=783006,
            ref="A",
            alt="G",
        )
        tuning = FailureTuning(
            scopes=FailureScopes(
                stage={
                    "vcf_observation": OperationEntry(
                        operation=FailureOverrides(retry_attempts=3, backoff_seconds=0)
                    )
                }
            )
        )
        timeout = TimeoutExpired(("bcftools",), 60)
        for name, faults, outcome, runs in (
            ("transient then answered", [timeout, timeout], "exact_allele_observed", 3),
            ("transient until exhausted", [timeout] * 3, "refused", 3),
            ("not transient", [OSError("bounded")], "refused", 1),
        ):
            with self.subTest(name=name):
                adapter, receipt = prepared(Interrupted())
                adapter.faults = list(faults)
                adapter.query_result = CompletedProcess(0, query_row(), b"")
                before = len(adapter.calls)
                with run_failure_tuning(tuning):
                    result = DeterministicVcfObservationTool().observe(
                        target, continuation(), receipt, adapter
                    )
                self.assertEqual(result.outcome, outcome)
                self.assertEqual(len(adapter.calls) - before, runs)
                if outcome == "refused":
                    self.assertEqual(
                        (result.code, result.violations),
                        ("tool_failure", (type(faults[-1]).__name__,)),
                    )

    def test_a_failed_regions_write_is_a_tool_failure_and_the_next_query_writes_again(
        self,
    ):
        """A regions file that was never published is no input: its query is a
        `tool_failure`, and the next query of that region writes the file again and
        reads what it wrote (#467 Standards r1 ST1, ADR 0029)."""

        class FailsOnce(FakeAdapter):
            failures = 1

            def write_fresh_file(self, path, data):
                if self.failures:
                    # The adapter publishes nothing on a failure (its contract).
                    self.failures -= 1
                    raise OSError("disk full")
                super().write_fresh_file(path, data)

        target = ObserveVariantInput(
            assembly=ASSEMBLY,
            contig="chr1",
            one_based_position=783006,
            ref="A",
            alt="G",
        )
        adapter, receipt = prepared(FailsOnce())
        adapter.query_result = CompletedProcess(0, query_row(), b"")
        before = len(adapter.calls)

        failed = DeterministicVcfObservationTool().observe(
            target, continuation(), receipt, adapter
        )
        self.assertEqual(
            (failed.outcome, failed.code, failed.violations),
            ("refused", "tool_failure", ("OSError",)),
        )
        self.assertEqual(len(adapter.calls), before)
        observed = DeterministicVcfObservationTool().observe(
            target, continuation(), receipt, adapter
        )

        self.assertEqual(observed.outcome, "exact_allele_observed")
        regions = observed.query_invocation.regions_file
        self.assertEqual(regions.text, "chr1\t783006\t783006\n")
        self.assertEqual(adapter.files[regions.path], regions.text.encode())

    def test_serialized_outcomes_reject_hostile_semantic_and_identity_drift(self):
        adapter, receipt = prepared()
        adapter.query_result = CompletedProcess(0, query_row(), b"")
        exact = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 783006,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )
        exact_hostiles = []
        changed_target = exact.model_dump(mode="json")
        changed_target["target"]["alt"] = "T"
        exact_hostiles.append(changed_target)
        changed_gt = exact.model_dump(mode="json")
        changed_gt["genotype"]["raw_genotype"] = "0/0"
        exact_hostiles.append(changed_gt)
        changed_query = exact.model_dump(mode="json")
        changed_query["query_invocation"]["argv"][4] = "chr1:1-1"
        exact_hostiles.append(changed_query)
        changed_id = exact.model_dump(mode="json")
        changed_id["continuation_id"] = "not-an-identity"
        exact_hostiles.append(changed_id)
        for hostile in exact_hostiles:
            with self.assertRaises(ValidationError):
                type(exact).model_validate(hostile)

        adapter, receipt = prepared()
        adapter.query_result = CompletedProcess(
            0, query_row(ref="C", alts="T", genotype="0/1"), b""
        )
        mismatch = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 783006,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )
        hostile_mismatch = mismatch.model_dump(mode="json")
        hostile_mismatch["records"][0].update(ref="A", alts=["G"])
        with self.assertRaises(ValidationError):
            type(mismatch).model_validate(hostile_mismatch)

        adapter, receipt = prepared()
        adapter.query_result = CompletedProcess(
            0, query_row() + query_row(genotype="0|1"), b""
        )
        ambiguous = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 783006,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )
        hostile_ambiguous = ambiguous.model_dump(mode="json")
        hostile_ambiguous["matching_records"].pop()
        with self.assertRaises(ValidationError):
            type(ambiguous).model_validate(hostile_ambiguous)

        adapter, receipt = prepared()
        adapter.query_result = CompletedProcess(0, b"", b"")
        no_record = DeterministicVcfObservationTool().observe(
            {
                "assembly": ASSEMBLY,
                "contig": "chr1",
                "one_based_position": 783006,
                "ref": "A",
                "alt": "G",
            },
            continuation(),
            receipt,
            adapter,
        )
        hostile_no_record = no_record.model_dump(mode="json")
        hostile_no_record["query_invocation"]["stdout_byte_count"] = 1
        with self.assertRaises(ValidationError):
            type(no_record).model_validate(hostile_no_record)


if __name__ == "__main__":
    unittest.main()
