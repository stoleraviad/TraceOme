"""A bound prebuilt index from planning to placement (#365, owner B; #464, owner I7).

The planning model reads Galaxy's candidate (its contigs from the `.ann`, each file's
size from a HEAD) and chooses it; the Execution Review binds it with its extra contigs.
After approval ADMIT_CANDIDATE trusts that choice (owner I7, 2026-10-29: "yes KISS ...
for now we trust fully"): each bound file streams from its URL into the Run's bucket,
checked against its bound size, with its contig names made NCBI's in flight, and the
eight prepared roles are placed with no CPU job. A changed size is drift and an
unreachable file a miss; either opens the CPU preparation Review.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.h100_stage_fakes import h100_provider
from tests.parabricks_operation_fakes import (
    approved_input,
    execute_and_read_path,
    input_storage_provider,
)
from tests.prebuilt_index_support import (
    GALAXY_FASTA,
    GALAXY_INDEX,
    GalaxyOpener,
    offline_galaxy_files,
)
from tests.reference_fakes import OFFLINE_ASSEMBLY_REPORT, SOURCE_GZIP

# Galaxy names the requested sequence by its UCSC name; NCBI's report names an alt
# scaffold Galaxy holds (an extra with a RefSeq name) and not chrUn_extra (one without).
REPORT = (
    OFFLINE_ASSEMBLY_REPORT
    + "alt\talt-scaffold\t1\tChromosome\tKI-alt.1\t=\tNT-alt.1\tALT_REF_LOCI_1"
    "\t16\tchrOffline_alt\n"
)
EXTRAS = (
    ("chrOffline_alt", "TTGACCATGGTTAACC"),
    ("chrUn_extra", "GGATCCATGGCCAATT"),
)
# What the placed files must be, written out by hand from the bundle above: NCBI's
# names, each `.fai` offset the first base's position in the renamed FASTA.
PLACED_FASTA = (
    b">chr-offline\nACGTACGTACGTACGT\n"
    b">NT-alt.1\nTTGACCATGGTTAACC\n"
    b">chrUn_extra\nGGATCCATGGCCAATT\n"
)
PLACED_FAI = (
    b"chr-offline\t16\t13\t50\t51\n"
    b"NT-alt.1\t16\t40\t50\t51\n"
    b"chrUn_extra\t16\t70\t50\t51\n"
)
PLACED_ANN = (
    b"48 3 11\n"
    b"0 chr-offline (null)\n0 16 0\n"
    b"0 NT-alt.1 (null)\n16 16 0\n"
    b"0 chrUn_extra (null)\n32 16 0\n"
)
PLACED_DICT = (
    b"@HD\tVN:1.0\tSO:unsorted\n"
    b"@SQ\tSN:chr-offline\tLN:16\n"
    b"@SQ\tSN:NT-alt.1\tLN:16\n"
    b"@SQ\tSN:chrUn_extra\tLN:16\n"
)
SEQUENCES = {
    "chr-offline": b"ACGTACGTACGTACGT",
    "NT-alt.1": b"TTGACCATGGTTAACC",
    "chrUn_extra": b"GGATCCATGGCCAATT",
}


def _bound_run(root: Path):
    """An approved Run whose Review bound Galaxy's served index, and its files."""
    files = offline_galaxy_files(root / "galaxy", name="chrOffline", extras=EXTRAS)
    opener = GalaxyOpener(files=files)
    approved = approved_input(root / "run", galaxy=opener, assembly_report=REPORT)
    return approved, files, opener


class _Execution:
    """The Run executed over offline providers serving ``served``."""

    def execute(self, approved, served):
        self.providers, self.plans = [], []

        def factory(plan, journal):
            self.plans.append(plan)
            provider = input_storage_provider(plan, journal)
            provider.sources.update(served)
            provider.sources[plan.reference.source_url] = SOURCE_GZIP
            self.providers.append(provider)
            return h100_provider(provider, plan, approved)

        return execute_and_read_path(approved, infrastructure_factory=factory)


class PlacedIndexTests(unittest.TestCase):
    """One bound Run executed once; each test reads what it placed."""

    @classmethod
    def setUpClass(cls) -> None:
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        cls.approved, cls.files, cls.opener = _bound_run(cls.root)
        cls.before = {path for path in cls.root.rglob("*") if path.is_file()}
        execution = _Execution()
        cls.state = execution.execute(cls.approved, cls.files)
        (cls.provider,) = execution.providers
        (cls.plan,) = execution.plans

    def placed(self) -> dict[str, bytes]:
        """Each placed role's bytes, found by its ReferenceResolved pointer into the
        Run's bucket (the admission manifest's pointer is to its Journey artifact)."""
        objects = {}
        for pointer in self.state.reference.pointers:
            if pointer.location.startswith("s3://"):
                bucket, key = pointer.location.removeprefix("s3://").split("/", 1)
                objects[pointer.role] = self.provider.objects[bucket, key]
        return objects

    def test_the_review_binds_the_chosen_candidate_without_a_download(self) -> None:
        bound = self.approved.proposal.execution_package.reference.prebuilt_index
        self.assertTrue(bound.contigs.holds_every_requested_sequence)
        self.assertEqual(
            {item.url: item.byte_count for item in bound.files},
            {url: len(body) for url, body in self.files.items()},
        )
        gets = [url for method, url in self.opener.sent if method == "GET"]
        self.assertNotIn(GALAXY_FASTA, gets)
        self.assertNotIn(f"{GALAXY_INDEX}.bwt", gets)

    def test_a_bound_index_streams_into_the_bucket_as_eight_roles_with_no_cpu_job(
        self,
    ) -> None:
        self.assertEqual(self.state.reference.state, "hit")
        placed = self.placed()
        self.assertEqual(
            sorted(placed),
            sorted(
                [
                    "genomic_fasta",
                    "fasta_fai",
                    "reference_dict",
                    "bwa_amb",
                    "bwa_ann",
                    "bwa_bwt",
                    "bwa_pac",
                    "bwa_sa",
                ]
            ),
        )
        self.assertEqual(
            [p.role for p in self.state.reference.pointers][-1],
            "prepared_reference_admission",
        )
        # Only the H100 job: no CPU preparation.
        self.assertEqual(
            list(self.provider.jobs_by_name), [self.plan.layout.h100_job_name]
        )
        # The files that name no contig are Galaxy's own bytes.
        for role, suffix in (
            ("bwa_amb", ".amb"),
            ("bwa_bwt", ".bwt"),
            ("bwa_pac", ".pac"),
            ("bwa_sa", ".sa"),
        ):
            self.assertEqual(placed[role], self.files[GALAXY_INDEX + suffix], role)
        # Execution wrote none of Galaxy's files to the controller's disk (planning
        # retained the `.ann` it read as a source answer, before approval).
        galaxy = set(self.files.values())
        written = [
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path not in self.before
            and path.read_bytes() in galaxy
        ]
        self.assertEqual(written, [])
        # The admission manifest is the candidate's last object.
        manifest = self.plan.layout.candidate_manifest_object_key
        prefix = manifest.rsplit("/", 1)[0] + "/"
        candidate = [key for _, key in self.provider.objects if key.startswith(prefix)]
        self.assertEqual(len(candidate), 9)
        self.assertTrue(
            candidate[-1].startswith(manifest.removesuffix(".json")), candidate[-1]
        )

    def test_the_placed_names_are_ncbis_and_the_fai_offsets_index_the_placed_fasta(
        self,
    ) -> None:
        placed = self.placed()
        self.assertEqual(placed["genomic_fasta"], PLACED_FASTA)
        self.assertEqual(placed["fasta_fai"], PLACED_FAI)
        self.assertEqual(placed["bwa_ann"], PLACED_ANN)
        # Each record read back from the placed FASTA at its placed .fai offset.
        for row in placed["fasta_fai"].decode().splitlines():
            name, length, offset, bases, width = row.split("\t")
            start, length, bases, width = (
                int(offset),
                int(length),
                int(bases),
                int(width),
            )
            lines = -(-length // bases)
            span = placed["genomic_fasta"][start : start + lines * width]
            self.assertEqual(span.replace(b"\n", b"")[:length], SEQUENCES[name], name)

    def test_the_dict_is_the_renamed_fai_s_contigs(self) -> None:
        placed = self.placed()
        self.assertEqual(placed["reference_dict"], PLACED_DICT)
        self.assertEqual(
            [
                line.split("\t")[1:3]
                for line in placed["reference_dict"].decode().splitlines()[1:]
            ],
            [
                [f"SN:{row.split()[0]}", f"LN:{row.split()[1]}"]
                for row in placed["fasta_fai"].decode().splitlines()
            ],
        )

    def test_an_extra_with_no_ncbi_name_keeps_galaxys(self) -> None:
        bound = self.approved.proposal.execution_package.reference.prebuilt_index
        self.assertEqual(
            [(e.name, e.ncbi_role, e.ncbi_name) for e in bound.contigs.extras],
            [
                ("chrOffline_alt", "alt-scaffold", "NT-alt.1"),
                ("chrUn_extra", None, None),
            ],
        )
        placed = self.placed()
        for role in ("genomic_fasta", "fasta_fai", "bwa_ann", "reference_dict"):
            self.assertIn(b"chrUn_extra", placed[role], role)
            self.assertNotIn(b"chrOffline", placed[role], role)

    def test_the_plan_streams_the_source_for_the_cpu_fallback(self) -> None:
        self.assertIsNotNone(self.plan.reference.prebuilt_index)
        self.assertIsNotNone(self.plan.input_source)


class UnplacedIndexTests(_Execution, unittest.TestCase):
    """A bound file Galaxy no longer serves as the Review bound it opens the CPU
    preparation Review, with no job (owner I7, #464: "a changed size or an
    unreachable file is drift or a miss")."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.approved, self.files, _ = _bound_run(Path(directory.name))

    def test_a_changed_size_is_drift(self) -> None:
        served = dict(self.files)
        served[f"{GALAXY_INDEX}.sa"] += b"\0" * 8

        state = self.execute(self.approved, served)

        self.assertEqual(state.reference.state, "drifted")
        self.assertEqual(state.reference.reason, "source_identity_mismatch")
        self.assertEqual(state.status, "cpu_preparation_review_pending")
        self.assertFalse(self.providers[0].jobs_by_name)

    def test_an_unreachable_file_is_a_miss(self) -> None:
        served = {url: body for url, body in self.files.items() if url != GALAXY_FASTA}

        state = self.execute(self.approved, served)

        self.assertEqual(state.reference.state, "miss")
        self.assertEqual(state.reference.reason, "candidate_absent")
        self.assertEqual(state.status, "cpu_preparation_review_pending")
        self.assertFalse(self.providers[0].jobs_by_name)


class TransformedTransferProofTests(unittest.TestCase):
    """A renamed transfer proves its original source separately (#464 review r4 ST6,
    ADR 0029: the acquired identity describes the bytes measured)."""

    def test_a_renamed_transfer_without_a_valid_source_proof_is_refused(self):
        import hashlib
        from types import SimpleNamespace

        from bio_neb.failure import ProviderError
        from bio_neb.parabricks.reference_stages import _stream
        from bio_neb.parabricks.worker_exchange import PREPARED_ROLES
        from bio_neb.prebuilt_index.rename import ContigRename
        from bio_neb.providers.ports import Receipt

        original, renamed = b">a\nAC\n", b">b\nAC\n"
        placed = {
            "bytes": len(renamed),
            "sha256": hashlib.sha256(renamed).hexdigest(),
        }
        plan = SimpleNamespace(
            storage=SimpleNamespace(bucket="fresh-run"),
            layout=SimpleNamespace(candidate_object_keys=PREPARED_ROLES),
            reference=SimpleNamespace(
                prebuilt_index=SimpleNamespace(
                    files=(
                        SimpleNamespace(
                            role="genomic_fasta",
                            url="https://example.test/a.fa",
                            byte_count=len(original),
                        ),
                    )
                )
            ),
        )
        rename = ContigRename("fasta", (("a", "b"),), len(renamed))

        def stream(values):
            class Store:
                def put(self, bucket, key, source):
                    return Receipt(values=values)

            return _stream(SimpleNamespace(plan=plan), Store(), "genomic_fasta", rename)

        for name, values in {
            "absent": placed,
            "empty digest": {**placed, "source": {"bytes": 6, "sha256": ""}},
            "not a digest": {
                **placed,
                "source": {"bytes": 6, "sha256": "not-a-sha256"},
            },
            "not hex": {**placed, "source": {"bytes": 6, "sha256": "z" * 64}},
            "no size": {**placed, "source": {"sha256": "0" * 64}},
        }.items():
            with self.subTest(name), self.assertRaises(ProviderError) as refused:
                stream(values)
            self.assertEqual(refused.exception.detail, "source_proof_invalid")
        source = {
            "bytes": len(original),
            "sha256": hashlib.sha256(original).hexdigest(),
        }
        _, origin = stream({**placed, "source": source})
        self.assertEqual(
            (origin["byte_count"], origin["sha256"]),
            (source["bytes"], source["sha256"]),
        )


if __name__ == "__main__":
    unittest.main()
