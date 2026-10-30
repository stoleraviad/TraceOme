"""The prebuilt-index modules (#365, #464): Galaxy's catalogues, the in-flight rename to
NCBI's names execution streams through (owner I7), and the M5 fit and role assembly lane
Rf's development tool (`tools/reference_index`) keeps using."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from bio_neb.parabricks.bwa_index import read_bwa_index
from bio_neb.parabricks.worker_exchange import PREPARED_ROLES
from bio_neb.prebuilt_index.catalogue import (
    DATACACHE,
    CatalogueUnreadable,
    datacache_url,
    galaxy_candidates,
    parse_loc,
    parse_tool_data,
)
from tests.prebuilt_index_support import (
    CHROMOSOME_I,
    CHROMOSOME_II,
    NCBI,
    bwa_text_files,
    galaxy_contigs,
    write_bundle,
    write_fasta,
)
from tools.reference_index.bundle import assemble_prepared_roles
from tools.reference_index.fit import (
    BUNDLE_SUFFIXES,
    BoundExtra,
    BundleUnreadable,
    FitReport,
    check_fit,
)


class FitFixture:
    """An NCBI-style requested FASTA and an empty Galaxy-style candidate folder."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.requested = self.root / "ncbi" / "GCF_000000001.1_genomic.fna"
        self.requested.parent.mkdir()
        write_fasta(self.requested, NCBI, width=80)
        (self.root / "galaxy").mkdir()
        self.candidate = self.root / "galaxy" / "GCF_000000001.1.fa"

    def check(self, bound_extras=()):
        return check_fit(
            requested=self.requested,
            requested_accession="GCF_000000001.1",
            candidate=self.candidate,
            candidate_id="galaxy:GCF_000000001.1",
            bound_extras=bound_extras,
        )


class FitCheckTests(FitFixture, unittest.TestCase):
    def test_an_index_with_ncbis_names_fits(self):
        write_bundle(self.candidate, tuple((n, "", s) for n, _, s in NCBI), width=60)

        report = self.check()

        self.assertEqual(
            (report.verdict, report.sequence_match, report.names, report.order),
            ("fits", "all", "identical", "same"),
        )
        self.assertEqual(report.reasons, ())
        self.assertEqual(
            [check.check for check in report.index_checks],
            ["fasta_fai", "ann_fai", "amb_ann", "pac_size", "bwt_sa_header"],
        )

    def test_a_renamed_only_index_fits_after_rename(self):
        write_bundle(self.candidate, galaxy_contigs(), width=50, lower=True)

        report = self.check()

        self.assertEqual(report.verdict, "fits_after_rename")
        self.assertEqual(report.reasons, ())
        self.assertEqual(report.sequence_match, "all")
        self.assertEqual(report.names, "renamed")
        self.assertEqual(
            [(c.requested_name, c.candidate_name, c.status) for c in report.contigs],
            [
                ("NC_000001.1", "chrI", "match"),
                ("NC_000002.1", "chrII", "match"),
                ("NC_000003.1", "chrM", "match"),
            ],
        )
        self.assertEqual(
            report.contigs[1].requested_m5,
            hashlib.md5(CHROMOSOME_II.encode(), usedforsecurity=False).hexdigest(),
        )
        self.assertTrue(all(check.passed for check in report.index_checks))

    def test_a_renamed_index_with_ambiguous_bases_fits_after_rename(self):
        gapped = (
            CHROMOSOME_II[:20]
            + "NNNNN"
            + CHROMOSOME_II[25:60]
            + "R"
            + CHROMOSOME_II[61:]
        )
        ncbi = tuple(
            (name, description, gapped if name == "NC_000002.1" else sequence)
            for name, description, sequence in NCBI
        )
        write_fasta(self.requested, ncbi, width=80)
        write_bundle(self.candidate, galaxy_contigs((1, ("chrII", "", gapped))))
        self.assertIn(b" 5 N\n", Path(f"{self.candidate}.amb").read_bytes())

        report = self.check()

        self.assertEqual((report.verdict, report.reasons), ("fits_after_rename", ()))

    def test_a_changed_contig_is_refused_with_its_name(self):
        changed = CHROMOSOME_II[:40] + "G" + CHROMOSOME_II[41:]
        self.assertNotEqual(changed, CHROMOSOME_II)
        write_bundle(self.candidate, galaxy_contigs((1, ("chrII", "", changed))))

        report = self.check()

        self.assertEqual(report.verdict, "refused")
        self.assertEqual(report.sequence_match, "partial")
        self.assertEqual(
            report.reasons, ("sequence_differs: NC_000002.1 (candidate chrII)",)
        )
        self.assertEqual(
            [c.status for c in report.contigs],
            ["match", "sequence_differs", "match"],
        )
        self.assertTrue(all(check.passed for check in report.index_checks))

    def test_a_missing_and_an_extra_contig_are_refused_by_name(self):
        decoy = ("chrUn_decoy", "", "GATTACA" * 5)
        write_bundle(self.candidate, (*galaxy_contigs()[:2], decoy))

        report = self.check()

        self.assertEqual(report.verdict, "refused")
        self.assertEqual(report.sequence_match, "partial")
        self.assertEqual(
            report.reasons,
            (
                "missing_from_candidate: NC_000003.1",
                "extra_in_candidate: chrUn_decoy",
            ),
        )

    def test_an_extra_contig_the_review_bound_fits_and_an_unbound_one_is_refused(self):
        # An index holding more than the Run's reference (human alt loci) is used,
        # its extras disclosed in the Execution Review; an extra the Review did not
        # bind is drift.
        decoy = ("chrUn_decoy", "", "GATTACA" * 5)
        write_bundle(self.candidate, (*galaxy_contigs(), decoy))

        bound = (BoundExtra(name="chrUn_decoy", length=35),)
        report = self.check(bound_extras=bound)

        self.assertEqual((report.verdict, report.reasons), ("fits_after_rename", ()))
        self.assertEqual(report.bound_extras, bound)
        self.assertEqual(
            report.renames(),
            {"chrI": "NC_000001.1", "chrII": "NC_000002.1", "chrM": "NC_000003.1"},
        )
        refused = self.check(bound_extras=(BoundExtra(name="chrUn_other", length=35),))
        self.assertEqual(refused.reasons, ("extra_in_candidate: chrUn_decoy",))

    def test_a_bound_extra_of_another_length_is_refused(self):
        # Spec r1 SP1: the Review bound each extra's name and length; the same name
        # at another length is not the contig the researcher approved.
        write_bundle(
            self.candidate, (*galaxy_contigs(), ("chrUn_decoy", "", "GATTACA" * 5))
        )

        refused = self.check(bound_extras=(BoundExtra(name="chrUn_decoy", length=34),))

        self.assertEqual(refused.verdict, "refused")
        self.assertEqual(
            refused.reasons,
            ("extra_in_candidate: chrUn_decoy is 35 bases; the Review bound 34",),
        )

    def test_a_pac_of_another_sequence_with_the_same_lengths_still_fits(self):
        # No `bwa fa2pac -f` reproduction in the product, so the `.pac` content is not
        # compared with the FASTA (a named gap). Its size and trailer still are.
        write_bundle(self.candidate, galaxy_contigs())
        other = galaxy_contigs((0, ("chrI", "", CHROMOSOME_I[::-1])))
        Path(f"{self.candidate}.pac").write_bytes(bwa_text_files(other)[".pac"])

        report = self.check()

        self.assertEqual(report.verdict, "fits_after_rename")

    def test_an_ann_that_disagrees_with_the_fai_is_refused(self):
        write_bundle(self.candidate, galaxy_contigs())
        renamed = Path(f"{self.candidate}.ann").read_text().replace("chrII", "chr2")
        Path(f"{self.candidate}.ann").write_text(renamed)

        report = self.check()

        self.assertEqual(report.verdict, "refused")
        self.assertEqual(
            [check.check for check in report.index_checks if not check.passed],
            ["ann_fai"],
        )

    def test_a_bwt_or_sa_with_a_misaligned_payload_is_refused(self):
        for suffix in (".bwt", ".sa"):
            with self.subTest(suffix=suffix):
                shutil.rmtree(self.root / "galaxy")
                (self.root / "galaxy").mkdir()
                write_bundle(self.candidate, galaxy_contigs())
                with open(f"{self.candidate}{suffix}", "ab") as index:
                    index.write(b"\0")

                report = self.check()

                self.assertEqual(
                    [check.check for check in report.index_checks if not check.passed],
                    ["bwt_sa_header"],
                )
                self.assertEqual(report.verdict, "refused")

    def test_a_file_that_is_not_its_format_is_refused_with_its_roles_reason(self):
        # The product reader's names for each role.
        cases = (
            (".fai", b"\xffchrI\n", "fai_structure_invalid"),
            ("", b">\xff\nACGT\n", "fasta_structure_invalid"),
            (".pac", b"", "local_role_not_positive_regular_file"),
        )
        for suffix, body, reason in cases:
            with self.subTest(reason=reason):
                write_bundle(self.candidate, galaxy_contigs())
                Path(f"{self.candidate}{suffix}").write_bytes(body)

                with self.assertRaises(BundleUnreadable) as raised:
                    self.check()

                self.assertEqual(
                    (raised.exception.reason, raised.exception.path),
                    (reason, Path(f"{self.candidate}{suffix}")),
                )

    def test_every_file_is_read_before_any_is_parsed(self):
        # The reader's order, as admission's, so a later missing file is refused
        # before an earlier malformed one.
        write_bundle(self.candidate, galaxy_contigs())
        Path(f"{self.candidate}.fai").write_text("chrI\t1\n")
        Path(f"{self.candidate}.sa").unlink()

        with self.assertRaises(BundleUnreadable) as raised:
            self.check()

        self.assertEqual(
            (raised.exception.reason, raised.exception.path.name),
            ("file_unreadable", "GCF_000000001.1.fa.sa"),
        )

    def test_a_report_that_contradicts_its_contigs_or_checks_does_not_validate(self):
        write_bundle(self.candidate, galaxy_contigs())
        honest = self.check().model_dump(mode="json")
        changed_row = {**honest["contigs"][1], "candidate_m5": "0" * 32}
        missing_row = {
            **honest["contigs"][2],
            "status": "missing_from_candidate",
            "candidate_name": None,
            "candidate_length": None,
            "candidate_m5": None,
        }
        contradictions = {
            "a match row with unequal M5": {
                "contigs": [*honest["contigs"][:1], changed_row, *honest["contigs"][2:]]
            },
            "no index checks": {"index_checks": []},
            "a missing index check": {"index_checks": honest["index_checks"][:-1]},
            "all claimed with a missing contig": {
                "contigs": [*honest["contigs"][:2], missing_row]
            },
            "identical names claimed for renamed contigs": {
                "names": "identical",
                "verdict": "fits",
            },
        }
        for name, change in contradictions.items():
            with self.subTest(name), self.assertRaises(ValueError):
                FitReport.model_validate({**honest, **change})

    def test_an_unreadable_requested_fasta_is_refused(self):
        write_bundle(self.candidate, galaxy_contigs())
        self.requested.unlink()

        with self.assertRaises(BundleUnreadable) as raised:
            self.check()

        self.assertEqual(
            (raised.exception.reason, raised.exception.path),
            ("fasta_unreadable", self.requested),
        )


class AssembleTests(FitFixture, unittest.TestCase):
    """A fitted candidate becomes the eight prepared roles, renamed to NCBI's names when
    the fit paired its contigs under other ones (owner Q6 a)."""

    def assemble(self):
        report = self.check()
        return report, assemble_prepared_roles(
            report, self.candidate, self.root / "prepared"
        )

    def read(self, assembled):
        roles = assembled.roles
        return read_bwa_index(
            roles["genomic_fasta"],
            {role: path for role, path in roles.items() if role != "genomic_fasta"},
        )

    def test_a_fit_under_ncbis_names_is_placed_as_published(self):
        write_bundle(self.candidate, tuple((n, "", s) for n, _, s in NCBI))

        report, assembled = self.assemble()

        self.assertEqual(assembled.derivation_rule, "as_published")
        self.assertEqual(tuple(assembled.roles), PREPARED_ROLES)
        for role, suffix in (("genomic_fasta", ""), *BUNDLE_SUFFIXES.items()):
            self.assertEqual(
                assembled.roles[role].read_bytes(),
                Path(f"{self.candidate}{suffix}").read_bytes(),
                role,
            )
        reading = self.read(assembled)
        self.assertEqual(reading.findings, ())
        self.assertEqual(
            [(c.name, c.m5) for c in reading.contigs],
            [(c.requested_name, c.requested_m5) for c in report.contigs],
        )

    def test_a_fit_under_other_names_is_renamed_to_ncbis(self):
        write_bundle(self.candidate, galaxy_contigs(), width=50, lower=True)

        report, assembled = self.assemble()

        self.assertEqual(report.verdict, "fits_after_rename")
        self.assertEqual(assembled.derivation_rule, "renamed_to_requested_names")
        fasta = assembled.roles["genomic_fasta"].read_text()
        self.assertEqual(
            [line for line in fasta.splitlines() if line.startswith(">")],
            [">NC_000001.1", ">NC_000002.1", ">NC_000003.1"],
        )
        for role in ("bwa_bwt", "bwa_sa", "bwa_pac", "bwa_amb"):
            self.assertEqual(
                assembled.roles[role].read_bytes(),
                Path(f"{self.candidate}{BUNDLE_SUFFIXES[role]}").read_bytes(),
                role,
            )
        reading = self.read(assembled)
        self.assertEqual(reading.findings, ())
        self.assertEqual(
            reading.ann.names, ("NC_000001.1", "NC_000002.1", "NC_000003.1")
        )
        self.assertEqual(
            [(c.name, c.length, c.m5) for c in reading.contigs],
            [
                (c.requested_name, c.requested_length, c.requested_m5)
                for c in report.contigs
            ],
        )

    def test_a_refused_fit_is_not_assembled(self):
        write_bundle(self.candidate, galaxy_contigs()[:2])

        with self.assertRaisesRegex(ValueError, "refused"):
            self.assemble()
        self.assertFalse((self.root / "prepared").exists())


BRC = "http://datacache.galaxyproject.org/brc/data/genomes"


def _loc(*rows):
    return "\n".join(
        ("#<unique_build_id>\t<dbkey>\t<display_name>\t<file_path>", *rows)
    )


def _row(value, path, *, dbkey=None, name=None):
    return f"{value}\t{dbkey or value}\t{name or value}\t{path}"


class CatalogueTests(unittest.TestCase):
    def index_rows(self):
        return parse_loc(
            _loc(
                _row(
                    "GCF_000002985.6",
                    "/cvmfs/brc.galaxyproject.org/data/genomes/GCF_000002985.6/"
                    "bwa_mem_index/v1/GCF_000002985.6/GCF_000002985.6.fa",
                    name="Caenorhabditis elegans (GCA_000002985.3_WBcel235)",
                ),
                _row(
                    "GCA_000002985.3",
                    "/cvmfs/brc.galaxyproject.org/data/genomes/GCA_000002985.3/"
                    "bwa_mem_index/v1/GCA_000002985.3/GCA_000002985.3.fa",
                ),
                _row("GCF_000146045.2", "/galaxy/local/sacCer3.fa"),
            ),
            catalogue="brc-bwa_mem_index.loc",
        )

    def fasta_rows(self):
        return parse_loc(
            _loc(
                _row(
                    "GCF_000002985.6",
                    "/cvmfs/brc.galaxyproject.org/data/genomes/GCF_000002985.6/"
                    "sam_fasta_index/v1/GCF_000002985.6/GCF_000002985.6.fa",
                ),
                _row(
                    "GCF_000146045.2",
                    "/cvmfs/brc.galaxyproject.org/data/genomes/GCF_000146045.2/"
                    "sam_fasta_index/v1/GCF_000146045.2/GCF_000146045.2.fa",
                ),
            ),
            catalogue="brc-fasta_indexes.loc",
        )

    def test_each_repository_s_cvmfs_path_has_its_datacache_mirror(self):
        # Galaxy's four datacache repositories: a path outside them has none.
        for path, url in {
            "/cvmfs/data.galaxyproject.org/byhand/hg38/hg38full/bwa/hg38full.fa": (
                f"{DATACACHE}/indexes/hg38/hg38full/bwa/hg38full.fa"
            ),
            "/cvmfs/data.galaxyproject.org/managed/bwa_mem_index/sacCer3/sacCer3.fa": (
                f"{DATACACHE}/managed/bwa_mem_index/sacCer3/sacCer3.fa"
            ),
            "/cvmfs/brc.galaxyproject.org/data/genomes/x/x.fa": (
                f"{DATACACHE}/brc/data/genomes/x/x.fa"
            ),
            "/cvmfs/vgp.galaxyproject.org/data/genomes/y/y.fa": (
                f"{DATACACHE}/vgp/data/genomes/y/y.fa"
            ),
            "/galaxy/local/sacCer3.fa": None,
        }.items():
            with self.subTest(path):
                self.assertEqual(datacache_url(path), url)

    def test_a_row_s_fasta_comes_from_its_own_repository(self):
        index = parse_loc(
            _loc(
                _row(
                    "hg38",
                    "/cvmfs/data.galaxyproject.org/byhand/hg38/hg38full/bwa/hg38full.fa",
                    name="Human (Homo sapiens) (b38): hg38",
                )
            ),
            catalogue="indexes-bwa_mem_index.loc",
        )
        fasta = "/cvmfs/data.galaxyproject.org/byhand/hg38/sam_index/hg38.fa"
        other = parse_loc(
            _loc(_row("hg38", "/cvmfs/brc.galaxyproject.org/hg38.fa")),
            catalogue="brc-fasta_indexes.loc",
        )
        own = parse_loc(
            _loc(_row("hg38", fasta)), catalogue="indexes-fasta_indexes.loc"
        )

        self.assertEqual(galaxy_candidates("hg38", index, other), ())
        (candidate,) = galaxy_candidates("hg38", index, (*other, *own))
        self.assertEqual(
            (candidate.candidate_id, candidate.value, candidate.dbkey),
            ("galaxy:indexes:hg38", "hg38", "hg38"),
        )
        self.assertEqual(
            candidate.fasta_url, f"{DATACACHE}/indexes/hg38/sam_index/hg38.fa"
        )

    def test_a_row_of_the_value_is_a_candidate_with_its_file_urls(self):
        (candidate,) = galaxy_candidates(
            "GCF_000002985.6", self.index_rows(), self.fasta_rows()
        )

        self.assertEqual(candidate.candidate_id, "galaxy:brc:GCF_000002985.6")
        self.assertEqual(candidate.catalogue, "brc-bwa_mem_index.loc")
        self.assertEqual(
            candidate.display_name, "Caenorhabditis elegans (GCA_000002985.3_WBcel235)"
        )
        fasta = f"{BRC}/GCF_000002985.6/sam_fasta_index/v1/GCF_000002985.6/GCF_000002985.6.fa"
        index = (
            f"{BRC}/GCF_000002985.6/bwa_mem_index/v1/GCF_000002985.6/GCF_000002985.6.fa"
        )
        self.assertEqual(
            candidate.files(),
            {
                "genomic_fasta": fasta,
                "fasta_fai": fasta + ".fai",
                "bwa_amb": index + ".amb",
                "bwa_ann": index + ".ann",
                "bwa_bwt": index + ".bwt",
                "bwa_pac": index + ".pac",
                "bwa_sa": index + ".sa",
            },
        )

    def test_another_version_or_prefix_of_the_assembly_is_not_a_candidate(self):
        self.assertEqual(
            galaxy_candidates("GCF_000002985.5", self.index_rows(), self.fasta_rows()),
            (),
        )
        self.assertEqual(
            [
                c.candidate_id
                for c in galaxy_candidates(
                    "GCA_000002985.3", self.index_rows(), self.fasta_rows()
                )
            ],
            [],
        )

    def test_a_row_without_a_datacache_path_or_fasta_row_is_not_a_candidate(self):
        # GCF_000146045.2's index row names a local path; GCA_000002985.3 has no
        # FASTA row: neither can be fetched.
        self.assertEqual(
            galaxy_candidates("GCF_000146045.2", self.index_rows(), self.fasta_rows()),
            (),
        )

    def test_usegalaxy_tool_data_rows_parse_by_their_columns(self):
        rows = parse_tool_data(
            json.dumps(
                {
                    "columns": ["value", "dbkey", "name", "path"],
                    "fields": [["ce11", "ce11", "C. elegans: ce11", "ce11.fa"]],
                }
            ),
            catalogue="usegalaxy-bwa_mem_indexes.json",
        )

        self.assertEqual(
            [(r.value, r.dbkey, r.name, r.path) for r in rows],
            [("ce11", "ce11", "C. elegans: ce11", "ce11.fa")],
        )

    def test_a_catalogue_that_is_not_its_shape_is_unreadable(self):
        with self.assertRaisesRegex(CatalogueUnreadable, "line 2 is not a loc row"):
            parse_loc("#header\nGCF_1\tonly two", catalogue="brc-bwa_mem_index.loc")
        with self.assertRaisesRegex(CatalogueUnreadable, "is not a tool-data table"):
            parse_tool_data("{}", catalogue="usegalaxy-bwa_mem_indexes.json")


class RenameTests(unittest.TestCase):
    """Galaxy's names made NCBI's in flight (owner I7, #464): only names, and the
    `.fai` offsets they move, change; the result is the same however the source's
    bytes arrive."""

    NAMES = (("chrI", "NC_000001.1"), ("chrII", "NC_000002.1"))

    def stream(self, rename, data, *, piece):
        """``data`` renamed, read from a source that answers ``piece`` bytes at a
        time, as a network response may."""

        class Trickle:
            def __init__(self):
                self.offset = 0

            def read(self, size=-1):
                start, self.offset = self.offset, self.offset + piece
                return data[start : self.offset]

        reader = rename.open(Trickle())
        parts = []
        while chunk := reader.read(7):
            parts.append(chunk)
        return b"".join(parts), reader.source_identity()

    def test_fasta_headers_are_renamed_whatever_the_chunking(self):
        from bio_neb.prebuilt_index.rename import ContigRename, renamed_byte_count

        source = b">chrI first\nACGT\nAC\n>chrII\nGGGG\n>chrM\nTT\n"
        expected = b">NC_000001.1 first\nACGT\nAC\n>NC_000002.1\nGGGG\n>chrM\nTT\n"
        size = renamed_byte_count(len(source), self.NAMES)
        self.assertEqual(size, len(expected))
        rename = ContigRename("fasta", self.NAMES, size)
        self.assertEqual(rename.whole(source), expected)
        for piece in (1, 2, 3, 5, 64):
            with self.subTest(piece=piece):
                renamed, identity = self.stream(rename, source, piece=piece)
                self.assertEqual(renamed, expected)
                self.assertEqual(
                    identity, (len(source), hashlib.sha256(source).hexdigest())
                )

    def test_fai_names_and_offsets_follow_each_header_s_change(self):
        from bio_neb.prebuilt_index.rename import ContigRename

        # chrI -> NC_000001.1 is 7 bytes longer, chrII -> NC_000002.1 6: offsets 6,
        # 95 and 152 become 13, 108 (+7+6, a digit more, so a .fai is rewritten
        # whole) and 165.
        source = b"chrI\t80\t6\t60\t61\nchrII\t50\t95\t60\t61\nchrM\t9\t152\t60\t61\n"
        expected = (
            b"NC_000001.1\t80\t13\t60\t61\n"
            b"NC_000002.1\t50\t108\t60\t61\n"
            b"chrM\t9\t165\t60\t61\n"
        )
        rename = ContigRename("fai", self.NAMES)
        self.assertEqual(rename.whole(source), expected)
        self.assertEqual(self.stream(rename, source, piece=3)[0], expected)

    def test_ann_descriptor_names_are_renamed_and_positions_kept(self):
        from bio_neb.prebuilt_index.rename import ContigRename, renamed_byte_count

        source = (
            b"10 3 11\n0 chrI (null)\n0 4 0\n0 chrII x y\n4 4 1\n0 chrM (null)\n8 2 0\n"
        )
        expected = (
            b"10 3 11\n0 NC_000001.1 (null)\n0 4 0\n"
            b"0 NC_000002.1 x y\n4 4 1\n0 chrM (null)\n8 2 0\n"
        )
        rename = ContigRename(
            "ann", self.NAMES, renamed_byte_count(len(source), self.NAMES)
        )
        self.assertEqual(rename.whole(source), expected)
        self.assertEqual(self.stream(rename, source, piece=4)[0], expected)


if __name__ == "__main__":
    unittest.main()
