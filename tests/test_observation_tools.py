"""The Run's observation tools come from the locked environment (owner, #107)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_VCF = (
    b"##fileformat=VCFv4.2\n"
    b"##contig=<ID=NC_000019.10,length=58617616>\n"
    # An HLA contig: its `*` and `:` are no region-string syntax in a file.
    b"##contig=<ID=HLA-A*01:01:01:01,length=3503>\n"
    b'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001\n"
    b"NC_000019.10\t44908684\t.\tT\tC\t50\tPASS\t.\tGT\t0/1\n"
    b"HLA-A*01:01:01:01\t10\t.\tA\tG\t50\tPASS\t.\tGT\t1/1\n"
)


def _tool(name: str) -> str:
    return str(Path(sys.prefix) / "bin" / f"bio-neb-{name}")


def _run(*argv: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, capture_output=True, check=False, timeout=120)


class BundledToolTests(unittest.TestCase):
    def test_each_tool_reports_the_bundled_tool_and_htslib_versions(self):
        # pysam 0.24.1 bundles samtools, bcftools and htslib 1.24; its own C
        # samtools reports "samtools 1.24 (pysam)" / "Using htslib 1.24".
        for name in ("bcftools", "samtools"):
            with self.subTest(name):
                completed = _run(_tool(name), "--version")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                lines = completed.stdout.decode().splitlines()
                self.assertIn(f"{name} 1.24", lines)
                self.assertIn("Using htslib 1.24", lines)

    def test_bcftools_runs_every_command_the_vcf_tool_issues(self):
        bcftools = _tool("bcftools")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "variants.vcf")
            source.write_bytes(_VCF)
            bgzf = Path(directory, "variants.vcf.gz")
            view = _run(
                bcftools,
                "view",
                "--no-version",
                "--output-type",
                "z",
                "--output",
                str(bgzf),
                str(source),
            )
            index = _run(
                bcftools, "index", "--csi", "--output", f"{bgzf}.csi", str(bgzf)
            )
            header = _run(bcftools, "head", str(bgzf))
            stats = _run(bcftools, "index", "--stats", str(bgzf))
            queries = []
            for name, region in (
                ("nc.tsv", "NC_000019.10\t44908684\t44908684\n"),
                ("hla.tsv", "HLA-A*01:01:01:01\t10\t10\n"),
            ):
                # The lookup's regions file: CHROM, BEG, END, one-based.
                regions = Path(directory, name)
                regions.write_text(region)
                queries.append(
                    _run(
                        bcftools,
                        "query",
                        "--allow-undef-tags",
                        "--regions-file",
                        str(regions),
                        "--format",
                        "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
                        str(bgzf),
                    )
                )
            query, hla = queries
        for completed in (view, index, header, stats, query, hla):
            self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(
            header.stdout.endswith(
                b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001\n"
            )
        )
        self.assertEqual(
            stats.stdout,
            b"NC_000019.10\t58617616\t1\nHLA-A*01:01:01:01\t3503\t1\n",
        )
        self.assertEqual(query.stdout, b"NC_000019.10\t44908684\tT\tC\t0/1\n")
        self.assertEqual(hla.stdout, b"HLA-A*01:01:01:01\t10\tA\tG\t1/1\n")

    def test_a_failed_command_exits_nonzero_with_its_message(self):
        completed = _run(_tool("bcftools"), "view", "/nonexistent/variants.vcf")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"nonexistent", completed.stderr)


def _named_tools(version: str = "0.24.1"):
    from bio_neb.agentic_journey.runtime_config import InvestigationTools

    tool = {"package": "pysam", "package_version": version}
    return InvestigationTools.model_validate({"bcftools": tool, "samtools": tool})


class ToolProbeTests(unittest.TestCase):
    def test_the_run_probes_each_named_tool_once_and_keeps_its_receipt(self):
        from bio_neb.investigation.observation_tools import probe_observation_tools

        with tempfile.TemporaryDirectory() as directory:
            receipts = Path(directory, "tools")
            probe = probe_observation_tools(_named_tools(), receipts)
            receipt = Path(probe.receipt_path)
            self.assertEqual(receipt.parent, receipts)
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o444)
            self.assertEqual(
                {
                    (
                        tool.name,
                        tool.executable,
                        tool.package_version,
                        tool.tool_version,
                        tool.htslib_version,
                    )
                    for tool in probe.receipt.tools
                },
                {
                    (name, _tool(name), "0.24.1", "1.24", "1.24")
                    for name in ("bcftools", "samtools")
                },
            )
            # A later need reuses the receipt; it runs no executable again.
            reused = probe_observation_tools(
                _named_tools(), receipts, bin_directory=Path(directory, "empty")
            )
            self.assertEqual(reused, probe)

    def test_a_tool_the_environment_cannot_provide_is_a_configuration_fault(self):
        from bio_neb.agentic_journey.runtime_config import RuntimeConfigError
        from bio_neb.investigation.observation_tools import probe_observation_tools

        with tempfile.TemporaryDirectory() as directory:
            cases = (
                (None, {}, ("investigation_tools",)),
                (
                    _named_tools("0.23.0"),
                    {},
                    (
                        "investigation_tools.bcftools.package_version",
                        "investigation_tools.samtools.package_version",
                    ),
                ),
                (
                    _named_tools(),
                    {"bin_directory": Path(directory, "empty")},
                    ("investigation_tools.bcftools", "investigation_tools.samtools"),
                ),
            )
            for index, (tools, options, violations) in enumerate(cases):
                receipts = Path(directory, f"tools-{index}")
                with self.subTest(violations=violations):
                    with self.assertRaises(RuntimeConfigError) as refused:
                        probe_observation_tools(tools, receipts, **options)
                    self.assertEqual(refused.exception.code, "runtime_config_invalid")
                    self.assertEqual(refused.exception.violations, violations)
                    self.assertFalse(receipts.exists())


if __name__ == "__main__":
    unittest.main()
