"""Shared fixtures of the reference-index tests (#319, #330, #332, #385): FASTA writing,
#319's local install and #332's synthetic pair and choice record."""

from __future__ import annotations

import os
import pwd
import random
from pathlib import Path

from tests.prebuilt_index_support import write_fasta
from tools.reference_index.contracts import Choice, SnvToMove
from tools.reference_index.local_tools import DEFAULT_PREFIX, LocalTools

# Moving SNVs to another assembly. The pair is synthetic: A holds five 1,000-base
# segments; B holds each one the way one result needs, so each expected place on B follows
# from how B was built, not from the tool. The install is found from the account's home:
# the batch's Run sandbox replaces HOME (ADR 0041), and these tests only run the
# executables, writing nothing outside their temporary directories.
INSTALL = LocalTools(
    Path(pwd.getpwuid(os.getuid()).pw_dir) / DEFAULT_PREFIX.relative_to(Path.home())
)
INSTALLED = (
    INSTALL.executable("bwa").exists() and INSTALL.executable("samtools").exists()
)
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _reverse_complement(sequence):
    return "".join(_COMPLEMENT[base] for base in reversed(sequence))


class SyntheticPair:
    """`chrA` on A: spacer, then segments 0-4 with a spacer after each, the SNV of segment
    `k` at its offset 500. On B (each contig flanked by 400 random bases):

    - `same`: segment 0 less its bases 400-401, so the SNV sits 2 bases earlier: 400 + 499 - 2
      + 1 = 899 (1-based);
    - `flipped`: segment 1 reverse-complemented: 400 + (999 - 500) + 1 = 900;
    - `changed`: segment 2 with its SNV base replaced by the ALT: 901;
    - `twice`: segment 3 at 400 and again at 1,800: 901 and 2,301;
    - segment 4 is nowhere on B;
    - `half`: segment 5 only up to 20 bases past its SNV, then random bases, so the flanks
      align over only 221 of their 401 bases: 901, low-scoring;
    - `lacks`: segment 6 without its SNV base (between a C and a T on A).
    """

    SEGMENT, SITE, SPACER, FILLER = 1000, 500, 300, 400

    def __init__(self, root):
        rng = random.Random(332)

        def dna(length):
            return "".join(rng.choice("ACGT") for _ in range(length))

        segments = [dna(self.SEGMENT) for _ in range(7)]
        segments[6] = (
            segments[6][: self.SITE - 1] + "CGT" + segments[6][self.SITE + 2 :]
        )
        spacers = [dna(self.SPACER) for _ in range(8)]
        chr_a = spacers[0] + "".join(s + spacers[k + 1] for k, s in enumerate(segments))
        self.positions = [
            self.SPACER + k * (self.SEGMENT + self.SPACER) + self.SITE + 1
            for k in range(7)
        ]
        self.refs = [segment[self.SITE] for segment in segments]
        self.alts = ["C" if ref == "A" else "A" for ref in self.refs]
        same = segments[0][:400] + segments[0][402:]
        changed = segments[2][: self.SITE] + self.alts[2] + segments[2][self.SITE + 1 :]
        twice = segments[3] + dna(self.FILLER) + segments[3]
        b = [
            ("same", same),
            ("flipped", _reverse_complement(segments[1])),
            ("changed", changed),
            ("twice", twice),
            ("half", segments[5][: self.SITE + 21] + dna(self.FILLER)),
            ("lacks", segments[6][: self.SITE] + segments[6][self.SITE + 1 :]),
        ]
        self.source = root / "a.fa"
        self.target = root / "b.fa"
        write_fasta(self.source, [("chrA", "", chr_a)], width=60)
        write_fasta(
            self.target,
            [
                (name, "", dna(self.FILLER) + body + dna(self.FILLER))
                for name, body in b
            ],
            width=60,
        )
        INSTALL.run("bwa", "index", str(self.target))
        INSTALL.faidx(self.source)
        INSTALL.faidx(self.target)

    def snv(self, k, label, **fields):
        return SnvToMove(
            label=label,
            assembly="synthetic A",
            contig="chrA",
            position=self.positions[k],
            ref=self.refs[k],
            alt=self.alts[k],
            **fields,
        )


def choice_record(requested, target, bundle_digest, snvs):
    """A taken newer-prebuilt choice in #330's layout (`choices/GCF_000002315.4.json`,
    made 2026-10-15), with synthetic accessions, digests and SNVs."""
    return Choice.model_validate(
        {
            "requested": {
                "assembly_accession": requested,
                "assembly_level": "Chromosome",
                "assembly_name": "SynA",
                "assembly_status": "suppressed",
                "current_accession": requested,
                "ncbi_tax_id": 99,
                "organism_name": "Synthetic species",
                "paired_accession": None,
                "refseq_category": None,
                "release_date": "2015-12-16",
            },
            "decision": "taken",
            "option": {
                "option_id": f"prebuilt:galaxy:brc:{target}",
                "action": "use_newer_prebuilt",
                "assembly_accession": target,
                "assembly_name": "SynB",
                "candidate_id": f"galaxy:brc:{target}",
                "cost": {
                    "work": "fetch_prebuilt",
                    "download_bytes": 1,
                    "basis": "estimated_from_fai",
                    "detail": "synthetic",
                },
                "fit": {
                    "sequence": "newer_assembly",
                    "detail": "synthetic",
                    "snv_chain": ["move: #332"],
                },
            },
            "card_sha256": "c" * 64,
            "bundle_digest": bundle_digest,
            "fit_report_sha256": "f" * 64,
            "snvs_to_move": snvs,
        }
    )
