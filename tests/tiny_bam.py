"""A tiny real BAM, the offline Runs' default ``aligned_bam`` (#414).

The Run's measurement runs its real probed samtools over this file, so its bytes
are a valid sorted BAM: built once here with pysam, deterministically. Its header
names the sequences the offline VCFs use; twelve 100-base reads cover
``NC_000019.10:44908684`` (the offline APOE target) twelve deep, so a 10-read
minimum is met there, below it beside the reads and zero elsewhere.
"""

from __future__ import annotations

import os
import tempfile
from functools import cache

SEQUENCES = (("NC_000019.10", 58_617_616), ("NC_012920.1", 16_569))
COVERED_POSITION = 44_908_684
READ_COUNT = 12
READ_LENGTH = 100
# The reads' first base: the covered position is each read's 51st base, onward.
FIRST_READ_START = COVERED_POSITION - 50


@cache
def tiny_bam() -> bytes:
    import pysam

    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in SEQUENCES],
    }
    descriptor, path = tempfile.mkstemp(suffix=".bam")
    os.close(descriptor)
    try:
        with pysam.AlignmentFile(path, "wb", header=header) as written:
            for index in range(READ_COUNT):
                read = pysam.AlignedSegment(written.header)
                read.query_name = f"read-{index:02d}"
                read.reference_id = 0
                # 0-based: the read starts `index` bases after the first start.
                read.reference_start = FIRST_READ_START - 1 + index
                read.query_sequence = "A" * READ_LENGTH
                read.flag = 0
                read.mapping_quality = 60
                read.cigarstring = f"{READ_LENGTH}M"
                read.query_qualities = pysam.qualitystring_to_array("I" * READ_LENGTH)
                written.write(read)
        with open(path, "rb") as built:
            return built.read()
    finally:
        os.unlink(path)


__all__ = [
    "COVERED_POSITION",
    "FIRST_READ_START",
    "READ_COUNT",
    "READ_LENGTH",
    "SEQUENCES",
    "tiny_bam",
]
