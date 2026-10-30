"""Reference-index spike: find prebuilt BWA indexes for an assembly and check their fit.

Developer tool lane (map #307, lane Rf). `src/` never imports this package.
"""

from .cache import ReferenceIndexCache
from .catalogue import attach_fit, carry_fits, find
from .contracts import Candidate, CandidateList, ToolIdentity
from .fetch import Fetcher, FetchFailed, FetchRefused, ncbi_md5
from .local_tools import LocalTools, ToolFailed

__all__ = [
    "Candidate",
    "CandidateList",
    "FetchFailed",
    "FetchRefused",
    "Fetcher",
    "LocalTools",
    "ReferenceIndexCache",
    "ToolFailed",
    "ToolIdentity",
    "attach_fit",
    "carry_fits",
    "find",
    "ncbi_md5",
]
