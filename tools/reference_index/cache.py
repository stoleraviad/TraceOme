"""The tool's local, git-ignored cache: fetched files with their receipts, and records.

Layout under the root: `catalogues/`, `bundles/galaxy/<brc|vgp>/<accession>/`,
`ncbi/<accession>/`, `candidates/<accession>.json` (replaced as fits are attached) and
`fit/<accession>/<candidate>-<sha256 prefix>.json` (written once). Each fetched file keeps
its receipt beside it (`<file>.fetched.json`). Records are canonical JSON, so a saved fit
report's file digest is the digest `attach_fit` records.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from bio_neb.content import canonical_json
from tools.reference_index.fit import FitReport

from .contracts import CandidateList, Fetched

RECEIPT_SUFFIX = ".fetched.json"


class ReferenceIndexCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def keep_receipt(self, fetched: Fetched) -> None:
        """Write a fetched file's receipt beside it."""
        Path(fetched.destination + RECEIPT_SUFFIX).write_text(
            canonical_json(fetched.model_dump(mode="json"))
        )

    def forget(self, path: Path) -> None:
        """Remove a fetched file that was refused, with its receipt."""
        path.unlink(missing_ok=True)
        Path(str(path) + RECEIPT_SUFFIX).unlink(missing_ok=True)

    def receipts_in(self, *folders: Path) -> tuple[Fetched, ...]:
        """The receipts of the files fetched into ``folders``, by file name."""
        return tuple(
            Fetched.model_validate_json(path.read_bytes())
            for folder in folders
            for path in sorted(folder.glob("*" + RECEIPT_SUFFIX))
        )

    def save_fit(self, report: FitReport) -> Path:
        content = canonical_json(report.model_dump(mode="json")).encode()
        digest = hashlib.sha256(content).hexdigest()
        path = (
            self.root
            / "fit"
            / report.requested_accession
            / f"{_slug(report.candidate_id)}-{digest[:12]}.json"
        )
        if path.exists() and path.read_bytes() == content:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            target.write(content)
        return path

    def load_fit(self, path: Path) -> FitReport:
        return FitReport.model_validate_json(path.read_bytes())

    def load_fit_by_digest(
        self, accession: str, candidate_id: str, sha256: str
    ) -> FitReport:
        """The saved report a `FitSummary` names, by the digest in its file name."""
        return self.load_fit(
            self.root / "fit" / accession / f"{_slug(candidate_id)}-{sha256[:12]}.json"
        )

    def save_candidates(self, listed: CandidateList) -> Path:
        path = self.root / "candidates" / f"{listed.requested_accession}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".json.part")
        partial.write_text(canonical_json(listed.model_dump(mode="json")))
        os.replace(partial, path)
        return path

    def load_candidates(self, accession: str) -> CandidateList:
        path = self.root / "candidates" / f"{accession}.json"
        return CandidateList.model_validate_json(path.read_bytes())


def _slug(candidate_id: str) -> str:
    return candidate_id.replace(":", "_").replace("/", "_")
