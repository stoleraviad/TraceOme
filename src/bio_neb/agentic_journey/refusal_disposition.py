"""Closed caller dispositions shared by typed domain refusals."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class RefusalDisposition(BaseModel):
    """State exactly what a caller may lawfully do after one refusal.

    ``hold_for_fix``: the failure module recorded the fault and holds the Run;
    fix the code or installation, then ``resume`` the same Run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "hold_for_fix",
        "caller_correction_and_reissue",
        "researcher_review_required",
    ]


__all__ = ["RefusalDisposition"]
