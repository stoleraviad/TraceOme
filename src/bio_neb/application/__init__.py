"""Public Run interfaces, resolved lazily to keep shared facts independently importable."""

from importlib import import_module

_EXPORTS = {
    "AnswerReview": "bio_neb.application.advancement",
    "BeginRun": "bio_neb.application.advancement",
    "ExecutionReviewAnswer": "bio_neb.application.advancement",
    "ModelAvailabilityReviewAnswer": "bio_neb.application.advancement",
    "ObserveRun": "bio_neb.application.queries",
    "PlanningReviewAnswer": "bio_neb.application.advancement",
    "RefusalDisposition": "bio_neb.agentic_journey.refusal_disposition",
    "ReviewAnswerRefused": "bio_neb.application.advancement",
    "ResumeRun": "bio_neb.application.advancement",
    "RunAdvancementRefused": "bio_neb.application.advancement",
    "advance": "bio_neb.application.advancement",
    "observe": "bio_neb.application.queries",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public operation only when requested, never while importing its facts."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
