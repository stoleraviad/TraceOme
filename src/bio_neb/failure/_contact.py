"""The model contact classifier: one verdict per failed model contact.

Every reader of a failed contact reads this verdict: the live contact loop, the
turn's resend note, the availability Review and reopen's replay of the window
(owner, #439 B3; #443). It reads only what is known about the contact: how it
failed, its HTTP status, Retry-After, its ordinal, the time already waited and
whether its request follows a stall.
"""

from dataclasses import dataclass
from typing import Literal

from bio_neb.failure._faults import (
    FailureClass,
    ModelContactPolicy,
    ModelContactStalled,
    ModelContextExhausted,
    ModelResponseMalformed,
)

# How one contact failed. ``stalled``: no reply within the stall limit, cut off by
# the loop; ``connection``: the transport lost the request with its completion
# unknown; ``not_sent``: refused before any byte left the process; ``unexpected``:
# any other exception, our fault.
ModelContactFailure = Literal[
    "http", "stalled", "connection", "malformed", "not_sent", "unexpected"
]
ModelContactDisposition = Literal[
    "automatic_retry_scheduled",
    "stall_noted",
    "rejection_shown",
    "availability_review_requested",
    "terminal_failure",
]
# Nebius answers its rate limit with 529, so it is sent again like a 429.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
# A request whose reply never arrived: sending it again is a resend the model is
# told about in its next note (ADR 0044).
_TIMED_OUT_STATUSES = frozenset({408, 504})


@dataclass(frozen=True)
class ContactVerdict:
    """What one failed contact means and what happens next.

    ``automatic_retry_scheduled`` sends the same request again after
    ``delay_seconds``; ``stall_noted`` ends the request and the next one carries
    the stall note; ``rejection_shown`` ends a refused request and the next one
    carries the provider's own words (#477 G1.5); ``availability_review_requested`` asks a person for the
    ``exhaustion_reason``; ``terminal_failure`` is no availability fault.
    ``resend`` marks a retry of a request whose reply never arrived.
    """

    failure_class: FailureClass
    failure_reason: str | None
    completion_certainty: Literal["response_received", "completion_unknown", "not_sent"]
    policy_disposition: ModelContactDisposition
    delay_seconds: float | None = None
    delay_source: str | None = None
    exhaustion_reason: str | None = None
    resend: bool = False


def model_contact_failure(error: BaseException) -> ModelContactFailure:
    """How a raised error that carries no HTTP status failed its contact.

    The Agent edge reads the framework's own errors (an HTTP status, a wrapped
    connection error) before it asks; this whitelists transport faults.
    """
    import httpx
    from openai import APIConnectionError

    if isinstance(error, ModelContactStalled):
        return "stalled"
    if isinstance(error, ModelResponseMalformed):
        return "malformed"
    if isinstance(error, ModelContextExhausted):
        return "not_sent"
    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            APIConnectionError,
        ),
    ):
        return "connection"
    return "unexpected"


def model_contact_reading(
    failure: ModelContactFailure, http_status_code: int | None = None
) -> tuple[FailureClass, str | None, str]:
    """A failure's class, its reason and whether the provider answered.

    The reason reads the HTTP status only, never provider-controlled content.
    """
    if failure == "http":
        if http_status_code is None:
            return "rejected", None, "response_received"
        return (
            "transient" if http_status_code in _TRANSIENT_STATUSES else "rejected",
            _http_reason(http_status_code),
            "response_received",
        )
    return {
        "stalled": ("transient", "contact_stalled", "completion_unknown"),
        "connection": ("transient", "non_http_failure", "completion_unknown"),
        "malformed": ("semantic", "malformed_response", "response_received"),
        "not_sent": ("exhausted", "context_window_exhausted", "not_sent"),
        "unexpected": ("invariant", "non_http_failure", "completion_unknown"),
    }[failure]


_STATEMENT_LIMIT = 2000


def contact_fault_key(
    failure: str, status: int | None = None, body_sha256: str | None = None
) -> str:
    """A failed model contact's cause (G4.1; Spec r2 SP4): the status and body
    of a retained answer, else the status, else how the contact failed."""
    if body_sha256 is not None and status is not None:
        return model_fault_key(status, body_sha256)
    if status is not None:
        return f"model endpoint {status}"
    return f"model endpoint: {failure}"


# Failures whose cause can hold every item: each asks the availability Review.
_HELD_FAILURES = {
    "http": None,
    "connection": "automatic_retry_exhausted",
    "stalled": "contact_stalled",
    "malformed": "malformed_response",
}


def model_fault_key(status: int, body_sha256: str) -> str:
    """One model endpoint fault's key (owner, #476 alignment G4.1): its status and
    the exact body it answered with. Every item failing with the same key shares
    one Review."""
    return f"model endpoint {status}: {body_sha256}"


def provider_statement(body: bytes) -> str | None:
    """The provider's own words in an error answer's body, as text (#477 G4.2).

    Read, never classified (ADR 0049): the body's text with its whitespace runs
    as one space, cut at a bound; None for an empty body."""
    text = " ".join(body.decode("utf-8", errors="replace").split())
    if not text:
        return None
    if len(text) > _STATEMENT_LIMIT:
        text = text[: _STATEMENT_LIMIT - 1] + "\u2026"
    return text


def _http_reason(status: int) -> str:
    if status == 401:
        return "authentication_rejected"
    if status == 403:
        return "authorization_rejected"
    if status == 408:
        return "request_timeout"
    if status == 413:
        return "request_too_large"
    if status in {429, 529}:
        return "rate_limited"
    if 400 <= status < 500:
        return "request_rejected"
    if 500 <= status < 600:
        return "provider_failure"
    return "http_failure"


def model_contact_verdict(
    policy: ModelContactPolicy,
    *,
    failure: ModelContactFailure,
    contact: int,
    waited: float,
    jitter: float,
    http_status_code: int | None = None,
    retry_after_disposition: str | None = None,
    retry_after_seconds: float | None = None,
    malformed_responses: int = 0,
    after_stall: bool = False,
    rejection_key: str | None = None,
    shown_key: str | None = None,
    rejoins: bool = False,
) -> ContactVerdict:
    """The one verdict on a failed contact (owner, #443 readiness).

    A stall ends its request with a note, and a stall of the request that
    carries the note asks the Review. Any other short failure is sent once more
    within the window, then asks. A request the endpoint refused with any 4xx and
    words in its body (``rejection_key``) ends with the provider's words for the
    model, unless the request carrying them (``shown_key``) was refused the same
    way: then it asks (owner, #476 alignment G1.5 B). An HTTP failure whose cause
    already has an open Review (``rejoins``) asks at once and joins it (owner
    amendment, #477 6483312280). Any other refusal asks at once; a malformed
    reply is sent again at once, and a second one asks.
    """
    failure_class, reason, completion = model_contact_reading(failure, http_status_code)
    verdict = {
        "failure_class": failure_class,
        "failure_reason": reason,
        "completion_certainty": completion,
    }

    def review(exhaustion: str) -> ContactVerdict:
        return ContactVerdict(
            **verdict,
            policy_disposition="availability_review_requested",
            exhaustion_reason=exhaustion,
        )

    if rejoins and failure in _HELD_FAILURES:
        return review(
            _HELD_FAILURES[failure]
            or (
                "request_rejected"
                if failure_class == "rejected"
                else "automatic_retry_exhausted"
            )
        )
    if (
        failure == "http"
        and rejection_key is not None
        and http_status_code is not None
        and 400 <= http_status_code < 500
    ):
        if rejection_key != shown_key:
            return ContactVerdict(**verdict, policy_disposition="rejection_shown")
        return review("request_rejected")
    if failure == "stalled":
        if after_stall:
            return review("contact_stalled")
        return ContactVerdict(**verdict, policy_disposition="stall_noted")
    if failure == "malformed":
        if malformed_responses > 1:
            return review("malformed_response")
        # The window's sends bound this retry too.
        if contact >= policy.contacts:
            return review("automatic_retry_exhausted")
        return ContactVerdict(
            **verdict,
            policy_disposition="automatic_retry_scheduled",
            delay_seconds=0.0,
            delay_source="malformed_response",
        )
    if failure_class == "rejected":
        return review("request_rejected")
    if failure_class != "transient":
        return ContactVerdict(**verdict, policy_disposition="terminal_failure")
    if contact >= policy.contacts:
        return review("automatic_retry_exhausted")
    if retry_after_disposition == "over_cap":
        return review("retry_after_over_cap")
    source = "retry_after" if retry_after_seconds is not None else "exponential_jitter"
    delay = (
        retry_after_seconds
        if retry_after_seconds is not None
        else min(
            policy.max_delay_seconds,
            policy.base_delay_seconds * 2 ** min(contact - 1, 30),
        )
        * jitter
    )
    if waited + delay > policy.max_wait_seconds:
        return review(
            "retry_after_over_cap"
            if source == "retry_after"
            else "wait_budget_exhausted"
        )
    return ContactVerdict(
        **verdict,
        policy_disposition="automatic_retry_scheduled",
        delay_seconds=delay,
        delay_source=source,
        resend=failure == "connection" or http_status_code in _TIMED_OUT_STATUSES,
    )


def model_retry_after(headers, *, now: float) -> tuple[str, float | None]:
    """Parse only safe Retry-After seconds or HTTP-date; retain no raw headers."""
    from collections.abc import Mapping
    from email.utils import parsedate_to_datetime

    if not isinstance(headers, Mapping):
        return "missing", None
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return "missing", None
    if not isinstance(value, str):
        return "unusable", None
    value = value.strip()
    if value and value.isascii() and value.isdigit():
        # Bound parsing before int conversion, including hostile enormous headers.
        if len(value.lstrip("0")) > 9:
            return "over_cap", None
        return "valid", float(int(value.lstrip("0") or "0"))
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return "unusable", None
        return "valid", max(0.0, date.timestamp() - now)
    except (TypeError, ValueError, OverflowError):
        return "unusable", None
