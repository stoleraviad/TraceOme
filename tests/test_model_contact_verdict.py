"""The model contact classifier: one verdict per failed contact (#443).

Owner, #439 B3 and #443 readiness (2026-10-24): budgets are per call. A send with
no reply after 180 s is a stall: the next request carries a note, and a second
stall in a row asks the availability Review. Every short failure (no connection,
502, 503, a fast 504, 408, 429, and 529, Nebius's rate limit: owner, #420 D2)
retries once within 60 s, then asks; a 429 or 529 follows a Retry-After of 60 s
or less.
"""

import unittest

import httpx
from openai import APIConnectionError

from bio_neb.failure import (
    ContactVerdict,
    ModelContactPolicy,
    ModelContactStalled,
    ModelContextExhausted,
    ModelResponseMalformed,
    model_contact_failure,
    model_contact_verdict,
)


def _verdict(failure, **facts):
    facts.setdefault("contact", 1)
    facts.setdefault("waited", 0.0)
    facts.setdefault("jitter", 0.5)
    policy = facts.pop("policy", ModelContactPolicy())
    return model_contact_verdict(policy, failure=failure, **facts)


def _retry(reason, delay, *, source="exponential_jitter", resend=False, **kw):
    return ContactVerdict(
        failure_class=kw.get("failure_class", "transient"),
        failure_reason=reason,
        completion_certainty=kw.get("completion", "response_received"),
        policy_disposition="automatic_retry_scheduled",
        delay_seconds=delay,
        delay_source=source,
        resend=resend,
    )


def _review(reason, exhaustion, **kw):
    return ContactVerdict(
        failure_class=kw.get("failure_class", "transient"),
        failure_reason=reason,
        completion_certainty=kw.get("completion", "response_received"),
        policy_disposition="availability_review_requested",
        exhaustion_reason=exhaustion,
    )


class ModelContactVerdictTests(unittest.TestCase):
    def test_the_default_window_is_two_sends_of_180_s_and_one_wait_of_60_s(self):
        policy = ModelContactPolicy()
        self.assertEqual(
            (
                policy.contacts,
                policy.stall_seconds,
                policy.max_delay_seconds,
                policy.max_wait_seconds,
            ),
            (2, 180, 60, 60),
        )

    def test_every_contact_fact_has_one_verdict(self):
        unknown = {"completion": "completion_unknown"}
        rows = (
            # A stall: no reply after 180 s. Its request ends; the next carries
            # the note; a stall of that request asks the Review.
            (
                "stalled",
                {},
                ContactVerdict(
                    failure_class="transient",
                    failure_reason="contact_stalled",
                    completion_certainty="completion_unknown",
                    policy_disposition="stall_noted",
                ),
            ),
            (
                "stalled",
                {"after_stall": True},
                _review("contact_stalled", "contact_stalled", **unknown),
            ),
            # A stall on a request's second send is still noted, not resent.
            (
                "stalled",
                {"contact": 2},
                ContactVerdict(
                    failure_class="transient",
                    failure_reason="contact_stalled",
                    completion_certainty="completion_unknown",
                    policy_disposition="stall_noted",
                ),
            ),
            # Short failures: one retry within 60 s, then the Review.
            (
                "http",
                {"http_status_code": 504},
                _retry("provider_failure", 15.0, resend=True),
            ),
            (
                "http",
                {"http_status_code": 504, "contact": 2},
                _review("provider_failure", "automatic_retry_exhausted"),
            ),
            ("http", {"http_status_code": 502}, _retry("provider_failure", 15.0)),
            (
                "http",
                {"http_status_code": 500, "jitter": 1.0},
                _retry("provider_failure", 30.0),
            ),
            (
                "http",
                {"http_status_code": 503, "contact": 2},
                _review("provider_failure", "automatic_retry_exhausted"),
            ),
            (
                "http",
                {"http_status_code": 408},
                _retry("request_timeout", 15.0, resend=True),
            ),
            (
                "http",
                {
                    "http_status_code": 429,
                    "retry_after_disposition": "valid",
                    "retry_after_seconds": 20.0,
                },
                _retry("rate_limited", 20.0, source="retry_after"),
            ),
            (
                "http",
                {
                    "http_status_code": 429,
                    "retry_after_disposition": "valid",
                    "retry_after_seconds": 61.0,
                },
                _review("rate_limited", "retry_after_over_cap"),
            ),
            (
                "http",
                {"http_status_code": 429, "retry_after_disposition": "over_cap"},
                _review("rate_limited", "retry_after_over_cap"),
            ),
            (
                "connection",
                {},
                _retry("non_http_failure", 15.0, resend=True, **unknown),
            ),
            (
                "connection",
                {"contact": 2},
                _review("non_http_failure", "automatic_retry_exhausted", **unknown),
            ),
            # The provider refused the request: a person decides.
            (
                "http",
                {"http_status_code": 401},
                _review(
                    "authentication_rejected",
                    "request_rejected",
                    failure_class="rejected",
                ),
            ),
            (
                "http",
                {"http_status_code": 400},
                _review(
                    "request_rejected", "request_rejected", failure_class="rejected"
                ),
            ),
            (
                "http",
                {"http_status_code": 501},
                _review(
                    "provider_failure", "request_rejected", failure_class="rejected"
                ),
            ),
            # A malformed reply is sent again at once, a second one asks.
            (
                "malformed",
                {"malformed_responses": 1},
                _retry(
                    "malformed_response",
                    0.0,
                    source="malformed_response",
                    failure_class="semantic",
                ),
            ),
            (
                "malformed",
                {"malformed_responses": 2, "contact": 2},
                _review(
                    "malformed_response", "malformed_response", failure_class="semantic"
                ),
            ),
            # The window's two sends bound a malformed reply too.
            (
                "malformed",
                {"malformed_responses": 1, "contact": 2},
                _review(
                    "malformed_response",
                    "automatic_retry_exhausted",
                    failure_class="semantic",
                ),
            ),
            # Never sent, or our own fault: no retry and no availability Review.
            (
                "not_sent",
                {},
                ContactVerdict(
                    failure_class="exhausted",
                    failure_reason="context_window_exhausted",
                    completion_certainty="not_sent",
                    policy_disposition="terminal_failure",
                ),
            ),
            (
                "unexpected",
                {},
                ContactVerdict(
                    failure_class="invariant",
                    failure_reason="non_http_failure",
                    completion_certainty="completion_unknown",
                    policy_disposition="terminal_failure",
                ),
            ),
        )
        for failure, facts, expected in rows:
            with self.subTest(failure=failure, **facts):
                self.assertEqual(_verdict(failure, **facts), expected)

    def test_a_refused_request_is_shown_to_the_model_once_then_asks(self):
        """Owner, #476 alignment G1.5 B: a model endpoint 4xx's body goes to the
        model once, as a failed turn; the same status and body on the next
        request asks the one Run-level Review; no word list reads the body."""
        from bio_neb.failure import model_fault_key

        key = model_fault_key(400, "a" * 64)
        shown = ContactVerdict(
            failure_class="rejected",
            failure_reason="request_rejected",
            completion_certainty="response_received",
            policy_disposition="rejection_shown",
        )
        self.assertEqual(
            _verdict("http", http_status_code=400, rejection_key=key), shown
        )
        # Another body after a shown refusal is shown too.
        self.assertEqual(
            _verdict(
                "http",
                http_status_code=400,
                rejection_key=key,
                shown_key=model_fault_key(400, "b" * 64),
            ),
            shown,
        )
        rejected = _review(
            "request_rejected", "request_rejected", failure_class="rejected"
        )
        self.assertEqual(
            _verdict("http", http_status_code=400, rejection_key=key, shown_key=key),
            rejected,
        )
        # Every 4xx with words is shown once, a 408 or 429 too (Spec r1 SP1).
        for status, reason in ((408, "request_timeout"), (429, "rate_limited")):
            with self.subTest(status=status):
                status_key = model_fault_key(status, "a" * 64)
                self.assertEqual(
                    _verdict("http", http_status_code=status, rejection_key=status_key),
                    ContactVerdict(
                        failure_class="transient",
                        failure_reason=reason,
                        completion_certainty="response_received",
                        policy_disposition="rejection_shown",
                    ),
                )
                self.assertEqual(
                    _verdict(
                        "http",
                        http_status_code=status,
                        rejection_key=status_key,
                        shown_key=status_key,
                    ),
                    _review(reason, "request_rejected"),
                )
        # A cause whose Review is still open recurs: any status asks and joins at
        # once, never a fresh retry window.
        self.assertEqual(
            _verdict("http", http_status_code=503, rejoins=True),
            _review("provider_failure", "automatic_retry_exhausted"),
        )
        # No body to show, or a 5xx the provider cannot serve: the Review at once.
        self.assertEqual(_verdict("http", http_status_code=400), rejected)
        self.assertEqual(
            _verdict(
                "http",
                http_status_code=501,
                rejection_key=model_fault_key(501, "a" * 64),
            ),
            _review("provider_failure", "request_rejected", failure_class="rejected"),
        )

    def test_a_529_is_nebius_rate_limit_retried_once_then_asks(self):
        """Owner, #420 D2: "529 is regular rate limit in Nebius". A 529 is sent
        once more after a jittered wait within the window, or after a Retry-After
        of 60 s or less, then asks the availability Review; it is never a
        refusal that asks at once (#420 post-mortem L2)."""
        self.assertEqual(
            _verdict("http", http_status_code=529),
            _retry("rate_limited", 15.0),
        )
        self.assertEqual(
            _verdict(
                "http",
                http_status_code=529,
                retry_after_disposition="valid",
                retry_after_seconds=20.0,
            ),
            _retry("rate_limited", 20.0, source="retry_after"),
        )
        self.assertEqual(
            _verdict(
                "http",
                http_status_code=529,
                retry_after_disposition="valid",
                retry_after_seconds=61.0,
            ),
            _review("rate_limited", "retry_after_over_cap"),
        )
        self.assertEqual(
            _verdict("http", http_status_code=529, contact=2),
            _review("rate_limited", "automatic_retry_exhausted"),
        )

    def test_a_wider_window_stops_at_its_wait_budget(self):
        policy = ModelContactPolicy(contacts=3)
        self.assertEqual(
            _verdict("connection", policy=policy, contact=2, waited=40.0, jitter=1.0),
            _review(
                "non_http_failure",
                "wait_budget_exhausted",
                completion="completion_unknown",
            ),
        )

    def test_a_contact_record_cannot_mix_http_and_transport_evidence(self):
        # A transport failure carries no HTTP status; a record that claims both
        # is refused, never read as one of them.
        from pydantic import ValidationError

        from bio_neb.agentic_journey.attempt_journey import ModelPhysicalContactFailed

        with self.assertRaisesRegex(ValidationError, "non-HTTP contact"):
            ModelPhysicalContactFailed(
                attempt_id="a",
                request_number=1,
                logical_request_event_id="r",
                contact_started_event_id="c",
                contact_ordinal=1,
                failure_class="transient",
                error_type="ConnectionError",
                contact_seconds=1.0,
                http_status_code=401,
                failure_reason="non_http_failure",
                completion_certainty="completion_unknown",
                policy_disposition="automatic_retry_scheduled",
            )

    def test_each_raised_error_names_its_contact_failure(self):
        # The Agent edge reads the framework's HTTP and wrapped errors first.
        cases = (
            (ModelContactStalled(180), "stalled"),
            (httpx.ConnectError("refused"), "connection"),
            (httpx.ReadTimeout("slow"), "connection"),
            (ConnectionResetError(), "connection"),
            (
                APIConnectionError(request=httpx.Request("POST", "https://x")),
                "connection",
            ),
            (ModelResponseMalformed(), "malformed"),
            (
                ModelContextExhausted(
                    input_estimate=1, safety_margin=1, context_window=1
                ),
                "not_sent",
            ),
            (ValueError("ours"), "unexpected"),
        )
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(model_contact_failure(error), expected)


if __name__ == "__main__":
    unittest.main()
