"""Galaxy's datacache view of the source transport family (#365).

The planning model's index Tools read Galaxy's catalogues and a candidate's `.ann` with
one bounded GET each, and each file's size with one HEAD (#464). Each GET the source
answered is reported to the Tool collecting answers (`source_answers.answer_received`,
#458). A redirect or a client status is the source's answer, refused as
``source_answered``; a server status or a lost connection is transient, retried by the
caller's failure surface. Nothing is retried here.
"""

from __future__ import annotations

from bio_neb.agentic_journey.source_answers import answer_received, source_answered
from bio_neb.agentic_journey.source_contacts import response_byte_bound
from bio_neb.failure import ProviderError
from bio_neb.providers.source_transport import (
    AnsweredStatus,
    Opener,
    SourceTransport,
)

DATACACHE_HOST = "datacache.galaxyproject.org"


class GalaxyDatacache:
    def __init__(self, *, opener: Opener | None = None) -> None:
        self._transport = SourceTransport((DATACACHE_HOST,), opener=opener)

    @property
    def contact_count(self) -> int:
        return self._transport.contact_count

    def get(self, url: str) -> bytes:
        try:
            answer = self._transport.exchange(
                url, max_bytes=response_byte_bound(), status_fails=True
            )
        except AnsweredStatus as status:
            code = status.answer.status
            if source_answered(code):
                answer_received(url, code, status.answer.body)
                raise ProviderError(
                    "rejected",
                    f"Galaxy's datacache answered HTTP {code}.",
                    evidence={"source_code": "source_answered", "status": code},
                ) from status
            raise _status_fault(code) from status
        answer_received(url, answer.status, answer.body)
        return answer.body

    def size(self, url: str) -> int:
        """One file's byte size, from a HEAD's Content-Length."""
        try:
            answer = self._transport.exchange(
                url, method="HEAD", max_bytes=0, status_fails=True
            )
        except AnsweredStatus as status:
            raise _status_fault(status.answer.status) from status
        length = answer.headers.get("Content-Length")
        if length is None or not length.isdigit() or int(length) < 1:
            raise ProviderError(
                "semantic", "Galaxy's datacache did not report the file's size."
            )
        return int(length)


def _status_fault(code: int) -> ProviderError:
    return ProviderError(
        "transient" if code == 429 or code >= 500 else "rejected",
        f"Galaxy's datacache returned HTTP {code}.",
        evidence={"source_code": "http_error", "status": code},
    )


__all__ = ["DATACACHE_HOST", "GalaxyDatacache"]
