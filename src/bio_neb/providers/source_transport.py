"""The one transport family for public source GETs.

NCBI's and EVA's metadata GETs, a reference download's GETs and a paper's all leave
through `SourceTransport`: one no-redirect opener, one user agent, one host check, a
timeout per socket operation and the byte bound of the surface whose contact runs
(`source_contacts.response_byte_bound`). Inside a Run's source scope each GET is one
physical contact in the Pass's contact journal (`source_physical_attempt`); outside one
nothing is recorded. An HTTP status is an answer, a lost connection a transient fault,
and nothing is retried here: the failure module retries on the surface (ADR 0031).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from typing import Any, Literal

from bio_neb.agentic_journey.source_contacts import (
    observe_source_transport,
    response_over_bound,
    source_physical_attempt,
)
from bio_neb.failure import ProviderError

USER_AGENT = "bio-neb-clean-agentic-journey/0.1"
TIMEOUT_SECONDS = 15.0
"""Per socket operation, not per answer: a long download fails only when it stalls
(owner, #417 Q6)."""


@dataclass(frozen=True)
class SourceHost:
    accept: str | None
    path_prefixes: tuple[str, ...] = ("/",)
    # Galaxy's datacache serves its index files over plain HTTP only.
    plain_http: bool = False


HOSTS: Mapping[str, SourceHost] = {
    "api.ncbi.nlm.nih.gov": SourceHost("application/json"),
    "ftp.ncbi.nlm.nih.gov": SourceHost("text/plain"),
    "eutils.ncbi.nlm.nih.gov": SourceHost(None, ("/entrez/eutils/",)),
    # EVA's web services and Europe PMC's REST service.
    "www.ebi.ac.uk": SourceHost(
        "application/json", ("/eva/webservices/", "/europepmc/webservices/rest/")
    ),
    "usegalaxy.org": SourceHost(None, ("/api/tool_data/",)),
    "datacache.galaxyproject.org": SourceHost(None, plain_http=True),
    # The PMC open-data bucket: its listing and its article files.
    "pmc-oa-opendata.s3.amazonaws.com": SourceHost(None),
}
"""Every host a source GET may reach, with its default Accept and its path."""


class SourceUrlRefused(ProviderError):
    def __init__(self, url: str) -> None:
        super().__init__("rejected", "The URL is outside the source hosts.")
        self.url = url


class SourceContactLost(ProviderError):
    def __init__(self, error: BaseException) -> None:
        super().__init__(
            "transient", f"The source request failed: {type(error).__name__}."
        )


class SourceAnswerTooLarge(ProviderError):
    def __init__(self) -> None:
        super().__init__("rejected", "The source's answer exceeded the byte bound.")


class AnsweredStatus(Exception):
    """An HTTP error status that failed its contact; its answer, a 404's body
    included, stays readable."""

    def __init__(self, answer: SourceAnswer) -> None:
        super().__init__(f"HTTP {answer.status}")
        self.answer = answer


@dataclass(frozen=True)
class SourceAnswer:
    url: str
    status: int
    body: bytes
    # Read in any case, as HTTP names its headers.
    headers: Message
    observed_at: datetime
    elapsed_time_ms: int


def _case_insensitive(headers: Any) -> Message:
    """An answer's headers, looked up in any case whatever mapping carried them."""
    kept = Message()
    for name, value in (headers or {}).items():
        kept[name] = value
    return kept


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        del req, fp, code, msg, headers, newurl
        return None


Opener = Callable[[urllib.request.Request, float], Any]


class SourceTransport:
    """One bounded HTTP exchange per call on the hosts it is given."""

    def __init__(
        self,
        hosts: Collection[str] = tuple(HOSTS),
        *,
        timeout_seconds: float = TIMEOUT_SECONDS,
        opener: Opener | None = None,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 30:
            raise ValueError("a source timeout must be in (0, 30]")
        if not set(hosts) <= set(HOSTS):
            raise ValueError("a source transport serves only the source hosts")
        self._hosts = frozenset(hosts)
        self._timeout_seconds = timeout_seconds
        self._opener = opener
        # The one no-redirect opener; its `open` is looked up when a request is sent.
        self._director = urllib.request.build_opener(_NoRedirect())
        self.contact_count = 0

    def exchange(
        self,
        url: str,
        *,
        max_bytes: int,
        method: Literal["GET", "HEAD"] = "GET",
        accept: str | None = None,
        status_fails: bool = False,
    ) -> SourceAnswer:
        """One GET read within ``max_bytes``, or one HEAD; the status is the answer's.

        With ``status_fails``, as NCBI's view keeps it, an HTTP error status ends the
        physical contact failed (its ``error_type`` the ``HTTPError``) and is raised as
        `AnsweredStatus`, its answer still readable; otherwise, as EVA's view keeps
        it, the status is the answer of a contact that succeeded.
        """
        if max_bytes < 0 or (method == "GET" and max_bytes < 1):
            raise ValueError("a source byte bound must be positive")
        request = self._request(url, method, accept, {})
        started = time.monotonic()
        refused: urllib.error.HTTPError | None = None
        try:
            with self._lost_contact() as fault, self._contact(url, method):
                try:
                    with fault(), self._send(request) as response:
                        status, headers = int(response.status), response.headers
                        body = response.read(max_bytes + 1) if max_bytes else b""
                except urllib.error.HTTPError as error:
                    status, headers = error.code, error.headers
                    # The error body is read in the contact, a stall classified.
                    with fault():
                        body = error.read(max_bytes + 1) if max_bytes else b""
                    if status_fails:
                        refused = error
                        raise
        except urllib.error.HTTPError as error:
            if error is not refused:
                raise
        if len(body) > max_bytes:
            # The surface's bound is the one asked for; record its breach.
            response_over_bound(len(body))
            raise SourceAnswerTooLarge()
        answer = SourceAnswer(
            url=url,
            status=status,
            body=body,
            headers=_case_insensitive(headers),
            observed_at=datetime.now(UTC),
            elapsed_time_ms=max(0, round((time.monotonic() - started) * 1000)),
        )
        if refused is not None:
            raise AnsweredStatus(answer) from refused
        return answer

    @contextmanager
    def stream(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> Iterator[Any]:
        """One GET whose open response the caller reads in blocks, as a download does.

        An HTTP status is the response's; a fault while reading is the caller's and
        ends the contact as failed.
        """
        request = self._request(url, "GET", None, headers or {})
        with self._lost_contact() as fault, self._contact(url, "GET"):
            try:
                with fault():
                    response = self._send(request)
            except urllib.error.HTTPError as error:
                response = error
            # The body is read inside the physical contact, so its terminal is the
            # read's; the caller's own fault reaches it unchanged.
            with response as opened:
                yield opened

    def _request(
        self,
        url: str,
        method: str,
        accept: str | None,
        headers: Mapping[str, str],
    ) -> urllib.request.Request:
        observe_source_transport()
        parsed = urllib.parse.urlsplit(url)
        host = HOSTS.get(parsed.hostname or "")
        if (
            host is None
            or parsed.hostname not in self._hosts
            or parsed.scheme
            not in ({"https", "http"} if host.plain_http else {"https"})
            or not parsed.path.startswith(host.path_prefixes)
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise SourceUrlRefused(url)
        sent = {"User-Agent": USER_AGENT, **headers}
        if accept or host.accept:
            sent["Accept"] = accept or host.accept
        return urllib.request.Request(url, method=method, headers=sent)

    @contextmanager
    def _contact(self, url: str, method: Literal["GET", "HEAD"]) -> Iterator[None]:
        self.contact_count += 1
        with source_physical_attempt(url, method=method):
            yield

    def _send(self, request: urllib.request.Request) -> Any:
        send = self._opener or self._director.open
        return send(request, timeout=self._timeout_seconds)

    @contextmanager
    def _lost_contact(self) -> Iterator[Callable[[], AbstractContextManager[None]]]:
        """Classify a lost connection after its contact records the actual fault; a
        local evidence-write failure is not a contact fault and propagates unchanged."""
        faults: list[BaseException] = []

        @contextmanager
        def fault() -> Iterator[None]:
            try:
                yield
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                faults.append(e)
                raise

        try:
            yield fault
        except BaseException as error:
            if faults and error is faults[-1]:
                raise SourceContactLost(error) from error
            raise


__all__ = [
    "HOSTS",
    "AnsweredStatus",
    "TIMEOUT_SECONDS",
    "USER_AGENT",
    "SourceAnswer",
    "SourceAnswerTooLarge",
    "SourceContactLost",
    "SourceHost",
    "SourceTransport",
    "SourceUrlRefused",
]
