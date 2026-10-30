"""Registry V2 mechanics for Nebius adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from bio_neb.providers.nebius_common import (
    ExecutionContactObserver,
    NebiusCliError,
    _canonical_line,
    observe_execution_contact_finish,
    observe_execution_contact_start,
)


@dataclass(frozen=True, slots=True)
class RegistryV2Lineage:
    index_digest: str
    linux_amd64_child_digest: str
    base_reference: str | None


_INDEX_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


class _NoRegistryRedirect(HTTPRedirectHandler):
    """Expose redirects so every physical exchange crosses the journal boundary."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class NebiusRegistryV2Resolver:
    """Resolve one exact OCI index and its Linux/amd64 child without retaining credentials."""

    def __init__(
        self,
        *,
        cli_runner: Any,
        profile: str,
        opener: Callable[..., Any] | None = None,
        execution_contact_observer: ExecutionContactObserver | None = None,
    ) -> None:
        self._observer = execution_contact_observer
        self._runner = cli_runner
        self._profile = _canonical_line(profile, "profile")
        self._opener = opener or build_opener(_NoRegistryRedirect()).open

    def __call__(self, tagged_reference: str) -> RegistryV2Lineage:
        host, repository, tag = self._split_reference(tagged_reference)
        basic = self._basic_credential(host)
        manifest_root = f"https://{host}/v2/{quote(repository, safe='/')}/manifests"
        index_url = f"{manifest_root}/{quote(tag, safe='')}"
        token = self._bearer_token(index_url, host, repository, basic)
        index_digest, index = self._json_get(
            index_url, token, _INDEX_ACCEPT, phase="index"
        )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", index_digest) is None or index.get(
            "mediaType"
        ) not in {
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        }:
            raise NebiusCliError("registry_index_mismatch")
        manifests = index.get("manifests")
        all_amd64 = (
            [
                item
                for item in manifests
                if isinstance(item, dict)
                and item.get("platform") == {"architecture": "amd64", "os": "linux"}
            ]
            if isinstance(manifests, list)
            else []
        )
        if len(all_amd64) != 1:
            raise NebiusCliError("registry_child_mismatch")
        selected_child_digest = all_amd64[0].get("digest")
        if (
            not isinstance(selected_child_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", selected_child_digest) is None
        ):
            raise NebiusCliError("registry_child_mismatch")
        child_url = f"{manifest_root}/{selected_child_digest}"
        child_digest, child = self._json_get(
            child_url,
            token,
            _MANIFEST_ACCEPT,
            expected_digest=selected_child_digest,
            phase="child",
        )
        if child_digest != selected_child_digest:
            raise NebiusCliError("registry_child_mismatch")
        descriptor = child.get("config")
        config_digest = (
            descriptor.get("digest") if isinstance(descriptor, dict) else None
        )
        if (
            not isinstance(config_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest) is None
        ):
            raise NebiusCliError("registry_config_invalid")
        config_url = (
            f"https://{host}/v2/{quote(repository, safe='/')}/blobs/{config_digest}"
        )
        _, image_config = self._json_get(
            config_url,
            token,
            "application/vnd.oci.image.config.v1+json",
            expected_digest=config_digest,
            phase="config",
        )
        labels = self._nested(image_config, "config", "Labels")
        base = (
            labels.get("org.opencontainers.image.base.name")
            if isinstance(labels, dict)
            else None
        )
        if base is not None and (not isinstance(base, str) or not base):
            raise NebiusCliError("registry_base_mismatch")
        return RegistryV2Lineage(index_digest, child_digest, base)

    @staticmethod
    def _split_reference(reference: str) -> tuple[str, str, str]:
        tagged_repository, separator, tag = reference.rpartition(":")
        host, slash, repository = tagged_repository.partition("/")
        if (
            not separator
            or not slash
            or not host
            or not repository
            or "@" in reference
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", tag)
        ):
            raise NebiusCliError("registry_tag_invalid")
        return host, repository, tag

    def _basic_credential(self, host: str) -> str:
        try:
            document = self._runner.json_from_stdin(
                f"{host}\n",
                "registry",
                "docker-credential",
                "get",
                "--profile",
                self._profile,
                "--retries",
                "1",
                "--no-check-update",
                "--no-progress",
            )
            server, username, secret = (
                document.get("ServerURL"),
                document.get("Username"),
                document.get("Secret"),
            )
        except (NebiusCliError, TypeError, ValueError) as error:
            raise NebiusCliError("registry_credentials_unavailable") from error
        if (
            server not in {host, f"https://{host}"}
            or not isinstance(username, str)
            or not username
            or not isinstance(secret, str)
            or not secret
        ):
            raise NebiusCliError("registry_credentials_invalid")
        return base64.b64encode(f"{username}:{secret}".encode()).decode("ascii")

    def _request(
        self,
        request: Request,
        *,
        phase: str,
        limit: int,
        expected_digest: str | None = None,
    ):
        previous = None
        for hop in range(6):
            status, headers, body = self._exchange(
                request,
                phase=phase,
                limit=limit,
                expected_digest=expected_digest,
                redirect_from_sha256=previous,
            )
            if status not in {301, 302, 303, 307, 308}:
                return headers, body
            if hop == 5:
                raise NebiusCliError("registry_redirect_limit")
            location = headers.get("Location")
            if not location:
                raise NebiusCliError("registry_redirect_invalid")
            target_url = urljoin(request.full_url, location)
            source, target = urlsplit(request.full_url), urlsplit(target_url)
            same_origin = (source.scheme, source.netloc) == (
                target.scheme,
                target.netloc,
            )
            if (
                target.scheme != "https"
                or not target.hostname
                or target.username is not None
                or target.password is not None
                or (not same_origin and phase != "config")
            ):
                raise NebiusCliError("registry_redirect_invalid")
            headers = dict(request.header_items())
            if not same_origin:
                headers.pop("Authorization", None)
                # The opener adds Host to the Registry request. A presigned
                # Storage URL must use its own host for signature validation.
                headers.pop("Host", None)
                # OCI content negotiation belongs to the Registry endpoint.
                headers.pop("Accept", None)
            previous = hashlib.sha256(request.full_url.encode()).hexdigest()
            request = Request(target_url, headers=headers)
        raise AssertionError("bounded redirect loop exhausted")

    def _exchange(
        self,
        request: Request,
        *,
        phase: str,
        limit: int,
        expected_digest: str | None,
        redirect_from_sha256: str | None,
    ):
        unit = observe_execution_contact_start(
            self._observer,
            category="provider",
            operation=f"registry.http.{phase}",
            executable_identity="registry-http-client",
        )
        status, body, headers = None, None, {}
        disposition = "failed"
        try:
            try:
                response = self._opener(request, timeout=30)
            except HTTPError as error:
                response = error
            with response:
                status = response.status
                headers = response.headers
                body = response.read(limit + 1) if limit else b""
            disposition = (
                "succeeded"
                if status < 400 or (phase == "challenge" and status == 401)
                else "failed"
            )
            if status >= 400 and not (phase == "challenge" and status == 401):
                raise HTTPError(
                    request.full_url, status, "registry_http_unavailable", {}, None
                )
            return status, headers, body
        finally:
            declared = headers.get("Docker-Content-Digest")
            observe_execution_contact_finish(
                self._observer,
                unit,
                disposition=disposition,
                parse_classification="response_received"
                if status is not None
                else "unavailable",
                registry_response=dict(
                    request_sha256=hashlib.sha256(
                        request.full_url.encode()
                    ).hexdigest(),
                    status=status,
                    redirect_from_sha256=redirect_from_sha256,
                    body_digest=("sha256:" + hashlib.sha256(body).hexdigest())
                    if phase in {"index", "child", "config"}
                    and body is not None
                    and len(body) <= limit
                    else None,
                    declared_digest=declared
                    if phase in {"index", "child", "config"}
                    and isinstance(declared, str)
                    and re.fullmatch(r"sha256:[0-9a-f]{64}", declared)
                    else None,
                    expected_digest=expected_digest,
                ),
            )

    def _bearer_token(self, url: str, host: str, repository: str, basic: str) -> str:
        try:
            headers, _ = self._request(
                Request(
                    url,
                    headers={
                        "Authorization": f"Basic {basic}",
                        "Accept": _INDEX_ACCEPT,
                    },
                ),
                phase="challenge",
                limit=0,
            )
            challenge = headers.get("WWW-Authenticate")
        except (OSError, URLError) as error:
            raise NebiusCliError("registry_index_unavailable") from error
        match = re.fullmatch(
            r'Bearer realm="([^"]+)",service="([^"]+)"(?:,scope="([^"]+)")?',
            challenge or "",
        )
        if match is None:
            raise NebiusCliError("registry_challenge_invalid")
        realm, service, challenge_scope = match.groups()
        selected = urlsplit(realm)
        scope = f"repository:{repository}:pull"
        if (
            selected.scheme != "https"
            or selected.hostname != host
            or service != host
            or challenge_scope not in {None, scope}
        ):
            raise NebiusCliError("registry_challenge_invalid")
        token_url = f"{realm}?{urlencode({'service': service, 'scope': scope})}"
        try:
            _, payload = self._request(
                Request(token_url, headers={"Authorization": f"Basic {basic}"}),
                phase="token",
                limit=64 * 1024,
            )
            document = json.loads(payload)
            token = document.get("token") or document.get("access_token")
        except (
            OSError,
            URLError,
            HTTPError,
            UnicodeError,
            json.JSONDecodeError,
            AttributeError,
        ) as error:
            raise NebiusCliError("registry_token_unavailable") from error
        if len(payload) > 64 * 1024 or not isinstance(token, str) or not token:
            raise NebiusCliError("registry_token_invalid")
        return token

    def _json_get(
        self,
        url: str,
        token: str,
        accept: str,
        *,
        expected_digest: str | None = None,
        phase: str = "index",
    ) -> tuple[str, dict[str, Any]]:
        try:
            headers, payload = self._request(
                Request(
                    url, headers={"Authorization": f"Bearer {token}", "Accept": accept}
                ),
                phase=phase,
                limit=8 * 1024 * 1024,
                expected_digest=expected_digest,
            )
            header_digest = headers.get("Docker-Content-Digest")
            document = json.loads(payload)
        except (
            OSError,
            URLError,
            HTTPError,
            UnicodeError,
            json.JSONDecodeError,
        ) as error:
            raise NebiusCliError("registry_document_unavailable") from error
        calculated = "sha256:" + hashlib.sha256(payload).hexdigest()
        digest = expected_digest or header_digest
        if (
            len(payload) > 8 * 1024 * 1024
            or not isinstance(document, dict)
            or not isinstance(digest, str)
            or calculated != digest
            or (header_digest is not None and header_digest != digest)
        ):
            raise NebiusCliError("registry_document_invalid")
        return digest, document

    @staticmethod
    def _nested(value: Any, *path: str) -> Any:
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return value


__all__ = [
    "NebiusRegistryV2Resolver",
    "RegistryV2Lineage",
]
