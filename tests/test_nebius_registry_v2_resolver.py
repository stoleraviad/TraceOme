from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import HTTPSHandler, build_opener
from urllib.response import addinfourl

from bio_neb.parabricks.execution_contact_accounting import (
    ExecutionContactJournal,
    compile_execution_contact_evidence,
    read_complete_execution_contacts,
)
from bio_neb.providers.nebius_common import NebiusCliError
from bio_neb.providers.nebius_registry_v2 import NebiusRegistryV2Resolver

_SECRET_KEY = "secret-key-sentinel"


class RegistryV2ResolverTests(unittest.TestCase):
    _REFERENCE = (
        "cr.eu-north1.nebius.cloud/e00md63vxjxams912c/bio-neb-parabricks-worker:r4"
    )

    def _http_case(self):
        # Synthetic OCI bytes, hashed independently of the resolver; no live fixture claim.
        config = b'{"config":{"Labels":{"org.opencontainers.image.base.name":"base:approved"}}}'
        config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
        child = json.dumps({"config": {"digest": config_digest}}).encode()
        child_digest = "sha256:" + hashlib.sha256(child).hexdigest()
        index = json.dumps(
            {
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [
                    {
                        "digest": child_digest,
                        "platform": {"architecture": "amd64", "os": "linux"},
                    }
                ],
            }
        ).encode()
        index_digest = "sha256:" + hashlib.sha256(index).hexdigest()
        host = "cr.eu-north1.nebius.cloud"
        responses = [
            (
                401,
                {
                    "WWW-Authenticate": f'Bearer realm="https://{host}/token",service="{host}"'
                },
                b"",
            ),
            (200, {}, b'{"token":"bearer-sentinel"}'),
            (200, {"Docker-Content-Digest": index_digest}, index),
            (200, {"Docker-Content-Digest": child_digest}, child),
            (200, {}, config),
        ]
        calls = []

        class Transport(HTTPSHandler):
            def https_open(self, request):
                calls.append(request)
                # The presigned config URL signs its Storage host. A copied
                # Registry Host header makes the Storage request fail.
                scripted = (
                    (403, {}, b"storage denied")
                    if request.full_url.startswith("https://blob.invalid/")
                    and request.get_header("Host") != "blob.invalid"
                    else responses.pop(0)
                )
                if isinstance(scripted, BaseException):
                    raise scripted
                status, headers, body = scripted
                message = Message()
                for key, value in headers.items():
                    message[key] = value

                class BrokenRead(io.BytesIO):
                    def read(self, *args):
                        raise OSError("read-secret")

                stream = BrokenRead() if isinstance(body, OSError) else io.BytesIO(body)
                response = addinfourl(stream, message, request.full_url, status)
                response.msg = "untrusted-error-sentinel"
                return response

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        journal = ExecutionContactJournal(
            Path(temporary.name) / "journal.jsonl", "parabricks-execution:" + "a" * 64
        )
        with patch(
            "bio_neb.providers.nebius_registry_v2.build_opener",
            side_effect=lambda *handlers: build_opener(*handlers, Transport()),
        ):
            resolver = NebiusRegistryV2Resolver(
                cli_runner=self._resolver()._runner,
                profile="default",
                execution_contact_observer=journal,
            )
        return (
            resolver,
            journal,
            responses,
            calls,
            (index_digest, child_digest, config_digest),
        )

    def test_http_lineage_is_retained_without_credentials(self):
        resolver, journal, _, calls, digests = self._http_case()
        lineage = resolver(self._REFERENCE)
        self.assertEqual(
            (
                lineage.index_digest,
                lineage.linux_amd64_child_digest,
                lineage.base_reference,
            ),
            (*digests[:2], "base:approved"),
        )
        binding = journal.finalize()
        rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
        terminals = [row for row in rows if row["row_kind"] == "terminal"]
        self.assertEqual(
            [row["operation"] for row in terminals],
            [
                "registry.http.challenge",
                "registry.http.token",
                "registry.http.index",
                "registry.http.child",
                "registry.http.config",
            ],
        )
        self.assertEqual(
            [row["registry_response"]["status"] for row in terminals],
            [401, 200, 200, 200, 200],
        )
        self.assertEqual(
            [row["registry_response"]["body_digest"] for row in terminals],
            [None, None, *digests],
        )
        self.assertEqual(
            [row["registry_response"]["expected_digest"] for row in terminals],
            [None, None, None, digests[1], digests[2]],
        )
        self.assertEqual(len(calls), 5)
        self.assertTrue(rows[-1]["complete"])
        read_complete_execution_contacts(
            journal.path.read_bytes(), journal.execution_id
        )
        review = compile_execution_contact_evidence(
            journal_bytes=journal.path.read_bytes(),
            binding=binding,
            expected_execution_id=journal.execution_id,
            journey_events=(),
            journey_binding="synthetic-no-outcome",
        )
        self.assertEqual(
            [unit.registry_response.body_digest for unit in review.units],
            [None, None, *digests],
        )
        self.assertEqual(review.logical_attempts, ())
        self.assertEqual(review.ledger.network_contact.status, "unmeasurable")
        retained = journal.path.read_text()
        for secret in (
            _SECRET_KEY,
            "bearer-sentinel",
            "contact-user",
            "WWW-Authenticate",
            "Authorization",
            "untrusted-error-sentinel",
            self._REFERENCE,
            "/token?",
        ):
            self.assertNotIn(secret, retained)

    def test_config_redirect_retains_each_exchange_and_strips_cross_origin_credentials(
        self,
    ):
        resolver, journal, responses, calls, digests = self._http_case()
        responses.insert(
            4,
            (
                307,
                {"Location": "https://blob.invalid/config?signature=signed-sentinel"},
                b"",
            ),
        )
        self.assertEqual(resolver(self._REFERENCE).linux_amd64_child_digest, digests[1])
        journal.finalize()
        rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
        terminals = [row for row in rows if row["row_kind"] == "terminal"]
        self.assertEqual(
            [row["registry_response"]["status"] for row in terminals],
            [401, 200, 200, 200, 307, 200],
        )
        self.assertIsNone(calls[-1].get_header("Authorization"))
        self.assertIsNone(calls[-1].get_header("Accept"))
        self.assertEqual(calls[-1].get_header("Host"), "blob.invalid")
        self.assertEqual(
            terminals[-1]["registry_response"]["redirect_from_sha256"],
            terminals[-2]["registry_response"]["request_sha256"],
        )
        self.assertNotIn("signed-sentinel", journal.path.read_text())
        self.assertTrue(rows[-1]["complete"])

    def test_failed_http_and_digest_drift_retain_observed_prefix_only(self):
        cases = (
            (0, (403, {}, b"denied-secret"), "registry_index_unavailable"),
            (1, OSError("transport-secret"), "registry_token_unavailable"),
            (1, (200, {}, b"not-json-token-secret"), "registry_token_unavailable"),
            (
                3,
                (200, {"Docker-Content-Digest": "sha256:" + "0" * 64}, b"{}"),
                "registry_document_invalid",
            ),
            (4, OSError("transport-secret"), "registry_document_unavailable"),
            (4, (200, {}, OSError("read-secret")), "registry_document_unavailable"),
        )
        for position, response, reason in cases:
            with self.subTest(reason=reason, position=position):
                resolver, journal, responses, calls, _ = self._http_case()
                responses[position] = response
                with self.assertRaises(NebiusCliError) as caught:
                    resolver(self._REFERENCE)
                self.assertEqual(caught.exception.code, reason)
                journal.finalize()
                rows = [
                    json.loads(line) for line in journal.path.read_text().splitlines()
                ]
                terminals = [row for row in rows if row["row_kind"] == "terminal"]
                self.assertEqual(len(terminals), position + 1)
                self.assertEqual(len(calls), position + 1)
                self.assertTrue(rows[-1]["complete"])
                # Transport completion is separate from semantic digest/JSON acceptance.
                self.assertEqual(
                    terminals[-1]["disposition"],
                    "failed"
                    if isinstance(response, OSError)
                    or response[0] >= 400
                    or isinstance(response[2], OSError)
                    else "succeeded",
                )
                if isinstance(response, OSError) or isinstance(response[2], OSError):
                    self.assertIsNone(terminals[-1]["registry_response"]["body_digest"])
                    self.assertEqual(
                        terminals[-1]["registry_response"]["status"],
                        None if isinstance(response, OSError) else 200,
                    )
                if position == 3:
                    evidence = terminals[-1]["registry_response"]
                    self.assertNotEqual(
                        evidence["body_digest"], evidence["expected_digest"]
                    )
                    self.assertEqual(evidence["declared_digest"], "sha256:" + "0" * 64)
                for secret in (
                    "transport-secret",
                    "denied-secret",
                    "not-json-token-secret",
                    _SECRET_KEY,
                    "bearer-sentinel",
                ):
                    self.assertNotIn(secret, journal.path.read_text())

    def test_redirect_refusals_never_contact_forbidden_targets(self):
        for position, target in (
            (1, "https://other.invalid/token"),
            (4, "http://blob.invalid/config"),
            (4, "https://user:password@blob.invalid/config"),
        ):
            with self.subTest(position=position, target=target):
                resolver, journal, responses, calls, _ = self._http_case()
                responses[position] = (302, {"Location": target}, b"")
                with self.assertRaisesRegex(
                    NebiusCliError, "registry_redirect_invalid"
                ):
                    resolver(self._REFERENCE)
                self.assertEqual(len(calls), position + 1)
                journal.finalize()
                self.assertTrue(
                    json.loads(journal.path.read_text().splitlines()[-1])["complete"]
                )
                self.assertNotIn(target, journal.path.read_text())

    def test_redirect_loop_is_bounded_and_every_attempt_is_retained(self):
        resolver, journal, responses, calls, _ = self._http_case()
        responses[4:] = [(307, {"Location": "/repeat"}, b"")] * 7
        with self.assertRaisesRegex(NebiusCliError, "registry_redirect_limit"):
            resolver(self._REFERENCE)
        self.assertEqual(len(calls), 10)  # Four prerequisites plus six config attempts.
        journal.finalize()
        rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
        self.assertEqual(
            len([row for row in rows if row["row_kind"] == "terminal"]), 10
        )
        self.assertTrue(rows[-1]["complete"])

    def test_default_provider_retains_attestation_before_refusing_drift(self):
        from dataclasses import asdict, replace
        from types import SimpleNamespace

        from bio_neb.failure import ProviderError
        from bio_neb.providers.nebius import NebiusProvider
        from tests.nebius_serverless_job_contact_fakes import _NativeCli, _request
        from tests.reference_stage_fakes import credentials

        resolver, journal, _, calls, _ = self._http_case()
        request = replace(_request(), provider_image_reference=self._REFERENCE)
        cli = _NativeCli()
        cli.json_from_stdin = resolver._runner.json_from_stdin
        with patch(
            "bio_neb.providers.nebius_registry_v2.build_opener",
            return_value=SimpleNamespace(open=resolver._opener),
        ):
            provider = NebiusProvider(
                selection=SimpleNamespace(topology=request.target),
                endpoint_url="https://storage.invalid",
                region="region",
                journal=journal,
                cli_runner=cli,
                credential_resolver=credentials,
            )
        values = asdict(request)
        values.pop("provider_job_name")
        values.update(
            image_index_digest="sha256:" + "0" * 64,
            image_child_digest="sha256:" + "1" * 64,
        )
        with self.assertRaisesRegex(ProviderError, "job_image_drift"):
            provider.jobs.submit(request.provider_job_name, values)
        self.assertEqual(len(calls), 5)
        journal.finalize()
        rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
        self.assertEqual(len([row for row in rows if row["row_kind"] == "terminal"]), 5)
        self.assertFalse(any(call[:3] == ("ai", "job", "create") for call in cli.calls))
        self.assertTrue(rows[-1]["complete"])

    def test_default_provider_completes_attestation_before_job_create(self):
        from dataclasses import asdict, replace
        from types import SimpleNamespace

        from bio_neb.providers.nebius import NebiusProvider
        from tests.nebius_serverless_job_contact_fakes import _NativeCli, _request
        from tests.reference_stage_fakes import credentials

        resolver, journal, _, _, digests = self._http_case()
        request = replace(_request(), provider_image_reference=self._REFERENCE)
        case = self

        class Cli(_NativeCli):
            def __call__(self, *arguments):
                if arguments[:3] == ("ai", "job", "create"):
                    rows = [
                        json.loads(line)
                        for line in journal.path.read_text().splitlines()
                    ]
                    terminals = [row for row in rows if row["row_kind"] == "terminal"]
                    case.assertEqual(
                        [row["registry_response"]["body_digest"] for row in terminals],
                        [None, None, *digests],
                    )
                    case.assertFalse(journal.closed)
                return super().__call__(*arguments)

        cli = Cli()
        cli.json_from_stdin = resolver._runner.json_from_stdin
        with patch(
            "bio_neb.providers.nebius_registry_v2.build_opener",
            return_value=SimpleNamespace(open=resolver._opener),
        ):
            provider = NebiusProvider(
                selection=SimpleNamespace(topology=request.target),
                endpoint_url="https://storage.invalid",
                region="region",
                journal=journal,
                cli_runner=cli,
                credential_resolver=credentials,
            )
        values = asdict(request)
        values.pop("provider_job_name")
        values.update(image_index_digest=digests[0], image_child_digest=digests[1])
        receipt = provider.jobs.submit(request.provider_job_name, values)
        self.assertEqual(receipt.provider_ids, ("aijob-exact",))
        self.assertEqual(
            sum(call[:3] == ("ai", "job", "create") for call in cli.calls), 1
        )
        journal.finalize()
        read_complete_execution_contacts(
            journal.path.read_bytes(), journal.execution_id
        )

    def test_invalid_diagnostic_status_preserves_registry_error_and_incomplete_journal(
        self,
    ):
        resolver, journal, responses, calls, _ = self._http_case()
        responses[0] = (600, {}, b"untrusted-error-sentinel")
        with self.assertRaises(NebiusCliError) as caught:
            resolver(self._REFERENCE)
        self.assertEqual(caught.exception.code, "registry_index_unavailable")
        self.assertEqual(len(calls), 1)
        journal.finalize()
        self.assertFalse(
            json.loads(journal.path.read_text().splitlines()[-1])["complete"]
        )
        self.assertNotIn("untrusted-error-sentinel", journal.path.read_text())

    def _resolver(self):
        class Runner:
            def __init__(self):
                self.calls = []

            def json_from_stdin(self, input_text, *arguments):
                self.calls.append((input_text, arguments))
                return {
                    "ServerURL": "cr.eu-north1.nebius.cloud",
                    "Username": "contact-user",
                    "Secret": _SECRET_KEY,
                }

        runner = Runner()
        return NebiusRegistryV2Resolver(
            cli_runner=runner,
            profile="default",
            opener=lambda *args, **kwargs: None,
        )

    def _documents(self):
        child = (
            "sha256:85ebd4d8db1c35940a89aa1b21face17e63958aa1a597d0205d80e499b963ee2"
        )
        config_digest = "sha256:" + "c" * 64
        return [
            (
                "sha256:2135d746222c5c2542e3d27725c7180b29bb608d449455f06b99750175a26dca",
                {
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "manifests": [
                        {
                            "digest": child,
                            "platform": {"architecture": "amd64", "os": "linux"},
                        },
                        {
                            "digest": "sha256:" + "e" * 64,
                            "platform": {"architecture": "arm64", "os": "linux"},
                        },
                    ],
                },
            ),
            (child, {"config": {"digest": config_digest}}),
            (
                config_digest,
                {
                    "config": {
                        "Labels": {
                            "org.opencontainers.image.base.name": "nvcr.io/nvidia/clara/clara-parabricks:4.7.0-1@sha256:28f2191158fd14421958fc275e219f22f6caaa3755ae3250c0f04883a880c212",
                            "io.bio-neb.parabricks.version": "4.7.0-1",
                        }
                    }
                },
            ),
        ]

    def _resolve_documents(self, documents):
        resolver = self._resolver()
        resolver._basic_credential = lambda _: "contact-basic"
        resolver._bearer_token = lambda *args: "contact-token"
        remaining = iter(documents)
        resolver._json_get = lambda *args, **kwargs: next(remaining)
        return resolver(self._REFERENCE)

    def test_resolves_exact_index_child_and_base(self) -> None:
        lineage = self._resolve_documents(self._documents())
        self.assertTrue(lineage.linux_amd64_child_digest.startswith("sha256:85ebd4"))

    def test_registry_credential_uses_native_runner_stdin_without_retaining_secret(
        self,
    ):
        resolver = self._resolver()

        self.assertEqual(
            resolver._basic_credential("cr.eu-north1.nebius.cloud"),
            base64.b64encode(f"contact-user:{_SECRET_KEY}".encode()).decode("ascii"),
        )
        self.assertEqual(
            resolver._runner.calls,
            [
                (
                    "cr.eu-north1.nebius.cloud\n",
                    (
                        "registry",
                        "docker-credential",
                        "get",
                        "--profile",
                        "default",
                        "--retries",
                        "1",
                        "--no-check-update",
                        "--no-progress",
                    ),
                )
            ],
        )
        self.assertNotIn(_SECRET_KEY, repr(resolver))

    def test_rejects_malformed_index_child_base_and_multiple_amd64(self) -> None:
        mutations = []
        wrong_index = self._documents()
        wrong_index[0] = ("sha256:not-a-digest", wrong_index[0][1])
        mutations.append(wrong_index)
        wrong_child = self._documents()
        wrong_child[0][1]["manifests"][0]["digest"] = "sha256:" + "0" * 64
        mutations.append(wrong_child)
        multiple = self._documents()
        multiple[0][1]["manifests"].append(dict(multiple[0][1]["manifests"][0]))
        mutations.append(multiple)
        wrong_base = self._documents()
        wrong_base[2][1]["config"]["Labels"]["org.opencontainers.image.base.name"] = ""
        mutations.append(wrong_base)
        for documents in mutations:
            with self.subTest(documents=documents), self.assertRaises(NebiusCliError):
                self._resolve_documents(documents)

    def test_rejects_malformed_auth_challenge_without_exposing_credentials(
        self,
    ) -> None:
        resolver = self._resolver()

        def opener(request, timeout):
            raise HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {"WWW-Authenticate": "Basic realm=bad"},
                None,
            )

        resolver._opener = opener
        with self.assertRaises(NebiusCliError) as caught:
            resolver._bearer_token(
                "https://cr.eu-north1.nebius.cloud/v2/repo/manifests/r4",
                "cr.eu-north1.nebius.cloud",
                "repo",
                "contact-basic",
            )
        self.assertEqual(caught.exception.code, "registry_challenge_invalid")
        self.assertNotIn("contact-basic", repr(caught.exception))


if __name__ == "__main__":
    unittest.main()
