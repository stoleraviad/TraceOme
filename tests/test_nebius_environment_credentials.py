from __future__ import annotations

import unittest

from bio_neb.providers.nebius_common import (
    EnvironmentAwsCredentialResolver,
)

_ACCESS_KEY = "access-key-sentinel"
_SECRET_KEY = "secret-key-sentinel"
_SECRET_SELECTOR = "secret-selector@secret-version"


class EnvironmentCredentialResolverTests(unittest.TestCase):
    def test_environment_credentials_retain_public_profile_and_secret_selector_without_session(
        self,
    ) -> None:
        credentials = EnvironmentAwsCredentialResolver(
            {
                "AWS_ACCESS_KEY_ID": _ACCESS_KEY,
                "AWS_SECRET_ACCESS_KEY": _SECRET_KEY,
                "BIO_NEB_S3_CREDENTIALS_PROFILE": "object-storage-profile",
                "BIO_NEB_WORKER_SECRET_SELECTOR": _SECRET_SELECTOR,
            }
        )()
        self.assertEqual(credentials.access_key_id, _ACCESS_KEY)
        self.assertEqual(credentials.secret_access_key, _SECRET_KEY)
        self.assertEqual(credentials.credentials_profile, "object-storage-profile")
        self.assertEqual(credentials.worker_secret_selector, _SECRET_SELECTOR)
        self.assertIsNone(credentials.session_token)


if __name__ == "__main__":
    unittest.main()
