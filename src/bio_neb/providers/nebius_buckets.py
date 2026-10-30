"""Exact-name bucket access; callers own fresh-only versus permitted reuse."""

from bio_neb.failure import ProviderError
from bio_neb.providers.nebius_common import AmbiguousNebiusCreate, NebiusCliError


def _safe_cause(error):
    if isinstance(error, ProviderError):
        return {"detail": error.detail}
    evidence = {"code": error.code}
    if error.failure_evidence is not None:
        evidence["failure_evidence"] = error.failure_evidence.public_fields()
    if error.provider_reason is not None:
        evidence["provider_reason"] = error.provider_reason
    return evidence


class NebiusBucketAccess:
    def __init__(self, runner, *, common, project_id, region, domain):
        self.runner = runner
        self.common = common
        self.project_id = project_id
        self.region = region
        self.domain = domain

    def find(self, name):
        payload = self.runner.json(
            "storage",
            "bucket",
            "list",
            "--parent-id",
            self.project_id,
            "--all",
            *self.common,
        )
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("metadata"), dict)
            or not isinstance(item["metadata"].get("name"), str)
            for item in items
        ):
            raise ProviderError("rejected", "bucket_lookup_invalid")
        return [item for item in items if item["metadata"]["name"] == name]

    def validate(self, name, payload):
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        status = payload.get("status") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict) or not isinstance(status, dict):
            raise ProviderError("rejected", "bucket_identity_mismatch")
        if status.get("suspension_state") not in (
            None,
            "NOT_SUSPENDED",
            "UNSUSPENDED",
            "NONE",
        ):
            raise ProviderError("rejected", "bucket_suspended")
        if status.get("state") != "ACTIVE":
            raise ProviderError("rejected", "bucket_not_active")
        if (
            metadata.get("parent_id") != self.project_id
            or metadata.get("name") != name
            or not isinstance(metadata.get("id"), str)
            or not metadata["id"]
        ):
            raise ProviderError("rejected", "bucket_identity_mismatch")
        if status.get("region") != self.region:
            raise ProviderError("rejected", "bucket_region_mismatch")
        if status.get("domain_name") != self.domain:
            raise ProviderError("rejected", "bucket_domain_mismatch")
        return payload

    def create(self, name):
        """Create once, reconciling an uncertain result with one exact-name read."""
        try:
            payload = self.runner.json(
                "storage",
                "bucket",
                "create",
                "--name",
                name,
                "--parent-id",
                self.project_id,
                "--default-storage-class",
                "intelligent",
                *self.common,
            )
            return self.validate(name, payload), 0
        except (NebiusCliError, ProviderError) as error:
            if isinstance(error, NebiusCliError) and not (
                isinstance(error, AmbiguousNebiusCreate)
                or error.code
                in {"provider_json_invalid", "provider_output_invalid_utf8"}
            ):
                raise
            cause = _safe_cause(error)
        try:
            matches = self.find(name)
            if len(matches) != 1:
                raise ProviderError(
                    "exhausted",
                    "bucket_match_not_unique",
                    {"match_count": len(matches)},
                )
            payload = self.validate(name, matches[0])
        except (NebiusCliError, ProviderError) as error:
            raise ProviderError(
                "exhausted",
                "bucket_create_ambiguous",
                {"create": cause, "reconciliation": _safe_cause(error)},
            ) from None
        return payload, 1
