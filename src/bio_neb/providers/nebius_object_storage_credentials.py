"""Transient resolution of the selected Nebius context's S3 credentials."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bio_neb.providers.nebius_common import (
    AwsCredentialMaterial,
    ExecutionContactObserver,
    NebiusCliError,
    S3ContactError,
    observe_execution_contact_finish,
    observe_execution_contact_start,
)
from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
    NebiusObjectStorageTopologyCandidate,
    NebiusProfileCandidate,
    NebiusProjectCandidate,
    NebiusTenantCandidate,
)
from bio_neb.providers.nebius_controller_runtime import (
    accepted_nebius_cli_identity,
    make_accepted_nebius_cli_runner,
)
from bio_neb.storage_contracts import (
    InputStorageNebiusTarget,
    ObjectStorageCredentialReuseReceipt,
)


def _nested(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _fetch_secret_payload(
    runner: Any, common: tuple[str, ...], secret_id: str, version_id: str
) -> str:
    payload = runner.json(
        "mysterybox",
        "payload",
        "get",
        "--secret-id",
        secret_id,
        "--version-id",
        version_id,
        *common,
    )
    data = payload.get("data")
    if (
        not isinstance(data, list)
        or len(data) != 1
        or not isinstance(data[0], dict)
        or not isinstance(data[0].get("string_value"), str)
    ):
        raise S3ContactError("credentials_unavailable")
    return data[0]["string_value"]


class NebiusCliExecutionContextDiscovery:
    """Project secret-free, read-only candidates from one installed Nebius CLI."""

    def __init__(
        self,
        *,
        cli_runner: Any,
    ) -> None:
        self._runner = cli_runner

    @staticmethod
    def _items(
        payload: dict[str, Any], collection_key: str = "items"
    ) -> list[dict[str, Any]]:
        items = payload.get(collection_key)
        if not isinstance(items, list) or not all(
            isinstance(item, dict) for item in items
        ):
            raise S3ContactError("credentials_unavailable")
        return items

    @staticmethod
    def _common(profile: str) -> tuple[str, ...]:
        return (
            "--profile",
            profile,
            "--retries",
            "1",
            "--no-check-update",
            "--no-progress",
            "--format",
            "json",
        )

    def _pages(
        self,
        prefix: tuple[str, ...],
        profile: str,
        *,
        collection_key: str = "items",
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        token: str | None = None
        seen: set[str] = set()
        while True:
            page = () if token is None else ("--page-token", token)
            payload = self._runner.json(*prefix, *page, *self._common(profile))
            result.extend(self._items(payload, collection_key))
            next_token = payload.get("next_page_token")
            if next_token in {None, ""}:
                return result
            if (
                not isinstance(next_token, str)
                or next_token.strip() != next_token
                or next_token in seen
            ):
                raise S3ContactError("credentials_unavailable")
            seen.add(next_token)
            token = next_token

    def profiles(self) -> tuple[NebiusProfileCandidate, ...]:
        names: list[str] = []
        for line in self._runner(
            "profile", "list", "--no-check-update", "--no-progress", "--retries", "1"
        ).stdout.splitlines():
            value = line.removesuffix(" [default]")
            if (
                not value
                or any(char.isspace() for char in value)
                or any(c in value for c in "[]")
            ):
                raise S3ContactError("credentials_unavailable")
            names.append(value)
        return tuple(
            NebiusProfileCandidate(profile=item) for item in dict.fromkeys(names)
        )

    def tenants(self, profile: str) -> tuple[NebiusTenantCandidate, ...]:
        items = self._pages(("iam", "tenant", "list"), profile)
        return tuple(
            NebiusTenantCandidate(
                profile=profile,
                tenant_id=tenant_id,
                name=name if isinstance(name, str) else None,
            )
            for item in items
            if isinstance(tenant_id := _nested(item, "metadata", "id"), str)
            and (name := _nested(item, "metadata", "name")) is not False
        )

    def projects(
        self, profile: str, tenant_id: str
    ) -> tuple[NebiusProjectCandidate, ...]:
        items = self._pages(
            ("iam", "project", "list", "--parent-id", tenant_id), profile
        )
        return tuple(
            NebiusProjectCandidate(
                profile=profile,
                tenant_id=tenant_id,
                project_id=project_id,
                name=name if isinstance(name, str) else None,
            )
            for item in items
            if _nested(item, "metadata", "parent_id") == tenant_id
            and isinstance(project_id := _nested(item, "metadata", "id"), str)
            and (name := _nested(item, "metadata", "name")) is not False
        )

    def _topology_for_access(
        self,
        access: dict[str, Any],
        *,
        profile: str,
        tenant_id: str,
        project_id: str,
    ) -> list[NebiusObjectStorageTopologyCandidate]:
        access_resource_id = _nested(access, "metadata", "id")
        aws_access_key_id = _nested(access, "status", "aws_access_key_id")
        service_account_id = _nested(access, "spec", "account", "service_account", "id")
        if (
            not all(
                isinstance(value, str)
                for value in (access_resource_id, aws_access_key_id, service_account_id)
            )
            or _nested(access, "metadata", "parent_id") != project_id
            or _nested(access, "status", "state") != "ACTIVE"
            or str(_nested(access, "spec", "secret_delivery_mode")).casefold()
            != "mystery_box"
            or not isinstance(_nested(access, "status", "secret_reference_id"), str)
        ):
            return []
        account = self._runner.json(
            "iam",
            "service-account",
            "get",
            "--id",
            service_account_id,
            *self._common(profile),
        )
        if (
            _nested(account, "metadata", "id") != service_account_id
            or _nested(account, "metadata", "parent_id") != project_id
        ):
            return []
        matches: list[NebiusObjectStorageTopologyCandidate] = []
        groups = self._pages(
            (
                "iam",
                "group-membership",
                "list-member-of",
                "--subject-id",
                service_account_id,
            ),
            profile,
        )
        group_ids = tuple(
            dict.fromkeys(
                group_id
                for group in groups
                if isinstance(group_id := _nested(group, "metadata", "id"), str)
                and _nested(group, "metadata", "parent_id") == tenant_id
            )
        )
        for group_id in group_ids:
            group = self._runner.json(
                "iam", "group", "get", "--id", group_id, *self._common(profile)
            )
            if (
                _nested(group, "metadata", "id") != group_id
                or _nested(group, "metadata", "parent_id") != tenant_id
            ):
                continue
            memberships = self._pages(
                (
                    "iam",
                    "group-membership",
                    "list-members",
                    "--parent-id",
                    group_id,
                ),
                profile,
                collection_key="memberships",
            )
            membership_ids = [
                membership_id
                for membership in memberships
                if isinstance(
                    membership_id := _nested(membership, "metadata", "id"), str
                )
                and _nested(membership, "metadata", "parent_id") == group_id
                and _nested(membership, "spec", "member_id") == service_account_id
            ]
            permits = self._pages(
                ("iam", "access-permit", "list", "--parent-id", group_id), profile
            )
            permit_ids = [
                permit_id
                for permit in permits
                if isinstance(permit_id := _nested(permit, "metadata", "id"), str)
                and _nested(permit, "metadata", "parent_id") == group_id
                and _nested(permit, "spec", "resource_id") == project_id
                and _nested(permit, "spec", "role") == "editor"
            ]
            for membership_id in membership_ids:
                for permit_id in permit_ids:
                    matches.append(
                        NebiusObjectStorageTopologyCandidate.bind(
                            profile=profile,
                            tenant_id=tenant_id,
                            project_id=project_id,
                            service_account_id=service_account_id,
                            access_key_resource_id=access_resource_id,
                            access_key_id=aws_access_key_id,
                            group_id=group_id,
                            group_membership_id=membership_id,
                            access_permit_id=permit_id,
                        )
                    )
        if len(matches) != 1:
            return []
        secret_id = _nested(access, "status", "secret_reference_id")
        secret = self._runner.json(
            "mysterybox", "secret", "get", "--id", secret_id, *self._common(profile)
        )
        version_id = _nested(secret, "spec", "primary_version_id")
        if (
            _nested(secret, "metadata", "id") != secret_id
            or _nested(secret, "metadata", "parent_id") != project_id
            or _nested(secret, "status", "state") != "ACTIVE"
            or not isinstance(version_id, str)
        ):
            return []
        version = self._runner.json(
            "mysterybox",
            "secret-version",
            "get",
            "--id",
            version_id,
            *self._common(profile),
        )
        if (
            _nested(version, "metadata", "id") != version_id
            or _nested(version, "metadata", "parent_id") != secret_id
        ):
            return []
        return matches

    def topologies(
        self, profile: str, tenant_id: str, project_id: str
    ) -> tuple[NebiusObjectStorageTopologyCandidate, ...]:
        accesses = self._pages(
            ("iam", "v2", "access-key", "list", "--parent-id", project_id), profile
        )
        candidates = [
            candidate
            for access in accesses
            for candidate in self._topology_for_access(
                access, profile=profile, tenant_id=tenant_id, project_id=project_id
            )
        ]
        return tuple(dict.fromkeys(candidates))


class SelectedNebiusCredentialResolver:
    """Re-read an APP selection, resolve its current primary secret, then memoize."""

    def __init__(
        self,
        target: InputStorageNebiusTarget | None,
        selection: NebiusExecutionContextSelection,
        *,
        payload_fetcher: Callable[[str, str], str] | None = None,
        cli_runner: Any | None = None,
        execution_contact_observer: ExecutionContactObserver | None = None,
    ) -> None:
        self._target = target
        self._selection = selection
        if cli_runner is not None:
            self._runner = cli_runner
        else:
            expected = accepted_nebius_cli_identity(selection.nebius_executable)
            if (
                selection.resolved_nebius_executable != expected.resolved_executable
                or selection.nebius_cli_version != expected.version
                or selection.nebius_executable_byte_count != expected.byte_count
                or selection.nebius_executable_sha256 != expected.sha256
            ):
                raise S3ContactError("credentials_unavailable")
            self._runner = make_accepted_nebius_cli_runner(selection.nebius_executable)
        self._payload_fetcher = payload_fetcher
        self._execution_contact_observer = execution_contact_observer
        self._material: AwsCredentialMaterial | None = None
        self._receipt: ObjectStorageCredentialReuseReceipt | None = None

    @property
    def receipt(self) -> ObjectStorageCredentialReuseReceipt:
        if self._receipt is None:
            raise S3ContactError("credentials_unavailable")
        return self._receipt

    def _common(self) -> tuple[str, ...]:
        return (
            "--profile",
            self._selection.topology.profile,
            "--retries",
            "1",
            "--no-check-update",
            "--no-progress",
            "--format",
            "json",
        )

    def _get(self, *arguments: str) -> dict[str, Any]:
        return self._runner.json(*arguments, *self._common())

    def _resolve(self) -> AwsCredentialMaterial:
        if self._material is not None:
            return self._material
        selected = self._selection.topology
        if self._target is not None:
            target = self._target
            runtime_mismatch = (
                target.controller_runtime != "native_ubuntu"
                or self._selection.nebius_executable != target.executable
                or self._selection.resolved_nebius_executable
                != target.resolved_executable
                or self._selection.nebius_cli_version != target.cli_version
                or self._selection.nebius_executable_byte_count
                != target.executable_byte_count
                or self._selection.nebius_executable_sha256 != target.executable_sha256
            )
            if (
                runtime_mismatch
                or selected.profile != target.nebius_cli_profile
                or selected.tenant_id != target.tenant_id
                or selected.project_id != target.project_id
            ):
                raise S3ContactError("credentials_unavailable")
        try:
            current = NebiusCliExecutionContextDiscovery(
                cli_runner=self._runner,
            ).topologies(selected.profile, selected.tenant_id, selected.project_id)
            if sum(candidate == selected for candidate in current) != 1:
                raise S3ContactError("credentials_unavailable")
            access = self._get(
                "iam",
                "v2",
                "access-key",
                "get",
                "--id",
                selected.access_key_resource_id,
            )
            secret_id = _nested(access, "status", "secret_reference_id")
            if (
                _nested(access, "metadata", "id") != selected.access_key_resource_id
                or _nested(access, "metadata", "parent_id") != selected.project_id
                or _nested(access, "status", "state") != "ACTIVE"
                or _nested(access, "status", "aws_access_key_id")
                != selected.access_key_id
                or str(_nested(access, "spec", "secret_delivery_mode")).casefold()
                != "mystery_box"
                or _nested(access, "spec", "account", "service_account", "id")
                != selected.service_account_id
                or not isinstance(secret_id, str)
            ):
                raise S3ContactError("credentials_unavailable")
            account = self._get(
                "iam", "service-account", "get", "--id", selected.service_account_id
            )
            group = self._get("iam", "group", "get", "--id", selected.group_id)
            membership = self._get(
                "iam", "group-membership", "get", "--id", selected.group_membership_id
            )
            permit = self._get(
                "iam", "access-permit", "get", "--id", selected.access_permit_id
            )
            if (
                _nested(account, "metadata", "id") != selected.service_account_id
                or _nested(account, "metadata", "parent_id") != selected.project_id
                or _nested(group, "metadata", "id") != selected.group_id
                or _nested(group, "metadata", "parent_id") != selected.tenant_id
                or _nested(membership, "metadata", "id") != selected.group_membership_id
                or _nested(membership, "metadata", "parent_id") != selected.group_id
                or _nested(membership, "spec", "member_id")
                != selected.service_account_id
                or _nested(permit, "metadata", "id") != selected.access_permit_id
                or _nested(permit, "metadata", "parent_id") != selected.group_id
                or _nested(permit, "spec", "resource_id") != selected.project_id
                or _nested(permit, "spec", "role") != "editor"
            ):
                raise S3ContactError("credentials_unavailable")
            secret = self._get("mysterybox", "secret", "get", "--id", secret_id)
            version_id = _nested(secret, "spec", "primary_version_id")
            if (
                _nested(secret, "metadata", "id") != secret_id
                or _nested(secret, "metadata", "parent_id") != selected.project_id
                or _nested(secret, "status", "state") != "ACTIVE"
                or not isinstance(version_id, str)
            ):
                raise S3ContactError("credentials_unavailable")
            version = self._get(
                "mysterybox", "secret-version", "get", "--id", version_id
            )
            if (
                _nested(version, "metadata", "id") != version_id
                or _nested(version, "metadata", "parent_id") != secret_id
            ):
                raise S3ContactError("credentials_unavailable")
            payload = (
                self._payload_fetcher(secret_id, version_id)
                if self._payload_fetcher is not None
                else _fetch_secret_payload(
                    self._runner, self._common(), secret_id, version_id
                )
            )
            self._material = AwsCredentialMaterial(
                access_key_id=selected.access_key_id,
                secret_access_key=payload,
                credentials_profile=f"object-storage:{selected.access_key_resource_id}",
                worker_secret_selector=f"{secret_id}@{version_id}",
            )
            self._receipt = ObjectStorageCredentialReuseReceipt(
                service_account_id=selected.service_account_id,
                access_key_resource_id=selected.access_key_resource_id,
                access_key_id=selected.access_key_id,
                exact_topology_match_count=1,
                access_key_state="ACTIVE",
                secret_state="ACTIVE",
                secret_delivery_mode="mystery_box",
                host_secret_resolved=True,
                worker_secret_version_bound=True,
            )
        except S3ContactError:
            raise
        except (KeyError, NebiusCliError, TypeError, ValueError) as error:
            raise S3ContactError("credentials_unavailable") from error
        return self._material

    def __call__(self) -> AwsCredentialMaterial:
        cached = self._material is not None
        unit_id = observe_execution_contact_start(
            self._execution_contact_observer,
            category="credential",
            operation="selected-object-storage-credential",
            executable_identity="credential-resolver",
            credential_names=(
                "control_plane_authentication",
                "host_secret_resolution",
                "worker_secret_binding",
                "session_credentials",
            ),
            selected_identity=self._selection.topology.candidate_id,
        )
        try:
            material = self._resolve()
        except BaseException:
            observe_execution_contact_finish(
                self._execution_contact_observer,
                unit_id,
                disposition="failed",
                parse_classification="credential_unavailable",
            )
            raise
        observe_execution_contact_finish(
            self._execution_contact_observer,
            unit_id,
            disposition="succeeded",
            parse_classification="credential_cached"
            if cached
            else "credential_resolved",
        )
        return material


__all__ = [
    "NebiusCliExecutionContextDiscovery",
    "SelectedNebiusCredentialResolver",
]
