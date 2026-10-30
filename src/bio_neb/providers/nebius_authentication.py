"""Invocation-scoped authentication shared by storage and job callers."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any

from bio_neb.content import canonical_json
from bio_neb.providers.nebius_common import (
    AUTHENTICATION_EXIT_CODES,
    NebiusAccessScope,
    NebiusAuthenticationRequired,
    NebiusCliError,
    NebiusTenantNotActive,
    authenticated_profile_kind,
    nebius_common_arguments,
)
from bio_neb.providers.runtime import NebiusCliTarget

ADAPTER_ENTRY_AUTHENTICATION_SCHEMA = "bio-neb.adapter-entry-authentication/1"
_PROBE_ARGUMENTS = ("iam", "whoami")
_PROBE_OPERATION = ".".join(_PROBE_ARGUMENTS)


def _emit_record(record: Mapping[str, Any]) -> None:
    """Publish one secret-free declaration on this process's operator record."""

    print(canonical_json(dict(record)), file=sys.stderr, flush=True)


class NebiusAuthenticationProbe:
    """Share successful identity observations within one invocation and selected scope.

    The bound runner owns accepted CLI identity. Job preflight additionally verifies its
    runtime target; storage authentication checks active user and project ownership.
    No proof survives a newly constructed invocation.
    """

    def __init__(
        self,
        *,
        cli_runner: Any,
        target: NebiusCliTarget | NebiusAccessScope | None = None,
        emit: Callable[[Mapping[str, Any]], None] = _emit_record,
    ) -> None:
        self._runner = cli_runner
        self._target = target
        self._emit = emit
        self._proved: dict[NebiusAccessScope, tuple[str, bool]] = {}
        self._projects: set[NebiusAccessScope] = set()

    def __call__(self) -> None:
        """Run the adapter-entry precondition against the bound target."""

        if self._target is None:
            raise NebiusCliError("provider_authentication_target_missing")
        self.prove(self._target)

    def prove(self, target: NebiusCliTarget | NebiusAccessScope) -> None:
        """Prove authentication and the selected tenant, at most once per target."""

        scope = NebiusAccessScope(target.profile, target.tenant_id, target.project_id)
        if scope in self._proved:
            return
        common = nebius_common_arguments(target)
        auth_timeout = self._stated_auth_timeout(common)
        # Declared before the call, not after.  Without it a ~654-second auth stall is
        # invisible: the APP sees a long-running subprocess and nothing else.  This line is
        # what an operator sees during that wait, and it names both the contact being spent
        # and the human action it may be waiting on.
        self._emit(
            {
                "schema_version": ADAPTER_ENTRY_AUTHENTICATION_SCHEMA,
                "kind": "authentication_probe_started",
                "operation": _PROBE_OPERATION,
                "profile": target.profile,
                "auth_timeout": auth_timeout,
                "provider_contact_count": 1,
                "provider_mutation_count": 0,
                "create_ceiling_consumed": 0,
                "human_authentication_may_be_required": True,
            }
        )
        try:
            payload = self._runner.json(*_PROBE_ARGUMENTS, *common)
        except NebiusCliError as error:
            evidence = error.failure_evidence
            required = (
                evidence is not None
                and evidence.child_exit_code in AUTHENTICATION_EXIT_CODES
            )
            self._finished(
                "authentication_required" if required else "contact_failed",
                None if evidence is None else evidence.child_exit_code,
            )
            if evidence is not None and required:
                # `iam whoami` names no project, so the other exit-7 case -- a
                # call against a nonexistent project -- is unreachable here.  For this
                # one command exit 7 means only "auth cannot be established".
                raise NebiusAuthenticationRequired(evidence) from error
            raise
        kind = authenticated_profile_kind(
            payload, tenant_id=target.tenant_id, project_id=target.project_id
        )
        if kind is None:
            # Authenticated, but the selected tenant (or, for a service account, the
            # selected project) is not usable.  No browser login fixes that, so it is
            # reported under its own name and never as the halt that asks for one.
            self._finished("tenant_not_active", 0)
            raise NebiusTenantNotActive()
        # Retained only here, after both refusals.  A proof that ended in either of them
        # must not be memoized, or a stage would inherit a pass this call never gave.
        user = payload.get("user_profile", {})
        self._proved[scope] = (
            kind,
            isinstance(user, dict) and user.get("user_account_state") == "ACTIVE",
        )
        self._finished("authenticated", 0)

    def authenticate(self, target: NebiusCliTarget | NebiusAccessScope) -> str:
        """Establish the selected active identity and project, once per invocation."""
        scope = NebiusAccessScope(target.profile, target.tenant_id, target.project_id)
        self.prove(scope)
        kind, user_active = self._proved[scope]
        if scope in self._projects:
            return kind
        try:
            if kind == "user_profile" and not user_active:
                raise NebiusCliError("provider_authentication_unavailable")
            project = self._runner.json(
                "iam",
                "project",
                "get",
                "--id",
                scope.project_id,
                *nebius_common_arguments(scope),
            )
            metadata = project.get("metadata") if isinstance(project, dict) else None
            if not isinstance(metadata, dict) or (
                metadata.get("id") != scope.project_id
                or metadata.get("parent_id") != scope.tenant_id
            ):
                raise NebiusCliError("provider_authentication_unavailable")
        except NebiusCliError:
            self._proved.pop(scope, None)
            raise
        self._projects.add(scope)
        return kind

    def _finished(self, outcome: str, child_exit_code: int | None) -> None:
        self._emit(
            {
                "schema_version": ADAPTER_ENTRY_AUTHENTICATION_SCHEMA,
                "kind": "authentication_probe_finished",
                "operation": _PROBE_OPERATION,
                "outcome": outcome,
                "child_exit_code": child_exit_code,
                "provider_contact_count": 1,
                "provider_mutation_count": 0,
                "create_ceiling_consumed": 0,
            }
        )

    @staticmethod
    def _stated_auth_timeout(common: tuple[str, ...]) -> str:
        """Read the CLI's own outer bound out of the argv that is about to run.

        Ticket 140 put `--auth-timeout` in every Nebius argv so the CLI's bound is a
        stated decision rather than an inherited default.  Taking the value from that
        argv rather than from a second constant means this probe refuses instead of
        declaring a bound it is not actually passing.
        """

        try:
            return common[common.index("--auth-timeout") + 1]
        except (ValueError, IndexError):
            raise NebiusCliError("provider_auth_timeout_unstated") from None


__all__ = [
    "ADAPTER_ENTRY_AUTHENTICATION_SCHEMA",
    "NebiusAuthenticationProbe",
]
