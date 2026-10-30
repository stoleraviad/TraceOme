"""Production host composition for readiness and Object Storage tracing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bio_neb.providers.nebius_context import (
    NebiusExecutionContextSelection,
)
from bio_neb.providers.nebius_controller_runtime import (
    accepted_nebius_cli_identity,
    make_accepted_nebius_cli_runner,
)
from bio_neb.providers.nebius_object_storage import (
    NebiusObjectStorageMechanics,
    ReadContact,
)
from bio_neb.providers.nebius_object_storage_credentials import (
    SelectedNebiusCredentialResolver,
)
from bio_neb.providers.nebius_storage_readiness import NebiusStorageReadinessAdapter
from bio_neb.storage_contracts import ExecutionStorageReadinessConfig


@dataclass(frozen=True, slots=True)
class _ProductionObjectStorageTracerAdapters:
    readiness: NebiusStorageReadinessAdapter
    objects: NebiusObjectStorageMechanics


def _accepted_selected_resolver(
    config: ExecutionStorageReadinessConfig,
    selection: NebiusExecutionContextSelection,
    *,
    payload_fetcher: Callable[[str, str], str] | None,
    credential_cli_runner: Any | None,
) -> tuple[SelectedNebiusCredentialResolver, Any]:
    expected = accepted_nebius_cli_identity(selection.nebius_executable)
    if (
        selection.resolved_nebius_executable != expected.resolved_executable
        or selection.nebius_cli_version != expected.version
        or selection.nebius_executable_byte_count != expected.byte_count
        or selection.nebius_executable_sha256 != expected.sha256
    ):
        raise ValueError("selection changed the accepted native CLI identity")
    runner = credential_cli_runner or make_accepted_nebius_cli_runner(
        selection.nebius_executable
    )
    resolver = SelectedNebiusCredentialResolver(
        config.nebius,
        selection,
        payload_fetcher=payload_fetcher,
        cli_runner=runner,
    )
    return resolver, runner


def make_nebius_storage_readiness_adapter(
    config: ExecutionStorageReadinessConfig,
    selection: NebiusExecutionContextSelection,
    *,
    payload_fetcher: Callable[[str, str], str] | None = None,
    credential_cli_runner: Any | None = None,
) -> NebiusStorageReadinessAdapter:
    """Construct the production input-storage Adapter around the selected resolver."""

    resolver, runner = _accepted_selected_resolver(
        config,
        selection,
        payload_fetcher=payload_fetcher,
        credential_cli_runner=credential_cli_runner,
    )
    return NebiusStorageReadinessAdapter(
        credential_resolver=resolver,
        cli_runner_factory=lambda _: runner,
    )


def make_nebius_object_storage_tracer_adapters(
    config: ExecutionStorageReadinessConfig,
    selection: NebiusExecutionContextSelection,
    contact_observer: Callable[[ReadContact], None],
    *,
    payload_fetcher: Callable[[str, str], str] | None = None,
    credential_cli_runner: Any | None = None,
) -> _ProductionObjectStorageTracerAdapters:
    """Build the public tracer around one selected resolver and shared mechanics."""

    resolver, runner = _accepted_selected_resolver(
        config,
        selection,
        payload_fetcher=payload_fetcher,
        credential_cli_runner=credential_cli_runner,
    )
    return _ProductionObjectStorageTracerAdapters(
        readiness=NebiusStorageReadinessAdapter(
            credential_resolver=resolver,
            cli_runner_factory=lambda _: runner,
            contact_observer=contact_observer,
        ),
        objects=NebiusObjectStorageMechanics(
            endpoint_url=config.object_storage.endpoint_url,
            region=config.object_storage.region,
            bucket=config.bucket_name,
            credential_resolver=resolver,
            contact_observer=contact_observer,
        ),
    )


__all__ = [
    "make_nebius_object_storage_tracer_adapters",
    "make_nebius_storage_readiness_adapter",
]
