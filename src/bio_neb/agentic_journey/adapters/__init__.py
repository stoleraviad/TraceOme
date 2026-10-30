"""External mechanics and dependency-light Adapter construction entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bio_neb.agentic_journey.runtime_config import (
    LoadedRuntimeConfig,
    NebiusModelEndpointSelection,
)

NebiusModelEndpointFactory = Callable[
    [LoadedRuntimeConfig, NebiusModelEndpointSelection], Any
]


def create_nebius_model_endpoint(
    loaded: LoadedRuntimeConfig, selected: NebiusModelEndpointSelection
) -> Any:
    """Construct the concrete Adapter only on an authorized model run path."""

    from bio_neb.providers.nebius_model_endpoint import (
        NebiusModelEndpointAdapter,
    )

    return NebiusModelEndpointAdapter(loaded, selected)


__all__ = ["NebiusModelEndpointFactory", "create_nebius_model_endpoint"]
