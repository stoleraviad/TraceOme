"""Source contacts through the product's failure module, one Tool surface each.

Every dbSNP and EVA GET runs inside `call_source`, which applies `call_with_failure_policy`
under the policy `policy_for` resolves for the contact's Tool surface: a transient fault (a 5xx,
429 or a lost connection) is retried within the policy's allowance, then reclassified
`exhausted`; any other classified fault is not retried. No Journey scope is open here, so the
module records nothing; the fault it gave up on becomes the claim's own `SourceFailure`, with
the route the policy gives it (`ask` by default), and the other claims still run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from bio_neb.agentic_journey.source_contacts import call_source
from bio_neb.failure import FailureSurface, ProviderError, policy_for

from .records import SourceFailure

T = TypeVar("T")


class SourceFailed(Exception):
    """A source contact ended in a fault the failure module routed; the claim records it."""

    def __init__(self, failure: SourceFailure) -> None:
        super().__init__(failure.detail)
        self.failure = failure


def through_failure_module(tool: str, operation: Callable[[], T]) -> T:
    """`operation` under `tool`'s failure policy; a routed fault raises `SourceFailed`."""
    try:
        return call_source(tool, operation)
    except ProviderError as error:
        policy = policy_for(FailureSurface(scope="tool", name=tool))
        raise SourceFailed(
            SourceFailure(
                tool=tool,
                failure_class=error.failure_class,
                original_failure_class=error.evidence["original_failure_class"],
                contact_attempts=error.evidence["contact_attempt"],
                route=policy.effective.route(error.failure_class),
                detail=error.detail,
                policy=policy,
            )
        ) from error
