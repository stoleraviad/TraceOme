"""Review vocabulary: the asks a held fault opens and the answers they take."""

from typing import Literal

from pydantic import (
    Field,
    JsonValue,
    model_validator,
)

from bio_neb.contracts import FrozenContract


def continuation_destination(request, answer):
    """Select only the exact stop's declared consequence, without domain routing.

    Domains declare legal actions and each one's destination. Shared policy checks
    that declaration; the Run coordinator dispatches the selected owner.
    """
    if (
        answer.request_id != request.request_id
        or answer.action not in request.legal_actions
    ):
        raise ValueError("continuation answer exceeds its exact request")
    destination = request.action_destinations.get(answer.action)
    if destination is None:
        raise ValueError("continuation request does not declare this destination")
    return destination


class Ask(FrozenContract):
    """A failure-review intent, also the retained request payload.

    Only declared actions may be answered. The engine adds field-scoped
    amendments where the stage and plan allow them. CPU approval stays separate.
    """

    state: Literal["pending"] = "pending"
    stage: str = Field(min_length=1)
    failure_event_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    legal_actions: tuple[
        Literal["retry", "abort", "amend", "stop", "continue"], ...
    ] = Field(default=("retry", "abort"), min_length=1)

    amendable: tuple[str, ...] = ()
    retry_consequence: Literal["reuse_intent", "replace_job"] = "reuse_intent"
    amendment_consequences: dict[str, Literal["new_job_intent"]] = Field(
        default_factory=dict
    )


def retry_consequence(context):
    """Only retained terminal resource evidence offers replacement on retry."""
    if (
        context.get("job_name")
        and context.get("provider_job_id")
        and context.get("terminal_state") in {"FAILED", "CANCELLED", "ERROR"}
    ):
        return "replace_job"
    return "reuse_intent"


class ReviewAnswer(FrozenContract):
    state: Literal["answered"] = "answered"
    request_id: str = Field(min_length=1)
    action: Literal[
        "retry", "abort", "approve_cpu_preparation", "amend", "stop", "continue"
    ]
    by: str = Field(min_length=1)

    reason: str | None = Field(default=None, min_length=1, max_length=2000)
    field: str | None = None
    value: JsonValue = None

    @model_validator(mode="after")
    def amendment(self):
        if (self.action == "amend") != (self.field is not None):
            raise ValueError(
                "amend requires a declared field; other actions have no field"
            )
        if self.action != "amend" and self.value is not None:
            raise ValueError("only amendment carries a value")
        return self


# An item whose check ended with an issue. Its verdict stays the domain's
# result, never a fault class; the failure owner keys it and asks the item's own
# Review at once, with no correction allowance. The researcher keeps the finding
# as checked but not verified (approve), amends it, which sends that item alone
# back, or approves removing the item (reject), while other work continues.
ITEM_ISSUE_ACTIONS = ("approve", "amend", "reject")


class ItemIssue(FrozenContract):
    """One item's issue, keyed by the source and verdict its finding names.

    ``finding_event_id`` is the accepted finding (the validated output) the issue
    keys; ``evidence_event_ids`` the source results that finding cites.
    """

    key: str = Field(min_length=1)
    source: str = Field(min_length=1)
    verdict: str = Field(min_length=1)
    conflicts: tuple[str, ...] = Field(min_length=1)
    finding_event_id: str = Field(min_length=1)
    evidence_event_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def keyed(self):
        if self.key != item_issue_key(self.source, self.verdict):
            raise ValueError("an item issue is keyed by its source and verdict")
        return self


def item_issue_key(source: str, verdict: str) -> str:
    """The issue's key: the same source verdict is the same issue for any item."""
    return f"{source}: {verdict}"
