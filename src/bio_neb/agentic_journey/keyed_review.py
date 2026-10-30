"""One keyed Run-level Review per cause (owner, #476 alignment G4.1; #477).

The failure module keys a fault by its cause. The first item to hit it opens
the Review; every other item with that cause joins it, at its failure or before
it would send to the failing endpoint or source. Each joined item is answered on
its own (owner, 2026-10-28, #477 comment 6483312280), and the Review closes when
its last item is answered. The model endpoint's availability Review and the
operational (source and provider) Review share this one reading of which Review
is open; each owner supplies its own fact types and what a cause matches.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def open_keyed_review(
    events: Iterable[Mapping[str, Any]],
    *,
    requested: str,
    responded: str,
    matches: Callable[[Mapping[str, Any]], bool],
) -> str | None:
    """The open Review a new item joins, or None.

    A Review is the first item's request (it joins no other) together with the
    items that joined it; it is open while any of those items is unanswered.
    ``matches`` reads a lead request's payload: its cause's key, or the endpoint
    or surface the new item is about to reach.
    """
    events = list(events)
    answered = {
        event["payload"]["request_event_id"]
        for event in events
        if event["event_type"] == responded
    }
    unanswered_leads = set()
    leads = []
    for event in events:
        if event["event_type"] != requested:
            continue
        lead_id = event["payload"].get("joins_event_id")
        if lead_id is None:
            leads.append(event)
            lead_id = event["event_id"]
        if event["event_id"] not in answered:
            unanswered_leads.add(lead_id)
    return next(
        (
            lead["event_id"]
            for lead in leads
            if lead["event_id"] in unanswered_leads and matches(lead["payload"])
        ),
        None,
    )
