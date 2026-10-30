"""Read-only assignment context and the bounded same-Run Journey search."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from bio_neb.agentic_journey.action_policy import call_identity
from bio_neb.agentic_journey.tool_execution import (
    ModelView,
    ToolPrecondition,
    function_tool_contract,
)
from bio_neb.application.planning_read import read_planning
from bio_neb.content import canonical_json, sha256_bytes
from bio_neb.recording import read_evidence

# A search page must leave room for current context and subsequent native turns.
# Half the page is reserved for inline facts; provenance uses the rest.
_SEARCH_INLINE_PAYLOAD_BYTES = 32 * 1024
_SEARCH_PAGE_BYTES = 64 * 1024
_SEARCH_PAYLOAD_SLICE_CHARACTERS = 4096
SEARCH_TEXT_CHARACTERS = 200
SEARCH_FILTER_ITEMS = 20
SEARCH_TOOL_NAME = "search_journey"
# The model sees no recorded Tool input: it repeats its fact's work item and reads
# as a call to copy, while each offered Tool states its own arguments. Facts keep
# their results.
_CALL_SHAPES = {"facts": {"__all__": {"tool_input"}}}


class JourneySearchRefused(ValueError):
    """A search the Run's retained Journey cannot answer, for one named cause.

    ``code`` is the refusal the model reads, explained by the search Tool's
    registry row (#360: every cause was once read as outside the Run); ``facts``
    are the typed values its sentence names (#409).
    """

    def __init__(
        self, code: str, message: str, facts: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.facts = dict(facts or {})


class JourneyPayloadSlice(BaseModel):
    """Read exact retained payload text without expanding a whole transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: str = Field(
        min_length=1,
        description="Exact same-Run event ID from a search result payload_reference.",
    )
    offset: StrictInt = Field(
        default=0,
        ge=0,
        description="Zero-based Unicode character offset in canonical JSON, or the previous slice's next_offset.",
    )
    length: StrictInt = Field(
        default=_SEARCH_PAYLOAD_SLICE_CHARACTERS,
        ge=1,
        le=_SEARCH_PAYLOAD_SLICE_CHARACTERS,
        description="Maximum characters to return; follow next_offset for the next bounded slice.",
    )


def journey_search_query(
    *,
    text=None,
    event_ids=(),
    kinds=(),
    limit=20,
    before_event_id=None,
    payload_slice=None,
    scope_event_id=None,
):
    """One normalized search receipt shape for both Agent callers and replay.

    A scoped search names the retained fact that lists the events it may read
    (``scope_event_id``); an unscoped search records no scope key.
    """
    query = {
        "event_ids": list(event_ids),
        "kinds": list(kinds),
        "limit": limit,
        "before_event_id": before_event_id,
    }
    if text is not None:
        query["text"] = text
    if payload_slice is not None:
        query["payload_slice"] = JourneyPayloadSlice.model_validate(
            payload_slice
        ).model_dump(mode="json")
    if scope_event_id is not None:
        query["scope_event_id"] = scope_event_id
    return query


def current_assignment_context(journey, objective, *, collection_member_id=None):
    """Useful current meaning with provenance; the Journey stays searchable on demand.

    An Investigation member's context holds that member's own evidence only.
    """
    events = read_evidence(journey)
    investigation = next(
        (e for e in events if e["event_type"] == "InvestigationStarted"), None
    )
    if investigation is not None:
        return _investigation_context(
            events, investigation, objective, collection_member_id
        )
    request = next(e for e in events if e["event_type"] == "ResearchRequestRegistered")
    registration = next(e for e in events if e["event_type"] == "RunInputsRegistered")
    planning = read_planning(journey)
    selection = journey.checked_view("input_selection")
    registered = journey.checked_view("investigation_request")
    return {
        "journey_id": str(journey.journey_id),
        "objective": objective,
        "original_intent": {
            "event_id": request["event_id"],
            "grounding": "human",
            "prompt": request["payload"]["prompt"],
        },
        # Each registered request file lists targets; its items stay searchable, so
        # a long list never fills this context. Planning cites each by event id.
        "registered_requests": [
            {
                "event_id": file.source_event_id,
                "request_file": file.request_file,
                "asserted_by": file.request.asserted_by,
                "target_count": len(file.request.items),
            }
            for file in registered
        ],
        "supplied_inputs": {
            "event_id": registration["event_id"],
            "input_set_id": registration["payload"]["input_set_id"],
            "inputs": registration["payload"]["inputs"],
            "current_selection": {
                "source_event_id": selection.source_event_id,
                "inputs": [i.model_dump(mode="json") for i in selection.inputs],
                "latest_observation": selection.inspections[-1]
                if selection.inspections
                else None,
                "correction_direction": selection.receipts[-1].model_dump(mode="json")
                if selection.correction_pending and selection.receipts
                else None,
            },
        },
        "requirements": None
        if planning.requirements is None
        else planning.requirements.model_dump(mode="json"),
        "reference": None
        if planning.reference is None
        else planning.reference.model_dump(mode="json"),
        "companions": None
        if planning.companions is None
        else planning.companions.model_dump(mode="json", exclude=_CALL_SHAPES),
        "selections": None
        if planning.execution is None
        else planning.execution.model_dump(
            mode="json", exclude={**_CALL_SHAPES, "recipe_tool_inputs": True}
        ),
        "open_questions": [
            q.model_dump(mode="json") for q in planning.hitl.pending_requests
        ],
        "authority": {
            "history_grants_effect_authority": False,
            "scope": "Only currently admitted assignment capabilities; historical approvals and User assertions grant no new execution or scientific authority.",
        },
    }


def search_journey(
    journey,
    *,
    text=None,
    event_ids=(),
    kinds=(),
    limit=20,
    before_event_id=None,
    payload_slice=None,
    scope_event_id=None,
):
    """Return isolated retained evidence, never an executable authority handle.

    Ids resolve only within the owning Journey's admitted snapshot. Transcripts
    and oversized payloads retain exact event references, payload hashes and
    original artifact links. They are never rewritten as scientific findings.
    """
    return search_journey_events(
        read_evidence(journey),
        str(journey.journey_id),
        text=text,
        event_ids=event_ids,
        kinds=kinds,
        limit=limit,
        before_event_id=before_event_id,
        payload_slice=payload_slice,
        scope_event_id=scope_event_id,
    )


def _scope(events, scope_event_id, named):
    """The events a scoped search may read: those its scope fact lists.

    The scope fact is a retained event of the prefix whose payload names
    ``evidence_event_ids``; any other named event of the Run is refused.
    """
    scope = next(
        (event for event in events if event["event_id"] == scope_event_id), None
    )
    if scope is None:
        raise ValueError("search scope is not a retained fact of this prefix")
    allowed = frozenset(scope["payload"]["evidence_event_ids"])
    if outside := tuple(identity for identity in named if identity not in allowed):
        raise JourneySearchRefused(
            "search_outside_scope",
            "searched evidence is outside this assignment's evidence scope",
            {"outside": outside},
        )
    return allowed


def search_journey_events(
    events,
    journey_id,
    *,
    text=None,
    event_ids=(),
    kinds=(),
    limit=20,
    before_event_id=None,
    payload_slice=None,
    representation="retained_evidence_v2",
    scope_event_id=None,
):
    """Interpret an exact admitted prefix, shared by live access and replay.

    The prefix is only read; the returned page holds the caller's own copies.
    A scoped search returns only the events its scope fact lists; it reads
    the whole prefix, which its result names, so replay finds its scope fact.
    """
    from copy import deepcopy

    if type(limit) is not int or not 1 <= limit <= 100:
        raise JourneySearchRefused(
            "search_limit_invalid", "search limit must be between 1 and 100"
        )
    if text is not None and (
        not isinstance(text, str) or not 1 <= len(text) <= SEARCH_TEXT_CHARACTERS
    ):
        raise JourneySearchRefused(
            "search_text_invalid", "search text must be 1 to 200 characters"
        )
    if len(event_ids) > SEARCH_FILTER_ITEMS or len(kinds) > SEARCH_FILTER_ITEMS:
        raise JourneySearchRefused(
            "search_filters_too_many", "search filters name at most 20 items each"
        )
    if representation not in {
        None,
        "retained_evidence_v2",
        "retained_payload_slice_v1",
    }:
        raise ValueError("unknown history representation")
    by_id = {event["event_id"]: index for index, event in enumerate(events)}
    if any(identity not in by_id for identity in event_ids) or (
        before_event_id is not None and before_event_id not in by_id
    ):
        raise JourneySearchRefused(
            "search_outside_run", "searched evidence is outside this Run"
        )
    allowed = None
    if scope_event_id is not None:
        allowed = _scope(
            events,
            scope_event_id,
            (
                *event_ids,
                *(() if before_event_id is None else (before_event_id,)),
                *(
                    ()
                    if not isinstance(payload_slice, dict)
                    or not isinstance(payload_slice.get("event_id"), str)
                    else (payload_slice["event_id"],)
                ),
            ),
        )
        through = events[-1]["event_id"]
    if before_event_id is not None:
        events = events[: by_id[before_event_id]]
    if allowed is not None:
        events = [event for event in events if event["event_id"] in allowed]
    if event_ids and any(
        identity not in {e["event_id"] for e in events} for identity in event_ids
    ):
        raise JourneySearchRefused(
            "search_outside_prefix",
            "searched evidence is outside the selected prefix",
        )
    if payload_slice is not None:
        if representation is None:
            raise ValueError("payload slice cannot use the legacy representation")
        if fields := tuple(
            name
            for name, sent in (
                ("text", text is not None),
                ("event_ids", bool(event_ids)),
                ("kinds", bool(kinds)),
                ("limit", limit != 20),
            )
            if sent
        ):
            raise JourneySearchRefused(
                "search_slice_with_page_filter",
                "payload slice cannot mix event page filters",
                {"fields": fields},
            )
        try:
            requested = JourneyPayloadSlice.model_validate(payload_slice)
        except ValidationError as error:
            raise JourneySearchRefused(
                "search_slice_invalid", "payload slice is not one exact slice"
            ) from error
        event = next(
            (event for event in events if event["event_id"] == requested.event_id), None
        )
        if event is None:
            raise JourneySearchRefused(
                "search_outside_prefix"
                if requested.event_id in by_id
                else "search_outside_run",
                "searched payload is outside this Run prefix",
            )
        # The append-only Journey is checked at open; the event ID is the identity.
        text = canonical_json(event["payload"])
        if requested.offset > len(text):
            raise JourneySearchRefused(
                "search_slice_offset_outside_text",
                "payload offset is outside retained text",
            )
        end = min(requested.offset + requested.length, len(text))
        return {
            "journey_id": str(journey_id),
            "through_event_id": events[-1]["event_id"] if allowed is None else through,
            "representation": "retained_payload_slice_v1",
            "payload_slice": {
                "event_id": requested.event_id,
                "offset": requested.offset,
                "text": text[requested.offset : end],
                "total_characters": len(text),
                "next_offset": end if end < len(text) else None,
            },
        }
    if representation == "retained_payload_slice_v1":
        raise ValueError("payload slice representation requires an exact slice query")
    needle = None if text is None else text.casefold()
    selected = [
        event
        for event in events
        if (not event_ids or event["event_id"] in event_ids)
        and (not kinds or event["event_type"] in kinds)
        # Text matches the retained event as recorded, case-insensitively.
        and (needle is None or needle in canonical_json(event).casefold())
    ]
    result = {
        "journey_id": str(journey_id),
        "through_event_id": (
            through
            if allowed is not None
            else events[-1]["event_id"]
            if events
            else None
        ),
        "events": deepcopy(selected[-limit:]),
        "has_earlier_matches": len(selected) > limit,
    }
    if representation is not None:
        result["representation"] = representation
        _bound_search_payloads(result)
    return result


def _bound_search_payloads(result):
    """Keep original locators and facts; never copy transcripts recursively."""
    indirect = {
        "ModelRequestRecorded",
        "ModelResponseRecorded",
        "ModelTransportRequestRecorded",
        "ModelTransportResponseRecorded",
        "JourneySearchRead",
    }
    remaining_payload_bytes = _SEARCH_INLINE_PAYLOAD_BYTES
    for event in result["events"]:
        payload = canonical_json(event["payload"]).encode("utf-8")
        if event["event_type"] in indirect or len(payload) > remaining_payload_bytes:
            event.pop("payload")
            event["payload_reference"] = {
                "event_id": event["event_id"],
                "sha256": sha256_bytes(payload),
                "byte_count": len(payload),
                "representation": "retained_event_payload",
            }
        else:
            remaining_payload_bytes -= len(payload)
    # A page is bounded as well as its inline payloads. Omitted older matches can
    # be paged with before_event_id; through_event_id still names the exact prefix.
    while len(canonical_json(result).encode("utf-8")) > _SEARCH_PAGE_BYTES:
        if len(result["events"]) <= 1:
            raise JourneySearchRefused(
                "search_page_too_large",
                "search event metadata exceeds the page byte budget",
            )
        result["events"].pop(0)
        result["has_earlier_matches"] = True


class JourneySearchRead(BaseModel):
    """A correlated search receipt, never scientific evidence or authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    event_type: Literal["JourneySearchRead"] = "JourneySearchRead"
    attempt_id: str
    query: dict[str, Any]
    result: dict[str, Any]


def search_action_errors(events, arguments, *, scope_event_id=None):
    """Admit a search of this Run's prefix, refusing it for its own cause.

    Admission validated the argument shape; arguments no query can take are
    ``search_arguments_invalid``. A scoped assignment's search is admitted only
    inside its scope.
    """
    try:
        search_journey_events(
            events,
            events[0]["journey_id"],
            **arguments,
            scope_event_id=scope_event_id,
        )
    except JourneySearchRefused as refused:
        return (refused.code,)
    except (ValueError, TypeError):
        return ("search_arguments_invalid",)
    return ()


def search_refusal_facts(events, arguments, code, *, scope_event_id=None):
    """The typed facts a refused search's sentence names, from the same read
    admission made (#409)."""
    try:
        search_journey_events(
            events,
            events[0]["journey_id"],
            **arguments,
            scope_event_id=scope_event_id,
        )
    except JourneySearchRefused as refused:
        return refused.facts if refused.code == code else {}
    except (ValueError, TypeError):
        return {}
    return {}


def assignment_admission_errors(
    action, *, events, domain_errors, search_scope_event_id=None
):
    """Admit the shared search Tool for every assignment; domains admit their own."""
    if action.tool_name == SEARCH_TOOL_NAME:
        return search_action_errors(
            events(), action.arguments, scope_event_id=search_scope_event_id
        )
    return domain_errors(action)


def assignment_repeat_identity(action):
    """The identity a domain refuses a repeated call of its own Tools by.

    The call as called, without the arguments its registry row declares
    result-shaping (``max_hits``: how much of one result comes back), so a read
    asked again for more or less of its result is the same read (owner, #395 9a).
    Every other row declares none. A search is never refused as a repeat:
    ``repeated_calls`` counts it on its result.
    """
    if action.tool_name == SEARCH_TOOL_NAME:
        return None
    from dataclasses import replace

    from bio_neb.agentic_journey.tool_execution import result_shaping_of

    shaping = result_shaping_of(action.tool_name)
    if shaping:
        action = replace(
            action,
            arguments={
                name: value
                for name, value in action.arguments.items()
                if name not in shaping
            },
        )
    return call_identity(action)


def validate_search_receipt(prefix, fact, arguments):
    query = fact["payload"]["query"]
    # A scoped receipt is re-read under its recorded scope; the assignment that
    # owns the Attempt checks that the scope is its own bound fact.
    scope = query["scope_event_id"] if "scope_event_id" in query else None
    if query != journey_search_query(**arguments, scope_event_id=scope):
        raise ValueError("search receipt changed its query")
    reread_search_receipt(prefix, fact)


def reread_search_receipt(prefix, fact, _owner=None) -> None:
    """The receipt's result is the Run's prefix read again through its query; it
    runs before the receipt is appended too (owner, #476 C8)."""
    payload = fact["payload"]
    query = payload["query"]
    # The search read the Run through the event its result names; a concurrent
    # sibling may write before this receipt, so it is re-read there.
    through = payload["result"].get("through_event_id")
    if through is not None:
        ids = [event["event_id"] for event in prefix]
        if through not in ids:
            raise ValueError("search receipt changed its exact Run prefix")
        end = ids.index(through)
        # A page before an event ends just before it, but the search read the
        # event it names, so the re-read keeps it.
        if query["before_event_id"] in ids:
            end = max(end, ids.index(query["before_event_id"]))
        prefix = prefix[: end + 1]
    result = search_journey_events(
        prefix,
        fact["journey_id"],
        **query,
        representation=payload["result"].get("representation"),
    )
    if result != payload["result"]:
        raise ValueError("search receipt changed its exact Run prefix")


def _investigation_context(events, started, objective, member_id):
    """The scientific caller keeps assertions, observed sources and choices distinct.

    Each member's current evidence is its own latest source results (#360)."""
    from bio_neb.investigation.member_context import latest_source_results

    return member_assignment(
        started, objective, latest_source_results(events, member_id)
    )


def member_assignment(started, objective, results):
    """An Investigation member's assignment context from its exact source results.

    Every member's original intent is the research goal in the researcher's words;
    each member's own request is its ``user_assertion``, never another member's.
    Its current evidence is ``results``, its own latest source result of each
    kind: another member's, and the collection's proposals and Reviews that hold
    them, stay searchable (#360)."""
    from bio_neb.investigation.investigation import UserAssertion

    assertion = UserAssertion.model_validate_json(
        canonical_json(started["payload"]["assertion"]), strict=True
    )
    return {
        "journey_id": started["journey_id"],
        "objective": objective,
        "original_intent": {
            "event_id": started["event_id"],
            "grounding": "user_assertion",
            "research_goal": list(assertion.research_goals),
        },
        "supplied_inputs": {
            "event_id": started["event_id"],
            **started["payload"]["investigation_input"],
        },
        "current_evidence": {
            kind: {"event_id": event["event_id"], "payload": event["payload"]}
            for kind, event in results.items()
        },
        "authority": {
            "history_grants_effect_authority": False,
            "scope": "Only the current Investigation source operations are admitted; historical approvals and User assertions do not authorize observation, execution, provider or storage effects.",
        },
    }


# The shared Journey search's refusals, one per cause the search names; every
# role's search row declares them.
SEARCH_PRECONDITIONS = (
    ToolPrecondition(
        "search_outside_run",
        "query",
        "The search must name events, a page or a payload slice inside this Run's retained Journey.",
    ),
    ToolPrecondition(
        "search_outside_prefix",
        "query",
        "A named event comes at or after before_event_id; name earlier events or leave before_event_id out.",
    ),
    ToolPrecondition(
        "search_slice_with_page_filter",
        "payload_slice",
        "A payload slice read takes no other filter; you also sent {fields}. Send payload_slice alone.",
    ),
    ToolPrecondition(
        "search_slice_invalid",
        "payload_slice",
        "A slice names one event_id from a payload_reference, an offset of 0 or more and a length of 1 to 4096 characters, and no other field.",
    ),
    ToolPrecondition(
        "search_slice_offset_outside_text",
        "payload_slice",
        "The offset is past the end of that event's retained text; use the previous slice's next_offset.",
    ),
    ToolPrecondition(
        "search_limit_invalid",
        "limit",
        "The page limit is a whole number from 1 to 100.",
    ),
    ToolPrecondition(
        "search_text_invalid",
        "text",
        "Search text is 1 to 200 characters.",
    ),
    ToolPrecondition(
        "search_filters_too_many",
        "query",
        "event_ids and kinds name at most 20 items each.",
    ),
    ToolPrecondition(
        "search_page_too_large",
        "query",
        "One matching event's metadata alone exceeds the page budget; narrow the search by kind or event id.",
    ),
    ToolPrecondition(
        "search_arguments_invalid",
        "query",
        "These arguments form no single search: send text, kinds, event_ids, limit and before_event_id, or payload_slice alone.",
    ),
)


# A scoped search's refusals: the shared ones and the one its scope adds.
SCOPED_SEARCH_PRECONDITIONS = (
    *SEARCH_PRECONDITIONS,
    ToolPrecondition(
        "search_outside_scope",
        "query",
        "A named event is outside the evidence this assignment may read: {outside}. Name only events from its evidence index.",
    ),
)


# What a search shows: the recorded events it matched, as recorded, for the model
# to read and cite.
SEARCH_VIEW = ModelView(
    next_step={
        "*": "Read the matched events above and cite their event ids; a search "
        "grants no effect authority."
    },
    status=lambda _payload: "read",
    result=lambda payload: payload["result"],
    scrubbed=False,
)


# This domain's Tool rows (ADR 0040): handler, row, bound values and refusal
# facts sit together, and the loop assembles every domain's rows into the one
# inventory (`tool_execution.tool_row_modules`).
TOOL_ROWS = (
    function_tool_contract(
        "search_journey",
        "planning_agent",
        (
            "text: str | None",
            "event_ids: list[str]",
            "kinds: list[str]",
            "limit: int",
            "before_event_id: str | None",
        ),
        "JourneySearchRead",
        "journey_read",
        "JourneySearchRead",
        None,
        None,
        SEARCH_VIEW,
        "generic_mechanical_failure_only",
        stages=(),
        examples=(
            '{"text": "HG001", "limit": 5}',
            '{"text": "MM-04", "limit": 5}',
            '{"kinds": ["PlanningHitlResponded"]}',
        ),
        preconditions=SEARCH_PRECONDITIONS,
    ),
    function_tool_contract(
        "search_journey",
        "investigation_planner",
        (
            "text: str | None",
            "event_ids: list[str]",
            "kinds: list[str]",
            "limit: int",
            "before_event_id: str | None",
        ),
        "JourneySearchRead",
        "journey_read",
        "JourneySearchRead",
        None,
        None,
        SEARCH_VIEW,
        "generic_mechanical_failure_only",
        stages=(),
        examples=(
            '{"text": "HG001", "limit": 5}',
            '{"text": "MM-04", "limit": 5}',
            '{"kinds": ["PlanningHitlResponded"]}',
        ),
        preconditions=SEARCH_PRECONDITIONS,
    ),
    function_tool_contract(
        "search_journey",
        "table_reader",
        (
            "text: str | None",
            "event_ids: list[str]",
            "kinds: list[str]",
            "limit: int",
            "before_event_id: str | None",
        ),
        "JourneySearchRead",
        "journey_read",
        "JourneySearchRead",
        None,
        None,
        SEARCH_VIEW,
        "generic_mechanical_failure_only",
        stages=(),
        examples=(
            '{"text": "<an rsID this Run states>", "limit": 5}',
            '{"kinds": ["InvestigationEvidenceRecorded"], "limit": 3}',
        ),
        preconditions=SEARCH_PRECONDITIONS,
    ),
    function_tool_contract(
        "search_journey",
        "report_explainer",
        (
            "text: str | None",
            "event_ids: list[str]",
            "kinds: list[str]",
            "limit: int",
            "before_event_id: str | None",
        ),
        "JourneySearchRead",
        "journey_read",
        "JourneySearchRead",
        None,
        None,
        SEARCH_VIEW,
        "generic_mechanical_failure_only",
        stages=(),
        examples=(
            '{"kinds": ["InvestigationEvidenceRecorded"], "limit": 3}',
            '{"text": "<an rsID this Run states>", "limit": 5}',
        ),
        preconditions=SCOPED_SEARCH_PRECONDITIONS,
    ),
    function_tool_contract(
        "search_journey",
        "paper_reader",
        (
            "text: str | None",
            "event_ids: list[str]",
            "kinds: list[str]",
            "limit: int",
            "before_event_id: str | None",
        ),
        "JourneySearchRead",
        "journey_read",
        "JourneySearchRead",
        None,
        None,
        SEARCH_VIEW,
        "generic_mechanical_failure_only",
        stages=(),
        examples=(
            '{"kinds": ["ReadingItemCompleted"], "limit": 5}',
            '{"kinds": ["PaperReadingRecorded"], "text": "TP53"}',
        ),
        preconditions=SEARCH_PRECONDITIONS,
    ),
)
