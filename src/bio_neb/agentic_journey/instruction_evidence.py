"""Read an Attempt's instructions from its own immutable Journey artifact."""

import re
from typing import Any, Mapping

from bio_neb.agentic_journey.role_instructions import RoleInstructions
from bio_neb.journey import ArtifactReference, Journey

INSTRUCTION_MEDIA_TYPE = "text/plain; charset=utf-8"


def retained_instruction_reference(started: Mapping[str, Any]) -> ArtifactReference:
    """Validate the retained role:version:form and its exact artifact ownership.

    The grammar is checked, never whether the version or form still ships: a
    retained Attempt reads its own artifact. A 2-part identity is refused (#441).
    """
    payload = started["payload"]
    role = payload["agent_role"]
    identity = payload.get("instruction_identity")
    references = started.get("artifact_references", ())
    if (
        started.get("event_type") != "AttemptStarted"
        or not isinstance(identity, str)
        or re.fullmatch(re.escape(role) + r":v[1-9][0-9]*:[a-z][a-z0-9_]*", identity)
        is None
        or len(references) != 1
    ):
        raise ValueError(
            "Attempt instruction identity requires its exact role and artifact"
        )
    reference = ArtifactReference.model_validate(references[0])
    if (
        reference.source_event_id != started["event_id"]
        or reference.journey_id != started["journey_id"]
        or reference.media_type != INSTRUCTION_MEDIA_TYPE
    ):
        raise ValueError("Attempt instruction artifact changed its retained binding")
    return reference


def read_attempt_instructions(journey: Journey, attempt_id: str) -> RoleInstructions:
    """Answer which instructions ran, without consulting installed versions/defaults.

    Documented reader: no product path calls it; tests and inspection read retained evidence through it.
    """
    from bio_neb.recording import read_evidence

    starts = [
        event
        for event in read_evidence(journey)
        if event["event_type"] == "AttemptStarted"
        and event["payload"].get("attempt_id") == attempt_id
    ]
    if len(starts) != 1:
        raise ValueError("instruction lookup requires one exact retained Attempt")
    started = starts[0]
    reference = retained_instruction_reference(started)
    role, version, form = started["payload"]["instruction_identity"].split(":")
    return RoleInstructions(
        role, version, form, journey.read_artifact(reference).decode("utf-8")
    )
