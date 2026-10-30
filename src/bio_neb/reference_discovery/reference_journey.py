"""Typed Journey facts and contact-free projection for reference identity work."""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from bio_neb.agentic_journey.source_answers import SourceAnswer
from bio_neb.identifier_rule import word_aligned_spans
from bio_neb.journey import JourneyEvent, ReadOnlyJsonObject
from bio_neb.parabricks.planning_hitl import (
    PlanningHitlProjection,
    PlanningHitlResponded,
)
from bio_neb.parabricks.proposal_information import ProposalInformationRequirement
from bio_neb.parabricks.proposal_journey import (
    ProposalRequirementsIdentified,
)
from bio_neb.reference_discovery.assembly_suggestion import SuggestedTarget
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_RULE,
    SAMPLE_IDENTIFIER_RULE,
    AssemblySourceRecord,
    DescribeReferenceInput,
    DescribeReferenceObserved,
    DescribeReferenceResult,
    DiscoverAssembliesObserved,
    DiscoverReferenceAssembliesInput,
    DiscoverReferenceAssembliesResult,
    RequirementGrounding,
    ResolvedTaxonEvidence,
    ResolveTaxonInput,
    ResolveTaxonObserved,
    ResolveTaxonResult,
    SampleAttestationInput,
    SampleAttestationResult,
    SuppliedAssemblyEvidence,
    ValidatedReference,
)

if TYPE_CHECKING:
    from bio_neb.application.planning_read import PrefixSnapshot


def _canonical_uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: StrictStr

    @model_validator(mode="after")
    def canonical_attempt(self) -> "_Fact":
        _canonical_uuid(self.attempt_id, "attempt_id")
        return self


class ReferenceSourceRetrieved(_Fact):
    """One GET a planning source lookup made and its source answered, its bytes
    retained on this fact: the lookup's fact follows its GETs in call order (#458)."""

    event_type: Literal["ReferenceSourceRetrieved"] = "ReferenceSourceRetrieved"
    operation: StrictStr
    request_identity: StrictStr
    status_code: StrictInt = Field(ge=100, le=599)
    byte_count: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


_SOURCE_LOOKUP_FACTS = frozenset(
    {
        "TaxonResolutionRecorded",
        "ReferenceAssembliesRecorded",
        "ReferenceDescriptionRecorded",
        "CompanionObservationRecorded",
        "PrebuiltIndexesListed",
        "PrebuiltIndexRead",
        "AssemblyWithIndexSuggested",
        "PaperSearchRecorded",
    }
)


def validate_source_lookup_gets(by_id: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a planning source lookup whose shown answers are not exactly its own
    GETs: the retained `ReferenceSourceRetrieved` records on its causal chain, in call
    order, each holding its own bytes (#458 Standards r2).

    ``by_id`` is a prefix in order whose every parent is an earlier event
    (`PrefixSnapshot` checks that first), so each walk back ends."""
    for event in by_id.values():
        if event["event_type"] == "ReferenceSourceRetrieved":
            retained = ReferenceSourceRetrieved.model_validate(
                {"event_type": event["event_type"], **event["payload"]}
            )
            references = event["artifact_references"]
            if len(references) != 1 or (
                references[0]["artifact_id"],
                references[0]["byte_count"],
                references[0]["source_event_id"],
            ) != (f"sha256:{retained.sha256}", retained.byte_count, event["event_id"]):
                raise ValueError("a planning source GET changed its retained bytes")
        if event["event_type"] not in _SOURCE_LOOKUP_FACTS:
            continue
        if "source_answers" not in event["payload"]:
            raise ValueError("a planning source lookup lost its source answers")
        gets = []
        parents = event["parent_ids"]
        while len(parents) == 1 and (
            by_id[parents[0]]["event_type"] == "ReferenceSourceRetrieved"
        ):
            got = by_id[parents[0]]
            if got["payload"]["attempt_id"] != event["payload"]["attempt_id"]:
                break
            gets.append(
                tuple(
                    got["payload"][field]
                    for field in (
                        "operation",
                        "request_identity",
                        "status_code",
                        "byte_count",
                        "sha256",
                    )
                )
            )
            parents = got["parent_ids"]
        shown = [
            (
                answer["operation"],
                answer["request_identity"],
                answer["status_code"],
                answer["byte_count"],
                answer["sha256"],
            )
            for answer in event["payload"]["source_answers"]
        ]
        if shown != gets[::-1]:
            raise ValueError("a planning source lookup changed its source answers")


class SampleAttestationRecorded(_Fact):
    event_type: Literal["SampleAttestationRecorded"] = "SampleAttestationRecorded"
    tool_input: SampleAttestationInput
    result: SampleAttestationResult

    @model_validator(mode="after")
    def exact_result(self) -> "SampleAttestationRecorded":
        if (
            self.result.grounding != self.tool_input.grounding
            or self.result.input_set_id != self.tool_input.input_set_id
            or self.result.sample_identifier != self.tool_input.sample_identifier
            or self.result.assertion_event_id != self.tool_input.assertion_event_id
        ):
            raise ValueError("sample attestation result changed its exact Tool input")
        return self


class TaxonResolutionRecorded(_Fact):
    event_type: Literal["TaxonResolutionRecorded"] = "TaxonResolutionRecorded"
    tool_input: ResolveTaxonInput
    result: ResolveTaxonResult
    # What each of the Tool's GETs' sources answered, as the model is shown it; each
    # GET's bytes are retained on its own `ReferenceSourceRetrieved`.
    source_answers: tuple[SourceAnswer, ...]

    @model_validator(mode="after")
    def exact_result(self) -> "TaxonResolutionRecorded":
        if self.result.grounding != self.tool_input.grounding:
            raise ValueError("taxon result changed its requirement grounding")
        if isinstance(self.result, ResolveTaxonObserved) and (
            self.result.query != self.tool_input.query
        ):
            raise ValueError("taxon result changed its exact query")
        return self


class ReferenceAssembliesRecorded(_Fact):
    event_type: Literal["ReferenceAssembliesRecorded"] = "ReferenceAssembliesRecorded"
    tool_input: DiscoverReferenceAssembliesInput
    result: DiscoverReferenceAssembliesResult
    # What each of the Tool's GETs' sources answered, as the model is shown it; each
    # GET's bytes are retained on its own `ReferenceSourceRetrieved`.
    source_answers: tuple[SourceAnswer, ...]

    @model_validator(mode="after")
    def exact_result(self) -> "ReferenceAssembliesRecorded":
        if self.result.grounding != self.tool_input.grounding:
            raise ValueError("reference discovery changed its requirement grounding")
        if isinstance(self.result, DiscoverAssembliesObserved) and (
            self.result.taxon != self.tool_input.taxon
        ):
            raise ValueError("reference discovery changed its resolved taxon")
        return self


class ReferenceDescriptionRecorded(_Fact):
    event_type: Literal["ReferenceDescriptionRecorded"] = "ReferenceDescriptionRecorded"
    tool_input: DescribeReferenceInput
    result: DescribeReferenceResult
    # What each of the Tool's GETs' sources answered, as the model is shown it; each
    # GET's bytes are retained on its own `ReferenceSourceRetrieved`.
    source_answers: tuple[SourceAnswer, ...]

    @model_validator(mode="after")
    def exact_result(self) -> "ReferenceDescriptionRecorded":
        if self.result.grounding != self.tool_input.grounding:
            raise ValueError("reference description changed its requirement grounding")
        if isinstance(self.result, DescribeReferenceObserved) and (
            self.result.selected != self.tool_input.selected
            or self.result.assembly_accession != self.tool_input.assembly_accession
        ):
            raise ValueError("reference description changed its selected assembly")
        return self


class ReferenceChoiceRecorded(_Fact):
    """A reasoned choice or explicit ambiguity, never exact validation.

    A discovered choice names a row of the discovery page it cites and the
    researcher's words it rests on (#471 Q2 a); explicit ambiguity names the page and
    no row; a supplied choice names the researcher's own accession, the statement that
    supplies it and its words there (#472 Q2 a)."""

    event_type: Literal["ReferenceChoiceRecorded"] = "ReferenceChoiceRecorded"
    grounding: RequirementGrounding
    discovery_event_id: StrictStr | None = None
    assembly_accession: StrictStr | None = None
    quote: StrictStr | None = Field(default=None, min_length=1, max_length=300)
    assertion_event_id: StrictStr | None = None
    reason: StrictStr = Field(min_length=1, max_length=2000)
    # The reading fact whose kept span holds a discovered choice's quote, when the
    # quote stands in a cited paper's words rather than the researcher's; left out
    # otherwise, so earlier choices keep their bytes.
    quote_event_id: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def exact_choice(self):
        if self.discovery_event_id is not None:
            _canonical_uuid(self.discovery_event_id, "discovery_event_id")
            if self.assertion_event_id is not None:
                raise ValueError(
                    "discovery choice cannot manufacture a supplied assertion"
                )
            if (self.assembly_accession is None) != (self.quote is None):
                raise ValueError("a discovered choice names its row and its quote")
        elif self.quote_event_id is not None:
            raise ValueError("a supplied choice cites its statement, not a quote fact")
        elif (
            self.assembly_accession is None
            or self.assertion_event_id is None
            or self.quote is None
        ):
            raise ValueError("direct choice needs exact supplied-reference provenance")
        if self.quote is not None and self.quote.strip() != self.quote:
            raise ValueError("a choice's quote is exact text")
        if self.reason.strip() != self.reason or not self.reason.strip():
            raise ValueError("reference choice needs a concise reason")
        return self


def researcher_text(event) -> str | None:
    """The words one researcher statement writes: the request's prompt, or the
    instruction of a planning answer or an Execution Review's reference revision; None
    for any other event and for an answer that writes none."""
    payload = event["payload"]
    if event["event_type"] == "ResearchRequestRegistered":
        return payload["prompt"]
    if _researcher_statement(event):
        # Only a clarify, select or revise answer writes an instruction.
        return payload["response"].get("instruction") or None
    return None


def researcher_words(events) -> tuple[str, ...]:
    """The researcher's own words a reference choice may quote: the request and each
    answer's instruction (planning Reviews, and an Execution Review's reference
    revision)."""
    return tuple(
        text for event in events if (text := researcher_text(event)) is not None
    )


def quoted(words: tuple[str, ...], quote: str) -> bool:
    """Whether ``quote`` stands, whole words, in one of the researcher's texts."""
    return any(word_aligned_spans(text, quote) for text in words)


def paper_quote_event(events, quote: str) -> str | None:
    """The first reading fact whose kept span holds ``quote`` on whole words: a cited
    paper's words, copied by code from its stored text (#383 A2, ADR 0049
    membership); None when no reading kept it."""
    from bio_neb.paper_reading.planning import kept_statements

    return next(
        (
            event_id
            for event_id, texts in kept_statements(events)
            if quoted(texts, quote)
        ),
        None,
    )


def statement_texts(events, event_id: str) -> tuple[str, ...]:
    """The words one cited statement writes: a researcher statement's text, or the
    spans a reading fact of a cited paper keeps (#383 A2)."""
    from bio_neb.paper_reading.planning import kept_texts

    text = next(
        (researcher_text(event) for event in events if event["event_id"] == event_id),
        None,
    )
    return (text,) if text is not None else kept_texts(events, event_id)


def cited_statement_states(events, event_id: str, quote: str, rule, value: str) -> bool:
    """``statement_states`` over a researcher statement or a reading fact of a cited
    paper: the quote stands, whole words, in the words it writes, and states
    ``value`` under ``rule`` (#383 A2)."""
    return any(
        word_aligned_spans(text, quote) for text in statement_texts(events, event_id)
    ) and rule.matches(value, quote)


def statement_states(events, event_id: str, quote: str, rule, value: str) -> bool:
    """Whether the researcher statement ``event_id`` holds ``quote`` on whole words and
    the quote states ``value`` under ``rule`` (#472 Q2 a): the membership check for a
    value the model typed from a researcher's answer or request."""
    text = next(
        (researcher_text(event) for event in events if event["event_id"] == event_id),
        None,
    )
    return (
        text is not None
        and bool(word_aligned_spans(text, quote))
        and rule.matches(value, quote)
    )


class ReferenceChoiceView(ReferenceChoiceRecorded):
    source_event_id: StrictStr


ReferenceIdentityFact = (
    SampleAttestationRecorded
    | TaxonResolutionRecorded
    | ReferenceAssembliesRecorded
    | ReferenceDescriptionRecorded
)

_FACT_TYPES = {
    owner.model_fields["event_type"].default: owner
    for owner in (
        SampleAttestationRecorded,
        TaxonResolutionRecorded,
        ReferenceAssembliesRecorded,
        ReferenceDescriptionRecorded,
    )
}


def validate_reference_fact(
    event: Mapping[str, Any],
) -> ReferenceIdentityFact | None:
    owner = _FACT_TYPES.get(event.get("event_type"))
    if owner is None:
        return None
    return owner.model_validate(
        {"event_type": event["event_type"], **event.get("payload", {})}
    )


class ReferenceFactView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_event_id: StrictStr
    event_type: Literal[
        "SampleAttestationRecorded",
        "TaxonResolutionRecorded",
        "ReferenceAssembliesRecorded",
        "ReferenceDescriptionRecorded",
    ]
    attempt_id: StrictStr
    result: ReadOnlyJsonObject


_REQUIREMENT_ID = re.compile(r"^requirement_[0-9a-f]{32}$")
_SUBJECT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


class ReferenceRequirementLocator(BaseModel):
    """One immutable reference-work item from the accepted proposal plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: StrictStr
    kind: Literal["sample_identity", "taxon_identity", "reference_assembly_identity"]
    subject_key: StrictStr

    @model_validator(mode="after")
    def canonical_identity(self) -> "ReferenceRequirementLocator":
        if not _REQUIREMENT_ID.fullmatch(self.requirement_id):
            raise ValueError("reference requirement identity is not canonical")
        if not _SUBJECT_KEY.fullmatch(self.subject_key):
            raise ValueError("reference requirement subject is not canonical")
        return self


class ChosenSuggestion(BaseModel):
    """The researcher's answer choosing a suggested assembly with a fitting Galaxy
    index for the whole Run (#459 D4): the answer, the suggestion it answered, the
    assembly it replaced, the chosen assembly and index, and the known targets as the
    suggestion checked them there (D3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_event_id: StrictStr
    suggestion_event_id: StrictStr
    requested_assembly: StrictStr
    assembly_accession: StrictStr
    candidate_id: StrictStr
    targets: Tuple[SuggestedTarget, ...]


class ReferenceIdentityProjection(BaseModel):
    """The current reference frontier, derived only from retained Journey facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    journey_id: StrictStr
    plan_event_id: StrictStr
    plan_id: StrictStr
    input_set_id: StrictStr
    requirements: Tuple[ReferenceRequirementLocator, ...]
    status: Literal[
        "reference_identity_ready",
        "sample_identity",
        "taxon_identity",
        "reference_candidates",
        "planning_hitl_pending",
        "reference_identity",
    ]
    facts: Tuple[ReferenceFactView, ...]
    planning_hitl: PlanningHitlProjection
    sample_event_id: StrictStr | None = None
    resolved_taxon: ResolvedTaxonEvidence | None = None
    discovery_event_id: StrictStr | None = None
    # Every discovery page read since the current resolved taxon, oldest first; a
    # choice may cite any of them. ``discovery_event_id`` is the latest.
    discovery_event_ids: Tuple[StrictStr, ...] = ()
    description_event_id: StrictStr | None = None
    verified_reference: ValidatedReference | None = None
    choices: Tuple[ReferenceChoiceView, ...] = ()
    revision_event_id: StrictStr | None = None
    chosen_suggestion: ChosenSuggestion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The suggested assembly the researcher chose in place of the verified one (#459
    D4): the earlier reference's companion facts are history. A reference revise
    clears it."""

    @property
    def standing_suggestion(self) -> ChosenSuggestion | None:
        """The chosen suggestion while the verified reference is its assembly: what the
        Run's reference basis and its known targets read (#459)."""
        chosen = self.chosen_suggestion
        if (
            chosen is None
            or self.verified_reference is None
            or self.verified_reference.assembly_accession != chosen.assembly_accession
        ):
            return None
        return chosen

    @property
    def requested_assembly(self) -> str | None:
        """The assembly the researcher asked for, when a standing suggestion replaced
        it."""
        standing = self.standing_suggestion
        return None if standing is None else standing.requested_assembly

    @model_validator(mode="after")
    def exact_requirement_set(self) -> "ReferenceIdentityProjection":
        _canonical_uuid(self.journey_id, "journey_id")
        required = {
            "sample_identity",
            "taxon_identity",
            "reference_assembly_identity",
        }
        kinds = [item.kind for item in self.requirements]
        identities = [item.requirement_id for item in self.requirements]
        subjects = [item.subject_key for item in self.requirements]
        if (
            len(self.requirements) != 3
            or set(kinds) != required
            or len(identities) != len(set(identities))
            or len(subjects) != len(set(subjects))
        ):
            raise ValueError("reference projection changed its exact requirement set")
        return self


def _requirement(
    plan: ProposalRequirementsIdentified, kind: str
) -> ProposalInformationRequirement:
    selected = tuple(item for item in plan.plan.requirements if item.kind == kind)
    if not selected:
        raise ValueError(f"reference slice requires a {kind} requirement")
    return selected[0]


def _requirement_evidence_ids(
    requirement: ProposalInformationRequirement,
) -> set[str]:
    identities: set[str] = set()
    for evidence in requirement.evidence:
        value = evidence.model_dump(mode="json")
        identities.update(
            item
            for key, item in value.items()
            if key.endswith("event_id") and isinstance(item, str)
        )
    return identities


def _validate_grounding(
    grounding: RequirementGrounding,
    *,
    journey_id: str,
    plan_event_id: str,
    plan: ProposalRequirementsIdentified,
    requirement: ProposalInformationRequirement,
    ancestors: set[str],
) -> None:
    if (
        grounding.journey_id != journey_id
        or grounding.plan_id != plan.plan.plan_id
        or grounding.requirement_id != requirement.requirement_id
        or grounding.kind != requirement.kind
        or grounding.subject_key != requirement.subject_key
    ):
        raise ValueError("reference Tool grounding changed its exact plan requirement")
    required_sources = {plan_event_id, *_requirement_evidence_ids(requirement)}
    supplied_sources = set(grounding.source_event_ids)
    if not required_sources <= supplied_sources:
        raise ValueError("reference Tool grounding omitted immutable plan evidence")
    if not supplied_sources <= ancestors:
        raise ValueError("reference Tool grounding cites non-ancestral evidence")


def _owning_attempt(
    fact: ReferenceIdentityFact,
    *,
    ancestors: set[str],
    by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    starts = [
        event_id
        for event_id in ancestors
        if by_id[event_id].get("event_type") == "AttemptStarted"
        and by_id[event_id].get("payload", {}).get("attempt_id") == fact.attempt_id
    ]
    if len(starts) != 1:
        raise ValueError("reference Tool fact lacks its exact owning Attempt")


def project_reference_identity(
    events: Iterable[JourneyEvent | Mapping[str, Any]],
) -> ReferenceIdentityProjection:
    """Project exact reference progress without model, graph, adapter, or source contact."""

    from bio_neb.application.planning_read import PrefixSnapshot

    return PrefixSnapshot(events).reference


def _validate_description_basis(
    fact,
    choices,
    resolved_taxon,
    discoveries,
    receipts_by_event,
    plan,
    reference_requirement,
):
    grounding = fact.tool_input.grounding
    selected = fact.tool_input.selected
    if isinstance(selected, SuppliedAssemblyEvidence):
        choice = choices[-1] if choices else None
        if (
            choice is None
            or selected.selection_event_id != choice.source_event_id
            or selected.assembly_accession != choice.assembly_accession
            or selected.assertion_event_id != choice.assertion_event_id
            or selected.expected_taxon != resolved_taxon
            or selected.selection_event_id not in grounding.source_event_ids
            or selected.assertion_event_id not in grounding.source_event_ids
        ):
            raise ValueError("reference description changed its exact supplied basis")
        return
    discovery = discoveries.get(selected.discovery_event_id)
    if (
        discovery is None
        or selected.discovery_evidence_id != discovery.evidence_id
        or selected.row not in discovery.rows
        or selected.discovery_event_id not in grounding.source_event_ids
    ):
        raise ValueError("reference description changed its exact discovery selection")
    if any(c.source_event_id == selected.selection_event_id for c in choices):
        choice = choices[-1]
        if (
            choice.source_event_id != selected.selection_event_id
            or choice.discovery_event_id != selected.discovery_event_id
            or choice.assembly_accession != selected.row.assembly_accession
            or selected.selection_event_id not in grounding.source_event_ids
        ):
            raise ValueError("reference description changed its retained Agent choice")
        return
    receipt = receipts_by_event.get(selected.selection_event_id)
    if (
        receipt is None
        or receipt.plan_id != plan.plan.plan_id
        or receipt.requirement_id != reference_requirement.requirement_id
        or receipt.assertion.selected_candidate_id
        != f"assembly:{selected.row.assembly_accession}"
        or selected.discovery_event_id not in receipt.assertion.evidence_event_ids
        or selected.selection_event_id not in grounding.source_event_ids
    ):
        raise ValueError("reference selection lacks its exact Planning HITL assertion")


def _researcher_statement(event) -> bool:
    if event["event_type"] in {"ResearchRequestRegistered", "PlanningHitlResponded"}:
        return True
    return (
        event["event_type"] == "ExecutionHitlResponded"
        and event["payload"].get("response", {}).get("revalidation_scope")
        == "reference"
    )


def _validate_sample_attestation(fact, plan, earlier) -> None:
    """A sample attestation binds the plan's inputs and the name the model typed from an
    earlier researcher statement, whose words it quotes (#472 Q2 a)."""
    if fact.tool_input.input_set_id != plan.plan.input_set_id:
        raise ValueError("sample attestation changed the registered input set")
    if not statement_states(
        earlier,
        fact.tool_input.assertion_event_id,
        fact.tool_input.quote,
        SAMPLE_IDENTIFIER_RULE,
        fact.tool_input.sample_identifier,
    ):
        raise ValueError("sample attestation quotes no researcher statement")


def _validate_choice_basis(
    choice, context, event, discoveries, discovery_event_id, choices, revision_event_id
):
    earlier = context.timeline[: context.timeline.index(event)]
    if choice.discovery_event_id is None:
        # The model typed the accession from an earlier researcher statement; its quote
        # stands there and states it.
        if not cited_statement_states(
            earlier,
            choice.assertion_event_id,
            choice.quote,
            ASSEMBLY_ACCESSION_RULE,
            choice.assembly_accession,
        ):
            raise ValueError(
                "direct reference choice lacks an exact supplied assertion"
            )
        if choice.assertion_event_id not in choice.grounding.source_event_ids:
            raise ValueError("direct reference choice omitted assertion grounding")
    else:
        discovery = discoveries.get(choice.discovery_event_id)
        if (
            discovery is None
            or choice.discovery_event_id not in choice.grounding.source_event_ids
        ):
            raise ValueError("reference choice lacks current discovery evidence")
        if (
            choice.assembly_accession is not None
            and discovery.row(choice.assembly_accession) is None
        ):
            raise ValueError("reference choice is not a row of its discovery page")
        if choice.quote_event_id is not None:
            # A quote of a cited paper's words stands in the span its reading kept.
            from bio_neb.paper_reading.planning import kept_texts

            if not quoted(kept_texts(earlier, choice.quote_event_id), choice.quote):
                raise ValueError("reference choice quotes words no reading kept")
        elif choice.quote is not None and not quoted(
            researcher_words(earlier), choice.quote
        ):
            raise ValueError("reference choice quotes words the researcher never wrote")
    if (
        revision_event_id is not None
        and revision_event_id not in choice.grounding.source_event_ids
    ):
        raise ValueError("reference correction omitted its exact revision authority")
    # A researcher answer about a discovery page or about the current choice.
    cited = {*discoveries, *(c.source_event_id for c in choices[-1:])}
    receipts = tuple(
        receipt
        for receipt in context.hitl.receipts
        if cited & set(receipt.assertion.evidence_event_ids)
        and receipt.response_event_id in choice.grounding.source_event_ids
    )
    if (
        receipts
        and receipts[-1].assertion.selected_candidate_id is not None
        and choice.assembly_accession is not None
        and f"assembly:{choice.assembly_accession}"
        != receipts[-1].assertion.selected_candidate_id
    ):
        raise ValueError("reference choice contradicts the exact researcher selection")
    fresh_clarification = bool(
        receipts
        and choices
        and receipts[-1].response_event_id not in choices[-1].grounding.source_event_ids
    )
    if (
        choices
        and not fresh_clarification
        and (
            revision_event_id is None
            or revision_event_id in choices[-1].grounding.source_event_ids
        )
    ):
        raise ValueError(
            "reference choice cannot silently replace an existing decision"
        )


def _chosen_suggestion(
    context, event_id, event, verified_reference
) -> ChosenSuggestion | None:
    """The suggestion an answer chooses: a selection of the suggested assembly, other
    than the verified one, on a question presenting a suggestion of another assembly's
    index (#459 D4); None for any other answer."""
    from bio_neb.reference_discovery.reference_facts_journey import (
        AssemblyWithIndexSuggested,
    )

    response = PlanningHitlResponded.model_validate(
        {"event_type": event["event_type"], **event["payload"]}
    ).response
    if response.action != "select":
        return None
    request = context.by_id[event["payload"]["request_event_id"]]
    suggestions = [
        (evidence, fact)
        for evidence in request["payload"]["evidence_event_ids"]
        if isinstance(
            fact := context.prebuilt_index_facts.get(evidence),
            AssemblyWithIndexSuggested,
        )
    ]
    if len(suggestions) != 1:
        return None
    suggestion_event_id, fact = suggestions[0]
    other = fact.tool_input.assembly_accession
    if (
        response.candidate_id != f"assembly:{other}"
        or other == verified_reference.assembly_accession
    ):
        return None
    return ChosenSuggestion(
        answer_event_id=event_id,
        suggestion_event_id=suggestion_event_id,
        requested_assembly=verified_reference.assembly_accession,
        assembly_accession=other,
        candidate_id=fact.tool_input.candidate.candidate.candidate_id,
        targets=fact.result.targets,
    )


def _project_reference_identity(  # noqa: C901, PLR0915 - Existing domain grammar retained; #459 adds the suggested-assembly switch.
    context: PrefixSnapshot,
) -> ReferenceIdentityProjection:
    by_id = context.by_id
    ancestors_by_id = context.ancestry
    plan_event, plan = context.require_plan()
    journey_id = _canonical_uuid(plan_event.get("journey_id"), "journey_id")
    plan_event_id = plan_event["event_id"]
    sample_requirement = _requirement(plan, "sample_identity")
    taxon_requirement = _requirement(plan, "taxon_identity")
    reference_requirement = _requirement(plan, "reference_assembly_identity")
    requirements = {
        sample_requirement.requirement_id: sample_requirement,
        taxon_requirement.requirement_id: taxon_requirement,
        reference_requirement.requirement_id: reference_requirement,
    }

    planning_hitl = context.hitl.for_requirements(set(requirements))
    receipts_by_event = {
        receipt.response_event_id: receipt for receipt in planning_hitl.receipts
    }
    fact_views: list[ReferenceFactView] = []
    sample_event_id: str | None = None
    resolved_taxon: ResolvedTaxonEvidence | None = None
    discovery_event_id: str | None = None
    discoveries: dict[str, DiscoverAssembliesObserved] = {}
    description_event_id: str | None = None
    verified_reference: ValidatedReference | None = None

    choices: list[ReferenceChoiceView] = []
    revision_event_id = None
    chosen_suggestion = None
    for event_id, event in by_id.items():
        if (
            event["event_type"] == "ExecutionHitlResponded"
            and event["payload"].get("response", {}).get("action") == "revise"
            and event["payload"]["response"].get("revalidation_scope") == "reference"
        ):
            from bio_neb.parabricks.execution_planning_journey import (
                _execution_revision_context,
            )

            direction = _execution_revision_context(
                context.timeline[: context.timeline.index(event) + 1], by_id
            )
            revision_event_id = direction.revision_event_id
            description_event_id = None
            verified_reference = None
            chosen_suggestion = None
            continue
        if (
            event["event_type"] == "PlanningHitlResponded"
            and verified_reference is not None
            and (
                chosen := _chosen_suggestion(
                    context, event_id, event, verified_reference
                )
            )
            is not None
        ):
            # The researcher chose the suggested assembly for the whole Run: the
            # reference is chosen again, citing the answer.
            chosen_suggestion = chosen
            description_event_id = None
            verified_reference = None
            continue
        if event["event_type"] == "ReferenceChoiceRecorded":
            choice = ReferenceChoiceRecorded.model_validate(
                {"event_type": event["event_type"], **event["payload"]}
            )
            ancestors = ancestors_by_id[event_id]
            _owning_attempt(choice, ancestors=ancestors, by_id=by_id)
            _validate_grounding(
                choice.grounding,
                journey_id=journey_id,
                plan_event_id=plan_event_id,
                plan=plan,
                requirement=reference_requirement,
                ancestors=ancestors,
            )
            _validate_choice_basis(
                choice,
                context,
                event,
                discoveries,
                discovery_event_id,
                choices,
                revision_event_id,
            )
            choices.append(
                ReferenceChoiceView(source_event_id=event_id, **choice.model_dump())
            )
            description_event_id = None
            verified_reference = None
            continue
        fact = context.reference_facts.get(event_id)
        if fact is None:
            continue
        ancestors = ancestors_by_id[event_id]
        _owning_attempt(fact, ancestors=ancestors, by_id=by_id)
        grounding = fact.tool_input.grounding
        requirement = requirements.get(grounding.requirement_id)
        if requirement is None:
            raise ValueError("reference Tool fact targets a non-reference requirement")
        _validate_grounding(
            grounding,
            journey_id=journey_id,
            plan_event_id=plan_event_id,
            plan=plan,
            requirement=requirement,
            ancestors=ancestors,
        )

        if isinstance(fact, SampleAttestationRecorded):
            _validate_sample_attestation(
                fact, plan, context.timeline[: context.timeline.index(event)]
            )
            sample_event_id = event_id if fact.result.status == "verified" else None
            resolved_taxon = None
            discovery_event_id = None
            discoveries = {}
            description_event_id = None
            verified_reference = None
        elif isinstance(fact, TaxonResolutionRecorded):
            if (
                sample_event_id is None
                or sample_event_id not in grounding.source_event_ids
            ):
                raise ValueError("taxon resolution lacks verified sample ancestry")
            resolved_taxon = None
            if (
                isinstance(fact.result, ResolveTaxonObserved)
                and fact.result.resolution == "resolved"
            ):
                candidate = fact.result.candidates[0]
                resolved_taxon = ResolvedTaxonEvidence(
                    candidate_id=candidate.candidate_id,
                    ncbi_tax_id=candidate.ncbi_tax_id,
                    candidate_evidence_id=candidate.evidence_id,
                    result_evidence_id=fact.result.evidence_id,
                    source_event_id=event_id,
                )
            discovery_event_id = None
            discoveries = {}
            description_event_id = None
            verified_reference = None
        elif isinstance(fact, ReferenceAssembliesRecorded):
            if resolved_taxon is None or fact.tool_input.taxon != resolved_taxon:
                raise ValueError("reference discovery lacks the current resolved taxon")
            if resolved_taxon.source_event_id not in grounding.source_event_ids:
                raise ValueError("reference discovery omitted resolved taxon evidence")
            if isinstance(fact.result, DiscoverAssembliesObserved):
                # Each page read stays citable; a failed call adds none.
                discoveries = {**discoveries, event_id: fact.result}
                discovery_event_id = event_id
            if verified_reference is None:
                description_event_id = None
            # A page read during the index work lists the organism's other
            # assemblies beside the verified one, which stays verified.
        else:
            assert isinstance(fact, ReferenceDescriptionRecorded)
            _validate_description_basis(
                fact,
                choices,
                resolved_taxon,
                discoveries,
                receipts_by_event,
                plan,
                reference_requirement,
            )
            description_event_id = event_id
            verified_reference = None
            if (
                isinstance(fact.result, DescribeReferenceObserved)
                and fact.result.resolution == "resolved"
            ):
                verified_reference = fact.result.references[0]

        projected_result = fact.result.model_dump(mode="json")
        if isinstance(fact, TaxonResolutionRecorded) and fact.result.status == "failed":
            projected_result["query"] = fact.tool_input.query
        fact_views.append(
            ReferenceFactView(
                source_event_id=event_id,
                event_type=fact.event_type,
                attempt_id=fact.attempt_id,
                result=projected_result,
            )
        )

    if planning_hitl.pending_request is not None:
        status = "planning_hitl_pending"
    elif verified_reference is not None and sample_event_id and resolved_taxon:
        status = "reference_identity"
    elif discovery_event_id is not None:
        status = "reference_candidates"
    elif resolved_taxon is not None:
        status = "taxon_identity"
    elif sample_event_id is not None:
        status = "sample_identity"
    else:
        status = "reference_identity_ready"
    return ReferenceIdentityProjection(
        journey_id=journey_id,
        plan_event_id=plan_event_id,
        plan_id=plan.plan.plan_id,
        input_set_id=plan.plan.input_set_id,
        requirements=tuple(
            ReferenceRequirementLocator(
                requirement_id=requirement.requirement_id,
                kind=requirement.kind,
                subject_key=requirement.subject_key,
            )
            for requirement in (
                sample_requirement,
                taxon_requirement,
                reference_requirement,
            )
        ),
        status=status,
        facts=tuple(fact_views),
        planning_hitl=planning_hitl,
        sample_event_id=sample_event_id,
        resolved_taxon=resolved_taxon,
        discovery_event_id=discovery_event_id,
        discovery_event_ids=tuple(discoveries),
        description_event_id=description_event_id,
        verified_reference=verified_reference,
        choices=tuple(choices),
        revision_event_id=revision_event_id,
        chosen_suggestion=chosen_suggestion,
    )


__all__ = [
    "ReferenceAssembliesRecorded",
    "ReferenceDescriptionRecorded",
    "ReferenceFactView",
    "ReferenceIdentityFact",
    "ReferenceIdentityProjection",
    "ReferenceRequirementLocator",
    "SampleAttestationRecorded",
    "TaxonResolutionRecorded",
    "project_reference_identity",
    "validate_reference_fact",
]


class ReferenceDecisionBasis(BaseModel):
    """Frozen selection and its separate exact validation for proposal inspection:
    the discovery page the choice cites, as NCBI listed it, and NCBI's count of the
    whole listing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    choice: ReferenceChoiceView
    alternatives: Tuple[AssemblySourceRecord, ...]
    total_count: StrictInt | None
    next_page_token: StrictStr | None
    validation_event_id: StrictStr
    validation: DescribeReferenceObserved
    requested_assembly: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The assembly the researcher asked for, when they chose a suggested assembly with
    a fitting Galaxy index in its place (#459 D4)."""


def reference_decision_basis(reference: ReferenceIdentityProjection):
    if not reference.choices or reference.description_event_id is None:
        return None
    choice = reference.choices[-1]
    discovery = next(
        (
            DiscoverAssembliesObserved.model_validate(f.result)
            for f in reference.facts
            if f.source_event_id == choice.discovery_event_id
        ),
        None,
    )
    validation = next(
        f
        for f in reference.facts
        if f.source_event_id == reference.description_event_id
    )
    return ReferenceDecisionBasis(
        choice=choice,
        alternatives=() if discovery is None else discovery.rows,
        total_count=None if discovery is None else discovery.total_count,
        next_page_token=None if discovery is None else discovery.next_page_token,
        validation_event_id=reference.description_event_id,
        validation=DescribeReferenceObserved.model_validate(validation.result),
        requested_assembly=reference.requested_assembly,
    )
