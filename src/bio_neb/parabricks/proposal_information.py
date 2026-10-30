"""Strict contracts for a grounded Parabricks proposal-information plan."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, Tuple, Union, get_args

from pydantic import (
    Field,
    ModelWrapValidatorHandler,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import ErrorType

from bio_neb.agentic_journey.action_policy import CANONICAL_TEXT_PATTERN
from bio_neb.contracts import FrozenContract
from bio_neb.identifier_rule import IdentifierRule, word_aligned_spans
from bio_neb.input_inspection.tool import InputMetadataResult
from bio_neb.investigation.investigation_request import (
    RSID_CANONICAL_PATTERN,
    RSID_RULE,
    SiteDraft,
    StatedChange,
    StatedPosition,
)
from bio_neb.journey import canonical_json_digest
from bio_neb.parabricks.slot_satisfaction import guidance as slot_guidance
from bio_neb.parabricks.slot_satisfaction import (
    repeated_single_requirement_kinds,
    unsatisfiable_reason,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    ASSEMBLY_ACCESSION_PATTERN,
    ASSEMBLY_ACCESSION_RULE,
    NCBI_TAXON_ID_RULE,
    SAMPLE_IDENTIFIER_RULE,
)

ProposalInformationKind = Literal[
    "research_goal",
    "registered_input",
    "sample_identity",
    "taxon_identity",
    "reference_assembly_identity",
    "companion_asset",
    "analysis_recipe",
    "resource_profile",
    "scientific_investigation",
    # A paper the researcher's words cite, one per paper; optional.
    "cited_paper",
]
ScientificInvestigationPurpose = Literal[
    "known_variant",
    "literature_claim",
    "hypothesis",
    "open_question",
    "discovery_request",
    "other",
]
# Every kind a proposal plan records, in the order a refusal names them.
_REQUIRED_KIND_ORDER = (
    "research_goal",
    "registered_input",
    "sample_identity",
    "taxon_identity",
    "reference_assembly_identity",
    "companion_asset",
    "analysis_recipe",
    "resource_profile",
    # Every Run names its targets before Execution.
    "scientific_investigation",
)
REQUIRED_PROPOSAL_INFORMATION_KINDS = frozenset(_REQUIRED_KIND_ORDER)


def missing_proposal_kinds(kinds: Iterable[object]) -> tuple[str, ...]:
    """The required kinds a draft lacks, in their recorded order (#372 W4).

    Both refusal paths name these: a field-error correction and the grounding
    precondition.
    """
    present = set(kinds)
    return tuple(
        kind
        for kind in _REQUIRED_KIND_ORDER
        if kind in REQUIRED_PROPOSAL_INFORMATION_KINDS and kind not in present
    )


AcquisitionRoute = Literal[
    "use_supplied_assertion",
    "use_accepted_pre_parabricks_fact",
    "validate_with_tool",
    "derive_deterministically",
    "ask_planning_hitl",
]
ObservedInputFact = Literal[
    "role", "registered_path", "exists", "regular_file", "size_bytes"
]

_KEY = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


def _nonempty(value: str, name: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


_REQUIREMENT_IDENTIFIER_RULES = {
    "sample_identity": SAMPLE_IDENTIFIER_RULE,
    "taxon_identity": NCBI_TAXON_ID_RULE,
    "reference_assembly_identity": ASSEMBLY_ACCESSION_RULE,
}
# Fact slots carry a value a source can verify (or, for a sample name, none can);
# prose requirements are the researcher's intent.
_FACT_SLOT_KINDS = frozenset(
    {"sample_identity", "taxon_identity", "reference_assembly_identity"}
)


def requirement_identifier_rule(kind: str) -> IdentifierRule | None:
    """The one format matcher for a typed value of this requirement kind, if any."""

    return _REQUIREMENT_IDENTIFIER_RULES.get(kind)


_QUOTE_DESCRIPTION = (
    "The researcher request's words that state this, copied exactly: whole words, "
    "spaces and case kept. It may occur more than once in the request."
)


def _quote_field(description: str = _QUOTE_DESCRIPTION, **options: Any) -> Any:
    return Field(description=description, pattern=CANONICAL_TEXT_PATTERN, **options)


class _Quoted(FrozenContract):
    @field_validator("quote", check_fields=False)
    @classmethod
    def canonical_quote(cls, value: str) -> str:
        return _nonempty(value, "quote")


class ToFindKnowledgeDraft(_Quoted):
    """A value the request does not state: a Tool or the researcher finds it."""

    state: Literal["to_find"] = "to_find"
    evidence_event_ids: Tuple[StrictStr, ...] = Field(
        min_length=1,
        description="Context event IDs that show the value is needed and not stated.",
    )
    quote: StrictStr | None = _quote_field(
        "Optional. The request's words that describe what to find, such as an "
        "organism name for a TaxId, copied exactly; they are not the value.",
        default=None,
    )


class AssertedTextDraft(_Quoted):
    """Prose the request states; the quote is the value."""

    state: Literal["asserted"] = "asserted"
    quote: StrictStr = _quote_field()


class AssertedCitationDraft(_Quoted):
    """The researcher's words that cite one paper; the quote is all code keeps of
    them, checked only by membership in the researcher's words (ADR 0049): no code
    reads an id out of it."""

    state: Literal["asserted"] = "asserted"
    quote: StrictStr = _quote_field(
        "The researcher's exact words that cite this paper (an id, a link, or a "
        "title, author and year), copied on whole words from their request or a "
        "registered request file's item."
    )
    evidence_event_ids: Tuple[StrictStr, ...] = Field(
        default=(),
        description="The PaperRecorded event of this paper when you retrieved it "
        "before recording the plan; omitted when it is still to find.",
    )


class AssertedSampleDraft(_Quoted):
    state: Literal["asserted"] = "asserted"
    value: StrictStr = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        description="The sample identifier exactly as the request spells it, e.g. HG001 "
        "or MM-04.",
    )
    quote: StrictStr = _quote_field()


class AssertedTaxonDraft(_Quoted):
    state: Literal["asserted"] = "asserted"
    value: StrictInt = Field(
        gt=0,
        description="The NCBI Taxonomy ID the request states, as a number, e.g. 9606 "
        "or 10090. "
        "An organism name alone states no ID: use to_find with that name as its quote.",
    )
    quote: StrictStr = _quote_field()


class AssertedAccessionDraft(_Quoted):
    state: Literal["asserted"] = "asserted"
    value: StrictStr = Field(
        pattern=ASSEMBLY_ACCESSION_PATTERN,
        description="The versioned GCA_/GCF_ assembly accession the request states, "
        "exactly as it writes it.",
    )
    quote: StrictStr = _quote_field()


class AssertedInvestigationDraft(_Quoted):
    state: Literal["asserted"] = "asserted"
    quote: StrictStr = _quote_field()
    value: StrictStr | None = Field(
        default=None,
        pattern=RSID_CANONICAL_PATTERN,
        description="Optional. The one rsID this item asks about when its quote names "
        "one, typed as rs followed by digits, e.g. rs429358 for a quote RS429358; an "
        "rsID the request does not give is to_find, never a placeholder.",
    )
    site: SiteDraft | None = Field(
        default=None,
        description="Optional. The site the quote writes by coordinates, typed: its "
        "sequence as written, position, and REF and ALT when written, e.g. "
        '{"contig": "V", "position": 2129317, "ref": "C", "alt": "T"} for V:2129317 '
        "C>T. The server checks that the quote holds each of them.",
    )
    stated_version: StrictStr | None = Field(
        default=None,
        pattern=CANONICAL_TEXT_PATTERN,
        description="Optional. The genome version the request states for the site, as "
        "it writes it, e.g. hg19 for chr19:45411941 T>C (hg19), or mm10. A sample, "
        "strain or other name is no version; leave it out when the request names none.",
    )
    gene_change: StatedChange | None = Field(
        default=None,
        description="Optional. The gene and change the quote writes, typed: the gene "
        "and change as written, the transcript when written, and the protein change, "
        "the coding change or both, e.g. "
        '{"gene": "BRAF", "change": "V600E", "protein": {"ref": "V", "codon": 600, '
        '"alt": "E"}} for BRAF V600E. The server checks that the quote holds the gene, '
        "the change and its number.",
    )


# Names an inspected fact; grounding fills its value from the inspection.
class ObservedKnowledgeDraft(FrozenContract):
    state: Literal["observed"] = "observed"
    input_role: StrictStr = Field(pattern=CANONICAL_TEXT_PATTERN)
    fact: ObservedInputFact

    @field_validator("input_role")
    @classmethod
    def canonical_value(cls, value: str, info) -> str:
        return _nonempty(value, info.field_name)


class NoRemainingValidation(FrozenContract):
    kind: Literal["none"] = "none"


class AcceptedPreParabricksValidation(FrozenContract):
    """A fact accepted at the deterministic Adapter-acceptance boundary."""

    kind: Literal["accepted_pre_parabricks"] = "accepted_pre_parabricks"
    binding_id: StrictStr = Field(pattern=r"^adapter-acceptance-binding:[0-9a-f]{64}$")
    planning_contact_count: Literal[0] = 0
    model_contact_count: Literal[0] = 0
    source_contact_count: Literal[0] = 0


class ToolEvidenceValidation(FrozenContract):
    kind: Literal["tool_evidence"] = "tool_evidence"
    requirement: StrictStr = Field(pattern=CANONICAL_TEXT_PATTERN)

    @field_validator("requirement")
    @classmethod
    def canonical_requirement(cls, value: str) -> str:
        return _nonempty(value, "validation requirement")


class DeterministicValidation(FrozenContract):
    kind: Literal["deterministic_derivation"] = "deterministic_derivation"
    requirement: StrictStr = Field(pattern=CANONICAL_TEXT_PATTERN)

    @field_validator("requirement")
    @classmethod
    def canonical_requirement(cls, value: str) -> str:
        return _nonempty(value, "validation requirement")


class ResearcherDecisionValidation(FrozenContract):
    kind: Literal["researcher_decision"] = "researcher_decision"
    requirement: StrictStr = Field(pattern=CANONICAL_TEXT_PATTERN)

    @field_validator("requirement")
    @classmethod
    def canonical_requirement(cls, value: str) -> str:
        return _nonempty(value, "validation requirement")


RemainingValidation = Annotated[
    Union[
        NoRemainingValidation,
        AcceptedPreParabricksValidation,
        ToolEvidenceValidation,
        DeterministicValidation,
        ResearcherDecisionValidation,
    ],
    Field(discriminator="kind"),
]


# The cross-field admission rules the model sees come from the one slot table.
PROPOSAL_REQUIREMENT_RULES = slot_guidance()
PROPOSAL_REQUIREMENT_REPAIR_GUIDANCE = (
    "For record_proposal_requirements: preserve already valid fields. "
    + PROPOSAL_REQUIREMENT_RULES
)


_PYDANTIC_ERROR_TYPES = frozenset(get_args(ErrorType))


def _reraised(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """One field error raised again: a draft's own error type as its custom error."""
    if item["type"] in _PYDANTIC_ERROR_TYPES:
        return item
    return {
        "type": PydanticCustomError(item["type"], item["msg"]),
        "loc": item["loc"],
        "input": item["input"],
    }


class _RequirementDraft(FrozenContract):
    """What every requirement kind declares; each kind types its own knowledge."""

    subject_key: StrictStr = Field(
        pattern=_KEY.pattern,
        description="A lowercase snake-case key for this requirement, not a quotation.",
    )
    domain: Literal["scientific", "operational"]
    meaning: StrictStr = Field(
        description="Your normalized reading, e.g. 'Parabricks germline variant calling'.",
        pattern=CANONICAL_TEXT_PATTERN,
    )
    # The route and validation rules are in the Tool's description.
    acquisition_route: AcquisitionRoute
    validation: RemainingValidation

    @field_validator("subject_key")
    @classmethod
    def canonical_subject_key(cls, value: str) -> str:
        if not _KEY.fullmatch(value):
            raise ValueError("subject_key must be lowercase snake case")
        return value

    @field_validator("meaning")
    @classmethod
    def canonical_meaning(cls, value: str) -> str:
        return _nonempty(value, "meaning")

    @field_validator("knowledge", mode="before", check_fields=False)
    @classmethod
    def knowledge_names_its_state(cls, value: Any) -> Any:
        """Knowledge without a state is told the states this kind takes; the
        union's own error names only its discriminator (#409, #380 S3)."""
        if isinstance(value, Mapping) and "state" not in value:
            annotation = cls.model_fields["knowledge"].annotation
            states = ", ".join(
                f"'{state}'"
                for member in get_args(annotation) or (annotation,)
                for state in get_args(member.model_fields["state"].annotation)
            )
            raise PydanticCustomError(
                "knowledge_state_missing",
                f"knowledge names its state: one of {states}",
            )
        return value

    @model_validator(mode="after")
    def coherent_route(self) -> "_RequirementDraft":
        reason = unsatisfiable_reason(
            self.kind,  # type: ignore[attr-defined]
            self.knowledge.state,  # type: ignore[attr-defined]
            self.validation.kind,
            self.acquisition_route,
        )
        if reason is not None:
            raise ValueError(reason)
        return self


class RegisteredInputRequirementDraft(_RequirementDraft):
    kind: Literal["registered_input"]
    knowledge: Annotated[
        Union[ObservedKnowledgeDraft, ToFindKnowledgeDraft],
        Field(discriminator="state"),
    ]


class SampleIdentityRequirementDraft(_RequirementDraft):
    kind: Literal["sample_identity"]
    knowledge: Annotated[
        Union[AssertedSampleDraft, ToFindKnowledgeDraft], Field(discriminator="state")
    ]


class TaxonIdentityRequirementDraft(_RequirementDraft):
    kind: Literal["taxon_identity"]
    knowledge: Annotated[
        Union[AssertedTaxonDraft, ToFindKnowledgeDraft], Field(discriminator="state")
    ]


class ReferenceAssemblyRequirementDraft(_RequirementDraft):
    kind: Literal["reference_assembly_identity"]
    knowledge: Annotated[
        Union[AssertedAccessionDraft, ToFindKnowledgeDraft],
        Field(discriminator="state"),
    ]


class ScientificInvestigationRequirementDraft(_RequirementDraft):
    """A post-execution question or claim; it stays open until after execution."""

    kind: Literal["scientific_investigation"]
    knowledge: Annotated[
        Union[AssertedInvestigationDraft, ToFindKnowledgeDraft],
        Field(discriminator="state"),
    ]
    investigation_purpose: ScientificInvestigationPurpose


class CitedPaperRequirementDraft(_RequirementDraft):
    """One paper the researcher's words cite: the paper stage finds and retrieves it
    before the reference work (owner, #420 D1 b)."""

    kind: Literal["cited_paper"]
    knowledge: AssertedCitationDraft


class TextRequirementDraft(_RequirementDraft):
    kind: Literal[
        "research_goal", "companion_asset", "analysis_recipe", "resource_profile"
    ]
    knowledge: Annotated[
        Union[AssertedTextDraft, ToFindKnowledgeDraft], Field(discriminator="state")
    ]


ProposalInformationRequirementDraft = Annotated[
    Union[
        RegisteredInputRequirementDraft,
        SampleIdentityRequirementDraft,
        TaxonIdentityRequirementDraft,
        ReferenceAssemblyRequirementDraft,
        ScientificInvestigationRequirementDraft,
        CitedPaperRequirementDraft,
        TextRequirementDraft,
    ],
    Field(discriminator="kind"),
]


# Grounding supplies the Journey scope (request, registration, inspection, input set).
class ProposalInformationPlanDraft(FrozenContract):
    requirements: Tuple[ProposalInformationRequirementDraft, ...] = Field(min_length=1)

    @model_validator(mode="wrap")
    @classmethod
    def retain_required_kind_error_with_field_errors(
        cls,
        value: Any,
        handler: ModelWrapValidatorHandler["ProposalInformationPlanDraft"],
    ) -> "ProposalInformationPlanDraft":
        raw_requirements = (
            value.get("requirements") if isinstance(value, dict) else None
        )
        raw_kinds = (
            {
                item.get("kind")
                for item in raw_requirements
                if isinstance(item, dict) and isinstance(item.get("kind"), str)
            }
            if isinstance(raw_requirements, (list, tuple))
            else set()
        )
        try:
            return handler(value)
        except ValidationError as error:
            missing = missing_proposal_kinds(raw_kinds)
            if not missing:
                raise
            raise ValidationError.from_exception_data(
                cls.__name__,
                [
                    *map(_reraised, error.errors(include_url=False)),
                    {
                        "type": PydanticCustomError(
                            "proposal_requirement_kinds_incomplete",
                            "add one requirement of each missing kind: {missing}",
                            {"missing": ", ".join(missing)},
                        ),
                        "loc": ("requirements",),
                        "input": raw_requirements,
                    },
                ],
            ) from error

    @model_validator(mode="after")
    def unique_subjects(self) -> "ProposalInformationPlanDraft":
        keys = [item.subject_key for item in self.requirements]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise ValueError(
                "requirement subject keys must be unique; repeated: "
                + ", ".join(repeated)
            )
        return self

    @model_validator(mode="after")
    def one_requirement_per_bound_kind(self) -> "ProposalInformationPlanDraft":
        repeated = repeated_single_requirement_kinds(
            item.kind for item in self.requirements
        )
        if repeated:
            raise ValueError(
                "; ".join(
                    f"more than one {kind} requirement: its Tool binds exactly one, "
                    "so record it once"
                    for kind in repeated
                )
            )
        return self


class RequestAssertionEvidence(FrozenContract):
    source_kind: Literal["research_request"] = "research_request"
    source_event_id: StrictStr
    start_offset: StrictInt = Field(ge=0)
    end_offset: StrictInt = Field(gt=0)
    exact_text: StrictStr


class InspectedInputEvidence(FrozenContract):
    source_kind: Literal["input_inspection"] = "input_inspection"
    registration_event_id: StrictStr
    inspection_event_id: StrictStr
    input_set_id: StrictStr
    role: StrictStr
    registered_path: StrictStr
    fact: ObservedInputFact


class ContextEvidence(FrozenContract):
    source_kind: Literal["planning_context"] = "planning_context"
    source_event_id: StrictStr


class AcceptedPreParabricksEvidence(FrozenContract):
    source_kind: Literal["accepted_pre_parabricks"] = "accepted_pre_parabricks"
    source_event_id: StrictStr
    binding_id: StrictStr = Field(pattern=r"^adapter-acceptance-binding:[0-9a-f]{64}$")


EvidenceReference = Annotated[
    Union[
        RequestAssertionEvidence,
        InspectedInputEvidence,
        ContextEvidence,
        AcceptedPreParabricksEvidence,
    ],
    Field(discriminator="source_kind"),
]


class SlotVerification(FrozenContract):
    """The source that can verify a fact slot's value; none for a sample name, or for
    an rsID, whose variant source the Investigation's model chooses (#469, owner Q6 a)."""

    source: StrictStr | None


class ProposalInformationRequirement(FrozenContract):
    requirement_id: StrictStr
    kind: ProposalInformationKind
    subject_key: StrictStr
    domain: Literal["scientific", "operational"]
    meaning: StrictStr
    knowledge_state: Literal["to_find", "asserted", "observed"]
    # The typed value: the model's, checked against its quote by
    # ``identifier_rule``; the quote itself for prose; the inspected value.
    value: StrictStr | None
    evidence: Tuple[EvidenceReference, ...]
    acquisition_route: AcquisitionRoute
    validation: RemainingValidation
    investigation_purpose: ScientificInvestigationPurpose | None = None
    identifier_rule: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Present on a fact slot: sample, taxon, reference assembly and an rsID item.
    verification: SlotVerification | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The genome version a site item's source states for it, as the model named it
    # (no code check, the placement lookup resolves it).
    stated_version: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The site the model typed from the item's quote, which holds it.
    site: StatedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The gene change the model typed from the item's quote, which holds it.
    gene_change: StatedChange | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def exact_investigation_shape(self) -> "ProposalInformationRequirement":
        if (self.kind == "scientific_investigation") != (
            self.investigation_purpose is not None
        ):
            raise ValueError(
                "only scientific investigation items require an investigation purpose"
            )
        return self


class ProposalInformationPlan(FrozenContract):
    plan_id: StrictStr
    request_event_id: StrictStr
    registration_event_id: StrictStr
    inspection_event_id: StrictStr
    input_set_id: StrictStr
    requirements: Tuple[ProposalInformationRequirement, ...] = Field(min_length=1)


class ProposalPlanGroundingError(ValueError):
    """The grounding refusals, with the typed facts each names (#372 D2)."""

    def __init__(
        self,
        errors: Tuple[str, ...],
        facts: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        super().__init__(";".join(errors))
        self.errors = errors
        self.facts = dict(facts or {})


class AnsweredTargetsDraft(FrozenContract):
    """The targets a researcher's answer to a target question states, read by the
    planning model as it reads the request (#462, owner 2026-10-26). An answer that
    states none the model can name is recorded with no target, and the coordinator
    asks again."""

    targets: Tuple[AssertedInvestigationDraft, ...] = Field(
        description="One item per target the answer names, in its order: its quote, "
        "the answer's own words for it, copied exactly; its rsID when the quote gives "
        "one; the site, typed, when the quote writes one by coordinates; the gene "
        "change, typed, when the quote writes a gene and its change; and the genome "
        "version the answer states for a site. Empty when the answer names no target."
    )


class AnsweredTarget(FrozenContract):
    """One target an answer states: the answer's words for it (located in the
    answer), its rsID and site when the model typed them from those words, and the
    version the model named."""

    quote: StrictStr
    start_offset: int = Field(ge=0)
    rsid: StrictStr | None = Field(
        default=None,
        pattern=RSID_CANONICAL_PATTERN,
        exclude_if=lambda value: value is None,
    )
    stated_version: StrictStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    site: StatedPosition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    gene_change: StatedChange | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def ground_answered_targets(
    draft: AnsweredTargetsDraft, answer: str
) -> Tuple[AnsweredTarget, ...]:
    """Accept the model's reading of an answer under the plan's own quote rules: each
    quote is the answer's words on whole words, an rsID is one its quote states, and
    a site and a gene change are ones its quote writes (#472 Q1 a, #473 Q2 a). The version is the model's, unchecked
    (owner, 2026-10-26)."""

    errors: list[str] = []
    grounded: list[AnsweredTarget] = []
    for index, target in enumerate(draft.targets):
        spans = word_aligned_spans(answer, target.quote)
        if not spans:
            errors.append(f"target_{index}:answer_quote_ungrounded")
        elif target.value is not None and not RSID_RULE.matches(
            target.value, target.quote
        ):
            errors.append(f"target_{index}:asserted_value_not_in_quote")
        elif target.site is not None and not target.site.in_quote(target.quote):
            errors.append(f"target_{index}:site_not_in_quote")
        elif target.gene_change is not None and not target.gene_change.in_quote(
            target.quote
        ):
            errors.append(f"target_{index}:gene_change_not_in_quote")
        else:
            grounded.append(
                AnsweredTarget(
                    quote=target.quote,
                    start_offset=spans[0],
                    rsid=target.value,
                    stated_version=target.stated_version,
                    site=None if target.site is None else target.site.stated(),
                    gene_change=target.gene_change,
                )
            )
    if errors:
        raise ProposalPlanGroundingError(tuple(errors))
    return tuple(grounded)


def _observed_value(result: InputMetadataResult, role: str, fact: str) -> str | None:
    selected = next((item for item in result.inputs if item.role == role), None)
    if selected is None:
        return None
    values = {
        "role": selected.role,
        "registered_path": selected.registered_path,
        "exists": "true",
        "regular_file": "true",
        "size_bytes": str(selected.size_bytes),
    }
    return values[fact]


def _typed_value(item: Any) -> tuple[str | None, IdentifierRule | None]:
    """An asserted draft's typed value and the matcher that checks it, if any."""

    knowledge = item.knowledge
    if isinstance(knowledge, AssertedTextDraft | AssertedCitationDraft):
        return knowledge.quote, None
    if isinstance(knowledge, AssertedInvestigationDraft):
        if knowledge.value is None:
            return knowledge.quote, None
        return knowledge.value, RSID_RULE
    rule = requirement_identifier_rule(item.kind)
    assert rule is not None
    return str(knowledge.value), rule


def _stated_version(item: Any) -> str | None:
    """The genome version an asserted site item names for its quote, if any."""
    return getattr(item.knowledge, "stated_version", None)


def _typed_target_unquoted(item: Any) -> str | None:
    """Why an asserted item's typed site or gene change is not its quote's, if it is
    not (#472 Q1 a, #473 Q2 a: membership only)."""
    quote = item.knowledge.quote
    if (site := _site(item)) is not None and not site.in_quote(quote):
        return "site_not_in_quote"
    if (change := _gene_change(item)) is not None and not change.in_quote(quote):
        return "gene_change_not_in_quote"
    return None


def _gene_change(item: Any) -> StatedChange | None:
    """The gene change an asserted investigation item typed from its quote, if any."""
    return getattr(item.knowledge, "gene_change", None)


def _site(item: Any) -> SiteDraft | None:
    """The site an asserted investigation item typed from its quote, if any."""
    return getattr(item.knowledge, "site", None)


def _slot_verification(
    kind: str, rule: IdentifierRule | None
) -> SlotVerification | None:
    if kind in _FACT_SLOT_KINDS:
        slot_rule = requirement_identifier_rule(kind)
        assert slot_rule is not None
        return SlotVerification(source=slot_rule.source)
    if rule is RSID_RULE:
        return SlotVerification(source=RSID_RULE.source)
    return None


def _request_order(item: ProposalInformationRequirement) -> tuple:
    """Kind, then where the request first states it, so targets keep the
    researcher's order; a requirement with no quote follows, by subject key."""

    offsets = [
        evidence.start_offset
        for evidence in item.evidence
        if isinstance(evidence, RequestAssertionEvidence)
    ]
    return (item.kind, not offsets, min(offsets, default=0), item.subject_key)


def _cited(
    item: Any, citation_words: tuple[tuple[str, str], ...], request_event_id: str
) -> str | None:
    """The event whose researcher words hold a cited paper's quote on whole words:
    the request first, else the first other statement; None when none does."""
    holding = [
        event_id
        for event_id, text in citation_words
        if word_aligned_spans(text, item.knowledge.quote)
    ]
    if request_event_id in holding:
        return request_event_id
    return holding[0] if holding else None


def _citation_evidence(
    item: Any,
    citation_words: tuple[tuple[str, str], ...],
    recorded_papers: Mapping[str, str],
    located: Any,
    request_event_id: str,
    errors: list[str],
    facts: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    """A cited paper's evidence: the statement its quote stands in (the request
    span, else that statement's event), then each retrieved paper it cites, one of
    the Run's citation papers (membership, ADR 0049); none, with its refusal
    recorded, when either is not."""
    cited = _cited(item, citation_words, request_event_id)
    if cited is None:
        errors.append(f"{item.subject_key}:cited_paper_quote_ungrounded")
        return ()
    citation_papers = tuple(
        event_id for event_id, origin in recorded_papers.items() if origin == "citation"
    )
    if not set(item.knowledge.evidence_event_ids) <= set(citation_papers):
        code = f"{item.subject_key}:cited_paper_evidence_ungrounded"
        errors.append(code)
        facts[code] = {"allowed": citation_papers or ("none",)}
        return ()
    return (
        located(item, item.knowledge.quote)
        if cited == request_event_id
        else ContextEvidence(source_event_id=cited),
        *(
            ContextEvidence(source_event_id=event_id)
            for event_id in item.knowledge.evidence_event_ids
        ),
    )


def _registered_request_citations(
    draft: ProposalInformationPlanDraft, registered_request_event_ids: tuple[str, ...]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Each registered request file is cited by exactly one to_find scientific
    investigation item, which stands for its targets and takes no quote: an uncited
    file's targets would reach the Investigation unseen (#342), and a second citing
    item would stand for them twice (#367, #482)."""
    citing = [
        item.knowledge
        for item in draft.requirements
        if item.kind == "scientific_investigation"
        and isinstance(item.knowledge, ToFindKnowledgeDraft)
    ]
    counts = {
        registered: sum(
            registered in knowledge.evidence_event_ids for knowledge in citing
        )
        for registered in registered_request_event_ids
    }
    errors: list[str] = []
    facts: dict[str, dict[str, Any]] = {}
    if uncited := tuple(registered for registered, n in counts.items() if n == 0):
        errors.append("registered_request_uncited")
        facts["registered_request_uncited"] = {"event_ids": uncited}
    if twice := tuple(registered for registered, n in counts.items() if n > 1):
        errors.append("registered_request_cited_twice")
        facts["registered_request_cited_twice"] = {"event_ids": twice}
    if any(
        set(registered_request_event_ids) & set(knowledge.evidence_event_ids)
        and knowledge.quote is not None
        for knowledge in citing
    ):
        # The citing item stands for the file's targets; a target the request
        # itself names is its own asserted item, reconciled with the file.
        errors.append("registered_request_quoted")
    return errors, facts


def ground_proposal_information_plan(
    draft: ProposalInformationPlanDraft,
    *,
    journey_id: str,
    research_request: str,
    request_event_id: str,
    registration_event_id: str,
    inspection_event_id: str,
    input_set_id: str,
    inspection_result: InputMetadataResult,
    registered_request_event_ids: tuple[str, ...] = (),
    citation_words: tuple[tuple[str, str], ...] = (),
    recorded_papers: Mapping[str, str] | None = None,
) -> ProposalInformationPlan:
    """Accept a model plan only when every claim matches supplied Journey projections.

    The scope ids and observed input values come from the Journey, never the draft.
    A quote is located in the request (its first whole-word occurrence), and a typed
    value must be one its quote states under its kind's format matcher; the server
    checks the value and never extracts one. Each registered request file is
    planning's to place: a scientific investigation item must cite it, one file per
    item (#367), so its targets never reach the Investigation beside the plan unseen.
    A cited paper's quote stands in one of ``citation_words``, the researcher's
    words a paper citation may quote, each with its event (the paper Tools' own
    membership, ADR 0049); its evidence is the request span or that event, and the
    paper it cites when it was retrieved before the plan. ``recorded_papers`` are
    the Run's ``PaperRecorded`` events with their origins: a to_find value the paper
    will supply may cite one (owner redirect, 2026-10-29), and a cited paper cites
    its own, one of origin ``citation``.
    """

    errors: list[str] = []
    facts: dict[str, dict[str, Any]] = {}
    scope = {
        "journey_id": journey_id,
        "request_event_id": request_event_id,
        "registration_event_id": registration_event_id,
        "inspection_event_id": inspection_event_id,
        "input_set_id": input_set_id,
    }
    scope_digest = canonical_json_digest(scope)
    grounded: list[ProposalInformationRequirement] = []
    observed_roles: set[str] = set()
    ids_by_key = {
        item.subject_key: "requirement_"
        + canonical_json_digest([scope_digest, item.kind, item.subject_key])[:32]
        for item in draft.requirements
    }

    allowed_context = (
        request_event_id,
        registration_event_id,
        inspection_event_id,
        *registered_request_event_ids,
        *(recorded_papers or {}),
    )
    files_text = ", ".join(f"`{event_id}`" for event_id in registered_request_event_ids)

    def located(item: Any, quote: str) -> RequestAssertionEvidence | None:
        spans = word_aligned_spans(research_request, quote)
        if not spans:
            code = f"{item.subject_key}:request_assertion_ungrounded"
            errors.append(code)
            # A target the request does not state may be one only a file lists:
            # each file's one to_find item stands for its targets.
            facts[code] = {
                "file": (
                    " A target only a registered request file lists is not "
                    "quoted: each file's targets are the one to_find "
                    f"scientific_investigation item that cites it: {files_text}."
                    if registered_request_event_ids
                    and item.kind == "scientific_investigation"
                    else ""
                )
            }
            return None
        return RequestAssertionEvidence(
            source_event_id=request_event_id,
            start_offset=spans[0],
            end_offset=spans[0] + len(quote),
            exact_text=quote,
        )

    # A declared cited paper still to find: its citation names no PaperRecorded yet.
    paper_pending = any(
        item.kind == "cited_paper" and not item.knowledge.evidence_event_ids
        for item in draft.requirements
    )
    for item in draft.requirements:
        evidence: list[EvidenceReference] = []
        value: str | None = None
        rule: IdentifierRule | None = None
        if isinstance(item.knowledge, ToFindKnowledgeDraft):
            cited = set(item.knowledge.evidence_event_ids)
            files = cited & set(registered_request_event_ids)
            if not cited <= set(allowed_context):
                code = f"{item.subject_key}:context_evidence_ungrounded"
                errors.append(code)
                facts[code] = {"allowed": allowed_context}
            elif (
                item.kind == "scientific_investigation"
                and registered_request_event_ids
                and not files
                and not (
                    item.acquisition_route == "validate_with_tool"
                    and (
                        cited & set(recorded_papers or {})
                        or (request_event_id in cited and paper_pending)
                    )
                )
            ):
                # A to_find item citing no file asks the researcher to name targets,
                # which a registered file already lists: its targets are the file's
                # item and each target the request names, asserted. A target the
                # researcher says a cited paper gives is the paper's to supply, beside
                # the file: it cites that paper's PaperRecorded, or the request while a
                # declared cited paper is still to find.
                code = f"{item.subject_key}:target_question_beside_registered_request"
                errors.append(code)
                facts[code] = {"event_ids": registered_request_event_ids}
            elif files and item.kind != "scientific_investigation":
                # The file stands for the targets only. It is the registration's
                # sibling, not an ancestor of later planning facts, so no other
                # requirement's Tool grounding could ever hold it.
                errors.append(f"{item.subject_key}:registered_request_not_a_target")
            elif len(files) > 1:
                # One item stands for one file, so each file's targets keep their own
                # asserter and order.
                code = f"{item.subject_key}:registered_requests_cited_together"
                errors.append(code)
                facts[code] = {
                    "event_ids": tuple(
                        event_id
                        for event_id in registered_request_event_ids
                        if event_id in files
                    )
                }
            else:
                evidence.extend(
                    ContextEvidence(source_event_id=event_id)
                    for event_id in item.knowledge.evidence_event_ids
                )
            if item.knowledge.quote is not None:
                quote = located(item, item.knowledge.quote)
                if quote is not None:
                    evidence.append(quote)
        elif isinstance(item.knowledge, ObservedKnowledgeDraft):
            value = _observed_value(
                inspection_result, item.knowledge.input_role, item.knowledge.fact
            )
            if value is None:
                code = f"{item.subject_key}:input_observation_ungrounded"
                errors.append(code)
                facts[code] = {
                    "roles": tuple(entry.role for entry in inspection_result.inputs),
                    "facts": get_args(ObservedInputFact),
                }
            else:
                selected = next(
                    entry
                    for entry in inspection_result.inputs
                    if entry.role == item.knowledge.input_role
                )
                observed_roles.add(selected.role)
                evidence.append(
                    InspectedInputEvidence(
                        registration_event_id=registration_event_id,
                        inspection_event_id=inspection_event_id,
                        input_set_id=input_set_id,
                        role=selected.role,
                        registered_path=selected.registered_path,
                        fact=item.knowledge.fact,
                    )
                )
        elif isinstance(item.knowledge, AssertedCitationDraft):
            value = item.knowledge.quote
            evidence.extend(
                _citation_evidence(
                    item,
                    citation_words,
                    recorded_papers or {},
                    located,
                    request_event_id,
                    errors,
                    facts,
                )
            )
        else:
            typed, rule = _typed_value(item)
            quote = located(item, item.knowledge.quote)
            # The first failing check is reported: a quote the request lacks first.
            if quote is None:
                pass
            elif rule is not None and not rule.matches(typed, item.knowledge.quote):
                errors.append(f"{item.subject_key}:asserted_value_not_in_quote")
            elif (unquoted := _typed_target_unquoted(item)) is not None:
                errors.append(f"{item.subject_key}:{unquoted}")
            else:
                value = typed
                evidence.append(quote)
        grounded.append(
            ProposalInformationRequirement(
                requirement_id=ids_by_key[item.subject_key],
                kind=item.kind,
                subject_key=item.subject_key,
                domain=item.domain,
                meaning=item.meaning,
                knowledge_state=item.knowledge.state,
                value=value,
                evidence=tuple(evidence),
                acquisition_route=item.acquisition_route,
                validation=item.validation,
                investigation_purpose=getattr(item, "investigation_purpose", None),
                identifier_rule=None if rule is None else rule.name,
                verification=_slot_verification(item.kind, rule),
                stated_version=_stated_version(item) if value is not None else None,
                site=(
                    None
                    if value is None or (site := _site(item)) is None
                    else site.stated()
                ),
                gene_change=None if value is None else _gene_change(item),
            )
        )
    required_roles = {item.role for item in inspection_result.inputs}
    if observed_roles != required_roles:
        errors.append("registered_inputs_not_fully_represented")
        facts["registered_inputs_not_fully_represented"] = {
            "roles": tuple(sorted(required_roles - observed_roles))
        }
    if missing := missing_proposal_kinds(item.kind for item in draft.requirements):
        errors.append("proposal_requirement_kinds_incomplete")
        facts["proposal_requirement_kinds_incomplete"] = {"missing": missing}
    file_errors, file_facts = _registered_request_citations(
        draft, registered_request_event_ids
    )
    errors.extend(file_errors)
    facts.update(file_facts)
    if errors:
        raise ProposalPlanGroundingError(tuple(errors), facts)
    ordered = tuple(sorted(grounded, key=_request_order))
    plan_payload = {
        "scope": scope,
        "requirements": TypeAdapter(
            tuple[ProposalInformationRequirement, ...]
        ).dump_python(ordered, mode="json"),
    }
    return ProposalInformationPlan(
        plan_id="plan_" + canonical_json_digest(plan_payload),
        request_event_id=request_event_id,
        registration_event_id=registration_event_id,
        inspection_event_id=inspection_event_id,
        input_set_id=input_set_id,
        requirements=ordered,
    )
