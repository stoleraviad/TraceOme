"""One declaration per Agent role over one global role default (ADR 0040, #446).

Shaped like the model registry: every role inherits ``ROLE_DEFAULT`` (its
per-Attempt budgets and its stops) and declares only its differences: its job,
stages, output Tools, closure facts, member scope and instruction version. The
runner, the assembly and the Reviews read the declaration; a Run's runtime file
may override the default or one role (``agent_invocation.scopes.role``). The fault
and correction allowances are not here: they are a model fact, read from the model
policy bound on the Attempt (#440).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, NamedTuple, get_args

AgentRole = Literal[
    "planning_agent",
    "investigation_planner",
    "table_reader",
    "report_explainer",
    "paper_reader",
]
MemberScope = Literal["member", "role"]
# A refused output is one more failed turn: its stop is the Agent work stop.
StopKind = Literal["agent_work", "effort", "operational", "model_availability"]


class AttemptClosure(NamedTuple):
    """Outer terminal facts that end one role's Attempt without continuing."""

    stage: str
    # A role whose Attempts end in different facts names each.
    success_type: str | frozenset[str] | None
    failure_type: str


@dataclass(frozen=True, kw_only=True)
class RoleDefault:
    """What every role inherits: its per-Attempt budgets and its stops.

    Each member of a job is its own Attempt with its own budget; no Run total
    exists (owner, #439). Every stop kind opens a Review, scoped by the role's
    member scope (owner B2: "member scope applies to every stop kind").
    """

    model_requests: int
    failed_turns: int
    stops: tuple[StopKind, ...]


ROLE_DEFAULT = RoleDefault(
    model_requests=200,
    # One failure rule for every role: the fourth reply in a row in which nothing
    # worked stops the Attempt for the Agent work Review.
    failed_turns=4,
    stops=get_args(StopKind),
)


def _agent_role(name: str) -> str:
    if name not in get_args(AgentRole):
        raise ValueError(f"not an Agent role: {name}")
    return name


@dataclass(frozen=True, kw_only=True)
class RoleDeclaration:
    """One role's differences from the global default; every field is stated.

    ``jobs`` names, for each stage, the instruction block (``use = "message"``)
    its own message states as its job; the block's words live in the instruction
    module (#442). A member-scoped role's stop of any kind settles that member
    alone, its siblings going on; a role-scoped stop holds the role's work.
    """

    name: AgentRole
    jobs: Mapping[str, str]
    stages: tuple[str, ...]
    output_tools: frozenset[str]
    closure: AttemptClosure
    member_scope: MemberScope
    # What rejecting one member's stop does, in the role's words, shown with the Review.
    member_reject_effect: str | None
    # What dismissing the stopped item does (a member, a chunk), in the role's words;
    # None where a stop names no item.
    dismiss_effect: str | None
    # Whether the role's own domain settles a stopped member, so a stop of any kind,
    # model availability included, ends only that member and not the Run's work (the
    # shared availability projection reads it).
    settles_member_stops: bool
    instruction_version: str

    def __post_init__(self) -> None:
        _agent_role(self.name)
        if not self.stages:
            raise ValueError(f"{self.name} declares no stage")
        for stage in self.stages:
            if not self.jobs.get(stage):
                raise ValueError(f"{self.name} names no job block for stage {stage}")
        if set(self.jobs) - set(self.stages):
            raise ValueError(f"{self.name} names a job for a stage it has not")
        if self.member_scope not in get_args(MemberScope):
            raise ValueError(f"{self.name}'s scope is member or role")
        if (self.member_scope == "member") != (self.member_reject_effect is not None):
            raise ValueError(
                f"{self.name}: a member scope states its reject effect; "
                "a role scope states no member reject effect"
            )
        if self.settles_member_stops and self.member_scope != "member":
            raise ValueError(f"{self.name}: only a member scope settles member stops")
        object.__setattr__(self, "jobs", MappingProxyType(dict(self.jobs)))

    @property
    def job(self) -> str:
        """The one job block every stage of the role states; a role whose stages
        state different jobs is read by stage (``jobs``)."""
        (block,) = set(self.jobs.values())
        return block


ROLES: Mapping[str, RoleDeclaration] = MappingProxyType(
    {
        item.name: item
        for item in (
            RoleDeclaration(
                name="planning_agent",
                # The coordinator picks one stage per request from checked state.
                stages=(
                    "input_correction",
                    # A target answer goes back to planning's target reading.
                    "target_answer",
                    "proposal_requirements",
                    # A plan that cites a paper settles it before the reference
                    # work; a Run that cites none never enters it.
                    "paper",
                    "reference",
                    "companions",
                    "execution_planning",
                ),
                jobs={
                    stage: "job"
                    for stage in (
                        "input_correction",
                        "target_answer",
                        "proposal_requirements",
                        "paper",
                        "reference",
                        "companions",
                        "execution_planning",
                    )
                },
                output_tools=frozenset(),
                closure=AttemptClosure(
                    "planning_attempt", "AttemptSucceeded", "AttemptFailed"
                ),
                member_scope="role",
                member_reject_effect=None,
                dismiss_effect=None,
                settles_member_stops=False,
                instruction_version="v22",
            ),
            RoleDeclaration(
                name="investigation_planner",
                stages=("source_reading",),
                jobs={"source_reading": "job"},
                output_tools=frozenset({"compile_proposal"}),
                closure=AttemptClosure(
                    "investigation_planning_attempt",
                    "InvestigationAttemptCompleted",
                    "InvestigationAttemptFailed",
                ),
                # A request member is settled alone and its siblings go on.
                member_scope="member",
                member_reject_effect=(
                    "Stop this member's Agent work: the reject settles this member as "
                    "not resolved, with your reason retained in the accounting, and the "
                    "other members continue."
                ),
                dismiss_effect=(
                    "Dismiss this request: it leaves the pool as not resolved, with "
                    "your reason retained in the accounting, and the other members "
                    "continue."
                ),
                settles_member_stops=False,
                instruction_version="v24",
            ),
            RoleDeclaration(
                name="table_reader",
                # The VCF table reading offers no domain Tool: the table is in its
                # prompt.
                stages=("vcf_table_reading",),
                jobs={"vcf_table_reading": "job"},
                output_tools=frozenset({"record_item_readings"}),
                closure=AttemptClosure(
                    "investigation_table_reading_attempt",
                    "InvestigationTableChunkRead",
                    "InvestigationTableChunkFailed",
                ),
                # A chunk's stop holds the role's reading, and its domain folds the
                # remaining chunks not_run (ADR 0047).
                member_scope="role",
                member_reject_effect=None,
                dismiss_effect=(
                    "Dismiss this chunk: its items are recorded as not read, with your "
                    "reason, and the other chunks continue."
                ),
                settles_member_stops=False,
                instruction_version="v4",
            ),
            RoleDeclaration(
                name="report_explainer",
                # The explanation offers no domain Tool: its rows are in its prompt
                # and its search reads only the approved basis's evidence. Each
                # request's member explains it, then one combination answers over
                # them.
                stages=("report_explanation", "report_combination"),
                jobs={
                    "report_explanation": "job",
                    "report_combination": "combination_job",
                },
                output_tools=frozenset(
                    {"record_explanation", "record_combined_answer"}
                ),
                # A member's admitted section, or the combination's (or a one-Attempt
                # explanation's) publication.
                closure=AttemptClosure(
                    "report_explanation_attempt",
                    frozenset(
                        {
                            "ReportExplanationMemberAdmitted",
                            "ReportExplanationPublished",
                        }
                    ),
                    "ReportExplanationFailed",
                ),
                member_scope="member",
                member_reject_effect=(
                    "Stop this request's explanation member: the reject settles this "
                    "request as not explained, with your reason kept in the "
                    "explanation, and the other requests' members and the combined "
                    "answer continue."
                ),
                dismiss_effect=(
                    "Dismiss this request's explanation: it is settled as not "
                    "explained, with your reason kept in the explanation, and the "
                    "other requests and the combined answer continue."
                ),
                settles_member_stops=False,
                instruction_version="v20",
            ),
            RoleDeclaration(
                name="paper_reader",
                # One stage for both reading items: the paper item and its area or
                # id items differ by their output Tool, not by the reading Tools
                # they see.
                stages=("paper_reading",),
                jobs={"paper_reading": "job"},
                # The paper item proposes areas; an area or id item answers by
                # pointers.
                output_tools=frozenset(
                    {"propose_reading_areas", "answer_reading_item"}
                ),
                # An item's terminal; the reading's answer is the coordinator's,
                # recorded once every item is settled.
                closure=AttemptClosure(
                    "paper_reading_attempt", "ReadingItemCompleted", "AttemptFailed"
                ),
                # Each reading item is a member of its reading's collection: a stopped
                # item holds only itself and its siblings go on (ADR 0039, which ADR
                # 0046's revisit_if named).
                member_scope="member",
                member_reject_effect=(
                    "Stop this reading item: it is settled as not read, with your reason "
                    "retained, and the reading's other items continue."
                ),
                dismiss_effect=(
                    "Dismiss this reading item: it is settled as not read, with your "
                    "reason retained, and the reading's other items continue."
                ),
                settles_member_stops=True,
                instruction_version="v14",
            ),
        )
    }
)


def role(name: str) -> RoleDeclaration:
    """The role's declaration; an unknown name is refused."""
    return ROLES[_agent_role(name)]


def output_tool_names() -> frozenset[str]:
    """Every role's output Tool names."""
    return frozenset().union(*(item.output_tools for item in ROLES.values()))
