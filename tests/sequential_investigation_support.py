"""Offline support fakes for sequential Investigation contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from bio_neb.agentic_journey.continuation_policy import AgentInvocationOverrides
from bio_neb.agentic_journey.runtime_config import RuntimeConfiguration
from bio_neb.application.input_journey import (
    RegisteredFastqInput,
    RunInputsRegistered,
    fact_payload,
)
from bio_neb.investigation.investigation import (
    ApproveInvestigationResponse,
    InvestigationInput,
    InvestigationRequestItem,
    InvestigationStarted,
    InvestigationVcfIdentity,
    UserAssertion,
)
from bio_neb.investigation.investigation_commands import (
    BeginInvestigationCommand,
    RespondInvestigationReviewCommand,
    plan_collection_members,
    respond_investigation_review,
)
from bio_neb.investigation.investigation_observation import (
    InvestigationCollectionEvidenceRecorded,
    InvestigationObservationCommand,
)
from bio_neb.investigation.investigation_observation_commands import (
    MaterializationObservationEvidence,
    RuntimeProbeObservationEvidence,
    investigation_observe,
)
from bio_neb.investigation.investigation_policy import InvestigationPolicyAmendment
from bio_neb.investigation.investigation_policy_commands import (
    AmendInvestigationPolicyCommand,
    amend_investigation_policy,
)
from bio_neb.investigation.investigation_request import RSID_RULE
from bio_neb.investigation.vcf_observation import (
    AttestedPlainVcf,
    BcftoolsRuntimeBinding,
    CompletedProcess,
    LocalFileObservation,
    trimmed_allele,
)
from bio_neb.journey import Journey, canonical_json
from bio_neb.providers.ncbi_investigation import (
    InvestigationSourceResponse,
)
from bio_neb.recording import processing_pass, read_evidence
from bio_neb.verified_artifacts import VerifiedObject
from tests.location_retained_bodies import (
    GRCH37_NAMES,
    GRCH38_NAMES,
    SPDI_UNLISTED,
)
from tests.model_endpoint_fakes import ScriptedModelEndpoint
from tests.prerequisite_builds import measured_build
from tests.proposal_information_fakes import runtime_configuration
from tests.terminal_journey_fakes import forked_child
from tests.vcf_lookup_fakes import (
    lookup_command,
    query_position,
    query_row,
    table_reading_reply,
    vcf_header,
)

REFSNP_BODY = json.dumps(
    {
        "refsnp_id": "429358",
        "primary_snapshot_data": {
            "placements_with_allele": [
                {
                    "seq_id": "NC_000019.10",
                    "is_ptlp": True,
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [
                            {
                                "assembly_accession": "GCF_000001405.40",
                                "assembly_name": "GRCh38.p14",
                            }
                        ]
                    },
                    "alleles": [
                        {
                            "allele": {
                                "spdi": {
                                    "seq_id": "NC_000019.10",
                                    "position": 44908683,
                                    "deleted_sequence": "T",
                                    "inserted_sequence": "T",
                                }
                            }
                        },
                        {
                            "allele": {
                                "spdi": {
                                    "seq_id": "NC_000019.10",
                                    "position": 44908683,
                                    "deleted_sequence": "T",
                                    "inserted_sequence": "C",
                                }
                            }
                        },
                    ],
                }
            ]
        },
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()

ASSEMBLY_BODY = json.dumps(
    {
        "reports": [
            {
                "accession": "GCF_000001405.40",
                "assembly_info": {
                    "assembly_name": "GRCh38.p14",
                    "assembly_status": "current",
                },
            }
        ]
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()


class SourceResponsesOnly:
    def __init__(
        self,
        refsnp_body: bytes = REFSNP_BODY,
        assembly_body: bytes = ASSEMBLY_BODY,
        refsnp_bodies: dict[str, bytes] | None = None,
        reference_bases: dict[str, str] | None = None,
        listed_rsids: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self.refsnp_body = refsnp_body
        self.assembly_body = assembly_body
        # A body per queried rsID (merged and current records); others get
        # ``refsnp_body``.
        self.refsnp_bodies = refsnp_bodies or {}
        # A position lookup's answers: the reference bases at a span key and
        # the rsIDs listed at an SPDI; an unlisted SPDI answers 404.
        self.reference_bases = reference_bases or {}
        self.listed_rsids = listed_rsids or {}
        self.operations: list[str] = []

    @property
    def contact_count(self) -> int:
        return len(self.operations)

    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        self.operations.append(f"fetch_refsnp:{rsid}")
        return InvestigationSourceResponse(
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
                + rsid.removeprefix("rs")
            ),
            status_code=200,
            body=self.refsnp_bodies.get(rsid, self.refsnp_body),
            observed_at="2026-08-26T00:00:00Z",
            elapsed_time_ms=11,
        )

    def _answer(self, operation, key, url, status, body):
        from bio_neb.agentic_journey.source_answers import answer_received

        self.operations.append(f"{operation}:{key}")
        # Each answer reaches a Tool collecting answers, as the real transports report
        # theirs.
        answer_received(url, status, body)
        return InvestigationSourceResponse(
            request_identity=url,
            status_code=status,
            body=body,
            observed_at="2026-10-17T00:00:00Z",
            elapsed_time_ms=7,
        )

    def fetch_taxon_assemblies(self, taxon_query: str):  # noqa: ANN201
        """NCBI's genome reports for any query: #466's recorded human search for
        GRCh37, every version and its patch releases (#471; whatever the query asks,
        a test stand-in for NCBI's search)."""
        from tests.recorded_sources import fixture

        [answer] = [
            item
            for item in fixture("assembly-stated-grch37").answers
            if "search_text=GRCh37" in item.url
        ]
        return self._answer(
            "ncbi_taxon_assemblies", taxon_query, answer.url, 200, answer.body
        )

    def fetch_sequence_names(self, assembly_accession: str):  # noqa: ANN201
        """The assembly report's answer: its rows as the FTP report holds them, the
        GRCh38 and GRCh37 bodies derived from the retained sequence reports (#467)."""
        return self._answer(
            "ncbi_sequence_names",
            assembly_accession,
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
            f"{assembly_accession}_x/{assembly_accession}_x_assembly_report.txt",
            200,
            GRCH37_NAMES if assembly_accession == "GCF_000001405.25" else GRCH38_NAMES,
        )

    def _placed(self):
        """Every (rsID, sequence, start0, REF, ALT) the configured records place:
        this fake's dbSNP, read the way the real one lists an SPDI allele."""
        default = f"rs{json.loads(self.refsnp_body)['refsnp_id']}"
        bodies = {
            default: self.refsnp_body,
            **dict.fromkeys(self.positions_rsids(), None),
            **self.refsnp_bodies,
        }
        for rsid, body in bodies.items():
            document = json.loads(body if body is not None else self._body_for(rsid))
            for placement in document.get("primary_snapshot_data", {}).get(
                "placements_with_allele", ()
            ):
                for allele in placement["alleles"]:
                    spdi = allele["allele"]["spdi"]
                    yield (
                        rsid,
                        spdi["seq_id"],
                        spdi["position"],
                        spdi["deleted_sequence"],
                        spdi["inserted_sequence"],
                    )

    def positions_rsids(self):  # noqa: ANN201
        return ()

    def _body_for(self, rsid):  # noqa: ANN001, ANN201
        return self.refsnp_body

    def fetch_reference_span(self, span: str):  # noqa: ANN201
        sequence, start, length = span.split(":")
        # A configured span answers as given; otherwise the bases a configured
        # record places there, else bases no stated REF holds.
        bases = self.reference_bases.get(span) or next(
            (
                ref
                for _, seq, at, ref, _ in self._placed()
                if (seq, at, len(ref)) == (sequence, int(start), int(length))
            ),
            "N" * int(length),
        )
        body = canonical_json(
            {
                "data": {
                    "seq_id": sequence,
                    "position": int(start),
                    "deleted_sequence": bases,
                    "inserted_sequence": "N" * int(length),
                }
            }
        ).encode()
        url = f"https://api.ncbi.nlm.nih.gov/variation/v0/spdi/{span}/contextual"
        return self._answer("ncbi_reference_span", span, url, 200, body)

    def fetch_spdi_rsids(self, spdi: str):  # noqa: ANN201
        url = f"https://api.ncbi.nlm.nih.gov/variation/v0/spdi/{spdi}/rsids"
        sequence, start, ref, alt = spdi.split(":")
        # The service matches an allele in any equivalent form, as SPDI's own
        # normalization does: compared here trimmed.
        asked = (sequence, *trimmed_allele(int(start) + 1, ref, alt))
        listed = self.listed_rsids.get(spdi) or tuple(
            dict.fromkeys(
                int(rsid.removeprefix("rs"))
                for rsid, seq, at, deleted, inserted in self._placed()
                if deleted != inserted
                and (seq, *trimmed_allele(at + 1, deleted, inserted)) == asked
            )
        )
        if not listed:
            return self._answer("ncbi_spdi_rsids", spdi, url, 404, SPDI_UNLISTED)
        body = canonical_json({"data": {"rsids": list(listed)}})
        return self._answer("ncbi_spdi_rsids", spdi, url, 200, body.encode())

    def fetch_assembly_report(
        self, assembly_accession: str
    ) -> InvestigationSourceResponse:
        self.operations.append(f"fetch_assembly_report:{assembly_accession}")
        return InvestigationSourceResponse(
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
                f"{assembly_accession}/dataset_report"
            ),
            status_code=200,
            body=self.assembly_body,
            observed_at="2026-08-26T00:00:01Z",
            elapsed_time_ms=9,
        )


def begin_in_pass(
    command: BeginInvestigationCommand,
    *,
    environ,
    model_factory=None,
    source_adapter_factory=None,
):
    """Begin Investigation as its one caller does: inside ``advance``'s Pass.

    Returns the Journey's Investigation projection at the stop ``advance``
    reached, which is what the begin command itself returns.
    """
    from bio_neb.application.advancement import (
        BeginInvestigation,
        advance,
        use_run_advancement_dependencies,
    )
    from bio_neb.investigation.investigation import project_investigation
    from bio_neb.recording import read_evidence

    with use_run_advancement_dependencies(
        environ=environ,
        investigation_model_factory=model_factory,
        investigation_source_adapter_factory=source_adapter_factory,
    ):
        advance(BeginInvestigation(**command.model_dump()))
    return project_investigation(
        read_evidence(Journey.open(Path(command.journey_file)))
    )


def run_attempt(
    root: Path,
    model,
    adapter: SourceResponsesOnly,
    *,
    investigation_input: InvestigationInput | None = None,
    assertion: UserAssertion | None = None,
    continuation_policy: dict[str, int] | None = None,
    invocation_overrides: dict | None = None,
    agent_invocation: dict | None = None,
    investigation_table: dict | None = None,
    investigation_concurrency: dict | None = None,
    model_name: str | None = None,
    planning_gets: tuple[tuple[str, bytes, str], ...] = (),
):
    """Start an ordinary Investigation and run its first planning Attempt.

    ``planning_gets`` are GETs planning retained before the Investigation began
    (URL, bytes, media type), as `ReferenceSourceRetrieved` records (#479 G).

    A registration binds the runtime file, as an ordinary begin requires; the
    Attempt then runs through the shared lifecycle with these adapters.
    """
    selected_input, selected_assertion = input_and_assertion()
    selected_input = investigation_input or selected_input
    selected_assertion = assertion or selected_assertion
    journey = Journey.create(root)
    configuration = (
        runtime_configuration()
        if model_name is None
        else runtime_configuration(model=model_name)
    )
    if continuation_policy is not None:
        # Output repair is the role default's; correction allowances the model's.
        configuration["agent_invocation"] = {
            **configuration.get("agent_invocation", {}),
            **dict(continuation_policy),
        }
    if agent_invocation is not None:
        configuration["agent_invocation"] = {
            **configuration.get("agent_invocation", {}),
            **agent_invocation,
        }
    if investigation_table is not None:
        # The Run's selected VCF table chunk size and window.
        configuration["investigation_table"] = investigation_table
    # The Run's bound on planner members and table chunks at once. Scripted
    # models answer members in request order, so members run one at a time
    # here unless a test selects a bound; each still has its own Attempt.
    configuration["investigation_concurrency"] = investigation_concurrency or {
        "planner_members": 1,
        "table_chunks": 4,
    }
    runtime = RuntimeConfiguration.model_validate(configuration)
    runtime_file = journey.run_directory / "investigation-runtime.json"
    runtime_file.write_text(runtime.model_dump_json(), encoding="utf-8")
    registered = RunInputsRegistered(
        input_set_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "test-investigation")),
        run_root=str(journey.run_directory),
        runtime_config_file=str(runtime_file),
        runtime_selections=runtime.runtime_selections(),
        inputs=(
            RegisteredFastqInput(
                role="sample", registered_path=str(root / "sample.fastq.gz")
            ),
        ),
    )
    journey.record(registered.event_type, fact_payload(registered))
    for url, body, media_type in planning_gets:
        journey.record_with_artifact(
            "ReferenceSourceRetrieved",
            {
                "attempt_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "planning")),
                "operation": "inspect_reference",
                "request_identity": url,
                "status_code": 200,
                "byte_count": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            },
            body,
            media_type,
        )
    from bio_neb.investigation.run_reference import (
        carried_reference_sources,
        run_species_taxid,
    )

    prefix = read_evidence(journey)
    started = InvestigationStarted(
        continuation_id="scientific-continuation:" + "b" * 64,
        investigation_input=selected_input,
        assertion=selected_assertion,
        # What the start carries from planning, read as the begin command reads it
        # (Q4 A); a test with no planning facts carries none.
        run_reference_sources=carried_reference_sources(
            prefix, selected_input.assembly_accession
        ),
        run_taxid=run_species_taxid(prefix, selected_input.assembly_accession),
    )
    journey.record(
        started.event_type,
        started.model_dump(mode="json", exclude={"event_type"}),
    )
    # Each member is planned in its own Attempt; the proposal closes once every
    # member settles.
    with processing_pass(journey) as interval:
        projection = plan_collection_members(
            journey,
            environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
            model_factory=model.model,
            source_adapter_factory=lambda: adapter,
            invocation_overrides=lambda _member: AgentInvocationOverrides.model_validate(
                invocation_overrides or {}
            ),
        )
        interval.close("waiting" if "pending" in projection.status else "completed")
    return projection, journey


def _file(
    path: str, sha256: str, *, executable: bool = False, size: int = 10
) -> LocalFileObservation:
    return LocalFileObservation(
        path=path,
        exists=True,
        is_regular=True,
        is_symlink=False,
        is_writable=False,
        is_executable=executable,
        byte_count=size,
        sha256=sha256,
        device="1",
        inode=path,
        modified_at_ns=1,
    )


class ObservationEvidence:
    def __init__(self, descriptor: VerifiedObject) -> None:
        self.descriptor = descriptor
        self.calls: list[str] = []

    def materialization(
        self, *, receipt_file: str, vcf_file: str
    ) -> MaterializationObservationEvidence:
        self.calls.append("materialization")
        receipt = _file(receipt_file, "6" * 64, size=100)
        return MaterializationObservationEvidence(
            receipt_id="vcf-materialization-receipt:" + "7" * 64,
            descriptor=self.descriptor,
            attested_vcf=AttestedPlainVcf(
                local_path=vcf_file,
                byte_count=self.descriptor.byte_count,
                sha256=self.descriptor.sha256,
                acquisition_receipt_path=receipt_file,
                acquisition_receipt_sha256=receipt.sha256,
            ),
            receipt_file=receipt,
        )

    def runtime(
        self, *, receipt_file: str, executable: str
    ) -> RuntimeProbeObservationEvidence:
        self.calls.append("runtime")
        receipt = _file(receipt_file, "8" * 64, size=100)
        return RuntimeProbeObservationEvidence(
            runtime=BcftoolsRuntimeBinding(
                executable_path=executable,
                executable_sha256="9" * 64,
                bcftools_version="1.24",
                htslib_version="1.24",
                probe_receipt_path=receipt_file,
                probe_receipt_sha256=receipt.sha256,
            ),
            receipt_file=receipt,
        )


class ObservationAdapter:
    def __init__(
        self,
        command: InvestigationObservationCommand,
        descriptor: VerifiedObject,
        sample_id: str,
    ) -> None:
        self.command = command
        self.descriptor = descriptor
        self.sample_id = sample_id
        self.calls: list[tuple[str, ...]] = []
        self.files = {
            command.vcf_file: _file(
                command.vcf_file, descriptor.sha256, size=descriptor.byte_count
            ),
            command.materialization_receipt_file: _file(
                command.materialization_receipt_file, "6" * 64, size=100
            ),
            command.bcftools_executable: _file(
                command.bcftools_executable,
                "9" * 64,
                executable=True,
                size=100,
            ),
            command.bcftools_probe_receipt_file: _file(
                command.bcftools_probe_receipt_file, "8" * 64, size=100
            ),
        }

    def inspect_file(self, path: str) -> LocalFileObservation:
        return copy.deepcopy(
            self.files.get(
                path,
                LocalFileObservation(
                    path=path,
                    exists=False,
                    is_regular=False,
                    is_symlink=False,
                    is_writable=False,
                    is_executable=False,
                    byte_count=None,
                    sha256=None,
                    device=None,
                    inode=None,
                    modified_at_ns=None,
                ),
            )
        )

    def path_exists(self, path: str) -> bool:
        return path in self.files

    def create_fresh_directory(self, path: str) -> None:
        if self.path_exists(path):
            raise FileExistsError(path)

    def write_fresh_file(self, path, data):
        # A query's regions file: kept as written, never inspected.
        if self.path_exists(path):
            raise FileExistsError(path)
        self.files[path] = data

    def run(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: int,
        stdout_path: str | None = None,
    ) -> CompletedProcess:
        del timeout_seconds
        self.calls.append(tuple(argv))
        step = lookup_command(argv)
        if step == "bgzf_derivation":
            assert stdout_path is not None
            self.files[stdout_path] = _file(stdout_path, "a" * 64, size=50)
            return CompletedProcess(0, b"", b"")
        if step == "csi_derivation":
            index = argv[-1] + ".csi"
            self.files[index] = _file(index, "b" * 64, size=20)
            return CompletedProcess(0, b"", b"")
        if step == "header_inventory":
            return CompletedProcess(0, vcf_header(self.sample_id), b"")
        return CompletedProcess(
            0,
            query_row("NC_000019.10", 44908684, "T", "C", self.sample_id, "0/1"),
            b"",
        )


class SequentialSourceResponses(SourceResponsesOnly):
    positions = {
        "rs429358": 44908684,
        "rs7412": 44908822,
        "rs4149056": 21331549,
    }

    def positions_rsids(self):  # noqa: ANN201
        return tuple(self.positions)

    def _body_for(self, rsid):  # noqa: ANN001, ANN201
        body = json.loads(REFSNP_BODY)
        body["refsnp_id"] = int(rsid.removeprefix("rs"))
        for placement in body["primary_snapshot_data"]["placements_with_allele"]:
            for allele in placement["alleles"]:
                allele["allele"]["spdi"]["position"] = self.positions[rsid] - 1
        return canonical_json(body).encode()

    def fetch_refsnp(self, rsid: str):  # noqa: ANN201
        self.refsnp_body = self._body_for(rsid)
        return super().fetch_refsnp(rsid)


def view_result(content):
    """A lookup Tool's model view read back as the scripted model reads it (#478
    G1.3): its copied result, with its status and the source's statement."""
    return {
        **content.get("result", {}),
        "status": content["status"],
        "conflicts": [content["statement"]] if "statement" in content else [],
    }


def _current_member_messages(messages):
    """Read the active assignment and actual returns, excluding historical context."""
    retained = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    parts = [part for message in retained for part in message["parts"]]
    assignments = []
    for index, part in enumerate(parts):
        if part["part_kind"] != "user-prompt" or not isinstance(part["content"], str):
            continue
        try:
            context = json.loads(part["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(context, dict) and isinstance(
            context.get("user_assertion"), dict
        ):
            assignments.append((index, context["user_assertion"]))
    if not assignments:
        raise AssertionError("sequential model requires a direct current assignment")
    index, request = assignments[-1]
    # A source result comes back as its model view, which names the id it is cited
    # by; a refused call returns its correction (text, or a refusal's view), which
    # is no result and leaves the latest recorded one in place.
    returned = [
        (part["tool_name"], part["content"])
        for part in parts[index + 1 :]
        if part["part_kind"] == "tool-return"
        and part["tool_name"]
        in {
            "fetch_refsnp",
            "fetch_eva_variant",
            "locate_variant",
            "fetch_assembly_report",
        }
        and isinstance(part["content"], dict)
        and "cite" in part["content"]
    ]
    returns = dict(returned)
    # The decisions recorded since the latest lookup result: the next lookup
    # reads them, so they are not made again.
    last_lookup = max(
        (
            position
            for position, part in enumerate(parts[index + 1 :])
            if part["part_kind"] == "tool-return"
            and part["tool_name"]
            in {"fetch_refsnp", "fetch_eva_variant", "locate_variant"}
            and isinstance(part["content"], dict)
            and "cite" in part["content"]
        ),
        default=-1,
    )
    decided = [
        part["content"]["result"]
        for part in parts[index + 1 :][last_lookup + 1 :]
        if part["part_kind"] == "tool-return"
        and part["tool_name"] in DECISION_TOOLS
        and isinstance(part["content"], dict)
        and isinstance(part["content"].get("result"), dict)
    ]
    # Each assembly's latest page of sequence names the member searched.
    names = {
        part["content"]["result"]["assembly_accession"]: part["content"]["result"]
        for part in parts[index + 1 :]
        if part["part_kind"] == "tool-return"
        and part["tool_name"] == "search_sequence_names"
        and isinstance(part["content"], dict)
        and "rows" in part["content"].get("result", {})
    }
    # Every Run is offered both lookup Tools: the latest lookup result is read.
    lookups = [content for name, content in returned if name != "fetch_assembly_report"]
    refsnp, assembly = (
        lookups[-1] if lookups else None,
        returns.get("fetch_assembly_report"),
    )
    return (
        request,
        None if refsnp is None else view_result(refsnp),
        None if assembly is None else view_result(assembly),
        {
            "refsnp_event_id": "missing"
            if refsnp is None
            else refsnp["cite"]["refsnp_event_id"],
            "assembly_event_id": None
            if assembly is None
            else assembly["cite"]["assembly_event_id"],
        },
        names,
        decided,
    )


DECISION_TOOLS = (
    "name_sequence",
    "name_version",
    "name_placement",
    "quote_listed_none",
)
_NAME_COLUMNS = ("Sequence-Name", "GenBank-Accn", "RefSeq-Accn", "UCSC-style-name")
_ASSEMBLY = re.compile(r"GC[AF]_[0-9]{9}\.[0-9]+")


def scripted_listed_none(refsnp, messages):
    """The scripted model's reading of a listing service's answer (#468, owner R5 b):
    after a position lookup that ended on the listing's non-2xx answer, it quotes the
    answer's own words when they say no rsID is listed there (NCBI's "does not match any
    RSID"), or cites an answer's status when it has no body; ``None`` otherwise."""
    if (
        refsnp is None
        or refsnp["status"] != "source_answered"
        or refsnp["location"] == "not_applicable"
        or refsnp["location"].get("listed_none") is not None
    ):
        return None
    retained = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    content = next(
        part["content"]
        for message in reversed(retained)
        for part in reversed(message["parts"])
        if part["part_kind"] == "tool-return"
        and part["tool_name"] == "locate_variant"
        and isinstance(part["content"], dict)
        and "source_answers" in part["content"]
    )
    answer = content["source_answers"][-1]
    body = answer["body"]
    if body == "":
        # An answer with no body has no words to quote: the decision cites its
        # status.
        return {"status_code": answer["status_code"]}
    message = body.get("error", {}).get("message") if isinstance(body, dict) else None
    if isinstance(message, str) and "does not match any RSID" in message:
        return {"quote": "does not match any RSID"}
    return None


def scripted_placement(refsnp, lookup, judgments, decided):
    """The scripted model's answer to ``placement_unnamed`` (#468, owner Q1 a): it names
    the placement it judges the rsID means (``judgments`` by rsID, else the first one
    the result copies on the Run's assembly) with `name_placement`, then calls the same
    Tool again, which reads it (#479 G2.2)."""
    rsid = refsnp.get("verified_rsid") or refsnp.get("requested_rsid")
    if any(item["kind"] == "placement" for item in decided):
        if refsnp["location"] == "not_applicable":
            return lookup, {"rsid": refsnp.get("requested_rsid") or rsid}
        return "locate_variant", {}
    on_run = [p["run_contig"] for p in refsnp.get("placements", ()) if p["run_contig"]]
    return "name_placement", {
        "placement": judgments.get(rsid, on_run[0] if on_run else None)
    }


LOOKUP_SOURCES = {"fetch_refsnp": "ncbi_dbsnp", "fetch_eva_variant": "eva"}
LOOKUP_TOOLS = {source: tool for tool, source in LOOKUP_SOURCES.items()}
_HUMAN_SERIES = ("000001405", "009914755")
"""GRCh37/GRCh38 and T2T-CHM13: the scripted model's own knowledge of which Runs are
human, standing in for a model that reads the Run's organism (#469)."""


def scripted_lookups(assignment) -> tuple[str, ...]:
    """The scripted model's choice of lookup Tool from the Tools' metadata (dbSNP: human
    variants; EVA: non-human; #469): dbSNP on a human Run's assembly, else EVA."""
    accession = assignment["investigation_input"]["assembly_accession"]
    if accession[4:13] in _HUMAN_SERIES:
        return ("fetch_refsnp",)
    return ("fetch_eva_variant",)


def scripted_naming(contig, refsnp, names, judgments=None, decided=()):
    """The scripted model's naming of a contig's sequence (#467, owner D1 a): after a
    ``sequence_unnamed`` result, search each assembly its conflict names, then name on
    each the one row it judges the written contig means with `name_sequence`, quoting
    the contig as the request writes it, and call `locate_variant` again once each is
    named (#479 G2.2). ``None`` when it judges none."""
    wanted = _ASSEMBLY.findall(refsnp["conflicts"][0])
    for accession in wanted:
        if accession not in names:
            # It searches for the molecule it judges the contig means.
            return "search_sequence_names", {
                "assembly_accession": accession,
                "text": (judgments or {}).get(contig, contig),
            }
    named = {
        item["choice"]["assembly_accession"]
        for item in decided
        if item["kind"] == "sequence"
    }
    if set(wanted) <= named:
        return "locate_variant", {}
    for accession, page in names.items():
        if accession not in wanted or accession in named:
            continue
        # The scripted model's judgment: a row literally named the contig, else the
        # one assembled molecule the contig names by its Assigned-Molecule (a match
        # the lookup records as model_matched).
        held = [
            row
            for row in page["rows"]
            if contig in {row[column] for column in _NAME_COLUMNS}
        ] or [
            row
            for row in page["rows"]
            if row["Assigned-Molecule"] == (judgments or {}).get(contig, contig)
            and row["Sequence-Role"] == "assembled-molecule"
        ]
        if len(held) != 1:
            return None
        row = held[0]
        return "name_sequence", {
            "assembly_accession": accession,
            "record": row["RefSeq-Accn"] or row["GenBank-Accn"],
            "quote": contig,
        }
    return None


def scripted_searches(messages) -> list[dict]:
    """The rows of every search_assemblies page the member read (#471), newest last."""
    retained = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    return [
        row
        for message in retained
        for part in message["parts"]
        if part["part_kind"] == "tool-return"
        and part["tool_name"] == "search_assemblies"
        and isinstance(part["content"], dict)
        and "result" in part["content"]
        for row in part["content"]["result"]["rows"]
    ]


def scripted_version(request, refsnp, searches, assignment, judgments=None, decided=()):
    """The scripted model's naming of a stated genome version (#471, owner Q3 a): after
    ``version_unnamed``, the assembly whose name is the stated version (or the one it
    judges it means, ``judgments``), or one of its patch releases, among the Run's own,
    the dbSNP record's placement assemblies and the rows it searched; with none, it
    searches NCBI's genome reports for the stated words on a human Run (its own
    knowledge of the organism, as `scripted_lookups`). ``None`` when it judges none."""
    if any(item["kind"] == "version" for item in decided):
        # The named version is read by the next lookup.
        return "locate_variant", {}
    stated = request["stated_assembly"]
    wanted = (judgments or {}).get(stated["name"], stated["name"])
    location = refsnp["location"]
    candidates = [
        (refsnp["expected_assembly_accession"], location["run_assembly_name"]),
        (refsnp["expected_assembly_accession"], location.get("run_assembly_synonym")),
        *(
            (assembly["assembly_accession"], assembly["assembly_name"])
            for placement in refsnp.get("placements", ())
            for assembly in (placement.get("dbsnp") or {}).get("assemblies", ())
        ),
        *((row["assembly_accession"], row.get("assembly_name")) for row in searches),
        *((row["assembly_accession"], row.get("synonym")) for row in searches),
    ]
    named = [
        accession for accession, name in candidates if wanted in (name, accession)
    ] or [
        accession
        for accession, name in candidates
        if name and name.startswith(wanted + ".p")
    ]
    if not named:
        accession = assignment["investigation_input"]["assembly_accession"]
        if searches or accession[4:13] not in _HUMAN_SERIES:
            return None
        return "search_assemblies", {
            "search_text": [wanted],
            "assembly_version": "all_assemblies",
        }
    quote = (
        stated["name"]
        if stated["name"] in stated["cited_line"]
        else stated["cited_line"]
    )
    return "name_version", {"assembly_accession": named[0], "quote": quote}


def source_basis(info):
    """Read this request's public dynamic findings for exact observation citations."""
    # The request's view is the JSON line its instructions end with.
    findings = json.loads(info.instructions.rpartition("\n")[2])["stage_findings"][
        "findings"
    ]
    refsnp = [row for row in findings if row["kind"] == "RefSnpToolResultRecorded"]
    assembly = [
        row for row in findings if row["kind"] == "AssemblyReportToolResultRecorded"
    ]
    return {
        "refsnp_event_id": refsnp[-1]["source_event_id"] if refsnp else "missing",
        "assembly_event_id": assembly[-1]["source_event_id"] if assembly else None,
    }


class SequentialAdaptiveModel:
    rsids = ("rs429358", "rs7412", "rs4149056")
    positions = (44908684, 44908822, 21331549)

    def __init__(
        self,
        judgments: dict[str, str] | None = None,
        placements: dict[str, str] | None = None,
        lookups: tuple[str, ...] | None = None,
    ) -> None:
        self.calls = 0
        # The lookup Tools it tries in order, the next after a "no record": a test's
        # stand-in for the model's choice from the Tools' metadata; by default it reads
        # the Run's organism from its assembly (`scripted_lookups`).
        self.lookups = lookups
        self.visible_messages: list[str] = []
        # The molecule the scripted model judges a written contig means, by the
        # written contig: a test's stand-in for the model's own judgment.
        self.judgments = judgments or {}
        # The placement it judges a record's rsID means when the record has several
        # on the Run's assembly, by rsID; else the first copied.
        self.placements = placements or {}
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        if (reply := table_reading_reply(info, messages)) is not None:
            return reply
        if (reply := member_words_reply(messages, info)) is not None:
            self.calls += 1
            return reply
        visible = ModelMessagesTypeAdapter.dump_json(messages).decode()
        self.visible_messages.append(visible)
        # Cite the observations by the source_event_id their Tool returns carry,
        # as a model reading its own transcript would; a resumed request's
        # retained findings may predate the latest lookup.
        request, refsnp, assembly, basis, names, decided = _current_member_messages(
            messages
        )
        answers = latest_assignment(messages).get("researcher_answers", ())
        written = (
            (answers[-1].get("target") or {}).get("position") if answers else None
        ) or (request.get("position") or request.get("region") or {})
        naming = (
            scripted_naming(
                written.get("contig"), refsnp, names, self.judgments, decided
            )
            if refsnp is not None and refsnp["status"] == "sequence_unnamed"
            else None
        )
        rsid = request["rsid"]
        self.calls += 1
        lookups = self.lookups or scripted_lookups(latest_assignment(messages))
        lookup = lookups[0]
        source = LOOKUP_SOURCES[lookup]
        placement = request.get("placement") or {}
        version = (
            scripted_version(
                request,
                refsnp,
                scripted_searches(messages),
                latest_assignment(messages),
                self.judgments,
                decided,
            )
            if refsnp is not None and refsnp["status"] == "version_unnamed"
            else None
        )
        if naming is not None:
            name, arguments = naming
        elif version is not None:
            name, arguments = version
        elif refsnp is not None and refsnp["status"] == "placement_unnamed":
            name, arguments = scripted_placement(
                refsnp, lookup, self.placements, decided
            )
        elif (quote := scripted_listed_none(refsnp, messages)) is not None:
            # The listing's words are quoted once, then the lookup reads
            # them.
            name, arguments = (
                ("locate_variant", {})
                if any(item["kind"] == "listed_none" for item in decided)
                else ("quote_listed_none", quote)
            )
        elif (
            request.get("region") is not None
            and refsnp is not None
            and (refsnp["status"] == "region_located")
        ):
            # A located region is complete on its own, citing no assembly
            # report; code copies its fields.
            name = info.output_tools[0].name
            arguments = {"outcome": "complete", "region": True}
        elif refsnp is None and request.get("region") is not None:
            name, arguments = "locate_variant", {}
        elif refsnp is None and (
            placement.get("position") is not None
            or (request.get("position") is not None and placement.get("step") != "none")
        ):
            # A stated position is placed first, from the source the model chose.
            name, arguments = "locate_variant", {"source": source}
        elif refsnp is None:
            name = lookup
            arguments = {"rsid": rsid}
        elif (
            refsnp["status"] in {"not_found", "source_answered"}
            and refsnp["location"] == "not_applicable"
            and (tried := LOOKUP_TOOLS[refsnp["lookup_source"]]) in lookups
            and lookups.index(tried) + 1 < len(lookups)
        ):
            # A source with no record: the next source it chose, same rsID.
            name = lookups[lookups.index(tried) + 1]
            arguments = {"rsid": rsid}
        elif refsnp["status"] == "merged":
            # Follow dbSNP's merge to the current rsID it names.
            name = "fetch_refsnp"
            arguments = {"rsid": refsnp["merged_into"][0]}
        elif refsnp["status"] not in {"resolved", "unlisted"} or (
            assembly is not None and assembly["status"] != "compatible"
        ):
            name = info.output_tools[0].name
            arguments = {"outcome": "unresolved"}
        elif assembly is None:
            name = "fetch_assembly_report"
            arguments = {
                "assembly_accession": refsnp["candidates"][0]["assembly_accession"]
            }
        else:
            # The model chooses the outcome and cites; code copies every field.
            name = info.output_tools[0].name
            arguments = {"outcome": "complete"}
        if name == "fetch_assembly_report" or name in DECISION_TOOLS:
            arguments["refsnp_event_id"] = basis["refsnp_event_id"]
        elif name == "locate_variant" and request.get("region") is None:
            # A position's lookup names its source on every call.
            arguments.setdefault("source", source)
        elif name == "compile_proposal":
            arguments.update(basis)
            if arguments.pop("region", False) or arguments["assembly_event_id"] is None:
                arguments.pop("assembly_event_id")
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name,
                    canonical_json(arguments),
                    tool_call_id=f"sequential-investigation-{self.calls}",
                )
            ]
        )


# rsIDs dbSNP does not know, as a recorded live Run saw
# rs9999999999.
NOT_FOUND = "rs9999999999"
ALSO_NOT_FOUND = "rs8888888888"


class IssueSourceResponses(SequentialSourceResponses):
    """dbSNP as the #107 Run saw it: one rsID it does not know, the rest current."""

    def fetch_refsnp(self, rsid: str) -> InvestigationSourceResponse:
        if rsid not in {NOT_FOUND, ALSO_NOT_FOUND}:
            return super().fetch_refsnp(rsid)
        self.operations.append(f"fetch_refsnp:{rsid}")
        return InvestigationSourceResponse(
            request_identity=(
                "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/"
                + rsid.removeprefix("rs")
            ),
            status_code=404,
            body=b'{"error":{"code":404,"message":"RefSNP not found"}}',
            observed_at="2026-08-26T00:00:00Z",
            elapsed_time_ms=11,
        )


class AmendedLookupModel(SequentialAdaptiveModel):
    """Looks up the rsID the researcher's amend states, as a model reading its
    assignment's researcher answers would; otherwise the sequential model."""

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        if scripted_target_reading(messages, info) is not None:
            # An answer's words are read before any lookup.
            return await super().respond(messages, info)
        _request, refsnp, _assembly, _basis, _names, _decided = (
            _current_member_messages(messages)
        )
        answers = latest_assignment(messages).get("researcher_answers", ())
        typed = answers[-1].get("target", {}) if answers else {}
        if refsnp is None and "position" not in typed and typed.get("rsid"):
            # A typed amend states only an rsID: that rsID is looked up.
            self.calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_refsnp",
                        canonical_json({"rsid": typed["rsid"]}),
                        tool_call_id=f"amended-lookup-{self.calls}",
                    )
                ]
            )
        if refsnp is None and answers and "position" in answers[-1].get("target", {}):
            # A typed amend states a position: it is placed first, from the source
            # the model chooses.
            self.calls += 1
            lookups = self.lookups or scripted_lookups(latest_assignment(messages))
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "locate_variant",
                        canonical_json({"source": LOOKUP_SOURCES[lookups[0]]}),
                        tool_call_id=f"amended-lookup-{self.calls}",
                    )
                ]
            )
        stated = [
            word.rstrip(".")
            for answer in latest_assignment(messages).get("researcher_answers", ())
            for word in answer["instruction"].split()
            if word.startswith("rs")
        ]
        if refsnp is None and stated:
            self.calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_refsnp",
                        canonical_json({"rsid": stated[-1]}),
                        tool_call_id=f"amended-lookup-{self.calls}",
                    )
                ]
            )
        return await super().respond(messages, info)


def target_reading_reply(messages, info) -> ModelResponse | None:
    """The scripted model's ``record_target_reading`` call while its member has words
    to read (#472 Q3 a), else None: the first reply of any scripted Investigation
    model, as ``table_reading_reply`` is for the table reader."""
    if (reading := scripted_target_reading(messages, info)) is None:
        return None
    return ModelResponse(
        parts=[
            ToolCallPart(
                "record_target_reading",
                reading,
                tool_call_id=f"target-reading-{reading['source_event_id']}",
            )
        ]
    )


def member_words_reply(messages, info) -> ModelResponse | None:
    """A scripted model's reply about its member's words (#472 Q3 a, R1 b): it reads
    words it has yet to read, then asks which target words that name none mean, before
    any lookup; else None."""
    if (reply := target_reading_reply(messages, info)) is not None:
        return reply
    request, refsnp, *_ = _current_member_messages(messages)
    answers = latest_assignment(messages).get("researcher_answers", ())
    # A reading typed in this Attempt is the model's own call, after its assignment.
    read = [
        {"target": part.args_as_dict()}
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
        and part.tool_name == "record_target_reading"
        and {"rsid", "site"} & set(part.args_as_dict())
    ]
    return target_question(request, [*answers, *read]) if refsnp is None else None


def target_question(request: dict, answers) -> ModelResponse | None:
    """The question a scripted model asks when its member's words, read, name no
    target (#472 R1 b), else None."""
    if untyped_target_known(request, answers):
        return None
    return ModelResponse(
        parts=[
            ToolCallPart(
                "request_investigation_information",
                {
                    "question": "Which rsID or site do you mean by "
                    f"{request['request_text']!r}?",
                    "missing_information": "The request's words name no target.",
                    "asks_about": "target",
                },
                tool_call_id="target-question",
            )
        ]
    )


def untyped_target_known(request: dict, answers) -> bool:
    """Whether the member has a typed target: its request's own, or one an answer or
    a reading of its words typed (#472)."""
    return any(
        request.get(field) is not None
        for field in ("rsid", "position", "region", "gene_change", "placement")
    ) or any(answer.get("target") for answer in answers)


def scripted_target_reading(messages, info) -> dict | None:
    """The reading a model makes of the first words its member still has to read
    (#472 Q3 a, R1 b): the last rsID the words state, quoting them whole, else why
    they state none. The test's stand-in for the model's reading, not the product's."""
    from bio_neb.investigation.investigation_request import RSID_RULE

    tool = next(
        (t for t in info.function_tools if t.name == "record_target_reading"), None
    )
    if tool is None:
        return None
    unread = tool.parameters_json_schema["properties"]["source_event_id"].get("enum")
    if not unread:
        return None
    context = latest_assignment(messages)
    words = {
        answer["response_event_id"]: answer["instruction"]
        for answer in context.get("researcher_answers", ())
        if "instruction" in answer
    }
    if "request_words_event_id" in context:
        words[context["request_words_event_id"]] = context["user_assertion"][
            "request_text"
        ]
    source = unread[0]
    text = words[source]
    stated = RSID_RULE.stated(text)
    if stated:
        return {"source_event_id": source, "quote": text, "rsid": stated[-1]}
    return {"source_event_id": source, "reason": f"The words {text!r} name no rsID."}


def latest_assignment(messages) -> dict:
    contexts = []
    for message in ModelMessagesTypeAdapter.dump_python(messages, mode="json"):
        for part in message["parts"]:
            if part["part_kind"] != "user-prompt" or not isinstance(
                part["content"], str
            ):
                continue
            try:
                context = json.loads(part["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(context, dict) and "user_assertion" in context:
                contexts.append(context)
    return contexts[-1]


class SecondOutputRepairOutageModel(SequentialAdaptiveModel):
    """Sends the second member's first output without its outcome, so it needs
    repair, then is unavailable for that repair request (the sixth call)."""

    async def respond(self, messages, info):  # noqa: ANN001, ANN201
        if self.calls == 6:
            raise ModelHTTPError(503, "offline", headers={"retry-after": "999"})
        response = await super().respond(messages, info)
        if self.calls == 6:
            arguments = response.parts[0].args_as_dict()
            del arguments["outcome"]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "compile_proposal",
                        arguments,
                        tool_call_id="invalid-second-member",
                    )
                ]
            )
        return response


def collection_rsids(cardinality: int) -> tuple[str, ...]:
    """The collection helper's targets, in its order."""
    return (
        *SequentialAdaptiveModel.rsids,
        "rs28503599",
        *(f"rs{1000 + index}" for index in range(cardinality)),
    )[:cardinality]


def collection_request(cardinality: int) -> dict:
    """The collection helper's request document; a forked collection's parent
    registers it, so the parent's plan names these targets (#342)."""
    return {
        "schema_version": "bio-neb.investigation-request/2",
        "asserted_by": "researcher:owner",
        "requests": [
            {
                "request": f"Investigate {rsid} in sample HG001.",
                "target": {"rsid": rsid},
            }
            for rsid in collection_rsids(cardinality)
        ],
    }


@measured_build("approved-collection-observation")
def approved_collection_observation(
    root: Path,
    *,
    cardinality: int = 3,
    forked: bool = False,
    approve_count: int | None = None,
    vcf_content: bytes | None = None,
    parent_file: Path | None = None,
    investigation_table: dict | None = None,
    investigation_concurrency: dict | None = None,
):  # noqa: ANN201
    if parent_file is not None and not forked:
        raise ValueError("parent_file applies only to a forked collection")

    class CollectionSources(SequentialSourceResponses):
        positions = {
            **SequentialSourceResponses.positions,
            "rs28503599": 14574,
            **{f"rs{1000 + index}": 15000 + index for index in range(cardinality)},
        }

    rsids = (
        *SequentialAdaptiveModel.rsids,
        "rs28503599",
        *(f"rs{1000 + index}" for index in range(cardinality)),
    )
    investigation_input = InvestigationInput.create(
        vcf=InvestigationVcfIdentity.create(
            bucket="accepted-run-bucket",
            object_key="outputs/HG001.vcf.gz",
            byte_count=1234 if vcf_content is None else len(vcf_content),
            sha256="a" * 64
            if vcf_content is None
            else hashlib.sha256(vcf_content).hexdigest(),
        ),
        sample_id="HG001",
        assembly_accession="GCF_000001405.40",
    )
    requests = tuple(
        InvestigationRequestItem(
            request_text=f"Investigate {rsid} in sample HG001.", rsid=rsid
        )
        for rsid in rsids[:cardinality]
    )
    assertion = (
        UserAssertion.create(
            asserted_by="researcher:owner",
            request_text=requests[0].request_text,
            rsid=requests[0].rsid,
            predecessor_ids=(investigation_input.input_id,),
        )
        if cardinality == 1
        else UserAssertion.create_collection(
            asserted_by="researcher:owner",
            requests=requests,
            predecessor_ids=(investigation_input.input_id,),
        )
    )
    if forked:
        document = collection_request(cardinality)
        journey = forked_child(
            root,
            parent_file=parent_file,
            investigation_request=None if parent_file else document,
        )
        request_file = root / "investigation-request.json"
        request_file.write_text(canonical_json(document), encoding="utf-8")
        pending = begin_in_pass(
            BeginInvestigationCommand(
                journey_file=str(journey.path),
                request_files=(str(request_file),),
            ),
            environ={"LOCAL_MODEL_KEY": "offline-test-value"},
            model_factory=lambda loaded, selected: SequentialAdaptiveModel().model(
                loaded, selected
            ),
            source_adapter_factory=lambda: CollectionSources(
                assembly_body=ASSEMBLY_BODY
            ),
        )
        assert pending.investigation_input is not None
        investigation_input = pending.investigation_input
    else:
        pending, journey = run_attempt(
            root / "planning",
            SequentialAdaptiveModel(),
            CollectionSources(assembly_body=ASSEMBLY_BODY),
            investigation_input=investigation_input,
            assertion=assertion,
            investigation_table=investigation_table,
            investigation_concurrency=investigation_concurrency,
        )
    assert pending.pending_request is not None
    approved = respond_investigation_review(
        RespondInvestigationReviewCommand(
            journey_file=str(journey.path),
            request_event_id=pending.pending_request.request_event_id,
            response=ApproveInvestigationResponse(
                action="approve",
                proposal_id=pending.pending_request.proposal.proposal_id,
                **(
                    {
                        "work_item_ids": tuple(
                            item.work_item_id
                            for item in pending.pending_request.proposal.work_items[
                                :approve_count
                            ]
                        )
                    }
                    if approve_count is not None
                    else {}
                ),
            ),
        )
    )
    authority = approved.authority
    assert authority is not None
    descriptor = VerifiedObject(
        role="VCF",
        bucket=investigation_input.vcf.bucket,
        object_key=investigation_input.vcf.object_key,
        byte_count=investigation_input.vcf.byte_count,
        sha256=investigation_input.vcf.sha256,
    )
    command = InvestigationObservationCommand(
        journey_file=str(journey.path),
        authority_id=authority.authority_id,
        vcf_file=str(root / "variants.vcf"),
        materialization_receipt_file=str(root / "materialization.json"),
        bcftools_executable=str(root / "bcftools"),
        bcftools_probe_receipt_file=str(root / "probe.json"),
        derivation_root=str(root / "derived"),
    )
    return investigation_input, journey, authority, descriptor, command


__all__ = [
    "ASSEMBLY_BODY",
    "ObservationAdapter",
    "ObservationEvidence",
    "REFSNP_BODY",
    "SequentialAdaptiveModel",
    "SequentialSourceResponses",
    "SourceResponsesOnly",
    "approved_collection_observation",
    "forked_child",
    "run_attempt",
]


def input_and_assertion() -> tuple[InvestigationInput, UserAssertion]:
    investigation_input = InvestigationInput.create(
        vcf=InvestigationVcfIdentity.create(
            bucket="accepted-run-bucket",
            object_key="outputs/HG001.vcf.gz",
            byte_count=1234,
            sha256="a" * 64,
        ),
        sample_id="HG001",
        assembly_accession="GCF_000001405.40",
    )
    assertion = UserAssertion.create(
        asserted_by="researcher:owner",
        request_text="Investigate rs429358 in sample HG001.",
        rsid="rs429358",
        predecessor_ids=(investigation_input.input_id,),
    )
    return investigation_input, assertion


def _typed_results(value):
    if isinstance(value, dict):
        if "result_id" in value and "status" in value:
            yield value
        elif "cite" in value and "status" in value:
            # A source Tool's model view.
            yield view_result(value)
        for item in value.values():
            yield from _typed_results(item)
    elif isinstance(value, list):
        for item in value:
            yield from _typed_results(item)
    elif isinstance(value, str) and value.startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return
        yield from _typed_results(parsed)


class _AdaptiveInvestigationModel:
    """Choose each next action from the typed responses visible in messages."""

    def __init__(self, *, first_action_ineligible: bool = False) -> None:
        self.calls = 0
        self.first_action_ineligible = first_action_ineligible
        self.visible_messages: list[str] = []
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        if (reply := table_reading_reply(info, messages)) is not None:
            return reply
        visible = ModelMessagesTypeAdapter.dump_json(messages).decode()
        self.visible_messages.append(visible)
        self.calls += 1
        if self.first_action_ineligible and self.calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_assembly_report",
                        canonical_json(
                            {
                                "assembly_accession": "GCF_000001405.40",
                                "refsnp_event_id": "missing",
                            }
                        ),
                        tool_call_id=f"investigation-{self.calls}",
                    )
                ]
            )
        returned = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        latest = returned[-1] if returned else None
        if isinstance(latest, dict) and "cite" in latest:
            latest = view_result(latest)  # A source Tool returns its model view.
        latest_status = (
            latest.get("status")
            if isinstance(latest, dict)
            else getattr(latest, "status", None)
        )
        if latest_status is None:
            retained_results = list(_typed_results(json.loads(visible)))
            if retained_results:
                latest_status = retained_results[-1]["status"]
        if latest_status in {
            "ambiguous",
            "conflicted",
            "incompatible",
            "not_found",
            "malformed",
            # A source's own answer and an unnamed placement: this model
            # reports them unresolved rather than reading or naming.
            "source_answered",
            "placement_unnamed",
        }:
            name = info.output_tools[0].name
            arguments = {"outcome": "unresolved"}
        elif '"cite":{"refsnp_event_id"' not in visible:
            # Look up the member's own rsID: another one is refused.
            request = _current_member_messages(messages)[0]
            name = "fetch_refsnp"
            # The scripted model's own reading of the words (no product code reads
            # an rsID out of them).
            arguments = {
                "rsid": request.get("rsid")
                or next(iter(RSID_RULE.stated(request["request_text"])), None)
                or "rs429358"
            }
        elif '"cite":{"assembly_event_id"' not in visible:
            name = "fetch_assembly_report"
            arguments = {"assembly_accession": "GCF_000001405.40"}
        else:
            name = info.output_tools[0].name
            arguments = {"outcome": "complete"}
        if name == "fetch_assembly_report":
            arguments["refsnp_event_id"] = source_basis(info)["refsnp_event_id"]
        elif name == "compile_proposal":
            arguments.update(source_basis(info))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name,
                    canonical_json(arguments),
                    tool_call_id=f"investigation-{self.calls}",
                )
            ]
        )


class _IndependentCorrectionsModel(_AdaptiveInvestigationModel):
    def __init__(self, *, wrong_actions: int = 2, wrong_outputs: int = 2) -> None:
        super().__init__()
        self.invalid_outputs = 0
        self.wrong_actions = wrong_actions
        self.wrong_outputs = wrong_outputs

    async def respond(self, messages, info):
        if self.calls < self.wrong_actions:
            self.calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "fetch_assembly_report",
                        canonical_json(
                            {
                                "assembly_accession": "GCF_000001405.40",
                                "refsnp_event_id": "missing",
                            }
                        ),
                        tool_call_id=f"wrong-action-{self.calls}",
                    )
                ]
            )
        response = await super().respond(messages, info)
        if (
            response.parts[0].tool_name == "compile_proposal"
            and self.invalid_outputs < self.wrong_outputs
        ):
            self.invalid_outputs += 1
            arguments = response.parts[0].args_as_dict()
            arguments["one_based_position"] = self.invalid_outputs
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "compile_proposal",
                        canonical_json(arguments),
                        tool_call_id=f"wrong-output-{self.invalid_outputs}",
                    )
                ]
            )
        return response


class _RepeatedActionModel:
    """Retain one result, then exhaust corrections for a missing source basis."""

    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        del messages, info
        self.calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "fetch_refsnp" if self.calls == 1 else "fetch_assembly_report",
                    canonical_json(
                        {"rsid": "rs429358"}
                        if self.calls == 1
                        else {
                            "assembly_accession": "GCF_000001405.40",
                            "refsnp_event_id": "missing",
                        }
                    ),
                    tool_call_id=f"repeated-{self.calls}",
                )
            ]
        )


class _MalformedOutputModel:
    def __init__(self) -> None:
        self.calls = 0
        self.model = ScriptedModelEndpoint(self.respond)

    async def respond(self, messages, info):
        del messages
        self.calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    canonical_json({"outcome": "complete"}),
                    tool_call_id=f"malformed-{self.calls}",
                )
            ]
        )


def bundle_bytes(directory: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


# The three-member parent's request document; its executed parent registers it.
THREE_MEMBER_REQUEST = {
    "schema_version": "bio-neb.investigation-request/2",
    "asserted_by": "researcher:owner",
    "requests": [
        {
            "request": f"Investigate {rsid} in sample-001.",
            "target": {"rsid": rsid},
            "normalized_question_action": "observe_variant",
        }
        for rsid in SequentialAdaptiveModel.rsids
    ],
}


def retained_three_member_parent(
    root: Path,
    *,
    strict_policy: bool = False,
    unavailable_origin: bool = False,
    coordinator_pass: bool = False,
    parent_file: Path | None = None,
) -> Journey:
    if unavailable_origin and parent_file is not None:
        raise ValueError("an unavailable origin needs its own executed parent")
    journey = forked_child(
        root,
        parent_file=parent_file,
        investigation_request=None if parent_file else THREE_MEMBER_REQUEST,
    )
    if unavailable_origin:
        origin_root = root / "source"
        origin_bytes = bundle_bytes(origin_root)
        origin_root.rename(root / "unavailable-origin")
        assert not origin_root.exists()
    request_file = root / "investigation-request.json"
    request_file.write_text(
        json.dumps(THREE_MEMBER_REQUEST, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    pending = begin_in_pass(
        BeginInvestigationCommand(
            journey_file=str(journey.path),
            request_files=(str(request_file),),
        ),
        environ={"LOCAL_MODEL_KEY": "secret-value-not-retained"},
        model_factory=lambda loaded, selected: SequentialAdaptiveModel().model(
            loaded, selected
        ),
        source_adapter_factory=lambda: SequentialSourceResponses(
            assembly_body=ASSEMBLY_BODY
        ),
    )
    assert pending.pending_request is not None

    def approve_and_observe():
        approved = respond_investigation_review(
            RespondInvestigationReviewCommand(
                journey_file=str(journey.path),
                request_event_id=pending.pending_request.request_event_id,
                response=ApproveInvestigationResponse(
                    action="approve",
                    proposal_id=pending.pending_request.proposal.proposal_id,
                ),
            )
        )
        assert approved.authority is not None
        assert approved.investigation_input is not None
        if strict_policy:
            amend_investigation_policy(
                AmendInvestigationPolicyCommand(
                    journey_file=str(journey.path),
                    amendment=InvestigationPolicyAmendment(
                        proposal_id=pending.pending_request.proposal.proposal_id,
                        expected_head_event_id=Journey.open(journey.path).read()[-1][
                            "event_id"
                        ],
                        previous_policy_id=None,
                        asserted_by="researcher:owner",
                        reason="Retain strict completion through the forked collection.",
                        completion="require_terminal_members",
                    ),
                )
            )
        descriptor = VerifiedObject(
            role="VCF",
            bucket=approved.investigation_input.vcf.bucket,
            object_key=approved.investigation_input.vcf.object_key,
            byte_count=approved.investigation_input.vcf.byte_count,
            sha256=approved.investigation_input.vcf.sha256,
        )
        command = InvestigationObservationCommand(
            journey_file=str(journey.path),
            authority_id=approved.authority.authority_id,
            vcf_file=str(root / "variants.vcf"),
            materialization_receipt_file=str(root / "materialization.json"),
            bcftools_executable=str(root / "bcftools"),
            bcftools_probe_receipt_file=str(root / "probe.json"),
            derivation_root=str(root / "derived"),
        )
        recorded = investigation_observe(
            command,
            adapter=ObservationAdapter(
                command,
                descriptor,
                approved.investigation_input.sample_id,
            ),
            evidence_reader=ObservationEvidence(descriptor),
        )
        return recorded

    if coordinator_pass:
        # As in the coordinator, the approval answer and the collection it starts
        # share one processing Pass, whose closure becomes the physical head.
        owner = Journey.open(journey.path)
        with owner.processing(), processing_pass(owner) as interval:
            recorded = approve_and_observe()
            interval.close("completed")
    else:
        recorded = approve_and_observe()
    assert isinstance(recorded, InvestigationCollectionEvidenceRecorded)
    if unavailable_origin:
        assert bundle_bytes(root / "unavailable-origin") == origin_bytes
    return Journey.open(journey.path)


class SequentialObservationAdapter(ObservationAdapter):
    def run(self, argv, *, timeout_seconds, stdout_path=None):  # noqa: ANN001, ANN201
        if lookup_command(argv) != "variant_query":
            return super().run(
                argv, timeout_seconds=timeout_seconds, stdout_path=stdout_path
            )
        del timeout_seconds
        self.calls.append(tuple(argv))
        return CompletedProcess(
            0,
            query_row(
                "NC_000019.10",
                query_position(self.files, argv),
                "T",
                "C",
                self.sample_id,
                "0/1",
            ),
            b"",
        )


def offline_investigation_dependencies() -> dict:
    """Bindings for a Run that begins Investigation after its verified Execution.

    Since #342 a fresh Run's plan names its targets, so the call that verifies
    Execution also begins Investigation. These offline model and source adapters,
    with the model credential, let it reach its first Investigation Review.
    """
    return {
        "environ": {"LOCAL_MODEL_KEY": "secret"},
        "investigation_model_factory": _AdaptiveInvestigationModel().model,
        "investigation_source_adapter_factory": lambda: SourceResponsesOnly(
            assembly_body=ASSEMBLY_BODY
        ),
    }


@contextmanager
def members_one_at_a_time():
    """Plan collection members one at a time, each still in its own Attempt (#105).

    For scripted models that answer by global call order on a Run whose
    registration (a Fork's inherited one) selects no planner bound, so the
    named default would run members together.
    """
    from unittest.mock import patch

    from bio_neb.investigation.observation_defaults import (
        InvestigationConcurrency,
        selected_concurrency,
    )

    def one_at_a_time(runtime_selections):
        return InvestigationConcurrency(
            planner_members=1,
            table_chunks=selected_concurrency(runtime_selections).table_chunks,
        )

    with patch(
        "bio_neb.investigation.observation_defaults.selected_concurrency",
        one_at_a_time,
    ):
        yield


def amend_to_partial(journey: Journey, reason: str = "Report what has run now."):
    """The researcher's explicit partial choice: the only earlier eligibility (#105)."""
    from bio_neb.application.queries import read_investigation
    from bio_neb.investigation.investigation_policy import (
        InvestigationPolicyAmendment,
    )
    from bio_neb.investigation.investigation_policy_commands import (
        AmendInvestigationPolicyCommand,
        amend_investigation_policy,
    )

    read = read_investigation(Journey.open(journey.path))
    return amend_investigation_policy(
        AmendInvestigationPolicyCommand(
            journey_file=str(journey.path),
            amendment=InvestigationPolicyAmendment(
                # The policy's current proposal and head, as the command reads them.
                proposal_id=read.work_policy.proposal_id,
                expected_head_event_id=read.timeline[-1]["event_id"],
                previous_policy_id=read.work_policy.policy_id,
                asserted_by="researcher:owner",
                reason=reason,
                completion="allow_partial_with_limitations",
            ),
        )
    )


def observed_collection(
    root: Path,
    *,
    cardinality: int = 3,
    adapter_type=None,
    investigation_table: dict | None = None,
    investigation_concurrency: dict | None = None,
    forked: bool = False,
):
    """An approved collection observed by its lookups, before its table stage:
    the reopened Journey and the observation adapter its stage reuses. ``forked``
    observes it in a Fork child of an executed Run, which holds a verified BAM and VCF.
    """
    investigation_input, journey, _, descriptor, command = (
        approved_collection_observation(
            root,
            cardinality=cardinality,
            forked=forked,
            investigation_table=investigation_table,
            investigation_concurrency=investigation_concurrency,
        )
    )
    adapter = (adapter_type or ObservationAdapter)(
        command, descriptor, investigation_input.sample_id
    )
    investigation_observe(
        command, adapter=adapter, evidence_reader=ObservationEvidence(descriptor)
    )
    return Journey.open(journey.path), adapter


def run_taxon(tax_id: int):
    """The Run organism's species-level taxon, as planning would have recorded it,
    for a test whose Journey holds no planning facts (#479 Q4 A): patches the one
    reader the start and its reopen share."""
    from unittest.mock import patch

    return patch(
        "bio_neb.investigation.run_reference.run_species_taxid",
        lambda _events, _accession: tax_id,
    )
