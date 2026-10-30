"""The end-of-dev-cleanup Run's approved Execution input and path, from its registered fixtures.

Source: Run `266-full-20261013T020000Z-39475b5a-a1`, Journey
`3b17ac33-133c-47f2-8271-b95fbec21a71` (`docs/development/reviews/issue266-live/README.md`).
`issue266_planning` holds the reviewed proposal and its launch; `issue266_execution` holds the
verified input and reference pointers, the H100 objects and the contact journal binding. The
fixtures export path-event names only, so the rows below are rebuilt from those values.

Supplied from the Run's Journey, not exported: the approval event id, the Journey file, the path
event ids and the H100 pointers' journal locators. The H100 pointers are derived from the verified
objects; the Run also recorded a local manifest copy and its agent-composed result context.

`approved_input` derives the current proposal shape from the retained one (#299): the Run's
configuration typed the reference source and plain-FASTA identity, and approval bound its sha256.
The derivation drops those configuration fields (the local replica keeps its path), binds a supplied publisher md5
(`SUPPLIED_SOURCE_MD5`, synthetic: the Run never observed one) and recomputes the admission,
package and proposal identities. The InputsVerified row's acquisition receipt pairs that md5
with the Run's observed source pointer and prepared FASTA object.

`reference_shape` derives #471's copied reference shape (owner Q6 a, no backward compatibility):
the validated reference's fields become its copied NCBI record (`identity_status` is NCBI's status
word, `linked_accession` the pair; the derived `namespace` goes), and the decision basis drops the
retired choice id, uncertainties and coverage for the page's count and token (none: both Runs'
choices are a supplied accession, with no discovery). #472 adds the direct choice's quote, which
the Run never recorded: the shape quotes the accession itself. #464's package names the Run's
reference sequences and the ones a CPU build drops, which the Run never read: the shape names
none, and renames the admission's and package's derivation rule to #464's (its admission id
recomputed). It also drops the launch's two
retired prepared-candidate fields (`_RETIRED_LAUNCH_FIELDS`): a Run's index is the Execution
Review's bound prebuilt index or one CPU job (owner, 2026-10-29, #464, no backward compatibility).
The #266 Run admitted its account candidate; the derived launch names none, so its rebuilt plan
prepares on CPU, while its retained path still holds the hit it recorded.

`prepared_path_events` is a synthetic variation: the Run hit its prepared reference, so its
CPU-preparation branch (a miss, then `ReferencePrepared`) never ran. Nothing here grants live,
continuation or review authority.
"""

import copy

from bio_neb.journey import canonical_json_digest
from bio_neb.parabricks.execution_proposal import (
    ExecutionAuthority,
    ExecutionProposal,
    execution_authority_id,
)
from bio_neb.parabricks.plan import ApprovedParabricksInput
from bio_neb.reference_discovery.reference_identity_tool import AssemblySourceRecord
from bio_neb.reference_discovery.reference_source import (
    GZIP_REFERENCE_SEQUENCES,
    reference_source_admission_id,
)
from tests.retained_live_fixture import load_retained_live_fixture

APPROVAL_EVENT_ID = "6125cb7a-44e3-5754-81e8-00e69c6a745d"
JOURNEY_FILE = (
    "/workspaces/bio-neb-work-issue29-retry/.bio-neb-local/live-runs/"
    "266-full-20261013T020000Z-39475b5a-a1/runs/"
    "3b17ac33-133c-47f2-8271-b95fbec21a71/journey.jsonl"
)
# The Run's own path event ids, in path order.
EVENT_IDS = {
    "ExecutionBound": "1226c829-a374-46a3-b65b-a27d3298e44b",
    "InputsVerified": "5e5dba85-def1-4071-a8c8-001b438b38a3",
    "ReferenceResolved": "43dd24ee-7994-4e33-a088-0d218473cf30",
    "H100ArtifactsVerified": "597ee58d-e0fa-4efc-9c8e-87ca31216cfe",
    "ExecutionOutcome": "8e88175d-4eb4-4c64-a408-cd7847793b2f",
}
_H100_PROOF_REFS = {
    "aligned_bam": "cj:1:1609",
    "VCF": "cj:1:1609",
    "worker_receipt": "cj:1:1609",
    "execution_manifest": "cj:1:1621",
}
# Synthetic ids for the CPU-preparation variation only.
_PREPARED_IDS = {
    "ReferenceResolved": "reference-miss",
    "ReferencePrepared": "reference-prepared",
}


# The retired launch fields; every retained launch names them, with or without a value.
_RETIRED_LAUNCH_FIELDS = ("local_prepared_bundle", "remote_prepared_candidate")

# Supplied, not observed: the retained Run's approval bound a sha256 source identity.
SUPPLIED_SOURCE_MD5 = "5" * 32


def _copied_reference(reference: dict) -> dict:
    record = {
        "assembly_accession": reference["assembly_accession"],
        "assembly_name": reference["assembly_name"],
        "organism_name": reference["organism_name"],
        "ncbi_tax_id": reference["ncbi_tax_id"],
        "assembly_status": reference["identity_status"],
        "current_accession": reference["current_accession"],
        "paired_accession": reference["linked_accession"],
        "release_date": reference["release_date"],
    }
    return {
        "candidate_id": reference["candidate_id"],
        "assembly_accession": reference["assembly_accession"],
        # As the proposal's identity dumps it: every copied field, unset ones null.
        "record": AssemblySourceRecord.model_validate(
            {key: value for key, value in record.items() if value is not None}
        ).model_dump(mode="json"),
        "source": reference["source"],
        "evidence_id": reference["evidence_id"],
    }


def reference_shape(retained: dict) -> dict:
    """The retained proposal in #471's copied reference shape; see the module notes."""
    proposal = copy.deepcopy(retained)
    launch = proposal["configuration"]["launch"]
    for retired in _RETIRED_LAUNCH_FIELDS:
        del launch[retired]
    proposal["reference"] = _copied_reference(proposal["reference"])
    basis = proposal["reference_basis"]
    basis["choice"].pop("candidate_id")
    # A direct choice stores its quote; the Run recorded none, so it is the accession.
    basis["choice"]["quote"] = basis["choice"]["assembly_accession"]
    for retired in ("uncertainties", "source_coverage"):
        basis.pop(retired)
    basis.update(total_count=None, next_page_token=None)
    validation = basis["validation"]
    validation["references"] = [
        _copied_reference(item) for item in validation["references"]
    ]
    # The package records the Run's reference sequences at the genomic FASTA's
    # description; the Run read no assembly report, so its derived package names none
    # (it binds no prebuilt index to fit against them).
    package = proposal["execution_package"]
    package["reference"]["reference_sequences"] = []
    # The package names the sequences a CPU build drops, and the source derivation
    # rule says the gzip source keeps the Run's reference sequences: the Run read no
    # report, so it drops none, and its admission is renamed to that rule.
    package["reference"]["excluded_sequences"] = []
    admission = proposal["reference_source_admission"]
    admission["derivation_rule"] = GZIP_REFERENCE_SEQUENCES
    admission["admission_id"] = reference_source_admission_id(
        {k: v for k, v in admission.items() if k != "admission_id"}
    )
    package["reference"].update(
        derivation_rule=GZIP_REFERENCE_SEQUENCES,
        source_admission_id=admission["admission_id"],
    )
    package["package_id"] = "execution-package:" + canonical_json_digest(
        {k: v for k, v in package.items() if k != "package_id"}
    )
    proposal["proposal_id"] = "execution-proposal:" + canonical_json_digest(
        {k: v for k, v in proposal.items() if k != "proposal_id"}
    )
    return proposal


def _current_shape(retained: dict, excluded_sequences=()) -> dict:
    """Derive the reviewed proposal in the source-identity shape; see the module notes."""
    proposal = reference_shape(retained)
    configuration = proposal["configuration"]
    del configuration["reference_source"]
    launch = configuration["launch"]
    del launch["accepted_plain_fasta"]
    # The Run's local replica keeps its path; upload checks the approved md5.
    launch["local_reference_source"] = {
        "path": launch["local_reference_source"]["path"]
    }
    admission = proposal["reference_source_admission"]
    admission.update(
        expected_checksum_algorithm="md5", expected_checksum=SUPPLIED_SOURCE_MD5
    )
    admission["admission_id"] = reference_source_admission_id(
        {k: v for k, v in admission.items() if k != "admission_id"}
    )
    package = proposal["execution_package"]
    package["reference"].update(
        source_admission_id=admission["admission_id"],
        expected_checksum_algorithm="md5",
        expected_checksum=SUPPLIED_SOURCE_MD5,
        excluded_sequences=list(excluded_sequences),
    )
    package["package_id"] = "execution-package:" + canonical_json_digest(
        {k: v for k, v in package.items() if k != "package_id"}
    )
    proposal["proposal_id"] = "execution-proposal:" + canonical_json_digest(
        {k: v for k, v in proposal.items() if k != "proposal_id"}
    )
    return proposal


def approved_input(excluded_sequences=()) -> ApprovedParabricksInput:
    """The exact approved input the Run bound, rebuilt from its reviewed proposal.
    ``excluded_sequences`` is a synthetic variation: the package names sequences a CPU
    build drops, as a description that read NCBI's report would (#464)."""
    (scenario,) = load_retained_live_fixture("issue266_planning").scenarios
    proposal = ExecutionProposal.model_validate(
        _current_shape(scenario.execution_proposal, excluded_sequences)
    )
    request = scenario.execution_review_event_id
    return ApprovedParabricksInput(
        journey_file=JOURNEY_FILE,
        proposal=proposal,
        authority=ExecutionAuthority(
            authority_id=execution_authority_id(
                proposal.proposal_id, request, APPROVAL_EVENT_ID
            ),
            proposal_id=proposal.proposal_id,
            request_event_id=request,
            approval_event_id=APPROVAL_EVENT_ID,
        ),
        launch=proposal.configuration.launch,
    )


def _row(event_type, kind, payload, rests_on, *, grounding="verified", event_id=None):
    return {
        "event_id": event_id or EVENT_IDS[event_type],
        "event_type": event_type,
        "kind": kind,
        "grounding": grounding,
        "pass": 1,
        "payload": payload,
        "rests_on": list(rests_on),
    }


def path_events() -> list[dict]:
    """The Run's five Execution path rows: bound, inputs, reference hit, H100, outcome."""
    approved = approved_input()
    (scenario,) = load_retained_live_fixture("issue266_execution").scenarios
    authority = approved.authority.authority_id
    inputs = [item.model_dump(mode="json") for item in scenario.inputs]
    reference = [item.model_dump(mode="json") for item in scenario.reference]
    h100 = [
        {
            "role": item.role,
            "content_id": f"sha256:{item.sha256}",
            "location": f"s3://{item.bucket}/{item.object_key}",
            "bytes": item.byte_count,
            "producer": {"stage": "h100", "tool": None, "authority_id": authority},
            "grounding": "verified",
            "proof_ref": _H100_PROOF_REFS[item.role],
        }
        for item in scenario.verified_objects
    ]
    journal = scenario.contacts.journal_binding
    ids = EVENT_IDS
    return [
        _row(
            "ExecutionBound",
            "authority",
            {
                "execution_id": approved.execution_id,
                "authority_id": authority,
                "proposal_id": approved.proposal.proposal_id,
                "run_id": scenario.journey_id,
                "approved_input": approved.model_dump(mode="json"),
            },
            (),
            grounding="human",
        ),
        _row(
            "InputsVerified",
            "verified_artifact",
            {
                "state": "verified",
                "pointers": inputs,
                "receipt_id": inputs[-1]["content_id"],
                "reference_acquisition": _acquisition(inputs, reference),
            },
            (ids["ExecutionBound"],),
        ),
        _row(
            "ReferenceResolved",
            "decision",
            {
                "reference": {
                    "state": "hit",
                    "pointers": reference,
                    "receipt_id": reference[-1]["content_id"],
                }
            },
            (ids["InputsVerified"],),
        ),
        _row(
            "H100ArtifactsVerified",
            "verified_artifact",
            {
                "state": "verified",
                "pointers": h100,
                "receipt_id": h100[-1]["content_id"],
            },
            (ids["ExecutionBound"], ids["InputsVerified"], ids["ReferenceResolved"]),
        ),
        _row(
            "ExecutionOutcome",
            "outcome",
            {
                "disposition": "succeeded",
                "contact_journal": journal.locator,
                "journal_bytes": journal.byte_count,
                "journal_sha256": journal.sha256,
                "milestone": None,
            },
            (ids["H100ArtifactsVerified"],),
        ),
    ]


def _acquisition(inputs, reference):
    """The Run's observed source and prepared FASTA, with the supplied md5."""
    (source,) = (p for p in inputs if p["role"] == "reference_source")
    (fasta,) = (p for p in reference if p["role"] == "genomic_fasta")
    return {
        "source_md5": SUPPLIED_SOURCE_MD5,
        "source_byte_count": source["bytes"],
        "source_sha256": source["content_id"].removeprefix("sha256:"),
        "plain_fasta_byte_count": fasta["bytes"],
        "plain_fasta_sha256": fasta["content_id"].removeprefix("sha256:"),
    }


def stored_objects(pointers):
    """The object-store pointers; local bundle documents have no exported bytes."""
    return [pointer for pointer in pointers if pointer["location"].startswith("s3://")]


def prepared_path_events() -> list[dict]:
    """Synthetic six-row variation: a reference miss, then CPU preparation of the same index.

    Rows: bound, inputs, miss, prepared, H100, outcome. The prepared pointers are the Run's
    stored reference objects relabelled as CPU-preparation output. Pointers to local bundle
    documents are left out, since their bytes are not exported, so a writer can record every row.
    """
    bound, inputs, hit, h100, outcome = path_events()
    inputs = {
        **inputs,
        "payload": {
            **inputs["payload"],
            "pointers": stored_objects(inputs["payload"]["pointers"]),
        },
    }
    prepared = [
        {**pointer, "producer": {**pointer["producer"], "stage": "cpu_preparation"}}
        for pointer in stored_objects(hit["payload"]["reference"]["pointers"])
    ]
    miss = _row(
        "ReferenceResolved",
        "decision",
        {"reference": {"state": "miss", "reason": "candidate_absent"}},
        (bound["event_id"], inputs["event_id"]),
        event_id=_PREPARED_IDS["ReferenceResolved"],
    )
    reference = _row(
        "ReferencePrepared",
        "verified_artifact",
        {
            "state": "prepared",
            "pointers": prepared,
            "receipt_id": hit["payload"]["reference"]["receipt_id"],
            # Input storage streamed this Run's source, so it holds the acquisition.
            "reference_acquisition": None,
        },
        (inputs["event_id"], miss["event_id"]),
        event_id=_PREPARED_IDS["ReferencePrepared"],
    )
    h100 = {**h100, "rests_on": [inputs["event_id"], reference["event_id"]]}
    return [bound, inputs, miss, reference, h100, outcome]
