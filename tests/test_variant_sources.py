"""The variant-source spike asks dbSNP or EVA for an assembly-tied variant claim, the
source its caller names (#469: no code routes by assembly)."""

import contextlib
import copy
import hashlib
import inspect
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from typing import get_args
from unittest import mock

from pydantic import ValidationError

from bio_neb.failure import (
    FailureOverrides,
    FailureSurface,
    FailureTuning,
    run_failure_tuning,
)
from bio_neb.investigation.investigation import (
    EvidenceAssessment,
    RefSnpStatus,
    RefSnpToolResult,
    ResolvedVariantCandidate,
    RetrievedSource,
)
from bio_neb.investigation.investigation_request import RequestedVariantIdentity
from bio_neb.investigation.investigation_sources import normalize_refsnp
from tests.eva_retained_bodies import EVA_CONTIG_ALIAS as EVA_ALIAS
from tests.eva_retained_bodies import (
    PIG_CM000823_5_ALIAS,
    PIG_RS5334475172,
    PIG_RS5334475172_SUBMITTED,
)
from tests.location_retained_bodies import (
    EVA_AT_PIG_DEL,
    GRCH38_MOLECULES,
    GRCH38_REPORT,
    SPAN_NC_000019_10_T,
    SPDI_RS429358,
    SSCROFA11_MOLECULES,
)
from tests.location_retained_bodies import SSCROFA11_REPORT as LOCATION_SSCROFA11_REPORT
from tests.refsnp_retained_bodies import (
    RS429358,
    RS4149056,
    RS4149056_DOCUMENT,
    RS60037639,
    RS61228756,
    refsnp_body,
)
from tests.sequence_naming_support import names_answers
from tools.variant_sources import ClaimAnswer, VariantClaim, VariantSources
from tools.variant_sources import __main__ as variant_sources_main


class _Answer:
    """One scripted HTTP answer from the fake opener."""

    def __init__(self, status, body=b"", headers=None):
        self.status = status
        self.body = body
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    """Answers by URL; a status of 300 or more is raised as `HTTPError`, as urllib does.

    A list of answers is given out in order, one per request (a retried contact)."""

    def __init__(self, answers):
        self.answers = {
            url: list(answer) if isinstance(answer, list) else answer
            for url, answer in answers.items()
        }
        self.sent = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.sent.append(url)
        answer = self.answers[url]
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if answer.status >= 300:
            raise urllib.error.HTTPError(
                url, answer.status, "scripted", answer.headers, io.BytesIO(answer.body)
            )
        return answer


EVA_RS = "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants/5"


class ProductTransportTests(unittest.TestCase):
    """`look`'s GETs leave through the product's source transports (#417): the transport
    family's own mechanics are `tests/test_source_transport.py`'s; this pins that the
    tool's default path reaches them."""

    def test_a_redirect_is_not_followed_and_is_read_as_the_claims_answer(self):
        sent = []

        def network(handler, request):
            """Below urllib's redirect handling: every HTTPS request answers 302."""
            sent.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://elsewhere.org/"
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=302
            )
            response.msg = "Found"
            return response

        sources = VariantSources.public()
        with mock.patch.object(urllib.request.HTTPSHandler, "https_open", network):
            answer = sources.answer(
                VariantClaim(
                    label="v", assembly_accession="GCF_000003025.6", rsid="rs5"
                ),
                "eva",
            )

        # One GET, never followed; its 302 is EVA's answer, read with its status
        # named, with no verdict on it.
        self.assertEqual(sent, [EVA_RS])
        self.assertEqual(sources.contact_count, 1)
        self.assertIsNone(answer.failure)
        self.assertEqual(answer.eva.status, "source_answered")
        self.assertIn("HTTP 302", answer.eva.conflicts[0])


# --- Recorded and synthetic source answers -------------------------------------------------
#
# Recorded: the run plan's step 1 probe (contact log of
# `.bio-neb-local/variant-sources/`), copied byte for byte.
EVA = "https://www.ebi.ac.uk/eva/webservices/identifiers/v1/clustered-variants"
PIG_RS_URL = f"{EVA}/5334475172"
PIG_RS = (
    b'[{"accession":5334475172,"version":1,"data":{"assemblyAccession":"GCA_000003025.6",'
    b'"taxonomyAccession":9823,"contig":"CM000823.5","start":3556402,"type":"DEL",'
    b'"validated":false,"mapWeight":null,"createdDate":"2023-03-24T15:21:19.475"}}]'
)
PIG_SUBMITTED = (
    b'[{"accession":7881321083,"version":1,"data":{"referenceSequenceAccession":'
    b'"GCA_000003025.6","taxonomyAccession":9823,"projectAccession":"PRJEB49109",'
    b'"contig":"CM000823.5","start":3556402,"referenceAllele":"GTGTCCCCCTCGC",'
    b'"alternateAllele":"","clusteredVariantAccession":5334475172,'
    b'"supportedByEvidence":true,"assemblyMatch":true,"allelesMatch":true,'
    b'"validated":false,"mapWeight":null,"createdDate":"2022-01-25T17:27:51.445",'
    b'"remappedFrom":null,"remappedDate":null,"remappingId":null,'
    b'"backPropagatedVariantAccession":null}}]'
)
PIG_AT = f"{EVA}?assemblyId=GCA_000003025.6&referenceName=CM000823.5&start=3556402&variantType=DEL"

# Synthetic, in the shape of the product's own fixture
# (`tests/sequential_investigation_support.py` `REFSNP_BODY`).
DBSNP_URL = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/429358"
DBSNP_BODY = json.dumps(
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
                                    "inserted_sequence": base,
                                }
                            }
                        }
                        for base in ("T", "C")
                    ],
                }
            ]
        },
    }
).encode()


NO_BACKOFF = FailureTuning(operation=FailureOverrides(backoff_seconds=0))
"""The failure section a Run may record: retries without waiting, so tests do not sleep."""


def _source_of(assembly: str) -> str:
    """The source these tests' developer names (#469: the caller chooses, no code routes):
    dbSNP for a human assembly (GRCh37/GRCh38, T2T-CHM13), EVA for any other."""
    return "dbsnp" if assembly[4:13] in ("000001405", "009914755") else "eva"


class _Sources:
    """`VariantSources.public` over the fake opener: the product's NCBI and EVA
    transports, with only their opener replaced."""

    def __init__(self, test, answers):
        test.enterContext(run_failure_tuning(NO_BACKOFF))
        self.opener = _Opener(answers)
        self.sources = VariantSources.public(self.opener)

    def answer(self, source=None, **claim):
        return self.sources.answer(
            VariantClaim(label="v", **claim),
            source or _source_of(claim["assembly_accession"]),
        )


class RoutingTests(unittest.TestCase):
    def test_the_named_source_is_asked_whatever_the_assembly(self):
        """No code route (#469, owner RC2 1, ADR 0049): a human-assembly claim the
        developer sends to EVA asks EVA, and EVA's no record is EVA's answer."""
        tool = _Sources(self, {f"{EVA}/429358": _Answer(404, b"")})

        answer = tool.answer(
            source="eva", assembly_accession="GCF_000001405.40", rsid="rs429358"
        )

        self.assertEqual(answer.route, "eva")
        self.assertEqual(answer.route_reason, "the caller named EVA")
        self.assertEqual(tool.opener.sent, [f"{EVA}/429358"])
        self.assertEqual(answer.eva.lookup_source, "eva")

    def test_a_human_rs_claim_goes_to_dbsnp_and_keeps_normalize_refsnps_result(self):
        tool = _Sources(self, {DBSNP_URL: _Answer(200, DBSNP_BODY)})

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs429358")

        self.assertEqual(answer.route, "dbsnp")
        self.assertEqual(tool.opener.sent, [DBSNP_URL])
        self.assertEqual(tool.sources.contact_count, 1)
        # The GET's time and duration are the transport's observation; every other
        # field of the source is stated here.
        source = RetrievedSource.create(
            operation="fetch_refsnp",
            request_identity=DBSNP_URL,
            status_code=200,
            observed_at=answer.dbsnp.source.observed_at,
            elapsed_time_ms=answer.dbsnp.source.elapsed_time_ms,
            raw_response=DBSNP_BODY,
        )
        expected = normalize_refsnp(
            rsid="rs429358",
            assembly_accession="GCF_000001405.40",
            source=source,
            raw_response=DBSNP_BODY,
        )
        self.assertEqual(answer.dbsnp, expected)
        [candidate] = answer.dbsnp.candidates
        self.assertEqual(
            (answer.dbsnp.status, candidate.contig, candidate.one_based_position),
            ("resolved", "NC_000019.10", 44908684),
        )
        self.assertIsNone(answer.eva)

    def test_a_dbsnp_404_is_dbsnps_answer(self):
        """No status has a fixed meaning (ADR 0049; owner #468 R1 b)."""
        tool = _Sources(self, {DBSNP_URL: _Answer(404, b'{"error": "no such rs"}')})

        answer = tool.answer(assembly_accession="GCF_000001405.25", rsid="rs429358")

        self.assertEqual(answer.dbsnp.status, "source_answered")
        self.assertEqual(answer.dbsnp.source.status_code, 404)

    def test_a_dbsnp_503_then_an_answer_is_retried_and_answered(self):
        tool = _Sources(
            self, {DBSNP_URL: [_Answer(503, b"busy"), _Answer(200, DBSNP_BODY)]}
        )

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs429358")

        self.assertEqual(tool.opener.sent, [DBSNP_URL, DBSNP_URL])
        self.assertEqual(answer.dbsnp.status, "resolved")
        self.assertEqual(tool.sources.contact_count, 2)

    def test_repeated_dbsnp_503s_exhaust_the_claim_which_asks(self):
        tool = _Sources(self, {DBSNP_URL: [_Answer(503, b"busy")] * 3})

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs429358")

        self.assertEqual(tool.opener.sent, [DBSNP_URL] * 3)
        self.assertIsNone(answer.dbsnp)
        failure = answer.failure
        self.assertEqual(
            (
                failure.tool,
                failure.failure_class,
                failure.original_failure_class,
                failure.contact_attempts,
                failure.route,
            ),
            ("fetch_refsnp", "exhausted", "transient", 3, "ask"),
        )
        self.assertEqual(
            failure.policy.surface, FailureSurface(scope="tool", name="fetch_refsnp")
        )
        self.assertEqual(failure.policy.effective.retry_attempts, 3)
        self.assertRegex(failure.detail, "HTTP 503")

    def test_a_lost_connection_is_transient_and_retried_for_either_source(self):
        dbsnp = _Sources(
            self, {DBSNP_URL: [TimeoutError("timed out"), _Answer(200, DBSNP_BODY)]}
        )
        eva = _Sources(self, {f"{EVA}/1": [ConnectionResetError("reset")] * 3})

        human = dbsnp.answer(assembly_accession="GCF_000001405.40", rsid="rs429358")
        pig = eva.answer(assembly_accession="GCF_000003025.6", rsid="rs1")

        self.assertEqual(human.dbsnp.status, "resolved")
        self.assertEqual(dbsnp.opener.sent, [DBSNP_URL] * 2)
        self.assertEqual(
            (
                pig.failure.failure_class,
                pig.failure.original_failure_class,
                pig.failure.contact_attempts,
            ),
            ("exhausted", "transient", 3),
        )
        self.assertEqual(eva.opener.sent, [f"{EVA}/1"] * 3)
        self.assertEqual(eva.sources.contact_count, 3)

    def test_a_dbsnp_400_is_read_as_dbsnps_answer_at_once(self):
        # A client status is the source's answer about the request: read, with its
        # status named, never retried and never a failure that asks.
        tool = _Sources(self, {DBSNP_URL: [_Answer(400, b"bad request")] * 3})

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs429358")

        self.assertEqual(tool.opener.sent, [DBSNP_URL])
        self.assertIsNone(answer.failure)
        self.assertEqual(
            (answer.dbsnp.status, answer.dbsnp.source.status_code),
            ("source_answered", 400),
        )
        self.assertIn("HTTP 400", answer.dbsnp.conflicts[0])

    def test_an_hgvs_only_claim_is_not_checked(self):
        tool = _Sources(self, {})

        answer = tool.answer(
            assembly_accession="GCF_000002285.3", hgvs=("c.1052T>C", "p.(Leu351Pro)")
        )

        self.assertEqual(answer.route, "eva")
        self.assertRegex(answer.not_checked, "HGVS")
        self.assertEqual(tool.opener.sent, [])


REFSNP = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp"


class DbsnpAnswerTests(unittest.TestCase):
    """dbSNP answers as #343's `normalize_refsnp` gives them, on retained dbSNP bodies
    (`tests/refsnp_retained_bodies.py` states each body's provenance)."""

    def test_a_merged_rsid_is_followed_to_its_current_rsid_keeping_both(self):
        tool = _Sources(
            self,
            {
                f"{REFSNP}/60037639": _Answer(200, RS60037639),
                f"{REFSNP}/4149056": _Answer(200, RS4149056),
            },
        )

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs60037639")

        self.assertEqual(tool.opener.sent, [f"{REFSNP}/60037639", f"{REFSNP}/4149056"])
        self.assertEqual(
            (answer.merge.status, answer.merge.merged_into),
            ("merged", ("rs4149056",)),
        )
        current = answer.dbsnp
        self.assertEqual(
            (current.status, current.requested_rsid, current.verified_rsid),
            ("resolved", "rs60037639", "rs4149056"),
        )
        self.assertEqual(current.merged_rsids, ("rs52816141", "rs60037639"))
        self.assertEqual(
            [
                (item.contig, item.one_based_position, item.ref, item.alt)
                for item in current.candidates
            ],
            [
                ("NC_000012.12", 21178615, "T", "A"),
                ("NC_000012.12", 21178615, "T", "C"),
            ],
        )

    def test_each_snv_alt_of_the_rsid_is_its_own_candidate(self):
        tool = _Sources(self, {f"{REFSNP}/4149056": _Answer(200, RS4149056)})

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs4149056")

        self.assertEqual(answer.dbsnp.status, "resolved")
        self.assertEqual(
            [
                (item.contig, item.one_based_position, item.ref, item.alt, item.aliases)
                for item in answer.dbsnp.candidates
            ],
            [
                ("NC_000012.12", 21178615, "T", "A", ("rs4149056",)),
                ("NC_000012.12", 21178615, "T", "C", ("rs4149056",)),
            ],
        )

    def test_a_claim_naming_one_exact_allele_keeps_that_candidate(self):
        # A claim with a position takes the product's position lookup: its listing names
        # the claim's rsID, whose record binds the stated allele. The span and the
        # listing are synthetic, in the retained bodies' shape.
        tool = _Sources(
            self,
            _ncbi_location(
                "GCF_000001405.40",
                GRCH38_REPORT,
                GRCH38_MOLECULES,
                **{
                    _span("NC_000012.12", 21178614, 1): _Answer(
                        200,
                        b'{"data":{"seq_id":"NC_000012.12","position":21178614,'
                        b'"deleted_sequence":"T","inserted_sequence":"N"}}',
                    ),
                    f"{NCBI}/variation/v0/spdi/NC_000012.12:21178614:T:C/rsids": _Answer(
                        200, b'{"data":{"rsids":[4149056]}}'
                    ),
                    f"{REFSNP}/4149056": _Answer(200, RS4149056),
                },
            ),
        )

        answer = tool.answer(
            assembly_accession="GCF_000001405.40",
            rsid="rs4149056",
            contig="NC_000012.12",
            position=21178615,
            ref="T",
            alt="C",
        )

        self.assertEqual(answer.position.status, "resolved")
        self.assertEqual(
            [(item.ref, item.alt) for item in answer.position.candidates], [("T", "C")]
        )

    def test_the_indel_allele_of_an_rsid_beside_an_snv_is_a_candidate(self):
        # The derivation in `tests/test_investigation_sources.py`: rs4149056's T>A
        # allele replaced by the insertion T>TT; T>C stays as observed. Both are
        # candidates in dbSNP's SPDI form.
        mixed = copy.deepcopy(RS4149056_DOCUMENT)
        placement = mixed["primary_snapshot_data"]["placements_with_allele"][0]
        placement["alleles"][1]["allele"]["spdi"]["inserted_sequence"] = "TT"
        tool = _Sources(self, {f"{REFSNP}/4149056": _Answer(200, refsnp_body(mixed))})

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs4149056")

        self.assertEqual(answer.dbsnp.status, "resolved")
        self.assertEqual(
            [
                (item.contig, item.one_based_position, item.ref, item.alt)
                for item in answer.dbsnp.candidates
            ],
            [
                ("NC_000012.12", 21178615, "T", "C"),
                ("NC_000012.12", 21178615, "T", "TT"),
            ],
        )

    def test_a_failed_follow_keeps_the_merge_beside_the_failure(self):
        tool = _Sources(
            self,
            {
                f"{REFSNP}/60037639": _Answer(200, RS60037639),
                f"{REFSNP}/4149056": [_Answer(503, b"busy")] * 3,
            },
        )

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs60037639")

        self.assertEqual(
            tool.opener.sent, [f"{REFSNP}/60037639"] + [f"{REFSNP}/4149056"] * 3
        )
        self.assertEqual(answer.merge.merged_into, ("rs4149056",))
        self.assertIsNone(answer.dbsnp)
        self.assertEqual(
            (
                answer.failure.tool,
                answer.failure.failure_class,
                answer.failure.contact_attempts,
                answer.failure.route,
            ),
            ("fetch_refsnp", "exhausted", 3, "ask"),
        )

    def test_a_follow_that_answers_no_record_is_read_beside_the_merge(self):
        # Bytes that are not JSON are dbSNP's answer, typed malformed by the
        # product reader, no longer a semantic failure that asks.
        tool = _Sources(
            self,
            {
                f"{REFSNP}/60037639": _Answer(200, RS60037639),
                f"{REFSNP}/4149056": _Answer(200, b"not json"),
            },
        )

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs60037639")

        self.assertEqual(tool.opener.sent, [f"{REFSNP}/60037639", f"{REFSNP}/4149056"])
        self.assertEqual(answer.merge.merged_into, ("rs4149056",))
        self.assertIsNone(answer.failure)
        self.assertEqual(answer.dbsnp.status, "malformed")

    def test_a_recorded_merge_is_bound_to_the_claim_and_its_follow_to_the_query(self):
        tool = _Sources(
            self,
            {
                f"{REFSNP}/60037639": _Answer(200, RS60037639),
                f"{REFSNP}/4149056": _Answer(200, RS4149056),
                f"{REFSNP}/61228756": _Answer(200, RS61228756),
                DBSNP_URL: [_Answer(200, DBSNP_BODY)] * 2,
            },
        )
        followed = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs60037639")
        other = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs429358")
        another_merge = tool.answer(
            assembly_accession="GCF_000001405.40", rsid="rs61228756"
        ).merge
        ClaimAnswer(**dict(followed))
        # The merge's and the follow's own lookups, normalized on GRCh37 instead of the
        # claim's GRCh38.
        grch37_merge = normalize_refsnp(
            rsid="rs60037639",
            assembly_accession="GCF_000001405.25",
            source=followed.merge.source,
            raw_response=RS60037639,
        )
        grch37_follow = normalize_refsnp(
            rsid="rs4149056",
            requested_rsid="rs60037639",
            assembly_accession="GCF_000001405.25",
            source=followed.dbsnp.source,
            raw_response=RS4149056,
        )
        # Lookups of rs429358 that kept rs60037639 as requested: answers the merge (into
        # rs4149056) never asked for, though their requested rsID fits.
        unasked = {
            status: normalize_refsnp(
                rsid="rs429358",
                requested_rsid="rs60037639",
                assembly_accession="GCF_000001405.40",
                source=RetrievedSource.create(
                    operation="fetch_refsnp",
                    request_identity=DBSNP_URL,
                    status_code=code,
                    observed_at="2026-10-15T19:00:00Z",
                    elapsed_time_ms=1,
                    raw_response=body,
                ),
                raw_response=body,
            )
            for status, code, body in (
                ("resolved", 200, DBSNP_BODY),
                ("source_answered", 404, b'{"error": "no such rs"}'),
                ("conflicted", 200, RS4149056),
            )
        }
        self.assertEqual(
            [result.status for result in unasked.values()],
            ["resolved", "source_answered", "conflicted"],
        )

        # The follow's own result retyped onto an EVA request whose path ends in the same
        # number: the product result itself refuses a dbSNP result on an EVA source, so
        # no such follow can reach a claim's answer.
        eva_source = RetrievedSource.create(
            operation="eva_clustered_variant",
            request_identity=f"{EVA}/4149056",
            status_code=200,
            observed_at="2026-10-15T19:00:00Z",
            elapsed_time_ms=1,
            raw_response=RS4149056,
        )
        with self.assertRaisesRegex(
            ValidationError, "a dbSNP result reads fetch_refsnp"
        ):
            RefSnpToolResult.create(
                **followed.dbsnp.model_dump(
                    exclude={"result_id", "source", "assessment"}
                ),
                source=eva_source,
                assessment=EvidenceAssessment.create(
                    source_id=eva_source.source_id,
                    verdict=followed.dbsnp.assessment.verdict,
                    rationale=followed.dbsnp.assessment.rationale,
                ),
            )

        merge_refused = (
            "a merge is the claim's dbSNP record merged into one current rsID"
        )
        follow_refused = (
            "a followed lookup asks for the merge's current rsID for the claim's rsID"
        )
        for case, (answer, reason) in {
            "another claim's lookup": ({"dbsnp": other.dbsnp}, follow_refused),
            "a resolved lookup of another rsID": (
                {"dbsnp": unasked["resolved"]},
                follow_refused,
            ),
            "a 404 for another rsID": (
                {"dbsnp": unasked["source_answered"]},
                follow_refused,
            ),
            "a follow on another assembly": ({"dbsnp": grch37_follow}, follow_refused),
            # rs4149056's own record, but returned to a lookup of rs429358.
            "a conflicted lookup of another rsID": (
                {"dbsnp": unasked["conflicted"]},
                follow_refused,
            ),
            "a record that is not merged": (
                {"merge": other.dbsnp, "dbsnp": None, "not_checked": "x"},
                merge_refused,
            ),
            "another rsID's merge": (
                {"merge": another_merge, "dbsnp": None, "not_checked": "x"},
                merge_refused,
            ),
            "a merge on another assembly": (
                {"merge": grch37_merge, "dbsnp": None, "not_checked": "x"},
                merge_refused,
            ),
            "an EVA claim": (
                {"route": "eva", "dbsnp": None, "not_checked": "x"},
                merge_refused,
            ),
        }.items():
            with self.subTest(case=case):
                with self.assertRaises(ValidationError) as refused:
                    ClaimAnswer(**{**dict(followed), **answer})

                self.assertIn(reason, str(refused.exception))

    def test_a_followed_lookup_keeps_its_typed_negative_outcome(self):
        cases = {
            "source_answered": (_Answer(404, b'{"error": "no such rs"}'), None),
            # Synthetic: a current-rsID record whose merge record names no rsID.
            "malformed": (
                _Answer(
                    200,
                    b'{"refsnp_id":"4149056","merged_snapshot_data":{"merged_into":[]}}',
                ),
                "rs4149056",
            ),
            # The current record's body answers for another rsID.
            "conflicted": (_Answer(200, DBSNP_BODY), "rs429358"),
        }
        for status, (followed, verified) in cases.items():
            with self.subTest(status=status):
                tool = _Sources(
                    self,
                    {
                        f"{REFSNP}/60037639": _Answer(200, RS60037639),
                        f"{REFSNP}/4149056": followed,
                    },
                )

                answer = tool.answer(
                    assembly_accession="GCF_000001405.40", rsid="rs60037639"
                )

                self.assertEqual(answer.merge.merged_into, ("rs4149056",))
                self.assertEqual(
                    (
                        answer.dbsnp.status,
                        answer.dbsnp.requested_rsid,
                        answer.dbsnp.verified_rsid,
                    ),
                    (status, "rs60037639", verified),
                )

    def test_a_merge_into_several_rsids_is_the_answer_and_is_not_followed(self):
        # Synthetic: RS60037639's retained merge record with a second current rsID added.
        body = RS60037639.replace(
            b'"merged_into":["4149056"]', b'"merged_into":["4149056","429358"]'
        )
        self.assertNotEqual(body, RS60037639)
        tool = _Sources(self, {f"{REFSNP}/60037639": _Answer(200, body)})

        answer = tool.answer(assembly_accession="GCF_000001405.40", rsid="rs60037639")

        self.assertEqual(tool.opener.sent, [f"{REFSNP}/60037639"])
        self.assertEqual(
            (answer.dbsnp.status, answer.dbsnp.merged_into),
            ("merged", ("rs4149056", "rs429358")),
        )
        self.assertIsNone(answer.merge)


def _alias(contig, gca, gcf, refseq, name, taxid=9823):
    """EVA's contig-alias answer for one INSDC sequence (synthetic, in the recorded shape
    of `tests/eva_retained_bodies.py`)."""
    return json.dumps(
        {
            "_embedded": {
                "chromosomeEntities": [
                    {
                        "insdcAccession": contig,
                        "refseq": refseq,
                        "assembly": {
                            "insdcAccession": gca,
                            "refseq": gcf,
                            "name": name,
                            "taxid": taxid,
                        },
                    }
                ]
            }
        }
    ).encode()


PIG_ALIAS_URL = f"{EVA_ALIAS}/chromosomes/genbank/CM000823.5"


PIG_REPORT_URL = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_000003025.6"
    "/dataset_report"
)
NEWER_PIG_REPORT_URL = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_054392235.1"
    "/dataset_report"
)
NEWER_PIG_REPORT = (
    b'{"reports":[{"accession":"GCF_054392235.1","paired_accession":"GCA_054392235.1",'
    b'"assembly_info":{"assembly_name":"Sscrofa_newer","assembly_status":"current"}}]}'
)
"""Synthetic: NCBI's report of a newer pig assembly, in its recorded shape."""


def _pig_rs(**more):
    # The Run's INSDC twin comes from NCBI's report of the Run's assembly: the
    # recorded Sscrofa11.1 report, and a synthetic newer one.
    return {
        PIG_RS_URL: _Answer(200, PIG_RS),
        PIG_REPORT_URL: _Answer(200, LOCATION_SSCROFA11_REPORT),
        NEWER_PIG_REPORT_URL: _Answer(200, NEWER_PIG_REPORT),
        PIG_ALIAS_URL: _Answer(200, PIG_CM000823_5_ALIAS),
        f"{PIG_RS_URL}/submitted": _Answer(200, PIG_SUBMITTED),
        **more,
    }


class EvaRsTests(unittest.TestCase):
    """An rs claim is looked up by the product's EVA lookup (#366); these pin that the tool
    reaches it through the product's transports and the failure module."""

    def test_the_pig_rs_id_resolves_on_sscrofa11_with_its_alleles(self):
        tool = _Sources(self, _pig_rs())

        answer = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs5334475172")

        self.assertEqual(answer.route, "eva")
        result = answer.eva
        self.assertEqual((result.status, result.lookup_source), ("resolved", "eva"))
        self.assertEqual(result.eva.run_insdc_assembly_accession, "GCA_000003025.6")
        [placement] = result.placements
        self.assertEqual(
            (
                placement.eva.assembly_accession,
                placement.sequence,
                placement.eva.start,
                placement.eva.type,
                placement.on_run_assembly,
                placement.run_contig,
            ),
            ("GCA_000003025.6", "CM000823.5", 3556402, "DEL", True, "NC_010454.4"),
        )
        [alleles] = placement.eva.submitted
        self.assertEqual(
            (alleles.ss, alleles.reference, alleles.alternate),
            ("ss7881321083", "GTGTCCCCCTCGC", ""),
        )
        self.assertEqual(
            [(c.contig, c.one_based_position, c.ref, c.alt) for c in result.candidates],
            [("NC_010454.4", 3556402, "GTGTCCCCCTCGC", "")],
        )
        self.assertEqual(
            [
                source.raw_response_sha256
                for source in (result.source, *result.eva.supporting_sources)
            ],
            [
                hashlib.sha256(PIG_RS).hexdigest(),
                hashlib.sha256(LOCATION_SSCROFA11_REPORT).hexdigest(),
                hashlib.sha256(PIG_CM000823_5_ALIAS).hexdigest(),
                hashlib.sha256(PIG_SUBMITTED).hexdigest(),
            ],
        )
        self.assertFalse(any("ftp.ncbi" in url for url in tool.opener.sent))

    def test_the_pig_rs_id_has_no_placement_on_the_newer_assembly(self):
        tool = _Sources(self, _pig_rs())

        answer = tool.answer(assembly_accession="GCF_054392235.1", rsid="rs5334475172")

        result = answer.eva
        self.assertEqual(result.status, "incompatible")
        self.assertEqual(result.candidates, ())
        [placement] = result.placements
        self.assertEqual(
            (
                placement.eva.assembly_accession,
                placement.on_run_assembly,
                placement.run_contig,
            ),
            ("GCA_000003025.6", False, None),
        )
        self.assertNotIn(f"{PIG_RS_URL}/submitted", tool.opener.sent)

    def test_an_eva_404_is_evas_answer_with_eva_named(self):
        """No status has a fixed meaning (ADR 0049; owner #468 R1 b)."""
        tool = _Sources(self, {f"{EVA}/1": _Answer(404, b"")})

        answer = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs1")

        self.assertEqual(answer.eva.status, "source_answered")
        source = answer.eva.source
        self.assertIsInstance(source, RetrievedSource)
        self.assertEqual(
            (source.authority, source.operation, source.status_code),
            ("eva", "eva_clustered_variant", 404),
        )
        self.assertEqual(source.raw_response_sha256, hashlib.sha256(b"").hexdigest())
        self.assertRegex(answer.eva.assessment.rationale, "EVA")

    def test_an_eva_answer_that_is_not_a_list_is_malformed(self):
        tool = _Sources(self, {f"{EVA}/1": _Answer(200, b'{"accession": 1}')})

        answer = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs1")

        self.assertEqual(answer.eva.status, "malformed")

    def test_an_eva_503_then_an_answer_is_retried_and_answered(self):
        tool = _Sources(
            self, _pig_rs(**{PIG_RS_URL: [_Answer(503, b"busy"), _Answer(200, PIG_RS)]})
        )

        answer = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs5334475172")

        self.assertEqual(answer.eva.status, "resolved")
        self.assertEqual(tool.opener.sent.count(PIG_RS_URL), 2)

    def test_repeated_eva_500s_exhaust_the_claim_on_evas_own_surface(self):
        tool = _Sources(self, {f"{EVA}/1": [_Answer(500, b"")] * 3})

        answer = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs1")

        self.assertIsNone(answer.eva)
        self.assertEqual(
            (
                answer.failure.tool,
                answer.failure.failure_class,
                answer.failure.contact_attempts,
                answer.failure.route,
                answer.failure.policy.surface,
            ),
            (
                "eva_clustered_variant",
                "exhausted",
                3,
                "ask",
                FailureSurface(scope="tool", name="eva_clustered_variant"),
            ),
        )
        self.assertEqual(tool.opener.sent.count(f"{EVA}/1"), 3)

    def test_an_eva_4xx_is_read_and_an_access_fault_asks_at_once(self):
        # EVA's 400 for a contig's aliases is its answer: the lookup ends on it, with
        # its status named, for the model to read (ADR 0049). A 403 is no answer about
        # the record: it asks at once.
        tool = _Sources(self, _pig_rs(**{PIG_ALIAS_URL: _Answer(400, b"")}))

        read = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs5334475172")

        self.assertIsNone(read.failure)
        self.assertEqual(read.eva.status, "source_answered")
        self.assertIn("HTTP 400", read.eva.conflicts[0])
        self.assertEqual(tool.opener.sent.count(PIG_ALIAS_URL), 1)

        submitted = f"{PIG_RS_URL}/submitted"
        tool = _Sources(self, _pig_rs(**{submitted: _Answer(403, b"")}))

        refused = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs5334475172")

        self.assertEqual(
            (
                refused.failure.tool,
                refused.failure.failure_class,
                refused.failure.contact_attempts,
                refused.failure.route,
            ),
            ("eva_submitted_variants", "rejected", 1, "ask"),
        )
        self.assertEqual(tool.opener.sent.count(submitted), 1)

    def test_an_rs_placed_twice_on_the_claims_assembly_asks_for_a_placement(self):
        """Which placement the claim means is the model's to name (ADR 0049; owner
        #468 Q1 a); this tool has no model, so the answer stays unnamed."""
        # Synthetic: the recorded pig records with a second placement 100 bases on.
        second = PIG_RS.replace(b"3556402", b"3556502")
        rs_twice = PIG_RS[:-1] + b"," + second[1:]
        submitted_twice = (
            PIG_SUBMITTED[:-1]
            + b","
            + PIG_SUBMITTED.replace(b"3556402", b"3556502").replace(
                b"7881321083", b"7881321084"
            )[1:]
        )
        tool = _Sources(
            self,
            _pig_rs(
                **{
                    PIG_RS_URL: _Answer(200, rs_twice),
                    f"{PIG_RS_URL}/submitted": _Answer(200, submitted_twice),
                }
            ),
        )

        answer = tool.answer(assembly_accession="GCF_000003025.6", rsid="rs5334475172")

        result = answer.eva
        self.assertEqual(result.status, "placement_unnamed")
        self.assertRegex(result.conflicts[0], "2 loci on the Run's assembly")
        self.assertEqual(
            [c.one_based_position for c in result.candidates], [3556402, 3556502]
        )
        self.assertEqual(
            [[a.ss for a in p.eva.submitted] for p in result.placements],
            [["ss7881321083"], ["ss7881321084"]],
        )
        self.assertEqual(tool.opener.sent.count(f"{PIG_RS_URL}/submitted"), 1)
        self.assertEqual(tool.opener.sent.count(PIG_ALIAS_URL), 1)

    def test_an_rs_eva_does_not_hold_reads_no_ncbi_assembly_report(self):
        tool = _Sources(
            self, {f"{EVA}/1": _Answer(404, b""), f"{EVA}/2": _Answer(404, b"")}
        )

        tool.answer(assembly_accession="GCF_000003025.6", rsid="rs1")
        tool.answer(assembly_accession="GCF_000003025.6", rsid="rs2")

        self.assertEqual(tool.opener.sent, [f"{EVA}/1", f"{EVA}/2"])


NCBI = "https://api.ncbi.nlm.nih.gov"


def _ncbi_location(assembly, report, molecules, **more):
    """The product position lookup's first two GETs on `assembly`, then `more`."""
    genome = f"{NCBI}/datasets/v2/genome/accession/{assembly}"
    return {
        f"{genome}/dataset_report": _Answer(200, report),
        # The assembly's sequence names: its links, then its FTP report.
        **{
            url: _Answer(200, body)
            for url, body in names_answers(assembly, molecules).items()
        },
        **more,
    }


# The retained pig span (as `tests/test_variant_location.py` keeps it): the fourteen
# bases at NC_010454.4:3556401.
SPAN_NC_010454_4 = (
    b'{"data":{"seq_id":"NC_010454.4","position":3556400,'
    b'"deleted_sequence":"GGTGTCCCCCTCGC","inserted_sequence":"NNNNNNNNNNNNNN"}}'
)


def _span(sequence, start0, bases):
    return (
        f"{NCBI}/variation/v0/spdi/{sequence}:{start0}:{bases}:{'N' * bases}/contextual"
    )


class PositionTests(unittest.TestCase):
    """A position claim goes through the product's `look_up_stated_position` on either
    route (#393 Q10), each GET sent through the product's transports and the failure module. The
    lookup's own outcomes are the product's (`tests/test_variant_location.py`); these check
    that `look` reaches them, with the claim's assembly as the stated version."""

    def test_a_pig_deletion_is_found_at_its_eva_form_and_its_cluster_read(self):
        answers = _ncbi_location(
            "GCF_000003025.6",
            LOCATION_SSCROFA11_REPORT,
            SSCROFA11_MOLECULES,
            **{
                _span("NC_010454.4", 3556400, 14): _Answer(200, SPAN_NC_010454_4),
                PIG_AT: _Answer(200, EVA_AT_PIG_DEL),
                PIG_RS_URL: _Answer(200, PIG_RS5334475172),
                PIG_ALIAS_URL: _Answer(200, PIG_CM000823_5_ALIAS),
                f"{PIG_RS_URL}/submitted": _Answer(200, PIG_RS5334475172_SUBMITTED),
            },
        )
        tool = _Sources(self, answers)

        answer = tool.answer(
            assembly_accession="GCF_000003025.6",
            contig="chr12",
            position=3556401,
            ref="GGTGTCCCCCTCGC",
            alt="G",
        )

        self.assertEqual(answer.route, "eva")
        self.assertEqual(
            (answer.position.status, answer.position.verified_rsid),
            ("resolved", "rs5334475172"),
        )
        self.assertEqual(
            [
                (c.contig, c.one_based_position, c.ref, c.alt)
                for c in answer.position.candidates
            ],
            [("NC_010454.4", 3556402, "GTGTCCCCCTCGC", "")],  # dbSNP's SPDI form
        )
        self.assertEqual(tool.opener.sent, list(answers))

    def test_a_human_position_without_an_rsid_is_placed_by_its_listed_rsid(self):
        answers = _ncbi_location(
            "GCF_000001405.40",
            GRCH38_REPORT,
            GRCH38_MOLECULES,
            **{
                _span("NC_000019.10", 44908683, 1): _Answer(200, SPAN_NC_000019_10_T),
                f"{NCBI}/variation/v0/spdi/NC_000019.10:44908683:T:C/rsids": _Answer(
                    200, SPDI_RS429358
                ),
                DBSNP_URL: _Answer(200, RS429358),
            },
        )
        tool = _Sources(self, answers)

        answer = tool.answer(
            assembly_accession="GCF_000001405.40",
            contig="19",
            position=44908684,
            ref="T",
            alt="C",
        )

        self.assertEqual(answer.route, "dbsnp")
        self.assertEqual(
            (answer.position.status, answer.position.verified_rsid),
            ("resolved", "rs429358"),
        )
        self.assertEqual(
            answer.position.location.stated_assembly.name, "GCF_000001405.40"
        )
        self.assertEqual(tool.opener.sent, list(answers))

    def test_a_ref_the_assemblys_reference_does_not_hold_conflicts(self):
        answers = _ncbi_location(
            "GCF_000001405.40",
            GRCH38_REPORT,
            GRCH38_MOLECULES,
            **{_span("NC_000019.10", 44908683, 1): _Answer(200, SPAN_NC_000019_10_T)},
        )
        tool = _Sources(self, answers)

        answer = tool.answer(
            assembly_accession="GCF_000001405.40",
            contig="NC_000019.10",
            position=44908684,
            ref="G",
            alt="C",
        )

        self.assertEqual(answer.position.status, "conflicted")
        self.assertEqual(
            answer.position.conflicts,
            (
                "The Run's reference holds T at NC_000019.10:44908684; the request "
                "states G.",
            ),
        )

    def test_a_contig_no_record_literally_names_is_left_unnamed(self):
        """This tool has no model, so a sequence is named only by literal membership
        (owner #467 R4 a); a contig no record names stays unnamed, with no code
        verdict on its spelling (ADR 0049)."""
        answers = _ncbi_location("GCF_000001405.40", GRCH38_REPORT, GRCH38_MOLECULES)
        tool = _Sources(self, answers)

        answer = tool.answer(
            assembly_accession="GCF_000001405.40",
            contig="chr99",
            position=5,
            ref="A",
            alt="G",
        )

        self.assertEqual(answer.position.status, "sequence_unnamed")
        self.assertEqual(tool.opener.sent, list(answers))

    def test_a_position_get_fails_on_its_own_surface_and_asks(self):
        answers = _ncbi_location(
            "GCF_000003025.6",
            LOCATION_SSCROFA11_REPORT,
            SSCROFA11_MOLECULES,
            **{
                _span("NC_010454.4", 3556400, 14): _Answer(200, SPAN_NC_010454_4),
                PIG_AT: [_Answer(503, b"") for _ in range(3)],
            },
        )
        tool = _Sources(self, answers)

        answer = tool.answer(
            assembly_accession="GCF_000003025.6",
            contig="chr12",
            position=3556401,
            ref="GGTGTCCCCCTCGC",
            alt="G",
        )

        self.assertEqual(
            (
                answer.failure.tool,
                answer.failure.failure_class,
                answer.failure.contact_attempts,
                answer.failure.route,
            ),
            ("eva_clustered_variants_at", "exhausted", 3, "ask"),
        )
        self.assertEqual(tool.opener.sent.count(PIG_AT), 3)


class ProductShapePinTests(unittest.TestCase):
    """The product shapes the dbSNP route passes through and calls, as #368 left them (#343's
    at main `0ec72245`, less its typed unsupported alleles). A later pull of main that changes them turns this test red,
    and the lane follows the change before the pin moves."""

    def test_the_refsnp_result_and_the_normalize_refsnp_call_are_as_the_lane_uses_them(
        self,
    ):
        self.assertEqual(
            tuple(RefSnpToolResult.model_fields),
            (
                "result_id",
                "requested_rsid",
                "verified_rsid",
                "expected_assembly_accession",
                "status",
                "candidates",
                "conflicts",
                "merged_into",
                "merged_rsids",
                # Every placement copied, dbSNP's record state, the model's named
                # placement.
                "placements",
                "record_state",
                "named_placement",
                # The typed origin of a lookup ending, whose audit sentence is the
                # conflicts and the rationale.
                "origin",
                "source",
                "assessment",
                "lookup_source",
                "eva",
                # A position lookup's placement, "not_applicable" otherwise.
                "location",
            ),
        )
        self.assertEqual(
            tuple(ResolvedVariantCandidate.model_fields),
            (
                "assembly_accession",
                "assembly_name",
                "contig",
                "one_based_position",
                "ref",
                "alt",
                "aliases",
            ),
        )
        self.assertEqual(
            get_args(RefSnpStatus),
            (
                "resolved",
                "unlisted",
                "merged",
                "ambiguous",
                "conflicted",
                "incompatible",
                "not_found",
                "malformed",
                # A region target's own lookup.
                "region_located",
                # The lookup asks the model to name a sequence.
                "sequence_unnamed",
                # The lookup asks the model to name a stated version.
                "version_unnamed",
                # The model names a placement; the source's own answer, with no
                # verdict (R4 a).
                "placement_unnamed",
                "source_answered",
            ),
        )
        keyword, empty = inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty
        self.assertEqual(
            {
                name: (parameter.kind, parameter.default)
                for name, parameter in inspect.signature(
                    normalize_refsnp
                ).parameters.items()
            },
            {
                "rsid": (keyword, empty),
                "assembly_accession": (keyword, empty),
                "source": (keyword, empty),
                "raw_response": (keyword, empty),
                "requested_variant": (keyword, None),
                "requested_rsid": (keyword, None),
                "location": (keyword, "not_applicable"),
                "placement": (keyword, None),
            },
        )
        self.assertEqual(
            RequestedVariantIdentity._fields,
            ("contig", "one_based_position", "ref", "alt"),
        )


class CommandLineTests(unittest.TestCase):
    def test_look_without_a_source_is_refused_before_any_contact(self):
        """The developer names the source (#469): `look` has no default route."""
        self.opener = _Opener({})
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(stderr):
            variant_sources_main.main(
                [
                    "--cache",
                    str(self.cache),
                    "look",
                    "--assembly",
                    "GCF_000001405.40",
                    "--rsid",
                    "rs429358",
                ],
                opener=self.opener,
            )

        self.assertEqual(stop.exception.code, 2)
        # The refusal is the missing source (review r1 ST2), not another argument.
        self.assertIn(
            "the following arguments are required: --source", stderr.getvalue()
        )
        self.assertEqual(self.opener.sent, [])

    def setUp(self):
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = io.StringIO()

    def _main(self, argv, answers):
        self.opener = _Opener(answers)
        if argv[0] == "look" and "--source" not in argv:
            # The developer names the source; these tests' choice by assembly.
            argv = [*argv, "--source", _source_of(argv[argv.index("--assembly") + 1])]
        with run_failure_tuning(NO_BACKOFF), contextlib.redirect_stdout(self.out):
            return variant_sources_main.main(
                ["--cache", str(self.cache), *argv], opener=self.opener
            )

    def _record(self):
        [path] = sorted((self.cache / "lookups").iterdir())
        return json.loads(path.read_text())

    def test_look_writes_one_record_with_the_answers_and_this_runs_contact_count(self):
        # An earlier run in the same cache: its GET is its own record's, not this run's.
        self._main(
            ["look", "--assembly", "GCF_000003025.6", "--rsid", "rs7"],
            {f"{EVA}/7": _Answer(404, b"")},
        )
        [earlier] = (self.cache / "lookups").iterdir()

        code = self._main(
            ["look", "--assembly", "GCF_000003025.6", "--rsid", "rs5334475172"],
            _pig_rs(),
        )

        self.assertEqual(code, 0)
        [path] = set((self.cache / "lookups").iterdir()) - {earlier}
        record = json.loads(path.read_text())
        self.assertEqual(record["schema_version"], "bio-neb.variant-lookup/7")
        [answer] = record["answers"]
        self.assertEqual(
            (answer["claim"]["label"], answer["route"], answer["eva"]["status"]),
            ("rs5334475172", "eva", "resolved"),
        )
        self.assertEqual(
            self.opener.sent,
            [PIG_RS_URL, PIG_REPORT_URL, PIG_ALIAS_URL, f"{PIG_RS_URL}/submitted"],
        )
        self.assertEqual(record["contact_count"], 4)
        self.assertNotIn("contacts", record)
        self.assertEqual(json.loads(earlier.read_text())["contact_count"], 1)
        self.assertIn("rs5334475172\teva\tresolved", self.out.getvalue())

    def test_a_followed_merge_is_recorded_and_reported_with_both_rsids(self):
        code = self._main(
            ["look", "--assembly", "GCF_000001405.40", "--rsid", "rs60037639"],
            {
                f"{REFSNP}/60037639": _Answer(200, RS60037639),
                f"{REFSNP}/4149056": _Answer(200, RS4149056),
            },
        )

        self.assertEqual(code, 0)
        [answer] = self._record()["answers"]
        self.assertEqual(
            (
                answer["merge"]["status"],
                answer["merge"]["merged_into"],
                answer["dbsnp"]["requested_rsid"],
                answer["dbsnp"]["verified_rsid"],
                [candidate["alt"] for candidate in answer["dbsnp"]["candidates"]],
            ),
            ("merged", ["rs4149056"], "rs60037639", "rs4149056", ["A", "C"]),
        )
        self.assertIn(
            "rs60037639\tdbsnp\tresolved (merged into rs4149056)\n",
            self.out.getvalue(),
        )

    def test_a_followed_lookup_answered_404_is_recorded_and_later_claims_still_run(
        self,
    ):
        code = self._main(
            [
                "look",
                "--assembly",
                "GCF_000001405.40",
                "--rsid",
                "rs60037639",
                "--rsid",
                "rs429358",
            ],
            {
                f"{REFSNP}/60037639": _Answer(200, RS60037639),
                f"{REFSNP}/4149056": _Answer(404, b'{"error": "no such rs"}'),
                DBSNP_URL: _Answer(200, DBSNP_BODY),
            },
        )

        self.assertEqual(code, 0)
        record = self._record()
        self.assertIsNone(record["stopped"])
        merged, later = record["answers"]
        self.assertEqual(
            (
                merged["merge"]["merged_into"],
                merged["dbsnp"]["status"],
                merged["dbsnp"]["requested_rsid"],
            ),
            (["rs4149056"], "source_answered", "rs60037639"),
        )
        self.assertEqual(
            (later["claim"]["rsid"], later["dbsnp"]["status"]), ("rs429358", "resolved")
        )
        self.assertIn(
            "rs60037639\tdbsnp\tsource_answered (merged into rs4149056)\n",
            self.out.getvalue(),
        )

    def test_sites_are_read_as_label_contig_position_ref_and_alt(self):
        answers = _ncbi_location(
            "GCF_000001405.40",
            GRCH38_REPORT,
            GRCH38_MOLECULES,
            **{_span("NC_000019.10", 44908683, 1): _Answer(200, SPAN_NC_000019_10_T)},
        )

        self._main(
            [
                "look",
                "--assembly",
                "GCF_000001405.40",
                "--site",
                "S1=NC_000019.10:44908684:G>C",
            ],
            answers,
        )

        [answer] = self._record()["answers"]
        claim = answer["claim"]
        self.assertEqual(
            (
                claim["label"],
                claim["contig"],
                claim["position"],
                claim["ref"],
                claim["alt"],
            ),
            ("S1", "NC_000019.10", 44908684, "G", "C"),
        )
        self.assertEqual(answer["position"]["status"], "conflicted")
        self.assertIn("S1\tdbsnp\tconflicted\n", self.out.getvalue())

    def test_a_malformed_site_is_refused_before_any_contact(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as stop,
        ):
            self._main(
                ["look", "--assembly", "GCF_000002315.4", "--site", "SV1=2:5"], {}
            )

        self.assertEqual(stop.exception.code, 2)
        self.assertEqual(self.opener.sent, [])

    def test_a_failed_claim_asks_and_the_claims_after_it_still_run(self):
        answers = _pig_rs(**{f"{EVA}/7": [_Answer(503, b"busy")] * 3})

        with contextlib.redirect_stderr(io.StringIO()) as error:
            code = self._main(
                [
                    "look",
                    "--assembly",
                    "GCF_000003025.6",
                    "--rsid",
                    "rs7",
                    "--rsid",
                    "rs5334475172",
                ],
                answers,
            )

        self.assertEqual(code, 2)
        record = self._record()
        self.assertIsNone(record["stopped"])
        failed, answered = record["answers"]
        self.assertEqual(
            (
                failed["claim"]["label"],
                failed["failure"]["failure_class"],
                failed["failure"]["route"],
            ),
            ("rs7", "exhausted", "ask"),
        )
        self.assertEqual(
            (answered["claim"]["label"], answered["eva"]["status"]),
            ("rs5334475172", "resolved"),
        )
        self.assertEqual(
            self.opener.sent,
            [f"{EVA}/7"] * 3
            + [PIG_RS_URL, PIG_REPORT_URL, PIG_ALIAS_URL, f"{PIG_RS_URL}/submitted"],
        )
        self.assertEqual(record["contact_count"], 7)
        self.assertIn("rs7\teva\texhausted ask", self.out.getvalue())
        self.assertRegex(error.getvalue(), "rs7 asks: exhausted after 3 attempts")

    def test_an_unexpected_error_on_either_route_stops_after_one_attempt(self):
        """Our bug is no transport fault: it is not retried, asks nothing and stops the run."""
        eva = {
            f"{EVA}/7": [RuntimeError("bug in the reader")] * 3,
            PIG_RS_URL: _Answer(200, PIG_RS),
        }
        routes = {
            "eva": (eva, "GCF_000003025.6", "rs7", f"{EVA}/7", PIG_RS_URL),
            "dbsnp": (
                {DBSNP_URL: [RuntimeError("bug in the reader")] * 3},
                "GCF_000001405.40",
                "rs429358",
                DBSNP_URL,
                "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/7412",
            ),
        }
        for route, (answers, assembly, first, failed_url, later_url) in routes.items():
            with self.subTest(route=route):
                self.cache = Path(self.enterContext(tempfile.TemporaryDirectory()))
                later = "rs5334475172" if route == "eva" else "rs7412"

                with contextlib.redirect_stderr(io.StringIO()):
                    code = self._main(
                        ["look", "--assembly", assembly, "--rsid", first]
                        + ["--rsid", later],
                        answers,
                    )

                self.assertEqual(code, 1)
                record = self._record()
                self.assertEqual(record["answers"], [])
                self.assertEqual(
                    (record["stopped"]["claim_label"], record["stopped"]["error"]),
                    (first, "RuntimeError: bug in the reader"),
                )
                self.assertEqual(record["stopped"]["unattempted"], [later])
                self.assertEqual(self.opener.sent.count(failed_url), 1)
                self.assertNotIn(later_url, self.opener.sent)
                self.assertEqual(record["contact_count"], 1)


if __name__ == "__main__":
    unittest.main()
