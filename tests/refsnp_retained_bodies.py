"""NCBI RefSNP bodies derived from retained dbSNP responses (#343).

Provenance (read-only sources, never modified):

- ``RS60037639``, ``RS61228756`` and ``RS10735810`` are the exact retained
  merged-rsID bodies from #340's Run root
  ``.bio-neb-local/live-runs/340-refsnp-20261015T104934Z-5d1c805b-a1``
  (``bodies/sha256/b8cbd5ec…a1c01f``, 339 bytes; ``bf4c9f93…e1f653``, 338 bytes;
  ``1b06930c…2f8261``, 491 bytes; listed in
  ``docs/development/reviews/issue340-vcf-spike/manifest/responses.jsonl``).
- ``RS4149056`` and ``RS429358`` derive from #300's retained bodies under
  ``.bio-neb-local/live-runs/300-e2e-20261014T090134Z-e6030e44-a1/runs/3ac7744f-aab4-4102-ae64-e60d7cf4fadf/artifacts/sha256/``
  (``1723b625…f02021``, 118,346 bytes; ``8cc8f8a0…f8fea6``, 117,598 bytes);
  ``RS2228570`` from #340's ``bodies/sha256/0669cfa4…75e112`` (140,241 bytes).
- ``RS4988235`` derives from #107's live Run, the body behind the planner stop
  (#360):
  ``.bio-neb-local/live-runs/107-full-20261016T033105Z-2ac5fd60-a1/9b1e840c-f8c7-4398-b848-7340968db734/artifacts/sha256/07fd6b80…efb95c``
  (56,475 bytes), its ``RefSnpToolResultRecorded`` ``a92fdc53``.
- ``RS756427959``, ``RS754204844`` and ``RS878871521`` (#368, the indel cases of
  #340 rows 18 and 65) derive from #340's ``bodies/sha256/1e6f3394…6f1563``
  (16,232 bytes), ``cec5903d…9aaaec`` (52,801 bytes) and ``b880b4f9…ed57f9``
  (8,126 bytes).

The current-rsID bodies are trimmed to what RefSNP normalization reads:
``refsnp_id``, ``dbsnp1_merges``, and each assembly-bound placement's ``seq_id``,
``is_ptlp``, ``seq_id_traits_by_assembly`` and allele SPDIs. Annotations, support,
HGVS strings and placements on no assembly are dropped; every kept value is the
retained value. They are observed dbSNP facts, not expectations: tests state their
expected candidates independently.
"""

from __future__ import annotations

import json

RS60037639 = (
    b'{"refsnp_id":"60037639","create_date":"2007-12-15T21:04Z",'
    b'"last_update_date":"2007-12-17T16:57Z","last_update_build_id":"130",'
    b'"dbsnp1_merges":[],"citations":[],"lost_obs_movements":[],'
    b'"present_obs_movements":[],"merged_snapshot_data":{"proxy_time":'
    b'"2008-05-25T21:18Z","proxy_build_id":"130","merged_into":["4149056"]},'
    b'"mane_select_ids":[]}'
)
RS60037639_SHA256 = "b8cbd5ec0d0722593f254140818c18d5e03c2397349e559f86dc4fbbfa51c01f"

RS61228756 = (
    b'{"refsnp_id":"61228756","create_date":"2007-12-16T06:39Z",'
    b'"last_update_date":"2007-12-17T17:08Z","last_update_build_id":"130",'
    b'"dbsnp1_merges":[],"citations":[],"lost_obs_movements":[],'
    b'"present_obs_movements":[],"merged_snapshot_data":{"proxy_time":'
    b'"2008-05-26T20:59Z","proxy_build_id":"130","merged_into":["429358"]},'
    b'"mane_select_ids":[]}'
)
RS61228756_SHA256 = "bf4c9f937b2fa808425c509fe83d10859841e5847668d2edc60c0c2cbce1f653"

RS10735810 = (
    b'{"refsnp_id":"10735810","create_date":"2004-02-27T09:30Z",'
    b'"last_update_date":"2008-04-4T16:09Z","last_update_build_id":"130",'
    b'"dbsnp1_merges":[{"merged_rsid":"17881966","revision":"126",'
    b'"merge_date":"2006-03-11T05:33Z"},{"merged_rsid":"52811041",'
    b'"revision":"128","merge_date":"2007-09-21T15:55Z"}],"citations":[],'
    b'"lost_obs_movements":[],"present_obs_movements":[],"merged_snapshot_data":'
    b'{"proxy_time":"2008-05-23T17:11Z","proxy_build_id":"130",'
    b'"merged_into":["2228570"]},"mane_select_ids":[]}'
)
RS10735810_SHA256 = "1b06930ca407634efc6fb181fd85768ce9228b71a1c25a3dd1fd0d257f2a8261"


def _assembly(name: str, accession: str) -> dict:
    return {
        "assembly_name": name,
        "assembly_accession": accession,
        "is_top_level": True,
        "is_alt": False,
        "is_patch": False,
        "is_chromosome": True,
    }


def _placement(
    seq_id: str,
    *,
    is_ptlp: bool,
    assembly: dict,
    position: int,
    ref: str,
    alts: tuple[str, ...],
) -> dict:
    return {
        "seq_id": seq_id,
        "is_ptlp": is_ptlp,
        "placement_annot": {"seq_id_traits_by_assembly": [assembly]},
        "alleles": [
            {
                "allele": {
                    "spdi": {
                        "seq_id": seq_id,
                        "position": position,
                        "deleted_sequence": ref,
                        "inserted_sequence": inserted,
                    }
                }
            }
            for inserted in (ref, *alts)
        ],
    }


def _merges(*rsids: tuple[str, str, str]) -> list[dict]:
    return [
        {"merged_rsid": rsid, "revision": revision, "merge_date": date}
        for rsid, revision, date in rsids
    ]


_GRCH38 = _assembly("GRCh38.p14", "GCF_000001405.40")
_GRCH37 = _assembly("GRCh37.p13", "GCF_000001405.25")

RS429358_DOCUMENT = {
    "refsnp_id": "429358",
    "dbsnp1_merges": _merges(
        ("630496", "85", "2000-09-19T13:56Z"),
        ("61228756", "130", "2008-05-26T20:59Z"),
    ),
    "primary_snapshot_data": {
        "variant_type": "snv",
        "placements_with_allele": [
            _placement(
                "NC_000019.10",
                is_ptlp=True,
                assembly=_GRCH38,
                position=44908683,
                ref="T",
                alts=("C",),
            ),
            _placement(
                "NC_000019.9",
                is_ptlp=False,
                assembly=_GRCH37,
                position=45411940,
                ref="T",
                alts=("C",),
            ),
        ],
    },
}

RS4149056_DOCUMENT = {
    "refsnp_id": "4149056",
    "dbsnp1_merges": _merges(
        ("52816141", "128", "2007-09-21T15:49Z"),
        ("60037639", "130", "2008-05-25T21:18Z"),
    ),
    "primary_snapshot_data": {
        "variant_type": "snv",
        "placements_with_allele": [
            _placement(
                "NC_000012.12",
                is_ptlp=True,
                assembly=_GRCH38,
                position=21178614,
                ref="T",
                alts=("A", "C"),
            ),
            _placement(
                "NC_000012.11",
                is_ptlp=False,
                assembly=_GRCH37,
                position=21331548,
                ref="T",
                alts=("A", "C"),
            ),
        ],
    },
}

RS2228570_DOCUMENT = {
    "refsnp_id": "2228570",
    "dbsnp1_merges": _merges(
        ("8179174", "130", "2008-05-23T17:11Z"),
        ("10735810", "130", "2008-05-23T17:11Z"),
        ("17881966", "126", "2006-03-11T05:33Z"),
        ("52811041", "128", "2007-09-21T15:55Z"),
        ("56641119", "130", "2008-05-27T11:21Z"),
        ("57067622", "130", "2008-05-23T17:11Z"),
        ("117559231", "132", "2010-08-16T13:01Z"),
    ),
    "primary_snapshot_data": {
        "variant_type": "snv",
        "placements_with_allele": [
            _placement(
                "NC_000012.12",
                is_ptlp=True,
                assembly=_GRCH38,
                position=47879111,
                ref="A",
                alts=("C", "G", "T"),
            ),
            _placement(
                "NC_000012.11",
                is_ptlp=False,
                assembly=_GRCH37,
                position=48272894,
                ref="A",
                alts=("C", "G", "T"),
            ),
        ],
    },
}

RS4988235_DOCUMENT = {
    "refsnp_id": "4988235",
    "dbsnp1_merges": [],
    "primary_snapshot_data": {
        "variant_type": "snv",
        "placements_with_allele": [
            _placement(
                "NC_000002.12",
                is_ptlp=True,
                assembly=_GRCH38,
                position=135851075,
                ref="G",
                alts=("A", "C", "T"),
            ),
            _placement(
                "NC_000002.11",
                is_ptlp=False,
                assembly=_GRCH37,
                position=136608645,
                ref="G",
                alts=("A", "C", "T"),
            ),
        ],
    },
}

RS756427959_DOCUMENT = {
    "refsnp_id": "756427959",
    "dbsnp1_merges": [],
    "primary_snapshot_data": {
        "variant_type": "delins",
        "placements_with_allele": [
            _placement(
                "NC_000001.11",
                is_ptlp=True,
                assembly=_GRCH38,
                position=14397,
                ref="TGTTG",
                alts=("TG",),
            ),
            _placement(
                "NC_000001.10",
                is_ptlp=False,
                assembly=_GRCH37,
                position=14397,
                ref="TGTTG",
                alts=("TG",),
            ),
        ],
    },
}

RS754204844_DOCUMENT = {
    "refsnp_id": "754204844",
    "dbsnp1_merges": _merges(("67444599", "136", "2011-10-15T12:15Z")),
    "primary_snapshot_data": {
        "variant_type": "del",
        "placements_with_allele": [
            _placement(
                "NC_000001.11",
                is_ptlp=True,
                assembly=_GRCH38,
                position=175160809,
                ref="C",
                alts=("",),
            ),
            _placement(
                "NC_000001.10",
                is_ptlp=False,
                assembly=_GRCH37,
                position=175129945,
                ref="C",
                alts=("",),
            ),
        ],
    },
}

# The mitochondrial placement names both assemblies in one traits list.
_RS878871521_MT = _placement(
    "NC_012920.1",
    is_ptlp=True,
    assembly=_GRCH37,
    position=302,
    ref="C" * 7,
    alts=tuple("C" * count for count in (8, 9, 10, 18, 21)),
)
_RS878871521_MT["placement_annot"]["seq_id_traits_by_assembly"].append(_GRCH38)
RS878871521_DOCUMENT = {
    "refsnp_id": "878871521",
    "dbsnp1_merges": [],
    "primary_snapshot_data": {
        "variant_type": "delins",
        "placements_with_allele": [_RS878871521_MT],
    },
}


def refsnp_body(document: dict) -> bytes:
    """One JSON response body, as the RefSNP adapter returns it."""
    return json.dumps(document, separators=(",", ":")).encode()


RS429358 = refsnp_body(RS429358_DOCUMENT)
RS4149056 = refsnp_body(RS4149056_DOCUMENT)
RS2228570 = refsnp_body(RS2228570_DOCUMENT)
RS4988235 = refsnp_body(RS4988235_DOCUMENT)
RS756427959 = refsnp_body(RS756427959_DOCUMENT)
RS754204844 = refsnp_body(RS754204844_DOCUMENT)
RS878871521 = refsnp_body(RS878871521_DOCUMENT)
