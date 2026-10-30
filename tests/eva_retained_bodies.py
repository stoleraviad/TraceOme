"""European Variation Archive bodies, copied byte for byte from recorded answers (#366).

Provenance (read-only sources, never modified):

- ``PIG_RS5334475172`` and ``PIG_RS5334475172_SUBMITTED`` are lane Rf's run-plan step 1
  probe (2026-10-15), recorded in ``tests/test_variant_sources.py`` ("Recorded and
  synthetic source answers"): EVA's clustered variant ``rs5334475172`` (a deletion on
  Sscrofa11.1, ``GCA_000003025.6`` ``CM000823.5:3556402``) and its submitted variants.
- ``PIG_CM000823_5_ALIAS`` is #366's development GET of EVA's contig-alias service,
  2026-10-17,
  ``https://www.ebi.ac.uk/eva/webservices/contig-alias/v1/chromosomes/genbank/CM000823.5``
  (HTTP 200, 689 bytes, sha256 ``40e5bc17…e19e9c``).

- ``RS160719159_GONE`` is EVA's HTTP 410 answer for ``rs160719159`` in #457's C. elegans
  check, taken by that worker outside the Run after its stop:
  ``docs/development/reviews/issue457-live-checkpoint/celegans-a1/evidence/eva-410-rs160719159.body``
  (232 bytes, ``application/json``; the status in ``eva-410-rs160719159.status``).

They are observed EVA facts, not expectations: tests state their expected outcomes
independently.
"""

EVA_IDENTIFIERS = "https://www.ebi.ac.uk/eva/webservices/identifiers/v1"
EVA_CONTIG_ALIAS = "https://www.ebi.ac.uk/eva/webservices/contig-alias/v1"

PIG_RS5334475172 = (
    b'[{"accession":5334475172,"version":1,"data":{"assemblyAccession":"GCA_000003025.6",'
    b'"taxonomyAccession":9823,"contig":"CM000823.5","start":3556402,"type":"DEL",'
    b'"validated":false,"mapWeight":null,"createdDate":"2023-03-24T15:21:19.475"}}]'
)
PIG_RS5334475172_SUBMITTED = (
    b'[{"accession":7881321083,"version":1,"data":{"referenceSequenceAccession":'
    b'"GCA_000003025.6","taxonomyAccession":9823,"projectAccession":"PRJEB49109",'
    b'"contig":"CM000823.5","start":3556402,"referenceAllele":"GTGTCCCCCTCGC",'
    b'"alternateAllele":"","clusteredVariantAccession":5334475172,'
    b'"supportedByEvidence":true,"assemblyMatch":true,"allelesMatch":true,'
    b'"validated":false,"mapWeight":null,"createdDate":"2022-01-25T17:27:51.445",'
    b'"remappedFrom":null,"remappedDate":null,"remappingId":null,'
    b'"backPropagatedVariantAccession":null}}]'
)
PIG_CM000823_5_ALIAS = (
    b'{"_embedded":{"chromosomeEntities":[{"genbankSequenceName":"Chr12",'
    b'"enaSequenceName":"12","insdcAccession":"CM000823.5","refseq":"NC_010454.4",'
    b'"seqLength":61602749,"ucscName":"chr12","md5checksum":'
    b'"040a18ab3cdb44465035749d9f27309a","trunc512checksum":null,'
    b'"contigType":"CHROMOSOME","assembly":{"insdcAccession":"GCA_000003025.6",'
    b'"name":"Sscrofa11.1","organism":"Sus scrofa (pig)","taxid":9823,'
    b'"refseq":"GCF_000003025.6","md5checksum":null,"trunc512checksum":null,'
    b'"genbankRefseqIdentical":false}}]},"_links":{"self":{"href":'
    b'"http://www.ebi.ac.uk/eva/webservices/contig-alias/v1/chromosomes/genbank/'
    b'CM000823.5?page=0&size=10"}},"page":{"size":10,"totalElements":1,'
    b'"totalPages":1,"number":0}}'
)
RS160719159_GONE = (
    b'[{"accession":160719159,"version":1,"data":{"assemblyAccession":"GCA_000298735.2",'
    b'"taxonomyAccession":9940,"contig":"AMGL02039708.1","start":1933,"type":"SNV",'
    b'"validated":false,"mapWeight":null,"createdDate":"2011-07-22T16:49:00"}}]'
)
