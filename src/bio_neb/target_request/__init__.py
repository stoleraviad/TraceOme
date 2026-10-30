"""The target-request module (#367): a source's targets placed on the Run's assembly and
written as the Run's request file.

`paper_targets` is the one record paper reading writes (`bio-neb.paper-targets/1`) and
`paper_text` its general parser; a contig keeps its written spelling (#467).
`placement.place` answers each target on the Run's assembly through the product's NCBI source
operations (`ncbi_sources`, G15's gene and change in `gene_change` and `annotation`) and the
resolver order (`resolver`), with the flank move behind the narrow `resolver.FlankMover` port.
`request_document.write_request_file` is the only writer of a request document
(`investigation_request.REQUEST_SCHEMA`, `/3`); the Run reads it through
`investigation.investigation_request`.
"""
