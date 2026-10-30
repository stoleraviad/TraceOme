"""NCBI Datasets reads as product source operations.

`bio_neb.providers.ncbi_datasets.NcbiDatasetsReferenceAdapter` is called, not copied, over
the product's NCBI transport, and each read runs on the surface of the product Tool that
makes it (`source_contacts.call_source`): a taxon's assemblies on `discover_assemblies`, an
accession's description on `inspect_reference`. A replaced or suppressed assembly is
described from its any-version report by the adapter's own rule
(`describe_assembly`).
"""

from __future__ import annotations

from bio_neb.agentic_journey.source_contacts import call_source
from bio_neb.providers.ncbi_datasets import (
    MetadataTransport,
    NcbiDatasetsReferenceAdapter,
)
from bio_neb.reference_discovery.reference_identity_tool import (
    AssemblyAdapterObservation,
    AssemblyQuery,
)


class Datasets:
    """The product adapter's assembly reads, each on its own failure surface."""

    def __init__(self, transport: MetadataTransport) -> None:
        self.adapter = NcbiDatasetsReferenceAdapter(transport)

    @property
    def contact_count(self) -> int:
        return self.adapter.contact_count

    def species(self, ncbi_tax_id: int) -> AssemblyAdapterObservation:
        return call_source(
            "discover_assemblies",
            lambda: self.adapter.discover_assemblies(ncbi_tax_id, AssemblyQuery()),
        )

    def describe(self, accession: str) -> AssemblyAdapterObservation:
        """The accession's description, current or, when NCBI has no current version
        of it, its any-version one."""
        return call_source(
            "inspect_reference",
            lambda: self.adapter.describe_assembly(accession),
        )
