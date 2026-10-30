"""Reference metadata required by the supported Parabricks variant-calling consumer."""


def reference_asset_roles():
    # Annotation is not consumed by the current FASTQ-to-BAM/VCF operation.
    return ("genomic_fasta",)
