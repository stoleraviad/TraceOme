"""Prebuilt BWA indexes from a reliable source, in place of the CPU preparation job (#365).

A source's catalogue lists an index for the Run's assembly accession; the planning model
reads its contigs beside NCBI's rows and chooses it, and the Execution Review binds it.
Execution trusts that choice: it streams the files into the Run's bucket as the eight
prepared roles, renamed to NCBI's names in flight (`rename`; owner I7, #464). The M5 fit
and the assembly of fitted roles live only in lane Rf's development tool
(`tools/reference_index/fit.py`, `bundle.py`; #459).
"""
