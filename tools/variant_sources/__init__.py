"""Variant-source developer entries (lane Rf, #331; failure path #352; #343's result shape
#353), thin over the product's source operations since #417."""

from .contracts import ClaimAnswer, VariantClaim
from .lookup import VariantSources

__all__ = ["ClaimAnswer", "VariantClaim", "VariantSources"]
