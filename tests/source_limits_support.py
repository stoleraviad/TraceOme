"""A test's own source limits, declared as a Tool row or an operation registration
would declare them (#447: no limits table to patch)."""

from unittest.mock import patch

from bio_neb.agentic_journey import source_contacts
from bio_neb.failure import ContactLimitOverrides


def declaring(limits: dict[str, ContactLimitOverrides]):
    """Patch these surfaces' declared limits; every other surface keeps its own."""
    declared = source_contacts.declared_limits
    return patch.object(
        source_contacts,
        "declared_limits",
        lambda name: limits[name] if name in limits else declared(name),
    )
