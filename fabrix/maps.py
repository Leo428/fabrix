"""Task maps: pure functions ``phi(q) -> task coordinates``, built from a provider.

A map depends on ``q`` alone (params live in the leaf's force law), so ``fabrix.diff`` can
take its Jacobian and curvature term by autodiff. M3 will add an SE(3) pose map via jaxlie.
"""
from __future__ import annotations


def site_position_map(provider):
    """Return ``phi(q) -> (3,)``, the world position of the provider's tracked site."""

    def phi(q):
        return provider.site_pos(q)

    return phi
