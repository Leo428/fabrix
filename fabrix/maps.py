"""Task maps: pure functions ``phi(q) -> task coordinates``, built from a provider.

A map depends on ``q`` alone (params live in the leaf's force law), so ``fabrix.diff`` can
take its Jacobian and curvature term by autodiff. M3 will add an SE(3) pose map via jaxlie.

Distance maps for barrier geometries return a ``(1,)`` array (not a bare scalar) so the spec
algebra sees a 1-D task space and ``pullback`` shapes work uniformly.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp


def site_position_map(provider):
    """Return ``phi(q) -> (3,)``, the world position of the provider's tracked site."""

    def phi(q):
        return provider.site_pos(q)

    return phi


def sphere_sdf_map(provider, center, radius: float, site_name: Optional[str] = None):
    """Signed distance from the tracked site to a sphere: ``phi(q) = ||p - center|| - radius``.

    Returns ``(1,)``; positive outside, zero on the surface. ``site_name`` is accepted for API
    symmetry but the provider already tracks one site, so it is currently informational.
    """
    c = jnp.asarray(center)

    def phi(q):
        p = provider.site_pos(q)
        return (jnp.linalg.norm(p - c) - radius).reshape(1)

    return phi


def plane_sdf_map(provider, point, normal, site_name: Optional[str] = None):
    """Signed distance from the tracked site to a plane: ``phi(q) = n . (p - point)``.

    Returns ``(1,)``; positive on the ``+normal`` side. ``normal`` is normalized here.
    """
    p0 = jnp.asarray(point)
    n = jnp.asarray(normal)
    n = n / jnp.linalg.norm(n)

    def phi(q):
        p = provider.site_pos(q)
        return jnp.dot(p - p0, n).reshape(1)

    return phi
