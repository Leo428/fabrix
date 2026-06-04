"""Task maps: pure functions ``phi(q) -> task coordinates``, built from a provider.

A map depends on ``q`` alone (params live in the leaf's force law), so ``fabrix.diff`` can
take its Jacobian and curvature term by autodiff. M3 will add an SE(3) pose map via jaxlie.

Distance maps for barrier geometries return a ``(1,)`` array (not a bare scalar) so the spec
algebra sees a 1-D task space and ``pullback`` shapes work uniformly.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import jaxlie


def _se3(pos, quat_wxyz):
    """Build a ``jaxlie.SE3`` from a position and a wxyz quaternion (MuJoCo convention).

    jaxlie stores rotations as xyzw, so the scalar ``w`` is rolled to the back.
    """
    rot = jaxlie.SO3.from_quaternion_xyzw(jnp.concatenate([quat_wxyz[1:], quat_wxyz[:1]]))
    return jaxlie.SE3.from_rotation_and_translation(rot, pos)


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


def se3_pose_error_map(provider, target_pos, target_quat):
    """Full-pose error map ``phi(q) = Log(T_target^{-1} T_current(q)) in se(3)``, shape ``(6,)``.

    A coupled SE(3) pose error: zero exactly when the tracked site reaches the target pose, and a
    geodesic (screw) twist otherwise. ``target_pos``/``target_quat`` (wxyz) are held fixed, so the
    autodiff Jacobian and curvature term differentiate through ``T_current(q)`` only. jaxlie's Log
    has a Taylor fallback at the identity, so ``J``/``Jdq`` stay finite as the error -> 0.

    Tangent ordering follows jaxlie's SE3: ``[v_translation (3), w_rotation (3)]``.
    """
    T_tgt_inv = _se3(target_pos, target_quat).inverse()

    def phi(q):
        p, quat = provider.site_pose(q)
        # jaxlie carries float64 constants; under jax_enable_x64 that would promote a float32 config
        # to float64 (e.g. breaking a float32 scan carry). Anchor the error to the config dtype.
        return (T_tgt_inv @ _se3(p, quat)).log().astype(q.dtype)

    return phi
