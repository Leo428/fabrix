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


def _so3(quat_wxyz):
    """Build a ``jaxlie.SO3`` from a wxyz quaternion (MuJoCo convention).

    jaxlie stores rotations as xyzw, so the scalar ``w`` is rolled to the back.
    """
    return jaxlie.SO3.from_quaternion_xyzw(jnp.concatenate([quat_wxyz[1:], quat_wxyz[:1]]))


def _se3(pos, quat_wxyz):
    """Build a ``jaxlie.SE3`` from a position and a wxyz quaternion (MuJoCo convention)."""
    return jaxlie.SE3.from_rotation_and_translation(_so3(quat_wxyz), pos)


def _rotate_points(quat_wxyz, pts):
    """Rotate local points ``pts`` ``(P,3)`` into the world frame by a wxyz quaternion → ``(P,3)``."""
    return pts @ _so3(quat_wxyz).as_matrix().T


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


def control_point_jack(radius: float, full: bool = True):
    """Local control-point offsets ``(P,3)`` for :func:`control_points_error_map`: a jack of radius ``r``.

    ``full=True`` → the 7-point NVlabs jack (origin + ±x/±y/±z) — symmetric, robust, well-conditioned.
    ``full=False`` → the minimal 4-point set (origin + +x/+y/+z) — cheaper, still pins the full pose
    (≥3 non-collinear points suffice). ``radius`` is a *virtual* lever arm (the points need not be
    physical): it sets the rotation/translation balance — larger ``r`` gives more rotational stiffness &
    observability (``∝ r²``) but a *lower* saturated angular rate (``ω_rot ≈ f_max/(b·r)``), while
    translation speed is ``r``-independent. ~0.1 m is a good start for an arm EE; sweep in sim.
    """
    r = float(radius)
    if full:
        pts = [[0, 0, 0], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]]
    else:
        pts = [[0, 0, 0], [r, 0, 0], [0, r, 0], [0, 0, r]]
    return jnp.asarray(pts)


def control_points_error_map(provider, target_pos, target_quat, offsets):
    """Control-points pose error: stacked world-position error of ``P`` rigid points on the EE, ``(3P,)``.

    Represents the EE pose by ``P≥3`` non-collinear points fixed in the site frame (``offsets``, a
    ``(P,3)`` local jack — see :func:`control_point_jack`). Each point's world-position error is stacked
    into ``phi(q) -> (3P,)``, **all in meters** — no SE(3) twist, no mixed (m, rad) units. Matching ≥3
    non-collinear points pins the full 6-DOF pose (the rigid-registration fact), so driving this error to
    zero drives position **and** orientation to zero. Orientation authority is geometric: a rotation
    error displaces the offset points by ``≈ r·θ`` and the positional attractor restores them (torque
    ``∝ k·r²``) — the offset radius ``r``, not a separate gain, sets the rotation/translation balance.
    This is the NVlabs/FABRICS palm-points construction; contrast :func:`se3_pose_error_map` (the coupled
    twist, whose single ``f_max`` over mixed units throttles rotation).

    ``target_pos``/``target_quat`` (wxyz) are held fixed, so the autodiff Jacobian and curvature flow
    through ``T_current(q)`` only. The error is cast to ``q.dtype`` (jaxlie carries float64 constants).
    """
    offs = jnp.asarray(offsets)
    t_pts = target_pos + _rotate_points(target_quat, offs)   # (P,3) world target points, fixed

    def phi(q):
        p, quat = provider.site_pose(q)
        cur_pts = p + _rotate_points(quat, offs)             # (P,3) current world points
        return (cur_pts - t_pts).reshape(-1).astype(q.dtype)  # (3P,), all meters

    return phi
