"""HD2 geometries, the energization operator, and barrier geometry leaves.

A *geometry* produces a speed-independent path: an acceleration ``a_g(x, xd)`` homogeneous of
degree 2 (HD2) in ``xd``, so rescaling speed retraces the same path. **Energization** adds a
force purely along ``xd`` to make a geometry conserve a Finsler energy (:mod:`fabrix.energy`),
turning a bare path into a stable, speed-regulated fabric.

**Why energize at the root.** A barrier lives in a 1-D leaf space, where energization is
degenerate: any 1-D acceleration changes speed, so the energy-conserving projection has no room
to act and would cancel the geometry entirely. So bare geometries are pulled back and *combined*
at the root (config space, n-D), and the combined geometry is energized there — where the
projection orthogonal to ``qd`` is nontrivial. :class:`fabrix.fabric.GeometricFabric` wires this.

**Barriers.** A barrier is an HD2 geometry whose acceleration ``k_b xd^2 / d^p`` blows up as the
distance ``d`` to a limit/obstacle goes to zero, gated to act only while *approaching*
(``xd < 0``). The blow-up is what makes the constraint invariant (you cannot reach ``d = 0`` from
``d > 0`` — the deceleration diverges first); energization keeps the barrier from injecting energy.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from fabrix.diff import value_jac_curv
from fabrix.maps import plane_sdf_map, sphere_sdf_map
from fabrix.spec import Spec, dynamic_gain


def energize(a_g, v, M_e, f_e, eps: float = 1e-8):
    """Energize a geometry acceleration ``a_g`` to conserve the energy with spec ``(M_e, f_e)``.

    Adds the unique multiple of the velocity ``v`` that zeroes the energy rate
    ``v^T (M_e a + f_e)``:

        a = a_g - [ v^T (M_e a_g + f_e) / (v^T M_e v) ] v.

    By construction ``v^T (M_e a + f_e) = 0``, so the energy is conserved *exactly* in continuous
    time. The correction is along ``v`` only, so the geometry's path is preserved — energization
    regulates speed, it does not bend the path. ``eps`` floors the ``v^T M_e v -> 0`` (near-rest)
    singularity; at rest the correction is harmless (numerator -> 0 with no motion to conserve).
    """
    Mv = M_e @ v
    alpha = (v @ (M_e @ a_g + f_e)) / (v @ Mv + eps)
    return a_g - alpha * v


def _barrier_accel(d, dd, k_b: float, power: float):
    """HD2 repulsion ``k_b dd^2 / d^p`` along +d, active only while approaching (``dd < 0``).

    ``dd^2`` is HD2 in velocity and the ``dd < 0`` switch is sign-only (homogeneous degree 0), so
    the product is HD2 and ``C1`` at ``dd = 0`` (it vanishes there). Returns the +``d`` acceleration.
    """
    approaching = (dd < 0.0).astype(d.dtype)
    return k_b * (dd * dd) / d**power * approaching


def _barrier_metric(d, dd, m_b: float):
    """Barrier priority metric ``m_b / d`` while approaching, else 0.

    Grows as the boundary nears so the about-to-be-violated constraint dominates the
    metric-weighted geometry combination; vanishes when moving away so it never fights free motion.
    """
    approaching = (dd < 0.0).astype(d.dtype)
    return m_b / d * approaching


def _band(d, d0: float):
    """C1 standoff envelope in [0, 1]: 1 at the boundary, smoothly 0 at and beyond ``d0``.

    Fades a barrier *potential*'s metric in/out across the standoff band without the metric jump
    (and resulting q_ddot step) a hard ``d < d0`` switch would inject into the root solve.
    """
    r = jnp.clip(1.0 - d / d0, 0.0, 1.0)
    return r * r


def _barrier_potential_grad(d, k_p: float, d0: float):
    """``d(psi)/dd`` for the localized barrier potential ``psi = 1/2 k_p (1/d - 1/d0)^2`` (d < d0).

    The potential diverges as ``d -> 0`` and is ``0`` (value and slope) at ``d = d0``, so it acts
    only inside the standoff band ``d0``. The gradient is ``<= 0``, so the resulting force
    ``-dpsi/dd >= 0`` always pushes toward larger ``d`` (away from the boundary). This is what makes
    the constraint a hard invariant: with bounded kinetic energy, the diverging potential is
    unreachable, so ``d`` cannot reach ``0``. Energy-conserving geometries can deflect but not
    stop a head-on approach — only a potential like this (or dissipation) can.
    """
    active = (d < d0).astype(d.dtype)
    s = 1.0 / d - 1.0 / d0
    return k_p * s * (-1.0 / (d * d)) * active


def _pullback_diag(J, Jdq, m, f) -> Spec:
    """Pull back a *batch* of independent 1-D barrier specs sharing a diagonal task metric.

    For ``k`` scalar tasks stacked into ``(k,)`` — Jacobian ``J`` (k,nq), curvature ``Jdq`` (k,),
    diagonal task metric ``diag(m)`` and task force ``f`` (k,) — summing the per-task pullbacks
    ``Jᵢᵀ mᵢ Jᵢ`` and ``Jᵢᵀ(fᵢ + mᵢ Jdqᵢ)`` gives, without ever forming the dense ``(k,k)`` metric,

        M_q = Jᵀ diag(m) J = J.T @ (m[:, None] * J),      f_q = Jᵀ (f + m ⊙ Jdq).

    For ``k = 1`` this equals :func:`fabrix.spec.pullback` of the scalar barrier spec, so the single
    obstacle/plane/joint barriers are bit-identical; for ``k > 1`` it is the batched barrier (one FK +
    one Jacobian shared across all ``k`` distances — see :mod:`fabrix.collision`).
    """
    return Spec(J.T @ (m[:, None] * J), J.T @ (f + m * Jdq))


def joint_limit_geometry(provider, k_b: float = 0.4, power: float = 2.0, m_b: float = 1.0,
                         margin: float = 0.0, eps: float = 1e-3):
    """Barrier geometry that keeps each *limited* joint inside its range.

    For joint ``j`` with range ``[lo, hi]`` it runs two barriers — on ``d_lo = q_j - lo`` and
    ``d_hi = hi - q_j`` — each repelling when the joint approaches that bound. The task maps are
    the identity per joint, so this is a config-space (pre-pullback) spec: a diagonal metric and
    a per-joint acceleration, built fully vectorized over joints (no Python loop, no scalar pack).

    ``margin`` shifts the effective limit inward (stay-out band); ``eps`` floors the distance so
    the barrier stays large-but-finite at the boundary instead of producing NaNs.
    """
    m = provider.mj_model
    nq = provider.nq
    limited = jnp.asarray(m.jnt_limited.astype(bool))            # (nq,) which joints have a range
    lo = jnp.asarray(m.jnt_range[:, 0])
    hi = jnp.asarray(m.jnt_range[:, 1])
    # jnt order == qpos order for an all-hinge serial arm (1 dof/joint); assert to be safe.
    if not (np.array_equal(m.jnt_qposadr, np.arange(nq)) and m.nv == nq):
        raise ValueError("joint_limit_geometry assumes 1-dof joints with qpos order == joint order")

    def leaf(q, qd, params):
        dtype = q.dtype
        mask = limited.astype(dtype)
        d_lo = jnp.clip(q - lo - margin, eps, None)              # distance to lower bound (+ => q up)
        d_hi = jnp.clip(hi - q - margin, eps, None)              # distance to upper bound (+ => q down)
        dd_lo = qd                                               # d/dt d_lo
        dd_hi = -qd                                              # d/dt d_hi
        a_lo = _barrier_accel(d_lo, dd_lo, k_b, power)           # pushes q up  (+)
        a_hi = _barrier_accel(d_hi, dd_hi, k_b, power)           # pushes q down (-)
        a_geo = (a_lo - a_hi) * mask                             # (nq,) config-space accel
        diag = (_barrier_metric(d_lo, dd_lo, m_b)
                + _barrier_metric(d_hi, dd_hi, m_b)) * mask      # (nq,) metric weight
        M = jnp.diag(diag)
        f = -diag * a_geo                                        # f = -M a_des (M diagonal)
        return Spec(M, f)

    return leaf


def sdf_barrier_geometry(dist, k_b: float = 1.0, power: float = 2.0, m_b: float = 2.0,
                         d0: Optional[float] = None, margin: float = 0.0, eps: float = 1e-3):
    """Barrier *geometry* on any signed-distance field ``dist(q, params) -> (k,)``, ``k >= 1``.

    The shared core of obstacle/plane/self-collision avoidance: an HD2 barrier on each distance,
    repelling while the site approaches that surface, pulled back to config space through the distance
    Jacobian (with its curvature term). ``dist`` is passed ``params`` too, so a surface can move at
    runtime (a traced obstacle center) with no structural change — autodiff differentiates ``q`` only.

    When ``dist`` returns a vector (``k > 1``) this is a single **batched** barrier: one FK and one
    Jacobian feed all ``k`` distances, and the ``k`` per-task pullbacks are summed in closed form
    (:func:`_pullback_diag`) — the efficient way to run many barriers (e.g. collision-sphere pairs),
    far cheaper than ``k`` separate leaves. ``k = 1`` is the original single-surface barrier, unchanged.

    ``d0`` optionally localizes the deflection: the priority *metric* is faded out (smooth C1 band)
    beyond ``d0`` of the surface, so the geometry only bends the path within that range. Without it
    the ``m_b / d`` metric reaches (weakly) at every approaching distance, which makes the arm start
    detouring from far away. The HD2 acceleration is left untouched, so the path stays speed-independent.
    """

    def leaf(q, qd, params):
        phi = lambda qq: dist(qq, params)                       # noqa: E731  (q-only for autodiff)
        x, J, Jdq = value_jac_curv(phi, q, qd)                  # x:(k,) J:(k,nq) Jdq:(k,)
        d = jnp.clip(x - dynamic_gain(margin, params), eps, None)   # (k,)
        dd = J @ qd                                             # (k,) d/dt of each distance
        a_g = _barrier_accel(d, dd, dynamic_gain(k_b, params), power)   # (k,) accel away from surfaces
        m = _barrier_metric(d, dd, dynamic_gain(m_b, params))  # (k,) per-surface priority weight
        if d0 is not None:
            m = m * _band(d, dynamic_gain(d0, params))          # fade priority out beyond d0
        return _pullback_diag(J, Jdq, m, -m * a_g)              # = -M a_des, batched over k

    return leaf


def _sphere_dist(provider, center, radius, site_name):
    """Distance fn ``(q, params) -> (1,)`` for a sphere; ``center=None`` reads
    ``params.obstacle_center`` (a draggable / moving obstacle)."""
    if center is None:
        return lambda q, params: sphere_sdf_map(provider, params.obstacle_center, radius,
                                                site_name=site_name)(q)
    phi = sphere_sdf_map(provider, center, radius, site_name=site_name)
    return lambda q, params: phi(q)


def obstacle_geometry(provider, center, radius: float, k_b: float = 1.0, power: float = 2.0,
                      m_b: float = 2.0, d0: Optional[float] = None, margin: float = 0.0,
                      eps: float = 1e-3, site_name: Optional[str] = None):
    """Barrier geometry that keeps the tracked site outside a sphere obstacle.

    Task map is the signed distance ``d(q) = ||p_site(q) - center|| - radius``; the 1-D barrier
    repels when the site approaches the surface, pulled back through the SDF Jacobian (with its
    curvature term). ``center`` is a fixed point (baked, static obstacle) or ``None`` to read
    ``params.obstacle_center`` each step — a reactive moving/draggable-obstacle fabric. ``d0``
    localizes the deflection to within that distance of the surface (see :func:`sdf_barrier_geometry`).
    """
    return sdf_barrier_geometry(_sphere_dist(provider, center, radius, site_name),
                                k_b=k_b, power=power, m_b=m_b, d0=d0, margin=margin, eps=eps)


def plane_geometry(provider, point, normal, k_b: float = 1.0, power: float = 2.0, m_b: float = 2.0,
                   d0: Optional[float] = None, margin: float = 0.0, eps: float = 1e-3,
                   site_name: Optional[str] = None):
    """Barrier geometry keeping the tracked site on the ``+normal`` side of a plane (e.g. a floor).

    Distance is ``d(q) = normal . (p_site(q) - point)`` (see :func:`fabrix.maps.plane_sdf_map`).
    ``d0`` localizes the deflection to within that distance of the plane.
    """
    phi = plane_sdf_map(provider, point, normal, site_name=site_name)
    return sdf_barrier_geometry(lambda q, params: phi(q),
                                k_b=k_b, power=power, m_b=m_b, d0=d0, margin=margin, eps=eps)


# ---------------------------------------------------------------------------
# Barrier *potentials* (forcing leaves). These — not the energized geometries — are what make a
# constraint a hard invariant: the potential diverges at the boundary, so finite kinetic energy
# can never reach it. Pair each with its energized geometry for reactive, smooth avoidance.
# ---------------------------------------------------------------------------
def joint_limit_potential(provider, k_p: float = 0.05, d0: float = 0.3, m_p: float = 2.0,
                          margin: float = 0.0, eps: float = 1e-3):
    """Repulsive barrier potential keeping each *limited* joint inside its range (forcing leaf).

    A diverging potential on each joint's distance to its bounds; the identity task map per joint
    makes it a direct config-space spec, vectorized over joints. ``d0`` is the standoff band.
    """
    m = provider.mj_model
    nq = provider.nq
    limited = jnp.asarray(m.jnt_limited.astype(bool))
    lo = jnp.asarray(m.jnt_range[:, 0])
    hi = jnp.asarray(m.jnt_range[:, 1])
    if not (np.array_equal(m.jnt_qposadr, np.arange(nq)) and m.nv == nq):
        raise ValueError("joint_limit_potential assumes 1-dof joints with qpos order == joint order")

    def leaf(q, qd, params):
        dtype = q.dtype
        mask = limited.astype(dtype)
        d_lo = jnp.clip(q - lo - margin, eps, None)             # distance to lower bound
        d_hi = jnp.clip(hi - q - margin, eps, None)             # distance to upper bound
        f_lo = _barrier_potential_grad(d_lo, k_p, d0)           # dpsi/dd_lo  (d d_lo / dq = +1)
        f_hi = _barrier_potential_grad(d_hi, k_p, d0)           # dpsi/dd_hi  (d d_hi / dq = -1)
        f = (f_lo - f_hi) * mask
        band = _band(d_lo, d0) + _band(d_hi, d0)
        return Spec(jnp.diag(m_p * band * mask), f)

    return leaf


def sdf_barrier_potential(dist, k_p: float = 0.5, d0: float = 0.2, m_p: float = 4.0,
                          margin: float = 0.0, eps: float = 1e-3):
    """Repulsive barrier *potential* on any signed-distance field ``dist(q, params) -> (k,)``, ``k >= 1``.

    The shared core of the obstacle/plane/self-collision potentials: a diverging potential force on
    each distance, localized to the standoff band ``d0``. This — not the geometry — makes the clearance
    a hard invariant (the diverging potential is unreachable with finite kinetic energy). Pair with the
    matching geometry. ``dist`` is passed ``params`` so a surface can move at runtime; a vector ``dist``
    (``k > 1``) is one **batched** potential over all ``k`` distances (see :func:`sdf_barrier_geometry`).
    """

    def leaf(q, qd, params):
        phi = lambda qq: dist(qq, params)                       # noqa: E731  (q-only for autodiff)
        x, J, Jdq = value_jac_curv(phi, q, qd)                  # x:(k,) J:(k,nq) Jdq:(k,)
        d = jnp.clip(x - dynamic_gain(margin, params), eps, None)   # (k,)
        d0_ = dynamic_gain(d0, params)
        f = _barrier_potential_grad(d, dynamic_gain(k_p, params), d0_)  # (k,) diverging task forces
        m = dynamic_gain(m_p, params) * _band(d, d0_)           # (k,) smooth standoff envelope
        return _pullback_diag(J, Jdq, m, f)                     # batched over k

    return leaf


def obstacle_potential(provider, center, radius: float, k_p: float = 0.5, d0: float = 0.2,
                       m_p: float = 4.0, margin: float = 0.0, eps: float = 1e-3,
                       site_name: Optional[str] = None):
    """Repulsive barrier potential keeping the tracked site outside a sphere (forcing leaf).

    The position-only counterpart to :func:`obstacle_geometry`: its force diverges at the surface,
    so it can halt even a head-on approach — the leaf that makes non-penetration a hard invariant.
    ``center=None`` reads ``params.obstacle_center`` (draggable). Localized to the standoff band ``d0``.
    """
    return sdf_barrier_potential(_sphere_dist(provider, center, radius, site_name),
                                 k_p=k_p, d0=d0, m_p=m_p, margin=margin, eps=eps)


def plane_potential(provider, point, normal, k_p: float = 0.5, d0: float = 0.2, m_p: float = 4.0,
                    margin: float = 0.0, eps: float = 1e-3, site_name: Optional[str] = None):
    """Repulsive barrier potential keeping the tracked site on the ``+normal`` side of a plane."""
    phi = plane_sdf_map(provider, point, normal, site_name=site_name)
    return sdf_barrier_potential(lambda q, params: phi(q),
                                 k_p=k_p, d0=d0, m_p=m_p, margin=margin, eps=eps)
