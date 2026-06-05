"""Leaf factories: each returns ``leaf(q, qd, params) -> Spec`` in configuration space.

Every leaf follows one pattern: define a **desired acceleration** ``a_des`` and a **priority
metric** ``M``, and emit force ``f = -M @ a_des`` so the leaf's *isolated* acceleration is
exactly ``a_des`` while ``M`` sets how strongly the leaf wins the metric-weighted combination.
This is the RMP/fabric design: the task attractor carries a large metric so it dominates in the
task directions, while posture/damping act where the task metric is small (the nullspace) —
so redundancy is resolved without biasing the end-effector equilibrium.

Task-space leaves (``attractor``) are pulled back to configuration space; config-space leaves
(``posture``, ``config_damping``) are identity-map leaves returned directly. M1 leaves are
*forced* (metric + potential-gradient + damping); HD2 geometries + energization arrive in M2.
Gains are baked at construction (mink-style); ``params.target``/``params.q_default`` are traced.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from fabrix.diff import value_jac_curv
from fabrix.maps import se3_pose_error_map, site_position_map
from fabrix.spec import Spec, dynamic_gain, pullback


def _restoring(e, k: float, f_max: Optional[float], eps: float = 1e-3):
    """Restoring 'force' of an attractor potential as a function of the task error ``e``.

    ``f_max=None`` → quadratic potential: gradient ``k·e`` (force grows without bound, so a far
    target commands a large acceleration). Otherwise a **gradient-saturating** potential: magnitude
    ``k·‖e‖`` near the goal (same stiffness/feel as the quadratic) but capped at ``f_max`` far away,
    so the commanded acceleration stays bounded no matter how distant the target — gentle, lunge-free
    large moves. ``f(e) = f_max·tanh(k‖e‖/f_max)·ê``; the softened norm keeps it smooth at ``e=0``.
    """
    if f_max is None:
        return k * e
    r = jnp.sqrt(e @ e + eps * eps)              # softened norm: smooth + NaN-free at e=0
    return f_max * jnp.tanh(k * r / f_max) * (e / r)


def attractor(provider, k: float = 16.0, b: float = 8.0, m: float = 50.0,
              f_max: Optional[float] = None):
    """EE-position attractor. Desired accel ``-(g(x-x*) + b ẋ)``; priority metric ``m·I₃``.

    ``k`` sets stiffness (ω=√k), ``b`` damping (b=2√k is critical), ``m`` the task priority
    relative to posture/damping. ``g`` is the restoring force: quadratic (``k·e``) by default, or
    gradient-saturating with magnitude capped at ``f_max`` (see :func:`_restoring`).
    """
    phi = site_position_map(provider)

    def leaf(q, qd, params):
        x, J, Jdq = value_jac_curv(phi, q, qd)
        xd = J @ qd
        k_, b_, m_, fm_ = (dynamic_gain(g, params) for g in (k, b, m, f_max))
        M = m_ * jnp.eye(3, dtype=x.dtype)
        f = m_ * (_restoring(x - params.target, k_, fm_) + b_ * xd)  # = -M @ a_des
        return pullback(Spec(M, f), J, Jdq)

    return leaf


def pose_attractor(provider, k: float = 16.0, b: float = 8.0, m: float = 50.0,
                   f_max: Optional[float] = None):
    """Full 6-DOF SE(3) pose attractor. Drives the coupled pose error to zero.

    Task = ``e(q) = Log(T*^{-1} T(q)) in se(3)`` (``params.target`` position + ``params.target_quat``
    orientation, wxyz). Desired accel ``-(g(e) + b ė)``; priority metric ``m·I₆`` (one shared metric
    over the 6 twist coordinates, the coupled-SE(3) choice). ``g`` is the restoring force: quadratic
    by default, or gradient-saturating capped at ``f_max`` (bounded accel on far/commanded moves; note
    one ``f_max`` mixes the translation (m) and rotation (rad) scales of the 6-D twist). Use alongside
    :func:`posture`/:func:`config_damping` to resolve redundancy; the approach is a geodesic screw.
    """

    def leaf(q, qd, params):
        phi = se3_pose_error_map(provider, params.target, params.target_quat)
        e, J, Jdq = value_jac_curv(phi, q, qd)
        ed = J @ qd
        k_, b_, m_, fm_ = (dynamic_gain(g, params) for g in (k, b, m, f_max))
        M = m_ * jnp.eye(6, dtype=e.dtype)
        f = m_ * (_restoring(e, k_, fm_) + b_ * ed)  # = -M @ a_des
        return pullback(Spec(M, f), J, Jdq)

    return leaf


def posture(nq: int, k=1.0, b: float = 2.0, weight=0.5):
    """Config-space attractor toward ``params.q_default``; low priority → acts in the nullspace.

    Resolves the arm's redundancy toward a nominal (e.g. upright/compact) posture without biasing the
    EE. ``weight`` and ``k`` may be scalars **or per-joint ``(nq,)`` arrays** — use a per-joint
    ``weight`` to hold the uprightness-critical joints (shoulder/elbow) firmly toward ``q_default``
    while leaving the wrist free for the task. Per-joint ``weight`` cancels from the isolated
    acceleration, so it only re-weights priority (which joints win the spare DOF), never the target.

    Ceiling worth knowing: posture acts only in the *task nullspace*. A full 6-DOF pose task leaves a
    7-DOF arm just 1 spare DOF (the elbow swivel), so posture's reach there is small; a position-only
    (or low-orientation-weight) task frees 4 DOF and posture becomes far more effective. Hard
    "never enter this pose" guarantees come from joint-limit barriers, not posture.
    """
    def leaf(q, qd, params):
        w = jnp.asarray(dynamic_gain(weight, params))
        kk = jnp.asarray(dynamic_gain(k, params))
        b_ = dynamic_gain(b, params)
        wv = jnp.broadcast_to(w, (nq,)).astype(q.dtype)
        M = jnp.diag(wv)
        f = wv * (kk * (q - params.q_default) + b_ * qd)  # = -M @ a_des (per-joint weight cancels in a_des)
        return Spec(M, f)

    return leaf


def config_damping(nq: int, b: float = 2.0, mass: float = 1.0):
    """Pure joint-space damping: global dissipation + a full-rank metric contribution."""

    def leaf(q, qd, params):
        mass_, b_ = dynamic_gain(mass, params), dynamic_gain(b, params)
        M = mass_ * jnp.eye(nq, dtype=q.dtype)
        f = mass_ * b_ * qd  # = -M @ (-b q̇)  → isolated accel -b·q̇
        return Spec(M, f)

    return leaf
