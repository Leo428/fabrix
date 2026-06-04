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

import jax.numpy as jnp

from fabrix.diff import value_jac_curv
from fabrix.maps import site_position_map
from fabrix.spec import Spec, pullback


def attractor(provider, k: float = 16.0, b: float = 8.0, m: float = 50.0):
    """EE-position attractor. Desired accel ``-(k(x-x*) + b ẋ)``; priority metric ``m·I₃``.

    ``k`` sets stiffness (ω=√k), ``b`` damping (b=2√k is critical), ``m`` the task priority
    relative to posture/damping. ``f = M @ (k(x-x*) + b ẋ)`` pulled back to config space.
    """
    phi = site_position_map(provider)

    def leaf(q, qd, params):
        x, J, Jdq = value_jac_curv(phi, q, qd)
        xd = J @ qd
        M = m * jnp.eye(3, dtype=x.dtype)
        f = m * (k * (x - params.target) + b * xd)  # = -M @ a_des
        return pullback(Spec(M, f), J, Jdq)

    return leaf


def posture(nq: int, k: float = 1.0, b: float = 2.0, weight: float = 0.5):
    """Config-space attractor toward a nominal posture; low priority → acts in the nullspace."""

    def leaf(q, qd, params):
        M = weight * jnp.eye(nq, dtype=q.dtype)
        f = weight * (k * (q - params.q_default) + b * qd)  # = -M @ a_des
        return Spec(M, f)

    return leaf


def config_damping(nq: int, b: float = 2.0, mass: float = 1.0):
    """Pure joint-space damping: global dissipation + a full-rank metric contribution."""

    def leaf(q, qd, params):
        M = mass * jnp.eye(nq, dtype=q.dtype)
        f = mass * b * qd  # = -M @ (-b q̇)  → isolated accel -b·q̇
        return Spec(M, f)

    return leaf
