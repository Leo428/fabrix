"""Differentiation helpers: value, Jacobian, and the curvature term ``Jdot @ qdot``.

The curvature term is the one piece of fabric math that is painful to hand-derive. Here it
falls out of a nested ``jvp`` (forward-over-forward), validated to ~1e-10 against finite
differences in ``experiments/``. This module is the single home for that pattern, reused by
every leaf via the task-map pullback.
"""
from __future__ import annotations

import jax


def value_jac_curv(phi, q, qd):
    """For ``x = phi(q)`` return ``(x, J, Jdq)``.

    - ``J = dphi/dq``                       (shape: out x nq), via forward-mode autodiff
    - ``Jdq = Jdot @ qd``                   the task-space curvature, with qddot = 0

    ``Jdq`` is the second directional derivative of ``phi`` along ``qd``:
    ``Jdq = d/deps [ J(q + eps*qd) @ qd ]|_0``, i.e. ``jvp`` of (``q -> J(q) @ qd``).
    ``phi`` must be a function of ``q`` alone (close over any params before calling).
    """
    x = phi(q)
    J = jax.jacfwd(phi)(q)

    def jac_times_qd(qq):
        # forward-mode directional derivative: J(qq) @ qd
        return jax.jvp(phi, (qq,), (qd,))[1]

    Jdq = jax.jvp(jac_times_qd, (q,), (qd,))[1]
    return x, J, Jdq
