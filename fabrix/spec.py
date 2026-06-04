"""Spec algebra for geometric fabrics.

A *Spec* is a second-order specification ``(M, f)`` on some space, encoding the dynamics

    M @ qddot + f = 0      =>      qddot = -solve(M, f).

Leaves emit Specs in their task space; :func:`pullback` maps a Spec to configuration
space through a task map; :func:`combine` sums Specs that share a space (the
metric-weighted combination of accelerations); :func:`resolve` solves for the
acceleration. This is the algebra every fabric is assembled from.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class Spec:
    """Second-order spec ``M @ qddot + f = 0``. A JAX pytree (M, f are leaves)."""

    M: jnp.ndarray  # (n, n) metric — SPD once posture/damping are combined in
    f: jnp.ndarray  # (n,) force


def pullback(spec: Spec, J: jnp.ndarray, Jdq: jnp.ndarray) -> Spec:
    """Pull a leaf-space Spec back through a task map to configuration space.

    With ``x = phi(q)``, ``xdot = J qdot``, ``xddot = J qddot + Jdot qdot``, substituting
    into ``M xddot + f = 0`` and projecting with ``Jᵀ`` gives

        M_q = Jᵀ M J,      f_q = Jᵀ (f + M @ Jdq),

    where ``Jdq = Jdot @ qdot`` is the task-space curvature term. Including ``M @ Jdq`` is
    what makes the pullback correct — dropping it is the classic RMP/fabric bug.
    """
    MJ = spec.M @ J
    M_q = J.T @ MJ
    f_q = J.T @ (spec.f + spec.M @ Jdq)
    return Spec(M_q, f_q)


def combine(specs) -> Spec:
    """Sum Specs defined on the same space. ``specs`` is a static-length sequence."""
    M, f = specs[0].M, specs[0].f
    for s in specs[1:]:
        M = M + s.M
        f = f + s.f
    return Spec(M, f)


def resolve(spec: Spec, reg: float = 1e-6) -> jnp.ndarray:
    """Solve ``M qddot + f = 0`` for ``qddot`` with Tikhonov regularization.

    The combined root metric is SPD (posture + config-damping make it full-rank), so we
    solve via Cholesky rather than forming an inverse.
    """
    n = spec.M.shape[0]
    M = spec.M + reg * jnp.eye(n, dtype=spec.M.dtype)
    return -jsl.cho_solve(jsl.cho_factor(M), spec.f)
