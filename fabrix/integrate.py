"""Semi-implicit Euler integration and a ``lax.scan`` rollout of a fabric policy.

Semi-implicit (symplectic) Euler updates velocity first, then position with the new velocity —
better behaved for second-order systems than explicit Euler. ``rollout`` runs the closed loop
inside a single ``lax.scan`` (one compiled loop, no Python iteration) and returns the stacked
commanded trajectory, whose derivatives we plot to show C2 smoothness.
"""
from __future__ import annotations

import jax


def step(policy, q, qd, params, dt):
    """One semi-implicit Euler step. Returns ``(q_next, qd_next, qddot)``."""
    qddot = policy(q, qd, params)
    qd = qd + dt * qddot
    q = q + dt * qd
    return q, qd, qddot


def rollout(policy, q0, qd0, params, dt, steps, ee_fn):
    """Roll the policy for ``steps`` steps via ``lax.scan``.

    ``policy`` and ``ee_fn`` are pure functions of ``(q, qd, params)`` / ``(q,)``; ``dt`` and
    ``steps`` are static. Returns a dict of stacked arrays: ``q, qd, qdd`` (steps, nq) and
    ``ee`` (steps, 3). The whole loop compiles once.
    """

    def body(carry, _):
        q, qd = carry
        qddot = policy(q, qd, params)
        qd = qd + dt * qddot
        q = q + dt * qd
        return (q, qd), (q, qd, qddot, ee_fn(q))

    _, (qs, qds, qdds, ees) = jax.lax.scan(body, (q0, qd0), xs=None, length=steps)
    return {"q": qs, "qd": qds, "qdd": qdds, "ee": ees}
