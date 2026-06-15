"""Semi-implicit Euler integration and a ``lax.scan`` rollout of a fabric policy.

Semi-implicit (symplectic) Euler updates velocity first, then position with the new velocity —
better behaved for second-order systems than explicit Euler. ``rollout`` runs the closed loop
inside a single ``lax.scan`` (one compiled loop, no Python iteration) and returns the stacked
commanded trajectory, whose derivatives we plot to show C2 smoothness.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def limit_accel(qdd, a_max):
    """Bound the acceleration to ``a_max`` by **direction-preserving** uniform scaling.

    Unlike a per-axis clip (``clip(qdd, -a_max, a_max)``), which bends the commanded direction when one
    joint saturates — distorting the fabric's path — this scales the whole vector by the single factor
    that brings the most-saturated joint to ``a_max``, so the path direction is preserved. A no-op when
    ``max|qdd_i| <= a_max`` (the common case). ``a_max`` is a scalar.
    """
    peak = jnp.max(jnp.abs(qdd))
    return qdd * jnp.minimum(1.0, a_max / (peak + 1e-12))


def limit_jerk(qdd, qdd_prev, dqdd_max):
    """Rate-limit the per-step change in acceleration to ``±dqdd_max`` (a per-tick jerk bound).

    Caps ``|qdd − qdd_prev|`` per joint, so the commanded acceleration cannot step discontinuously —
    gentler starts/stops and a bounded τ feedforward near a person. ``dqdd_max`` is the max ``|Δqdd|``
    per call (i.e. ``jerk_max · dt`` for a fixed-rate loop); a huge value is a no-op. The caller carries
    ``qdd_prev`` across ticks.
    """
    return qdd_prev + jnp.clip(qdd - qdd_prev, -dqdd_max, dqdd_max)


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
