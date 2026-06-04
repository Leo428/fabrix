"""High-level fabric assembly: a set of leaves over a shared configuration space.

``Fabric.policy(q, qd, params) -> qddot`` evaluates every leaf, combines their config-space
Specs, and resolves the root Spec. Exactly one ``jax.jit`` boundary — the control step — so a
real-time loop pays a single dispatch per tick.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from fabrix.spec import combine, resolve


@jdc.pytree_dataclass
class FabricParams:
    """Runtime (traced) parameters; change between steps without recompiling."""

    target: jnp.ndarray     # (3,) end-effector position target
    q_default: jnp.ndarray  # (nq,) nominal posture for redundancy resolution


class Fabric:
    """A geometric fabric assembled from leaves sharing one configuration space."""

    def __init__(self, leaves, reg: float = 1e-6):
        self.leaves = tuple(leaves)  # static structure (length unrolls at trace time)
        self.reg = float(reg)
        self.policy = jax.jit(self._policy)

    def _policy(self, q, qd, params):
        specs = [leaf(q, qd, params) for leaf in self.leaves]
        return resolve(combine(specs), self.reg)
