"""High-level fabric assembly: a set of leaves over a shared configuration space.

``Fabric.policy(q, qd, params) -> qddot`` evaluates every leaf, combines their config-space
Specs, and resolves the root Spec. Exactly one ``jax.jit`` boundary — the control step — so a
real-time loop pays a single dispatch per tick.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from fabrix.geometry import energize
from fabrix.spec import Spec, combine, resolve


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


class GeometricFabric:
    """A full geometric fabric: energized HD2 geometries + forcing potentials + damping.

    Each control step:

      1. evaluate the geometry leaves (barriers), combine + resolve to the **root geometry
         acceleration** ``a_g`` in config space (metric-weighted, so the about-to-be-violated
         constraint dominates);
      2. **energize** ``a_g`` against the execution energy -> ``a_e`` (energy-conserving,
         path-preserving), emitted as the spec ``(M_e, -M_e a_e)``;
      3. add the forcing leaves (the attractor that drives to goal) and damping, combine + resolve
         to ``qddot``.

    Geometries are energized at the root because a single barrier's 1-D leaf space leaves no room
    for the energy-conserving projection (see :mod:`fabrix.geometry`). With no geometries and the
    fixed-metric energy this reduces to the forcing+damping fabric of M1.

    ``geometries`` / ``forcing`` / ``damping`` are leaf sequences (``leaf(q, qd, params) -> Spec``);
    ``energy`` is a callable ``energy(q, qd) -> (M_e, f_e)`` (see :mod:`fabrix.energy`).
    """

    def __init__(self, *, geometries=(), forcing=(), damping=(), energy,
                 reg: float = 1e-6, geom_reg: float = 1e-4):
        self.geometries = tuple(geometries)  # static structure (unrolls at trace time)
        self.forcing = tuple(forcing)
        self.damping = tuple(damping)
        self.energy = energy
        self.reg = float(reg)
        # geom_reg regularizes the (often rank-deficient) barrier-metric solve for a_g; it must sit
        # comfortably above float32 eps (~1e-7), else a small barrier metric amplifies float32 noise
        # into q_ddot chatter. 1e-4 is float32-safe and does not change the active-geometry result.
        self.geom_reg = float(geom_reg)
        self.policy = jax.jit(self._policy)

    def _policy(self, q, qd, params):
        # 1. combined root geometry acceleration (zero if there are no geometries)
        if self.geometries:
            a_g = resolve(combine([g(q, qd, params) for g in self.geometries]), self.geom_reg)
        else:
            a_g = jnp.zeros_like(q)
        # 2. energize against the execution energy -> path-preserving, energy-conserving accel
        M_e, f_e = self.energy(q, qd)
        a_e = energize(a_g, qd, M_e, f_e)
        geo = Spec(M_e, -M_e @ a_e)
        # 3. add forcing + damping and resolve
        specs = [geo]
        specs += [leaf(q, qd, params) for leaf in self.forcing]
        specs += [leaf(q, qd, params) for leaf in self.damping]
        return resolve(combine(specs), self.reg)
