"""High-level fabric assembly: a set of leaves over a shared configuration space.

``Fabric.policy(q, qd, params) -> qddot`` evaluates every leaf, combines their config-space
Specs, and resolves the root Spec. Exactly one ``jax.jit`` boundary — the control step — so a
real-time loop pays a single dispatch per tick.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from fabrix.geometry import energize
from fabrix.spec import Spec, combine, dynamic_gain, resolve


@jdc.pytree_dataclass
class FabricParams:
    """Runtime (traced) parameters; change between steps without recompiling."""

    target: jnp.ndarray     # (3,) end-effector position target
    q_default: jnp.ndarray  # (nq,) nominal posture for redundancy resolution
    # (4,) wxyz orientation target for the SE(3) pose_attractor; identity default so
    # position-only fabrics (M1/M2) need not supply it.
    target_quat: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0]))
    # (3,) center of a draggable sphere obstacle (read by obstacle leaves built with center=None);
    # far-away default so an unset/unused obstacle leaf stays inert.
    obstacle_center: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array([0.0, 0.0, 100.0]))
    # Optional live-tuning gains: any pytree (e.g. a dataclass / NamedTuple of scalars) read by
    # leaves built with *callable* gains (``lambda p: p.gains...``). ``None`` when gains are baked at
    # construction (the M1–M3 default), so this stays fully backward-compatible — see
    # :func:`fabrix.spec.dynamic_gain`.
    gains: Any = None
    # (nq,) integrated REFERENCE velocity q̇_ref, consumed only by :func:`fabrix.leaves.reference_damping`
    # (NVlabs' cspace_damping on the reference). The closed-loop fabric is evaluated on the MEASURED q̇,
    # so the control node passes its integrator's q̇_ref here for the one leaf that must damp the
    # reference — not the measured — velocity. ``None`` ⇒ that leaf is inert (every other fabric).
    qd_ref: Any = None


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

    def __init__(self, *, geometries=(), forcing=(), damping=(), energy, ref_damp=None,
                 reg: float = 1e-6, geom_reg: float = 1e-4):
        self.geometries = tuple(geometries)  # static structure (unrolls at trace time)
        self.forcing = tuple(forcing)
        self.damping = tuple(damping)
        self.energy = energy
        # Optional NVlabs-style reference-velocity damping (cspace_damping on q̇_ref), a scalar or a
        # callable gain ``lambda p: p.gains...``. Applied POST-combine (see _policy) so it cancels the
        # metric and adds exactly −b·q̇_ref — the closed-loop placement of the damping NVlabs' open-loop
        # fabric applies to its own integrated reference. ``None`` ⇒ off (every M1–M3 fabric).
        self.ref_damp = ref_damp
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
        # 3. add forcing + damping, combine
        specs = [geo]
        specs += [leaf(q, qd, params) for leaf in self.forcing]
        specs += [leaf(q, qd, params) for leaf in self.damping]
        root = combine(specs)
        # 4. NVlabs cspace_damping on the REFERENCE velocity (fabrics_sim fabric.py:521,
        # ``force += gain·M·q̇``), applied to the COMBINED metric so it cancels: the accel gains exactly
        # −b·q̇_ref (unweighted), matching the reference damping NVlabs' integrator feeds back open-loop.
        # ``q̇_ref`` is None for fabrics that don't stream a reference ⇒ this is skipped (static branch).
        if self.ref_damp is not None and params.qd_ref is not None:
            b_ref = dynamic_gain(self.ref_damp, params)
            root = Spec(root.M, root.f + b_ref * (root.M @ params.qd_ref))
        return resolve(root, self.reg)
