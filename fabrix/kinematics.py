"""Kinematics provider: ``q -> site world position`` as a differentiable JAX function.

:class:`CustomFK` is a lean vectorized serial-chain FK (the real-time provider; ~8 us / FK,
~27 us / curvature term), built purely from the model's joint frames with no scalar packing. It
is differentiable, so ``J`` and ``Jdot @ qdot`` come from autodiff (see ``fabrix.diff``).

It sits behind the :class:`KinematicsProvider` Protocol, so an alternative backend (e.g. an MJX
wrapper for non-serial models or batched/GPU use) can be dropped in without touching the fabric.
A provider is constructed once and closed over by the fabric; ``self`` is static under jit and the
model arrays bake into the compiled graph.
"""
from __future__ import annotations

from typing import Optional, Protocol

import jax.numpy as jnp
import mujoco
import numpy as np


class KinematicsProvider(Protocol):
    """Minimal kinematics interface used by task maps (M3 will add ``site_pose``)."""

    mj_model: mujoco.MjModel
    site_id: int
    nq: int

    def site_pos(self, q: jnp.ndarray) -> jnp.ndarray:
        """World position (3,) of the tracked site for configuration ``q`` (nq,)."""
        ...


def _resolve_site(model: mujoco.MjModel, site_name: Optional[str]) -> int:
    if site_name is None:
        return model.nsite - 1
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)


# ---------------------------------------------------------------------------
# Vectorized quaternion helpers (NO jnp.array([scalar, ...]) — that anti-pattern
# compiles to thousands of scalar ops and was a ~1000x slowdown).
# Quaternions are (4,) arrays in MuJoCo [w, x, y, z] convention.
# ---------------------------------------------------------------------------
def _qmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    w1, v1 = a[0], a[1:]
    w2, v2 = b[0], b[1:]
    w = w1 * w2 - jnp.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + jnp.cross(v1, v2)
    return jnp.concatenate([w[None], v])


def _qrot(q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    w, u = q[0], q[1:]
    t = 2.0 * jnp.cross(u, p)
    return p + w * t + jnp.cross(u, t)


def _axisangle_quat(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    axis = axis / jnp.linalg.norm(axis)
    h = 0.5 * angle
    return jnp.concatenate([jnp.cos(h)[None], jnp.sin(h) * axis])


class CustomFK:
    """Lean serial-chain forward kinematics, pure JAX. Real-time provider.

    Supports bodies with zero or one hinge joint (covers serial arms like the Gen3). The
    body-tree loop unrolls over the model's *static* topology, so the compiled graph is a
    flat sequence of vectorized ops.
    """

    def __init__(self, xml_path: str, site_name: Optional[str] = None, dtype=jnp.float32):
        m = mujoco.MjModel.from_xml_path(xml_path)
        site = _resolve_site(m, site_name)
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)

        # static topology (numpy) + constant frames (jnp, baked into the graph)
        self._parent = np.array(m.body_parentid)
        self._jntnum = np.array(m.body_jntnum)
        self._jntadr = np.array(m.body_jntadr)
        self._jnt_type = np.array(m.jnt_type)
        self._jqadr = np.array(m.jnt_qposadr)
        self._nbody = m.nbody
        self._hinge = hinge
        self._site_body = int(m.site_bodyid[site])

        bpos = jnp.array(m.body_pos, dtype)
        bquat = jnp.array(m.body_quat, dtype)
        jaxis = jnp.array(m.jnt_axis, dtype)
        jpos = jnp.array(m.jnt_pos, dtype)
        spos = jnp.array(m.site_pos[site], dtype)
        ident_q = jnp.array([1.0, 0.0, 0.0, 0.0], dtype)
        zero3 = jnp.zeros(3, dtype)

        def fk(q):
            pw = [zero3] * self._nbody          # world position of each body frame
            qw = [ident_q] * self._nbody        # world orientation (quat) of each body frame
            for b in range(1, self._nbody):
                p = int(self._parent[b])
                wp = pw[p] + _qrot(qw[p], bpos[b])
                wq = _qmul(qw[p], bquat[b])
                if self._jntnum[b] == 1 and self._jnt_type[self._jntadr[b]] == hinge:
                    ja = int(self._jntadr[b])
                    jq = _axisangle_quat(jaxis[ja], q[self._jqadr[ja]])
                    anchor = jpos[ja]
                    wp = wp + _qrot(wq, anchor - _qrot(jq, anchor))
                    wq = _qmul(wq, jq)
                pw[b] = wp
                qw[b] = wq
            return pw[self._site_body] + _qrot(qw[self._site_body], spos)

        self._fk = fk
        self.mj_model = m
        self.site_id = site
        self.nq = m.nq

    def site_pos(self, q: jnp.ndarray) -> jnp.ndarray:
        return self._fk(q)
