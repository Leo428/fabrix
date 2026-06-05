"""fabrix collision-sphere test suite: per-link FK, the batched barrier core, and the invariants.

Claims under test:
  - ``CustomFK.body_poses`` reproduces every MuJoCo body frame (position + orientation) exactly;
  - autodiff ``J``/``Jdq`` through [all-links FK -> sphere placement -> pairwise SDF] match finite
    differences (the self-collision map is differentiated correctly and stays finite);
  - the generalized (vector) barrier core is **exactly** ``k`` separate single-barrier leaves summed —
    so batching is free of behavioral change, and ``k=1`` equals the original ``pullback`` (M2 regression);
  - the potentials are hard invariants: a fabric driven toward a self-colliding / link-obstacle-hitting
    configuration does **not** penetrate, whereas the same drive without the barrier does;
  - the full collision fabric stays real-time.

Math/behavior gates use x64 for clean margins; invariants + latency run the float32 deployment path.
"""
import pathlib
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede array creation

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from fabrix import (
    CustomFK, FabricParams, GeometricFabric, arm_obstacle_geometry, arm_obstacle_potential,
    arm_plane_geometry, arm_plane_potential, auto_arm_spheres, config_damping, fixed_metric_energy,
    nonadjacent_pairs, pose_attractor, posture, rollout, self_collision_geometry,
    self_collision_potential,
)
from fabrix.collision import SphereModel, _body_id, _centers
from fabrix.diff import value_jac_curv
from fabrix.geometry import _pullback_diag
from fabrix.spec import Spec, combine, pullback

XML = str(pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie/kinova_gen3/gen3.xml")
Q_FOLD = jnp.asarray([0.0, 2.0, 0.0, 2.8, 0.0, 1.5, 0.0])   # a clearly self-colliding config (~-73 mm)


@pytest.fixture(scope="module")
def prov():
    return CustomFK(XML, dtype=jnp.float64)


@pytest.fixture(scope="module")
def prov32():
    return CustomFK(XML, dtype=jnp.float32)


def _params(q_default, dtype=jnp.float64, obstacle=(0.0, 0.0, 100.0)):
    nq = q_default.shape[0]
    return FabricParams(target=jnp.zeros(3, dtype), q_default=jnp.asarray(q_default, dtype),
                        target_quat=jnp.asarray([1.0, 0, 0, 0], dtype),
                        obstacle_center=jnp.asarray(obstacle, dtype))


def _self_dist_phi(prov, sph, pairs):
    """The self-collision distance map ``phi(q) -> (Npairs,)`` (the leaf's ``dist``, sans params)."""
    link, local, rad = jnp.asarray(sph.link), jnp.asarray(sph.local), jnp.asarray(sph.radius)
    pa, pb = jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])
    rsum = rad[pa] + rad[pb]
    cf = _centers(prov, link, local)

    def phi(q):
        c = cf(q)
        diff = c[pa] - c[pb]
        return jnp.sqrt(jnp.sum(diff * diff, axis=1) + 1e-12) - rsum

    return phi


def _self_gaps(prov, sph, pairs, qs):
    """Min over time of every pair gap; ``qs`` is ``(T, nq)``. Returns the ``(T, Npairs)`` gap array."""
    cf = _centers(prov, jnp.asarray(sph.link), jnp.asarray(sph.local))
    C = jax.vmap(cf)(qs)                                   # (T, Ns, 3)
    pa, pb = pairs[:, 0], pairs[:, 1]
    d = jnp.linalg.norm(C[:, pa] - C[:, pb], axis=2)       # (T, Npairs)
    return np.asarray(d - (sph.radius[pa] + sph.radius[pb]))


# ---------------- per-link forward kinematics (#8) ----------------
def test_body_poses_matches_mujoco(prov):
    m = prov.mj_model
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    max_pos, max_rot = 0.0, 0.0
    for _ in range(20):
        q = rng.uniform(-2.0, 2.0, prov.nq)
        d.qpos[:prov.nq] = q
        mujoco.mj_forward(m, d)
        P, Q = prov.body_poses(jnp.asarray(q))
        assert P.shape == (m.nbody, 3) and Q.shape == (m.nbody, 4)
        max_pos = max(max_pos, float(jnp.max(jnp.abs(P - d.xpos))))
        for b in range(m.nbody):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.asarray(Q[b]))
            max_rot = max(max_rot, float(np.max(np.abs(R.reshape(3, 3) - d.xmat[b].reshape(3, 3)))))
    assert max_pos < 1e-9, f"body position mismatch {max_pos:.2e}"
    assert max_rot < 1e-9, f"body orientation mismatch {max_rot:.2e}"


# ---------------- collision-map autodiff ----------------
def test_self_collision_jac_curv_finite_diff(prov):
    sph = auto_arm_spheres(prov, 2)
    pairs = nonadjacent_pairs(sph, prov)
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.uniform(-1, 1, prov.nq))
    qd = jnp.asarray(rng.uniform(-1, 1, prov.nq))
    phi = _self_dist_phi(prov, sph, pairs)
    x, J, Jdq = value_jac_curv(phi, q, qd)
    assert x.shape == (len(pairs),) and J.shape == (len(pairs), prov.nq)
    assert bool(jnp.all(jnp.isfinite(J))) and bool(jnp.all(jnp.isfinite(Jdq)))
    eps = 1e-6
    v = jnp.asarray(rng.uniform(-1, 1, prov.nq))
    fd_J = (phi(q + eps * v) - phi(q - eps * v)) / (2 * eps)
    assert float(jnp.linalg.norm(fd_J - J @ v)) < 1e-7
    fd_Jdq = (jax.jacfwd(phi)(q + eps * qd) @ qd - jax.jacfwd(phi)(q - eps * qd) @ qd) / (2 * eps)
    assert float(jnp.linalg.norm(fd_Jdq - Jdq)) < 1e-7


# ---------------- batched core == separate leaves ----------------
def test_batched_equals_separate_leaves(prov):
    # The whole point of the batched leaf: one k-vector barrier must equal k single-barrier leaves
    # summed -- bit-for-bit (same FK, same pullback math), so batching changes cost, never behavior.
    sph = auto_arm_spheres(prov, 2)
    pairs = nonadjacent_pairs(sph, prov)[:10]
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.uniform(-0.5, 0.5, prov.nq))
    qd = jnp.asarray(rng.uniform(-0.5, 0.5, prov.nq))
    p = _params(q)
    batched = self_collision_geometry(prov, sph, pairs, d0=None)(q, qd, p)
    separate = combine([self_collision_geometry(prov, sph, pairs[i:i + 1], d0=None)(q, qd, p)
                        for i in range(len(pairs))])
    assert float(jnp.max(jnp.abs(batched.M - separate.M))) < 1e-9
    assert float(jnp.max(jnp.abs(batched.f - separate.f))) < 1e-9


def test_pullback_diag_matches_pullback_k1(prov):
    # k=1 regression: the generalized diagonal pullback must equal fabrix.spec.pullback of the scalar
    # barrier spec, so the single obstacle/plane/joint barriers (and all of M2) are unchanged.
    nq = prov.nq
    rng = np.random.default_rng(3)
    J = jnp.asarray(rng.uniform(-1, 1, (1, nq)))
    Jdq = jnp.asarray(rng.uniform(-1, 1, (1,)))
    m = jnp.asarray(rng.uniform(0.1, 2.0, (1,)))
    f = jnp.asarray(rng.uniform(-1, 1, (1,)))
    a = _pullback_diag(J, Jdq, m, f)
    b = pullback(Spec(m.reshape(1, 1), f), J, Jdq)
    assert float(jnp.max(jnp.abs(a.M - b.M))) < 1e-12
    assert float(jnp.max(jnp.abs(a.f - b.f))) < 1e-12


# ---------------- invariants: the potential is a hard wall ----------------
def test_self_collision_prevents_penetration(prov32):
    # Drive the arm toward a self-colliding posture. WITH the self-collision barrier it must not
    # penetrate; the same drive WITHOUT it must penetrate (proving the scenario is non-trivial).
    prov = prov32
    nq = prov.nq
    sph = auto_arm_spheres(prov, 2)
    pairs = nonadjacent_pairs(sph, prov)
    q0 = jnp.asarray(prov.mj_model.key_qpos[0, :nq], jnp.float32)
    params = _params(Q_FOLD, jnp.float32)                  # posture target = the colliding config
    energy = fixed_metric_energy(nq, jnp.float32)
    pull = [posture(nq, k=4.0, b=4.0, weight=5.0)]
    damp = [config_damping(nq, b=8.0)]
    fab = GeometricFabric(geometries=[self_collision_geometry(prov, sph, pairs, d0=0.06)],
                          forcing=pull + [self_collision_potential(prov, sph, pairs, k_p=0.5, d0=0.06, m_p=6.0)],
                          damping=damp, energy=energy)
    tr = rollout(fab.policy, q0, jnp.zeros(nq, jnp.float32), params, 0.002, 2500, prov.site_pos)
    assert bool(jnp.all(jnp.isfinite(tr["q"])))
    gaps = _self_gaps(prov, sph, pairs, tr["q"][::25])
    assert gaps.min() > -5e-3, f"self-collision: penetrated {gaps.min()*1e3:.1f} mm"

    ctrl = GeometricFabric(forcing=pull, damping=damp, energy=energy)   # no barrier
    tr0 = rollout(ctrl.policy, q0, jnp.zeros(nq, jnp.float32), params, 0.002, 2500, prov.site_pos)
    gaps0 = _self_gaps(prov, sph, pairs, tr0["q"][::25])
    assert gaps0.min() < -0.01, f"control did not self-collide ({gaps0.min()*1e3:.1f} mm) — test is vacuous"


def test_arm_obstacle_protects_whole_arm(prov32):
    # An EE reach whose natural arm motion sweeps a FOREARM sphere through an off-path obstacle. WITH
    # the whole-arm barrier the forearm deflects around it (positive clearance) while the EE still
    # reaches; WITHOUT it the EE-only fabric ignores the obstacle and the forearm rams through it.
    prov = prov32
    nq = prov.nq
    sph = auto_arm_spheres(prov, 2)
    cf = _centers(prov, jnp.asarray(sph.link), jnp.asarray(sph.local))
    fore = np.flatnonzero(sph.link == 5)                   # forearm spheres
    q0 = jnp.asarray(prov.mj_model.key_qpos[0, :nq], jnp.float32)
    tgt = jnp.asarray([0.1, -0.45, 0.35], jnp.float32)
    R_OBS = 0.06
    energy = fixed_metric_energy(nq, jnp.float32)
    base = [pose_attractor(prov, k=25.0, b=10.0, f_max=8.0), posture(nq, weight=0.5)]
    damp = [config_damping(nq, b=4.0)]
    params = FabricParams(target=tgt, q_default=q0, target_quat=jnp.asarray([1.0, 0, 0, 0], jnp.float32))

    def min_clear(qs, X):
        C = np.asarray(jax.vmap(cf)(qs))                   # (T, Ns, 3)
        return float((np.linalg.norm(C - X, axis=2) - (sph.radius + R_OBS)).min())

    # control: EE-only reach; place the obstacle on the most-traveled forearm sphere's path midpoint
    ctrl = GeometricFabric(forcing=base, damping=damp, energy=energy)
    tc = rollout(ctrl.policy, q0, jnp.zeros(nq, jnp.float32), params, 0.002, 2500, prov.site_pos)
    assert bool(jnp.all(jnp.isfinite(tc["q"])))
    C = np.asarray(jax.vmap(cf)(tc["q"]))
    s = fore[int(np.linalg.norm(C[-1, fore] - C[0, fore], axis=1).argmax())]
    X = C[len(C) // 2, s]
    assert min_clear(tc["q"], X) < -0.01, "control did not hit the obstacle — test is vacuous"

    # with the whole-arm barrier: the forearm must clear it and the EE must still reach past it
    fab = GeometricFabric(
        geometries=[arm_obstacle_geometry(prov, sph, tuple(X), R_OBS, d0=0.12, m_b=3.0)],
        forcing=base + [arm_obstacle_potential(prov, sph, tuple(X), R_OBS, k_p=0.5, d0=0.04, m_p=6.0)],
        damping=damp, energy=energy)
    tr = rollout(fab.policy, q0, jnp.zeros(nq, jnp.float32), params, 0.002, 2500, prov.site_pos)
    assert bool(jnp.all(jnp.isfinite(tr["q"])))
    assert min_clear(tr["q"], X) > -5e-3, "whole-arm obstacle: a sphere penetrated"
    assert float(jnp.linalg.norm(prov.site_pos(tr["q"][-1]) - tgt)) < 0.02, "EE failed to reach past it"


# ---------------- performance guard ----------------
def test_collision_latency_guard(prov32):
    prov = prov32
    nq = prov.nq
    sph = auto_arm_spheres(prov, 2)
    pairs = nonadjacent_pairs(sph, prov)
    fp, fn = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    fab = GeometricFabric(
        geometries=[self_collision_geometry(prov, sph, pairs), arm_obstacle_geometry(prov, sph, None, 0.08),
                    arm_plane_geometry(prov, sph, fp, fn)],
        forcing=[pose_attractor(prov, k=36.0, b=12.0, f_max=10.0), posture(nq, weight=1.0),
                 self_collision_potential(prov, sph, pairs), arm_obstacle_potential(prov, sph, None, 0.08),
                 arm_plane_potential(prov, sph, fp, fn)],
        damping=[config_damping(nq, b=2.0)], energy=fixed_metric_energy(nq, jnp.float32))
    qz = jnp.asarray(prov.mj_model.key_qpos[0, :nq], jnp.float32)
    p = _params(qz, jnp.float32, obstacle=(0.5, 0.0, 0.15))
    fab.policy(qz, jnp.zeros(nq, jnp.float32), p).block_until_ready()
    ts = []
    for _ in range(500):
        s = time.perf_counter()
        fab.policy(qz, jnp.zeros(nq, jnp.float32), p).block_until_ready()
        ts.append(time.perf_counter() - s)
    us = min(ts) * 1e6
    assert us < 400.0, f"collision policy {us:.0f} us"  # measured ~124 us min; generous guard


# ---------------- hand-tuning API (auto -> edit -> export -> reload) ----------------
def test_to_dict_from_dict_roundtrip(prov):
    # The hand-tuning pipeline's persistence: an auto model dumped to the from_dict literal and
    # reloaded must reproduce the spheres exactly (lossless at micron precision), grouped by body.
    sph = auto_arm_spheres(prov, 2)
    spec = sph.to_dict(prov)
    assert sum(len(v) for v in spec.values()) == len(sph)            # every sphere preserved
    assert len(spec) == len(set(sph.names(prov)))                    # grouped by distinct body
    back = SphereModel.from_dict(prov, spec)
    assert np.array_equal(back.link, sph.link)                       # same links, same order
    assert np.allclose(back.local, sph.local, atol=1e-6)
    assert np.allclose(back.radius, sph.radius, atol=1e-6)


def test_scaled_per_link(prov):
    # scaled() multiplies only the named link's radii; everything else (radii, centers, links) is fixed.
    sph = auto_arm_spheres(prov, 2)
    fore = sph.link == _body_id(prov.mj_model, "forearm_link")
    out = sph.scaled(prov, {"forearm_link": 0.5})
    assert np.allclose(out.radius[fore], 0.5 * sph.radius[fore])
    assert np.allclose(out.radius[~fore], sph.radius[~fore])         # other links untouched
    assert np.array_equal(out.link, sph.link) and np.allclose(out.local, sph.local)
    out.radius[0] = -1.0                                             # returned model is an independent copy
    assert sph.radius[0] != -1.0


def test_radius_scale_matches_scaled(prov):
    # The auto-time radius_scale kwarg is exactly scaled() applied to the plain auto model.
    base = auto_arm_spheres(prov, 2)
    via_auto = auto_arm_spheres(prov, 2, radius_scale={"forearm_link": 0.7, "shoulder_link": 1.3})
    via_scaled = base.scaled(prov, {"forearm_link": 0.7, "shoulder_link": 1.3})
    assert np.allclose(via_auto.radius, via_scaled.radius)
    assert np.array_equal(via_auto.link, base.link) and np.allclose(via_auto.local, base.local)


def test_body_id_rejects_unknown(prov):
    with pytest.raises(ValueError, match="unknown body"):
        auto_arm_spheres(prov, 2, radius_scale={"no_such_link": 0.5})
