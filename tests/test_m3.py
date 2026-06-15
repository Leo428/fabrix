"""fabrix M3 test suite: SE(3) pose kinematics + the coupled full-pose attractor.

M3 adds orientation. The claims under test:
  - ``CustomFK.site_pose`` reproduces MuJoCo's site frame (position *and* orientation) exactly;
  - the SE(3) error map ``Log(T*^{-1} T(q))`` and its autodiff ``J``/``Jdq`` (the curvature term
    that flows through jaxlie's ``SE3.log``) are correct to finite-difference precision;
  - a ``pose_attractor`` fabric converges in *both* position and orientation, smoothly (C2), and the
    float32 deployment path stays finite and real-time.

Math/behavior gates use x64 for clean margins; the float32 path is covered by a finiteness rollout
and the latency guard.
"""
import pathlib
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede array creation

import jax.numpy as jnp
import jaxlie
import mujoco
import numpy as np
import pytest

from fabrix import (
    CustomFK, FabricParams, GeometricFabric, config_damping, fixed_metric_energy,
    pose_attractor, posture, rollout, se3_pose_error_map,
)
from fabrix.diff import value_jac_curv

XML = str(pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie/kinova_gen3/gen3.xml")


@pytest.fixture(scope="module")
def prov():
    return CustomFK(XML, dtype=jnp.float64)


@pytest.fixture(scope="module")
def prov32():
    return CustomFK(XML, dtype=jnp.float32)


def _ori_err_deg(quat_a, quat_b):
    """Geodesic angle (degrees) between two wxyz quaternions."""
    def so3(qwxyz):
        return jaxlie.SO3.from_quaternion_xyzw(jnp.concatenate([qwxyz[1:], qwxyz[:1]]))
    return float(np.degrees(jnp.linalg.norm((so3(quat_a).inverse() @ so3(quat_b)).log())))


def _pose_fabric(prov, dtype):
    nq = prov.nq
    return GeometricFabric(forcing=[pose_attractor(prov), posture(nq)],
                           damping=[config_damping(nq, b=6.0)],
                           energy=fixed_metric_energy(nq, dtype))


# ---------------- forward kinematics: orientation ----------------
def test_site_pose_matches_mujoco(prov):
    # site_pose must match MuJoCo's site frame in BOTH position and orientation. The Gen3 pinch_site
    # has a non-identity local frame (site_quat = [0,1,0,0]), so this also guards that coupling.
    m = prov.mj_model
    d = mujoco.MjData(m)
    site = prov.site_id
    rng = np.random.default_rng(0)
    max_pos, max_rot = 0.0, 0.0
    for _ in range(25):
        q = rng.uniform(-2.0, 2.0, prov.nq)
        d.qpos[:prov.nq] = q
        mujoco.mj_forward(m, d)
        pos, quat = prov.site_pose(jnp.asarray(q))
        assert jnp.allclose(pos, prov.site_pos(jnp.asarray(q)))  # pose[0] == site_pos (shared loop)
        max_pos = max(max_pos, float(jnp.linalg.norm(pos - d.site_xpos[site])))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat))
        max_rot = max(max_rot, float(jnp.linalg.norm(R.reshape(3, 3) - d.site_xmat[site].reshape(3, 3))))
    assert max_pos < 1e-9, f"position mismatch {max_pos:.2e}"
    assert max_rot < 1e-9, f"orientation mismatch {max_rot:.2e}"


# ---------------- SE(3) error map + autodiff ----------------
def test_se3_error_zero_at_target(prov):
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.uniform(-1, 1, prov.nq))
    p, quat = prov.site_pose(q)
    e = se3_pose_error_map(prov, p, quat)(q)  # target == current pose
    assert float(jnp.linalg.norm(e)) < 1e-10


def test_pose_jacobian_curvature_finite_diff(prov):
    # J = de/dq and the curvature Jdq = Jdot @ qd, both differentiated through jaxlie's SE3.log,
    # must match finite differences -- the M3 analogue of the J/Jdq validation in experiments/.
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.uniform(-1, 1, prov.nq))
    qd = jnp.asarray(rng.uniform(-1, 1, prov.nq))
    pt, qt = prov.site_pose(q + 0.3)            # a distinct, reachable target pose
    phi = se3_pose_error_map(prov, pt, qt)
    e, J, Jdq = value_jac_curv(phi, q, qd)
    assert e.shape == (6,) and J.shape == (6, prov.nq)
    assert bool(jnp.all(jnp.isfinite(J))) and bool(jnp.all(jnp.isfinite(Jdq)))
    eps = 1e-6
    v = jnp.asarray(rng.uniform(-1, 1, prov.nq))
    fd_J = (phi(q + eps * v) - phi(q - eps * v)) / (2 * eps)
    assert float(jnp.linalg.norm(fd_J - J @ v)) < 1e-7
    fd_Jdq = (jax.jacfwd(phi)(q + eps * qd) @ qd - jax.jacfwd(phi)(q - eps * qd) @ qd) / (2 * eps)
    assert float(jnp.linalg.norm(fd_Jdq - Jdq)) < 1e-7


# ---------------- pose convergence ----------------
def test_pose_convergence(prov):
    # A reachable target pose (read off a perturbed config) must be reached in BOTH position and
    # orientation, with a smooth (C2) command. Drives the SE(3) error all the way to ~0, exercising
    # jaxlie's near-identity Log fallback under the closed loop.
    nq = prov.nq
    q0 = jnp.asarray(prov.mj_model.key_qpos[0, :nq])
    q_goal = q0 + jnp.asarray([0.3, -0.4, 0.2, 0.3, -0.2, 0.4, -0.3])
    pt, qt = prov.site_pose(q_goal)
    fab = _pose_fabric(prov, jnp.float64)
    params = FabricParams(target=pt, q_default=q0, target_quat=qt)
    tr = rollout(fab.policy, q0, jnp.zeros(nq), params, 0.002, 4000, prov.site_pos)
    pf, quatf = prov.site_pose(tr["q"][-1])
    assert bool(jnp.all(jnp.isfinite(tr["qdd"])))
    assert float(jnp.linalg.norm(pf - pt)) < 2e-3, "position did not converge"
    assert _ori_err_deg(qt, quatf) < 0.5, "orientation did not converge"
    assert float(jnp.abs(jnp.diff(tr["qdd"], axis=0)).max()) < 1.0  # C2: no per-step chatter


# ---------------- dynamic (distance-scaled) attractor mass ----------------
def test_scaled_mass_recovers_constant():
    # The schedule must (a) reduce to a constant when m_max is None OR == m_min (backward-compat), and
    # (b) when m_max>m_min, ride high near the goal, low far away, monotone decreasing in ‖e‖.
    from fabrix.leaves import _scaled_mass
    e_near, e_far = jnp.array([1e-3, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0])
    assert float(_scaled_mass(e_near, 50.0, None, 10.0, 0.1)) == 50.0      # m_max=None ⇒ exact constant
    assert float(_scaled_mass(e_far, 50.0, None, 10.0, 0.1)) == 50.0
    for e in (e_near, e_far):                                              # m_max==m_min ⇒ constant (no-op)
        assert abs(float(_scaled_mass(e, 50.0, 50.0, 10.0, 0.1)) - 50.0) < 1e-9
    near = float(_scaled_mass(e_near, 50.0, 150.0, 30.0, 0.15))           # genuine schedule
    far = float(_scaled_mass(e_far, 50.0, 150.0, 30.0, 0.15))
    assert near > 145.0, f"near-goal mass {near:.1f} should approach m_max=150"
    assert far < 55.0, f"far-field mass {far:.1f} should approach m_min=50"
    ms = [float(_scaled_mass(jnp.array([r, 0.0, 0.0]), 50.0, 150.0, 30.0, 0.15))
          for r in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)]
    assert all(ms[i] >= ms[i + 1] - 1e-9 for i in range(len(ms) - 1)), f"not monotone: {ms}"


def test_dynamic_mass_kills_posture_leak(prov):
    # The headline fix. At a goal DISPLACED from q_default, a posture(weight=2) leaf biases the
    # constant-mass attractor's EE equilibrium (the documented ~10 mm offset + slow orbit). A high
    # near-goal m_max lets the attractor dominate the metric-weighted combine → tight convergence.
    # Frictionless rollout, so this proves the effect is the metric competition, not stiction.
    nq = prov.nq
    q0 = jnp.asarray(prov.mj_model.key_qpos[0, :nq])
    q_goal = q0 + jnp.asarray([0.4, -0.5, 0.3, 0.4, -0.3, 0.5, -0.4])      # reachable, well off home
    pt, qt = prov.site_pose(q_goal)
    params = FabricParams(target=pt, q_default=q0, target_quat=qt)         # posture pulls to q0, away from q_goal

    def tail_err(att):
        fab = GeometricFabric(forcing=[att, posture(nq, weight=2.0)],
                              damping=[config_damping(nq, b=6.0)],
                              energy=fixed_metric_energy(nq, jnp.float64))
        tr = rollout(fab.policy, q0, jnp.zeros(nq), params, 0.002, 6000, prov.site_pos)
        assert bool(jnp.all(jnp.isfinite(tr["qdd"])))
        errs = jnp.linalg.norm(tr["ee"][-500:] - pt, axis=1)              # mean over last 1 s (robust to orbit)
        return float(errs.mean())

    const_err = tail_err(pose_attractor(prov))                             # constant m=50 → ~4.3 mm offset
    dyn_err = tail_err(pose_attractor(prov, m_max=300.0, sharp=20.0, offset=0.1))  # ~0.77 mm (offset ∝ 1/m_max)
    assert const_err > 3e-3, f"posture leak should bias the constant-mass EE; only {const_err*1e3:.2f} mm"
    assert dyn_err < 1e-3, f"dynamic mass should converge sub-mm: {dyn_err*1e3:.2f} mm"
    assert dyn_err < 0.3 * const_err, f"dynamic {dyn_err*1e3:.2f} mm not <0.3x constant {const_err*1e3:.2f} mm"


def test_pose_float32_finite(prov32):
    # float32 deployment path: a short rollout toward a reachable pose must stay finite and make
    # progress (jaxlie's Log threshold must not blow up in single precision).
    nq = prov32.nq
    q0 = jnp.asarray(prov32.mj_model.key_qpos[0, :nq], jnp.float32)
    pt, qt = prov32.site_pose(q0 + 0.3)
    fab = _pose_fabric(prov32, jnp.float32)
    params = FabricParams(target=pt, q_default=q0, target_quat=qt)
    tr = rollout(fab.policy, q0, jnp.zeros(nq, jnp.float32), params, 0.002, 1000, prov32.site_pos)
    assert bool(jnp.all(jnp.isfinite(tr["q"])))
    start = float(jnp.linalg.norm(prov32.site_pos(q0) - pt))
    end = float(jnp.linalg.norm(tr["ee"][-1] - pt))
    assert end < start, "made no progress toward the target"


# ---------------- performance guard ----------------
def test_pose_latency_guard(prov32):
    nq = prov32.nq
    fab = _pose_fabric(prov32, jnp.float32)
    qz = jnp.zeros(nq, jnp.float32)
    pt, qt = prov32.site_pose(qz + 0.2)
    p = FabricParams(target=pt, q_default=qz, target_quat=qt)
    fab.policy(qz, qz, p).block_until_ready()
    ts = []
    for _ in range(500):
        s = time.perf_counter(); fab.policy(qz, qz, p).block_until_ready(); ts.append(time.perf_counter() - s)
    us = min(ts) * 1e6
    assert us < 300.0, f"pose policy {us:.0f} us"  # measured ~88 us; generous guard for the jaxlie path
