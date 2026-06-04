"""fabrix M1 test suite: correctness, convergence, smoothness, and a latency guard.

x64 is enabled for clean correctness comparisons; the convergence/smoothness/latency tests
use the float32 provider (what we'd deploy). The latency guard exists to catch performance
regressions — most importantly the `jnp.array([scalar, ...])` anti-pattern that once cost ~1000x.
"""
import pathlib
import statistics
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede array creation

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from fabrix import (
    CustomFK, Fabric, FabricParams, Spec, attractor, combine,
    config_damping, posture, pullback, resolve, rollout, value_jac_curv,
)
from fabrix.leaves import _restoring

XML = str(pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie/kinova_gen3/gen3.xml")


@pytest.fixture(scope="module")
def prov64():
    return CustomFK(XML, dtype=jnp.float64)


@pytest.fixture(scope="module")
def prov32():
    return CustomFK(XML, dtype=jnp.float32)


@pytest.fixture(scope="module")
def setup(prov32):
    nq = prov32.nq
    q0 = jnp.zeros(nq, jnp.float32)
    target = prov32.site_pos(q0.at[1].set(0.4).at[3].set(0.5).at[5].set(-0.3))
    fabric = Fabric([attractor(prov32), posture(nq), config_damping(nq)])
    params = FabricParams(target=target, q_default=q0)
    return prov32, fabric, params, q0, jnp.zeros(nq, jnp.float32)


def _mj_site(model, site_id, q):
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    return d.site_xpos[site_id].copy()


# ---------------- correctness ----------------
def test_customfk_matches_mujoco(prov64):
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1, 1, prov64.nq)
        gt = _mj_site(prov64.mj_model, prov64.site_id, q)
        got = np.asarray(prov64.site_pos(jnp.array(q)))
        assert np.abs(got - gt).max() < 1e-10


def test_curvature_vs_finitediff(prov64):
    rng = np.random.default_rng(2)
    q = jnp.array(rng.uniform(-1, 1, prov64.nq))
    qd = jnp.array(rng.uniform(-1, 1, prov64.nq))
    _, _, Jdq = value_jac_curv(prov64.site_pos, q, qd)
    eps = 1e-6
    Jp = jax.jacfwd(prov64.site_pos)(q + eps * qd)
    Jm = jax.jacfwd(prov64.site_pos)(q - eps * qd)
    fd = (Jp @ qd - Jm @ qd) / (2 * eps)
    assert float(jnp.abs(Jdq - fd).max()) < 1e-6


def test_pullback_identity():
    n = 5
    rng = np.random.default_rng(3)
    M = jnp.array(rng.uniform(size=(n, n)))
    f = jnp.array(rng.uniform(size=n))
    s = pullback(Spec(M, f), jnp.eye(n), jnp.zeros(n))
    assert jnp.allclose(s.M, M) and jnp.allclose(s.f, f)


def test_resolve_solves_spd():
    n = 7
    rng = np.random.default_rng(4)
    A = jnp.array(rng.uniform(size=(n, n)))
    M = A @ A.T + jnp.eye(n)  # SPD
    f = jnp.array(rng.uniform(size=n))
    reg = 1e-6
    qdd = resolve(Spec(M, f), reg=reg)
    # resolve solves (M + reg I) qdd = -f
    assert float(jnp.abs((M + reg * jnp.eye(n)) @ qdd + f).max()) < 1e-8


def test_combine_sums():
    s = combine([Spec(jnp.eye(3), jnp.ones(3)), Spec(2 * jnp.eye(3), jnp.full(3, 2.0))])
    assert jnp.allclose(s.M, 3 * jnp.eye(3)) and jnp.allclose(s.f, jnp.full(3, 3.0))


# ---------------- behavior ----------------
def test_convergence(setup):
    prov, fabric, params, q0, qd0 = setup
    traj = rollout(fabric.policy, q0, qd0, params, 0.002, 1500, prov.site_pos)
    err = jnp.linalg.norm(traj["ee"] - params.target, axis=1)
    assert float(err[-1]) < 2e-3          # converges within 2 mm
    assert float(err[-1]) < float(err[0])  # made progress


def test_smoothness(setup):
    prov, fabric, params, q0, qd0 = setup
    traj = rollout(fabric.policy, q0, qd0, params, 0.002, 1500, prov.site_pos)
    qdd = traj["qdd"]
    assert bool(jnp.all(jnp.isfinite(qdd)))
    # acceleration is continuous: step-to-step change stays small (no per-step chatter)
    assert float(jnp.abs(jnp.diff(qdd, axis=0)).max()) < 0.5
    # starts from rest
    assert float(jnp.abs(traj["qd"][0]).max()) < 0.1


# ---------------- leaf enhancements: saturating attractor, per-joint posture ----------------
def test_saturating_restoring():
    # f_max=None is the plain quadratic gradient; with f_max set, the force matches k*e near zero
    # (same stiffness) but its magnitude saturates at f_max far away, pointing along e throughout.
    rng = np.random.default_rng(10)
    k, f_max = 36.0, 8.0
    e_small = jnp.array(rng.uniform(-1, 1, 6)) * 1e-4
    assert jnp.allclose(_restoring(e_small, k, None), k * e_small)
    assert float(jnp.linalg.norm(_restoring(e_small, k, f_max) - k * e_small)) < 1e-6
    for s in (0.5, 2.0, 50.0):
        e = jnp.array(rng.uniform(-1, 1, 6)); e = s * e / jnp.linalg.norm(e)
        g = _restoring(e, k, f_max)
        assert float(jnp.linalg.norm(g)) <= f_max + 1e-6           # never exceeds the cap
        alpha = float((g @ e) / (e @ e))
        assert alpha > 0 and float(jnp.linalg.norm(g - alpha * e)) < 1e-6   # parallel to e
    e_far = 50.0 * jnp.ones(6) / jnp.linalg.norm(jnp.ones(6))
    assert float(jnp.linalg.norm(_restoring(e_far, k, f_max))) > 0.99 * f_max  # saturated far away


def test_saturating_attractor_bounds_acceleration(prov64):
    # End-to-end: on a far target the saturating attractor caps the commanded acceleration well below
    # the quadratic one (no lunge), while still converging.
    nq = prov64.nq
    q0 = jnp.zeros(nq)
    far = prov64.site_pos(q0.at[1].set(0.7).at[3].set(0.7).at[5].set(-0.4))

    def run(f_max):
        fab = Fabric([attractor(prov64, k=36.0, b=12.0, f_max=f_max), posture(nq), config_damping(nq, b=2.0)])
        tr = rollout(fab.policy, q0, jnp.zeros(nq), FabricParams(target=far, q_default=q0), 0.002, 4000, prov64.site_pos)
        return float(jnp.max(jnp.abs(tr["qdd"]))), float(jnp.linalg.norm(tr["ee"][-1] - far))

    quad_peak, quad_err = run(None)
    sat_peak, sat_err = run(8.0)
    assert sat_peak < 0.7 * quad_peak, f"saturating peak {sat_peak:.1f} vs quadratic {quad_peak:.1f}"
    assert quad_err < 5e-3 and sat_err < 5e-3   # both reach the goal


def test_posture_per_joint_weight(prov64):
    # A per-joint posture weight holds the heavily-weighted joint nearer q_default than a uniform
    # weight does, in a redundant (position-only) task — without losing EE convergence.
    nq = prov64.nq
    q0 = jnp.zeros(nq)
    target = prov64.site_pos(q0.at[1].set(0.5).at[3].set(0.6).at[5].set(-0.3))
    q_start = q0.at[1].set(0.1)

    def final(weight):
        fab = Fabric([attractor(prov64), posture(nq, weight=weight), config_damping(nq)])
        tr = rollout(fab.policy, q_start, jnp.zeros(nq), FabricParams(target=target, q_default=q0),
                     0.002, 2500, prov64.site_pos)
        return tr["q"][-1], float(jnp.linalg.norm(tr["ee"][-1] - target))

    q_uni, err_uni = final(0.5)                                   # uniform low weight
    q_pj, err_pj = final(jnp.full(nq, 0.5).at[5].set(20.0))       # heavy on joint 5
    assert err_uni < 5e-3 and err_pj < 5e-3                       # both still reach the EE target
    assert abs(float(q_pj[5])) < abs(float(q_uni[5]))            # joint 5 held closer to q_default (0)


# ---------------- performance guard ----------------
def test_latency_guard(setup):
    prov, fabric, params, q0, qd0 = setup

    def jdq(q, qd):
        f1 = lambda qq: jax.jvp(prov.site_pos, (qq,), (qd,))[1]
        return jax.jvp(f1, (q,), (qd,))[1]
    jdq_j = jax.jit(jdq)

    def bench(fn, args, M=1000):
        fn(*args).block_until_ready()
        ts = []
        for _ in range(M):
            s = time.perf_counter(); fn(*args).block_until_ready(); ts.append(time.perf_counter() - s)
        return min(ts) * 1e6

    policy_us = bench(fabric.policy, (q0, qd0, params))
    jdq_us = bench(jdq_j, (q0, qd0))
    # generous thresholds: catch the ~1000x scalar-packing regression without CI flakiness
    assert policy_us < 200.0, f"policy {policy_us:.0f} us"
    assert jdq_us < 80.0, f"Jdotqd {jdq_us:.0f} us"
