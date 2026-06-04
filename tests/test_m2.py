"""fabrix M2 test suite: energization correctness, energy conservation, barrier invariants.

The math gates (energy spec identity, the energization operator, energy conservation) use x64 for
clean comparisons. The behavior gates (invariants, reach+avoid) also run in x64 for robust margins;
the float32 deployment path is covered by the latency guard here and by the float32 M1 tests.

The two load-bearing claims of M2:
  - energization conserves the execution energy *exactly* in continuous time (instantaneously to
    ~1e-9; a discrete rollout drifts only by integrator error), and only regulates speed — it never
    bends the geometry's path;
  - barrier *potentials* make joint limits and obstacle clearance *hard invariants*.
"""
import pathlib
import time

import jax

jax.config.update("jax_enable_x64", True)  # must precede array creation

import jax.numpy as jnp
import numpy as np
import pytest

from fabrix import (
    CustomFK, FabricParams, GeometricFabric, attractor, config_damping, energize,
    energy_spec, fixed_metric_energy, joint_limit_geometry, joint_limit_potential,
    lagrangian_energy, obstacle_geometry, obstacle_potential, posture, rollout,
)

XML = str(pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie/kinova_gen3/gen3.xml")


@pytest.fixture(scope="module")
def prov():
    return CustomFK(XML, dtype=jnp.float64)


@pytest.fixture(scope="module")
def prov32():
    return CustomFK(XML, dtype=jnp.float32)


# A non-trivial x-dependent SPD metric and its Lagrangian energy, for the energy-spec tests.
def _G(x):
    B = jnp.diag(1.0 + 0.5 * jnp.sin(x)) + 0.1
    return B @ B.T + jnp.eye(x.shape[0])


def _L(x, xd):
    return 0.5 * xd @ (_G(x) @ xd)


# ---------------- energy spec ----------------
def test_fixed_metric_energy():
    energy = fixed_metric_energy(5, jnp.float64)
    M, f = energy(jnp.arange(5.0), jnp.arange(5.0))
    assert jnp.allclose(M, jnp.eye(5)) and jnp.allclose(f, 0.0)


def test_energy_spec_metric_is_hessian():
    # For L = 1/2 xd^T G(x) xd, the energy metric must equal G(x).
    rng = np.random.default_rng(0)
    x = jnp.array(rng.uniform(-1, 1, 5))
    xd = jnp.array(rng.uniform(-1, 1, 5))
    M, _ = energy_spec(_L, x, xd)
    assert float(jnp.abs(M - _G(x)).max()) < 1e-12


def test_energy_spec_rate_identity():
    # The identity the whole framework rests on: dH_e/dt == xd^T (M_e xddot + f_e),
    # with H_e = xd . dL/dxd - L. Computed two independent ways; must agree to machine eps.
    rng = np.random.default_rng(1)
    x = jnp.array(rng.uniform(-1, 1, 5))
    xd = jnp.array(rng.uniform(-1, 1, 5))
    xdd = jnp.array(rng.uniform(-1, 1, 5))

    def H(xx, vv):
        return jax.grad(_L, argnums=1)(xx, vv) @ vv - _L(xx, vv)

    dHdt = jax.grad(H, 0)(x, xd) @ xd + jax.grad(H, 1)(x, xd) @ xdd
    M, f = energy_spec(_L, x, xd)
    assert float(abs(dHdt - xd @ (M @ xdd + f))) < 1e-10


# ---------------- energization operator ----------------
@pytest.mark.parametrize("make_energy", [
    lambda n: fixed_metric_energy(n, jnp.float64),
    lambda n: lagrangian_energy(_G),
])
def test_energize_conserves_instantaneously(make_energy):
    rng = np.random.default_rng(2)
    n = 5
    x = jnp.array(rng.uniform(-1, 1, n))
    v = jnp.array(rng.uniform(-1, 1, n))
    a_g = jnp.array(rng.uniform(-1, 1, n))
    M_e, f_e = make_energy(n)(x, v)
    a_e = energize(a_g, v, M_e, f_e)
    # the defining property: the energy rate is zeroed (to the eps floor)
    assert float(abs(v @ (M_e @ a_e + f_e))) < 1e-7


def test_energize_preserves_path():
    # the correction a_e - a_g is purely along v (speed regulated, direction unchanged)
    rng = np.random.default_rng(3)
    n = 5
    v = jnp.array(rng.uniform(-1, 1, n))
    a_g = jnp.array(rng.uniform(-1, 1, n))
    M_e, f_e = lagrangian_energy(_G)(jnp.zeros(n), v)
    corr = energize(a_g, v, M_e, f_e) - a_g
    off_axis = corr - (corr @ v) / (v @ v) * v
    assert float(jnp.abs(off_axis).max()) < 1e-9


def test_geometry_only_conserves_energy(prov):
    # A bare energized geometry (no forcing, no damping) must conserve 1/2 ||qd||^2 along a
    # rollout, up to integrator error. Drive joint 1 toward its limit so the barrier is active.
    nq = prov.nq
    fab = GeometricFabric(geometries=[joint_limit_geometry(prov)],
                          energy=fixed_metric_energy(nq, jnp.float64), reg=0.0)
    q0 = jnp.zeros(nq).at[1].set(2.0)
    qd0 = jnp.zeros(nq).at[1].set(1.5)
    tr = rollout(fab.policy, q0, qd0, FabricParams(jnp.zeros(3), jnp.zeros(nq)), 1e-3, 300, prov.site_pos)
    E = 0.5 * jnp.sum(tr["qd"] ** 2, axis=1)
    assert float(jnp.max(jnp.abs(E - E[0])) / E[0]) < 5e-3  # measured ~7.5e-4


# ---------------- barrier invariants ----------------
def test_joint_limit_invariant(prov):
    nq = prov.nq
    m = prov.mj_model
    limited = np.where(m.jnt_limited)[0]
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    fab = GeometricFabric(
        forcing=[posture(nq, k=4.0, b=4.0, weight=3.0),
                 joint_limit_potential(prov, k_p=0.05, d0=0.4, m_p=3.0)],
        damping=[config_damping(nq, b=4.0)], energy=fixed_metric_energy(nq, jnp.float64))
    # posture target sits *outside* the limits on every limited joint -> barrier must hold them in
    q_out = jnp.zeros(nq)
    for j in limited:
        q_out = q_out.at[j].set(hi[j] + 0.5)
    tr = rollout(fab.policy, jnp.zeros(nq), jnp.zeros(nq),
                 FabricParams(jnp.zeros(3), q_out), 0.002, 2000, prov.site_pos)
    q = tr["q"]
    for j in limited:
        assert float(jnp.max(q[:, j])) <= hi[j] + 1e-4, f"joint {j} exceeded upper limit"
        assert float(jnp.min(q[:, j])) >= lo[j] - 1e-4, f"joint {j} exceeded lower limit"


def _obstacle_fabric(prov, center, radius):
    nq = prov.nq
    return GeometricFabric(
        geometries=[obstacle_geometry(prov, center, radius, k_b=1.0, m_b=2.0)],
        forcing=[attractor(prov), obstacle_potential(prov, center, radius, k_p=0.6, d0=0.2, m_p=6.0)],
        damping=[config_damping(nq, b=6.0)], energy=fixed_metric_energy(nq, jnp.float64))


def test_obstacle_invariant(prov):
    # target placed on the far side of the sphere: the straight path would penetrate, so the
    # barrier potential must keep the end-effector outside the sphere (it need not reach the goal).
    nq = prov.nq
    p_home = np.asarray(prov.site_pos(jnp.zeros(nq)))
    p_tgt = np.asarray(prov.site_pos(jnp.zeros(nq).at[1].set(0.4).at[3].set(0.5).at[5].set(-0.3)))
    radius = 0.12
    center = jnp.asarray(p_home + 0.5 * (p_tgt - p_home))
    fab = _obstacle_fabric(prov, center, radius)
    tr = rollout(fab.policy, jnp.zeros(nq), jnp.zeros(nq),
                 FabricParams(jnp.asarray(p_tgt), jnp.zeros(nq)), 0.002, 2500, prov.site_pos)
    dist = jnp.linalg.norm(tr["ee"] - center, axis=1)
    assert float(jnp.min(dist)) >= radius - 1e-3, "end-effector penetrated the obstacle"


def test_reach_and_avoid(prov):
    # sphere offset off the direct line: the fabric should route around it AND reach the goal,
    # with a smooth (C2) command throughout.
    nq = prov.nq
    p_home = np.asarray(prov.site_pos(jnp.zeros(nq)))
    p_tgt = np.asarray(prov.site_pos(jnp.zeros(nq).at[1].set(0.4).at[3].set(0.5).at[5].set(-0.3)))
    radius = 0.10
    center = jnp.asarray(p_home + 0.5 * (p_tgt - p_home) + np.array([0.0, 0.06, 0.0]))
    fab = _obstacle_fabric(prov, center, radius)
    tr = rollout(fab.policy, jnp.zeros(nq), jnp.zeros(nq),
                 FabricParams(jnp.asarray(p_tgt), jnp.zeros(nq)), 0.002, 2500, prov.site_pos)
    dist = jnp.linalg.norm(tr["ee"] - center, axis=1)
    err = float(jnp.linalg.norm(tr["ee"][-1] - p_tgt))
    qdd = tr["qdd"]
    assert float(jnp.min(dist)) >= radius - 1e-3, "penetrated obstacle while reaching"
    assert err < 0.025, f"did not reach goal: {err*1000:.1f} mm"
    assert bool(jnp.all(jnp.isfinite(qdd)))
    assert float(jnp.abs(jnp.diff(qdd, axis=0)).max()) < 1.0  # C2: no per-step chatter


# ---------------- performance guard ----------------
def test_m2_latency_guard(prov32):
    nq = prov32.nq
    p_tgt = prov32.site_pos(jnp.zeros(nq).at[1].set(0.4))
    center = prov32.site_pos(jnp.zeros(nq).at[1].set(0.2))
    fab = GeometricFabric(
        geometries=[obstacle_geometry(prov32, center, 0.1), joint_limit_geometry(prov32)],
        forcing=[attractor(prov32), obstacle_potential(prov32, center, 0.1),
                 joint_limit_potential(prov32)],
        damping=[config_damping(nq)], energy=fixed_metric_energy(nq, jnp.float32))
    p = FabricParams(p_tgt, jnp.zeros(nq, jnp.float32))
    qz = jnp.zeros(nq, jnp.float32)
    fab.policy(qz, qz, p).block_until_ready()
    ts = []
    for _ in range(500):
        s = time.perf_counter(); fab.policy(qz, qz, p).block_until_ready(); ts.append(time.perf_counter() - s)
    us = min(ts) * 1e6
    assert us < 250.0, f"policy {us:.0f} us"  # measured ~66 us; generous guard vs the scalar-pack trap
