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
    energy_spec, fixed_metric_energy, joint_limit_geometry, joint_limit_potential, joint_speed_limit,
    lagrangian_energy, obstacle_geometry, obstacle_potential, posture, rollout, speed_control,
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


# ---------------- speed control / kinetic-energy cap ----------------
def test_speed_control_reduces_to_damping(prov):
    # Backward-compat contract: beta_speed=0 ⇒ speed_control is bit-identical to config_damping, so the
    # default Gains (speed_beta=0) leave the proven behavior untouched.
    nq = prov.nq
    rng = np.random.default_rng(7)
    sc, cd = speed_control(nq, b=2.0, beta_speed=0.0), config_damping(nq, b=2.0)
    params = FabricParams(jnp.zeros(3), jnp.zeros(nq))
    for _ in range(20):
        q, qd = jnp.asarray(rng.uniform(-1, 1, nq)), jnp.asarray(rng.uniform(-4, 4, nq))
        s1, s2 = sc(q, qd, params), cd(q, qd, params)
        assert float(jnp.abs(s1.M - s2.M).max()) < 1e-12 and float(jnp.abs(s1.f - s2.f).max()) < 1e-12


def test_geometric_fabric_reference_damping(prov):
    # GeometricFabric's post-combine reference damping is NVlabs' cspace_damping (fabrics_sim
    # fabric.py:521, force += gain·M·q̇): applied to the COMBINED metric so it cancels, adding EXACTLY
    # -b·qd_ref to the accel regardless of the metric — and only when params.qd_ref is set. Integrated by
    # the control node's pure q̇_ref += dt·a, this is the leaky reference integrator (1-dt·b)q̇_ref+dt·a.
    nq = prov.nq
    rng = np.random.default_rng(13)
    # config_damping gives the combined metric a full-rank mass·I floor (the real fabric always has
    # speed_control), so the post-combine cancellation is exact (reg=1e-6 ≪ the eigenvalues).
    base = dict(forcing=[attractor(prov, k=20.0, b=8.0, m=50.0)], damping=[config_damping(nq, b=2.0)],
                energy=fixed_metric_energy(nq, jnp.float64))
    fab0 = GeometricFabric(**base)                              # no reference damping (ref_damp=None)
    fabb = GeometricFabric(**base, ref_damp=3.0)               # b=3 reference damping, post-combine
    q0 = jnp.asarray(prov.mj_model.key_qpos[0, :nq])
    pt = prov.site_pos(q0)
    for _ in range(15):
        q = q0 + jnp.asarray(rng.uniform(-0.3, 0.3, nq))
        qd = jnp.asarray(rng.uniform(-2, 2, nq))
        qd_ref = jnp.asarray(rng.uniform(-2, 2, nq))
        a0 = fab0.policy(q, qd, FabricParams(pt, q0, qd_ref=qd_ref))   # ref_damp None ⇒ qd_ref ignored
        ab = fabb.policy(q, qd, FabricParams(pt, q0, qd_ref=qd_ref))
        # = -b·qd_ref, metric-cancelled; the residual is just the resolve's Tikhonov reg=1e-6 floor
        # (reg·(M+reg)⁻¹·b·qd_ref ≈ µrad/s²), far below the rad/s²-scale damping term.
        assert float(jnp.abs((ab - a0) - (-3.0 * qd_ref)).max()) < 1e-4
    # params.qd_ref is None ⇒ reference damping inert: backward-compatible with every non-reference fabric
    a_none = fabb.policy(q, qd, FabricParams(pt, q0))
    a_zero = fab0.policy(q, qd, FabricParams(pt, q0))
    assert float(jnp.abs(a_none - a_zero).max()) < 1e-6


def test_speed_cap_respected(prov):
    # The overspeed boost bounds peak kinetic energy E=½‖q̇‖². With a saturating attractor (bounded
    # drive) the cap holds near E_max; it is a smooth soft-cap so a step command overshoots ~30% on the
    # transient (the hard per-axis qd_max backstop lives in the control node, not here). No qd clip in
    # the rollout, so this isolates the fabric's own cap.
    nq = prov.nq
    q0 = jnp.asarray(prov.mj_model.key_qpos[0, :nq])
    pt = prov.site_pos(q0 + 0.8)                      # far target → strong (but f_max-bounded) drive
    E_max = 0.3

    def peak_E(beta):
        fab = GeometricFabric(forcing=[attractor(prov, k=36.0, b=12.0, m=50.0, f_max=10.0)],
                              damping=[speed_control(nq, b=2.0, beta_speed=beta, E_max=E_max, k_gate=20.0)],
                              energy=fixed_metric_energy(nq, jnp.float64))
        tr = rollout(fab.policy, q0, jnp.zeros(nq), FabricParams(pt, q0), 0.002, 3000, prov.site_pos)
        assert bool(jnp.all(jnp.isfinite(tr["qd"])))
        return float(jnp.max(0.5 * jnp.sum(tr["qd"] ** 2, axis=1)))

    uncapped, capped = peak_E(0.0), peak_E(120.0)
    assert uncapped > E_max, f"setup: uncapped peak KE {uncapped:.3f} must exceed the cap to test it"
    assert capped < 0.6 * uncapped, f"cap should materially cut peak KE: {capped:.3f} vs {uncapped:.3f}"
    assert capped <= E_max * 1.4, f"capped peak KE {capped:.3f} not held near E_max={E_max}"


def test_baseline_speed_monotone(prov):
    # b is the baseline speed/damping knob: with no drive, a higher b bleeds an initial velocity faster
    # (lower b ⇒ faster motion). Minimal damping-only fabric isolates the knob from any attractor.
    nq = prov.nq
    qd0 = jnp.ones(nq)

    def speed_after(b):
        fab = GeometricFabric(damping=[speed_control(nq, b=b)], energy=fixed_metric_energy(nq, jnp.float64))
        tr = rollout(fab.policy, jnp.zeros(nq), qd0, FabricParams(jnp.zeros(3), jnp.zeros(nq)),
                     0.002, 500, prov.site_pos)
        return float(jnp.linalg.norm(tr["qd"][-1]))

    speeds = [speed_after(b) for b in (1.0, 2.0, 4.0, 8.0)]
    assert all(speeds[i] > speeds[i + 1] for i in range(len(speeds) - 1)), f"not monotone in b: {speeds}"
    assert speeds[-1] < 0.5 * speeds[0], f"b=8 should bleed much faster than b=1: {speeds}"


def test_joint_speed_limit_caps_velocity(prov):
    # The per-joint velocity barrier holds |q̇_j| under qd_lim when a strong attractor would otherwise
    # drive it past — a smooth in-fabric speed cap (complements the global KE cap). Must stay finite.
    nq = prov.nq
    q0 = jnp.asarray(prov.mj_model.key_qpos[0, :nq])
    pt = prov.site_pos(q0 + 0.8)
    qd_lim = 0.6

    def peak(use_barrier):
        forcing = [attractor(prov, k=36.0, b=12.0, m=50.0, f_max=10.0)]
        if use_barrier:
            forcing.append(joint_speed_limit(nq, qd_lim=qd_lim, k_b=5.0, m_b=20.0, d0=0.4))
        fab = GeometricFabric(forcing=forcing, damping=[config_damping(nq, b=2.0)],
                              energy=fixed_metric_energy(nq, jnp.float64))
        tr = rollout(fab.policy, q0, jnp.zeros(nq), FabricParams(pt, q0), 0.002, 3000, prov.site_pos)
        assert bool(jnp.all(jnp.isfinite(tr["qd"])))
        return float(jnp.max(jnp.abs(tr["qd"])))

    uncapped, capped = peak(False), peak(True)
    assert uncapped > qd_lim, f"setup: uncapped peak |q̇| {uncapped:.3f} should exceed the limit to test it"
    assert capped <= qd_lim, f"barrier should cap |q̇| under the limit: {capped:.3f} > {qd_lim}"


def test_joint_speed_limit_inert_by_default(prov):
    # k_b=m_b=0 (default) ⇒ identically zero (no-op); and even with active gains it is zero far from
    # the limit (|q̇| well below the band) so it never fights free motion.
    nq = prov.nq
    params = FabricParams(jnp.zeros(3), jnp.zeros(nq))
    s = joint_speed_limit(nq)(jnp.zeros(nq), jnp.ones(nq) * 0.3, params)              # defaults k_b=m_b=0
    assert float(jnp.abs(s.M).max()) == 0.0 and float(jnp.abs(s.f).max()) == 0.0
    on = joint_speed_limit(nq, qd_lim=2.0, k_b=5.0, m_b=20.0, d0=0.3)
    s2 = on(jnp.zeros(nq), jnp.ones(nq) * 0.2, params)                                # |q̇|=0.2 ≪ 2.0−0.3 → inert
    assert float(jnp.abs(s2.M).max()) == 0.0 and float(jnp.abs(s2.f).max()) == 0.0


def test_limit_accel_direction_preserving():
    from fabrix import limit_accel
    qdd = jnp.array([10.0, -30.0, 5.0])
    out = limit_accel(qdd, 15.0)
    assert float(jnp.max(jnp.abs(out))) <= 15.0 + 1e-6, "not capped to a_max"
    assert float(jnp.linalg.norm(jnp.cross(out, qdd))) < 1e-6, "direction changed (not a uniform scale)"
    assert jnp.allclose(limit_accel(qdd, 100.0), qdd), "should be a no-op when already under a_max"


def test_limit_jerk_rate_limits():
    from fabrix import limit_jerk
    prev = jnp.zeros(3)
    assert jnp.allclose(limit_jerk(jnp.array([100.0, -100.0, 0.0]), prev, 5.0),
                        jnp.array([5.0, -5.0, 0.0])), "Δqdd not clamped to ±dqdd_max"
    assert jnp.allclose(limit_jerk(jnp.array([2.0, -3.0, 1.0]), prev, 1e9),
                        jnp.array([2.0, -3.0, 1.0])), "huge dqdd_max should be a no-op"


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
