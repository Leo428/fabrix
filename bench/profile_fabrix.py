"""Latency + scaling profiler for the fabrix control policy (permanent regression/design bench).

Run:  uv run python bench/profile_fabrix.py

CPU / float32 — what a real-time controller pays per step. Sections:
  A. end-to-end latency of the base pose fabric and the full whole-arm + self-collision fabric;
  B. per-component breakdown (FK, autodiff J + curvature, leaves, resolve);
  C. the scaling property behind the batched collision design — N barriers as separate leaves vs one
     batched vector-SDF leaf (the shipped `sdf_barrier_geometry` core): compile time + latency vs N;
  D. hygiene — no recompiles on param-value changes, float32 output.

A single dispatch with DISTINCT inputs each call is the honest controller number; the fused
`compute-only` estimate runs many evals in one `lax.scan` (loses thread parallelism, ~2x higher) and
is useful only for relative component costs.
"""
import os
import pathlib
import statistics
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from fabrix import (
    CustomFK, FabricParams, GeometricFabric, Spec, arm_obstacle_geometry, arm_obstacle_potential,
    arm_plane_geometry, arm_plane_potential, auto_arm_spheres, config_damping, fixed_metric_energy,
    joint_limit_geometry, joint_limit_potential, nonadjacent_pairs, obstacle_geometry, pose_attractor,
    posture, sdf_barrier_geometry, self_collision_geometry, self_collision_potential,
)
from fabrix.diff import value_jac_curv
from fabrix.maps import se3_pose_error_map, site_position_map
from fabrix.spec import resolve

XML = str(pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie/kinova_gen3/gen3.xml")
print("x64:", jax.config.jax_enable_x64, "| devices:", jax.devices(),
      "| ncpu:", os.cpu_count(), "| loadavg:", [round(x, 2) for x in os.getloadavg()], "\n")

prov = CustomFK(XML)
nq = prov.nq
rng = np.random.default_rng(0)
q0 = jnp.asarray(rng.uniform(-1, 1, nq), jnp.float32)
qd0 = jnp.asarray(rng.uniform(-1, 1, nq), jnp.float32)
q_home = jnp.asarray(prov.mj_model.key_qpos[0, :nq], jnp.float32)
FLOOR_PT, FLOOR_N = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
sph = auto_arm_spheres(prov, n_per_link=2)
pairs = nonadjacent_pairs(sph, prov)


def params(tgt=(0.45, 0.0, 0.5), obs=(0.5, 0.0, 0.15)):
    return FabricParams(target=jnp.asarray(tgt, jnp.float32), q_default=q_home,
                        target_quat=jnp.asarray([1.0, 0, 0, 0], jnp.float32),
                        obstacle_center=jnp.asarray(obs, jnp.float32))


P0 = params()
QS = [jnp.asarray(rng.uniform(-2, 2, nq), jnp.float32) for _ in range(3000)]   # distinct inputs


def per_call(jfn, args, budget_s=1.0, max_M=3000):
    """compile-once (ms), then steady-state min / median per-call latency (us), identical inputs."""
    t = time.perf_counter(); jax.block_until_ready(jfn(*args)); comp = (time.perf_counter() - t) * 1e3
    ts, start = [], time.perf_counter()
    while len(ts) < max_M and (time.perf_counter() - start) < budget_s:
        t = time.perf_counter(); jax.block_until_ready(jfn(*args)); ts.append(time.perf_counter() - t)
    return comp, min(ts) * 1e6, statistics.median(ts) * 1e6


def per_call_distinct(policy, p, M=3000):
    """Per-call latency with a DISTINCT q each call (what a controller pays — q changes every step)."""
    jax.block_until_ready(policy(QS[0], qd0, p))
    ts = []
    for i in range(M):
        t = time.perf_counter(); jax.block_until_ready(policy(QS[i], qd0, p)); ts.append(time.perf_counter() - t)
    return min(ts) * 1e6, statistics.median(ts) * 1e6


def compute_only(raw_fn, args, N=1500):
    """N fused evals in one dispatch, input perturbed to defeat cross-iteration CSE; us/eval."""
    def scanned(*a):
        q, rest = a[0], a[1:]
        def stp(c, i):
            return c, raw_fn(q + 1e-4 * jnp.sin(i.astype(jnp.float32)), *rest)
        return lax.scan(stp, None, jnp.arange(N))[1]
    j = jax.jit(scanned)
    jax.block_until_ready(j(*args))
    t = time.perf_counter(); jax.block_until_ready(j(*args)); return (time.perf_counter() - t) / N * 1e6


def base_fabric():
    return GeometricFabric(
        geometries=[obstacle_geometry(prov, None, 0.08, d0=0.12), joint_limit_geometry(prov)],
        forcing=[pose_attractor(prov, k=36.0, b=12.0, f_max=10.0), posture(nq, weight=1.0)],
        damping=[config_damping(nq, b=2.0)], energy=fixed_metric_energy(nq, jnp.float32))


def collision_fabric():
    return GeometricFabric(
        geometries=[self_collision_geometry(prov, sph, pairs), arm_obstacle_geometry(prov, sph, None, 0.08),
                    joint_limit_geometry(prov), arm_plane_geometry(prov, sph, FLOOR_PT, FLOOR_N)],
        forcing=[pose_attractor(prov, k=36.0, b=12.0, f_max=10.0), posture(nq, weight=1.0),
                 self_collision_potential(prov, sph, pairs), arm_obstacle_potential(prov, sph, None, 0.08),
                 arm_plane_potential(prov, sph, FLOOR_PT, FLOOR_N), joint_limit_potential(prov)],
        damping=[config_damping(nq, b=2.0)], energy=fixed_metric_energy(nq, jnp.float32))


# ===========================================================================
print("=" * 80)
print("SECTION A — end-to-end latency (single dispatch; distinct-input is the controller number)")
print("=" * 80)
for name, fab in [("base pose fabric", base_fabric()),
                  (f"full collision fabric ({len(pairs)} self + {2*len(sph)} env)", collision_fabric())]:
    c, mn, md = per_call(fab.policy, (q0, qd0, P0))
    dmn, dmd = per_call_distinct(fab.policy, P0)
    co = compute_only(lambda q, qd, p: fab._policy(q, qd, p), (q0, qd0, P0))
    print(f"  {name:<42} compile {c:5.0f} ms")
    print(f"  {'':<42} distinct-input min {dmn:6.1f}  median {dmd:6.1f} us  | fused {co:6.1f} us/eval")
print("  budget: 1 kHz = 1000 us/step, 500 Hz = 2000 us/step\n")

# ===========================================================================
print("=" * 80)
print("SECTION B — component breakdown (each jitted alone; compute-only fused, us/eval)")
print("=" * 80)
pos_map = site_position_map(prov)
pose_map = se3_pose_error_map(prov, P0.target, P0.target_quat)
obst = obstacle_geometry(prov, (0.5, 0.0, 0.15), 0.08, d0=0.12)
poseL = pose_attractor(prov, k=36.0, b=12.0, f_max=10.0)
selfL = self_collision_geometry(prov, sph, pairs)
M_spd = jnp.asarray(rng.uniform(-1, 1, (nq, nq)), jnp.float32)
M_spd = M_spd @ M_spd.T + nq * jnp.eye(nq, dtype=jnp.float32)
f_rand = jnp.asarray(rng.uniform(-1, 1, nq), jnp.float32)
comps = [
    ("FK site_pos",            jax.jit(prov.site_pos),                        prov.site_pos,                                 (q0,)),
    ("FK body_poses (all)",    jax.jit(lambda q: prov.body_poses(q)[0]),      lambda q: prov.body_poses(q)[0],               (q0,)),
    ("value_jac_curv pos",     jax.jit(lambda q, qd: value_jac_curv(pos_map, q, qd)[2]),  lambda q, qd: value_jac_curv(pos_map, q, qd)[2],  (q0, qd0)),
    ("value_jac_curv pose",    jax.jit(lambda q, qd: value_jac_curv(pose_map, q, qd)[2]), lambda q, qd: value_jac_curv(pose_map, q, qd)[2], (q0, qd0)),
    ("obstacle_geom leaf",     jax.jit(lambda q, qd: obst(q, qd, P0).f),       lambda q, qd: obst(q, qd, P0).f,               (q0, qd0)),
    ("pose_attractor leaf",    jax.jit(lambda q, qd: poseL(q, qd, P0).f),      lambda q, qd: poseL(q, qd, P0).f,              (q0, qd0)),
    (f"self_collision leaf (k={len(pairs)})", jax.jit(lambda q, qd: selfL(q, qd, P0).f), lambda q, qd: selfL(q, qd, P0).f,   (q0, qd0)),
    ("resolve (7x7 SPD)",      jax.jit(lambda M, f: resolve(Spec(M, f), 1e-6)), lambda M, f: resolve(Spec(M, f), 1e-6),      (M_spd, f_rand)),
]
print(f"  {'component':<28}{'min (us)':>10}{'median (us)':>14}{'compute-only':>15}")
for name, jfn, raw, args in comps:
    _, mn, md = per_call(jfn, args)
    print(f"  {name:<28}{mn:>10.1f}{md:>14.1f}{compute_only(raw, args):>15.2f}")
print("  (the FK primal is cheap; J + curvature dominate; config-space leaves + resolve are ~free)\n")

# ===========================================================================
print("=" * 80)
print("SECTION C — scaling: N obstacle barriers, separate leaves vs one batched sdf_barrier leaf")
print("=" * 80)
grid = np.stack(np.meshgrid(np.linspace(0.3, 0.6, 4), np.linspace(-0.3, 0.3, 4),
                            np.linspace(0.1, 0.5, 4)), -1).reshape(-1, 3)
forcing = [pose_attractor(prov, k=36.0, b=12.0), posture(nq, weight=1.0)]
damp = [config_damping(nq, b=2.0)]


def batched_obstacles(centers):
    C = jnp.asarray(centers, jnp.float32)
    return sdf_barrier_geometry(lambda q, p: jnp.linalg.norm(prov.site_pos(q) - C, axis=1) - 0.08, d0=0.12)


def fab_with(geoms):
    return GeometricFabric(geometries=geoms, forcing=forcing, damping=damp,
                           energy=fixed_metric_energy(nq, jnp.float32))


print(f"  {'N':>4} | {'separate leaves  (compile ms / min us)':>40} | {'batched leaf  (compile ms / min us)':>38} | match")
for N in (1, 2, 4, 8, 16, 32, 64):
    cs = grid[:N]
    fm = fab_with([obstacle_geometry(prov, tuple(c), 0.08, d0=0.12) for c in cs])
    fb = fab_with([batched_obstacles(cs)])
    cm, mnm, _ = per_call(fm.policy, (q0, qd0, P0))
    cb, mnb, _ = per_call(fb.policy, (q0, qd0, P0))
    match = float(np.max(np.abs(np.asarray(fm.policy(q0, qd0, P0)) - np.asarray(fb.policy(q0, qd0, P0)))))
    print(f"  {N:>4} | {cm:>22.0f} {mnm:>16.1f} | {cb:>20.0f} {mnb:>16.1f} | {match:.0e}")
print("  match = max|qddot_separate - qddot_batched| (bit-identical math; separate compile grows superlinearly)\n")

# ===========================================================================
print("=" * 80)
print("SECTION D — hygiene: recompilation, dtype")
print("=" * 80)
fab = collision_fabric()
traces = [0]
def counted(q, qd, p):
    traces[0] += 1
    return fab._policy(q, qd, p)
jc = jax.jit(counted)
for tgt in [(0.4, 0.1, 0.5), (0.5, -0.2, 0.4), (0.45, 0.0, 0.3)]:
    for obs in [(0.5, 0.0, 0.15), (0.4, 0.1, 0.2)]:
        jax.block_until_ready(jc(q0, qd0, params(tgt, obs)))
out = fab.policy(q0, qd0, P0)
print(f"  6 distinct param values dispatched -> {traces[0]} policy compile(s)  "
      f"[{'OK' if traces[0] == 1 else 'WARN: recompiling!'}]")
print(f"  qddot dtype {out.dtype}, finite {bool(jnp.all(jnp.isfinite(out)))}")
