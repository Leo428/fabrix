"""M2 demo: Kinova Gen3 reaches a target while routing around a sphere obstacle.

A full geometric fabric: an energized obstacle geometry (reactive, path-consistent, speed-
regulated) + a barrier *potential* (the hard non-penetration guarantee) + the M1 attractor and
damping. Produces `demos/obstacle_reach.png`:

  - end-effector error decays to ~0 (goal reached),
  - distance-to-surface stays >= 0 (obstacle never penetrated),
  - the commanded q_ddot is bounded and continuous (C2 -> quiet on hardware),
  - a 3-D view of the end-effector path bending around the obstacle.

Run:  uv run python demos/obstacle_reach.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root on sys.path

import statistics
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fabrix import (CustomFK, FabricParams, GeometricFabric, attractor, config_damping,
                    fixed_metric_energy, obstacle_geometry, obstacle_potential, rollout)

XML = "mujoco_menagerie/kinova_gen3/gen3.xml"
DT = 0.002            # 500 Hz control
STEPS = 2500          # 5.0 s
RADIUS = 0.10
OUT = "demos/obstacle_reach.png"


def main():
    prov = CustomFK(XML)
    nq = prov.nq
    q0 = jnp.zeros(nq, jnp.float32)
    target = prov.site_pos(q0.at[1].set(0.4).at[3].set(0.5).at[5].set(-0.3))

    p_home = np.asarray(prov.site_pos(q0))
    p_tgt = np.asarray(target)
    # sphere offset off the direct line so the straight path is blocked but the goal is reachable
    center = jnp.asarray(p_home + 0.5 * (p_tgt - p_home) + np.array([0.0, 0.06, 0.0], np.float32))

    fabric = GeometricFabric(
        geometries=[obstacle_geometry(prov, center, RADIUS, k_b=1.0, m_b=2.0)],
        forcing=[attractor(prov), obstacle_potential(prov, center, RADIUS, k_p=0.6, d0=0.2, m_p=6.0)],
        damping=[config_damping(nq, b=6.0)],
        energy=fixed_metric_energy(nq, jnp.float32))
    params = FabricParams(target=target, q_default=q0)
    qd0 = jnp.zeros(nq, jnp.float32)

    traj = rollout(fabric.policy, q0, qd0, params, DT, STEPS, prov.site_pos)
    qdd = np.asarray(traj["qdd"])
    ee = np.asarray(traj["ee"])
    c = np.asarray(center)
    err = np.linalg.norm(ee - p_tgt, axis=1)
    surf = np.linalg.norm(ee - c, axis=1) - RADIUS  # distance to obstacle surface (>=0 means clear)
    t = np.arange(STEPS) * DT

    print(f"final EE error      : {err[-1] * 1e3:.3f} mm")
    print(f"min dist to surface : {surf.min() * 1e3:.1f} mm  ({'CLEAR' if surf.min() >= 0 else 'PENETRATED'})")
    print(f"max step-to-step |d q_ddot| : {np.abs(np.diff(qdd, axis=0)).max():.4f}  (small => C2)")

    fabric.policy(q0, qd0, params).block_until_ready()
    ts = []
    for _ in range(2000):
        s = time.perf_counter()
        fabric.policy(q0, qd0, params).block_until_ready()
        ts.append(time.perf_counter() - s)
    print(f"policy latency      : min {min(ts) * 1e6:.1f} us  median {statistics.median(ts) * 1e6:.1f} us "
          f"(1 kHz budget = 1000 us)")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t, err * 1e3, color="k")
    ax.set(title="end-effector position error", xlabel="t (s)", ylabel="mm"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, surf * 1e3, color="C3")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="obstacle surface")
    ax.set(title="distance to obstacle surface  (>= 0 => never penetrated)", xlabel="t (s)", ylabel="mm")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 3)
    for j in range(nq):
        ax.plot(t, qdd[:, j])
    ax.set(title="commanded q_ddot  (bounded & continuous -> C2, quiet on hardware)",
           xlabel="t (s)", ylabel="rad/s^2"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="C0", lw=2, label="EE path")
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:12j]
    ax.plot_wireframe(c[0] + RADIUS * np.cos(u) * np.sin(v), c[1] + RADIUS * np.sin(u) * np.sin(v),
                      c[2] + RADIUS * np.cos(v), color="C3", alpha=0.3, lw=0.5)
    ax.scatter(*p_home, color="k", s=30, label="start")
    ax.scatter(*p_tgt, color="g", s=40, marker="*", label="target")
    ax.set(title="end-effector path around obstacle", xlabel="x", ylabel="y", zlabel="z")
    ax.legend(fontsize=8)

    fig.suptitle("fabrix M2 - energized obstacle fabric on Kinova Gen3", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
