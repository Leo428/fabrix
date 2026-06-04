"""M1 demo: drive the Kinova Gen3 end-effector to a target with a forced attractor fabric.

Produces `demos/attractor_reach.png` showing the commanded q / q_dot / q_ddot streams. The
point: q_dot and q_ddot are continuous and bounded (a C2 reference) — exactly the smoothness
the first-order mink stream lacked, which is what makes a real servo quiet instead of vibrating.

Run:  uv run python demos/attractor_reach.py
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

from fabrix import CustomFK, Fabric, FabricParams, attractor, config_damping, posture, rollout

XML = "mujoco_menagerie/kinova_gen3/gen3.xml"
DT = 0.002            # 500 Hz control
STEPS = 1500          # 3.0 s
OUT = "demos/attractor_reach.png"


def main():
    prov = CustomFK(XML)
    nq = prov.nq
    q0 = jnp.zeros(nq, jnp.float32)
    # a reachable target: EE position at a perturbed configuration
    target = prov.site_pos(q0.at[1].set(0.4).at[3].set(0.5).at[5].set(-0.3))

    fabric = Fabric([attractor(prov), posture(nq), config_damping(nq)])
    params = FabricParams(target=target, q_default=q0)
    qd0 = jnp.zeros(nq, jnp.float32)

    # closed-loop rollout (single compiled lax.scan)
    traj = rollout(fabric.policy, q0, qd0, params, DT, STEPS, prov.site_pos)
    q = np.asarray(traj["q"]); qd = np.asarray(traj["qd"]); qdd = np.asarray(traj["qdd"])
    ee = np.asarray(traj["ee"]); tgt = np.asarray(target)
    err = np.linalg.norm(ee - tgt, axis=1)
    t = np.arange(STEPS) * DT

    print(f"final EE error : {err[-1] * 1e3:.3f} mm")
    print(f"max |q_ddot|   : {np.abs(qdd).max():.2f} rad/s^2")
    print(f"max step-to-step |d q_ddot| : {np.abs(np.diff(qdd, axis=0)).max():.4f} "
          f"(small => continuous acceleration => C2)")

    # per-step controller latency (after warmup/compile)
    fabric.policy(q0, qd0, params).block_until_ready()
    ts = []
    for _ in range(2000):
        s = time.perf_counter()
        fabric.policy(q0, qd0, params).block_until_ready()
        ts.append(time.perf_counter() - s)
    print(f"policy latency : min {min(ts) * 1e6:.1f} us  median {statistics.median(ts) * 1e6:.1f} us "
          f"(1 kHz budget = 1000 us)")

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, err * 1e3, color="k")
    ax[0, 0].set(title="end-effector position error", xlabel="t (s)", ylabel="mm")
    ax[0, 0].grid(True, alpha=0.3)
    for j in range(nq):
        ax[0, 1].plot(t, q[:, j], label=f"j{j + 1}")
    ax[0, 1].set(title="commanded q", xlabel="t (s)", ylabel="rad")
    ax[0, 1].grid(True, alpha=0.3); ax[0, 1].legend(fontsize=7, ncol=2)
    for j in range(nq):
        ax[1, 0].plot(t, qd[:, j])
    ax[1, 0].set(title="commanded q_dot  (continuous -> no velocity steps)",
                 xlabel="t (s)", ylabel="rad/s")
    ax[1, 0].grid(True, alpha=0.3)
    for j in range(nq):
        ax[1, 1].plot(t, qdd[:, j])
    ax[1, 1].set(title="commanded q_ddot  (bounded & continuous -> C2, quiet on hardware)",
                 xlabel="t (s)", ylabel="rad/s^2")
    ax[1, 1].grid(True, alpha=0.3)
    fig.suptitle("fabrix M1 - attractor fabric on Kinova Gen3", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
