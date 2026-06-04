"""Interactive demo: drag a target, watch the Gen3 track it while avoiding an obstacle.

A live MuJoCo viewer. Double-click the GREEN target sphere, then Ctrl + right-drag to move it;
the geometric fabric drives the end-effector to follow — smoothly routing around the RED obstacle
and respecting joint limits. The arm shows the *commanded* configuration (the fabric integrated to
a joint reference), i.e. the C2-smooth stream a real servo would track quietly.

macOS needs mjpython (the interactive viewer must own the main thread):

    uv run mjpython demos/interactive_track.py        # macOS
    uv run python   demos/interactive_track.py        # Linux / Windows

Headless self-check (no GUI; scripts a target sweep across the obstacle and reports tracking +
clearance):

    uv run python demos/interactive_track.py --check
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root on sys.path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from fabrix import (CustomFK, FabricParams, GeometricFabric, attractor, config_damping,
                    fixed_metric_energy, joint_limit_geometry, joint_limit_potential,
                    obstacle_geometry, obstacle_potential, posture)

GEN3 = "mujoco_menagerie/kinova_gen3/gen3.xml"
SCENE = "mujoco_menagerie/kinova_gen3/scene.xml"
OBSTACLE_CENTER = (0.5, 0.0, 0.15)   # in front, low; clear of the home EE (~0.43 up) at start
OBSTACLE_RADIUS = 0.08
DT = 0.002        # 500 Hz control
SUB = 8           # control steps per render (~60 Hz render)


def build_model():
    """`scene.xml` + a draggable mocap target + a static (visual-only) obstacle sphere."""
    spec = mujoco.MjSpec.from_file(SCENE)
    t = spec.worldbody.add_body(); t.name = "target"; t.mocap = True; t.pos = [0.45, 0.0, 0.5]
    tg = t.add_geom(); tg.type = mujoco.mjtGeom.mjGEOM_SPHERE; tg.size = [0.025, 0, 0]
    tg.rgba = [0.1, 0.9, 0.1, 0.8]; tg.contype = 0; tg.conaffinity = 0
    o = spec.worldbody.add_body(); o.name = "obstacle"; o.pos = list(OBSTACLE_CENTER)
    og = o.add_geom(); og.type = mujoco.mjtGeom.mjGEOM_SPHERE; og.size = [OBSTACLE_RADIUS, 0, 0]
    og.rgba = [0.9, 0.3, 0.3, 0.45]; og.contype = 0; og.conaffinity = 0
    model = spec.compile()
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    return model, int(model.body_mocapid[tb])


def _controller():
    """Build the provider, fabric, home state, and a jitted SUB-step integrator."""
    prov = CustomFK(GEN3)
    nq = prov.nq
    center = jnp.asarray(OBSTACLE_CENTER, jnp.float32)
    fab = GeometricFabric(
        geometries=[obstacle_geometry(prov, center, OBSTACLE_RADIUS), joint_limit_geometry(prov)],
        forcing=[attractor(prov), posture(nq),
                 obstacle_potential(prov, center, OBSTACLE_RADIUS), joint_limit_potential(prov)],
        damping=[config_damping(nq, b=6.0)],
        energy=fixed_metric_energy(nq, jnp.float32))

    @jax.jit
    def advance(q, qd, params):  # SUB semi-implicit Euler steps, one dispatch per render
        def body(c, _):
            q, qd = c
            qdd = fab.policy(q, qd, params)
            qd = qd + DT * qdd
            q = q + DT * qd
            return (q, qd), None
        (q, qd), _ = jax.lax.scan(body, (q, qd), None, length=SUB)
        return q, qd

    return prov, advance


def main():
    import mujoco.viewer

    model, mocap_id = build_model()
    prov, advance = _controller()
    nq = prov.nq

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)               # "home"
    q_home = jnp.asarray(data.qpos[:nq], jnp.float32)
    q, qd = q_home, jnp.zeros(nq, jnp.float32)
    data.mocap_pos[mocap_id] = np.asarray(prov.site_pos(q))   # start the target at the EE (no jump)

    print(__doc__)
    print(">>> drag the GREEN target: double-click it, then Ctrl + right-drag <<<\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tic = time.perf_counter()
            params = FabricParams(target=jnp.asarray(data.mocap_pos[mocap_id], jnp.float32),
                                  q_default=q_home)
            q, qd = advance(q, qd, params)
            data.qpos[:nq] = np.asarray(q)
            mujoco.mj_forward(model, data)
            viewer.sync()
            lag = SUB * DT - (time.perf_counter() - tic)
            if lag > 0:
                time.sleep(lag)


def check():
    """Headless: sweep the target straight across the obstacle; report tracking + clearance."""
    model, mocap_id = build_model()
    prov, advance = _controller()
    nq = prov.nq
    center = np.asarray(OBSTACLE_CENTER)
    q_home = jnp.asarray(model.key_qpos[0, :nq], jnp.float32)
    q, qd = q_home, jnp.zeros(nq, jnp.float32)

    # target sweeps left->right at the obstacle's height; the straight line passes through it
    ys = np.linspace(-0.35, 0.35, 600)
    min_clear, max_track = np.inf, 0.0
    for y in ys:
        tgt = jnp.asarray([center[0], y, center[2]], jnp.float32)
        q, qd = advance(q, qd, FabricParams(target=tgt, q_default=q_home))
        ee = np.asarray(prov.site_pos(q))
        min_clear = min(min_clear, np.linalg.norm(ee - center) - OBSTACLE_RADIUS)
        max_track = max(max_track, np.linalg.norm(ee - np.asarray(tgt)))
    print(f"[check] target swept across obstacle: min clearance {min_clear*1e3:+.1f} mm "
          f"({'CLEAR' if min_clear >= 0 else 'PENETRATED'}); peak tracking lag {max_track*1e3:.0f} mm "
          f"(expected — the target line goes through the obstacle, so the EE detours)")
    assert bool(jnp.all(jnp.isfinite(q)))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
