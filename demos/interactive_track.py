"""Interactive demo: drag a target, watch the Gen3 track it while avoiding obstacles.

A live MuJoCo viewer. Double-click the GREEN target, then Ctrl + right-drag to move it and
Ctrl + left-drag to rotate it; the geometric fabric drives the end-effector to follow that full
6-DOF pose (M3) — smoothly routing around the RED obstacle (also draggable), staying above the
FLOOR, and respecting joint limits. Avoidance is **whole-arm**: the translucent blue collision
spheres (auto-placed on every link) all dodge the obstacle and floor, and a self-collision barrier
keeps the arm from folding into itself. The arm shows the *commanded* configuration (the fabric
integrated to a joint reference), i.e. the C2-smooth stream a real servo would track quietly.

macOS needs mjpython (the interactive viewer must own the main thread):

    uv run mjpython demos/interactive_track.py        # macOS
    uv run python   demos/interactive_track.py        # Linux / Windows

Headless self-check (no GUI; sweeps the target across the obstacle, down toward the floor, and folds
the arm inward, reporting whole-arm + self-collision clearances):

    uv run python demos/interactive_track.py --check
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root on sys.path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

from fabrix import (CustomFK, FabricParams, GeometricFabric, arm_obstacle_geometry,
                    arm_obstacle_potential, arm_plane_geometry, arm_plane_potential,
                    config_damping, fixed_metric_energy, joint_limit_geometry,
                    joint_limit_potential, load_spheres, nonadjacent_pairs, pose_attractor, posture,
                    self_collision_geometry, self_collision_potential)
from fabrix.collision import _centers

TUNED_SPHERES = pathlib.Path(__file__).with_name("spheres_tuned.py")  # written by demos/tune_spheres.py

GEN3 = "mujoco_menagerie/kinova_gen3/gen3.xml"
SCENE = "mujoco_menagerie/kinova_gen3/scene.xml"
OBSTACLE_CENTER = (0.5, 0.0, 0.15)   # initial spot; in front, low; clear of the home EE (~0.43 up)
OBSTACLE_RADIUS = 0.08
FLOOR_Z = 0.0                        # the scene's ground plane
DT = 0.01         # 100 Hz control — one policy eval per render; ~5x cooler than 500 Hz on a laptop
SUB = 1           # one control step per render; the loop caps the wall rate to 1/(SUB*DT) = 100 Hz
# Safety caps on the integrated reference. The barrier's approach-rate term sees only the ARM's motion,
# not the obstacle's, so a *fast-dragged* obstacle isn't anticipated — d collapses, the 1/d^2 wall spikes,
# and explicit Euler at 100 Hz would fling the arm to NaN. These per-joint caps (a real servo has them)
# bound the per-step jump so it can't blow up; they only bite on such pathological inputs.
QDD_MAX = 50.0    # rad/s^2  reactive-acceleration cap
QD_MAX = 4.0      # rad/s    joint-velocity cap (bounds the per-step q change)

# Attractor / damping gains (tuned for responsiveness — "setting C": ~2x snappier than the M2
# defaults, still critically damped). Raise POSE_K (with POSE_B = 2*sqrt(POSE_K)) for crisper.
POSE_K, POSE_B = 36.0, 12.0
CFG_DAMP = 2.0
POSE_FMAX = 10.0   # saturating attractor: cap the restoring accel so far/commanded moves don't lunge
                   # (incremental drag is unaffected — it only bites past ~POSE_FMAX/POSE_K of error)
POSTURE_W = 2.0    # posture priority toward q_default (the upright nominal); a per-joint (nq,) array
                   # also works — bias the shoulder/elbow hard, leave the wrist free for the task
POSTURE_K, POSTURE_B = 2.0, 2.83   # posture stiffness toward q_default (b≈2√k, critically damped)
# Two distinct ranges (see the obstacle docstrings): the GEOMETRY sets how far out the arm begins
# to smoothly bend its path around a surface; the POTENTIAL is the tight hard wall that guarantees
# no penetration. Keep the deflection local and the wall tight.
OBST_GEOM_D0 = 0.06   # start bending the path within 6 cm of the obstacle surface
OBST_POT_D0 = 0.02    # hard-wall standoff: the non-penetration potential acts within 2 cm
FLOOR_GEOM_D0 = 0.03  # floor: start cushioning a sphere within 3 cm of the ground
FLOOR_POT_D0 = 0.015  # floor: hard no-penetration wall within 1.5 cm (tighter than the deflection band)
# Collision spheres: the obstacle/floor barriers now guard the WHOLE arm (every sphere), not just the
# EE, and a self-collision barrier keeps non-adjacent links apart. Spheres are auto-placed from the
# kinematics and drawn translucent blue in the viewer (for tuning); the count is nearly free (batched).
SELF_GEOM_D0 = 0.03   # self-collision: start deflecting two links apart within 3 cm of contact
SELF_POT_D0 = 0.015   # hard self-collision wall: the no-penetration potential acts within 1.5 cm


def build_model():
    """`scene.xml` + a draggable mocap target (green) + a draggable mocap obstacle (red)."""
    spec = mujoco.MjSpec.from_file(SCENE)
    t = spec.worldbody.add_body(); t.name = "target"; t.mocap = True; t.pos = [0.45, 0.0, 0.5]
    tg = t.add_geom(); tg.type = mujoco.mjtGeom.mjGEOM_SPHERE; tg.size = [0.025, 0, 0]
    tg.rgba = [0.1, 0.9, 0.1, 0.8]; tg.contype = 0; tg.conaffinity = 0
    o = spec.worldbody.add_body(); o.name = "obstacle"; o.mocap = True; o.pos = list(OBSTACLE_CENTER)
    og = o.add_geom(); og.type = mujoco.mjtGeom.mjGEOM_SPHERE; og.size = [OBSTACLE_RADIUS, 0, 0]
    og.rgba = [0.9, 0.3, 0.3, 0.45]; og.contype = 0; og.conaffinity = 0
    model = spec.compile()
    mid = lambda name: int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])
    return model, mid("target"), mid("obstacle")


def _controller():
    """Build the provider, sphere model, fabric, and a jitted SUB-step integrator.

    Whole-arm avoidance: the obstacle and floor barriers act on *every* collision sphere (auto-placed
    along the links), and a self-collision barrier keeps non-adjacent links apart — all as single
    batched leaves. The sphere obstacle is param-driven (``center=None`` -> ``params.obstacle_center``)
    so it can be dragged live; the floor is a static plane barrier on the scene's ground plane.
    """
    prov = CustomFK(GEN3)
    nq = prov.nq
    sph, tuned = load_spheres(prov, TUNED_SPHERES, n_per_link=2)   # hand-tuned file if present, else auto
    print(f"[spheres] {'hand-tuned (demos/spheres_tuned.py)' if tuned else 'auto-generated'}: "
          f"{len(sph)} collision spheres  (tune with: uv run --group viz python demos/tune_spheres.py)")
    pairs = nonadjacent_pairs(sph, prov)              # non-adjacent link pairs for self-collision
    floor_pt, floor_n = (0.0, 0.0, FLOOR_Z), (0.0, 0.0, 1.0)
    fab = GeometricFabric(
        geometries=[arm_obstacle_geometry(prov, sph, None, OBSTACLE_RADIUS, d0=OBST_GEOM_D0),
                    self_collision_geometry(prov, sph, pairs, d0=SELF_GEOM_D0),
                    joint_limit_geometry(prov),
                    arm_plane_geometry(prov, sph, floor_pt, floor_n, d0=FLOOR_GEOM_D0)],
        forcing=[pose_attractor(prov, k=POSE_K, b=POSE_B, f_max=POSE_FMAX),
                 posture(nq, k=POSTURE_K, b=POSTURE_B, weight=POSTURE_W),
                 arm_obstacle_potential(prov, sph, None, OBSTACLE_RADIUS, d0=OBST_POT_D0),
                 self_collision_potential(prov, sph, pairs, d0=SELF_POT_D0),
                 arm_plane_potential(prov, sph, floor_pt, floor_n, d0=FLOOR_POT_D0),
                 joint_limit_potential(prov)],
        damping=[config_damping(nq, b=CFG_DAMP)],
        energy=fixed_metric_energy(nq, jnp.float32))

    @jax.jit
    def advance(q, qd, params):  # SUB semi-implicit Euler steps, one dispatch per render
        def body(c, _):
            q, qd = c
            qdd = jnp.clip(fab.policy(q, qd, params), -QDD_MAX, QDD_MAX)   # bound reactive accel
            qd = jnp.clip(qd + DT * qdd, -QD_MAX, QD_MAX)                  # ...and velocity -> no fly-away
            q = q + DT * qd
            return (q, qd), None
        (q, qd), _ = jax.lax.scan(body, (q, qd), None, length=SUB)
        return q, qd

    return prov, advance, sph


def main():
    import mujoco.viewer

    model, tgt_id, obs_id = build_model()
    prov, advance, sph = _controller()
    nq = prov.nq
    centers_fn = jax.jit(_centers(prov, jnp.asarray(sph.link), jnp.asarray(sph.local)))
    sph_rgba = np.array([0.2, 0.6, 1.0, 0.35], np.float32)   # translucent blue collision spheres
    eye = np.eye(3).flatten()

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)               # "home"
    q_home = jnp.asarray(data.qpos[:nq], jnp.float32)
    q, qd = q_home, jnp.zeros(nq, jnp.float32)
    # start the target at the EE pose (no jump in position or orientation); obstacle at its spot
    data.mocap_pos[tgt_id] = np.asarray(prov.site_pos(q))
    data.mocap_quat[tgt_id] = np.asarray(prov.site_rot(q))
    data.mocap_pos[obs_id] = np.asarray(OBSTACLE_CENTER)

    print(__doc__)
    print(">>> double-click the GREEN target or RED obstacle, then Ctrl + right-drag (move) "
          "/ Ctrl + left-drag (rotate the target) <<<")
    print(">>> translucent blue = the arm's collision spheres (whole-arm + self-collision avoidance) <<<\n")
    rate = RateLimiter(frequency=1.0 / (SUB * DT), warn=False)   # cap the loop to 1/(SUB*DT) = 100 Hz
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            params = FabricParams(target=jnp.asarray(data.mocap_pos[tgt_id], jnp.float32),
                                  q_default=q_home,
                                  target_quat=jnp.asarray(data.mocap_quat[tgt_id], jnp.float32),
                                  obstacle_center=jnp.asarray(data.mocap_pos[obs_id], jnp.float32))
            q, qd = advance(q, qd, params)
            data.qpos[:nq] = np.asarray(q)
            mujoco.mj_forward(model, data)
            c = np.asarray(centers_fn(q))                     # draw the collision spheres as decorations
            viewer.user_scn.ngeom = 0
            for i in range(len(c)):
                mujoco.mjv_initGeom(viewer.user_scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([sph.radius[i], 0.0, 0.0]), c[i].astype(np.float64), eye, sph_rgba)
                viewer.user_scn.ngeom += 1
            viewer.sync()
            rate.sleep()                                       # hold the target rate without drift


def check():
    """Headless: sweep the target across the obstacle, then down toward the floor; report clearances
    and the distance at which the arm *begins* to detour around the obstacle."""
    model, _, _ = build_model()
    prov, advance, sph = _controller()
    nq = prov.nq
    center = np.asarray(OBSTACLE_CENTER)
    q_home = jnp.asarray(model.key_qpos[0, :nq], jnp.float32)
    quat_home = jnp.asarray(prov.site_rot(q_home), jnp.float32)  # hold orientation while sweeping
    obs = jnp.asarray(OBSTACLE_CENTER, jnp.float32)

    def params(tgt):
        return FabricParams(target=tgt, q_default=q_home, target_quat=quat_home, obstacle_center=obs)

    # (1) target sweeps left->right through the obstacle's center: the straight line penetrates it
    q, qd = q_home, jnp.zeros(nq, jnp.float32)
    for _ in range(400):                      # settle at the start so onset excludes startup catch-up
        q, qd = advance(q, qd, params(jnp.asarray([center[0], -0.35, center[2]], jnp.float32)))
    min_clear, onset = np.inf, 0.0
    for y in np.linspace(-0.35, 0.35, 600):
        tgt = jnp.asarray([center[0], y, center[2]], jnp.float32)
        q, qd = advance(q, qd, params(tgt))
        ee = np.asarray(prov.site_pos(q))
        clear = float(np.linalg.norm(ee - center) - OBSTACLE_RADIUS)
        # "detour" = deviation perpendicular to the sweep line (which runs along y at fixed x,z), so
        # it isolates the obstacle's sideways push from the harmless along-track lag behind a moving target
        perp = float(np.hypot(ee[0] - center[0], ee[2] - center[2]))
        min_clear = min(min_clear, clear)
        if perp > 0.01:                      # detouring; track the largest clearance where it does
            onset = max(onset, clear)
    print(f"[obstacle] swept through center: min EE clearance {min_clear*1e3:+.1f} mm "
          f"({'CLEAR' if min_clear >= 0 else 'PENETRATED'}); EE begins to deviate at "
          f"~{onset*1e3:.0f} mm clearance (whole-arm: a forearm/elbow sphere reaches the obstacle "
          f"before the EE does, so the EE deflects earlier than the EE-only band)")

    # (2) target descends toward the floor, away from the obstacle: tests the plane barrier
    q, qd = q_home, jnp.zeros(nq, jnp.float32)
    for _ in range(400):
        q, qd = advance(q, qd, params(jnp.asarray([0.45, 0.25, 0.45], jnp.float32)))
    min_z = np.inf
    for z in np.linspace(0.45, -0.10, 500):
        tgt = jnp.asarray([0.45, 0.25, z], jnp.float32)
        q, qd = advance(q, qd, params(tgt))
        min_z = min(min_z, float(prov.site_pos(q)[2]))
    print(f"[floor] target driven below ground: min EE height {min_z*1e3:+.1f} mm "
          f"({'ABOVE floor' if min_z >= FLOOR_Z else 'BELOW floor'})")

    # (3) self-collision: command the EE down-and-in so the arm folds back over itself; the
    #     self-collision barrier must hold the non-adjacent links apart (min sphere-pair gap >= 0)
    pairs = nonadjacent_pairs(sph, prov)
    cf = _centers(prov, jnp.asarray(sph.link), jnp.asarray(sph.local))
    rsum = sph.radius[pairs[:, 0]] + sph.radius[pairs[:, 1]]
    q, qd = q_home, jnp.zeros(nq, jnp.float32)
    min_gap = np.inf
    for tgt in ([0.0, 0.15, 0.2], [0.0, 0.0, 0.15], [-0.1, 0.0, 0.2], [0.0, -0.15, 0.15]):
        for _ in range(500):
            q, qd = advance(q, qd, params(jnp.asarray(tgt, jnp.float32)))
            c = np.asarray(cf(q))
            gaps = np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1) - rsum
            min_gap = min(min_gap, float(gaps.min()))
    print(f"[self-collision] folding the arm inward: min non-adjacent sphere-pair gap "
          f"{min_gap*1e3:+.1f} mm ({'CLEAR' if min_gap >= 0 else 'CONTACT'})")
    assert bool(jnp.all(jnp.isfinite(q)))

    # (4) stability: SLAM the obstacle into the arm at high speed (a fast drag). The barrier's approach
    #     rate sees only the arm's motion, not the obstacle's, so the 1/d^2 wall spikes; without the
    #     integrator's accel/velocity caps, explicit Euler at 100 Hz flings the arm to NaN. Must stay sane.
    q, qd = q_home, jnp.zeros(nq, jnp.float32)
    ee = np.asarray(prov.site_pos(q_home))
    tgt = jnp.asarray([ee[0], ee[1], ee[2]], jnp.float32)              # hold position; only the ball moves
    finite, qmax = True, 0.0
    for s in list(np.linspace(0.6, 0.0, 8)) + [0.0] * 8:              # rush in ~7.5 m/s, then sit on the EE
        p = FabricParams(target=tgt, q_default=q_home, target_quat=quat_home,
                         obstacle_center=jnp.asarray([ee[0], ee[1] + s, ee[2]], jnp.float32))
        q, qd = advance(q, qd, p)
        finite = finite and bool(jnp.all(jnp.isfinite(q)))
        qmax = max(qmax, float(jnp.max(jnp.abs(q))))
    print(f"[stability] obstacle slammed into the arm: q finite={finite}, max|q|={qmax:.1f} rad "
          f"({'STABLE' if finite and qmax < 10 else 'BLEW UP'})")
    assert finite and qmax < 10.0, "fast obstacle drag blew up the integrator"


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
