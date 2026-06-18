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
from typing import NamedTuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root on sys.path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

from fabrix import (CustomFK, FabricParams, GeometricFabric, arm_obstacle_geometry,
                    arm_obstacle_potential, arm_plane_geometry, arm_plane_potential, cspace_attractor,
                    fixed_metric_energy, joint_limit_geometry, joint_limit_potential,
                    joint_speed_limit, limit_accel, load_spheres, nonadjacent_pairs, pose_attractor,
                    posture, self_collision_geometry, self_collision_potential, speed_control)
from fabrix.collision import _centers

TUNED_SPHERES = pathlib.Path(__file__).with_name("spheres_tuned.py")  # written by demos/tune_spheres.py

GEN3 = "mujoco_menagerie/kinova_gen3/gen3.xml"
SCENE = "mujoco_menagerie/kinova_gen3/scene.xml"
OBSTACLE_CENTER = (0.5, 0.0, 0.15)   # initial spot; in front, low; clear of the home EE (~0.43 up)
OBSTACLE_RADIUS = 0.08
FLOOR_Z = 0.0                        # the scene's ground plane
DT = 0.01         # 100 Hz control — one policy eval per render; ~5x cooler than 500 Hz on a laptop
SUB = 1           # one control step per render; the loop caps the wall rate to 1/(SUB*DT) = 100 Hz


class Gains(NamedTuple):
    """Every live-tunable fabric gain, as a JAX pytree the slider panel rewrites each step.

    These ride in ``FabricParams.gains`` and the fabric is built with *callable* gains
    (``lambda p: p.gains.<field>``), so changing any value is a **traced input change**: the next
    control step uses it with **no recompile** (the same trick that lets ``params.target`` move every
    frame). The defaults are the known-good M3 tuning, so ``Gains()`` reproduces the prior behavior.

    Each barrier splits into a wider GEOMETRY deflection band + a tighter POTENTIAL hard wall;
    ``*_d0`` = onset distance (m), ``*_kb``/``*_kp`` = strength, ``*_mb``/``*_mp`` = priority metric.
    """
    # task pose attractor: stiffness, damping (≈2√k critical), priority metric, saturating accel cap
    pose_k: float = 36.0
    pose_b: float = 12.0
    pose_m: float = 50.0
    pose_fmax: float = 10.0
    # distance-scaled attractor metric (D1): high near goal → dominates posture (no offset/orbit), low far.
    # pose_m_max == pose_m ⇒ constant (legacy); raise (~150-300) to engage. m(‖e‖) switches at pose_m_offset.
    pose_m_max: float = 50.0
    pose_m_sharp: float = 10.0
    pose_m_offset: float = 0.1
    # posture (nullspace → upright nominal) + global joint damping
    posture_w: float = 2.0
    posture_k: float = 2.0
    posture_b: float = 2.83
    # NVlabs-aligned config-space attractor (HD2 energized geometry toward home): saturating conical pull
    # ×‖q̇‖² (gentle during dexterous moves, zero force at rest). cspace_w=0 ⇒ inert (linear posture leads);
    # set cspace_w>0 (+ posture_w 0) to A/B the aligned mode. cspace_gain caps the far-field, near slope = gain·sharp.
    cspace_w: float = 0.0
    cspace_gain: float = 8.0
    cspace_sharp: float = 10.0
    cfg_damp: float = 2.0
    # speed control / KE cap (D2): cfg_damp is baseline damping b; speed_beta is the overspeed boost when
    # E=½‖q̇‖² exceeds speed_E_max (human-proximity safety). speed_beta=0 ⇒ no cap = constant config_damping.
    speed_beta: float = 0.0
    speed_E_max: float = 5.0     # demo: finite high cap so the slider has a range; speed_beta=0 ⇒ still off
    speed_k_gate: float = 20.0
    # obstacle barrier (whole-arm)
    obst_geom_d0: float = 0.06
    obst_geom_kb: float = 1.0
    obst_geom_mb: float = 2.0
    obst_pot_d0: float = 0.02
    obst_pot_kp: float = 0.5
    obst_pot_mp: float = 4.0
    # self-collision barrier (non-adjacent link pairs)
    self_geom_d0: float = 0.03
    self_geom_kb: float = 1.0
    self_geom_mb: float = 2.0
    self_pot_d0: float = 0.015
    self_pot_kp: float = 0.1
    self_pot_mp: float = 2.0
    # floor barrier (ground plane)
    floor_geom_d0: float = 0.03
    floor_geom_kb: float = 1.0
    floor_geom_mb: float = 2.0
    floor_pot_d0: float = 0.015
    floor_pot_kp: float = 0.5
    floor_pot_mp: float = 4.0
    # integrator safety caps (a real servo enforces these; they bound the per-step jump so a fast
    # obstacle drag can't fling the arm to NaN) + the draggable obstacle's radius (m)
    qdd_max: float = 50.0
    qd_max: float = 4.0
    # per-joint velocity-limit barrier: smoothly decelerate a joint near its qd_max bound (0 = off).
    # (Direction-preserving accel scaling to qdd_max is always on, replacing the per-axis clip.)
    qsl_kb: float = 0.0
    qsl_mb: float = 0.0
    qsl_d0: float = 0.5
    obstacle_radius: float = OBSTACLE_RADIUS


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
    # Build every leaf with a CALLABLE gain that reads params.gains.<name>, so a slider retunes it
    # live (traced input -> no recompile). g("x") closes over its own name (a function arg, no
    # late-binding bug). With a fixed Gains() in params this is identical to the old baked constants.
    g = lambda name: (lambda p: getattr(p.gains, name))
    fab = GeometricFabric(
        geometries=[arm_obstacle_geometry(prov, sph, None, g("obstacle_radius"),
                                          k_b=g("obst_geom_kb"), m_b=g("obst_geom_mb"), d0=g("obst_geom_d0")),
                    self_collision_geometry(prov, sph, pairs,
                                            k_b=g("self_geom_kb"), m_b=g("self_geom_mb"), d0=g("self_geom_d0")),
                    joint_limit_geometry(prov),
                    arm_plane_geometry(prov, sph, floor_pt, floor_n,
                                       k_b=g("floor_geom_kb"), m_b=g("floor_geom_mb"), d0=g("floor_geom_d0")),
                    cspace_attractor(nq, gain=g("cspace_gain"), sharp=g("cspace_sharp"), weight=g("cspace_w"))],
        forcing=[pose_attractor(prov, k=g("pose_k"), b=g("pose_b"), m=g("pose_m"), f_max=g("pose_fmax"),
                                m_max=g("pose_m_max"), sharp=g("pose_m_sharp"), offset=g("pose_m_offset")),
                 posture(nq, k=g("posture_k"), b=g("posture_b"), weight=g("posture_w")),
                 arm_obstacle_potential(prov, sph, None, g("obstacle_radius"),
                                        k_p=g("obst_pot_kp"), m_p=g("obst_pot_mp"), d0=g("obst_pot_d0")),
                 self_collision_potential(prov, sph, pairs,
                                          k_p=g("self_pot_kp"), m_p=g("self_pot_mp"), d0=g("self_pot_d0")),
                 arm_plane_potential(prov, sph, floor_pt, floor_n,
                                     k_p=g("floor_pot_kp"), m_p=g("floor_pot_mp"), d0=g("floor_pot_d0")),
                 joint_limit_potential(prov),
                 joint_speed_limit(nq, qd_lim=g("qd_max"), k_b=g("qsl_kb"), m_b=g("qsl_mb"), d0=g("qsl_d0"))],
        damping=[speed_control(nq, b=g("cfg_damp"), beta_speed=g("speed_beta"),
                               E_max=g("speed_E_max"), k_gate=g("speed_k_gate"))],
        energy=fixed_metric_energy(nq, jnp.float32))

    @jax.jit
    def advance(q, qd, params):  # SUB semi-implicit Euler steps, one dispatch per render
        qdd_max, qd_max = params.gains.qdd_max, params.gains.qd_max   # caps are live too (traced)
        def body(c, _):
            q, qd = c
            qdd = limit_accel(fab.policy(q, qd, params), qdd_max)          # direction-preserving accel cap
            qd = jnp.clip(qd + DT * qdd, -qd_max, qd_max)                  # ...and velocity -> no fly-away
            q = q + DT * qd
            return (q, qd), None
        (q, qd), _ = jax.lax.scan(body, (q, qd), None, length=SUB)
        return q, qd

    return prov, advance, sph


# --- live tuning panel: an optional viser web UI (same stack as the sphere tuner). Each (field, label,
#     min, max, step); falls back to a fixed Gains() if viser is absent so the plain run stays lean. ---
_GUI = [
    ("Attractor", [("pose_k", "k stiffness", 1.0, 300.0, 1.0), ("pose_b", "b damping", 0.0, 60.0, 0.5),
                   ("pose_m", "m far (floor)", 1.0, 200.0, 1.0), ("pose_fmax", "f_max accel cap", 1.0, 60.0, 0.5),
                   ("pose_m_max", "m near-goal (D1)", 1.0, 400.0, 1.0), ("pose_m_offset", "m switch r (m)", 0.01, 0.5, 0.01)]),
    ("Posture / damping", [("posture_w", "posture weight", 0.0, 20.0, 0.1), ("posture_k", "posture k", 0.0, 20.0, 0.1),
                           ("posture_b", "posture b", 0.0, 20.0, 0.05), ("cfg_damp", "damping b (cruise)", 0.0, 20.0, 0.1),
                           ("speed_beta", "overspeed β (D2)", 0.0, 100.0, 1.0), ("speed_E_max", "KE cap ½‖q̇‖²", 0.05, 5.0, 0.05)]),
    ("cspace attractor (NVlabs HD2)", [("cspace_w", "cspace weight (0=off)", 0.0, 20.0, 0.1),
                                       ("cspace_gain", "conical gain (far cap)", 0.0, 100.0, 1.0),
                                       ("cspace_sharp", "conical sharp (near slope)", 1.0, 40.0, 1.0)]),
    ("Obstacle barrier", [("obst_geom_d0", "geom d0 (m)", 0.005, 0.30, 0.005), ("obst_geom_kb", "geom k_b", 0.0, 5.0, 0.05),
                          ("obst_geom_mb", "geom m_b", 0.0, 10.0, 0.1), ("obst_pot_d0", "wall d0 (m)", 0.005, 0.20, 0.005),
                          ("obst_pot_kp", "wall k_p", 0.0, 5.0, 0.05), ("obst_pot_mp", "wall m_p", 0.0, 10.0, 0.1)]),
    ("Self-collision barrier", [("self_geom_d0", "geom d0 (m)", 0.005, 0.20, 0.005), ("self_geom_kb", "geom k_b", 0.0, 5.0, 0.05),
                                ("self_geom_mb", "geom m_b", 0.0, 10.0, 0.1), ("self_pot_d0", "wall d0 (m)", 0.005, 0.15, 0.005),
                                ("self_pot_kp", "wall k_p", 0.0, 5.0, 0.05), ("self_pot_mp", "wall m_p", 0.0, 10.0, 0.1)]),
    ("Floor barrier", [("floor_geom_d0", "geom d0 (m)", 0.005, 0.30, 0.005), ("floor_geom_kb", "geom k_b", 0.0, 5.0, 0.05),
                       ("floor_geom_mb", "geom m_b", 0.0, 10.0, 0.1), ("floor_pot_d0", "wall d0 (m)", 0.005, 0.20, 0.005),
                       ("floor_pot_kp", "wall k_p", 0.0, 5.0, 0.05), ("floor_pot_mp", "wall m_p", 0.0, 10.0, 0.1)]),
    ("Integrator / obstacle", [("qdd_max", "QDD_MAX rad/s²", 5.0, 500.0, 5.0), ("qd_max", "QD_MAX rad/s", 0.5, 20.0, 0.5),
                               ("qsl_kb", "vel-limit k_b", 0.0, 30.0, 0.5), ("qsl_mb", "vel-limit m_b", 0.0, 60.0, 1.0),
                               ("obstacle_radius", "obstacle radius (m)", 0.02, 0.20, 0.005)]),
]


class _TunePanel:
    """viser slider panel that emits a live :class:`Gains` each step + shows tracking/clearance readouts."""

    def __init__(self, model, obs_geom_id):
        import viser
        self.model, self.obs_geom_id = model, obs_geom_id
        self.server = viser.ViserServer()
        self.server.gui.add_markdown("**Live fabric tuning** — drag a slider; it takes effect next frame (no recompile).")
        d = Gains()._asdict()
        self.sl = {}
        for folder, items in _GUI:
            with self.server.gui.add_folder(folder):
                for name, label, lo, hi, step in items:
                    self.sl[name] = self.server.gui.add_slider(label, min=lo, max=hi, step=step,
                                                               initial_value=float(d[name]))
        with self.server.gui.add_folder("Readouts"):
            self.readout = self.server.gui.add_markdown("…")
        rb = self.server.gui.add_button("reset to defaults")

        @rb.on_click
        def _(_):
            for name, v in Gains()._asdict().items():
                if name in self.sl:
                    self.sl[name].value = float(v)

    def gains(self):
        gv = Gains(**{name: jnp.float32(s.value) for name, s in self.sl.items()})
        self.model.geom_size[self.obs_geom_id, 0] = float(gv.obstacle_radius)   # resize the red ball to match
        return gv

    def set_readout(self, text):
        self.readout.content = text


def _make_panel(model, obs_geom_id):
    try:
        return _TunePanel(model, obs_geom_id)
    except Exception as e:                                     # viser not installed / failed to start
        print(f"[tune] live panel unavailable ({type(e).__name__}: {e}); running with fixed Gains(). "
              f"For sliders: uv run --group viz mjpython demos/interactive_track.py")
        return None


def main():
    import mujoco.viewer

    model, tgt_id, obs_id = build_model()
    prov, advance, sph = _controller()
    nq = prov.nq
    centers_fn = jax.jit(_centers(prov, jnp.asarray(sph.link), jnp.asarray(sph.local)))
    pairs = nonadjacent_pairs(sph, prov)                      # for the live self-collision readout
    rsum = sph.radius[pairs[:, 0]] + sph.radius[pairs[:, 1]]
    sph_rgba = np.array([0.2, 0.6, 1.0, 0.35], np.float32)   # translucent blue collision spheres
    eye = np.eye(3).flatten()
    obs_geom_id = int(model.body_geomadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")])

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)               # "home"
    q_home = jnp.asarray(data.qpos[:nq], jnp.float32)
    q, qd = q_home, jnp.zeros(nq, jnp.float32)
    # start the target at the EE pose (no jump in position or orientation); obstacle at its spot
    data.mocap_pos[tgt_id] = np.asarray(prov.site_pos(q))
    data.mocap_quat[tgt_id] = np.asarray(prov.site_rot(q))
    data.mocap_pos[obs_id] = np.asarray(OBSTACLE_CENTER)

    panel = _make_panel(model, obs_geom_id)                   # viser sliders (None if viz not installed)
    print(__doc__)
    print(">>> double-click the GREEN target or RED obstacle, then Ctrl + right-drag (move) "
          "/ Ctrl + left-drag (rotate the target) <<<")
    print(">>> translucent blue = the arm's collision spheres (whole-arm + self-collision avoidance) <<<")
    print(">>> open the viser URL above to tune fabric gains LIVE (no recompile) <<<\n" if panel else "")
    rate = RateLimiter(frequency=1.0 / (SUB * DT), warn=False)   # cap the loop to 1/(SUB*DT) = 100 Hz
    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            gains = panel.gains() if panel else Gains()       # live slider values (or the fixed defaults)
            obs = np.asarray(data.mocap_pos[obs_id])
            params = FabricParams(target=jnp.asarray(data.mocap_pos[tgt_id], jnp.float32),
                                  q_default=q_home,
                                  target_quat=jnp.asarray(data.mocap_quat[tgt_id], jnp.float32),
                                  obstacle_center=jnp.asarray(obs, jnp.float32), gains=gains)
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
            if panel and frame % 8 == 0:                      # ~12 Hz readouts (cheap numpy on the centers)
                ee = np.asarray(prov.site_pos(q))
                pe = float(np.linalg.norm(ee - np.asarray(data.mocap_pos[tgt_id]))) * 1e3
                oc = float((np.linalg.norm(c - obs, axis=1) - (sph.radius + float(gains.obstacle_radius))).min()) * 1e3
                fc = float((c[:, 2] - sph.radius).min()) * 1e3
                sg = float((np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1) - rsum).min()) * 1e3
                qd_np = np.asarray(qd)
                spd, ke = float(np.linalg.norm(qd_np)), 0.5 * float(qd_np @ qd_np)
                cap = f" / cap {float(gains.speed_E_max):.2f}" if float(gains.speed_beta) > 0 else ""
                panel.set_readout(f"EE err **{pe:.1f} mm** · obstacle **{oc:+.0f} mm** · floor **{fc:+.0f} mm** "
                                  f"· self **{sg:+.0f} mm** · speed **{spd:.2f}** rad/s · KE **{ke:.2f}**{cap}")
            frame += 1
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
        return FabricParams(target=tgt, q_default=q_home, target_quat=quat_home, obstacle_center=obs,
                            gains=Gains())

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
                         obstacle_center=jnp.asarray([ee[0], ee[1] + s, ee[2]], jnp.float32), gains=Gains())
        q, qd = advance(q, qd, p)
        finite = finite and bool(jnp.all(jnp.isfinite(q)))
        qmax = max(qmax, float(jnp.max(jnp.abs(q))))
    print(f"[stability] obstacle slammed into the arm: q finite={finite}, max|q|={qmax:.1f} rad "
          f"({'STABLE' if finite and qmax < 10 else 'BLEW UP'})")
    assert finite and qmax < 10.0, "fast obstacle drag blew up the integrator"

    # (5) live tuning is a *traced* input change, not a recompile: retuning pose_k mid-run keeps the
    #     jit cache fixed and still takes effect (a stiffer k tracks a step target faster).
    tgt5 = jnp.asarray([ee[0] + 0.15, ee[1], ee[2]], jnp.float32)
    n_cache = advance._cache_size()

    def _track(pose_k):
        qq, qv = q_home, jnp.zeros(nq, jnp.float32)
        p = FabricParams(target=tgt5, q_default=q_home, target_quat=quat_home, obstacle_center=obs,
                         gains=Gains(pose_k=pose_k))
        for _ in range(40):
            qq, qv = advance(qq, qv, p)
        return float(np.linalg.norm(np.asarray(prov.site_pos(qq)) - np.asarray(tgt5)))

    err_soft, err_stiff = _track(6.0), _track(120.0)
    assert advance._cache_size() == n_cache, "a live gain change forced a recompile"
    assert err_stiff < err_soft, "stiffer pose_k did not track faster"
    print(f"[live-tuning] gains are traced (no recompile; jit cache={n_cache}); stiffer k tracks faster "
          f"({err_soft*1e3:.0f} -> {err_stiff*1e3:.0f} mm EE error)")
    # D1/D2/D-vel gains are traced too: toggling dynamic mass, the KE cap, and the velocity-limit
    # barrier mid-run must not recompile.
    for gn in (Gains(pose_m_max=300.0, pose_m_offset=0.08), Gains(speed_beta=60.0, speed_E_max=0.3),
               Gains(qsl_kb=5.0, qsl_mb=20.0),
               Gains(posture_w=0.0, cspace_w=1.0, cspace_gain=8.0)):  # toggle the NVlabs HD2 cspace attractor
        advance(q_home, jnp.zeros(nq, jnp.float32),
                FabricParams(target=tgt5, q_default=q_home, target_quat=quat_home, obstacle_center=obs, gains=gn))
    assert advance._cache_size() == n_cache, "a D1/D2/vel-limit/cspace gain change forced a recompile"
    print(f"[live-tuning] dynamic-mass + KE-cap + vel-limit + cspace gains also traced; jit cache still {n_cache}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
