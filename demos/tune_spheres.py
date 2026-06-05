"""Interactive collision-sphere tuner — a viser web UI.

Renders the Gen3's links (their visual meshes) with the collision spheres overlaid, and lets you
reshape the spheres in the browser, then export a hand-tuned model:

  * **select** a sphere from the dropdown or by clicking it (selected = yellow);
  * drag the **radius** slider to resize it, or the 3-axis **gizmo** to move it (in its link frame);
  * move precisely with the **x / y / z** inputs or the **nudge** buttons (viser has no global hotkeys);
  * **add / duplicate / delete** spheres on any link;
  * scrub the **pose** sliders to check coverage across the arm's range of motion;
  * **export** to ``demos/spheres_tuned.py`` — which ``demos/interactive_track.py`` then loads
    automatically (auto-generated spheres are used until that file exists).

This is the hand-tune half of the "auto first, tune if needed" pipeline
(:func:`fabrix.collision.auto_arm_spheres` -> tune here -> :meth:`SphereModel.from_dict`). It is
**purely kinematic** (no fabric / no jit): sizing spheres is a *coverage* question, so you shape them
here and check avoidance *behavior* in ``interactive_track.py`` with the exported model.

viser is an optional dependency (the ``viz`` group), and it serves a browser UI — so this works
headless / over SSH, with no local OpenGL and no ``mjpython``:

    uv run --group viz python demos/tune_spheres.py          # then open the printed URL
    uv run --group viz python demos/tune_spheres.py --check   # headless build smoke-test
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # repo root on sys.path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from fabrix import CustomFK, SphereModel
from fabrix.collision import _child_offset, _collision_radius, load_spheres

GEN3 = "mujoco_menagerie/kinova_gen3/gen3.xml"
TUNED = pathlib.Path(__file__).with_name("spheres_tuned.py")   # the export target / resume source

C_MESH = (170, 172, 180)        # link meshes: neutral gray
C_BLUE = (60, 140, 255)         # spheres: translucent blue
C_SEL = (255, 205, 40)          # selected sphere: yellow
OP_SPH, OP_SEL = 0.30, 0.6


def _visual_meshes(m):
    """``(body id, geom id, verts, faces, geom_pos, geom_quat)`` for each *visual* mesh geom.

    Visual = a mesh geom with both collision flags off. Vertices are in the geom frame, so they place
    correctly as a child of the body frame at the geom's local ``(pos, quat)`` (verified vs MuJoCo).
    """
    MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
    out = []
    for g in range(m.ngeom):
        if int(m.geom_type[g]) != MESH or m.geom_contype[g] or m.geom_conaffinity[g]:
            continue
        did = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        fa, fn = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
        out.append((int(m.geom_bodyid[g]), g,
                    np.asarray(m.mesh_vert[a:a + n], np.float32),
                    np.asarray(m.mesh_face[fa:fa + fn], np.int32),
                    np.asarray(m.geom_pos[g], np.float32),
                    np.asarray(m.geom_quat[g], np.float32)))
    return out


def _home(m):
    return np.asarray(m.key_qpos[0, :m.nq], float) if m.nkey else np.zeros(m.nq)


def _format_file(spec):
    """Render the ``{body: [((x,y,z), r), ...]}`` spec as a committable, from_dict-loadable module."""
    L = ['"""Hand-tuned collision spheres exported by demos/tune_spheres.py.',
         '',
         'Loaded automatically by demos that call fabrix.collision.load_spheres (e.g.',
         'demos/interactive_track.py) when this file exists. Re-export to update; the dict is plain',
         'data — edit it by hand to add or drop spheres (it round-trips through from_dict).',
         '"""',
         'from fabrix.collision import SphereModel',
         '',
         'SPHERES = {']
    for name, items in spec.items():
        L.append(f'    {name!r}: [')
        for c, r in items:
            L.append(f'        (({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}), {r:.6f}),')
        L.append('    ],')
    L += ['}', '', '', 'def load_tuned(provider):', '    return SphereModel.from_dict(provider, SPHERES)', '']
    return "\n".join(L)


class Tuner:
    """Holds the editable sphere arrays + viser scene/GUI, and keeps them in sync on every edit."""

    def __init__(self, server):
        self.prov = CustomFK(GEN3)
        self.m = self.prov.mj_model
        self.server = server
        self._wf = jax.jit(self.prov.body_poses)              # q -> (P, Q) world body frames, cached

        sph, self.tuned = load_spheres(self.prov, TUNED, n_per_link=2)   # resume a tuned file if present
        self.link = np.asarray(sph.link, int)
        self.local = np.asarray(sph.local, float)
        self.radius = np.asarray(sph.radius, float)
        self.base_local = self.local.copy()                  # "reset sphere" baseline (grows with adds)
        self.base_radius = self.radius.copy()
        self.names = sph.names(self.prov)
        self.link_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)   # add-sphere targets
                           for b in range(1, self.m.nbody)]
        self.q = _home(self.m)
        self.sel = 0
        self._sync = False                                    # guards programmatic widget writes
        self.sph_h, self.giz = {}, None

        self._build_scene()
        self._build_gui()
        self._set_pose(self.q)
        for i in range(len(self.link)):
            self._draw_sphere(i)
        self._move_gizmo(self.sel)

    # ---------------------------------------------------------------- scene
    def _build_scene(self):
        s = self.server.scene
        s.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1)
        self.frames = {b: s.add_frame(f"/body{b}", show_axes=False) for b in range(1, self.m.nbody)}
        for bid, g, verts, faces, gp, gq in _visual_meshes(self.m):
            s.add_mesh_simple(f"/body{bid}/mesh{g}", verts, faces, color=C_MESH,
                              position=gp, wxyz=gq, opacity=0.9)

    def _set_pose(self, q):
        P, Q = self._wf(jnp.asarray(q, jnp.float32))          # one FK to every body frame
        P, Q = np.asarray(P), np.asarray(Q)
        for b, fr in self.frames.items():                     # meshes + spheres + gizmo ride the frames
            fr.position, fr.wxyz = P[b], Q[b]

    def _draw_sphere(self, i):
        """(Re)create sphere ``i`` under its link frame, colored by selection. Overwrites by name."""
        bid = int(self.link[i])
        on = i == self.sel
        h = self.server.scene.add_icosphere(
            f"/body{bid}/sph{i}", radius=float(self.radius[i]), color=C_SEL if on else C_BLUE,
            opacity=OP_SEL if on else OP_SPH, position=self.local[i])
        self.sph_h[i] = h
        if hasattr(h, "on_click"):
            @h.on_click
            def _(_evt, i=i):
                self._select(i)

    def _move_gizmo(self, i):
        """Put the translation gizmo on sphere ``i`` (a child of its link frame, so it edits in that
        frame — its ``position`` is exactly the sphere's link-local center).

        ``depth_test=False`` draws the gizmo *on top of* the link mesh, so the z (blue) arrow stays
        grabbable even when it points into the link; the x/y/z number inputs are an exact alternative.
        """
        if self.giz is not None:
            self.giz.remove()
        bid = int(self.link[i])
        self.giz = self.server.scene.add_transform_controls(
            f"/body{bid}/giz", scale=0.12, line_width=3.5, depth_test=False,
            disable_rotations=True, position=self.local[i])

        @self.giz.on_update
        def _(_):
            if self._sync:
                return
            self.local[i] = np.asarray(self.giz.position, float)
            self._apply_local()                               # push to sphere + x/y/z inputs

    # ---------------------------------------------------------------- gui
    def _build_gui(self):
        g = self.server.gui
        g.add_markdown("**Collision-sphere tuner** — select a sphere, resize / move it, scrub the pose, export.")

        with g.add_folder("Pose"):
            self._jsliders = []
            for j in range(self.m.njnt):
                if int(self.m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                    continue
                qi = int(self.m.jnt_qposadr[j])
                lo, hi = ((float(self.m.jnt_range[j][0]), float(self.m.jnt_range[j][1]))
                          if self.m.jnt_limited[j] else (-np.pi, np.pi))
                lo, hi = np.floor(lo * 100) / 100, np.ceil(hi * 100) / 100  # round outward (continuous q2≈π)
                sl = g.add_slider(f"q{qi}", min=lo, max=hi, step=0.01,
                                  initial_value=float(np.clip(self.q[qi], lo, hi)))
                self._jsliders.append((qi, sl))

                @sl.on_update
                def _(_, qi=qi, sl=sl):
                    if self._sync:
                        return
                    self.q[qi] = sl.value
                    self._set_pose(self.q)
            rp = g.add_button("reset pose")

            @rp.on_click
            def _(_):
                self.q = _home(self.m)
                self._sync = True
                for qi, sl in self._jsliders:
                    sl.value = float(self.q[qi])
                self._sync = False
                self._set_pose(self.q)

        with g.add_folder("Selected sphere"):
            self.dd = g.add_dropdown("sphere", options=self._opts(), initial_value=self._opts()[0])

            @self.dd.on_update
            def _(_):
                if not self._sync:
                    self._select(int(self.dd.value.split(":")[0]))
            self.rad = g.add_slider("radius (mm)", min=5.0, max=120.0, step=0.5,
                                    initial_value=float(self.radius[self.sel] * 1e3))

            @self.rad.on_update
            def _(_):
                if self._sync:
                    return
                self.radius[self.sel] = self.rad.value * 1e-3
                self._draw_sphere(self.sel)
            for label, f in (("shrink 5%", 0.95), ("grow 5%", 1.05)):   # relative resize, selected only
                b = g.add_button(label)

                @b.on_click
                def _(_, f=f):
                    self.radius[self.sel] *= f
                    self._draw_sphere(self.sel)
                    self._sync = True
                    self.rad.value = float(self.radius[self.sel] * 1e3)
                    self._sync = False
            self.pos = []                                     # exact link-frame position (occlusion-proof)
            for k, ax in enumerate("xyz"):
                n = g.add_number(f"{ax} (mm)", initial_value=float(self.local[self.sel][k] * 1e3),
                                 min=-500.0, max=500.0, step=1.0)
                self.pos.append(n)

                @n.on_update
                def _(_, k=k, n=n):
                    if self._sync:
                        return
                    self.local[self.sel][k] = n.value * 1e-3
                    self._apply_local()
            # Nudge: viser exposes no keyboard events, so these buttons are the reliable "keyboard move"
            # (and the x/y/z fields above step with the ↑/↓ keys when focused). Step is in mm.
            self.step = g.add_number("nudge step (mm)", initial_value=5.0, min=0.5, max=50.0, step=0.5)
            for k, ax in enumerate("xyz"):
                for btn, s in ((g.add_button(f"{ax} −"), -1.0), (g.add_button(f"{ax} +"), +1.0)):
                    @btn.on_click
                    def _(_, k=k, s=s):
                        self.local[self.sel][k] += s * self.step.value * 1e-3
                        self._apply_local()
            rs = g.add_button("reset sphere")

            @rs.on_click
            def _(_):
                self.radius[self.sel] = float(self.base_radius[self.sel])
                self.local[self.sel] = self.base_local[self.sel].copy()
                self._refresh_selected()

        with g.add_folder("Add / remove"):
            self.link_dd = g.add_dropdown("link", options=self.link_names, initial_value=self.names[self.sel])
            ab = g.add_button("add sphere on link")

            @ab.on_click
            def _(_):
                self._add(self.link_dd.value)                 # default size/position from the link's bone
            db = g.add_button("duplicate selected")

            @db.on_click
            def _(_):
                self._add(self.names[self.sel], self.local[self.sel].copy(), float(self.radius[self.sel]))
            xb = g.add_button("delete selected")

            @xb.on_click
            def _(_):
                self._delete()

        with g.add_folder("Export"):
            self.status = g.add_markdown(
                f"loaded **{'tuned' if self.tuned else 'auto'}** model — {len(self.link)} spheres")
            eb = g.add_button("export -> demos/spheres_tuned.py")

            @eb.on_click
            def _(_):
                self._export()

    # ---------------------------------------------------------------- helpers
    def _opts(self):
        return [f"{i}: {n}" for i, n in enumerate(self.names)]

    def _select(self, i):
        old, self.sel = self.sel, i
        self._draw_sphere(old)                                # recolor the previously-selected one
        self._refresh_selected()

    def _refresh_selected(self):
        i = self.sel
        self._draw_sphere(i)
        self._move_gizmo(i)
        self._sync = True                                     # reflect state in widgets w/o re-firing
        self.rad.value = float(self.radius[i] * 1e3)
        self.dd.value = self._opts()[i]
        for k, n in enumerate(self.pos):
            n.value = float(self.local[i][k] * 1e3)
        self._sync = False

    def _apply_local(self):
        """Push the selected sphere's link-local center to the sphere, the gizmo, and the x/y/z inputs."""
        i = self.sel
        self.sph_h[i].position = self.local[i]
        self._sync = True
        self.giz.position = self.local[i]
        for k, n in enumerate(self.pos):
            n.value = float(self.local[i][k] * 1e3)
        self._sync = False

    def _full_redraw(self):
        """Rebuild every sphere handle from the current arrays (after add / delete renumbers indices)."""
        for h in self.sph_h.values():
            h.remove()
        self.sph_h.clear()
        for i in range(len(self.link)):
            self._draw_sphere(i)

    def _sync_dropdown(self):
        self._sync = True
        self.dd.options = self._opts()
        self.dd.value = self._opts()[self.sel]
        self._sync = False

    def _add(self, link_name, local=None, radius=None):
        """Append a sphere on link ``link_name``; default position = bone midpoint, radius = mesh width."""
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if local is None:
            bone = _child_offset(self.m, bid)
            local = 0.5 * bone if bone is not None else np.zeros(3)
        if radius is None:
            radius = _collision_radius(self.m, bid)
        self.link = np.append(self.link, int(bid))
        self.local = np.vstack([self.local, np.asarray(local, float)])
        self.radius = np.append(self.radius, float(radius))
        self.base_local = np.vstack([self.base_local, np.asarray(local, float)])
        self.base_radius = np.append(self.base_radius, float(radius))
        self.names.append(link_name)
        self.sel = len(self.link) - 1
        self._full_redraw()
        self._sync_dropdown()
        self._refresh_selected()

    def _delete(self):
        if len(self.link) <= 1:                               # keep at least one sphere
            return
        keep = [j for j in range(len(self.link)) if j != self.sel]
        self.link, self.local, self.radius = self.link[keep], self.local[keep], self.radius[keep]
        self.base_local, self.base_radius = self.base_local[keep], self.base_radius[keep]
        self.names = [self.names[j] for j in keep]
        self.sel = min(self.sel, len(self.link) - 1)
        self._full_redraw()
        self._sync_dropdown()
        self._refresh_selected()

    def _export(self):
        spec = SphereModel(self.link.copy(), self.local.copy(), self.radius.copy()).to_dict(self.prov)
        TUNED.write_text(_format_file(spec))
        msg = f"exported {len(self.link)} spheres -> {TUNED}"
        self.status.content = msg
        print(msg)


def main(check=False):
    import viser
    server = viser.ViserServer()
    tuner = Tuner(server)
    if check:
        n0 = len(tuner.link)                                  # exercise the mutation logic headlessly
        tuner._add("forearm_link")
        assert len(tuner.link) == n0 + 1 and tuner.names[-1] == "forearm_link" and tuner.sel == n0
        tuner._add(tuner.names[tuner.sel], tuner.local[tuner.sel].copy(), float(tuner.radius[tuner.sel]))
        assert len(tuner.link) == n0 + 2                      # duplicate
        tuner.local[tuner.sel][2] += 0.01; tuner._apply_local()    # nudge / position edit
        tuner._delete(); tuner._delete()
        assert len(tuner.link) == n0 and len(tuner.names) == n0
        assert len(tuner.dd.options) == n0 and tuner.local.shape == (n0, 3)
        print(f"[check] tuner OK — {n0} spheres, {len(_visual_meshes(tuner.m))} link meshes, "
              f"{'tuned' if tuner.tuned else 'auto'} model; add/duplicate/nudge/delete exercised")
        server.stop()
        return
    print("\n>>> open the URL above: select a sphere, drag radius / gizmo, scrub the pose, then Export <<<")
    print(">>> the exported model loads automatically in demos/interactive_track.py <<<\n")
    server.sleep_forever()


if __name__ == "__main__":
    main(check="--check" in sys.argv)
