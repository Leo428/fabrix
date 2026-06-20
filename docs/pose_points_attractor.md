# B2 — Control-points pose attractor (the NVlabs way)

**Status:** BUILT + sim-validated 2026-06-19 — fabrix lib (`pose_points_attractor`, `control_points_error_map`,
`control_point_jack`, 7 tests, full suite 60✓) + control_node wiring (`FabricController(pose_mode=…)`,
`gains.ctrl_radius`, `robot_node --pose-points`) + `sim_twin --check-pose-points` gate (ori lag 21.5°→0.52°,
rotation +37%, no ring). **Opt-in; default is still the SE(3) twist until the HW A/B.** Decided over B1
(splitting the twist gains). Remaining: HW A/B vs twist+`--pose-fmax 18`, then flip the default if it wins.

## Why

Teleop wrist rotation is fabric-limited (HW-diagnosed 6/19 — see memory `kinova-teleop-responsiveness`). Root cause: our `pose_attractor` (`fabrix/leaves.py`, `maps.py:se3_pose_error_map`) uses a **coupled 6-D SE(3) twist** with **one shared `k`/`b`/`f_max`/metric** over mixed (m + rad) units. Max reference twist rate ≈ `f_max/b`, shared across translation and rotation → rotation throttled to ~14°/s (commanded 15–60). The A-probe (`--pose-fmax` 10→18) confirmed the lever but the shared cap over-boosts translation (peak KE 2.1×) for a modest rotation gain — can't push further safely. The shared mixed-units cap is the root cause and must go.

## How NVlabs/FABRICS avoids it (the reference)

Local clone: `/private/tmp/nvlabs_fabrics/src/fabrics_sim`. Their EE pose fabric (`fabrics/kuka_allegro_pose_fabric.py:189 add_palm_points_attractor`) uses **NO SE(3) twist**. It places **7 control points** rigidly on the palm — origin + ±x/±y/±z:
```python
control_point_frames = ["palm_link","palm_x","palm_x_neg","palm_y","palm_y_neg","palm_z","palm_z_neg"]
taskmap = RobotFrameOriginsTaskMap(...)            # q -> stacked 3-D positions (21-D)
fabric  = Attractor(is_forcing, params['palm_attractor'], ...)   # purely POSITIONAL
```
Matching ≥3 non-collinear points pins the full 6-DOF pose, but **everything is in meters** → one uniform metric, one gain, one saturation (`conical_gain`/soft-relu = their `f_max`; `max_metric`+`metric_exploder_offset` = our D1 dynamic mass). Orientation is implicit: rotating the EE moves the offset points by `r·θ`; the positional attractor restores them, torque `∝ r²`. **The translation/rotation balance is geometric — the point offset radius `r`, not a separate gain.** No mixed units, no shared-cap problem by construction.

## Design (reuses almost all existing machinery)

Control points are built directly from the EXISTING `provider.site_pose(q) -> (pos, quat)` by applying fixed local offsets — no new FK needed (cheaper than collision's per-link FK: one site FK + P offset rotations).

1. **`fabrix/maps.py` — new `control_points_error_map(provider, target_pos, target_quat, offsets)`** (mirrors `se3_pose_error_map`):
   - `offsets`: `(P,3)` fixed local points in the EE/site frame (e.g. origin + ±x/±y/±z jack of radius `r`).
   - target points (world, fixed): `t_pts = target_pos + R(target_quat) @ offsetsᵀ`.
   - `phi(q)`: `p,quat = site_pose(q)`; `cur_pts = p + R(quat) @ offsetsᵀ`; return `(cur_pts - t_pts).reshape(-1)` → `(3P,)`, **all meters**. (Use jaxlie SO3 wxyz like `_se3`, or the existing `_qrot`.)
2. **`fabrix/leaves.py` — new `pose_points_attractor(provider, offsets, k, b, m, f_max, m_max=None, sharp, offset)`** (mirrors `pose_attractor`):
   - `phi = control_points_error_map(...)`; `e,J,Jdq = value_jac_curv(phi,q,qd)` (`e:(3P,)`, autodiff J/curvature);
   - `m_ = _scaled_mass(e, m, m_max, sharp, offset)` — **`‖e‖` now pure meters** (cleaner D1, no mixed units);
   - `M = m_·I_{3P}`; `f = m_·(_restoring(e,k,f_max) + b·ed)`; `return pullback(Spec(M,f), J, Jdq)`.
   - Reuses `_restoring`, `_scaled_mass`, `value_jac_curv`, `pullback` unchanged (all dimension-agnostic).
3. **`fabrix/__init__.py`** export `pose_points_attractor` (+ map). Keep `pose_attractor` (the twist) in the library for other users / fallback.

## Empirical sweep (sim, validated 6/19) — how to choose `r`

Jack-radius sweep (frictionless rollout, `f_max=10`, `posture`+`config_damping`; pure-orientation target
= +1.0 rad in place, plus a separate 12 cm translation target):

| `r` (m) | peak rotation (°/s) | final orient err (°) | peak translation (mm/s) |
|---|---|---|---|
| 0.03 | 3.8 | 41.5 ✗ | 166 |
| 0.05 | 8.5 | 20.9 ✗ | 166 |
| 0.08 | 16.2 | 5.4 | 164 |
| 0.10 | 21.2 | 3.2 | 163 |
| 0.15 | 32.2 | 1.5 | 161 |
| 0.20 | 40.2 | 0.8 | 160 |
| 0.30 | 48.4 | 0.4 | 159 |
| 0.50 | 47.6 | 0.1 | 158 |

**Findings (these CORRECT the earlier hand-analysis, which guessed `ω ∝ 1/r`):**
- **Translation speed is `r`-independent** (~160 mm/s, flat) — `f_max` alone sets it. ✓ as predicted.
- **Rotation speed *increases* with `r`** up to a broad plateau ~0.3–0.5. The dominant effect is the
  **metric competition**: the rotation DOF's task priority is `∝ m·r²`, so at small `r` the attractor
  loses the rotation direction to `posture`/`config_damping` (which carry `r`-independent metric) and
  rotation is suppressed. The idealized "saturated cruise `∝ 1/r`" is real but only bites at the very top
  (the 0.50 dip), where it finally overtakes the `r²` priority growth.
- **Small `r` (<0.08) fails to even converge orientation** (41°, 21° residual) — too little rotational
  signal (`∝ r`) and priority (`∝ r²`). That is the practical floor.
- Net: **larger `r` is simply better for rotation** (speed *and* convergence) until it plateaus; there is
  no strong tradeoff in the useful range because translation is decoupled. `r` sets the rotation⁄
  translation speed RATIO; `f_max` sets the absolute level.

**The r→rotation direction is REGIME-DEPENDENT** — an inverted-U (priority `∝ r²` lifts small r, the
saturated cruise `∝ 1/r` drags large r), whose peak moves with the damping:
- **Light damping** (this fabrix rollout: attractor `b=8`, no `ref_damp`): peak at LARGE r (~0.3–0.5) —
  the table above.
- **Heavy PRODUCTION damping** (`sim_twin`, `pose_b=40`, `ref_damp=14` — the deployed teleop cascade):
  peak at SMALL **r ≈ 0.12**. r-down sweep (~60°/s wrist command): sustained rotation peaks at r=0.12
  (17.7°/s) and the settled orientation lag bottoms there (1.0° vs the twist's 26°); peak rotation keeps
  rising as r↓ (33°/s at r=0.08, but with a softer hold).

**Deployment default `r = 0.12`** (`gains.ctrl_radius`), taken from the `sim_twin` regime — NOT the fabrix
table, which is the wrong (light-damping) regime for the deployed controller. `sim_twin --check-pose-points`
gates it on the teleop law: ori lag 21.5°→0.52°, sustained rotation 12.9→17.7°/s, push-ring ζ 0.165→0.361
(points is *better* damped), no divergence. The fabrix table still governs other (light-damping) users and
shows the mechanism; HW A/B refines the deployed value.

## Wiring (control_node)

- `fabric_controller.py:76`: swap `pose_attractor(...)` → `pose_points_attractor(self.prov, offsets=control_point_jack(g0("ctrl_radius")), k=g("pose_k"), b=g("pose_b"), m=g("pose_m"), f_max=g("pose_fmax"), m_max=g("pose_m_max"), ...)`. (`ctrl_radius` is a *static* geometry knob — read it once at build time via the plain gains object, not the traced `g()` closure; changing the jack changes array shapes ⇒ a recompile, unlike the scalar gains.)
- `gains.py`: add `ctrl_radius` (jack radius, **~0.18 m** per the sweep). **Gains are now POSITIONAL (m)** — `pose_k`/`pose_fmax`/`pose_b`/`pose_m*` must be RE-TUNED from the current mixed-units values (180/40/10/300 are not meaningful in pure meters). Start from the sweep above + the demo sliders.
- The rotation/translation balance = `ctrl_radius`: larger r → more rotation authority (`∝ r²` priority). Tune r + `f_max` so rotation reaches ~commanded 15–60°/s while translation stays bounded.

## Verification (sim-first, then HW)

- **fabrix tests** (`tests/test_m3.py`): (a) pulling P non-collinear points to a target drives full SE(3) pose error → 0 (parity vs `se3_pose_error_map` at convergence); (b) rotation authority MONOTONE in `ctrl_radius` (peak ref angular speed ↑ with r at fixed f_max); (c) `_scaled_mass` on pure-meters `‖e‖` still kills the posture leak (D1 behavior preserved); (d) latency guard (3P path bounded — expect cheap, one site FK).
- **demo** (`demos/interactive_track.py`): sliders for `ctrl_radius` + `f_max`; readouts for EE linear AND angular speed → watch rotation speed up with r/f_max independently of translation.
- **sim_twin** (`control_node/sim_twin.py`): extend `check_fabric_upgrades` with a ref-angular-speed metric; assert rotation reaches a target rate at given f_max/r WITHOUT translation over-speed. Keep `--check-torque-ctrl` green.
- **HW A/B** (user drives, e-stop): vs the current f_max-18 twist. Measure with `/tmp/fk_orient.py` (ref/act angular speed → should reach command) + `/tmp/analyze_ke.py` (translation KE bounded) + `/tmp/fk_split.py` (impedance EE err). Diagnostic scripts live on the Mac `/tmp`.

## Interim + state

- Current HW best = the SE(3) twist with `--pose-fmax 18` (rotation +43%, translation lag 63→44 mm, but peak KE 2.1×). Keep as interim working config until B2 lands (or dial `--pose-fmax 14` for gentler). Baked teleop defaults already = `fric-comp 1.0 + kd-scale 1.0` (low-level wins).
- Once B2 sets a comfortable speed, **enable the KE cap** (`--speed-beta`/`--speed-e-max`, built + unused) to bound peak KE near the seated person.

## Risks / open

- Re-tune effort: positional gains from scratch (sim sweep first).
- Min points: ≥3 non-collinear for full-pose observability; 7-point jack is robust, 4 (origin+x+y+z) is cheaper — pick in tests.
- `ctrl_radius` units now make `_scaled_mass` `offset` (switch radius) pure meters — re-pick it too.
- jaxlie/`_qrot` dtype: anchor to `q.dtype` (float32) as `se3_pose_error_map` does.
