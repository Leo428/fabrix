# fabrix — Roadmap, Status & TODOs

Living design+status doc. Project scope is **M1 → M3** — **all complete** (2026-06), plus a post-M3
**whole-arm + self-collision** capability (collision spheres, see below — also complete). (M4-style
batched-RL / learnable fabrics and hardware StableHLO/AOT export are explicitly out of scope.)

---

## Why this exists

On real arms, **differential IK (mink)** produces a position stream that is geometrically
correct but **not smooth in velocity/acceleration**. A stiff servo *differentiates the
reference* for velocity feedforward, so that non-smoothness becomes motor vibration / audible
noise. Simulation hides it (soft position actuator, no reference differentiation). Second-order
reactive policies (RMPflow, **geometric fabrics**) carry `(q, q̇)` state with damping and emit a
**C2 reference** — smooth and quiet on hardware. We're building a JAX-native fabrics library,
general beyond the immediate Kinova Gen3 + Robotiq 2F-85 use case.

## Why JAX (de-risked, see `experiments/`)

Every fabric operation is a derivative-of-a-derivative; autodiff supplies them:
- `J = jacfwd(phi)`; curvature `J̇q̇ = jvp(λq: jvp(phi,(q,),(q̇,))[1], (q,),(q̇,))[1]`.
- Validated: `J` vs `mj_jacSite` = **6e-16**; `J̇q̇` vs finite-diff = **2e-10**.
- Speed: vectorized custom FK `fk` ~8 µs, `J̇q̇` ~27 µs; JAX CPU **dispatch floor ~1.6 µs**
  → single-arm real-time from Python on CPU is feasible.
- **MJX** was evaluated — correct, but ~100 ms single-instance on CPU → a *batched-GPU /
  general-model* tool, not a real-time provider. Since M1–M3 run on the Gen3 via `CustomFK`, the
  `mujoco-mjx` dependency was dropped; the `KinematicsProvider` Protocol keeps re-adding it cheap.
- **Burned-in lesson:** never build vectors with `jnp.array([scalar, …])` — it compiles to
  thousands of scalar ops (~**1000× slowdown**). Use slicing + vectorized ops + `concatenate`.

## JAX discipline (enforced)

- No `jnp.array([scalar, …])`. - One `jit` boundary per dispatch (`Fabric.policy`, `rollout`).
- Static structure (provider, leaf set, dims, dtypes) baked at build; only `q/q̇/params` traced
  → no recompiles when targets/gains change. - Pytrees everywhere (`@jdc.pytree_dataclass`),
  pure-functional. - `lax.scan` for rollouts, `vmap` for any batching — no Python loops over
  data (static body-tree unroll in FK is fine). - Solve, don't invert (`cho_factor`/`cho_solve`
  on `M+λI`). - float32 default, x64 only in tests. - Latency-guard test catches regressions.

## Architecture

`Spec(M, f)` = second-order spec `M q̈ + f = 0`. Leaves emit Specs; `pullback` maps task-space
Specs to config space; `combine` sums them; `resolve` solves for `q̈`.

- **Spec algebra** (`fabrix/spec.py`): `pullback(s, J, Jdq) = Spec(JᵀMJ, Jᵀ(f + M·Jdq))` — the
  `M·J̇q̇` curvature term is what makes it correct. `combine` = (ΣM, Σf). `resolve` = Cholesky.
- **Derivatives** (`fabrix/diff.py`): `value_jac_curv(phi, q, qd) -> (x, J, Jdq)`.
- **Kinematics** (`fabrix/kinematics.py`): `KinematicsProvider` Protocol + `CustomFK` (vectorized
  serial-chain FK, the real-time provider). The Protocol leaves room for an alternative backend
  (e.g. an MJX wrapper for non-serial / batched use) without touching the fabric.
- **Leaves** (`fabrix/leaves.py`): the pattern is **`f = -M @ a_des`** — the leaf's isolated
  acceleration is exactly `a_des`, and the **metric `M` sets task priority** in the combination.
  This is what gives precise EE convergence without posture bias. `attractor` (task-space, large
  metric), `posture` + `config_damping` (config-space, identity map → returned directly).
- **Fabric** (`fabrix/fabric.py`): `Fabric(leaves).policy(q, qd, params) -> q̈`, one jit.
  `FabricParams{target, q_default}` are traced.
- **Integrate** (`fabrix/integrate.py`): semi-implicit Euler `step`; `rollout` via `lax.scan`.

---

## Status

### ✅ M1 — forced attractor fabric (DONE)
Smallest end-to-end slice: attractor + posture + config-damping forced fabric drives the Gen3 EE
to a target; integrate to a smooth joint reference.
- **Results:** EE reach **1.17 mm**; `policy` **~53 µs/step** (float32 CPU, ~5% of 1 kHz);
  `q̈` continuous & bounded (C2). `uv run pytest -q` → **9 passing**.
- **Files:** all of `fabrix/`, `demos/attractor_reach.py`, `tests/test_fabrix.py`.
- **Known transient:** `q̈` has a one-time step at `t=0` (step response from rest), not per-step
  chatter. A soft-start or M2 speed-control rounds it off.

### ✅ M2 — energization + obstacle/joint-limit geometries (DONE)
Turns the forced base case into a *full* geometric fabric.
- **`fabrix/energy.py`** — `energy_spec(L_e, x, xd)` builds the energy spec `(M_e, f_e)` from a
  Lagrangian by autodiff (`M_e = ∂²_{ẋ}L_e`, `f_e = ∂_x(∂_{ẋ}L_e)·ẋ − ∂_x L_e`); `fixed_metric_energy`
  (`L_e = ½‖ẋ‖²`) and `lagrangian_energy(G_fn)`. Validated: `M_e == G(x)` exact; the rate identity
  `dH_e/dt = ẋᵀ(M_e ẍ + f_e)` to **4e-16**.
- **`fabrix/geometry.py`** — `energize(a_g, v, M_e, f_e)` = `a_g − [vᵀ(M_e a_g + f_e)/(vᵀM_e v)] v`:
  zeroes the energy rate (instantaneously **~1e-9**) with the correction **purely along v** (path
  preserved, off-axis part ~1e-17). HD2 barrier geometries `joint_limit_geometry`,
  `obstacle_geometry` (SDF map) + barrier **potentials** `joint_limit_potential`, `obstacle_potential`.
- **Assembly** `GeometricFabric` (`fabrix/fabric.py`): geometries → combine → resolve = root
  geometry accel; energize; + forcing (attractor) + damping → resolve.
- **Three things learned (load-bearing):**
  1. **Energize at the *root*.** A barrier's 1-D leaf space makes energization degenerate (any 1-D
     accel changes speed → the projection kills it); combine geometries to config space first.
  2. **The invariant comes from the *potential*, not the geometry.** An energized geometry conserves
     speed, so it deflects but cannot *stop* a head-on approach; a barrier potential (diverging at the
     boundary) is what makes non-penetration / limit-respect a hard invariant.
  3. **`geom_reg` must clear float32 eps.** The root-geometry solve regularizer at `1e-6` ≈ float32
     eps amplified noise into q̈ chatter (max Δq̈ 0.36 → 10+); `geom_reg=1e-4` fixes it, behavior
     unchanged. Strict HD2 sign-switch kept; potential metrics use a C1 standoff band (`_band`).
- **Results:** geometry-only energy drift **7.5e-4**; joint-limit & obstacle invariants **never
  violated**; reach-with-obstacle **10 mm** + **no penetration** (77 mm standoff) + C2 (max Δq̈ **0.31**,
  float32); full policy **~66 µs/step**. `uv run pytest -q` → **19 passing** (8 M1 + 11 M2).
- **Files:** `fabrix/{energy,geometry}.py`, `GeometricFabric`, SDF maps; `demos/obstacle_reach.py`
  (→ `obstacle_reach.png`); `tests/test_m2.py`.

### ✅ M3 — orientation / full SE(3) pose (PROJECT ENDPOINT — DONE)
Adds orientation, completing the M1→M3 scope: the fabric now tracks a full 6-DOF pose.
- **`CustomFK.site_pose(q) -> (pos, quat)`** — the body loop already carries every body's world
  quat, so orientation is one extra `_qmul(qw[site_body], site_quat)`; the position hot path
  (`site_pos`) shares the same loop unchanged. Including the site's *local* frame mattered: the Gen3
  `pinch_site` has `site_quat = [0,1,0,0]` (180° about x), which the old position-only FK dropped.
  Verified vs MuJoCo `site_xpos`/`site_xmat` to **8.9e-16**.
- **`se3_pose_error_map`** (`maps.py`) — `φ(q) = Log(T*⁻¹ T(q)) ∈ se(3)` (6,) via **jaxlie**
  (`SO3.from_quaternion_xyzw` — note jaxlie is xyzw, MuJoCo wxyz — `SE3.from_rotation_and_translation`,
  `.inverse()`, `@`, `.log()`). The **flagged risk cleared**: `jacfwd` *and* the nested-jvp curvature
  flow through `SE3.log` (Taylor fallback keeps it finite at the identity); `J` vs finite-diff **3e-10**,
  `J̇q̇` **5e-11** — same precision as the position maps. Output is cast to the config dtype (jaxlie
  carries float64 constants that would otherwise promote a float32 config under `jax_enable_x64`).
- **`pose_attractor`** leaf (`leaves.py`) — coupled SE(3): one 6-DOF error, priority metric `m·I₆`,
  same `f = -M@a_des` pattern as the position attractor but in se(3); geodesic ("screw") approach.
  `FabricParams` gained `target_quat` (wxyz, identity default → M1/M2 constructions unchanged).
- **Design choice:** coupled **SE(3) Log** (geodesic, single 6D metric) over decoupled position⊕SO(3)
  — user's call; matches the roadmap's original wording.
- **Results:** pose reach **0.82 mm** + **0.019°**, C2-smooth; float32 pose policy **~88 µs/step**
  (+22 µs over M2 for the SE(3) log, still ~9% of 1 kHz). `uv run pytest -q` → **25 passing**
  (8 M1 + 11 M2 + 6 M3).
- **Demo:** `interactive_track.py` now reads `mocap_quat` → full 6-DOF target (Ctrl+left-drag rotates
  the target, and the arm tracks it — the rotation the demo previously ignored). `--check` 97 mm
  clearance, stable.
- **Files:** `kinematics.py` (`site_pose`/`site_rot`), `maps.py` (`se3_pose_error_map`, `_se3`),
  `leaves.py` (`pose_attractor`), `fabric.py` (`FabricParams.target_quat`); `tests/test_m3.py`.

### Backlog / follow-ups (noted from the interactive demo, 2026-06)
Driving `demos/interactive_track.py` surfaced these. With M1→M3 complete these are post-scope polish:
- **Rotation tracking** — ✅ **done in M3**: the demo reads `mocap_quat` and the `pose_attractor`
  tracks the full 6-DOF target.
- **Responsiveness** — ✅ **tuned**: bumped the demo to "setting C" (`pose_attractor` k=36/b=12,
  `config_damping` b=2) — ~2× snappier reach, still critically damped. Measured time-to-1 mm 2.66 s → 1.46 s.
- **Obstacle avoided from too far** — ✅ **fixed + generalized**: the early detour was the *geometry*
  (its `m_b/d` metric reaches at all approaching distances), not the potential. Added an optional
  standoff band `d0` to `sdf_barrier_geometry` (fades the priority metric out beyond `d0`, HD2
  acceleration untouched). Demo: geometry `d0=0.12` (detour onset ~100 mm, was ~150–200) + potential
  `d0=0.02` (tight 2 cm hard wall). Min clearance still +20 mm in the sweep-through-center test.
- **Ground/floor avoidance** — ✅ **done**: generalized the obstacle leaves into `sdf_barrier_*`
  taking any distance field; added `plane_geometry`/`plane_potential` and wired the scene's ground
  plane into the demo (EE held ~80 mm above the floor when the target is driven below it).
- **Draggable obstacle** — ✅ **done**: `obstacle_*` leaves accept `center=None` → read
  `params.obstacle_center` (new `FabricParams` field); the demo's red ball is a second mocap body.
- **Saturating attractor potential** — ✅ **done**: opt-in `f_max` on `attractor`/`pose_attractor`
  (`_restoring`): magnitude `f_max·tanh(k‖e‖/f_max)·ê` — same stiffness near the goal, accel capped
  far away so large/commanded moves don't lunge (on a 64 cm move, peak ‖q̈‖ 30.7 → 3.3 at f_max=8).
  Demo uses `f_max=10`. Caveat: for coupled SE(3) one `f_max` mixes the translation (m) and rotation
  (rad) scales of the twist. Doesn't speed the near-goal tail (that's stiffness).
- **Wheelchair: keep the arm upright** — ✅ **#1 (soft posture bias) done**: `posture` now takes
  per-joint `weight`/`k` (bias the shoulder/elbow toward `q_default`, leave the wrist free). **Key
  finding:** a full 6-DOF pose task leaves only **1 nullspace DOF** on the 7-DOF arm, so scalar
  posture barely shifts overall pose (weight 0.5→5 moved it 45.7°→44.6° but cost EE 0.5→4.7 mm);
  posture's real reach is the elbow swivel + position-dominant tasks. `q_default` (= the home
  keyframe today) **is** the upright nominal — set it from real wheelchair geometry later. Deferred
  to when the arm is mounted + dimensions known:
  - **#2 Hard joint no-go limits** — let `joint_limit_{geometry,potential}` take custom narrower
    per-joint ranges (override the model's `jnt_range`) → a *guaranteed* no-go region. Small addition.
  - **#3 Keep-out volume** (user's lap/torso) — a task-space plane/box barrier; to protect the elbow
    and links (not just the EE) it needs the whole-arm collision spheres below.
- **Self-collision / whole-arm obstacle avoidance** — ✅ **DONE (2026-06)**. Collision-sphere proxies
  (`fabrix/collision.py`): `auto_arm_spheres` places spheres along each link's bone (radius from the
  collision-mesh width; per-link `radius_scale`, or full hand-tuning — see the pipeline below), `nonadjacent_pairs`
  selects the self-collision pairs (excludes same-link + parent-child). A **batched / shared-FK**
  barrier — one FK to all link frames (`CustomFK.body_poses`, **#8**), vectorized pairwise/region SDFs,
  one summed pullback (generalized `sdf_barrier_*`, **#1**) — drives `self_collision_{geometry,potential}`
  and the whole-arm `arm_obstacle_*` / `arm_plane_*` (obstacle & floor now guard *every* sphere, not
  just the EE). Geometry deflects, potential is the hard wall (the M2 division of labour).
  - **Validated** (`tests/test_collision.py`, 11): `body_poses` vs MuJoCo 1e-9; self-collision `J`/`J̇q̇`
    vs finite-diff 1e-10; the batched leaf is **bit-identical** to ``k`` separate leaves (`max|Δ|<1e-9`,
    the #1 claim); driven into a −73 mm self-colliding fold it stays clear, and a forearm rammed −106 mm
    through an obstacle deflects to **+55 mm** while the EE still reaches (2 mm). Full collision fabric
    (k=84 self + 16 obstacle + 16 floor) **~124 µs/step** — only ~15 µs over the base fabric (the batched
    payoff). `--check`: obstacle +53 mm, floor +143 mm, self-collision +29 mm, all CLEAR; the demo draws
    the spheres translucent-blue.
  - **Tuning pipeline (auto → tune → reload)** — ✅ done. `auto_arm_spheres(radius_scale={…})` /
    `SphereModel.scaled` shrink over-covering links in one line; `SphereModel.to_dict`↔`from_dict`
    round-trips a committable literal; **`demos/tune_spheres.py`** is a **viser web UI** (optional `viz`
    dep) — select a sphere, drag its radius / position gizmo, scrub the pose to check coverage, export
    to `demos/spheres_tuned.py`, which **`load_spheres`** auto-loads into the demos (auto until that file
    exists). Browser-based → no local OpenGL / `mjpython`, works headless / over SSH.
  - **#1 batched `sdf_barrier`** + **#8 per-link FK** — ✅ done (the two foundations above).
  - Still deferred to real-arm dimensions: **#2** hard joint no-go limits, **#3** keep-out volume
    (the whole-arm sphere machinery #3 needs is now in place — add a keep-out plane/box barrier over the spheres).

### Performance profile (2026-06-04, CPU float32, single arm)
Profiled the real 10-leaf demo fabric (permanent harness `bench/profile_fabrix.py` —
`uv run python bench/profile_fabrix.py`).
- **~110–123 µs / control step** (single dispatch, distinct inputs) — ~8× under the 1 kHz budget.
  Hygiene clean: params traced (6 values → 1 compile, no recompiles), q̈ + pose-error stay float32,
  output finite. No anti-patterns.
- **Cost = per-leaf autodiff of the FK.** FK primal ~7 µs; +J+curvature 4–6× that; a full FK-leaf
  60–90 µs. Config-space leaves (posture/limit/damping) + `resolve` are **~free** (<1 µs). XLA
  dedupes the cheap primal across leaves but **not** the autodiff → the redundancy that #1 fixes.
- **`lax.scan` of the policy is ~2× per-eval vs independent dispatches** (loses thread-parallelism).
  A hard 1 kHz loop should dispatch once/step; the demo's 8-substep scan (~2.5 ms) is fine (16 ms budget).
- **Scaling (the collision decision):** N obstacle barriers, separate leaves vs one batched leaf —
  N=64: separate **41 s compile / 235 µs**, batched **2.7 s compile (flat) / 131 µs**, `max|Δq̈|=0`.

### Code-quality backlog (#2–#7, from the 2026-06 profile — deferred, low value)
Minor / cosmetic; XLA already absorbs most. Park here, revisit opportunistically (e.g. when editing
the file anyway). #1 (batched barriers) and #8 (per-link FK) are **done** — see the self-collision
entry above.
- **#2** `value_jac_curv` evals the primal `x = phi(q)` that `jacfwd(phi)(q)` also recomputes (redundant primal).
- **#3** `J @ qd` computed twice in barrier leaves (explicit `dd`, and again as the primal of the curvature jvp).
- **#4** `d ** power` with `power=2.0` (float pow) instead of `d*d` in `_barrier_accel`.
- **#5** `obstacle_*(center=None)` rebuilds the `sphere_sdf_map` closure inside the leaf each trace (harmless, hoistable).
- **#6** `resolve` has no graceful fallback if the metric is non-PD (relies on posture+damping invariant; reg=1e-6). Document louder.
- **#7** `combine` Python-loop sum — fine for static small N; the batched design sidesteps it for collision.

### Out of scope
M4-style batched-RL benchmarks / learnable fabrics; hardware StableHLO/AOT export. (Keep code
pure-functional + vmap-clean anyway — free hygiene.)

---

## Run

```bash
uv run pytest -q                          # 39 tests (11 M1 + 11 M2 + 6 M3 + 11 collision)
uv run python demos/attractor_reach.py    # M1 -> demos/attractor_reach.png
uv run python demos/obstacle_reach.py     # M2 -> demos/obstacle_reach.png
uv run mjpython demos/interactive_track.py        # M3 6-DOF pose + whole-arm/self-collision (macOS; --check headless)
uv run --group viz python demos/tune_spheres.py   # tune collision spheres in a viser web UI (browser; optional 'viz' dep)
```

## Environment / facts

- uv project (Python 3.12): jax, jax-dataclasses, jaxlie, loop-rate-limiters, matplotlib, mujoco;
  pytest (dev); viser (optional `viz` group — only `demos/tune_spheres.py`; core install stays lean).
  `pyproject.toml` has `[tool.pytest.ini_options] pythonpath=["."]`.
- Models: MuJoCo Menagerie sparse-checkout (`kinova_gen3`, `robotiq_2f85`) — not vendored.
- Gen3 EE site is `pinch_site`; arm is 7 hinge joints (`nq=nv=7`).
- Real-arm context (motivation): 200 Hz IK loop synced to feedback → 1 kHz low-level servo,
  currently ZOH (no interpolation). Open-loop IK was tried and tracks poorly — keep closed-loop.
