# fabrix — Roadmap, Status & TODOs

Living design+status doc. Project scope is **M1 → M3**. (M4-style batched-RL / learnable
fabrics and hardware StableHLO/AOT export are explicitly out of scope.)

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

### ☐ M3 — orientation / full SE(3) pose (PROJECT ENDPOINT)
- [ ] Extend `CustomFK` with `site_pose(q) -> (pos, quat)` (orientation is already available as
  `qmul(qw[site_body], site_quat)` in the FK — cheap to add).
- [ ] `maps.py`: SE(3) pose map; pose error `= Log(T_target⁻¹ T_current) ∈ se(3)` via
  **jaxlie** (`SE3.from_rotation_and_translation`, `.inverse()`, `@`, `.log()`).
- [ ] 6-DOF pose attractor leaf; `J`/`J̇q̇` through the jaxlie expression by autodiff (verify the
  nested `jvp` flows through jaxlie). Tests: position+orientation convergence, smoothness.
- [ ] Demo: upgrade `interactive_track.py` to also track the mocap target's **orientation** (read
  `mocap_quat`, 6-DOF target) — supplies the rotation tracking the interactive demo currently lacks.

### Backlog / follow-ups (noted from the interactive demo, 2026-06)
Driving `demos/interactive_track.py` surfaced these. None block M3; revisit after (or fold the
rotation one into M3):
- **Rotation tracking** — mocap can be rotated (Ctrl+left-drag) but the arm ignores it
  (position-only). Lands with **M3** (above).
- **Obstacle avoided from too far (~10–20 cm)** — the `obstacle_potential` standoff band is
  `d0=0.2` (20 cm). Lower it (~0.08–0.12) for tighter avoidance. Tunable, not a bug.
- **No ground/floor avoidance** — `plane_sdf_map` exists but no plane barrier leaf is wired in.
  Generalize the obstacle leaves to take any task map (or add a plane variant) + add a ground
  obstacle to the demo.
- **Self-collision is NOT handled** — the arm avoids self-collision only *incidentally* (posture +
  the motions tried); there are no self-collision leaves. Real support = pairwise body-distance task
  maps (capsule/sphere proxies), a separate feature.
- **Draggable obstacle** — lift the obstacle center to a traced `FabricParams` field (~10 lines) so
  the obstacle can be dragged live too.

### Out of scope
M4-style batched-RL benchmarks / learnable fabrics; hardware StableHLO/AOT export. (Keep code
pure-functional + vmap-clean anyway — free hygiene.)

---

## Run

```bash
uv run pytest -q                          # 19 tests
uv run python demos/attractor_reach.py    # M1 -> demos/attractor_reach.png
uv run python demos/obstacle_reach.py     # M2 -> demos/obstacle_reach.png
```

## Environment / facts

- uv project (Python 3.12): jax, jax-dataclasses, jaxlie, matplotlib, mujoco; pytest (dev).
  `pyproject.toml` has `[tool.pytest.ini_options] pythonpath=["."]`.
- Models: MuJoCo Menagerie sparse-checkout (`kinova_gen3`, `robotiq_2f85`) — not vendored.
- Gen3 EE site is `pinch_site`; arm is 7 hinge joints (`nq=nv=7`).
- Real-arm context (motivation): 200 Hz IK loop synced to feedback → 1 kHz low-level servo,
  currently ZOH (no interpolation). Open-loop IK was tried and tracks poorly — keep closed-loop.
