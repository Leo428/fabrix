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

### ☐ M2 — energization + obstacle/joint-limit geometries (NEXT)
Turns the forced base case into a *full* geometric fabric. TODO:
- [ ] `fabrix/energy.py`: Finsler energies (HD2 in `q̇`); energy metric `M_e = ∂²_{q̇} L_e` via
  `jax.hessian`; the Euler–Lagrange force terms. Start with a Lagrangian energy `½ q̇ᵀ G(x) q̇`.
- [ ] `fabrix/geometry.py`: the **energization operator** — given an HD2 geometry `q̈ = -h₂(x,ẋ)`
  and a Finsler energy, produce the energized acceleration that follows the geometry's path while
  conserving the energy (the energy-orthogonal "boost" along `ẋ`). Pin the exact formula to
  *Optimization Fabrics* (Ratliff et al.) / *Geometric Fabrics* (Van Wyk et al.).
  - [ ] **Gate with an energy-conservation test**: geometry alone conserves `L_e` to ~1e-6 over
    a rollout. This is the correctness check for the operator.
- [ ] HD2 geometry leaves:
  - [ ] joint-limit avoidance: barrier on `(q - q_min)`, `(q_max - q)`; HD2 geometry that pushes
    away as a limit nears; **invariant test: limits never violated**.
  - [ ] obstacle avoidance: task map = signed distance (start with analytic SDFs for sphere/plane
    primitives — MJX collision is thin); repulsive HD2 geometry; **invariant test: no penetration**.
- [ ] Assemble: energized geometries + M1 attractor (forcing potential) + damping → goal reached
  *and* constraints respected, still smooth. Tests: goal still reached; smoothness preserved.

### ☐ M3 — orientation / full SE(3) pose (PROJECT ENDPOINT)
- [ ] Extend `CustomFK` with `site_pose(q) -> (pos, quat)` (orientation is already available as
  `qmul(qw[site_body], site_quat)` in the FK — cheap to add).
- [ ] `maps.py`: SE(3) pose map; pose error `= Log(T_target⁻¹ T_current) ∈ se(3)` via
  **jaxlie** (`SE3.from_rotation_and_translation`, `.inverse()`, `@`, `.log()`).
- [ ] 6-DOF pose attractor leaf; `J`/`J̇q̇` through the jaxlie expression by autodiff (verify the
  nested `jvp` flows through jaxlie). Tests: position+orientation convergence, smoothness.

### Out of scope
M4-style batched-RL benchmarks / learnable fabrics; hardware StableHLO/AOT export. (Keep code
pure-functional + vmap-clean anyway — free hygiene.)

---

## Run

```bash
uv run pytest -q                          # 9 tests
uv run python demos/attractor_reach.py    # -> demos/attractor_reach.png
```

## Environment / facts

- uv project (Python 3.12): jax, mujoco, mujoco-mjx, jaxlie, matplotlib; pytest (dev).
  `pyproject.toml` has `[tool.pytest.ini_options] pythonpath=["."]`.
- Models: MuJoCo Menagerie sparse-checkout (`kinova_gen3`, `robotiq_2f85`) — not vendored.
- Gen3 EE site is `pinch_site`; arm is 7 hinge joints (`nq=nv=7`).
- Real-arm context (motivation): 200 Hz IK loop synced to feedback → 1 kHz low-level servo,
  currently ZOH (no interpolation). Open-loop IK was tried and tracks poorly — keep closed-loop.
