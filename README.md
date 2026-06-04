# fabrix

A JAX-native **geometric fabrics** motion-generation library for robot manipulators.

Geometric fabrics (the RMPflow successor, from Ratliff / Van Wyk et al.) are second-order
reactive motion policies that emit **C2-smooth** joint references — smooth and quiet on real
hardware, unlike first-order differential IK whose non-smooth reference makes stiff servos
vibrate. `fabrix` is built on autodiff: the task Jacobian and the curvature term `J̇q̇` come
straight from `jax.jacfwd` / nested `jvp`, so the whole policy is `jit`-fast and correct by
construction.

**Status: Milestone 1 complete** — a forced attractor fabric on a Kinova Gen3.
Full plan, progress, and TODOs in [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). The robot models are not vendored — clone them:

```bash
uv sync                                   # create env from uv.lock
# Gen3 + 2F-85 models (sparse clone of MuJoCo Menagerie):
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set kinova_gen3 robotiq_2f85 && cd ..
```

## Run

```bash
uv run pytest -q                          # correctness, convergence, smoothness, latency
uv run python demos/attractor_reach.py    # reach demo -> demos/attractor_reach.png
```

## Results (M1)

- End-effector reach error **1.17 mm**; controller **~53 µs/step** (float32, CPU) — ~5% of a
  1 kHz budget, so single-arm real-time is comfortable.
- Commanded `q̇`/`q̈` are continuous and bounded (a C2 reference).

## Layout

| path | contents |
|---|---|
| `fabrix/` | library: `spec` (spec algebra), `diff` (autodiff `J` + `J̇q̇`), `kinematics` (`CustomFK`, `MJXProvider`), `maps`, `leaves`, `fabric`, `integrate` |
| `demos/` | runnable demos |
| `tests/` | pytest suite |
| `experiments/` | de-risking scratch (autodiff-FK correctness + latency studies) |

Models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0),
cloned separately rather than vendored.
