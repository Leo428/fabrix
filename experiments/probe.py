"""Latency probe for the custom FK provider, controlled by env (XLA_FLAGS, thread vars).
Run under different env settings to find the per-dispatch CPU floor.
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp, numpy as np, mujoco
from gen3_fk import build_site_fk

print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS','')!r}  "
      f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','')!r}")

XML = "/Users/huzheyuan/Documents/kinova/mujoco_menagerie/kinova_gen3/gen3.xml"
fk, m, site = build_site_fk(XML)

# correctness vs MuJoCo at a random config
_q = np.random.default_rng(7).uniform(-1, 1, m.nq)
_d = mujoco.MjData(m); _d.qpos[:] = _q; mujoco.mj_kinematics(m, _d)
_err = float(np.abs(np.asarray(fk(jnp.array(_q, jnp.float32))) - _d.site_xpos[site]).max())
print(f"  correctness vs mujoco: {_err:.1e}")

def jdq(q, qd):
    f1 = lambda qq: jax.jvp(fk, (qq,), (qd,))[1]
    return jax.jvp(f1, (q,), (qd,))[1]

q0 = jnp.array(np.random.default_rng(0).uniform(-1, 1, m.nq), jnp.float32)
qd0 = jnp.array(np.random.default_rng(1).uniform(-1, 1, m.nv), jnp.float32)

def measure(jfn, args, budget_s=1.0, max_M=8000):
    jfn(*args).block_until_ready()
    ts = []; start = time.perf_counter()
    while len(ts) < max_M and (time.perf_counter()-start) < budget_s:
        t = time.perf_counter(); jfn(*args).block_until_ready(); ts.append(time.perf_counter()-t)
    return min(ts)*1e6, statistics.median(ts)*1e6, len(ts)

for name, jfn, args in [("fk", jax.jit(fk), (q0,)),
                        ("Jdotqd", jax.jit(jdq), (q0, qd0))]:
    mn, md, n = measure(jfn, args)
    print(f"  {name:<8} min={mn:8.1f}us  median={md:8.1f}us  (n={n})")
