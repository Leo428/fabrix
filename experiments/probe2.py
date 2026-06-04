"""Is the ~1.9ms floor per-dispatch overhead, or my scalar-packed FK compute?
Measure trivial jitted ops: if 'add scalar' is also ~1.9ms -> dispatch floor (Python).
If it's ~tens of us -> the FK compute is the problem (fixable by vectorizing)."""
import time, statistics
import jax, jax.numpy as jnp, numpy as np

def meas(jfn, args, budget=1.0, M=40000):
    jfn(*args).block_until_ready()
    ts = []; s = time.perf_counter()
    while len(ts) < M and time.perf_counter()-s < budget:
        t = time.perf_counter(); jfn(*args).block_until_ready(); ts.append(time.perf_counter()-t)
    return min(ts)*1e6, statistics.median(ts)*1e6, len(ts)

x = jnp.array(1.0, jnp.float32)
v = jnp.array(np.random.rand(7), jnp.float32)
A = jnp.array(np.random.rand(7, 7), jnp.float32)

cases = [
    ("add scalar",  jax.jit(lambda x: x + 1.0), (x,)),
    ("vec*2 (7,)",  jax.jit(lambda v: v * 2.0), (v,)),
    ("matvec 7x7",  jax.jit(lambda A, v: A @ v), (A, v)),
    ("solve 7x7",   jax.jit(lambda A, v: jnp.linalg.solve(A @ A.T + jnp.eye(7, dtype=A.dtype), v)), (A, v)),
]
for name, f, a in cases:
    mn, md, n = meas(f, a)
    print(f"{name:<12} min={mn:7.1f}us  median={md:7.1f}us  (n={n})")
